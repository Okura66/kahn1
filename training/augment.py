"""On-the-fly dynamic data augmentation pipeline.

Applied during training at each epoch without disk materialization.
Objective: prevent the model from memorizing dataset-specific class semantics,
forcing it to dynamically read and interpret the candidate option list supplied
in the prompt (analogous to taxonomic text encoding for open-vocabulary runtime resolution).

Six core transformations:
  1. Candidate option permutation (with exact label index remapping).
  2. Option cardinality sub-sampling: correct answer plus random subset
     (cardinality drawn between 2 and 25, i.e. up to 26 letters with the fallback).
  3. Cross-dataset distractor injection from a unified distractor pool.
  4. Correct answer omission mapping to the fallback 'other' label (-1)
     (5-10% of training instances) to ensure calibration of the rejection class.
  5. Prompt framing template variability across diverse phrasing styles.
  6. Bilingual diversification (French and English) to prevent linguistic overfitting.

For Score and Noul tasks, relevant transformations (framing templates, language,
directional reversal) are selectively applied.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Distractor Pool — Sampled from cross-dataset instances
# ---------------------------------------------------------------------------

@dataclass
class DistractorPool:
    """Distractor candidate repository indexed by question primitive.

    Populated once prior to training from the raw dataset mixture.
    For Choice: pool of candidate text options from cross-dataset instances.
    For Score: no distractors injected (ordinal levels remain structurally semantic).
    For Noul: no distractors injected (binary yes/no space).
    """
    choice_options: list[str]

    @classmethod
    def build(cls, examples: list[dict]) -> "DistractorPool":
        pool: set[str] = set()
        for ex in examples:
            # Fixed-option sources (a question with its own answers, e.g. reading
            # comprehension) would pour whole sentences into the intent-label pool.
            if ex["kind"] == "choice" and not ex.get("fixed_options"):
                # For single-label sources, options = [intent]; store intent.
                # For multi-option sources, index all option candidates.
                for opt in ex["options"]:
                    pool.add(opt)
        return cls(choice_options=list(pool))

    def sample_choice(self, n: int, exclude: set[str], rng: random.Random) -> list[str]:
        cands = [o for o in self.choice_options if o not in exclude]
        if not cands:
            return []
        if n >= len(cands):
            n = len(cands)
        return rng.sample(cands, n)


# ---------------------------------------------------------------------------
# Framing Templates (FR/EN)
# ---------------------------------------------------------------------------

CHOICE_TEMPLATES = [
    ("Quelle est l'intention de l'utilisateur ?", "FR"),
    ("What is the user's intent?", "EN"),
    ("Quelle catégorie correspond le mieux à ce message ?", "FR"),
    ("Which category best matches this message?", "EN"),
    ("Choisissez la meilleure option pour cet état.", "FR"),
    ("Select the best option for this state.", "EN"),
    ("Quelle est la bonne réponse parmi les suivantes ?", "FR"),
    ("Which is the correct answer among the following?", "EN"),
    ("Classez cet état dans la bonne option.", "FR"),
    ("Classify this state into the correct option.", "EN"),
]

SCORE_TEMPLATES = [
    ("Quelle est la note de cet avis ?", "FR"),
    ("What is the rating of this review?", "EN"),
    ("Évaluez l'intensité de cet état.", "FR"),
    ("Rate the intensity of this state.", "EN"),
    ("Quel niveau correspond le mieux ?", "FR"),
    ("Which level best applies?", "EN"),
    ("Quelle est la polarité ou le sentiment de ce message ?", "FR"),
    ("What is the polarity or sentiment of this text?", "EN"),
    ("Comment évaluez-vous cet élément ?", "FR"),
    ("How would you rate this item?", "EN"),
]

# Noul templates are grouped by the FORM of the ex["statement"] field, because the
# sources are not interchangeable: mnli/anli/snli provide declarative hypotheses,
# boolq provides yes/no questions, qnli provides wh-questions whose label means
# "does the state contain the answer". A single "is this statement true" template
# is semantically wrong for 40% of the mixture (see build_dataset._ex_noul).
#
# Each entry is (prompt, lang, flip). flip=True marks a negative-polarity framing
# ("is it false?", "does the state contradict it?"): the training label is inverted
# to match. This is the Noul analogue of Score's scale reversal — the answer cannot
# be predicted without reading the prompt, so a polarity inversion like the one
# that shipped in the first checkpoint would surface immediately at validation.
NOUL_TEMPLATES = {
    "statement": [
        ("La proposition suivante est-elle vraie pour cet état ?", "FR", False),
        ("Is the following statement true for this state?", "EN", False),
        ("Cette affirmation est-elle exacte au vu de l'état ?", "FR", False),
        ("Does the state support the following claim?", "EN", False),
        ("L'état permet-il d'affirmer la proposition suivante ?", "FR", False),
        ("Is the following claim consistent with the state?", "EN", False),
        ("La proposition suivante est-elle fausse pour cet état ?", "FR", True),
        ("Is the following statement false for this state?", "EN", True),
        ("L'état contredit-il la proposition suivante ?", "FR", True),
        ("Does the state contradict the following claim?", "EN", True),
    ],
    "yesno_question": [
        ("La réponse à la question suivante est-elle « oui » ?", "FR", False),
        ("Is the answer to the following question yes?", "EN", False),
        ("D'après l'état, répond-on oui à la question suivante ?", "FR", False),
        ("Based on the state, is the following question answered with yes?", "EN", False),
        ("La réponse à la question suivante est-elle « non » ?", "FR", True),
        ("Is the answer to the following question no?", "EN", True),
    ],
    "answerable_question": [
        ("L'état contient-il la réponse à la question suivante ?", "FR", False),
        ("Does the state contain the answer to the following question?", "EN", False),
        ("Peut-on répondre à la question suivante à partir de cet état ?", "FR", False),
        ("Can the following question be answered from this state?", "EN", False),
    ],
}


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------

@dataclass
class AugmentedExample:
    """Container for an augmented instance prepared for model training.

    For Choice: options (text list), label (0-based integer index), include_other flag.
    label = -1 when the correct option has been withheld (relegated to fallback 'other').
    """
    state: str
    kind: str
    prompt: str
    options: list[str]  # For choice/score
    levels: list[str]   # For score (alias)
    statement: str       # For noul
    label: int
    lang: str
    source: str
    # Whether the fallback option is displayed. Sampled independently of the label:
    # tying it to (label == -1) taught the model 'fallback shown => fallback correct'.
    include_other: bool = False


def augment_choice(
    ex: dict,
    pool: DistractorPool,
    rng: random.Random,
    p_remove_correct: float = 0.07,
    min_options: int = 2,
    # 25 + the fallback option = 26, the direct-evaluation ceiling: every letter A-Z
    # is a training target. The previous cap of 16 left Q..Z tokens never trained.
    max_options: int = 25,
    p_include_other: float = 0.5,
) -> AugmentedExample:
    """Apply dynamic augmentation transformations to a Choice task instance."""
    if ex.get("fixed_options"):
        return _augment_fixed_choice(ex, rng)
    # 1, 5, 6: template + language variation
    prompt, lang = rng.choice(CHOICE_TEMPLATES)

    # Correct candidate
    correct = ex["options"][ex["label"]] if ex["label"] >= 0 else None
    if correct is None:
        # Initial 'other' instance (e.g. out-of-scope) -> sample distractors and maintain label=-1
        correct_present = False
    else:
        correct_present = True

    # 4. Random omission of correct option -> label = -1 (fallback other)
    if correct_present and rng.random() < p_remove_correct:
        correct_present = False

    # 2. Sub-sampling option cardinality between min_options and max_options
    n_target = rng.randint(min_options, max_options)

    exclude = set()
    distractors: list[str] = []
    if correct is not None:
        exclude.add(correct)

    # 3. Inject distractors from distractor pool
    n_distractors = n_target - (1 if correct_present else 0)
    if n_distractors > 0:
        distractors = pool.sample_choice(n_distractors, exclude, rng)

    # Build options list
    options = []
    if correct_present and correct is not None:
        options.append(correct)
    options.extend(distractors)
    if not options:
        # Fallback when pool is empty
        options.append(correct or "N/A")

    # Determine ground truth index
    if correct_present and correct is not None:
        label = options.index(correct)
    else:
        label = -1

    # 1. Permute option ordering and remap ground truth label
    perm = list(range(len(options)))
    rng.shuffle(perm)
    new_options = [options[i] for i in perm]
    if label >= 0:
        new_label = perm.index(label)
    else:
        new_label = -1

    # The fallback option is mandatory when it IS the answer, and otherwise shown
    # p_include_other of the time so that its presence carries no information.
    include_other = True if new_label == -1 else (rng.random() < p_include_other)

    return AugmentedExample(
        state=ex["state"], kind="choice", prompt=prompt,
        options=new_options, levels=[], statement="",
        label=new_label, lang=lang, source=ex["source"],
        include_other=include_other,
    )


def _augment_fixed_choice(ex: dict, rng: random.Random, p_remove_correct: float = 0.05,
                          p_include_other: float = 0.3) -> AugmentedExample:
    """A question that carries its own prompt and answers: keep both, only reorder them.

    Swapping the question for a generic template or the answers for pool distractors
    would destroy what a reading-comprehension or rule question asks.
    """
    options = list(ex["options"])
    label = ex["label"]
    if label >= 0 and rng.random() < p_remove_correct and len(options) > 2:
        options.pop(label)
        label = -1
    perm = list(range(len(options)))
    rng.shuffle(perm)
    new_options = [options[i] for i in perm]
    new_label = perm.index(label) if label >= 0 else -1
    include_other = True if new_label == -1 else (rng.random() < p_include_other)
    return AugmentedExample(
        state=ex["state"], kind="choice", prompt=ex["prompt"],
        options=new_options, levels=[], statement="",
        label=new_label, lang="EN", source=ex["source"],
        include_other=include_other,
    )


def augment_score(ex: dict, rng: random.Random, reverse_prob: float = 0.0) -> AugmentedExample:
    """Augment a Score instance with template variation, language selection, and directional scale reversal.

    Semantic sequence is preserved, but the ordinal progression may be presented in
    ascending order (L_0..L_{M-1}) or descending order (L_{M-1}..L_0).
    When reversed, label l is remapped to M - 1 - l, forcing the model to attend to
    the prompt taxonomy rather than relying on absolute token positional priors.
    """
    prompt, lang = rng.choice(SCORE_TEMPLATES)
    if ex.get("fixed_options"):
        # A rubric question written for its levels ("Which risk tier applies?"): a generic
        # template would drop what the levels are measuring.
        prompt = ex["prompt"]
    levels = list(ex["levels"])
    label = ex["label"]
    if len(levels) >= 2 and rng.random() < reverse_prob:
        levels = list(reversed(levels))
        label = len(levels) - 1 - label

    return AugmentedExample(
        state=ex["state"], kind="score", prompt=prompt,
        options=levels, levels=levels, statement="",
        label=label, lang=lang, source=ex["source"],
    )


def augment_noul(ex: dict, rng: random.Random) -> AugmentedExample:
    """Augment a binary Noul instance with form-aware templates and polarity flips.

    The template family follows the statement's form (declarative hypothesis, yes/no
    question, or answerability question); negative-polarity templates invert the label.
    """
    form = ex.get("form", "statement")
    templates = NOUL_TEMPLATES.get(form, NOUL_TEMPLATES["statement"])
    prompt, lang, flip = rng.choice(templates)
    label = ex["label"]
    if flip:
        label = 1 - label
    return AugmentedExample(
        state=ex["state"], kind="noul", prompt=prompt,
        options=[], levels=[], statement=ex["statement"],
        label=label, lang=lang, source=ex["source"],
    )


def augment(
    ex: dict,
    pool: DistractorPool,
    rng: random.Random,
    reverse_prob: float = 0.0,
) -> AugmentedExample:
    """Dispatch augmentation by question primitive kind."""
    if ex["kind"] == "choice":
        return augment_choice(ex, pool, rng)
    elif ex["kind"] == "score":
        return augment_score(ex, rng, reverse_prob=reverse_prob)
    elif ex["kind"] == "noul":
        return augment_noul(ex, rng)
    else:
        raise ValueError(f"Unknown question kind: {ex['kind']}")


# ---------------------------------------------------------------------------
# PyTorch Dataset: Dynamic augmentation iterator
# ---------------------------------------------------------------------------

class AugmentingDataset:
    """PyTorch-compatible dataset wrapper applying on-the-fly augmentation on item access.

    Ensures that successive epochs observe distinct randomized augmentations.
    """

    def __init__(
        self,
        examples: list[dict],
        pool: DistractorPool,
        seed: int = 0,
        reverse_prob: float = 0.5,
    ):
        self.examples = examples
        self.pool = pool
        self.rng = random.Random(seed)
        self.reverse_prob = reverse_prob

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> AugmentedExample:
        return augment(self.examples[idx], self.pool, self.rng, reverse_prob=self.reverse_prob)
