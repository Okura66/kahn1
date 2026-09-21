"""Construction pipeline for training and evaluation datasets.

Downloads Hugging Face datasets and serializes them to the intermediate JSONL schema:

    {"state": "...", "kind": "choice", "prompt": "...",
     "options": ["...", "..."], "label": 2, "source": "clinc150"}

Training sources (300k+ instances):
  Choice : CLINC150 (including out-of-scope), AG News, DBpedia-14, GoEmotions,
           legal clause triage (synthetic)
  Score  : Yelp Review Full (5), Amazon Reviews (5), TweetEval sentiment (3),
           IMDB (2), medical triage / CSAT (synthetic) — deliberately mixed
           cardinalities, since `levels` is arbitrary at inference time
  Noul   : MNLI, SNLI, ANLI (entailment vs contradiction, neutral dropped),
           QNLI (answerability), BoolQ (yes/no questions) — each balanced 50/50

Reserved evaluation sources:
  Strict invariant: The evaluation split is partitioned at the DATASET LEVEL, not per row.
  Banking77 and MASSIVE (Choice), SST-5 (Score), RTE and SciTail (Noul) are strictly
  reserved for zero-shot generalization benchmarking and never appear in training
  data mixtures. Every primitive the engine answers must have at least one reserved
  source — build() refuses to write an eval split with a missing primitive.

Usage:
    python training/build_dataset.py --out data/train.jsonl --eval-out data/eval.jsonl --max-per-source 30000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# Reserved Evaluation Datasets (Never seen during training)
# ---------------------------------------------------------------------------

EVAL_CHOICE_SOURCES = {
    "banking77",  # 77 unobserved banking intent classes
    "massive",    # 60 unobserved user intent classes (SetFit/amazon_massive_intent_en-US)
}
EVAL_SCORE_SOURCES = {
    "sst5_eval",  # 5 unobserved sentence polarity levels (SetFit/sst5)
}
EVAL_NOUL_SOURCES = {
    "rte_eval",      # GLUE RTE validation: natively binary entailment, no 3-way collapse
    "scitail_eval",  # SciTail test: science-domain entailment, unobserved domain
}

# Training dataset registry
TRAIN_SOURCES = {
    "clinc150",
    "ag_news",
    "dbpedia_14",
    "go_emotions",
    "yelp_full",
    "amazon_reviews",
    "imdb",
    "tweet_sentiment",
    "medical_triage",
    "legal_triage",
    "csat_sentiment",
    "mnli",
    "snli",
    "boolq",
    "qnli",
    "anli",
}


# ---------------------------------------------------------------------------
# Primitive conversion helpers
# ---------------------------------------------------------------------------

# Unicode line separators that json.dumps(ensure_ascii=False) writes RAW: NEL, LS, PS.
# str.splitlines() breaks a JSONL record at these, so any reader using
# read_text().splitlines() sees an unterminated JSON string. Normalized to spaces.
_LINE_SEPARATORS = str.maketrans({"\x85": " ", " ": " ", " ": " "})


def _clean_text(text: str) -> str:
    return text.translate(_LINE_SEPARATORS)


def _ex_choice(state: str, prompt: str, options: list[str], label: int, source: str) -> dict:
    """Create a typed Choice instance. label = -1 indicates absence of valid options ('other')."""
    return {
        "state": _clean_text(state),
        "kind": "choice",
        "prompt": prompt,
        "options": options,
        "label": label,
        "source": source,
    }


def _ex_score(state: str, prompt: str, levels: list[str], label: int, source: str) -> dict:
    """Create a typed Score instance (invariant natural semantic ordering)."""
    return {
        "state": _clean_text(state),
        "kind": "score",
        "prompt": prompt,
        "levels": levels,
        "label": label,
        "source": source,
    }


def _ex_noul(state: str, statement: str, label: int, source: str,
             form: str = "statement") -> dict:
    """Create a typed Noul instance (binary: 1=yes/entailment, 0=no/contradiction).

    form describes what the statement field actually is, so augmentation can frame
    the prompt truthfully (see training/augment.py NOUL_TEMPLATES):
      - "statement": a declarative hypothesis (mnli, anli, snli, rte, scitail)
      - "yesno_question": a question whose answer is yes/no (boolq)
      - "answerable_question": a wh-question; label means the state answers it (qnli)
    """
    return {
        "state": _clean_text(state),
        "kind": "noul",
        "statement": _clean_text(statement),
        "label": label,
        "source": source,
        "form": form,
    }


def _balance_noul(examples: list[dict], seed: int = 0) -> list[dict]:
    """Subsample the majority class of a Noul source down to a 50/50 label balance.

    The 3-way NLI sources collapse to binary at roughly 33% yes; left as-is the
    mixture teaches a 'no' prior (40.6% yes overall in the first released training set).
    """
    rng = random.Random(seed)
    pos = [e for e in examples if e["label"] == 1]
    neg = [e for e in examples if e["label"] == 0]
    n = min(len(pos), len(neg))
    rng.shuffle(pos)
    rng.shuffle(neg)
    out = pos[:n] + neg[:n]
    rng.shuffle(out)
    return out


# ---------------------------------------------------------------------------
# Training Loaders (Compatible with datasets >= 5.0 and Parquet repositories)
# ---------------------------------------------------------------------------

def load_clinc150() -> list[dict]:
    """CLINC150 with out-of-scope (oos) split.

    Explicitly teaches the fallback rejection class 'none of the options apply' (allow_other).
    The out-of-scope label "oos" is mapped to label index -1.
    """
    from datasets import load_dataset
    ds = load_dataset("clinc/clinc_oos", "plus")
    names = ds["train"].features["intent"].names
    examples = []
    for split in ["train", "validation", "test"]:
        if split not in ds:
            continue
        for row in ds[split]:
            text = row["text"]
            intent_idx = row["intent"]
            intent_name = names[intent_idx]
            is_oos = (intent_name == "oos")
            examples.append(_ex_choice(
                state=text,
                prompt="Quelle est l'intention de l'utilisateur ?",
                options=[intent_name] if not is_oos else [],
                label=0 if not is_oos else -1,
                source="clinc150",
            ))
    return examples


def load_ag_news() -> list[dict]:
    """AG News: 4 thematic categories (World, Sports, Business, Sci/Tech)."""
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news")
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    examples = []
    for row in ds["train"]:
        examples.append(_ex_choice(
            state=row["text"],
            prompt="Quelle est la catégorie de cet article ?",
            options=labels,
            label=row["label"],
            source="ag_news",
        ))
    return examples


def load_dbpedia() -> list[dict]:
    """DBpedia-14: 14 encyclopedic topic categories."""
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/dbpedia_14")
    labels = [
        "Company", "EducationalInstitution", "Artist",
        "Athlete", "OfficeHolder", "MeanOfTransportation",
        "Building", "Animal", "Plant", "Album", "Film",
        "WrittenWork", "NaturalPlace", "Village",
    ]
    examples = []
    for row in ds["train"]:
        examples.append(_ex_choice(
            state=row["content"],
            prompt="Quelle est la catégorie de cet article ?",
            options=labels,
            label=row["label"],
            source="dbpedia_14",
        ))
    return examples


def load_go_emotions() -> list[dict]:
    """GoEmotions: 28 fine-grained emotional nuances."""
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/go_emotions")
    names = ds["train"].features["labels"].feature.names
    examples = []
    for row in ds["train"]:
        lbls = row["labels"]
        if not lbls:
            continue
        examples.append(_ex_choice(
            state=row["text"],
            prompt="Quelle est l'émotion principale de ce texte ?",
            options=names,
            label=lbls[0],
            source="go_emotions",
        ))
    return examples


def load_yelp_full(anti_midpoint: bool = True, max_total: int = 20000) -> list[dict]:
    """Yelp Review Full: 1 to 5 star customer ratings with anti-midpoint class weighting."""
    from datasets import load_dataset
    ds = load_dataset("Yelp/yelp_review_full")
    levels = ["1 étoile", "2 étoiles", "3 étoiles", "4 étoiles", "5 étoiles"]

    if anti_midpoint:
        # Quotas favoring polarity extremes and mitigating median midpoint bias (3 stars = 10%)
        quotas = {0: int(max_total * 0.25), 1: int(max_total * 0.20),
                  2: int(max_total * 0.10), 3: int(max_total * 0.20),
                  4: int(max_total * 0.25)}
        counts = {i: 0 for i in range(5)}
        examples = []
        for row in ds["train"]:
            lbl = row["label"]
            if counts[lbl] < quotas.get(lbl, 0):
                examples.append(_ex_score(
                    state=row["text"],
                    prompt="Quelle est la note de cet avis ?",
                    levels=levels,
                    label=lbl,
                    source="yelp_full",
                ))
                counts[lbl] += 1
            if len(examples) >= max_total:
                break
        return examples

    examples = []
    for row in ds["train"]:
        examples.append(_ex_score(
            state=row["text"],
            prompt="Quelle est la note de cet avis ?",
            levels=levels,
            label=row["label"],
            source="yelp_full",
        ))
    return examples


def load_tweet_sentiment(max_per_class: int = 4000) -> list[dict]:
    """TweetEval sentiment: 3 polarity classes (negative, neutral, positive).

    Applies anti-midpoint weighting: 40% negative, 20% neutral, 40% positive.
    (cardiffnlp/tweet_eval: the previously referenced SetFit mirror stopped resolving,
    which silently removed this source from the first released training mixture.)
    """
    from datasets import load_dataset
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    levels = ["négatif", "neutre", "positif"]
    # label mapping: 0=negative, 1=neutral, 2=positive
    quotas = {0: max_per_class, 1: max_per_class // 2, 2: max_per_class}
    counts = {0: 0, 1: 0, 2: 0}
    examples = []
    for row in ds["train"]:
        lbl = row["label"]
        if lbl in quotas and counts[lbl] < quotas[lbl]:
            examples.append(_ex_score(
                state=row["text"],
                prompt="Quelle est la polarité de ce tweet ?",
                levels=levels,
                label=lbl,
                source="tweet_sentiment",
            ))
            counts[lbl] += 1
        if all(counts[k] >= quotas[k] for k in quotas):
            break
    return examples


def load_amazon_reviews(max_total: int = 20000) -> list[dict]:
    """Amazon product reviews (English): 5 ordinal star levels, balanced per class.

    Second large 5-level source next to Yelp, in a different register (product
    reviews vs venue reviews) and with English level names next to Yelp's French ones.
    """
    from datasets import load_dataset
    ds = load_dataset("SetFit/amazon_reviews_multi_en")
    levels = ["1 star", "2 stars", "3 stars", "4 stars", "5 stars"]
    per_class = max_total // 5
    counts = {i: 0 for i in range(5)}
    examples = []
    for row in ds["train"]:
        lbl = row["label"]
        if counts.get(lbl, per_class) < per_class:
            examples.append(_ex_score(
                state=row["text"],
                prompt="How many stars does this review give?",
                levels=levels,
                label=lbl,
                source="amazon_reviews",
            ))
            counts[lbl] += 1
        if all(c >= per_class for c in counts.values()):
            break
    return examples


def load_imdb(max_total: int = 15000) -> list[dict]:
    """IMDB movie reviews: 2 ordinal polarity levels.

    The only 2-level Score source: levels is arbitrary at inference time, but every
    other ordinal source carries 3 to 5 levels, so binary scales were never trained.
    """
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/imdb")
    levels = ["negative", "positive"]
    per_class = max_total // 2
    counts = {0: 0, 1: 0}
    examples = []
    for row in ds["train"]:
        lbl = row["label"]
        if counts.get(lbl, per_class) < per_class:
            examples.append(_ex_score(
                state=row["text"][:1500],
                prompt="What is the overall sentiment of this review?",
                levels=levels,
                label=lbl,
                source="imdb",
            ))
            counts[lbl] += 1
        if all(c >= per_class for c in counts.values()):
            break
    return examples


def load_enterprise_medical_triage() -> list[dict]:
    """Emergency medical triage domain: 4 ordinal severity levels."""
    levels = ["Conseil / Automédication", "Consultation Simple", "Urgence Relative", "Urgence Vitale"]
    cases = [
        # Level 0: Self-care / Advice
        ("Patient de 28 ans présentant un rhume léger sans fièvre depuis 24h.", 0),
        ("Légère courbature après séance de sport, pas de douleur articulaire.", 0),
        ("Petite égratignure superficielle sur l'avant-bras après jardinage.", 0),
        ("Demande de renouvellement de traitement contraceptif habituel.", 0),
        ("Fatigue passagère en fin de semaine d'examen, bilan sanguin récent normal.", 0),
        # Level 1: Routine consultation
        ("Fièvre à 38.5°C depuis 48h accompagnée d'une toux grasse et maux de gorge.", 1),
        ("Entorse bénigne de la cheville suite à faux pas, appui encore possible avec boiterie.", 1),
        ("Éruption cutanée prurigineuse apparue hier sur les cuisses, sans gêne respiratoire.", 1),
        ("Douleur lombaire modérée survenue après port de charge lourde sans irradiation.", 1),
        ("Otite douloureuse unilatérale chez l'adulte sans écoulement purulent.", 1),
        # Level 2: Relative emergency
        ("Coupure profonde à la main avec saignement continu malgré compression de 10 min.", 2),
        ("Crise d'asthme inhabituelle ne cédant que partiellement à 2 bouffées de salbutamol.", 2),
        ("Douleur abdominale vive en fosse iliaque droite fébrile suspectant appendicite.", 2),
        ("Nourrisson de 3 mois avec fièvre à 39.2°C, léthargique et refusant le biberon.", 2),
        ("Traumatisme crânien avec brève perte de connaissance de 30 secondes chez un footballeur.", 2),
        # Level 3: Vital emergency
        ("Homme de 58 ans avec douleur thoracique constrictive irradiant dans le bras gauche et sueurs.", 3),
        ("Choc anaphylactique avec œdème de Quincke, stridor et cyanose après piqûre de guêpe.", 3),
        ("Déficit moteur brutal hémi-corporel droit avec aphasie brutale depuis 15 minutes (AVC).", 3),
        ("Arrêt cardio-respiratoire chez un adulte inconscient en arrêt de ventilation spontanée.", 3),
        ("Polytraumatisé inconscient incarcéré après accident de la route à haute cinétique.", 3),
    ]
    examples = []
    prompts = [
        "Quel est le niveau de gravité et d'urgence de cette situation ?",
        "Évaluez le degré de priorité clinique de ce patient.",
    ]
    for p in prompts:
        for text, lbl in cases:
            examples.append(_ex_score(
                state=text,
                prompt=p,
                levels=levels,
                label=lbl,
                source="medical_triage",
            ))
    return examples


def load_enterprise_legal_triage() -> list[dict]:
    """Legal contract clause classification (6 typed option categories)."""
    options = [
        "Confidentialité",
        "Résiliation",
        "Responsabilité & Indemnisation",
        "Propriété Intellectuelle",
        "Force Majeure",
        "Loi Applicable & Juridiction",
    ]
    clauses = [
        # 0: Confidentiality
        ("Chaque partie s'engage à préserver le secret le plus absolu sur les données confidentielles transmises.", 0),
        ("The recipient agrees to hold and maintain all proprietary information in strict confidence.", 0),
        ("Les obligations de non-divulgation survivront à l'extinction du contrat pour une durée de 5 ans.", 0),
        # 1: Termination
        ("Le présent accord peut être résilié par l'une ou l'autre des parties avec un préavis écrit de 30 jours.", 1),
        ("Either party may immediately terminate this agreement upon material breach not cured within 15 days.", 1),
        ("En cas de liquidation judiciaire du prestataire, le client pourra prononcer la rupture de plein droit.", 1),
        # 2: Liability
        ("La responsabilité totale de chaque partie sera expressément plafonnée au montant des sommes versées.", 2),
        ("In no event shall either party be liable for any indirect, consequential or punitive damages.", 2),
        ("Le prestataire indemnisera le client contre tout recours de tiers résultant d'une négligence grave.", 2),
        # 3: Intellectual Property
        ("Tous les droits d'auteur, brevets et codes sources développés demeurent la propriété exclusive du client.", 3),
        ("All Intellectual Property rights arising out of the performance of the services shall belong to Licensor.", 3),
        ("La présente concession de licence est non-exclusive, mondiale et incessible pour la durée du projet.", 3),
        # 4: Force Majeure
        ("Aucune des parties ne sera tenue pour responsable d'un manquement causé par une catastrophe naturelle ou guerre.", 4),
        ("Neither party shall be liable for delay caused by acts of God, pandemic, or government restrictions.", 4),
        ("L'évènement imprévisible et irrésistible suspend les obligations contractuelles pendant sa durée.", 4),
        # 5: Governing Law
        ("Le présent contrat est régi et interprété selon le droit français, compétence exclusive aux tribunaux de Paris.", 5),
        ("This agreement shall be governed by and construed in accordance with the laws of the State of Delaware.", 5),
        ("Tout litige sera soumis à l'arbitrage de la Chambre de Commerce Internationale.", 5),
    ]
    examples = []
    prompts = [
        "Quelle est la catégorie juridique de cette clause contractuelle ?",
        "Which legal category best describes this contractual clause?",
    ]
    for p in prompts:
        for text, lbl in clauses:
            examples.append(_ex_choice(
                state=text,
                prompt=p,
                options=options,
                label=lbl,
                source="legal_triage",
            ))
    return examples


def load_customer_satisfaction_csat() -> list[dict]:
    """E-commerce and SaaS customer support reviews (5 ordinal CSAT satisfaction levels)."""
    levels = ["Très insatisfait", "Insatisfait", "Neutre", "Satisfait", "Très satisfait"]
    tickets = [
        ("Produit reçu brisé en mille morceaux, service client injoignable, je demande un remboursement immédiat !", 0),
        ("Inadmissible : prélèvement indu sur mon compte bancaire et aucune réponse depuis 3 semaines.", 0),
        ("Délai de livraison non respecté de plus de 10 jours, l'emballage était déchiré.", 1),
        ("L'application plante fréquemment lors du paiement, expérience frustrante à améliorer.", 1),
        ("Commande reçue dans les temps. Produit conforme à la description, ni plus ni moins.", 2),
        ("Article standard qui fait le travail, rapport qualité prix correct sans enthousiasme.", 2),
        ("Bonne expérience d'achat globale, livraison rapide et produit conforme à mes attentes.", 3),
        ("Service après-vente courtois qui a résolu mon problème de mot de passe en 10 minutes.", 3),
        ("Absolument parfait ! Qualité exceptionnelle, expédié en 24h et emballage soigné. Je recommande vivement !", 4),
        ("Meilleur support client jamais testé, réponse ultra rapide et geste commercial généreux, bravo !", 4),
    ]
    examples = []
    prompts = [
        "Quel est le niveau de satisfaction exprimé dans cet avis ?",
        "Rate the customer satisfaction level of this message.",
    ]
    for p in prompts:
        for text, lbl in tickets:
            examples.append(_ex_score(
                state=text,
                prompt=p,
                levels=levels,
                label=lbl,
                source="csat_sentiment",
            ))
    return examples


def load_mnli() -> list[dict]:
    """MNLI: premise/hypothesis inference, entailment (1) vs contradiction (0).

    Neutral rows (label 1) are dropped rather than collapsed into 'no': training a
    hard, fully-confident 'no' on hypotheses the premise neither supports nor refutes
    teaches certainty on genuinely uncertain inputs — the opposite of calibration.
    """
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", "mnli")
    examples = []
    for row in ds["train"]:
        if row["label"] == 1:  # neutral
            continue
        examples.append(_ex_noul(
            state=row["premise"],
            statement=row["hypothesis"],
            label=1 if row["label"] == 0 else 0,
            source="mnli",
        ))
    return _balance_noul(examples)


def load_snli() -> list[dict]:
    """SNLI: image-caption premise/hypothesis pairs, entailment vs contradiction.

    Neutral rows dropped for the same reason as MNLI. Caption domain, disjoint from
    every other Noul source (news/wiki/fiction) and from the SciTail eval domain.
    """
    from datasets import load_dataset
    ds = load_dataset("stanfordnlp/snli")
    examples = []
    for row in ds["train"]:
        if row["label"] != 0 and row["label"] != 2:  # keep entailment/contradiction; -1 = unlabelled
            continue
        examples.append(_ex_noul(
            state=row["premise"],
            statement=row["hypothesis"],
            label=1 if row["label"] == 0 else 0,
            source="snli",
        ))
    return _balance_noul(examples)


def load_boolq() -> list[dict]:
    """BoolQ: boolean yes/no reading comprehension questions over text passages."""
    from datasets import load_dataset
    ds = load_dataset("google/boolq")
    examples = []
    for row in ds["train"]:
        examples.append(_ex_noul(
            state=row["passage"],
            statement=row["question"],
            label=1 if row["answer"] else 0,
            source="boolq",
            form="yesno_question",
        ))
    return _balance_noul(examples)


def load_qnli() -> list[dict]:
    """QNLI: does the context sentence answer the question? (0=entailment, 1=not)."""
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", "qnli")
    examples = []
    for row in ds["train"]:
        examples.append(_ex_noul(
            state=row["sentence"],
            statement=row["question"],
            label=1 if row["label"] == 0 else 0,
            source="qnli",
            form="answerable_question",
        ))
    return _balance_noul(examples)


def load_anli() -> list[dict]:
    """ANLI rounds 1-3: adversarial NLI, entailment vs contradiction (neutral dropped)."""
    from datasets import load_dataset
    ds = load_dataset("facebook/anli")
    examples = []
    for split in ["train_r1", "train_r2", "train_r3"]:
        if split in ds:
            for row in ds[split]:
                if row["label"] == 1:  # neutral
                    continue
                examples.append(_ex_noul(
                    state=row["premise"],
                    statement=row["hypothesis"],
                    label=1 if row["label"] == 0 else 0,
                    source="anli",
                ))
    return _balance_noul(examples)


# ---------------------------------------------------------------------------
# Reserved Evaluation Loaders (Never seen during training)
# ---------------------------------------------------------------------------

def load_banking77_eval() -> list[dict]:
    """Banking77 reserved for evaluation (77 unobserved banking intent categories)."""
    from datasets import load_dataset
    ds = load_dataset("mteb/banking77")
    names = sorted(list(set(ds["train"]["label_text"])))
    name_to_idx = {name: i for i, name in enumerate(names)}
    examples = []
    for row in ds["test"]:
        lbl_text = row["label_text"]
        examples.append(_ex_choice(
            state=row["text"],
            prompt="Quelle est la catégorie bancaire de cette demande ?",
            options=names,
            label=name_to_idx[lbl_text],
            source="banking77",
        ))
    return examples


def load_massive_eval() -> list[dict]:
    """MASSIVE reserved for evaluation (60 unobserved user intent categories)."""
    from datasets import load_dataset
    ds = load_dataset("SetFit/amazon_massive_intent_en-US")
    names = sorted(list(set(ds["train"]["label_text"])))
    name_to_idx = {name: i for i, name in enumerate(names)}
    examples = []
    for row in ds["test"]:
        lbl_text = row["label_text"]
        examples.append(_ex_choice(
            state=row["text"],
            prompt="What is the user's intent?",
            options=names,
            label=name_to_idx[lbl_text],
            source="massive",
        ))
    return examples


def load_sst5_eval() -> list[dict]:
    """SST-5 reserved for evaluation (5 unobserved fine-grained sentence polarity levels)."""
    from datasets import load_dataset
    ds = load_dataset("SetFit/sst5")
    levels = ["très négatif", "négatif", "neutre", "positif", "très positif"]
    examples = []
    for row in ds["test"]:
        examples.append(_ex_score(
            state=row["text"],
            prompt="Quelle est la polarité de cette phrase ?",
            levels=levels,
            label=row["label"],
            source="sst5_eval",
        ))
    return examples


def load_rte_eval() -> list[dict]:
    """RTE reserved for evaluation (binary entailment over an unobserved corpus).

    RTE is labelled entailment / not_entailment at the source, so unlike the MNLI and
    ANLI training sources it needs no 3-way collapse and stays near 50/50.
    """
    from datasets import load_dataset
    ds = load_dataset("nyu-mll/glue", "rte")
    examples = []
    for row in ds["validation"]:
        # GLUE RTE: 0 = entailment, 1 = not_entailment.
        examples.append(_ex_noul(
            state=row["sentence1"],
            statement=row["sentence2"],
            label=1 if row["label"] == 0 else 0,
            source="rte_eval",
        ))
    return examples


def load_scitail_eval() -> list[dict]:
    """SciTail reserved for evaluation (science-domain entailment, unobserved domain).

    Entailment pairs built from science exam questions: a domain none of the training
    Noul sources (qnli, mnli, boolq, anli) covers.
    """
    from datasets import load_dataset
    ds = load_dataset("allenai/scitail", "snli_format")
    examples = []
    for row in ds["test"]:
        examples.append(_ex_noul(
            state=row["sentence1"],
            statement=row["sentence2"],
            label=1 if row["gold_label"] == "entailment" else 0,
            source="scitail_eval",
        ))
    return examples


# ---------------------------------------------------------------------------
# Unified Construction Pipeline
# ---------------------------------------------------------------------------

def build(
    out: str,
    eval_out: str,
    max_per_source: int | None = 30000,
    seed: int = 42,
    eval_only: bool = False,
) -> tuple[int, int]:
    """Build train.jsonl and eval.jsonl dataset files with strict source isolation.

    eval_only rebuilds just the evaluation file, leaving an existing train.jsonl
    untouched: the eval registry gains sources more often than the training mixture does.
    """
    rng = random.Random(seed)
    train: list[dict] = []
    eval_: list[dict] = []

    # Theoretical isolation verification
    all_eval_sources = EVAL_CHOICE_SOURCES | EVAL_SCORE_SOURCES | EVAL_NOUL_SOURCES
    overlap = TRAIN_SOURCES.intersection(all_eval_sources)
    if overlap:
        raise ValueError(f"DATASET LEAK VIOLATION: datasets present in both train and eval registries: {overlap}")

    train_loaders = {
        "clinc150": load_clinc150,
        "ag_news": load_ag_news,
        "dbpedia_14": load_dbpedia,
        "go_emotions": load_go_emotions,
        "yelp_full": load_yelp_full,
        "amazon_reviews": load_amazon_reviews,
        "imdb": load_imdb,
        "tweet_sentiment": load_tweet_sentiment,
        "medical_triage": load_enterprise_medical_triage,
        "legal_triage": load_enterprise_legal_triage,
        "csat_sentiment": load_customer_satisfaction_csat,
        "mnli": load_mnli,
        "snli": load_snli,
        "boolq": load_boolq,
        "qnli": load_qnli,
        "anli": load_anli,
    }

    eval_loaders = {
        "banking77": load_banking77_eval,
        "massive": load_massive_eval,
        "sst5_eval": load_sst5_eval,
        "rte_eval": load_rte_eval,
        "scitail_eval": load_scitail_eval,
    }

    if eval_only:
        train_loaders = {}

    print("[build] Loading training sources...")
    # Deliberately fatal: a swallowed loader failure silently shrinks the mixture
    # (the first released training set lost tweet_sentiment and the synthetic
    # enterprise sources this way, with only a log line to show for it).
    for name, fn in train_loaders.items():
        exs = fn()
        if max_per_source and len(exs) > max_per_source:
            rng.shuffle(exs)
            exs = exs[:max_per_source]
        train.extend(exs)
        print(f"  + TRAIN : {name} -> {len(exs)} instances")

    print("\n[build] Loading reserved evaluation sources...")
    for name, fn in eval_loaders.items():
        # Deliberately fatal, unlike the training loop: a silently skipped eval source
        # produces a benchmark with a whole primitive missing, which is how the
        # published numbers came to cover 0 Noul examples.
        exs = fn()
        eval_.extend(exs)
        print(f"  + EVAL RESERVED : {name} -> {len(exs)} instances")

    missing = {"choice", "score", "noul"} - {ex["kind"] for ex in eval_}
    if missing:
        raise RuntimeError(
            f"Evaluation set covers no {sorted(missing)} example. Every primitive the "
            f"engine answers must be measured; add a reserved source for it."
        )

    # Strict empirical leak check. Compared against the declared training registry
    # rather than the loaded instances, so it still holds under eval_only.
    eval_srcs = {ex["source"] for ex in eval_}
    leak = (TRAIN_SOURCES | {ex["source"] for ex in train}).intersection(eval_srcs)
    if leak:
        raise RuntimeError(f"DATASET LEAK DETECTED BETWEEN TRAIN AND EVAL: {leak}")

    # Shuffle training instances to break consecutive blocks
    rng.shuffle(train)
    # Shuffle the eval set for the same reason: written in source order, any consumer
    # taking a prefix (validation NLL, --max-examples) silently measures one source.
    random.Random(seed).shuffle(eval_)

    Path(eval_out).parent.mkdir(parents=True, exist_ok=True)

    if not eval_only:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for ex in train:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(eval_out, "w", encoding="utf-8") as f:
        for ex in eval_:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    by_kind: dict[str, int] = {}
    for ex in eval_:
        by_kind[ex["kind"]] = by_kind.get(ex["kind"], 0) + 1

    print(f"\n=======================================================")
    print(f"Datasets generated successfully:")
    if eval_only:
        print(f"  TRAIN : untouched (eval-only build)")
    else:
        print(f"  TRAIN : {len(train):,} instances written to {out}")
    print(f"  EVAL  : {len(eval_):,} instances written to {eval_out}")
    print(f"          " + ", ".join(f"{k}={v:,}" for k, v in sorted(by_kind.items())))
    print(f"  Strict isolation guaranteed: 0 source overlap")
    print(f"=======================================================")

    return len(train), len(eval_)


def main():
    ap = argparse.ArgumentParser(description="Construct train and evaluation dataset mixtures for model training.")
    ap.add_argument("--out", default="data/train.jsonl", help="Train dataset output path")
    ap.add_argument("--eval-out", default="data/eval_reserved.jsonl", help="Reserved evaluation dataset output path")
    ap.add_argument("--max-per-source", type=int, default=30000,
                    help="Maximum instances per data source to balance classes")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--eval-only", action="store_true",
                    help="Rebuild only the evaluation file, leaving train.jsonl untouched")
    args = ap.parse_args()

    build(
        out=args.out,
        eval_out=args.eval_out,
        max_per_source=args.max_per_source,
        seed=args.seed,
        eval_only=args.eval_only,
    )


if __name__ == "__main__":
    main()
