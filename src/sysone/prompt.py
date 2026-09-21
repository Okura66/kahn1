"""Prompt template construction and prefix-caching alignment.

Canonical prompt structure:

    <system>You assess a state and answer with a single letter.</system>
    <user>
    ## State
    {state}

    ## Question
    {question_prompt}

    A. {option_0}
    B. {option_1}
    ...
    Z. None of these answers         <- appended if allow_other=True
    </user>
    <assistant>Answer:

Prefix Caching Optimization:
    The state representation appears first, and its template header must remain
    strictly byte-identical across all questions within a single query batch.
    vLLM hashes token blocks sequentially from index 0; placing the shared state
    at the very beginning maximizes KV cache reuse across multiple questions.

Binary (Noul) Primitives:
    Binary questions present a single proposition evaluated as true/false.
    They share the identical system/state prefix without displaying an A/B
    option list, with target continuation tokens restricted to 'yes' and 'no'.
"""

from __future__ import annotations

from dataclasses import dataclass

from .types import ChoiceQuestion, NoulQuestion, ScoreQuestion


SYSTEM_PROMPT = "You assess a state and answer with a single letter."

# Fallback option label.
# Appended at the end of the candidate list when allow_other=True for Choice questions.
# Assigned the final sequential letter and scored as a first-class token candidate.
OTHER_LABEL_TEXT = "None of these answers"

# Default framing for a Noul proposition. Overridable per question so that the
# bilingual NOUL_TEMPLATES augmentation actually reaches the prompt; previously
# this string was hardcoded here and every sampled template was discarded.
NOUL_PROMPT = "Is the following statement true for this state?"


def _format_options(options: list[str], offset: int = 0) -> str:
    """Format a list of candidate strings as sequentially lettered lines starting from offset (0 -> A, 1 -> B)."""
    lines = []
    for i, opt in enumerate(options):
        letter = chr(ord("A") + i + offset)
        lines.append(f"{letter}. {opt}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared State Prefix
# ---------------------------------------------------------------------------

def shared_state_prefix(state: str) -> str:
    """Build the prompt segment strictly identical across all questions in a query batch.

    Encompasses <system>, the opening <user> block, and the state text up to '## Question'.
    This common prefix is cached by the vLLM prefix-caching subsystem.
    """
    return (
        f"<system>{SYSTEM_PROMPT}</system>\n"
        f"<user>\n"
        f"## State\n{state}\n"
    )


# ---------------------------------------------------------------------------
# Full Prompt Assemblers
# ---------------------------------------------------------------------------

def build_choice_prompt(state: str, question: ChoiceQuestion, options: list[str], include_other: bool = True) -> str:
    """Construct the complete evaluation prompt for a categorical ChoiceQuestion.

    Args:
        state: Shared context state string.
        question: Choice question specification.
        options: Candidate string list, potentially permuted for debiasing.
        include_other: Whether to append the fallback 'Aucune de ces réponses' label.
    """
    body_lines = [_format_options(options)]
    n = len(options)
    if include_other and question.allow_other:
        letter = chr(ord("A") + n)
        body_lines.append(f"{letter}. {OTHER_LABEL_TEXT}")
    body = "\n".join(body_lines)
    return (
        f"{shared_state_prefix(state)}"
        f"\n## Question\n{question.prompt}\n"
        f"{body}\n"
        f"</user>\n"
        f"<assistant>Answer:"
    )


def build_score_prompt(state: str, question: ScoreQuestion, levels: list[str] | None = None) -> str:
    """Construct the evaluation prompt for an ordinal ScoreQuestion.

    Levels are formatted as lettered sequential candidates (A, B, C...).
    """
    levels_to_use = levels if levels is not None else question.levels
    body = _format_options(levels_to_use)
    return (
        f"{shared_state_prefix(state)}"
        f"\n## Question\n{question.prompt}\n"
        f"{body}\n"
        f"</user>\n"
        f"<assistant>Answer:"
    )


def build_noul_prompt(state: str, question: NoulQuestion) -> str:
    """Construct the evaluation prompt for a binary NoulQuestion.

    Presents the proposition directly under the shared state prefix without lettered options.
    """
    return (
        f"{shared_state_prefix(state)}"
        f"\n## Question\n{question.prompt or NOUL_PROMPT}\n"
        f"{question.statement}\n"
        f"</user>\n"
        f"<assistant>Answer:"
    )


# ---------------------------------------------------------------------------
# Prompt Specification & Token Resolution Metadata
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PromptSpec:
    """Constructed prompt container pairing raw prompt text with token resolution metadata."""

    full_text: str
    suffix: str  # Truncated suffix for token resolution (ends with 'Answer:')
    kind: str  # Question kind: 'choice', 'score', or 'noul'
    n_options: int  # Number of active candidate tokens to evaluate


def build_prompt_spec(state: str, question, options: list[str] | None = None, include_other: bool = True) -> PromptSpec:
    """Construct a typed PromptSpec for any supported question primitive.

    Args:
        state: Shared context state string.
        question: Question instance (ChoiceQuestion, ScoreQuestion, or NoulQuestion).
        options: Optional permuted candidate list (or alternative ordinal ordering).
        include_other: Whether to include the fallback other option for choice questions.
    """
    if isinstance(question, ChoiceQuestion):
        if options is None:
            options = question.options
        full = build_choice_prompt(state, question, options, include_other=include_other)
        n_opts = len(options) + (1 if include_other and question.allow_other else 0)
        return PromptSpec(full_text=full, suffix=full, kind="choice", n_options=n_opts)
    elif isinstance(question, ScoreQuestion):
        levels = options if options is not None else question.levels
        full = build_score_prompt(state, question, levels=levels)
        return PromptSpec(full_text=full, suffix=full, kind="score", n_options=len(levels))
    elif isinstance(question, NoulQuestion):
        full = build_noul_prompt(state, question)
        return PromptSpec(full_text=full, suffix=full, kind="noul", n_options=2)
    else:
        raise TypeError(f"Unsupported question type: {type(question)}")


def common_prefix_length(token_ids_lists: list[list[int]]) -> int:
    """Calculate the length of the identical common prefix across token ID sequences.

    Used to verify prefix sharing across multiple questions evaluated on the same state.
    """
    if not token_ids_lists:
        return 0
    ref = token_ids_lists[0]
    n = 0
    for i in range(min(len(lst) for lst in token_ids_lists)):
        val = ref[i]
        if all(lst[i] == val for lst in token_ids_lists):
            n += 1
        else:
            break
    return n
