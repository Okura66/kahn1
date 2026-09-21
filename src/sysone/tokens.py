"""Resolution and validation of single-token option labels.

Reliable logit scoring requires strict one-to-one mapping between candidate labels
and single vocabulary token IDs. Token IDs must never be guessed or assumed across models.

Key architectural rules:
    - Candidate labels for direct categorical evaluation are single capital letters:
      'A', 'B', ..., 'Z' (supporting up to 26 discrete options). For cardinality > 26,
      evaluation routes to a hierarchical two-stage candidate filter.
    - The prompt suffix ends strictly with 'Answer:' without trailing whitespace,
      ensuring that BPE/SentencePiece word-boundary prefixes (e.g. ' A') are resolved accurately.
    - For binary assessment (Noul), labels are fixed to 'yes' and 'no' under identical
      single-token verification.
    - Token collisions or multi-token splits immediately raise explicit exceptions,
      preventing silent model degradation.

Design rationale:
    Directly scoring full-text option strings is prohibited because surface-form competition
    mechanically penalizes longer textual options across variable token lengths. Single-letter
    labels act as neutral indirect pointers whose semantic content is bound in the prompt context.
"""

from __future__ import annotations

import string
from dataclasses import dataclass


def option_labels(n: int) -> list[str]:
    """Generate n unique sequential uppercase option labels: ['A', ..., 'Z'].

    For cardinality n > 26, direct single-token evaluation raises NotImplementedError,
    requiring two-stage candidate routing. The maximum theoretical cardinality cap is 255.
    """
    if n > 26:
        raise NotImplementedError(
            f"Cardinality {n} > 26 is not supported for direct Choice. "
            "Use the two-stage router (engine.evaluate_two_stage)."
        )
    if n < 1:
        raise ValueError("n must be >= 1")
    return [chr(ord("A") + i) for i in range(n)]


NOUL_LABELS = ["yes", "no"]


def noul_token_index(label: int) -> int:
    """Map a Noul dataset label to its index in NOUL_LABELS.

    The two conventions are deliberately opposite and must never be conflated:
      - dataset label: 1 = yes / entailment, 0 = no  (see training/build_dataset.py)
      - NOUL_LABELS index: 0 = 'yes', 1 = 'no'

    Indexing NOUL_LABELS with the raw dataset label therefore trains and calibrates
    on the inverted target. Always route Noul label -> token through this helper.
    """
    if label not in (0, 1):
        raise ValueError(f"Invalid noul label: {label!r} (expected 0 or 1).")
    return 0 if label == 1 else 1


@dataclass
class ResolvedTokens:
    """Container pairing symbolic string labels with their single vocabulary token IDs."""

    labels: list[str]
    token_ids: list[int]  # Strictly parallel to labels list

    def __post_init__(self):
        if len(self.labels) != len(self.token_ids):
            raise ValueError("labels and token_ids must have the same length.")
        if len(set(self.token_ids)) != len(self.token_ids):
            raise ValueError(
                f"Token collision: ids={self.token_ids} "
                f"for labels={self.labels}."
            )

    def id_of(self, label: str) -> int:
        """Return the integer token ID corresponding to a given string label."""
        idx = self.labels.index(label)
        return self.token_ids[idx]

    def label_of(self, token_id: int) -> str | None:
        """Return the string label matching a token ID, or None if not contained."""
        if token_id in self.token_ids:
            return self.labels[self.token_ids.index(token_id)]
        return None


def resolve_option_tokens(tokenizer, prompt_suffix: str, labels: list[str]) -> list[int]:
    """Resolve exactly one continuation token ID per option label against the prompt suffix.

    Validates that:
      1. Each label appends exactly one continuation token to the prompt prefix.
      2. All labels resolve under the SAME separator, so their logits are
         comparable surface forms (all bare, or all whitespace-prefixed).
      3. Token IDs are strictly unique across all labels (no collisions).

    Handles both direct prefix tokenizers (e.g., SentencePiece) and byte-level
    BPE tokenizers where prefix boundary bytes merge unless separated by whitespace (e.g., Qwen, Llama).
    """
    prefix_ids = tokenizer.encode(prompt_suffix, add_special_tokens=False)

    def _continuation(label: str, sep: str) -> list[int] | None:
        """Continuation tokens added by `sep + label`, or None if the prefix did not survive."""
        full_ids = tokenizer.encode(prompt_suffix + sep + label, add_special_tokens=False)
        if full_ids[:len(prefix_ids)] != prefix_ids:
            return None
        return full_ids[len(prefix_ids):]

    # The whole label set must resolve under ONE separator. Resolving labels
    # individually lets a byte-level BPE merge the boundary for some labels only
    # (e.g. ':no' merges but ':yes' does not), yielding a mix of bare and
    # space-prefixed tokens. Their logits then live on different surface forms
    # and comparing them is meaningless — which silently inverted Noul decisions.
    token_ids: list[int] = []
    attempts: dict[str, list[list[int] | None]] = {}
    for sep in ("", " "):
        conts = [_continuation(label, sep) for label in labels]
        attempts[sep] = conts
        if all(c is not None and len(c) == 1 for c in conts):
            token_ids = [c[0] for c in conts]
            break
    else:
        details = []
        for sep, conts in attempts.items():
            for label, cont in zip(labels, conts):
                if cont is None:
                    details.append(f"{sep + label!r}: prefix merged")
                elif len(cont) != 1:
                    details.append(f"{sep + label!r}: {len(cont)} tokens {cont!r}")
        raise ValueError(
            f"Cannot resolve labels {labels!r} to a single token under a shared "
            f"separator ('' or ' '). Details: {'; '.join(details)}. "
            f"Use shorter labels (a single uppercase letter). "
            f"NEVER score options as full text."
        )

    if len(set(token_ids)) != len(token_ids):
        raise ValueError(
            f"Token collision between labels {labels!r}: ids={token_ids!r}."
        )
    return token_ids


def resolve_choice_tokens(tokenizer, prompt_suffix: str, n_options: int) -> ResolvedTokens:
    """Helper to resolve uppercase option tokens for a Choice question of cardinality n_options."""
    labels = option_labels(n_options)
    ids = resolve_option_tokens(tokenizer, prompt_suffix, labels)
    return ResolvedTokens(labels=labels, token_ids=ids)


def resolve_noul_tokens(tokenizer, prompt_suffix: str) -> ResolvedTokens:
    """Helper to resolve binary decision tokens ('yes' / 'no') for a Noul question."""
    ids = resolve_option_tokens(tokenizer, prompt_suffix, NOUL_LABELS)
    return ResolvedTokens(labels=list(NOUL_LABELS), token_ids=ids)
