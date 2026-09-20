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
            f"Cardinalité {n} > 26 non supportée en Choice direct. "
            "Utiliser le système à deux étages (engine.evaluate_two_stage)."
        )
    if n < 1:
        raise ValueError("n doit être >= 1")
    return [chr(ord("A") + i) for i in range(n)]


NOUL_LABELS = ["yes", "no"]


@dataclass
class ResolvedTokens:
    """Container pairing symbolic string labels with their single vocabulary token IDs."""

    labels: list[str]
    token_ids: list[int]  # Strictly parallel to labels list

    def __post_init__(self):
        if len(self.labels) != len(self.token_ids):
            raise ValueError("labels et token_ids doivent avoir même longueur.")
        if len(set(self.token_ids)) != len(self.token_ids):
            raise ValueError(
                f"Collision de tokens : ids={self.token_ids} "
                f"pour labels={self.labels}."
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
      2. Token IDs are strictly unique across all labels (no collisions).

    Handles both direct prefix tokenizers (e.g., SentencePiece) and byte-level
    BPE tokenizers where prefix boundary bytes merge unless separated by whitespace (e.g., Qwen, Llama).
    """
    prefix_ids = tokenizer.encode(prompt_suffix, add_special_tokens=False)
    token_ids: list[int] = []
    for label in labels:
        full_ids = tokenizer.encode(prompt_suffix + label, add_special_tokens=False)
        if full_ids[:len(prefix_ids)] == prefix_ids:
            cont = full_ids[len(prefix_ids):]
        else:
            # Handle tokenizers that merge boundary characters across the suffix colon (e.g. ':A').
            # Test natural whitespace-prefixed tokenization (' A').
            full_ids_sp = tokenizer.encode(prompt_suffix + " " + label, add_special_tokens=False)
            if full_ids_sp[:len(prefix_ids)] == prefix_ids:
                cont = full_ids_sp[len(prefix_ids):]
            else:
                cont = []

        if len(cont) == 0:
            raise ValueError(
                f"Le label {label!r} n'ajoute aucun token au préfixe "
                f"(prefix={len(prefix_ids)} ids, full={len(full_ids)}). "
                f"Le tokenizer a peut-être fusionné le label."
            )
        if len(cont) != 1:
            raise ValueError(
                f"Le label {label!r} produit {len(cont)} tokens de "
                f"continuation {cont!r}; attendu exactement 1. "
                f"Choisir un label plus court (lettre majuscule). "
                f"Ne JAMAIS scorer les options en toutes lettres."
            )

        token_ids.append(cont[0])

    if len(set(token_ids)) != len(token_ids):
        raise ValueError(
            f"Collision de tokens entre labels {labels!r}: ids={token_ids!r}."
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
