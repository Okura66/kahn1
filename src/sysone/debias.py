"""Permutation-based positional debiasing for discrete option logit extraction.

Autoregressive language models inherently suffer from positional bias, frequently exhibiting
an artificial prior favoring early candidates (e.g. option 'A' or the midpoint).

Debiasing pipeline:
    1. Permutation generation: construct k candidate orderings (default k=3 for categorical
       choice; dual-pass bidirectional ascending/descending for ordinal scoring).
    2. Position remapping: invert each permuted display index back to the canonical option index.
    3. Probability-space aggregation: average distributions directly in the probability simplex
       (arithmetic mean or normalized geometric consensus). Averaging unconstrained logits
       is mathematically invalid as logit means do not represent linear belief mixtures.

Computational overhead:
    Requires k forward evaluation queries, but each query evaluates exactly one output token
    while reusing the shared cached state prefix, maintaining sub-millisecond incremental cost.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass


@dataclass
class Permutation:
    """Represents a discrete candidate permutation.

    Attributes:
        order: Mapping where order[i] is the canonical option index placed at display position i.
    """

    order: list[int]  # order[i] = canonical option placed at display position i

    def inverse(self) -> list[int]:
        """Compute inverse mapping: inv[canonical_idx] = displayed_position_idx."""
        inv = [0] * len(self.order)
        for pos, orig in enumerate(self.order):
            inv[orig] = pos
        return inv


def generate_permutations(n: int, k: int, seed: int | None = None) -> list[Permutation]:
    """Generate k distinct permutations of range(n).

    The identity permutation (natural order) is always preserved as the first entry (k >= 1).
    For k > 1, k - 1 distinct random permutations are sampled without replacement.
    """
    if n <= 1:
        return [Permutation(order=list(range(n)))]
    rng = random.Random(seed)
    perms: list[Permutation] = [Permutation(order=list(range(n)))]
    seen = {tuple(range(n))}
    attempts = 0
    while len(perms) < k and attempts < k * 20:
        order = list(range(n))
        rng.shuffle(order)
        key = tuple(order)
        if key not in seen:
            seen.add(key)
            perms.append(Permutation(order=order))
        attempts += 1
    # If distinct permutations cannot be generated (e.g. small n), return generated candidates
    return perms


def generate_bidirectional_permutations(n: int) -> list[Permutation]:
    """Generate canonical dual-pass bidirectional permutations for ordinal evaluation:
    Pass 1 (Ascending) : [0, 1, ..., n-1] (e.g. A=Level_1, ..., E=Level_M)
    Pass 2 (Descending): [n-1, n-2, ..., 0] (e.g. A=Level_M, ..., E=Level_1)

    For n <= 1, returns identity.
    """
    if n <= 1:
        return [Permutation(order=list(range(n)))]
    return [
        Permutation(order=list(range(n))),
        Permutation(order=list(reversed(range(n)))),
    ]


def remap_distribution(
    perm: Permutation,
    position_probs: list[float],
) -> list[float]:
    """Remap a position-indexed probability vector back to canonical option identity.

    Args:
        perm: The active permutation applied during prompt formatting.
        position_probs: Probabilities indexed by displayed candidate position.

    Returns:
        Probability distribution aligned with canonical option indices.
    """
    n = len(position_probs)
    out = [0.0] * n
    for pos, orig in enumerate(perm.order):
        out[orig] = position_probs[pos]
    return out


def average_distributions(
    remapped: list[list[float]],
    weights: list[float] | None = None,
) -> list[float]:
    """Compute the weighted arithmetic mean in the probability simplex.

    Args:
        remapped: List of probability vectors aligned to canonical option indices.
        weights: Optional non-negative weighting vector (normalized automatically).

    Returns:
        Averaged normalized probability distribution.
    """
    if not remapped:
        return []
    n = len(remapped[0])
    if weights is None:
        w = [1.0 / len(remapped)] * len(remapped)
    else:
        if len(weights) != len(remapped):
            raise ValueError("weights et remapped doivent avoir même longueur.")
        s = sum(weights)
        if s <= 0:
            raise ValueError("La somme des poids doit être > 0.")
        w = [x / s for x in weights]
    out = [0.0] * n
    for probs, wi in zip(remapped, w):
        for i in range(n):
            out[i] += wi * probs[i]
    return out


def geometric_mean_distributions(
    remapped: list[list[float]],
    weights: list[float] | None = None,
    eps: float = 1e-12,
) -> list[float]:
    """Compute the normalized geometric mean across remapped probability distributions.

    Enforces strict probabilistic consensus by heavily penalizing positional overconfidence
    unless mutually confirmed across inverse ordering passes.
    """
    if not remapped:
        return []
    n = len(remapped[0])
    if weights is None:
        w = [1.0 / len(remapped)] * len(remapped)
    else:
        if len(weights) != len(remapped):
            raise ValueError("weights et remapped doivent avoir même longueur.")
        s = sum(weights)
        if s <= 0:
            raise ValueError("La somme des poids doit être > 0.")
        w = [x / s for x in weights]

    import math
    log_probs = [0.0] * n
    for probs, wi in zip(remapped, w):
        for i in range(n):
            log_probs[i] += wi * math.log(max(probs[i], eps))

    max_log = max(log_probs)
    exps = [math.exp(lp - max_log) for lp in log_probs]
    total = sum(exps)
    if total <= 0:
        return [1.0 / n] * n
    return [e / total for e in exps]


def debias(
    per_perm_position_probs: list[tuple[Permutation, list[float]]],
) -> list[float]:
    """Execute end-to-end debiasing: remap display distributions and aggregate.

    Args:
        per_perm_position_probs: Sequence of (Permutation, position_probabilities) pairs.

    Returns:
        Canonical debiased probability distribution.
    """
    remapped = [remap_distribution(perm, probs) for perm, probs in per_perm_position_probs]
    return average_distributions(remapped)
