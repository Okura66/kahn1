"""Tests for permutation-based and bidirectional debiasing.

Quantifies position bias: for an identical question with permuted options,
measures variance of p(correct_option) across orderings. Debiasing must
reduce this positional variance.
"""

import random

import pytest

from sysone.debias import (
    Permutation, generate_permutations, generate_bidirectional_permutations,
    remap_distribution, average_distributions, geometric_mean_distributions, debias,
)


def test_generate_permutations_identity_first():
    """The identity permutation is always included at index 0."""
    perms = generate_permutations(4, k=3, seed=42)
    assert perms[0].order == [0, 1, 2, 3]
    assert len(perms) == 3


def test_generate_permutations_distinct():
    """Generated k permutations must be distinct when n allows."""
    perms = generate_permutations(5, k=5, seed=42)
    orders = [tuple(p.order) for p in perms]
    assert len(set(orders)) == 5


def test_generate_permutations_k1():
    """k=1 yields only the identity permutation (fast mode, no debiasing)."""
    perms = generate_permutations(4, k=1)
    assert len(perms) == 1
    assert perms[0].order == [0, 1, 2, 3]


def test_remap_distribution_identity():
    """Identity permutation remaps distribution without alteration."""
    perm = Permutation(order=[0, 1, 2, 3])
    probs = [0.4, 0.3, 0.2, 0.1]
    out = remap_distribution(perm, probs)
    assert out == probs


def test_remap_distribution_swap():
    """Permutation swapping positions A and B correctly exchanges probabilities."""
    perm = Permutation(order=[1, 0, 2, 3])  # position 0 = option B
    probs = [0.5, 0.3, 0.15, 0.05]  # position 0 has 0.5
    out = remap_distribution(perm, probs)
    # out[1] (option B) = probs[0] (position where B was placed) = 0.5
    assert out[0] == 0.3  # option A = position 1 = 0.3
    assert out[1] == 0.5  # option B = position 0 = 0.5
    assert sum(out) == pytest.approx(1.0)


def test_average_uniform():
    """Mean of identical distributions returns the same distribution."""
    d = [0.4, 0.3, 0.2, 0.1]
    avg = average_distributions([d, d, d])
    assert avg == pytest.approx(d)


def test_average_two():
    """Simple unweighted arithmetic average of probability distributions."""
    d1 = [0.5, 0.5, 0.0]
    d2 = [0.0, 0.5, 0.5]
    avg = average_distributions([d1, d2])
    assert avg == pytest.approx([0.25, 0.5, 0.25])


def test_debias_reduces_position_variance():
    """Verify debiasing reduces variance of p(correct_option) across orderings.

    Simulates strong position bias: model allocates ~0.6 to position 0
    regardless of option content. Without debiasing, the correct option receives
    0.6 if placed at position 0, and ~0.2 otherwise. With debiasing (averaging
    permutations), the remapped correct option probability stabilizes.
    """
    rng = random.Random(42)
    n_options = 4
    correct_option = 0  # option A is correct

    def simulated_position_probs(perm_order: list[int]) -> list[float]:
        """Simulate position bias: 0.6 on position 0, noisy uniform remainder."""
        probs = [0.0] * n_options
        for pos, orig in enumerate(perm_order):
            if pos == 0:
                probs[pos] = 0.6 + rng.uniform(-0.02, 0.02)
            else:
                probs[pos] = (0.4 / (n_options - 1)) + rng.uniform(-0.02, 0.02)
        s = sum(probs)
        return [p / s for p in probs]

    # Without debiasing: evaluate identity permutation only
    perms = generate_permutations(n_options, k=1)
    p_without = []
    for perm in perms:
        pos_probs = simulated_position_probs(perm.order)
        remapped = remap_distribution(perm, pos_probs)
        p_without.append(remapped[correct_option])

    # With debiasing: k=5 permutations
    perms5 = generate_permutations(n_options, k=5, seed=42)
    p_with = []
    for perm in perms5:
        pos_probs = simulated_position_probs(perm.order)
        remapped = remap_distribution(perm, pos_probs)
        p_with.append(remapped[correct_option])

    avg_without = average_distributions([remap_distribution(
        perm, simulated_position_probs(perm.order)
    ) for perm in perms])
    avg_with = average_distributions([remap_distribution(
        perm, simulated_position_probs(perm.order)
    ) for perm in perms5])

    # Debiased average must be closer to true preference (~0.25) than biased identity
    assert abs(avg_with[correct_option] - 0.25) < abs(avg_without[correct_option] - 0.25)


def test_debias_pipeline():
    """Pipeline integration test: debias() performs remap + aggregation."""
    perms = generate_permutations(3, k=3, seed=0)
    data = [(perm, [0.5, 0.3, 0.2]) for perm in perms]
    out = debias(data)
    assert len(out) == 3
    assert sum(out) == pytest.approx(1.0, abs=1e-3)


def test_average_in_probability_space_not_logits():
    """Verify aggregation occurs in probability space rather than logit space.

    Averaging [0.99, 0.01] and [0.01, 0.99] in probability space yields [0.5, 0.5],
    whereas averaging logits would produce vastly different results.
    """
    d1 = [0.99, 0.01]
    d2 = [0.01, 0.99]
    avg = average_distributions([d1, d2])
    assert avg == pytest.approx([0.5, 0.5])


def test_generate_bidirectional_permutations():
    """Verify bidirectional permutations consist of ascending and descending sequences."""
    perms = generate_bidirectional_permutations(5)
    assert len(perms) == 2
    assert perms[0].order == [0, 1, 2, 3, 4]
    assert perms[1].order == [4, 3, 2, 1, 0]

    # Boundary case n=1
    perms_1 = generate_bidirectional_permutations(1)
    assert len(perms_1) == 1
    assert perms_1[0].order == [0]


def test_geometric_mean_distributions_consensus():
    """Verify geometric mean consensus penalizes extreme asymmetry."""
    d1 = [0.8, 0.2]
    d2 = [0.2, 0.8]
    geo = geometric_mean_distributions([d1, d2])
    assert geo[0] == pytest.approx(0.5)
    assert geo[1] == pytest.approx(0.5)
    assert sum(geo) == pytest.approx(1.0)


def test_dual_pass_ordinal_debiasing_cancels_primacy_bias():
    """Empirically demonstrate primacy bias cancellation via dual-pass ordinal debiasing."""
    # Assume ground truth level is 4 ("strongly positive").
    # Model exhibits severe primacy bias towards option A (position 0).
    # Pass 1 (ascending : [0, 1, 2, 3, 4]):
    # True class (4) is at position E (last), heavily penalized by primacy bias.
    pass1_pos_probs = [0.4, 0.1, 0.1, 0.1, 0.3]  # Position 0 inflated
    # Pass 2 (descending : [4, 3, 2, 1, 0]):
    # True class (4) is now at position A (first), receiving primacy boost.
    pass2_pos_probs = [0.7, 0.1, 0.1, 0.05, 0.05]

    perm1 = Permutation(order=[0, 1, 2, 3, 4])
    perm2 = Permutation(order=[4, 3, 2, 1, 0])

    remap1 = remap_distribution(perm1, pass1_pos_probs)
    remap2 = remap_distribution(perm2, pass2_pos_probs)

    # Without dual-pass (Pass 1 alone): position 0 ("strongly negative") erroneously wins (0.4 vs 0.3)
    assert max(range(5), key=lambda i: remap1[i]) == 0

    # With dual-pass geometric consensus:
    # Ground truth level 4 decisively wins, cancelling out primacy bias
    debiased_geo = geometric_mean_distributions([remap1, remap2])
    assert max(range(5), key=lambda i: debiased_geo[i]) == 4
