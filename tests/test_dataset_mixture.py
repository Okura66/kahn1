"""Validation tests for the multi-task dataset mixture.

Verifies:
1. Ordinal directional reversal within `augment_score`.
2. Balanced stratified sampling within `sample_stratified_mixture`.
3. Structural validity of enterprise dataset loaders (medical, legal, CSAT).
4. Strict partition isolation between reserved TRAIN and EVAL sources.
"""

import random
import pytest

from training.augment import augment_score
from training.build_dataset import (
    TRAIN_SOURCES,
    EVAL_CHOICE_SOURCES,
    EVAL_SCORE_SOURCES,
    EVAL_NOUL_SOURCES,
    load_enterprise_medical_triage,
    load_enterprise_legal_triage,
    load_customer_satisfaction_csat,
)
from training.train_lora import sample_stratified_mixture


def test_augment_score_directional_reversal():
    """Verify scale reversal correctly reassigns ordinal levels and ground truth label."""
    rng = random.Random(42)
    ex = {
        "state": "Film magnifique et émouvant",
        "kind": "score",
        "levels": ["très négatif", "négatif", "neutre", "positif", "très positif"],
        "label": 4,  # "très positif"
        "source": "dummy",
    }

    # Test with forced reversal (prob=1.0)
    aug_rev = augment_score(ex, rng, reverse_prob=1.0)
    assert aug_rev.levels == ["très positif", "positif", "neutre", "négatif", "très négatif"]
    # True class was at index 4, now mapped to index 0 (5 - 1 - 4)
    assert aug_rev.label == 0
    assert aug_rev.levels[aug_rev.label] == "très positif"

    # Test without reversal (prob=0.0)
    aug_orig = augment_score(ex, rng, reverse_prob=0.0)
    assert aug_orig.levels == ex["levels"]
    assert aug_orig.label == 4


def test_sample_stratified_mixture_equal_quotas():
    """Verify stratified sampler enforces equal allocation across primitives."""
    # Synthetic unbalanced pool (analogous to base distribution with 100k choice, 10k score, 80k noul)
    dummy_pool = (
        [{"kind": "choice", "id": f"c_{i}"} for i in range(1000)]
        + [{"kind": "score", "id": f"s_{i}"} for i in range(200)]
        + [{"kind": "noul", "id": f"n_{i}"} for i in range(800)]
    )

    sampled = sample_stratified_mixture(dummy_pool, max_examples=300, seed=42)
    assert len(sampled) == 300

    counts = {k: sum(1 for x in sampled if x["kind"] == k) for k in ["choice", "score", "noul"]}
    assert counts["choice"] == 100
    assert counts["score"] == 100
    assert counts["noul"] == 100


def test_enterprise_loaders_schema_validity():
    """Verify enterprise loaders produce valid, strictly typed examples."""
    med_exs = load_enterprise_medical_triage()
    assert len(med_exs) >= 20
    for ex in med_exs:
        assert ex["kind"] == "score"
        assert len(ex["levels"]) == 4
        assert 0 <= ex["label"] < 4
        assert len(ex["state"]) > 10

    legal_exs = load_enterprise_legal_triage()
    assert len(legal_exs) >= 20
    for ex in legal_exs:
        assert ex["kind"] == "choice"
        assert len(ex["options"]) == 6
        assert 0 <= ex["label"] < 6
        assert len(ex["state"]) > 10

    csat_exs = load_customer_satisfaction_csat()
    assert len(csat_exs) >= 10
    for ex in csat_exs:
        assert ex["kind"] == "score"
        assert len(ex["levels"]) == 5
        assert 0 <= ex["label"] < 5


def test_strict_eval_split_isolation():
    """Ensure strict partition isolation: no holdout eval source in training mixture."""
    all_eval_sources = EVAL_CHOICE_SOURCES | EVAL_SCORE_SOURCES | EVAL_NOUL_SOURCES
    overlap = TRAIN_SOURCES.intersection(all_eval_sources)
    assert len(overlap) == 0, f"Fuite détectée entre train et eval : {overlap}"
    assert "banking77" in EVAL_CHOICE_SOURCES
    assert "massive" in EVAL_CHOICE_SOURCES
    assert "sst5_eval" in EVAL_SCORE_SOURCES
