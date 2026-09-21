"""Tests for dynamic data augmentation.

Validates the required transformations:
1. Permutation of candidate option order with exact label remapping.
2. Option cardinality sub-sampling (2 to 16 options).
3. Distractor injection from the shared distractor pool.
4. Random omission of correct answer mapping to fallback 'other' label (-1).
5. Template framing variability (bilingual FR and EN support).
6. Cross-primitive support across Choice, Score, and Noul tasks.
"""

import random
import pytest

from training.augment import (
    DistractorPool,
    AugmentedExample,
    augment_choice,
    augment_score,
    augment_noul,
    augment,
    AugmentingDataset,
    CHOICE_TEMPLATES,
    SCORE_TEMPLATES,
    NOUL_TEMPLATES,
)


@pytest.fixture
def sample_pool():
    examples = [
        {"kind": "choice", "options": ["Facturation", "Support technique", "Bug affichage"]},
        {"kind": "choice", "options": ["Remboursement", "Annulation", "Livraison"]},
        {"kind": "choice", "options": ["Mot de passe", "Compte bloqué"]},
    ]
    return DistractorPool.build(examples)


def test_distractor_pool_build(sample_pool):
    """Ensure distractor pool indexes all unique candidate options."""
    assert len(sample_pool.choice_options) == 8
    assert "Facturation" in sample_pool.choice_options
    assert "Compte bloqué" in sample_pool.choice_options


def test_distractor_pool_sample(sample_pool):
    """Ensure sampling respects exclusions and requested cardinality."""
    rng = random.Random(42)
    exclude = {"Facturation", "Support technique"}
    sampled = sample_pool.sample_choice(3, exclude=exclude, rng=rng)
    assert len(sampled) == 3
    for item in sampled:
        assert item not in exclude


def test_augment_choice_preserves_correct_option(sample_pool):
    """When correct option is retained, ensure presence and remapped label."""
    rng = random.Random(123)
    ex = {
        "state": "Mon écran reste noir lors du lancement.",
        "kind": "choice",
        "prompt": "Quel est le problème ?",
        "options": ["Bug affichage", "Facturation"],
        "label": 0,
        "source": "test_data",
    }
    # Force p_remove_correct = 0.0 to guarantee inclusion
    aug = augment_choice(ex, sample_pool, rng, p_remove_correct=0.0, min_options=3, max_options=5)

    assert aug.kind == "choice"
    assert aug.state == ex["state"]
    assert 3 <= len(aug.options) <= 5
    assert aug.label >= 0
    assert aug.options[aug.label] == "Bug affichage"
    assert aug.lang in {"FR", "EN"}


def test_augment_choice_remove_correct_sets_label_minus_one(sample_pool):
    """When correct option is removed, verify label is set to -1 (other fallback)."""
    rng = random.Random(456)
    ex = {
        "state": "Je souhaite changer mon mot de passe.",
        "kind": "choice",
        "prompt": "Quel est le problème ?",
        "options": ["Mot de passe", "Autre"],
        "label": 0,
        "source": "test_data",
    }
    # Force p_remove_correct = 1.0
    aug = augment_choice(ex, sample_pool, rng, p_remove_correct=1.0, min_options=2, max_options=4)

    assert aug.kind == "choice"
    assert aug.label == -1
    # The correct option must not appear in candidate list
    assert "Mot de passe" not in aug.options


def test_augment_choice_out_of_scope_handling(sample_pool):
    """Verify examples with initial label=-1 (e.g. out-of-scope) sample distractors and maintain label=-1."""
    rng = random.Random(789)
    ex = {
        "state": "Peux-tu me faire une recette de crêpes ?",
        "kind": "choice",
        "prompt": "Quelle est l'intention bancaire ?",
        "options": ["oos"],
        "label": -1,
        "source": "clinc150_oos",
    }
    aug = augment_choice(ex, sample_pool, rng, min_options=3, max_options=6)

    assert aug.kind == "choice"
    assert aug.label == -1
    assert 3 <= len(aug.options) <= 6
    assert "oos" not in aug.options


def test_augment_score():
    """For score questions, verify ordinal level semantic sequence is preserved while template varies."""
    rng = random.Random(42)
    levels = ["très mauvais", "mauvais", "moyen", "bon", "excellent"]
    ex = {
        "state": "Ce produit est de très haute qualité et fonctionne parfaitement.",
        "kind": "score",
        "prompt": "Donnez une note.",
        "levels": levels,
        "label": 4,
        "source": "yelp",
    }
    aug = augment_score(ex, rng)

    assert aug.kind == "score"
    assert aug.options == levels
    assert aug.levels == levels
    assert aug.label == 4
    assert aug.lang in {"FR", "EN"}


def test_augment_noul():
    """For noul questions, verify statement is preserved and label remains binary."""
    rng = random.Random(42)
    ex = {
        "state": "Le ciel est complètement bleu sans aucun nuage.",
        "kind": "noul",
        "statement": "Il fait beau.",
        "label": 1,
        "source": "mnli",
    }
    aug = augment_noul(ex, rng)

    assert aug.kind == "noul"
    assert aug.statement == "Il fait beau."
    assert aug.label == 1
    assert aug.lang in {"FR", "EN"}


def test_augmenting_dataset_iteration(sample_pool):
    """Verify AugmentingDataset yields dynamic augmented instances on access."""
    examples = [
        {
            "state": "Problème de connexion",
            "kind": "choice",
            "prompt": "Catégorie ?",
            "options": ["Support technique"],
            "label": 0,
            "source": "src1",
        },
        {
            "state": "Texte d'évaluation",
            "kind": "score",
            "prompt": "Note ?",
            "levels": ["1", "2", "3"],
            "label": 1,
            "source": "src2",
        },
    ]
    ds = AugmentingDataset(examples, sample_pool, seed=42)
    assert len(ds) == 2

    item0_call1 = ds[0]
    item0_call2 = ds[0]
    assert isinstance(item0_call1, AugmentedExample)
    assert isinstance(item0_call2, AugmentedExample)


def test_include_other_is_independent_of_label(sample_pool):
    """The fallback option's presence must not reveal whether it is the answer.

    Regression: include_other was derived as (label == -1), so 'Aucune de ces
    réponses' only ever appeared when it was correct. The model learned
    'fallback shown => fallback correct' and collapsed onto it at inference.
    """
    rng = random.Random(0)
    ex = {
        "state": "some customer message",
        "kind": "choice",
        "prompt": "Intent?",
        "options": ["billing", "delivery", "account"],
        "label": 0,
        "source": "unit",
    }
    # p_remove_correct=0 keeps every label >= 0, isolating the include_other draw.
    augs = [
        augment_choice(ex, sample_pool, rng, p_remove_correct=0.0, p_include_other=0.5)
        for _ in range(200)
    ]
    assert all(a.label >= 0 for a in augs)
    shown = sum(a.include_other for a in augs)
    # Both branches must occur: presence carries no information about the label.
    assert 0 < shown < len(augs)


def test_include_other_forced_when_fallback_is_the_answer(sample_pool):
    """When the correct option was withheld (label -1), the fallback must be shown."""
    rng = random.Random(1)
    ex = {
        "state": "out of scope message",
        "kind": "choice",
        "prompt": "Intent?",
        "options": [],
        "label": -1,
        "source": "unit",
    }
    for _ in range(20):
        aug = augment_choice(ex, sample_pool, rng)
        assert aug.label == -1
        assert aug.include_other is True
