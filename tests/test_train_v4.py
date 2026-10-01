"""The v4 training pieces: answer-position loss, left-padded collation, fixed-option augmentation."""

import random

import pytest

from training.augment import AugmentingDataset, DistractorPool, augment


class _Tok:
    """Whitespace tokenizer: enough to check collation."""
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [len(w) + 1 for w in text.split()]


def test_collate_left_pads_and_keeps_the_end():
    from training.train_lora import collate_fn

    batch = collate_fn([("a bb ccc", 7), ("dddd", 9)], _Tok(), max_len=2)
    assert batch["input_ids"].tolist() == [[3, 4], [0, 5]]          # cut on the left, padded on the left
    assert batch["attention_mask"].tolist() == [[1, 1], [0, 1]]
    assert batch["position_ids"].tolist() == [[0, 1], [0, 0]]
    assert batch["answer_ids"].tolist() == [7, 9]


def test_collate_single_row_needs_no_mask():
    from training.train_lora import collate_fn

    batch = collate_fn([("a bb", 7)], _Tok())
    assert set(batch) == {"input_ids", "answer_ids"}


def test_answer_loss_matches_the_full_label_loss():
    """Reading one position gives the same loss as labels masked everywhere but the answer."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from training.train_lora import answer_loss

    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=50))
    model.eval()
    ids = torch.tensor([[5, 6, 7, 8]])
    answer = torch.tensor([9])
    new = answer_loss(model, {"input_ids": ids, "answer_ids": answer}).item()
    full = torch.cat([ids, answer[:, None]], dim=1)
    labels = torch.full_like(full, -100)
    labels[0, -1] = 9
    old = model(input_ids=full, labels=labels).loss.item()
    assert new == pytest.approx(old, rel=1e-5)


def test_yesno_question_is_sometimes_asked_as_itself():
    ex = {"state": "A policy.", "kind": "noul", "statement": "Is the refund permitted?", "label": 0,
          "form": "yesno_question", "source": "teacher-judge"}
    pool = DistractorPool.build([ex])
    seen = [augment(ex, pool, random.Random(i)) for i in range(200)]
    own = [a for a in seen if a.prompt == ex["statement"]]
    assert 40 < len(own) < 110 and all(a.label == 0 for a in own)     # JEV form: no polarity flip
    assert any(a.prompt != ex["statement"] for a in seen)                # templates still used


def test_fixed_score_keeps_its_question():
    ex = {"state": "A ticket.", "kind": "score", "prompt": "Which risk tier applies?",
          "levels": ["low", "medium", "high"], "label": 2, "source": "teacher-rubric", "fixed_options": True}
    pool = DistractorPool.build([ex])
    a = augment(ex, pool, random.Random(0))
    assert a.prompt == "Which risk tier applies?" and a.levels == ex["levels"] and a.label == 2


def test_fixed_options_keep_their_question_and_answers():
    ex = {"state": "A long article.", "kind": "choice", "prompt": "Why did Si retire?",
          "options": ["money", "health", "love", "war"], "label": 2, "source": "quality", "fixed_options": True}
    pool = DistractorPool.build([ex, {"state": "s", "kind": "choice", "prompt": "p", "options": ["card_arrival"],
                                      "label": 0, "source": "banking77"}])
    assert pool.choice_options == ["card_arrival"]            # fixed options stay out of the pool
    rng = random.Random(0)
    for _ in range(200):
        a = augment(ex, pool, rng)
        assert a.prompt == "Why did Si retire?"
        assert set(a.options) <= set(ex["options"]) and "card_arrival" not in a.options
        if a.label >= 0:
            assert a.options[a.label] == "love"
        else:
            assert "love" not in a.options and a.include_other


def test_collate_pads_the_candidates():
    from training.train_lora import Target, collate_fn

    batch = collate_fn([Target("a bb", 7, (7, 8, 9), 0, True), Target("c", 4, (3, 4), 1, False)], _Tok())
    assert batch["option_ids"].tolist() == [[7, 8, 9], [3, 4, 3]]
    assert batch["option_mask"].tolist() == [[True, True, True], [True, True, False]]
    assert batch["answer_pos"].tolist() == [0, 1] and batch["ordinal"].tolist() == [True, False]


class _FixedLogits:
    """A model whose last-position logits are given: enough to check the loss terms."""

    def __init__(self, logits):
        self.logits = logits

    def __call__(self, input_ids, logits_to_keep=1, **_):
        from types import SimpleNamespace
        return SimpleNamespace(logits=self.logits[:, None, :])


def _batch(torch, option_ids, mask, pos, ordinal):
    return {"input_ids": torch.zeros(len(pos), 1, dtype=torch.long),
            "answer_ids": torch.tensor([o[p] for o, p in zip(option_ids, pos)]),
            "option_ids": torch.tensor(option_ids), "option_mask": torch.tensor(mask),
            "answer_pos": torch.tensor(pos), "ordinal": torch.tensor(ordinal)}


def test_restricted_loss_reads_the_candidates_only():
    torch = pytest.importorskip("torch")
    from training.train_lora import answer_loss, answer_terms

    logits = torch.tensor([[0.0, 2.0, 1.0, 5.0, -1.0]])      # token 3 is not a candidate
    batch = _batch(torch, [[1, 2, 4]], [[True, True, True]], [1], [False])
    expected = -torch.log_softmax(torch.tensor([2.0, 1.0, -1.0]), 0)[1]
    assert answer_terms(_FixedLogits(logits), batch)["options"].item() == pytest.approx(expected.item())
    assert answer_loss(_FixedLogits(logits), batch, restrict=True).item() == pytest.approx(expected.item())
    full = -torch.log_softmax(logits[0], 0)[2]
    assert answer_loss(_FixedLogits(logits), batch).item() == pytest.approx(full.item())
    # A padded candidate takes no probability.
    padded = _batch(torch, [[1, 2, 4, 1]], [[True, True, True, False]], [1], [False])
    assert answer_terms(_FixedLogits(logits), padded)["options"].item() == pytest.approx(expected.item())


def test_emd_grows_with_the_distance_to_the_true_level():
    torch = pytest.importorskip("torch")
    from training.train_lora import answer_loss, answer_terms

    def emd(peak, ordinal=True):
        logits = torch.full((1, 5), -30.0)
        logits[0, peak] = 30.0                                # all the mass on one level
        batch = _batch(torch, [[0, 1, 2, 3, 4]], [[True] * 5], [0], [ordinal])
        return answer_terms(_FixedLogits(logits), batch)["emd"].item()

    assert emd(0) == pytest.approx(0.0, abs=1e-6)
    assert [round(emd(p)) for p in range(5)] == [0, 1, 2, 3, 4]
    assert emd(4, ordinal=False) == 0.0                         # Choice and Noul rows: no term

    logits = torch.tensor([[0.0, 1.0, 2.0]])
    batch = _batch(torch, [[0, 1, 2]], [[True] * 3], [0], [True])
    terms = answer_terms(_FixedLogits(logits), batch)
    both = answer_loss(_FixedLogits(logits), batch, restrict=True, ordinal_weight=0.5).item()
    assert both == pytest.approx(terms["options"].item() + 0.5 * terms["emd"].item())


def test_correct_reads_the_argmax_among_candidates():
    torch = pytest.importorskip("torch")
    from training.train_lora import answer_terms

    logits = torch.tensor([[0.0, 2.0, 1.0, 9.0], [0.0, 2.0, 1.0, 9.0]])   # token 3 is no candidate
    batch = _batch(torch, [[0, 1, 2], [0, 1, 2]], [[True] * 3] * 2, [1, 2], [False, False])
    assert answer_terms(_FixedLogits(logits), batch)["correct"].tolist() == [1.0, 0.0]
