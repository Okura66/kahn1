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
