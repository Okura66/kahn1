"""Tests for shared prefix caching alignment.

Verifies that for N questions evaluated over the same state, all prompts share an
identical common prefix (comparing token ID sequences). This invariant ensures
optimal vLLM prefix-caching reuse.
"""

import pytest

from sysone.prompt import (
    shared_state_prefix, build_choice_prompt, build_score_prompt,
    build_noul_prompt, build_prompt_spec, common_prefix_length,
)
from sysone.types import ChoiceQuestion, ScoreQuestion, NoulQuestion


def test_shared_state_prefix_identical():
    """State prefix is strictly identical for identical input state."""
    state = "Un client est frustré."
    p1 = shared_state_prefix(state)
    p2 = shared_state_prefix(state)
    assert p1 == p2
    assert "## State" in p1
    assert state in p1


def test_prefix_common_across_questions():
    """N questions evaluated over identical state share an identical prompt prefix."""
    state = "Un utilisateur écrit au support : « Je veux annuler. »"
    q1 = ChoiceQuestion(key="q1", prompt="Intention ?",
                          options=["Annuler", "Rembourser", "Aide"])
    q2 = ScoreQuestion(key="q2", prompt="Satisfaction ?",
                        levels=["faible", "moyen", "élevé"])
    q3 = NoulQuestion(key="q3", statement="L'utilisateur veut partir.")

    spec1 = build_prompt_spec(state, q1)
    spec2 = build_prompt_spec(state, q2)
    spec3 = build_prompt_spec(state, q3)

    # State prefix must be a strict prefix of every question prompt
    pre = shared_state_prefix(state)
    assert spec1.full_text.startswith(pre)
    assert spec2.full_text.startswith(pre)
    assert spec3.full_text.startswith(pre)


def test_common_prefix_length_with_mock_tokenizer():
    """Verify common prefix length exceeds zero with mock tokenizer."""
    class MockTokenizer:
        def encode(self, text, add_special_tokens=False):
            # Tokenize by word split for test verification
            return [hash(w) % 10000 for w in text.split()]

    tok = MockTokenizer()
    state = "Un client est frustré et veut annuler."
    q1 = ChoiceQuestion(key="q1", prompt="Intention ?",
                          options=["Annuler", "Aide"])
    q2 = ScoreQuestion(key="q2", prompt="Niveau ?",
                        levels=["faible", "élevé"])
    spec1 = build_prompt_spec(state, q1)
    spec2 = build_prompt_spec(state, q2)
    ids1 = tok.encode(spec1.full_text)
    ids2 = tok.encode(spec2.full_text)
    cp = common_prefix_length([ids1, ids2])
    # Shared prefix contains ~8 words -> common prefix must be >= 8
    assert cp >= 8, f"Common prefix too short: {cp}"


def test_prompt_ends_with_answer_no_space():
    """Prompt must end exactly with 'Answer:' without trailing whitespace."""
    state = "x"
    q = ChoiceQuestion(key="q", prompt="p", options=["a", "b"])
    spec = build_prompt_spec(state, q)
    assert spec.full_text.rstrip().endswith("Answer:"), \
        f"Prompt must end with 'Answer:', got '{spec.full_text[-20:]}'"
    # No trailing space allowed
    assert not spec.full_text.endswith("Answer: "), \
        "Trailing whitespace prohibited — leading space belongs to token candidate ' A'"


def test_choice_with_other_appends_option():
    """allow_other=True appends the fallback option label."""
    state = "x"
    q = ChoiceQuestion(key="q", prompt="p", options=["a", "b"], allow_other=True)
    spec = build_prompt_spec(state, q)
    assert "None of these answers" in spec.full_text
    # n_options accounts for the fallback option
    assert spec.n_options == 3


def test_choice_without_other():
    """allow_other=False omits the fallback option label."""
    state = "x"
    q = ChoiceQuestion(key="q", prompt="p", options=["a", "b"], allow_other=False)
    spec = build_prompt_spec(state, q)
    assert "None of these answers" not in spec.full_text
    assert spec.n_options == 2


# ---------------------------------------------------------------------------
# Chat formats
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fmt", ["chatml", "qwen3"])
def test_chat_formats_share_the_state_prefix(fmt):
    """Every question of a state starts with the same system + user + state text."""
    state = "Order #4411 was billed twice."
    specs = [
        build_prompt_spec(state, ChoiceQuestion(key="a", prompt="Team?", options=["billing", "tech"]), fmt=fmt),
        build_prompt_spec(state, ScoreQuestion(key="b", prompt="Urgency?", levels=["low", "high"]), fmt=fmt),
        build_prompt_spec(state, NoulQuestion(key="c", statement="A refund is requested."), fmt=fmt),
    ]
    pre = shared_state_prefix(state, fmt)
    assert pre.startswith("<|im_start|>system\n") and pre.endswith(f"## State\n{state}\n")
    for s in specs:
        assert s.full_text.startswith(pre)
        assert s.suffix == s.full_text


def test_chat_format_endings():
    q = NoulQuestion(key="c", statement="A refund is requested.")
    assert build_prompt_spec("s", q, fmt="chatml").full_text.endswith("<|im_end|>\n<|im_start|>assistant\n")
    assert build_prompt_spec("s", q, fmt="qwen3").full_text.endswith(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n")
    # The default layout is unchanged.
    assert build_prompt_spec("s", q).full_text.endswith("</user>\n<assistant>Answer:")


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        shared_state_prefix("s", "json")


def test_chat_format_labels_resolve_with_a_qwen_tokenizer():
    """With a real Qwen tokenizer the letters and yes/no are single tokens after the header."""
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", local_files_only=True)
    except Exception:
        pytest.skip("Qwen tokenizer not cached")
    from sysone.tokens import resolve_choice_tokens, resolve_noul_tokens
    q = ChoiceQuestion(key="a", prompt="Team?", options=["billing", "tech", "sales"])
    spec = build_prompt_spec("Order #4411 was billed twice.", q, fmt="chatml")
    res = resolve_choice_tokens(tok, spec.suffix, spec.n_options)
    assert [tok.decode([t]) for t in res.token_ids] == ["A", "B", "C", "D"]
    noul = build_prompt_spec("s", NoulQuestion(key="c", statement="x"), fmt="chatml")
    assert [tok.decode([t]) for t in resolve_noul_tokens(tok, noul.suffix).token_ids] == ["yes", "no"]


def test_prompt_format_is_resolved_from_the_model(tmp_path):
    from sysone.engine import resolve_prompt_format

    assert resolve_prompt_format("Okura66/Kahn1-Qwen3.5-4B") == "qwen3"
    assert resolve_prompt_format("Qwen/Qwen3.5-4B") == "qwen3"
    assert resolve_prompt_format("Okura66/Kahn1-Qwen2.5-3B") == "tags"
    assert resolve_prompt_format("Okura66/Kahn1-Qwen3.5-4B", "tags") == "tags"   # explicit wins
    (tmp_path / "config.json").write_text('{"model_type": "qwen3_5_text"}')
    assert resolve_prompt_format(str(tmp_path)) == "qwen3"                       # local dir: config
    (tmp_path / "config.json").write_text('{"model_type": "qwen2"}')
    assert resolve_prompt_format(str(tmp_path)) == "tags"
