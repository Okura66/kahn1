"""Token resolution validation tests.

Verifies:
  - Collisions detected (raises exception)
  - Multi-token labels detected (raises exception)
  - Empty continuation labels detected (raises exception)
  - Uniqueness of target token candidate mappings
"""

import pytest


# --- Mock tokenizer for test execution (no GPU required) ---

class MockTokenizer:
    """Mock tokenizer simulating BPE where ' A' -> 1 token, ' B' -> 1 token, etc.

    encode(text) returns a sequence of simulated token IDs:
      - "Answer:" -> [100]
      - "Answer:A" -> [100, 200]  (A = 1 token)
      - "Answer:B" -> [100, 201]
      - "Answer:billing" -> [100, 300, 301, 302]  (multi-token)
      - "Answer:yes" -> [100, 400]
      - "Answer:no" -> [100, 401]
    """
    def encode(self, text, add_special_tokens=False):
        # strip trailing for matching but keep as-is
        if text.endswith("Answer:A"):
            return [100, 200]
        if text.endswith("Answer:B"):
            return [100, 201]
        if text.endswith("Answer:C"):
            return [100, 202]
        if text.endswith("Answer:D"):
            return [100, 203]
        if text.endswith("Answer:billing"):
            return [100, 300, 301, 302]  # multi-token
        if text.endswith("Answer:yes"):
            return [100, 400]
        if text.endswith("Answer:no"):
            return [100, 401]
        if text.endswith("Answer:AA"):
            return [100, 999, 998]  # multi-token for "AA"
        if text.endswith("Answer:"):
            return [100]
        # default fallback: 1 token
        return [100, hash(text) % 1000 + 5000]

    @property
    def pad_token(self):
        return None

    @pad_token.setter
    def pad_token(self, v):
        pass


def test_resolve_single_token_labels():
    from sysone.tokens import resolve_option_tokens
    tok = MockTokenizer()
    ids = resolve_option_tokens(tok, "Answer:", ["A", "B", "C", "D"])
    assert ids == [200, 201, 202, 203]
    assert len(set(ids)) == 4  # uniqueness


def test_resolve_noul_labels():
    from sysone.tokens import resolve_option_tokens
    tok = MockTokenizer()
    ids = resolve_option_tokens(tok, "Answer:", ["yes", "no"])
    assert ids == [400, 401]
    assert len(set(ids)) == 2


def test_multi_token_label_raises():
    """Multi-token label must raise an exception rather than silently degrading."""
    from sysone.tokens import resolve_option_tokens
    tok = MockTokenizer()
    with pytest.raises(ValueError, match="tokens de continuation"):
        resolve_option_tokens(tok, "Answer:", ["billing"])


def test_collision_raises():
    """Token collision across distinct labels must raise ValueError."""
    from sysone.tokens import resolve_option_tokens
    # MockTokenizer returning identical token ID for A and B
    class CollisionTokenizer(MockTokenizer):
        def encode(self, text, add_special_tokens=False):
            if text.endswith("Answer:A") or text.endswith("Answer:B"):
                return [100, 200]  # collision
            return super().encode(text, add_special_tokens)
    tok = CollisionTokenizer()
    with pytest.raises(ValueError, match="Collision"):
        resolve_option_tokens(tok, "Answer:", ["A", "B"])


def test_empty_continuation_raises():
    """Label appending zero continuation tokens must raise ValueError."""
    from sysone.tokens import resolve_option_tokens
    class EmptyTokenizer(MockTokenizer):
        def encode(self, text, add_special_tokens=False):
            if text.endswith("Answer:"):
                return [100]
            if text.endswith("Answer:X"):
                return [100]  # appends nothing
            return super().encode(text, add_special_tokens)
    tok = EmptyTokenizer()
    with pytest.raises(ValueError, match="n'ajoute aucun token"):
        resolve_option_tokens(tok, "Answer:", ["X"])


def test_option_labels_26():
    from sysone.tokens import option_labels
    labels = option_labels(26)
    assert labels == [chr(ord("A") + i) for i in range(26)]
    assert labels[0] == "A"
    assert labels[-1] == "Z"


def test_option_labels_over_26_raises():
    from sysone.tokens import option_labels
    with pytest.raises(NotImplementedError, match="deux étages"):
        option_labels(27)


def test_resolved_tokens_dataclass():
    from sysone.tokens import ResolvedTokens
    rt = ResolvedTokens(labels=["A", "B"], token_ids=[200, 201])
    assert rt.id_of("A") == 200
    assert rt.label_of(201) == "B"
    assert rt.label_of(999) is None


def test_resolved_tokens_mismatch_raises():
    from sysone.tokens import ResolvedTokens
    with pytest.raises(ValueError, match="même longueur"):
        ResolvedTokens(labels=["A", "B"], token_ids=[200])


def test_bpe_fusing_tokenizer():
    """Simulate BPE tokenizer fusing where whitespace continuation yields clean single tokens."""
    from sysone.tokens import resolve_option_tokens

    class FusingBpeTokenizer:
        def encode(self, text, add_special_tokens=False):
            if text == "Answer:":
                return [10, 20]
            if text == "Answer:A":
                return [10, 9999]  # fused :A
            if text == "Answer:B":
                return [10, 9998]  # fused :B
            if text == "Answer: A":
                return [10, 20, 301]  # valid single continuation ' A'
            if text == "Answer: B":
                return [10, 20, 302]  # valid single continuation ' B'
            return [10, 20]

    tok = FusingBpeTokenizer()
    ids = resolve_option_tokens(tok, "Answer:", ["A", "B"])
    assert ids == [301, 302]
