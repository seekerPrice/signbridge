"""Tests for the shared vocabulary module."""

from signbridge.vocab import VOCAB, VOCAB_SET, ALPHABET, DIGITS, WLASL_TOP50


def test_vocab_is_tuple_of_strings():
    assert isinstance(VOCAB, tuple)
    assert all(isinstance(t, str) for t in VOCAB)
    assert len(VOCAB) == len(set(VOCAB))  # no duplicates


def test_vocab_set_matches_tuple():
    assert VOCAB_SET == frozenset(VOCAB)


def test_alphabet_subset():
    assert all(letter in VOCAB_SET for letter in ALPHABET)
    assert len(ALPHABET) == 26


def test_digits_subset():
    assert all(d in VOCAB_SET for d in DIGITS)
    assert len(DIGITS) == 10


def test_wlasl_top50_subset():
    assert all(sign in VOCAB_SET for sign in WLASL_TOP50)
    assert len(WLASL_TOP50) >= 49


def test_unknown_sentinel_present():
    assert "unknown" in VOCAB_SET


def test_recognizer_uses_shared_vocab():
    from signbridge.recognizer.vlm import _VLM_VOCAB_SET
    assert _VLM_VOCAB_SET is VOCAB_SET


def test_classifier_uses_shared_vocab():
    from signbridge.recognizer.classifier import VOCABULARY
    assert tuple(VOCABULARY) == VOCAB
