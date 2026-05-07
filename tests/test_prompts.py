"""Tests for the prompt-builder module."""

from signbridge.recognizer.prompts import (
    build_single_frame_prompt,
    build_multi_frame_prompt,
)
from signbridge.vocab import VOCAB_SET  # noqa: F401 — reserved for future vocab-coverage tests


class TestSingleFramePrompt:
    def test_returns_string(self):
        assert isinstance(build_single_frame_prompt(), str)

    def test_mentions_asl(self):
        assert "ASL" in build_single_frame_prompt()

    def test_includes_vocab_tokens(self):
        prompt = build_single_frame_prompt()
        for token in ("A", "Z", "0", "9", "hello", "thank_you", "unknown"):
            assert token in prompt, f"missing {token!r}"

    def test_demands_single_token_output(self):
        prompt = build_single_frame_prompt()
        assert any(s in prompt for s in ("Single token only", "single token", "ONE token"))

    def test_includes_domain_priming(self):
        prompt = build_single_frame_prompt()
        assert "ASL" in prompt
        assert any(p in prompt for p in ("fingerspelled", "fingerspelling"))


class TestMultiFramePrompt:
    def test_returns_string(self):
        assert isinstance(build_multi_frame_prompt(4), str)

    def test_includes_frame_markers(self):
        prompt = build_multi_frame_prompt(4)
        for i in range(1, 5):
            assert f"Frame {i}" in prompt, f"missing 'Frame {i}' marker"

    def test_mentions_motion(self):
        prompt = build_multi_frame_prompt(4)
        assert "motion" in prompt.lower() or "across" in prompt.lower()

    def test_demands_single_token_output(self):
        prompt = build_multi_frame_prompt(4)
        assert any(s in prompt for s in ("Single token only", "single token", "ONE token"))

    def test_includes_vocab(self):
        prompt = build_multi_frame_prompt(4)
        for token in ("A", "hello", "unknown"):
            assert token in prompt

    def test_n_frames_parameter(self):
        p2 = build_multi_frame_prompt(2)
        p4 = build_multi_frame_prompt(4)
        assert p2 != p4
        assert "Frame 1" in p2 and "Frame 2" in p2 and "Frame 3" not in p2
        assert "Frame 4" in p4

    def test_too_few_frames_raises(self):
        import pytest
        with pytest.raises(ValueError):
            build_multi_frame_prompt(1)
        with pytest.raises(ValueError):
            build_multi_frame_prompt(0)
