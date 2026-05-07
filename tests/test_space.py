"""Tests for the Gradio space helpers (without launching the UI)."""

from __future__ import annotations

import numpy as np

from signbridge.space import (
    _capture_sign,
    _clear,
    _format_history,
    _new_session,
    _speak,
)


class TestSession:
    def test_new_session_is_empty(self) -> None:
        s = _new_session()
        assert s.sign_history == []
        assert s.last_sentence == ""
        assert s.last_audio_path is None


class TestFormatHistory:
    def test_empty_returns_hint(self) -> None:
        out = _format_history([])
        assert "no signs" in out.lower()

    def test_renders_signs_as_code(self) -> None:
        out = _format_history(["A", "B", "hello"])
        assert "`A`" in out
        assert "`B`" in out
        assert "`hello`" in out


class TestCaptureSign:
    def test_no_frame(self) -> None:
        s = _new_session()
        msg, history, new_state = _capture_sign(None, s)
        assert "no webcam" in msg.lower()
        assert new_state.sign_history == []

    def test_no_provider_says_unrecognized(self) -> None:
        # Without API keys, recognizer returns ("", 0.0) → UI shows hint.
        s = _new_session()
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        msg, history, new_state = _capture_sign(frame, s)
        assert "couldn't recognise" in msg.lower() or "couldn't recognize" in msg.lower()
        assert new_state.sign_history == []


class TestSpeak:
    def test_empty_history(self) -> None:
        s = _new_session()
        sentence, audio, new_state = _speak(s)
        assert "no signs" in sentence.lower()
        assert audio is None

    def test_with_history_invokes_composer_and_tts(self) -> None:
        s = _new_session()
        s.sign_history = ["L", "U", "C", "A", "S"]
        sentence, audio, new_state = _speak(s)
        # Naive joiner path
        assert "Lucas" in sentence
        # TTS produces a silent-stub WAV path
        assert audio is not None and audio != ""


class TestClear:
    def test_resets_state(self) -> None:
        s = _new_session()
        s.sign_history = ["A", "B"]
        s.last_sentence = "AB."
        s.last_audio_path = "/tmp/foo.wav"

        latest, history, sentence, audio, new_state = _clear(s)
        assert latest == ""
        assert "no signs" in history.lower()
        assert sentence == ""
        assert audio is None
        assert new_state.sign_history == []
        assert new_state.last_sentence == ""
        assert new_state.last_audio_path is None
