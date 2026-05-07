"""Tests for the VLM-based sign recognizer."""

from __future__ import annotations

import numpy as np
import pytest

from signbridge.recognizer.vlm import (
    _frame_to_data_url,
    _normalise,
    _resolve_client,
    recognize_sign_from_frame,
)


class TestNormalise:
    def test_letter_uppercased(self) -> None:
        assert _normalise("a") == "A"

    def test_already_upper_letter(self) -> None:
        assert _normalise("Z") == "Z"

    def test_digit_kept_as_string(self) -> None:
        assert _normalise("7") == "7"

    def test_word_lowercased(self) -> None:
        assert _normalise("HELLO") == "hello"

    def test_strips_quotes(self) -> None:
        assert _normalise('"hello"') == "hello"

    def test_strips_punctuation(self) -> None:
        assert _normalise("hello.") == "hello"

    def test_takes_first_token_of_multi(self) -> None:
        assert _normalise("hello world") == "hello"

    def test_empty(self) -> None:
        assert _normalise("") == ""

    def test_underscore_preserved(self) -> None:
        assert _normalise("thank_you") == "thank_you"


class TestResolveClient:
    def test_no_keys_returns_none(self) -> None:
        # conftest cleared env; default provider 'amd' but no creds → None
        client, _ = _resolve_client()
        assert client is None

    def test_unknown_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIGNBRIDGE_PROVIDER", "garbage")
        client, _ = _resolve_client()
        assert client is None

    def test_amd_with_creds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIGNBRIDGE_PROVIDER", "amd")
        monkeypatch.setenv("AMD_DEV_CLOUD_BASE_URL", "https://example.invalid/v1")
        monkeypatch.setenv("AMD_DEV_CLOUD_API_KEY", "test-key")
        client, model = _resolve_client()
        assert client is not None
        assert "Qwen" in model or "Llama" in model

    def test_openai_with_creds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIGNBRIDGE_PROVIDER", "openai")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        client, model = _resolve_client()
        assert client is not None
        # When we fall to OpenAI we use a smaller default
        assert model.startswith("gpt-") or "Qwen" in model


class TestFrameToDataUrl:
    def test_valid_jpeg_data_url(self) -> None:
        frame = np.full((32, 32, 3), 128, dtype=np.uint8)
        url = _frame_to_data_url(frame)
        assert url.startswith("data:image/jpeg;base64,")
        # Body should decode without error
        import base64

        body = url.split(",", 1)[1]
        raw = base64.b64decode(body)
        assert len(raw) > 0


class TestRecognizeSignFromFrame:
    def test_no_client_returns_empty(self) -> None:
        # No env keys; should return ("", 0.0)
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        token, conf = recognize_sign_from_frame(frame)
        assert token == ""
        assert conf == 0.0

    def test_with_mock_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from signbridge.recognizer import vlm

        class _FakeChoice:
            def __init__(self, content: str) -> None:
                self.message = type("M", (), {"content": content})()

        class _FakeResp:
            def __init__(self, content: str) -> None:
                self.choices = [_FakeChoice(content)]

        class _FakeClient:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**_: object) -> _FakeResp:
                        return _FakeResp("A")

        monkeypatch.setattr(vlm, "_resolve_client", lambda: (_FakeClient(), "test"))
        frame = np.full((32, 32, 3), 200, dtype=np.uint8)
        token, conf = recognize_sign_from_frame(frame)
        assert token == "A"
        assert conf == 0.85

    def test_unknown_token_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from signbridge.recognizer import vlm

        class _FakeResp:
            def __init__(self) -> None:
                self.choices = [
                    type("C", (), {"message": type("M", (), {"content": "unknown"})()})()
                ]

        class _FakeClient:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**_: object) -> _FakeResp:
                        return _FakeResp()

        monkeypatch.setattr(vlm, "_resolve_client", lambda: (_FakeClient(), "test"))
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        token, conf = recognize_sign_from_frame(frame)
        assert token == ""
        assert conf == 0.0

    def test_provider_failure_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from signbridge.recognizer import vlm

        class _FailingClient:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**_: object) -> object:
                        raise RuntimeError("boom")

        monkeypatch.setattr(vlm, "_resolve_client", lambda: (_FailingClient(), "test"))
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        token, conf = recognize_sign_from_frame(frame)
        assert token == ""
        assert conf == 0.0
