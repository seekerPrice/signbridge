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


class TestRecognizeSignFromFrames:
    def test_too_few_frames_raises(self):
        from signbridge.recognizer.vlm import recognize_sign_from_frames

        with pytest.raises(ValueError):
            recognize_sign_from_frames([])
        with pytest.raises(ValueError):
            recognize_sign_from_frames([np.zeros((32, 32, 3), dtype=np.uint8)])

    def test_no_client_returns_empty(self):
        from signbridge.recognizer.vlm import recognize_sign_from_frames

        frames = [np.full((32, 32, 3), 200, dtype=np.uint8) for _ in range(4)]
        token, conf = recognize_sign_from_frames(frames)
        assert token == ""
        assert conf == 0.0

    def test_with_mock_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from signbridge.recognizer import vlm

        captured: dict = {}

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
                    def create(**kwargs: object) -> _FakeResp:
                        captured.update(kwargs)
                        return _FakeResp("hello")

        monkeypatch.setattr(vlm, "_resolve_client", lambda: (_FakeClient(), "test"))
        frames = [np.full((32, 32, 3), 100 + i, dtype=np.uint8) for i in range(4)]
        token, conf = vlm.recognize_sign_from_frames(frames)
        assert token == "hello"
        assert conf == 0.85
        # Verify multi-image payload shape: 1 message with 1 text + 4 image_urls
        msgs = captured["messages"]
        assert len(msgs) == 1
        content = msgs[0]["content"]
        assert sum(1 for c in content if c["type"] == "text") == 1
        assert sum(1 for c in content if c["type"] == "image_url") == 4

    def test_off_vocab_token_suppressed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from signbridge.recognizer import vlm

        class _FakeClient:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**_: object) -> object:
                        return type(
                            "R",
                            (),
                            {
                                "choices": [
                                    type(
                                        "C",
                                        (),
                                        {"message": type("M", (), {"content": "fingerspelling"})()},
                                    )()
                                ]
                            },
                        )()

        monkeypatch.setattr(vlm, "_resolve_client", lambda: (_FakeClient(), "test"))
        frames = [np.full((32, 32, 3), 100, dtype=np.uint8) for _ in range(4)]
        token, conf = vlm.recognize_sign_from_frames(frames)
        # 'fingerspelling' is not in VOCAB_SET → suppressed
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
        frames = [np.full((32, 32, 3), 0, dtype=np.uint8) for _ in range(3)]
        token, conf = vlm.recognize_sign_from_frames(frames)
        assert token == ""
        assert conf == 0.0
