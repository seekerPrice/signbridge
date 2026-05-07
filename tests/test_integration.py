"""Integration tests — exercise multi-step user flows end-to-end."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from signbridge.backend import app
from signbridge.space import _capture_sign, _clear, _new_session, _speak


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _frame(rgb: tuple[int, int, int] = (180, 200, 160), size: int = 96) -> np.ndarray:
    return np.full((size, size, 3), rgb, dtype=np.uint8)


def _frame_b64(rgb: tuple[int, int, int] = (180, 200, 160), size: int = 96) -> str:
    arr = _frame(rgb, size)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class TestUserFlowFingerspell:
    """User fingerspells L-U-C-A-S then presses Speak."""

    def test_via_space_helpers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Stub VLM to return one letter at a time.
        from signbridge.recognizer import vlm

        responses = iter(["L", "U", "C", "A", "S"])

        class _Resp:
            def __init__(self, c: str) -> None:
                self.choices = [type("C", (), {"message": type("M", (), {"content": c})()})()]

        class _FakeClient:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**_: object) -> _Resp:
                        return _Resp(next(responses))

        monkeypatch.setattr(vlm, "_resolve_client", lambda: (_FakeClient(), "test"))

        state = _new_session()
        for _ in range(5):
            _, _, state = _capture_sign(_frame(), state)
        assert state.sign_history == ["L", "U", "C", "A", "S"]

        sentence, audio_path, state = _speak(state)
        # Composer fallback (no API keys for composer in this test) → naive joiner
        assert "Lucas" in sentence
        assert audio_path  # silent-stub WAV exists

    def test_via_backend_endpoints(self, client: TestClient) -> None:
        # Direct multi-step flow over HTTP, exercising every endpoint.
        for _letter in "LUCAS":
            r = client.post("/recognize", json={"frame": _frame_b64()})
            assert r.status_code == 200
            # No API keys → token is "" but endpoint succeeds.
            assert r.json()["token"] == ""

        # Compose a manually-curated sequence
        r = client.post("/compose", json={"signs": ["L", "U", "C", "A", "S"]})
        assert r.status_code == 200
        assert "Lucas" in r.json()["sentence"]

        # Speak
        r = client.post("/speak", json={"text": "My name is Lucas."})
        assert r.status_code == 200
        assert len(r.content) > 0


class TestClearResetsCleanly:
    def test_full_round_trip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from signbridge.recognizer import vlm

        class _FakeClient:
            class chat:  # noqa: N801
                class completions:  # noqa: N801
                    @staticmethod
                    def create(**_: object):
                        return type(
                            "R",
                            (),
                            {
                                "choices": [
                                    type(
                                        "C",
                                        (),
                                        {"message": type("M", (), {"content": "hello"})()},
                                    )()
                                ]
                            },
                        )()

        monkeypatch.setattr(vlm, "_resolve_client", lambda: (_FakeClient(), "test"))

        state = _new_session()
        _, _, state = _capture_sign(_frame(), state)
        _, _, state = _capture_sign(_frame(), state)
        assert state.sign_history == ["hello", "hello"]

        sentence, audio, state = _speak(state)
        assert sentence
        assert audio

        latest, history, sentence_box, audio_out, state = _clear(state)
        assert state.sign_history == []
        assert state.last_sentence == ""
        assert state.last_audio_path is None
        assert "no signs" in history.lower()


class TestEdgeCases:
    def test_huge_sign_sequence(self, client: TestClient) -> None:
        # 200 fingerspelled letters — make sure compose endpoint doesn't crash.
        signs = list("ABCDEFGHIJ" * 20)
        r = client.post("/compose", json={"signs": signs})
        assert r.status_code == 200
        assert r.json()["sentence"]  # non-empty

    def test_unicode_in_compose(self, client: TestClient) -> None:
        # Synthetic unicode token should pass through naive joiner unscathed.
        r = client.post("/compose", json={"signs": ["héllo", "wörld"]})
        assert r.status_code == 200

    def test_speak_very_long_text(self, client: TestClient) -> None:
        r = client.post("/speak", json={"text": "a " * 500})
        assert r.status_code == 200

    def test_recognize_jpeg_with_data_url_jpg(self, client: TestClient) -> None:
        b64 = _frame_b64()
        r = client.post(
            "/recognize", json={"frame": f"data:image/jpg;base64,{b64}"}
        )
        # Slightly malformed data URL (jpg vs jpeg) — should still work via tolerant decoder.
        assert r.status_code == 200

    def test_recognize_png_frame(self, client: TestClient) -> None:
        arr = _frame()
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        r = client.post("/recognize", json={"frame": b64})
        assert r.status_code == 200

    def test_compose_with_only_punctuation_glosses(self, client: TestClient) -> None:
        # Tokens that are 1 char, lowercase letters — should not be misread as fingerspelling.
        r = client.post("/compose", json={"signs": ["a", "b"]})
        assert r.status_code == 200
        # Naive joiner only treats UPPERCASE single letters as fingerspelling.
        # Lowercase 'a' / 'b' are full glosses → should appear with a space, no concat.
        assert r.json()["sentence"] == "A b."

    def test_health_after_recognize_failure(self, client: TestClient) -> None:
        # Even after a 400, /healthz should still respond.
        client.post("/recognize", json={"frame": "%%%bad%%%"})
        r = client.get("/healthz")
        assert r.status_code == 200


class TestBackendInfoEndpoint:
    def test_info_reflects_env(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIGNBRIDGE_PROVIDER", "openai")
        monkeypatch.setenv(
            "SIGNBRIDGE_COMPOSER_MODEL", "meta-llama/Llama-3.1-8B-Instruct"
        )
        r = client.get("/info")
        assert r.status_code == 200
        body = r.json()
        assert body["provider"] == "openai"
        assert body["composer_model"].endswith("Llama-3.1-8B-Instruct")
