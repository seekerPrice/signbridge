"""Tests for the FastAPI backend."""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from signbridge.backend import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _frame_b64(rgb: tuple[int, int, int] = (128, 128, 128), size: int = 64) -> str:
    arr = np.full((size, size, 3), rgb, dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("ascii")


class TestHealth:
    def test_healthz_returns_ok(self, client: TestClient) -> None:
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    def test_info_returns_provider_block(self, client: TestClient) -> None:
        r = client.get("/info")
        assert r.status_code == 200
        body = r.json()
        for key in ("provider", "composer_model", "vlm_model", "tts_model", "recognizer_mode"):
            assert key in body


class TestRecognize:
    def test_empty_frame_rejected(self, client: TestClient) -> None:
        r = client.post("/recognize", json={"frame": ""})
        assert r.status_code == 400

    def test_invalid_base64_rejected(self, client: TestClient) -> None:
        r = client.post("/recognize", json={"frame": "%%%not-base64%%%"})
        assert r.status_code == 400

    def test_valid_frame_no_provider_returns_empty_token(self, client: TestClient) -> None:
        # Without API keys, the VLM path returns ("", 0.0). The endpoint should
        # still respond 200 OK.
        r = client.post("/recognize", json={"frame": _frame_b64()})
        assert r.status_code == 200
        body = r.json()
        assert body == {"token": "", "confidence": 0.0}

    def test_data_url_prefix_tolerated(self, client: TestClient) -> None:
        b64 = _frame_b64()
        r = client.post(
            "/recognize", json={"frame": f"data:image/jpeg;base64,{b64}"}
        )
        assert r.status_code == 200


class TestCompose:
    def test_empty_signs(self, client: TestClient) -> None:
        r = client.post("/compose", json={"signs": []})
        assert r.status_code == 200
        assert r.json() == {"sentence": ""}

    def test_fingerspelled_word(self, client: TestClient) -> None:
        r = client.post(
            "/compose", json={"signs": ["L", "U", "C", "A", "S"]}
        )
        assert r.status_code == 200
        # Naive joiner produces "Lucas."
        assert "Lucas" in r.json()["sentence"]

    def test_glosses(self, client: TestClient) -> None:
        r = client.post(
            "/compose", json={"signs": ["hello", "thank_you"]}
        )
        assert r.status_code == 200
        out = r.json()["sentence"].lower()
        assert "hello" in out
        assert "thank you" in out


class TestSpeak:
    def test_empty_text_rejected(self, client: TestClient) -> None:
        r = client.post("/speak", json={"text": ""})
        assert r.status_code == 400

    def test_returns_audio(self, client: TestClient) -> None:
        r = client.post("/speak", json={"text": "hello"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("audio/")
        assert len(r.content) > 0
