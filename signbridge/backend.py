"""FastAPI inference backend.

Deployable on AMD Developer Cloud (single MI300X). Wraps the same modules
the Gradio Space uses, exposed over HTTP so the Space can run as a thin
client when separated. For Day-1 dev we run this in-process.

Endpoints:
- POST /recognize   { frame: base64-jpeg }    → { token: str, confidence: float }
- POST /compose     { signs: list[str] }      → { sentence: str }
- POST /speak       { text: str }             → audio/wav
- GET  /healthz                               → 200 OK
- GET  /info                                  → provider / model config
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from signbridge.composer.sentence import compose_sentence
from signbridge.imageio import load_rgb
from signbridge.recognizer.vlm import (
    recognize_sign_from_frame,
    recognize_sign_from_frames,
)
from signbridge.voice.tts import synthesize_speech

logger = logging.getLogger(__name__)

# Per-frame size cap: a 1280x720 JPEG q85 is ~120 KB → ~160 KB base64.
# 3 MB cap is generous and defends /recognize against accidental DoS via
# huge payloads (deep-check audit B.F3).
_MAX_FRAME_B64_LEN = 3 * 1024 * 1024
_MAX_FRAMES_IN_BURST = 12

app = FastAPI(title="SignBridge backend", version="0.1.0")


class RecognizeRequest(BaseModel):
    """Either single-frame or multi-frame; exactly one must be provided."""

    frame: Optional[str] = Field(
        default=None,
        description="Base64-encoded JPEG/PNG (single-frame mode).",
        max_length=_MAX_FRAME_B64_LEN,
    )
    frames: Optional[list[str]] = Field(
        default=None,
        description="Ordered list of base64 frames (multi-frame mode).",
        min_length=1,
        max_length=_MAX_FRAMES_IN_BURST,
    )

    @model_validator(mode="after")
    def _exactly_one_payload(self) -> "RecognizeRequest":
        if (self.frame is None) == (self.frames is None):
            raise ValueError("provide exactly one of 'frame' or 'frames'")
        if self.frames is not None:
            for f in self.frames:
                if len(f) > _MAX_FRAME_B64_LEN:
                    raise ValueError(
                        f"each frame must be at most {_MAX_FRAME_B64_LEN} bytes (base64)"
                    )
        return self


class RecognizeResponse(BaseModel):
    token: str
    confidence: float


class ComposeRequest(BaseModel):
    signs: list[str]


class ComposeResponse(BaseModel):
    sentence: str


class SpeakRequest(BaseModel):
    text: str


class InfoResponse(BaseModel):
    provider: str
    composer_model: str
    vlm_model: str
    tts_model: str
    recognizer_mode: str


def _decode_b64_image(b64: str) -> np.ndarray:
    try:
        # tolerate optional data URL prefix
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        raw = base64.b64decode(b64)
        return load_rgb(raw)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"bad frame: {exc}") from exc


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    return InfoResponse(
        provider=os.getenv("SIGNBRIDGE_PROVIDER", "amd"),
        composer_model=os.getenv(
            "SIGNBRIDGE_COMPOSER_MODEL", "meta-llama/Llama-3.1-8B-Instruct"
        ),
        vlm_model=os.getenv("SIGNBRIDGE_VLM_MODEL", "Qwen/Qwen2-VL-7B-Instruct"),
        tts_model=os.getenv(
            "SIGNBRIDGE_TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2"
        ),
        recognizer_mode=os.getenv("SIGNBRIDGE_RECOGNIZER_MODE", "vlm"),
    )


@app.post("/recognize", response_model=RecognizeResponse)
def recognize(req: RecognizeRequest) -> RecognizeResponse:
    if req.frames is not None:
        if len(req.frames) < 2:
            raise HTTPException(
                status_code=400,
                detail="multi-frame recognition needs at least 2 frames",
            )
        decoded_frames = [_decode_b64_image(b) for b in req.frames]
        token, conf = recognize_sign_from_frames(decoded_frames)
        return RecognizeResponse(token=token, confidence=conf)
    if not req.frame:
        raise HTTPException(status_code=400, detail="frame must be non-empty")
    decoded = _decode_b64_image(req.frame)
    token, conf = recognize_sign_from_frame(decoded)
    return RecognizeResponse(token=token, confidence=conf)


@app.post("/compose", response_model=ComposeResponse)
def compose(req: ComposeRequest) -> ComposeResponse:
    return ComposeResponse(sentence=compose_sentence(req.signs))


@app.post("/speak")
def speak(req: SpeakRequest) -> FileResponse:
    if not req.text:
        raise HTTPException(status_code=400, detail="text must be non-empty")
    path = synthesize_speech(req.text)
    if not path:
        raise HTTPException(status_code=500, detail="TTS unavailable")
    return FileResponse(path, media_type="audio/wav")


def main() -> None:
    import uvicorn

    uvicorn.run(
        "signbridge.backend:app",
        host="0.0.0.0",
        port=int(os.getenv("SIGNBRIDGE_BACKEND_PORT", "8000")),
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
