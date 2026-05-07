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
import io
import logging
import os

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel, Field

from signbridge.composer.sentence import compose_sentence
from signbridge.recognizer.vlm import recognize_sign_from_frame
from signbridge.voice.tts import synthesize_speech

logger = logging.getLogger(__name__)

app = FastAPI(title="SignBridge backend", version="0.1.0")


class RecognizeRequest(BaseModel):
    frame: str = Field(..., description="Base64-encoded JPEG/PNG frame.")


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
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
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
    if not req.frame:
        raise HTTPException(status_code=400, detail="frame must be non-empty")
    frame = _decode_b64_image(req.frame)
    token, conf = recognize_sign_from_frame(frame)
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
