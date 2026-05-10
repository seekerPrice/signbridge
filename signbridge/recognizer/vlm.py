"""VLM-based sign recognizer (Plan-B path, default for V1).

Sends a webcam frame + a structured prompt to a vision-language model
(Qwen2-VL / Llama-3.2-Vision / GPT-4o) and asks it to identify the
ASL sign or fingerspelled letter being shown. Returns a single token
plus a confidence estimate.

Why this is the V1 default
--------------------------
Training a custom WLASL classifier inside a 3-day window is risky.
A hosted VLM gets us to a working demo immediately and only requires
that the AMD Developer Cloud / HF Inference / OpenAI provider has a
working multimodal endpoint. The trained-classifier path
(`classifier.py` + `scripts/train_classifier.py`) is preserved as an
optional V2 upgrade and switched in via `SIGNBRIDGE_RECOGNIZER_MODE`.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
from functools import lru_cache

import numpy as np

# Closed vocabulary the VLM is asked to choose from. Imported from the
# shared `signbridge.vocab` module so the recognizer and the trained
# classifier (`signbridge.recognizer.classifier`) can never drift.
from signbridge.recognizer.prompts import (
    build_multi_frame_prompt,
    build_single_frame_prompt,
)
from signbridge.vocab import VOCAB_SET as _VLM_VOCAB_SET

logger = logging.getLogger(__name__)

DEFAULT_VLM_MODEL = os.getenv("SIGNBRIDGE_VLM_MODEL", "Qwen/Qwen3-VL-32B-Instruct")


@lru_cache(maxsize=4)
def _build_client(provider: str, base_url: str, api_key: str, model: str) -> tuple[object | None, str]:
    """Build (and cache) an OpenAI-compatible client for the given config.

    Cache key includes the full provider tuple so switching providers
    rebuilds; same provider re-uses the httpx connection pool. The
    `(None, model)` return is cached too — once a missing-deps state is
    detected we don't re-import on every frame.
    """
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("openai sdk not installed; recognizer returns 'unknown'.")
        return None, model
    if base_url:
        return OpenAI(base_url=base_url, api_key=api_key), model
    return OpenAI(api_key=api_key), model


def _resolve_client() -> tuple[object | None, str]:
    """Return (cached client, model_id) based on SIGNBRIDGE_PROVIDER env var."""
    provider = os.getenv("SIGNBRIDGE_PROVIDER", "amd").lower()

    if provider == "amd":
        base_url = os.getenv("AMD_DEV_CLOUD_BASE_URL", "").rstrip("/")
        api_key = os.getenv("AMD_DEV_CLOUD_API_KEY", "")
        if not base_url or not api_key:
            logger.info("AMD Dev Cloud not configured; recognizer in stub mode.")
            return None, DEFAULT_VLM_MODEL
        return _build_client(provider, base_url, api_key, DEFAULT_VLM_MODEL)

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.info("OPENAI_API_KEY not set; recognizer in stub mode.")
            return None, DEFAULT_VLM_MODEL
        model = os.getenv("SIGNBRIDGE_VLM_MODEL_OPENAI", "gpt-4o-mini")
        return _build_client(provider, "", api_key, model)

    if provider == "hf":
        api_key = os.getenv("HF_TOKEN", "")
        if not api_key:
            logger.info("HF_TOKEN not set; recognizer in stub mode.")
            return None, DEFAULT_VLM_MODEL
        base_url = os.getenv(
            "HF_INFERENCE_BASE_URL", "https://router.huggingface.co/v1"
        )
        model = os.getenv(
            "SIGNBRIDGE_VLM_MODEL_HF", "meta-llama/Llama-3.2-11B-Vision-Instruct"
        )
        return _build_client(provider, base_url, api_key, model)

    logger.warning("unknown SIGNBRIDGE_PROVIDER=%r; recognizer in stub mode.", provider)
    return None, DEFAULT_VLM_MODEL


def _frame_to_data_url(frame: np.ndarray) -> str:
    from PIL import Image

    from signbridge.imageio import array_to_rgb

    rgb = array_to_rgb(frame)
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _normalise(token: str) -> str:
    """Trim quotes/punctuation and clamp to one token."""
    t = re.sub(r'[\s.!?,;:"\'`]+', " ", token).strip()
    t = t.split()[0] if t else ""
    if len(t) == 1 and t.isalpha():
        return t.upper()
    if len(t) == 1 and t.isdigit():
        return t
    return t.lower()


def recognize_sign_from_frame(frame: np.ndarray) -> tuple[str, float]:
    """Run the VLM on a single frame.

    Returns (token, confidence). Confidence is a heuristic — VLMs don't
    expose true probabilities. We use a fixed 0.85 when the model returns
    a token from the vocab, 0.0 otherwise.
    """
    client, model = _resolve_client()
    if client is None:
        return "", 0.0

    try:
        data_url = _frame_to_data_url(frame)
        resp = client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_single_frame_prompt()},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=10,
        )
        raw = (resp.choices[0].message.content or "").strip()
        token = _normalise(raw)
    except Exception as exc:  # noqa: BLE001 — broad at the boundary on purpose
        # Log only the exception type — full message can include the request
        # URL with embedded credentials when the OpenAI-compatible client
        # bubbles up an httpx error. We pay log fidelity to avoid leaking the
        # provider key into a public HF Space stdout.
        logger.warning("VLM recognition failed: %s", type(exc).__name__)
        return "", 0.0

    # Suppress any token that isn't in the closed vocabulary the prompt
    # explicitly requested. Without this, a VLM that returns "letter" or
    # "no_sign" would be reported as a confident prediction.
    if token in {"", "unknown"} or token not in _VLM_VOCAB_SET:
        return "", 0.0
    return token, 0.85


def recognize_sign_from_frames(frames: list[np.ndarray]) -> tuple[str, float]:
    """Run the VLM on an ordered sequence of frames (multi-image prompt).

    Returns (token, confidence). Confidence semantics match the single-frame
    path: 0.85 when the VLM emits an in-vocab token, 0.0 otherwise.

    Raises:
        ValueError: if fewer than 2 frames are supplied (use the single-frame
            entry point for one frame).
    """
    if len(frames) < 2:
        raise ValueError(
            f"recognize_sign_from_frames requires at least 2 frames, got {len(frames)}"
        )

    client, model = _resolve_client()
    if client is None:
        return "", 0.0

    prompt = build_multi_frame_prompt(len(frames))
    content: list[dict[str, object]] = [{"type": "text", "text": prompt}]
    for frame in frames:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _frame_to_data_url(frame)},
            }
        )

    try:
        resp = client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=10,
        )
        raw = (resp.choices[0].message.content or "").strip()
        token = _normalise(raw)
    except Exception as exc:  # noqa: BLE001 — broad at the boundary on purpose
        # Same credential-leak guard as the single-frame path.
        logger.warning("multi-frame VLM recognition failed: %s", type(exc).__name__)
        return "", 0.0

    if token in {"", "unknown"} or token not in _VLM_VOCAB_SET:
        return "", 0.0
    return token, 0.85


def recognize_sign_from_video(video_path: str) -> tuple[str, float]:
    """Run the VLM on a recorded video clip via vLLM's video_url block.

    Pipeline: gradio webm → ffmpeg downscale (480p, 8 fps, ≤4 s, no audio,
    H.264 mp4) → base64 data URL → Qwen3-VL native video understanding.

    Why ffmpeg first (live-tested 2026-05-10 against the deployed
    signbridge-qwen3vl-8b-asl endpoint):
    - Direct webm upload fails — vLLM's opencv backend can't read VP8/VP9
      metadata, returns nonsense total_num_frames and Qwen3VLProcessor
      throws BadRequestError.
    - The deployed model has max_model_len=8192. A 10s/1080p video blows
      past that with a -3513 max_tokens budget. 480p @ 8fps capped at 4s
      gives ~1680 prompt_tokens, leaving headroom.

    Returns (token, confidence). 0.85 if the model emits an in-vocab
    token, 0.0 otherwise.
    """
    client, model = _resolve_client()
    if client is None:
        return "", 0.0

    if shutil.which("ffmpeg") is None:
        logger.warning("ffmpeg not on PATH; can't transcode video for VLM.")
        return "", 0.0

    tmp_mp4 = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_mp4.close()
    mp4_path = tmp_mp4.name
    try:
        ff = subprocess.run(
            [
                "ffmpeg", "-y", "-i", video_path,
                "-vf", "scale=480:-2,fps=8",
                "-c:v", "libx264", "-preset", "ultrafast",
                "-an", "-t", "4",
                mp4_path,
            ],
            capture_output=True, text=True, timeout=30,
        )
        if ff.returncode != 0:
            logger.warning(
                "ffmpeg transcode failed (rc=%d): %s",
                ff.returncode, ff.stderr[-300:],
            )
            return "", 0.0

        with open(mp4_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        data_url = f"data:video/mp4;base64,{b64}"

        resp = client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_multi_frame_prompt(8)},
                        {"type": "video_url", "video_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=10,
        )
        raw = (resp.choices[0].message.content or "").strip()
        token = _normalise(raw)
    except Exception as exc:  # noqa: BLE001 — broad at the boundary on purpose
        logger.warning("video VLM recognition failed: %s", type(exc).__name__)
        return "", 0.0
    finally:
        try:
            os.unlink(mp4_path)
        except OSError:
            pass

    if token in {"", "unknown"} or token not in _VLM_VOCAB_SET:
        return "", 0.0
    return token, 0.85
