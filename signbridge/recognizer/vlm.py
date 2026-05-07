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

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_VLM_MODEL = os.getenv("SIGNBRIDGE_VLM_MODEL", "Qwen/Qwen2-VL-7B-Instruct")

# Closed vocabulary the VLM is asked to choose from. Same shape as
# `classifier.VOCABULARY` but expressed as a prompt, not a softmax.
_VLM_VOCAB = (
    "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z "
    "0 1 2 3 4 5 6 7 8 9 "
    "hello thank_you name please sorry yes no good bad help "
    "want like love family friend mother father sister brother child "
    "home school work eat drink water food more finish today tomorrow "
    "yesterday where what who why when how go come "
    "see know understand think feel happy sad tired hungry wait "
    "unknown"
)

_PROMPT = (
    "You are an expert in American Sign Language (ASL). Look at this image of a "
    "single signed gesture. Identify which ASL sign or fingerspelled letter is "
    "being shown.\n\n"
    "Reply with EXACTLY ONE token from this list, no other text, no quotes, no "
    "explanation:\n"
    f"{_VLM_VOCAB}\n\n"
    "Rules:\n"
    "- Single uppercase letter (A-Z) for fingerspelling letters.\n"
    "- Single digit (0-9) for fingerspelled numbers.\n"
    "- Lowercase word with underscores for full-word signs (e.g. thank_you).\n"
    "- 'unknown' if no sign is visible or the gesture isn't in the list.\n"
    "- Do NOT explain. Do NOT add punctuation. Single token only."
)


def _resolve_client() -> tuple[object | None, str]:
    """Return (openai-compat client, model_id) based on SIGNBRIDGE_PROVIDER."""
    provider = os.getenv("SIGNBRIDGE_PROVIDER", "amd").lower()

    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("openai sdk not installed; recognizer returns 'unknown'.")
        return None, DEFAULT_VLM_MODEL

    if provider == "amd":
        base_url = os.getenv("AMD_DEV_CLOUD_BASE_URL", "").rstrip("/")
        api_key = os.getenv("AMD_DEV_CLOUD_API_KEY", "")
        if not base_url or not api_key:
            logger.info("AMD Dev Cloud not configured; recognizer in stub mode.")
            return None, DEFAULT_VLM_MODEL
        return OpenAI(base_url=base_url, api_key=api_key), DEFAULT_VLM_MODEL

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.info("OPENAI_API_KEY not set; recognizer in stub mode.")
            return None, DEFAULT_VLM_MODEL
        return OpenAI(api_key=api_key), os.getenv(
            "SIGNBRIDGE_VLM_MODEL_OPENAI", "gpt-4o-mini"
        )

    if provider == "hf":
        api_key = os.getenv("HF_TOKEN", "")
        if not api_key:
            logger.info("HF_TOKEN not set; recognizer in stub mode.")
            return None, DEFAULT_VLM_MODEL
        return (
            OpenAI(
                base_url=os.getenv(
                    "HF_INFERENCE_BASE_URL",
                    "https://api-inference.huggingface.co/v1",
                ),
                api_key=api_key,
            ),
            DEFAULT_VLM_MODEL,
        )

    logger.warning("unknown SIGNBRIDGE_PROVIDER=%r; recognizer in stub mode.", provider)
    return None, DEFAULT_VLM_MODEL


def _frame_to_data_url(frame: np.ndarray) -> str:
    from PIL import Image

    img = Image.fromarray(frame)
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
                        {"type": "text", "text": _PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=10,
        )
        raw = (resp.choices[0].message.content or "").strip()
        token = _normalise(raw)
    except Exception:  # noqa: BLE001 — broad at the boundary on purpose
        logger.exception("VLM recognition failed; returning stub.")
        return "", 0.0

    if token in {"", "unknown"}:
        return "", 0.0
    return token, 0.85
