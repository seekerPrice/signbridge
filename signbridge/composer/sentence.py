"""Llama-3.1-8B sentence composer.

Takes a stream of sign tokens (English glosses + fingerspelled letters)
and composes a grammatical English sentence. Backed by an OpenAI-compatible
endpoint — works with AMD Developer Cloud (vLLM), HF Inference, or OpenAI.

ASL is not English-word-by-English-word; it has its own grammar (topic-comment,
non-manual markers, spatial referents). For V1 we keep this simple: the LLM
gets a prompt explaining the sign sequence is ASL gloss and asked to render
it as natural English. Day 2 we may swap in a sign-language-specific model.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Sequence

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are SignBridge, a translator from American Sign Language (ASL) gloss to natural spoken English.

You will receive a sequence of ASL signs as a list. Some entries are full ASL signs (e.g. "hello", "name", "thank_you"). Others are individual letters from fingerspelling (uppercase A-Z) — concatenate consecutive letters into spelled-out words.

Rules:
1. Output ONLY the spoken English sentence. No explanations, no quotation marks, no preamble.
2. Convert ASL grammar to natural English (e.g. "NAME ME LUCAS" → "My name is Lucas.").
3. Concatenate consecutive uppercase single letters into a spelled word: ["L","U","C","A","S"] → "Lucas".
4. If the sequence is incomplete or ambiguous, prefer the most natural plausible English.
5. Never invent content not implied by the signs.
6. End with appropriate punctuation."""


@lru_cache(maxsize=4)
def _build_client(provider: str, base_url: str, api_key: str, model: str) -> tuple[object | None, str]:
    """Build (and cache) an OpenAI-compatible client for the given config."""
    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except ImportError:
        logger.warning("openai sdk not installed; composer returns naive joiner.")
        return None, model
    if base_url:
        return OpenAI(base_url=base_url, api_key=api_key), model
    return OpenAI(api_key=api_key), model


def _resolve_client() -> tuple[object | None, str]:
    """Return (cached client, model_id) based on SIGNBRIDGE_PROVIDER env var."""
    provider = os.getenv("SIGNBRIDGE_PROVIDER", "amd").lower()
    composer_model = os.getenv(
        "SIGNBRIDGE_COMPOSER_MODEL", "meta-llama/Llama-3.1-8B-Instruct"
    )

    if provider == "amd":
        base_url = os.getenv("AMD_DEV_CLOUD_BASE_URL", "").rstrip("/")
        api_key = os.getenv("AMD_DEV_CLOUD_API_KEY", "")
        if not base_url or not api_key:
            logger.info("AMD Dev Cloud not configured; falling back to naive joiner.")
            return None, composer_model
        return _build_client(provider, base_url, api_key, composer_model)

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            logger.info("OPENAI_API_KEY not set; falling back to naive joiner.")
            return None, composer_model
        model = os.getenv("SIGNBRIDGE_COMPOSER_MODEL_OPENAI", "gpt-4o-mini")
        return _build_client(provider, "", api_key, model)

    if provider == "hf":
        api_key = os.getenv("HF_TOKEN", "")
        if not api_key:
            logger.info("HF_TOKEN not set; falling back to naive joiner.")
            return None, composer_model
        base_url = os.getenv(
            "HF_INFERENCE_BASE_URL", "https://router.huggingface.co/v1"
        )
        return _build_client(provider, base_url, api_key, composer_model)

    logger.warning("unknown SIGNBRIDGE_PROVIDER=%r; using naive joiner.", provider)
    return None, composer_model


def _naive_join(signs: Sequence[str]) -> str:
    """Best-effort fallback: concatenate fingerspelled letters and lowercase glosses."""
    out: list[str] = []
    buf: list[str] = []
    for s in signs:
        if len(s) == 1 and s.isalpha() and s.isupper():
            buf.append(s)
            continue
        if buf:
            out.append("".join(buf).capitalize())
            buf.clear()
        # turn "thank_you" → "thank you"
        out.append(s.replace("_", " "))
    if buf:
        out.append("".join(buf).capitalize())
    sentence = " ".join(out).strip()
    if sentence and sentence[-1] not in ".!?":
        sentence += "."
    return sentence[:1].upper() + sentence[1:] if sentence else ""


def compose_sentence(signs: Sequence[str]) -> str:
    """Public entry-point. Returns a single English sentence."""
    if not signs:
        return ""

    client, model = _resolve_client()
    # System prompt examples use Python-list syntax (e.g. ["L","U","C","A","S"]);
    # match that format here so the LLM sees inputs and examples consistently.
    user_prompt = f"ASL signs: {list(signs)!r}"

    if client is None:
        return _naive_join(signs)

    try:
        resp = client.chat.completions.create(  # type: ignore[attr-defined]
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=120,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001 — broad catch is intentional at the boundary
        # Log only the exception type; full message can include the request
        # URL with embedded credentials when the OpenAI-compatible client
        # surfaces an httpx error.
        logger.warning("composer LLM call failed: %s", type(exc).__name__)
        return _naive_join(signs)

    cleaned = _strip_quotes(text)
    if not cleaned:
        # LLM returned empty content — fall back to the naive joiner so the
        # demo still produces *something* readable instead of silently
        # playing no audio.
        logger.info("composer LLM returned empty content; using naive joiner.")
        return _naive_join(signs)
    return cleaned


def _strip_quotes(text: str) -> str:
    return re.sub(r'^["\']|["\']$', "", text).strip()
