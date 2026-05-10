"""Text-to-speech via Coqui XTTS-v2.

Loaded lazily because XTTS-v2 weights are ~2 GB. While the model isn't
loaded (e.g. on a fresh local dev box), we fall back to a silent WAV stub
so the Gradio app still produces something playable.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv(
    "SIGNBRIDGE_TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2"
)
# Cap on consecutive transient load failures before we give up retrying.
# Without this we permanently mute the demo on a single bad cold-start.
_MAX_LOAD_FAILURES = 3


def _cache_key(text: str) -> str:
    """Stable per-text cache key. Python's `hash()` is salted per-process
    (PYTHONHASHSEED) so it changes every cold start and defeats the cache."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class _TTSEngine:
    def __init__(self) -> None:
        self._tts = None
        self._import_failed = False
        self._load_failures = 0
        self._cache_dir = Path(tempfile.gettempdir()) / "signbridge_tts"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._tts is not None or self._import_failed:
            return
        if self._load_failures >= _MAX_LOAD_FAILURES:
            return
        with self._load_lock:
            # Re-check after acquiring lock (double-checked locking).
            if self._tts is not None or self._import_failed:
                return
            try:
                from TTS.api import TTS  # type: ignore[import-not-found]
            except ImportError:
                logger.warning(
                    "TTS package not installed; voice output will be silent. "
                    "Install via `pip install TTS>=0.22`."
                )
                self._import_failed = True
                return

            try:
                self._tts = TTS(model_name=DEFAULT_MODEL, progress_bar=False)
            except Exception as exc:  # noqa: BLE001
                self._load_failures += 1
                logger.warning(
                    "XTTS-v2 load failed (attempt %d/%d): %s",
                    self._load_failures,
                    _MAX_LOAD_FAILURES,
                    type(exc).__name__,
                )

    def synthesize(self, text: str) -> str | None:
        if not text:
            return None

        # Disk cache hit — same text already synthesised this session.
        cached_wav = self._cache_dir / f"{_cache_key(text)}.wav"
        if cached_wav.exists():
            return str(cached_wav)
        cached_mp3 = self._cache_dir / f"{_cache_key(text)}.mp3"
        if cached_mp3.exists():
            return str(cached_mp3)

        # Tier 1: Coqui XTTS-v2 if installed locally (full quality, slow).
        self._ensure_loaded()
        if self._tts is not None:
            try:
                self._tts.tts_to_file(
                    text=text,
                    file_path=str(cached_wav),
                    language="en",
                )
                return str(cached_wav)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "XTTS synthesis failed (%s); falling through to gTTS.",
                    type(exc).__name__,
                )

        # Tier 2: gTTS — tiny dep, free, fast (Google's TTS API).
        try:
            from gtts import gTTS  # type: ignore[import-not-found]
            tts = gTTS(text=text, lang="en", tld="com")
            tts.save(str(cached_mp3))
            print(f"[tts] gTTS synthesised: {cached_mp3}", flush=True)
            return str(cached_mp3)
        except ImportError:
            logger.warning("gTTS not installed; falling through to silent stub.")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "gTTS synthesis failed (%s); falling through to silent stub.",
                type(exc).__name__,
            )

        # Tier 3: silent placeholder — better than crashing the audio component.
        return self._silent_stub(text)

    def _silent_stub(self, text: str) -> str | None:
        """Emit a 0.5 s silent WAV so the Gradio audio component has something to play.

        Returns None when even the stub can't be written (numpy/soundfile not
        available); callers must handle None and present a clean "no audio"
        UI rather than feeding "" into a Gradio Audio component.
        """
        out_path = self._cache_dir / f"silent_{_cache_key(text)}.wav"
        if out_path.exists():
            return str(out_path)
        try:
            import numpy as np
            import soundfile as sf  # type: ignore[import-not-found]
        except ImportError:
            return None
        sf.write(str(out_path), np.zeros(8000, dtype="int16"), 16000)
        return str(out_path)


_singleton = _TTSEngine()


def synthesize_speech(text: str) -> str | None:
    """Public entry-point. Returns path to a WAV file (or None)."""
    return _singleton.synthesize(text)
