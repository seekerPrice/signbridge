"""Text-to-speech via Coqui XTTS-v2.

Loaded lazily because XTTS-v2 weights are ~2 GB. While the model isn't
loaded (e.g. on a fresh local dev box), we fall back to a silent WAV stub
so the Gradio app still produces something playable.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv(
    "SIGNBRIDGE_TTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2"
)


class _TTSEngine:
    def __init__(self) -> None:
        self._tts = None
        self._loaded = False
        self._unavailable_logged = False
        self._cache_dir = Path(tempfile.gettempdir()) / "signbridge_tts"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        try:
            from TTS.api import TTS  # type: ignore[import-not-found]
        except ImportError:
            if not self._unavailable_logged:
                logger.info(
                    "TTS package not installed; voice output will be silent. "
                    "Install via `pip install TTS>=0.22`."
                )
                self._unavailable_logged = True
            self._loaded = True
            return

        try:
            self._tts = TTS(model_name=DEFAULT_MODEL, progress_bar=False)
        except Exception:  # noqa: BLE001
            logger.exception("XTTS-v2 load failed; voice output will be silent.")
            self._tts = None
        self._loaded = True

    def synthesize(self, text: str) -> str | None:
        if not text:
            return None
        self._ensure_loaded()
        if self._tts is None:
            return self._silent_stub(text)

        out_path = self._cache_dir / f"{abs(hash(text))}.wav"
        if out_path.exists():
            return str(out_path)

        try:
            self._tts.tts_to_file(
                text=text,
                file_path=str(out_path),
                language="en",
                # XTTS-v2 needs a speaker reference; omit to use the default voice.
            )
        except Exception:  # noqa: BLE001
            logger.exception("XTTS synthesis failed for %r; emitting silent stub.", text)
            return self._silent_stub(text)
        return str(out_path)

    def _silent_stub(self, text: str) -> str:
        """Emit a 0.5 s silent WAV so the Gradio audio component has something to play."""
        out_path = self._cache_dir / f"silent_{abs(hash(text))}.wav"
        if out_path.exists():
            return str(out_path)
        try:
            import numpy as np
            import soundfile as sf  # type: ignore[import-not-found]
        except ImportError:
            return ""
        sf.write(str(out_path), np.zeros(8000, dtype="int16"), 16000)
        return str(out_path)


_singleton = _TTSEngine()


def synthesize_speech(text: str) -> str | None:
    """Public entry-point. Returns path to a WAV file (or None)."""
    return _singleton.synthesize(text)
