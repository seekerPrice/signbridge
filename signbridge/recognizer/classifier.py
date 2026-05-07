"""Sign classifier — landmarks sequence → sign token.

V1: a small transformer encoder that maps a (T, 543) landmark window to a
sign-vocabulary logit vector. The trained checkpoint is published to
HF Hub at $SIGNBRIDGE_CLASSIFIER_HF_REPO; before training is done, this
module returns "(thinking…)" stubs so the Gradio app still boots.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

# Vocabulary imported from the shared module — must match the order the
# trained classifier head was trained against.
from signbridge.vocab import VOCAB

logger = logging.getLogger(__name__)

VOCABULARY = list(VOCAB)
VOCAB_SIZE = len(VOCABULARY)


class _Classifier:
    """Lazy-loaded torch model. Falls back to a no-op when no weights present."""

    def __init__(self) -> None:
        self._model = None
        self._device = "cpu"
        self._loaded = False
        self._weights_missing_logged = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        weights_path = os.getenv("SIGNBRIDGE_CLASSIFIER_PATH", "models/classifier.pt")
        path = Path(weights_path)
        if not path.exists():
            if not self._weights_missing_logged:
                logger.info(
                    "classifier weights not found at %s; classifier returns stub. "
                    "Train via `python -m signbridge.scripts.train_classifier`.",
                    weights_path,
                )
                self._weights_missing_logged = True
            self._loaded = True
            return

        try:
            import torch  # type: ignore[import-not-found]
        except ImportError:
            logger.warning("torch not installed; classifier returns stub.")
            self._loaded = True
            return

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = torch.jit.load(str(path), map_location=self._device).eval()
        self._loaded = True
        logger.info("classifier loaded from %s on %s", weights_path, self._device)

    def predict(self, window: np.ndarray) -> tuple[str, float]:
        """Predict sign for a (T, 543) landmark window.

        Returns (sign, confidence). When no model is loaded, returns
        ("", 0.0) — the UI treats low-confidence as "still listening".
        """
        self._ensure_loaded()
        if self._model is None:
            return "", 0.0

        try:
            import torch  # type: ignore[import-not-found]
        except ImportError:
            return "", 0.0

        with torch.no_grad():
            x = torch.from_numpy(window).unsqueeze(0).to(self._device)
            logits = self._model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            idx = int(probs.argmax().item())
            conf = float(probs[idx].item())
        return VOCABULARY[idx], conf


_singleton = _Classifier()


def classify_landmarks(window: np.ndarray) -> tuple[str, float]:
    """Public entry-point used by the Gradio handler."""
    return _singleton.predict(window)
