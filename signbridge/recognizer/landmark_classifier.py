"""MediaPipe Hand landmarks → MLP → ASL letter (A-Z).

This is the high-accuracy path for fingerspelling. It runs on CPU with
~50ms latency and 88% accuracy on Marxulia ASL holdout (vs ~19% for
Qwen3-VL zero-shot). Used by the Snapshot tab; the Record-sign tab
still uses Qwen3-VL for motion-dependent signs.

Lazy-loads MediaPipe + the trained MLP on first call. Falls back to
returning ("", 0.0) if either model is missing or no hand is detected,
so the upstream VLM path can take over.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Model files. Override via env for HF Space deploys.
_MLP_PATH = Path(
    os.getenv(
        "SIGNBRIDGE_LANDMARK_MLP_PATH",
        str(Path(__file__).resolve().parent.parent.parent / "models" / "asl_landmark_mlp.pt"),
    )
)
_HAND_MODEL_PATH = Path(
    os.getenv(
        "SIGNBRIDGE_HAND_LANDMARKER_PATH",
        str(Path(__file__).resolve().parent.parent.parent / "models" / "hand_landmarker.task"),
    )
)

_lock = threading.Lock()
_state: dict[str, object] = {"loaded": False, "landmarker": None, "mlp": None, "classes": None}


def _normalize_landmarks(coords3: np.ndarray) -> np.ndarray:
    """Zero at wrist, scale by middle-finger MCP norm — must match training."""
    out = coords3.copy().astype(np.float32)
    out -= out[0]
    scale = float(np.linalg.norm(out[9]))
    if scale > 1e-6:
        out /= scale
    return out


def _ensure_loaded() -> bool:
    """Lazy-load MediaPipe + MLP. Returns True if both ready."""
    if _state["loaded"]:
        return _state["landmarker"] is not None and _state["mlp"] is not None
    with _lock:
        if _state["loaded"]:
            return _state["landmarker"] is not None and _state["mlp"] is not None

        if not _MLP_PATH.exists():
            logger.info("landmark MLP weights missing at %s; classifier disabled.", _MLP_PATH)
            _state["loaded"] = True
            return False
        if not _HAND_MODEL_PATH.exists():
            logger.info(
                "MediaPipe hand_landmarker.task missing at %s; classifier disabled.",
                _HAND_MODEL_PATH,
            )
            _state["loaded"] = True
            return False

        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
            import torch  # type: ignore[import-not-found]
            import torch.nn as nn  # type: ignore[import-not-found]
        except ImportError as exc:
            logger.warning("landmark classifier deps missing (%s); disabled.", exc)
            _state["loaded"] = True
            return False

        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(_HAND_MODEL_PATH)),
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
        )
        landmarker = vision.HandLandmarker.create_from_options(opts)

        ckpt = torch.load(str(_MLP_PATH), map_location="cpu", weights_only=False)
        n_in = int(ckpt["n_in"])
        n_out = int(ckpt["n_out"])

        class _MLP(nn.Module):
            def __init__(self, n_in: int, n_out: int) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(n_in, 256), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(256, 128), nn.GELU(),
                    nn.Linear(128, n_out),
                )

            def forward(self, x):  # type: ignore[no-untyped-def]
                return self.net(x)

        mlp = _MLP(n_in, n_out)
        mlp.load_state_dict(ckpt["model_state_dict"])
        mlp.eval()

        _state["landmarker"] = landmarker
        _state["mlp"] = mlp
        _state["classes"] = list(ckpt["classes"])
        _state["loaded"] = True
        logger.info(
            "landmark classifier ready: %d classes, MLP=%s",
            len(_state["classes"]),  # type: ignore[arg-type]
            ckpt.get("arch"),
        )
        return True


def predict_letter(frame: np.ndarray) -> tuple[str, float]:
    """Single-frame letter prediction. Returns (letter, confidence) or ("", 0.0).

    `frame` is an HxWx3 uint8 RGB array. Returns ("", 0.0) when no hand is
    detected — the upstream caller should fall through to Qwen3-VL.
    """
    if not _ensure_loaded():
        return "", 0.0

    import mediapipe as mp
    import torch

    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    if frame.ndim != 3 or frame.shape[2] != 3:
        return "", 0.0

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    res = _state["landmarker"].detect(mp_img)  # type: ignore[union-attr]
    if not res.hand_landmarks:
        return "", 0.0

    lm = res.hand_landmarks[0]
    coords3 = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    norm = _normalize_landmarks(coords3).flatten()
    with torch.no_grad():
        logits = _state["mlp"](torch.from_numpy(norm).unsqueeze(0))  # type: ignore[operator]
        probs = torch.softmax(logits, dim=1).squeeze(0)
        idx = int(torch.argmax(probs).item())
        conf = float(probs[idx].item())

    classes = _state["classes"]  # type: ignore[assignment]
    return classes[idx], conf  # type: ignore[index,return-value]
