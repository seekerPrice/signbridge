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

# Weights live in a public HF model repo so the Space repo stays small and
# free of LFS. Local clones can override via the env vars below for offline
# tests; the default behaviour is `hf_hub_download` on first call (cached
# under HF_HOME / ~/.cache/huggingface).
_HF_REPO = os.getenv("SIGNBRIDGE_CLASSIFIER_HF_REPO", "LucasLooTan/signbridge-asl-classifier")
_MLP_FILENAME = os.getenv("SIGNBRIDGE_MLP_FILENAME", "asl_landmark_mlp.pt")
_HAND_FILENAME = os.getenv("SIGNBRIDGE_HAND_FILENAME", "hand_landmarker.task")
_MLP_LOCAL_OVERRIDE = os.getenv("SIGNBRIDGE_LANDMARK_MLP_PATH")
_HAND_LOCAL_OVERRIDE = os.getenv("SIGNBRIDGE_HAND_LANDMARKER_PATH")


def _resolve_weight(local_override: str | None, filename: str) -> Path | None:
    """Return a local Path for a weight file, downloading from HF Hub if needed."""
    if local_override:
        p = Path(local_override)
        if p.exists():
            return p
        logger.warning("override %s does not exist; falling back to HF Hub.", local_override)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.warning("huggingface_hub missing; cannot fetch %s.", filename)
        return None
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or None
    logger.info(
        "hf_hub_download(%s) attempt: repo=%s token_len=%d",
        filename, _HF_REPO, len(token) if token else 0,
    )
    # First attempt: with explicit token (if set). If that fails with
    # auth-flavoured RepositoryNotFoundError, retry anonymously — public
    # repos work without auth, and a stale/invalid token can poison even
    # public reads.
    last_exc: Exception | None = None
    for attempt_token in (token, None):
        try:
            local = hf_hub_download(
                repo_id=_HF_REPO,
                filename=filename,
                repo_type="model",
                token=attempt_token,
            )
            logger.info("hf_hub_download(%s) ok via %s", filename, "token" if attempt_token else "anonymous")
            return Path(local)
        except Exception as exc:  # noqa: BLE001 — many failure modes
            last_exc = exc
            logger.warning(
                "hf_hub_download(%s) failed (token=%s): %s — %s",
                filename, "yes" if attempt_token else "no", type(exc).__name__, str(exc)[:300],
            )
            if attempt_token is None:
                break  # already tried anonymously
    return None

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
    """Lazy-load MediaPipe + MLP. Returns True if both ready.

    Transient failures (HF Hub blip, momentary network) are NOT cached
    so the next call retries. Only deps-missing (ImportError) is fatal
    and cached, since it can't fix itself at runtime."""
    if _state["loaded"]:
        return _state["landmarker"] is not None and _state["mlp"] is not None
    with _lock:
        if _state["loaded"]:
            return _state["landmarker"] is not None and _state["mlp"] is not None

        mlp_path = _resolve_weight(_MLP_LOCAL_OVERRIDE, _MLP_FILENAME)
        if mlp_path is None:
            logger.warning("MLP weights download failed; will retry on next call.")
            return False
        hand_path = _resolve_weight(_HAND_LOCAL_OVERRIDE, _HAND_FILENAME)
        if hand_path is None:
            logger.warning("hand_landmarker.task download failed; will retry on next call.")
            return False

        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
            import torch  # type: ignore[import-not-found]
            import torch.nn as nn  # type: ignore[import-not-found]
        except ImportError as exc:
            logger.warning("landmark classifier deps missing (%s); disabled.", exc)
            _state["loaded"] = True  # cache: deps won't appear at runtime
            return False

        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(hand_path)),
            num_hands=1,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
        )
        landmarker = vision.HandLandmarker.create_from_options(opts)

        ckpt = torch.load(str(mlp_path), map_location="cpu", weights_only=False)
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
