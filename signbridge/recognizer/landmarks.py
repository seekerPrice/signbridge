"""MediaPipe Holistic wrapper.

Extracts a 543-dim landmark vector per frame:
- 33 pose landmarks × 4 (x, y, z, visibility) = 132
- 468 face landmarks × 3 (x, y, z) — we use the top 70 = 210
  (we drop most face landmarks; sign meaning is in pose + hands)
  Actually we keep all 468 × 3 only for V2; V1 takes 33 pose + 21 left hand + 21 right hand.
- 21 left-hand landmarks × 3 = 63
- 21 right-hand landmarks × 3 = 63

V1 vector size = 132 + 63 + 63 = 258. We pad to 543 for forward-compat
with the WLASL community's 543-dim convention.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

LANDMARK_DIM = 543  # WLASL community convention; pad shorter vectors with zeros


class LandmarkExtractor:
    """Wraps MediaPipe Holistic. Lazy-imports mediapipe to keep startup fast."""

    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        self._holistic = None
        self._mp_drawing = None
        self._mp_holistic = None
        self._min_detection_confidence = min_detection_confidence

    def _ensure_loaded(self) -> None:
        if self._holistic is not None:
            return
        try:
            import mediapipe as mp  # type: ignore[import-not-found]
        except ImportError:
            logger.warning(
                "mediapipe not installed; landmarks will be all zeros. "
                "Install via `pip install mediapipe>=0.10.18`."
            )
            return
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_holistic = mp.solutions.holistic
        self._holistic = self._mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=0.5,
        )

    def extract(self, frame: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Process a single frame.

        Returns (annotated_frame, landmark_vector_or_None).
        Landmark vector is shape (LANDMARK_DIM,) padded with zeros.
        """
        self._ensure_loaded()
        if self._holistic is None:
            return frame, None

        # MediaPipe expects RGB; Gradio webcam already gives us RGB ndarray.
        results = self._holistic.process(frame)

        landmarks = self._collect_landmarks(results)
        annotated = self._draw_annotations(frame.copy(), results)
        return annotated, landmarks

    def _collect_landmarks(self, results) -> Optional[np.ndarray]:  # noqa: ANN001
        """Concatenate pose + left-hand + right-hand landmark coords."""
        if results.pose_landmarks is None:
            return None  # no person detected — drop frame

        parts: list[float] = []
        for lm in results.pose_landmarks.landmark:
            parts.extend([lm.x, lm.y, lm.z, lm.visibility])

        for hand in (results.left_hand_landmarks, results.right_hand_landmarks):
            if hand is None:
                parts.extend([0.0] * (21 * 3))
            else:
                for lm in hand.landmark:
                    parts.extend([lm.x, lm.y, lm.z])

        vec = np.asarray(parts, dtype=np.float32)
        if vec.shape[0] < LANDMARK_DIM:
            vec = np.pad(vec, (0, LANDMARK_DIM - vec.shape[0]))
        elif vec.shape[0] > LANDMARK_DIM:
            vec = vec[:LANDMARK_DIM]
        return vec

    def _draw_annotations(self, frame: np.ndarray, results) -> np.ndarray:  # noqa: ANN001
        if self._mp_drawing is None or self._mp_holistic is None:
            return frame
        if results.pose_landmarks is not None:
            self._mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, self._mp_holistic.POSE_CONNECTIONS
            )
        if results.left_hand_landmarks is not None:
            self._mp_drawing.draw_landmarks(
                frame, results.left_hand_landmarks, self._mp_holistic.HAND_CONNECTIONS
            )
        if results.right_hand_landmarks is not None:
            self._mp_drawing.draw_landmarks(
                frame, results.right_hand_landmarks, self._mp_holistic.HAND_CONNECTIONS
            )
        return frame
