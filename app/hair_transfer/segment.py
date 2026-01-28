"""
Hair/Head segmentation utilities for hair transfer.
Uses MediaPipe Selfie Segmentation + FaceMesh for a lightweight mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class FaceLandmarks:
    keypoints: np.ndarray  # shape (5, 2)
    bounds: Tuple[int, int, int, int]  # x1, y1, x2, y2


class HairSegmenter:
    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        mask_blur: int = 21,
        scale: float = 1.0,
    ) -> None:
        self.segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=1
        )
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.mask_blur = mask_blur
        self.scale = max(0.3, min(1.0, scale))
        self.prev_mask: Optional[np.ndarray] = None

    def get_face_landmarks(self, frame: np.ndarray) -> Optional[FaceLandmarks]:
        scale = self.scale
        if scale < 1.0:
            small = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small = frame

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0].landmark
        h, w = small.shape[:2]

        indices = [33, 263, 1, 61, 291]
        keypoints = np.array(
            [[landmarks[i].x * w, landmarks[i].y * h] for i in indices],
            dtype=np.float32,
        )

        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        x1 = int(max(0, min(xs)))
        y1 = int(max(0, min(ys)))
        x2 = int(min(w - 1, max(xs)))
        y2 = int(min(h - 1, max(ys)))

        if scale < 1.0:
            keypoints /= scale
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

        return FaceLandmarks(keypoints=keypoints, bounds=(x1, y1, x2, y2))

    def segment_person(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns raw person mask from MediaPipe (0..1).
        """
        scale = self.scale
        if scale < 1.0:
            small = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            small = frame

        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        result = self.segmenter.process(rgb)
        mask = result.segmentation_mask
        if scale < 1.0:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        return mask

    def segment_hair_and_head(self, frame: np.ndarray) -> np.ndarray:
        """
        Returns a soft mask (0..1) for hair/head region.
        """
        h, w = frame.shape[:2]
        person_mask = self.segment_person(frame)

        face = self.get_face_landmarks(frame)
        if face is None:
            mask = np.where(person_mask > 0.6, 1.0, 0.0).astype(np.float32)
            return self._smooth_mask(mask)

        x1, y1, x2, y2 = face.bounds
        face_w = x2 - x1
        face_h = y2 - y1

        pad_x = int(face_w * 0.5)
        pad_y_top = int(face_h * 0.9)
        pad_y_bottom = int(face_h * 1.1)

        rx1 = max(0, x1 - pad_x)
        rx2 = min(w, x2 + pad_x)
        ry1 = max(0, y1 - pad_y_top)
        ry2 = min(h, y2 + pad_y_bottom)

        region = np.zeros((h, w), dtype=np.float32)
        region[ry1:ry2, rx1:rx2] = 1.0

        mask = person_mask * region
        mask = np.where(mask > 0.4, 1.0, 0.0).astype(np.float32)

        return self._smooth_mask(mask)

    def segment_hair_only(self, frame: np.ndarray) -> np.ndarray:
        """
        Approximate hair-only mask by removing face region from head mask.
        """
        mask = self.segment_hair_and_head(frame)
        face = self.get_face_landmarks(frame)
        if face is None:
            return mask

        x1, y1, x2, y2 = face.bounds
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        ax = max(1, (x2 - x1) // 2)
        ay = max(1, int((y2 - y1) * 0.55))
        face_mask = np.zeros(frame.shape[:2], dtype=np.float32)
        cv2.ellipse(face_mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)
        face_mask = cv2.GaussianBlur(face_mask, (51, 51), 0)
        return mask * (1 - face_mask)

    def _smooth_mask(self, mask: np.ndarray) -> np.ndarray:
        if self.mask_blur > 0:
            k = self.mask_blur * 2 + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        return mask

    def close(self) -> None:
        self.segmenter.close()
        self.face_mesh.close()

