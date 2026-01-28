"""
BiSeNet face parsing (ONNX) for hair/face masks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort


class BiSeNetHairSeg:
    def __init__(self, onnx_path: Path, provider: str = "cuda") -> None:
        if not onnx_path.exists():
            raise FileNotFoundError(f"BiSeNet model not found: {onnx_path}")

        providers = self._get_providers(provider)
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    @staticmethod
    def _get_providers(provider: str) -> list[str]:
        available = ort.get_available_providers()
        if provider == "cuda" and "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if provider == "dml" and "DmlExecutionProvider" in available:
            return ["DmlExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def get_masks(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (hair_mask, face_mask) as float32 in 0..1.
        """
        h, w = frame_bgr.shape[:2]
        resized, blob = self._preprocess(frame_bgr)

        logits = self.session.run(None, {self.input_name: blob})[0]
        labels = np.argmax(logits, axis=1)[0].astype(np.uint8)

        labels = cv2.resize(labels, (w, h), interpolation=cv2.INTER_NEAREST)

        hair_mask = (labels == 17).astype(np.float32)
        face_mask = np.isin(labels, [1, 2, 3, 4, 5, 10, 11, 12, 13]).astype(np.float32)

        return hair_mask, face_mask

    def segment_hair_and_head(self, frame_bgr: np.ndarray) -> np.ndarray:
        hair, face = self.get_masks(frame_bgr)
        return np.clip(hair + face, 0.0, 1.0)

    def segment_hair_only(self, frame_bgr: np.ndarray) -> np.ndarray:
        hair, _face = self.get_masks(frame_bgr)
        return hair

    def segment_person(self, frame_bgr: np.ndarray) -> np.ndarray:
        return self.segment_hair_and_head(frame_bgr)

    def _preprocess(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        input_h = int(self.input_shape[2])
        input_w = int(self.input_shape[3])

        resized = cv2.resize(frame_bgr, (input_w, input_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std

        chw = np.transpose(rgb, (2, 0, 1))[None, ...]
        return resized, chw

