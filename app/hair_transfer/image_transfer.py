"""
Image hair transfer wrapper.
Provides pluggable backends: hairfastgan, style-your-hair, stable-hair, none.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import subprocess
import tempfile

from app.hair_transfer.segment import HairSegmenter
from app.hair_transfer.hairfast_integration import HairFastWrapper


@dataclass
class HairTransferBackend:
    name: str
    command: Optional[str] = None


class HairImageTransfer:
    def __init__(self, backend: HairTransferBackend) -> None:
        self.backend = backend
        self.segmenter = HairSegmenter() if backend.name == "simple" else None

    def transfer(self, anchor_frame: np.ndarray, reference_image: np.ndarray) -> np.ndarray:
        """
        Returns an anchor frame with transferred hairstyle.
        """
        if self.backend.name == "simple":
            return self._simple_transfer(anchor_frame, reference_image)

        if self.backend.name == "none":
            return anchor_frame

        if self.backend.name == "hairfastgan":
            return self._hairfast_transfer(anchor_frame, reference_image)

        return self._run_command(anchor_frame, reference_image)

    def _hairfast_transfer(self, anchor_frame: np.ndarray, reference_image: np.ndarray) -> np.ndarray:
        if self.backend.command:
            return self._run_command(anchor_frame, reference_image)

        wrapper = HairFastWrapper(device="cuda")
        return wrapper.transfer(anchor_frame, reference_image, reference_image)

    def _run_command(self, anchor_frame: np.ndarray, reference_image: np.ndarray) -> np.ndarray:
        if not self.backend.command:
            raise RuntimeError("Hair transfer command is not configured.")

        with tempfile.TemporaryDirectory() as tmp:
            anchor_path = Path(tmp) / "anchor.png"
            ref_path = Path(tmp) / "reference.png"
            out_path = Path(tmp) / "output.png"

            cv2.imwrite(str(anchor_path), anchor_frame)
            cv2.imwrite(str(ref_path), reference_image)

            cmd = self.backend.command.format(
                anchor=str(anchor_path),
                reference=str(ref_path),
                output=str(out_path),
            )

            subprocess.run(cmd, shell=True, check=True)

            result = cv2.imread(str(out_path))
            if result is None:
                raise RuntimeError("Hair transfer backend did not produce output.")

            return result

    def _simple_transfer(self, anchor_frame: np.ndarray, reference_image: np.ndarray) -> np.ndarray:
        """
        Simple, fast baseline: segment hair from reference and warp to anchor.
        """
        if self.segmenter is None:
            return anchor_frame

        ref_landmarks = self.segmenter.get_face_landmarks(reference_image)
        anchor_landmarks = self.segmenter.get_face_landmarks(anchor_frame)
        if ref_landmarks is None or anchor_landmarks is None:
            return anchor_frame

        ref_mask = self.segmenter.segment_hair_and_head(reference_image)
        # Exclude face region so we don't paste the reference face
        ref_face = self.segmenter.get_face_landmarks(reference_image)
        if ref_face is not None:
            x1, y1, x2, y2 = ref_face.bounds
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            ax = max(1, (x2 - x1) // 2)
            ay = max(1, int((y2 - y1) * 0.55))
            face_mask = np.zeros(reference_image.shape[:2], dtype=np.float32)
            cv2.ellipse(face_mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)
            face_mask = cv2.GaussianBlur(face_mask, (51, 51), 0)
            ref_mask = ref_mask * (1 - face_mask)
        M, _ = cv2.estimateAffinePartial2D(
            ref_landmarks.keypoints, anchor_landmarks.keypoints, method=cv2.LMEDS
        )
        if M is None:
            return anchor_frame

        h, w = anchor_frame.shape[:2]
        warped_ref = cv2.warpAffine(
            reference_image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
        )
        warped_mask = cv2.warpAffine(
            ref_mask, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        mask_3ch = np.stack([warped_mask] * 3, axis=-1)
        return (warped_ref * mask_3ch + anchor_frame * (1 - mask_3ch)).astype(np.uint8)

