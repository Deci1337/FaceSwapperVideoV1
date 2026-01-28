"""
Hair color matching in LAB space.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def compute_mean_lab(image_bgr: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    if mask is None:
        return None
    mask_u8 = (mask > 0.1).astype(np.uint8) * 255
    if mask_u8.sum() == 0:
        return None
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab[mask_u8 > 0]
    return pixels.mean(axis=0)


def match_hair_color(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    ref_mean_lab: np.ndarray,
) -> np.ndarray:
    if ref_mean_lab is None:
        return image_bgr
    mask_u8 = (mask > 0.1).astype(np.uint8) * 255
    if mask_u8.sum() == 0:
        return image_bgr

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab[mask_u8 > 0]
    mean_lab = pixels.mean(axis=0)
    delta = ref_mean_lab - mean_lab

    lab[mask_u8 > 0] = np.clip(lab[mask_u8 > 0] + delta, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)



