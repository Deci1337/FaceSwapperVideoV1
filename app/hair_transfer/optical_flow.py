"""
Optical flow propagation for hair layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class HairFlowConfig:
    enabled: bool = True
    alpha: float = 0.7
    pyr_scale: float = 0.5
    levels: int = 3
    winsize: int = 25
    iterations: int = 3
    poly_n: int = 5
    poly_sigma: float = 1.2
    flags: int = 0


class HairOpticalFlow:
    def __init__(self, config: HairFlowConfig) -> None:
        self.config = config

    def propagate(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        prev_hair_layer: np.ndarray,
    ) -> np.ndarray:
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            self.config.pyr_scale,
            self.config.levels,
            self.config.winsize,
            self.config.iterations,
            self.config.poly_n,
            self.config.poly_sigma,
            self.config.flags,
        )
        h, w = prev_gray.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (grid_x + flow[..., 0]).astype(np.float32)
        map_y = (grid_y + flow[..., 1]).astype(np.float32)
        return cv2.remap(
            prev_hair_layer,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )



