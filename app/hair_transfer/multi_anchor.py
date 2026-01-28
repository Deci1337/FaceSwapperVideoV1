"""
Multi-anchor selection for hair transfer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AnchorCandidate:
    frame_idx: int
    yaw: float
    frame: np.ndarray


@dataclass
class AnchorData:
    frame: np.ndarray
    yaw: float
    hair: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    landmarks: Optional[np.ndarray] = None


class MultiAnchorManager:
    def __init__(self, yaw_threshold: float = 0.12) -> None:
        self.yaw_threshold = yaw_threshold
        self.front: Optional[AnchorCandidate] = None
        self.left: Optional[AnchorCandidate] = None
        self.right: Optional[AnchorCandidate] = None

    def consider_frame(self, frame_idx: int, frame: np.ndarray, yaw: float) -> None:
        if abs(yaw) <= self.yaw_threshold:
            if self.front is None or abs(yaw) < abs(self.front.yaw):
                self.front = AnchorCandidate(frame_idx, yaw, frame.copy())

        if yaw < -self.yaw_threshold:
            if self.left is None or yaw < self.left.yaw:
                self.left = AnchorCandidate(frame_idx, yaw, frame.copy())

        if yaw > self.yaw_threshold:
            if self.right is None or yaw > self.right.yaw:
                self.right = AnchorCandidate(frame_idx, yaw, frame.copy())

    def finalize(self) -> list[AnchorCandidate]:
        anchors: list[AnchorCandidate] = []
        if self.front:
            anchors.append(self.front)
        if self.left:
            anchors.append(self.left)
        if self.right:
            anchors.append(self.right)
        return anchors

    def get_anchor_for_pose(self, yaw: float, anchors: list[AnchorData]) -> Optional[AnchorData]:
        if not anchors:
            return None

        left = [a for a in anchors if a.yaw < -self.yaw_threshold]
        right = [a for a in anchors if a.yaw > self.yaw_threshold]
        front = [a for a in anchors if abs(a.yaw) <= self.yaw_threshold]

        if yaw < -self.yaw_threshold and left:
            return min(left, key=lambda a: a.yaw)
        if yaw > self.yaw_threshold and right:
            return max(right, key=lambda a: a.yaw)
        if front:
            return min(front, key=lambda a: abs(a.yaw))

        return min(anchors, key=lambda a: abs(a.yaw - yaw))



