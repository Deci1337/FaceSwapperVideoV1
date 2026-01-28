"""
Video hair transfer pipeline.
Takes hair from reference_image (PNG with alpha or regular image) and applies it to each video frame.
Optimized for speed and realism.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from app.hair_transfer.segment import HairSegmenter
from app.hair_transfer.segment_bisenet import BiSeNetHairSeg
from app.hair_transfer.optical_flow import HairFlowConfig, HairOpticalFlow
from app.hair_transfer.color_transfer import compute_mean_lab, match_hair_color


@dataclass
class HairTransferConfig:
    reference_image: Path
    backend: str = "none"
    backend_cmd: Optional[str] = None
    anchor_strategy: str = "middle"
    blend_strength: float = 0.8
    mask_blur: int = 21
    mask_ema: float = 0.7
    mask_stride: int = 1
    mask_scale: float = 1.0
    face_exclusion: float = 0.9
    head_extend: float = 2.0
    seg_mode: str = "mediapipe"
    seg_model: Optional[Path] = None
    seg_provider: str = "cuda"
    hair_mask_mode: str = "head"
    multi_anchor: bool = False
    anchor_stride: int = 10
    anchor_yaw: float = 0.12
    flow_enabled: bool = False
    flow_alpha: float = 0.7
    flow_pyr_scale: float = 0.5
    flow_levels: int = 3
    flow_winsize: int = 25
    flow_iterations: int = 3
    flow_poly_n: int = 5
    flow_poly_sigma: float = 1.2
    flow_flags: int = 0
    hair_color_match: bool = False


class HairVideoApplier:
    """
    Applies hair from reference_image directly to each video frame.
    Supports PNG with alpha channel for instant mask extraction.
    """

    def __init__(self, video_path: Path, config: HairTransferConfig) -> None:
        self.video_path = video_path
        self.config = config
        # Only initialize landmark segmenter (lightweight)
        self.landmark_segmenter = HairSegmenter(mask_blur=config.mask_blur, scale=config.mask_scale)
        
        # Reference image data (hair source)
        self.ref_image: Optional[np.ndarray] = None
        self.ref_hair_mask: Optional[np.ndarray] = None
        self.has_alpha: bool = False

        # Per-frame state
        self.prev_mask: Optional[np.ndarray] = None
        self.frame_idx: int = 0
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_hair_layer: Optional[np.ndarray] = None
        self.cached_face = None
        self.cached_frame_idx: int = -1

        self.flow = HairOpticalFlow(
            HairFlowConfig(
                enabled=config.flow_enabled,
                alpha=config.flow_alpha,
                pyr_scale=config.flow_pyr_scale,
                levels=config.flow_levels,
                winsize=config.flow_winsize,
                iterations=config.flow_iterations,
                poly_n=config.flow_poly_n,
                poly_sigma=config.flow_poly_sigma,
                flags=config.flow_flags,
            )
        )

        self._prepare_reference()

    def _prepare_reference(self) -> None:
        """Extract hair from reference image. If PNG with alpha, use alpha as mask."""
        # Read with alpha channel support
        reference = cv2.imread(str(self.config.reference_image), cv2.IMREAD_UNCHANGED)
        if reference is None:
            raise ValueError(f"Could not read reference image: {self.config.reference_image}")

        # Check if image has alpha channel (PNG with transparency)
        if reference.ndim == 3 and reference.shape[2] == 4:
            self.has_alpha = True
            # Extract BGR and alpha
            self.ref_image = reference[:, :, :3].copy()
            alpha = reference[:, :, 3].astype(np.float32) / 255.0
            # Smooth alpha edges for realistic blending
            self.ref_hair_mask = cv2.GaussianBlur(alpha, (15, 15), 0)
        else:
            self.has_alpha = False
            self.ref_image = reference[:, :, :3] if reference.ndim == 3 else cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)
            # Fallback: create mask from non-black pixels
            gray = cv2.cvtColor(self.ref_image, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
            self.ref_hair_mask = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (21, 21), 0)

    def _get_face_cached(self, frame: np.ndarray):
        """Get face landmarks with caching for speed."""
        if self.config.mask_stride > 1 and self.frame_idx % self.config.mask_stride != 1:
            if self.cached_face is not None:
                return self.cached_face
        face = self.landmark_segmenter.get_face_landmarks(frame)
        self.cached_face = face
        return face

    def _position_hair_on_face(self, frame: np.ndarray, face) -> Tuple[np.ndarray, np.ndarray]:
        """Position hair PNG above the face based on face bounding box."""
        h, w = frame.shape[:2]
        ref_h, ref_w = self.ref_image.shape[:2]

        if face is None:
            # No face detected - center hair at top
            scale = w / ref_w
            new_w = int(ref_w * scale)
            new_h = int(ref_h * scale)
            hair_resized = cv2.resize(self.ref_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            mask_resized = cv2.resize(self.ref_hair_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            mask_canvas = np.zeros((h, w), dtype=np.float32)
            x_off = (w - new_w) // 2
            y_off = 0
            x1, y1 = max(0, x_off), max(0, y_off)
            x2, y2 = min(w, x_off + new_w), min(h, y_off + new_h)
            src_x1, src_y1 = max(0, -x_off), max(0, -y_off)
            src_x2, src_y2 = src_x1 + (x2 - x1), src_y1 + (y2 - y1)
            canvas[y1:y2, x1:x2] = hair_resized[src_y1:src_y2, src_x1:src_x2]
            mask_canvas[y1:y2, x1:x2] = mask_resized[src_y1:src_y2, src_x1:src_x2]
            return canvas, mask_canvas

        # Get face bounds
        fx1, fy1, fx2, fy2 = face.bounds
        face_w = fx2 - fx1
        face_h = fy2 - fy1
        face_cx = (fx1 + fx2) // 2

        # Scale hair to match face width (hair should be wider than face)
        target_hair_w = int(face_w * 2.2)
        scale = target_hair_w / ref_w
        new_w = int(ref_w * scale)
        new_h = int(ref_h * scale)

        hair_resized = cv2.resize(self.ref_image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(self.ref_hair_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Position hair centered on face, above forehead
        x_off = face_cx - new_w // 2
        y_off = fy1 - int(face_h * 0.6)  # Start above forehead

        # Create output canvases
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        mask_canvas = np.zeros((h, w), dtype=np.float32)

        # Calculate valid regions for paste
        dst_x1 = max(0, x_off)
        dst_y1 = max(0, y_off)
        dst_x2 = min(w, x_off + new_w)
        dst_y2 = min(h, y_off + new_h)

        src_x1 = max(0, -x_off)
        src_y1 = max(0, -y_off)
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)

        if dst_x2 > dst_x1 and dst_y2 > dst_y1:
            canvas[dst_y1:dst_y2, dst_x1:dst_x2] = hair_resized[src_y1:src_y2, src_x1:src_x2]
            mask_canvas[dst_y1:dst_y2, dst_x1:dst_x2] = mask_resized[src_y1:src_y2, src_x1:src_x2]

        return canvas, mask_canvas

    def _adjust_hair_lighting(self, hair: np.ndarray, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Match hair lighting to frame for realism."""
        if mask.max() < 0.01:
            return hair
        
        # Get average brightness of frame in hair region
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_hair = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        mask_sum = mask.sum() + 1e-6
        frame_brightness = (gray_frame * mask).sum() / mask_sum
        hair_brightness = (gray_hair * mask).sum() / mask_sum
        
        if hair_brightness < 1:
            return hair
        
        # Adjust hair brightness to match frame
        ratio = frame_brightness / (hair_brightness + 1e-6)
        ratio = np.clip(ratio, 0.5, 1.5)  # Limit adjustment range
        
        adjusted = hair.astype(np.float32) * ratio
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Apply reference hair to current frame. Optimized for PNG with alpha."""
        self.frame_idx += 1
        h, w = frame.shape[:2]

        # Get face (cached for speed)
        face = self._get_face_cached(frame)

        # Position hair based on face location
        warped_hair, warped_mask = self._position_hair_on_face(frame, face)

        # Adjust lighting for realism
        warped_hair = self._adjust_hair_lighting(warped_hair, frame, warped_mask)

        # Exclude face area from hair
        if face is not None:
            x1, y1, x2, y2 = face.bounds
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            ax = max(1, int((x2 - x1) * 0.55))
            ay = max(1, int((y2 - y1) * 0.5))
            face_mask = np.zeros((h, w), dtype=np.float32)
            cv2.ellipse(face_mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)
            face_mask = cv2.GaussianBlur(face_mask, (41, 41), 0)
            warped_mask = warped_mask * (1 - face_mask * self.config.face_exclusion)

        # Temporal smoothing for stable edges
        if self.prev_mask is not None and self.prev_mask.shape == warped_mask.shape:
            warped_mask = self.config.mask_ema * warped_mask + (1 - self.config.mask_ema) * self.prev_mask
        self.prev_mask = warped_mask.copy()

        # Final mask
        mask = np.clip(warped_mask * self.config.blend_strength, 0.0, 1.0)
        mask_3ch = np.stack([mask] * 3, axis=-1)

        # Create hair layer
        hair_layer = (warped_hair.astype(np.float32) * mask_3ch)

        # Optical flow for smooth motion
        if self.flow.config.enabled and self.prev_gray is not None and self.prev_hair_layer is not None:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow_hair = self.flow.propagate(self.prev_gray, curr_gray, self.prev_hair_layer)
            alpha = self.flow.config.alpha
            hair_layer = alpha * hair_layer + (1 - alpha) * flow_hair
            self.prev_gray = curr_gray
        else:
            self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.prev_hair_layer = hair_layer.copy()

        # Composite: hair over frame
        result = hair_layer + frame.astype(np.float32) * (1 - mask_3ch)
        return np.clip(result, 0, 255).astype(np.uint8)

    def close(self) -> None:
        self.landmark_segmenter.close()

