"""
High-quality background replacement using MediaPipe Selfie Segmentation.
Focus on stability and preventing body part flickering.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class BackgroundRemover:
    """
    Removes background with focus on temporal stability.
    Prevents body parts from flickering/disappearing.
    """
    
    def __init__(self, model_selection: int = 1):
        """
        Initialize the segmentation model.
        
        Args:
            model_selection: 0 = general (faster), 1 = landscape (better quality)
        """
        self.segmenter = mp.solutions.selfie_segmentation.SelfieSegmentation(
            model_selection=model_selection
        )
        
        # Temporal smoothing with Exponential Moving Average
        self.prev_mask = None
        self.ema_alpha = 0.3  # Low alpha = more smoothing, less flickering
        
        # Hysteresis thresholds
        self.high_thresh = 0.6  # Confident foreground
        self.low_thresh = 0.3   # Keep if was foreground before
        
        logger.info(f"BackgroundRemover initialized (model={model_selection}, ema_alpha={self.ema_alpha})")
        
    def get_raw_mask(self, frame: np.ndarray) -> np.ndarray:
        """Get raw segmentation mask from MediaPipe."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self.segmenter.process(frame_rgb)
        return results.segmentation_mask.astype(np.float32)
    
    def apply_hysteresis(self, mask: np.ndarray, prev_mask: Optional[np.ndarray]) -> np.ndarray:
        """
        Apply hysteresis thresholding to prevent flickering.
        
        - Pixels above high_thresh are always foreground
        - Pixels between low_thresh and high_thresh stay foreground if they were before
        - Pixels below low_thresh are always background
        """
        if prev_mask is None:
            # First frame - use simple threshold
            return (mask > self.high_thresh).astype(np.float32)
        
        # High confidence foreground
        high_mask = mask > self.high_thresh
        
        # Medium confidence - keep if was foreground
        medium_mask = (mask > self.low_thresh) & (mask <= self.high_thresh)
        was_foreground = prev_mask > 0.5
        keep_medium = medium_mask & was_foreground
        
        # Combine
        result = (high_mask | keep_medium).astype(np.float32)
        
        return result
    
    def temporal_ema(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply Exponential Moving Average for smooth temporal transitions.
        Prevents sudden changes that cause flickering.
        """
        if self.prev_mask is None:
            self.prev_mask = mask.copy()
            return mask
        
        # EMA: new = alpha * current + (1 - alpha) * previous
        # Low alpha = more weight to previous = smoother but slower to react
        smoothed = self.ema_alpha * mask + (1 - self.ema_alpha) * self.prev_mask
        
        # Update previous mask
        self.prev_mask = smoothed.copy()
        
        return smoothed
    
    def keep_largest_component(self, mask: np.ndarray, min_area_ratio: float = 0.05) -> np.ndarray:
        """
        Keep only the largest connected component (the person).
        Removes stray background objects.
        
        Args:
            mask: Binary mask
            min_area_ratio: Minimum area as ratio of largest component to keep
        """
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find all connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        
        if num_labels <= 1:
            return mask
        
        # Find largest component (excluding background label 0)
        areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
        if len(areas) == 0:
            return mask
            
        largest_idx = np.argmax(areas) + 1  # +1 because we skipped background
        largest_area = areas[largest_idx - 1]
        
        # Create mask with only large enough components
        result = np.zeros_like(mask_uint8)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= largest_area * min_area_ratio:
                result[labels == i] = 255
        
        return result.astype(np.float32) / 255.0
    
    def fill_body_holes(self, mask: np.ndarray) -> np.ndarray:
        """
        Fill holes in the body to prevent missing limbs.
        """
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Closing to fill gaps (arms, shoulders)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
        
        # Find contours and fill holes inside the person
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Fill holes inside all contours
            mask_filled = np.zeros_like(mask_uint8)
            cv2.drawContours(mask_filled, contours, -1, 255, -1)
            return mask_filled.astype(np.float32) / 255.0
        
        return mask_uint8.astype(np.float32) / 255.0
    
    def gentle_edge_cleanup(self, mask: np.ndarray) -> np.ndarray:
        """
        Gentle edge cleanup - minimal erosion to avoid cutting body parts.
        """
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Very small erosion (only 2px) - just to clean noise, not cut body
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_uint8 = cv2.erode(mask_uint8, kernel_erode, iterations=1)
        
        # Small opening to remove tiny noise specks
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_open)
        
        return mask_uint8.astype(np.float32) / 255.0
    
    def smooth_edges(self, mask: np.ndarray, blur_size: int = 21) -> np.ndarray:
        """
        Smooth mask edges for natural blending.
        """
        # Gaussian blur for smooth edges
        smoothed = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        
        # Preserve solid interior, only blur edges
        # Create edge region
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        eroded = cv2.erode((mask * 255).astype(np.uint8), kernel)
        interior = eroded.astype(np.float32) / 255.0
        
        # Use original mask in interior, blurred at edges
        edge_region = mask - interior
        edge_region = np.clip(edge_region, 0, 1)
        
        result = interior + smoothed * edge_region
        return np.clip(result, 0, 1)
    
    def process_mask(self, frame: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Full mask processing pipeline optimized for stability.
        
        Args:
            frame: Input BGR frame
            threshold: Base threshold (used for hysteresis high_thresh)
        
        Returns:
            Stable, smooth mask (0.0-1.0)
        """
        # Update hysteresis thresholds based on input
        self.high_thresh = threshold
        self.low_thresh = max(0.2, threshold - 0.3)
        
        # 1. Get raw mask from MediaPipe
        raw_mask = self.get_raw_mask(frame)
        
        # 2. Apply hysteresis to prevent flickering
        mask = self.apply_hysteresis(raw_mask, self.prev_mask)
        
        # 3. Keep only largest component (remove stray background objects)
        mask = self.keep_largest_component(mask, min_area_ratio=0.02)
        
        # 4. Fill holes in body (prevent missing limbs)
        mask = self.fill_body_holes(mask)
        
        # 5. Gentle edge cleanup (minimal erosion)
        mask = self.gentle_edge_cleanup(mask)
        
        # 6. Temporal EMA smoothing (main anti-flicker)
        mask = self.temporal_ema(mask)
        
        # 7. Threshold the smoothed result
        mask = np.where(mask > 0.4, 1.0, mask)  # Solidify confident areas
        
        # 8. Keep only largest again after smoothing (remove any new artifacts)
        mask = self.keep_largest_component(mask, min_area_ratio=0.01)
        
        # 9. Smooth edges for natural blending
        mask = self.smooth_edges(mask, blur_size=15)
        
        return np.clip(mask, 0, 1).astype(np.float32)
    
    def composite(self, frame: np.ndarray, background: np.ndarray, 
                  mask: np.ndarray) -> np.ndarray:
        """Composite foreground onto background using mask."""
        mask_3ch = np.stack([mask] * 3, axis=-1)
        result = (frame * mask_3ch + background * (1 - mask_3ch)).astype(np.uint8)
        return result
    
    def replace_background(self, frame: np.ndarray, background: np.ndarray,
                           threshold: float = 0.5) -> np.ndarray:
        """
        Full background replacement pipeline.
        
        Args:
            frame: Input frame (BGR)
            background: Background image (BGR, must match frame dimensions)
            threshold: Segmentation threshold (0.4-0.7 recommended)
        
        Returns:
            Frame with replaced background
        """
        mask = self.process_mask(frame, threshold)
        result = self.composite(frame, background, mask)
        return result
    
    def reset(self):
        """Reset temporal state (call on scene changes)."""
        self.prev_mask = None
    
    def close(self):
        """Release resources."""
        self.segmenter.close()
        self.prev_mask = None
