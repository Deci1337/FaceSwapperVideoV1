import cv2
import numpy as np
from pathlib import Path
from gfpgan import GFPGANer
from app.models.loader import download_model
import logging

logger = logging.getLogger(__name__)


def create_blend_mask(height: int, width: int, border: int = 20) -> np.ndarray:
    """Create a gradient mask for smooth blending at edges."""
    mask = np.ones((height, width), dtype=np.float32)
    
    # Create gradients at borders
    for i in range(border):
        factor = i / border
        mask[i, :] *= factor          # Top
        mask[-(i+1), :] *= factor     # Bottom
        mask[:, i] *= factor          # Left
        mask[:, -(i+1)] *= factor     # Right
    
    return mask


class FaceEnhancer:
    def __init__(self, cache_dir: Path, blend_weight: float = 0.7):
        """
        Initialize GFPGAN face enhancer.
        
        Args:
            cache_dir: Directory for model cache
            blend_weight: How much to blend enhanced result (0.0-1.0).
                         Lower values = more natural, higher = more "enhanced" look.
                         Recommended: 0.5-0.7 to avoid "plastic" appearance.
        """
        model_path = download_model("GFPGANv1.4.pth", cache_dir)
        
        # upscale=1 to avoid size mismatch when pasting back
        self.enhancer = GFPGANer(
            model_path=str(model_path),
            upscale=1,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None
        )
        self.blend_weight = blend_weight
        
    def enhance(self, frame: np.ndarray, face_bbox: list) -> np.ndarray:
        """
        Enhance the face region using GFPGAN with smooth blending.
        
        Args:
            frame: Input frame (will be modified)
            face_bbox: [x1, y1, x2, y2] face bounding box
        
        Returns:
            Frame with enhanced face region
        """
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = [int(c) for c in face_bbox]
            
            # Add padding around face for context
            pad_x = int((x2 - x1) * 0.35)
            pad_y = int((y2 - y1) * 0.35)
            
            x1_pad = max(0, x1 - pad_x)
            y1_pad = max(0, y1 - pad_y)
            x2_pad = min(w, x2 + pad_x)
            y2_pad = min(h, y2 + pad_y)
            
            face_img = frame[y1_pad:y2_pad, x1_pad:x2_pad].copy()
            original_face = face_img.copy()
            crop_h, crop_w = face_img.shape[:2]
            
            # Enhance the cropped face
            _, _, output = self.enhancer.enhance(
                face_img,
                has_aligned=False,
                only_center_face=True,
                paste_back=True
            )
            
            if output is None:
                return frame
            
            # Resize output to match original crop size if needed
            if output.shape[:2] != (crop_h, crop_w):
                output = cv2.resize(output, (crop_w, crop_h))
            
            # Blend enhanced with original using weight (prevents "plastic" look)
            blended = cv2.addWeighted(
                output, self.blend_weight,
                original_face, 1.0 - self.blend_weight,
                0
            )
            
            # Create edge mask for smooth transition
            edge_mask = create_blend_mask(crop_h, crop_w, border=15)
            edge_mask_3ch = np.stack([edge_mask] * 3, axis=-1)
            
            # Apply edge blending
            final = (blended * edge_mask_3ch + original_face * (1 - edge_mask_3ch)).astype(np.uint8)
            
            # Paste back
            frame[y1_pad:y2_pad, x1_pad:x2_pad] = final
            return frame
            
        except Exception as e:
            logger.warning(f"Enhancement failed: {e}")
            return frame

