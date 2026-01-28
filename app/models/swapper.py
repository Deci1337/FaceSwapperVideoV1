import insightface
import onnxruntime
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
import logging
from app.models.loader import download_model
from app.utils.color_transfer import apply_face_color_correction

logger = logging.getLogger(__name__)

def get_onnx_providers(preferred: str = 'cuda') -> List[str]:
    """Get available ONNX providers with fallback."""
    available = onnxruntime.get_available_providers()
    
    # CUDA (NVIDIA)
    if preferred == 'cuda' and 'CUDAExecutionProvider' in available:
        logger.info("Using CUDA (NVIDIA GPU) for inference")
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    
    # DirectML (AMD/Intel GPU on Windows)
    if 'DmlExecutionProvider' in available:
        logger.info("Using DirectML (AMD/Intel GPU) for inference")
        return ['DmlExecutionProvider', 'CPUExecutionProvider']
    
    logger.warning("No GPU available, using CPU (this will be slow)")
    return ['CPUExecutionProvider']

def create_face_mask(frame: np.ndarray, landmarks: np.ndarray, blur_amount: int = 30) -> np.ndarray:
    """
    Create a smooth face mask from landmarks using convex hull.
    This prevents hair bleeding and square artifacts.
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    
    # Use facial landmarks to create convex hull (face outline only, not hair)
    # landmarks shape is (5, 2) for basic or (68, 2) / (106, 2) for detailed
    if landmarks is not None and len(landmarks) >= 5:
        # Convert to int points
        points = landmarks.astype(np.int32)
        
        # Create convex hull from landmarks
        hull = cv2.convexHull(points)
        
        # Fill the hull
        cv2.fillConvexPoly(mask, hull, 1.0)
        
        # Apply Gaussian blur for smooth edges
        if blur_amount > 0:
            mask = cv2.GaussianBlur(mask, (blur_amount * 2 + 1, blur_amount * 2 + 1), 0)
    
    return mask

def create_face_mask_from_bbox(bbox: np.ndarray, frame_shape: Tuple[int, int], 
                                shrink_factor: float = 0.15, blur_amount: int = 25) -> np.ndarray:
    """
    Create an elliptical face mask from bounding box.
    Shrinks the mask to avoid hair region.
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    
    x1, y1, x2, y2 = bbox.astype(int)
    
    # Calculate center and axes of ellipse
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    # Shrink to avoid hair (make ellipse smaller, especially at top)
    width = int((x2 - x1) * (1 - shrink_factor))
    height = int((y2 - y1) * (1 - shrink_factor * 1.5))  # Shrink more vertically
    
    # Shift center slightly down to avoid forehead/hair
    center_y = center_y + int(height * 0.1)
    
    # Draw ellipse
    cv2.ellipse(mask, (center_x, center_y), (width // 2, height // 2), 
                0, 0, 360, 1.0, -1)
    
    # Blur for smooth edges
    if blur_amount > 0:
        mask = cv2.GaussianBlur(mask, (blur_amount * 2 + 1, blur_amount * 2 + 1), 0)
    
    return mask

class FaceSwapper:
    def __init__(self, cache_dir: Path, provider: str = 'cuda', 
                 mask_mode: str = 'bbox'):
        """
        Initialize FaceSwapper.
        
        Args:
            cache_dir: Model cache directory
            provider: ONNX provider ('cuda', 'dml', 'cpu')
            mask_mode: 'bbox' (ellipse mask) or 'parsing' (face parsing mask)
        """
        self.cache_dir = cache_dir
        self.providers = get_onnx_providers(provider)
        self.mask_mode = mask_mode
        
        # Initialize Face Analysis (Detector + Recognition)
        self.face_analyser = insightface.app.FaceAnalysis(
            name='buffalo_l', 
            root=str(cache_dir),
            providers=self.providers
        )
        self.face_analyser.prepare(ctx_id=0, det_size=(640, 640))
        
        # Initialize Swapper Model
        model_path = download_model("inswapper_128.onnx", cache_dir)
        self.swapper = insightface.model_zoo.get_model(
            str(model_path), 
            providers=self.providers
        )
        
        # Initialize Face Parser (for 'parsing' mask mode)
        self.face_parser = None
        if mask_mode == 'parsing':
            from app.models.parsing import FaceParser
            self.face_parser = FaceParser(cache_dir, providers=self.providers)
            logger.info("Face Parser enabled for smart masking")
        
    def get_face(self, img: np.ndarray) -> Optional[object]:
        """Detect and return the largest face in image."""
        faces = self.face_analyser.get(img)
        if not faces:
            return None
        # Return largest face by bbox area
        return max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        
    def swap(self, source_face, target_face, frame: np.ndarray, 
             color_correction: float = 0.5, include_hair: bool = False) -> np.ndarray:
        """
        Swap source face onto target face in frame with smooth blending.
        
        Args:
            source_face: Source face embedding/data
            target_face: Target face in frame
            frame: Original frame
            color_correction: Strength of color matching (0.0-1.0)
            include_hair: If True and using parsing mode, include hair in mask
        
        Returns:
            Frame with swapped face
        """
        # Keep original for color reference
        original = frame.copy()
        
        # Get swapped frame (InsightFace does the swap + paste_back internally)
        swapped = self.swapper.get(frame, target_face, source_face, paste_back=True)
        
        # Create mask based on mode
        if self.mask_mode == 'parsing' and self.face_parser is not None:
            # Use face parsing for smart mask (better for gender swap)
            mask = self.face_parser.get_face_mask(
                swapped,  # Parse the swapped result to get face regions
                include_hair=include_hair,
                blur_amount=20
            )
        else:
            # Use simple ellipse mask (default)
            mask = create_face_mask_from_bbox(
                target_face.bbox, 
                frame.shape,
                shrink_factor=0.12,  # Shrink mask to avoid hair
                blur_amount=30       # Smooth edges
            )
        
        # Expand mask to 3 channels
        mask_3ch = np.stack([mask] * 3, axis=-1)
        
        # Blend: swapped face in mask region, original elsewhere
        result = (swapped * mask_3ch + original * (1 - mask_3ch)).astype(np.uint8)
        
        # Apply color correction to match original lighting
        if color_correction > 0:
            result = apply_face_color_correction(
                result, original, target_face.bbox, 
                strength=color_correction
            )
        
        return result

