"""
Face Parsing using BiSeNet ONNX model.
Provides detailed segmentation of face parts: skin, hair, eyes, nose, mouth, etc.
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Optional, Dict, Tuple
import logging
from app.models.loader import download_model

logger = logging.getLogger(__name__)

# Face parsing class indices (BiSeNet CelebAMask-HQ format)
# These may vary depending on the model - adjust if needed
PARSING_CLASSES = {
    'background': 0,
    'skin': 1,
    'l_brow': 2,
    'r_brow': 3,
    'l_eye': 4,
    'r_eye': 5,
    'eye_g': 6,  # glasses
    'l_ear': 7,
    'r_ear': 8,
    'ear_r': 9,  # earring
    'nose': 10,
    'mouth': 11,
    'u_lip': 12,
    'l_lip': 13,
    'neck': 14,
    'neck_l': 15,  # necklace
    'cloth': 16,
    'hair': 17,
    'hat': 18,
}

# Groups for easy mask creation
FACE_SKIN_CLASSES = [1, 2, 3, 4, 5, 10, 11, 12, 13]  # All face parts except hair
FACE_FULL_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14]  # Face + neck
HAIR_CLASSES = [17]


class FaceParser:
    """
    Face parsing model for detailed face segmentation.
    Uses BiSeNet architecture trained on CelebAMask-HQ.
    """
    
    def __init__(self, cache_dir: Path, providers: list = None):
        """
        Initialize the face parsing model.
        
        Args:
            cache_dir: Directory for model cache
            providers: ONNX Runtime providers (e.g., ['CUDAExecutionProvider'])
        """
        model_path = download_model("face_parsing.onnx", cache_dir)
        
        if providers is None:
            providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        
        # Get model input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        
        # Expected input size (typically 512x512 for BiSeNet)
        self.input_size = (512, 512)
        if len(self.input_shape) == 4:
            self.input_size = (self.input_shape[2], self.input_shape[3])
        
        logger.info(f"FaceParser initialized, input size: {self.input_size}")
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Preprocess image for the model.
        
        Args:
            image: BGR image
        
        Returns:
            Preprocessed tensor and original size
        """
        original_size = (image.shape[1], image.shape[0])  # (w, h)
        
        # Resize to model input size
        resized = cv2.resize(image, self.input_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to NCHW format
        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor, original_size
    
    def postprocess(self, output: np.ndarray, original_size: Tuple[int, int]) -> np.ndarray:
        """
        Postprocess model output to get class mask.
        
        Args:
            output: Model output tensor
            original_size: (width, height) of original image
        
        Returns:
            Segmentation mask with class indices
        """
        # Output shape: (1, num_classes, H, W) or (1, H, W)
        if len(output.shape) == 4:
            # Get argmax across classes
            mask = np.argmax(output[0], axis=0)
        else:
            mask = output[0]
        
        # Resize back to original size
        mask = cv2.resize(mask.astype(np.uint8), original_size, interpolation=cv2.INTER_NEAREST)
        
        return mask
    
    def parse(self, image: np.ndarray) -> np.ndarray:
        """
        Run face parsing on an image.
        
        Args:
            image: BGR image
        
        Returns:
            Segmentation mask with class indices (0-18)
        """
        tensor, original_size = self.preprocess(image)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: tensor})
        
        # Get segmentation mask
        mask = self.postprocess(outputs[0], original_size)
        
        return mask
    
    def get_mask(self, image: np.ndarray, classes: list, 
                 blur_amount: int = 15) -> np.ndarray:
        """
        Get binary mask for specific classes.
        
        Args:
            image: BGR image
            classes: List of class indices to include
            blur_amount: Gaussian blur for smooth edges
        
        Returns:
            Binary mask (0.0-1.0)
        """
        parsing = self.parse(image)
        
        # Create binary mask for selected classes
        mask = np.zeros(parsing.shape, dtype=np.float32)
        for cls in classes:
            mask[parsing == cls] = 1.0
        
        # Smooth edges
        if blur_amount > 0:
            mask = cv2.GaussianBlur(mask, (blur_amount * 2 + 1, blur_amount * 2 + 1), 0)
        
        return mask
    
    def get_face_mask(self, image: np.ndarray, include_hair: bool = False,
                      blur_amount: int = 15) -> np.ndarray:
        """
        Get mask for face region.
        
        Args:
            image: BGR image
            include_hair: Whether to include hair in the mask
            blur_amount: Gaussian blur for smooth edges
        
        Returns:
            Binary mask (0.0-1.0)
        """
        classes = FACE_SKIN_CLASSES.copy()
        if include_hair:
            classes.extend(HAIR_CLASSES)
        
        return self.get_mask(image, classes, blur_amount)
    
    def get_hair_mask(self, image: np.ndarray, blur_amount: int = 15) -> np.ndarray:
        """
        Get mask for hair region only.
        
        Args:
            image: BGR image
            blur_amount: Gaussian blur for smooth edges
        
        Returns:
            Binary mask (0.0-1.0)
        """
        return self.get_mask(image, HAIR_CLASSES, blur_amount)



