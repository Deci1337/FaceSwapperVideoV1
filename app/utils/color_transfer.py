"""
Color transfer utilities for matching skin tones between faces.
Uses LAB color space for perceptually uniform color matching.
"""

import cv2
import numpy as np
from typing import Tuple

def get_face_region(frame: np.ndarray, bbox: np.ndarray, padding: float = 0.1) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """Extract face region from frame with optional padding."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    
    # Add padding
    pad_w = int((x2 - x1) * padding)
    pad_h = int((y2 - y1) * padding)
    
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def compute_color_stats(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and std of image in LAB color space."""
    # Convert to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Compute stats for each channel
    mean = np.mean(lab, axis=(0, 1))
    std = np.std(lab, axis=(0, 1))
    
    # Avoid division by zero
    std = np.maximum(std, 1e-6)
    
    return mean, std


def transfer_color(source_img: np.ndarray, target_img: np.ndarray, 
                   strength: float = 0.8) -> np.ndarray:
    """
    Transfer color statistics from target to source image.
    
    This makes the source image match the color/lighting of the target.
    Useful for face swap to match the target video's lighting conditions.
    
    Args:
        source_img: Image to be color-corrected (swapped face region)
        target_img: Reference image to match colors from (original face region)
        strength: How much to apply the transfer (0.0 = none, 1.0 = full)
    
    Returns:
        Color-corrected source image
    """
    if source_img.size == 0 or target_img.size == 0:
        return source_img
        
    # Compute stats
    src_mean, src_std = compute_color_stats(source_img)
    tgt_mean, tgt_std = compute_color_stats(target_img)
    
    # Convert source to LAB
    src_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # Apply transfer: normalize, scale, shift
    for i in range(3):
        src_lab[:, :, i] = (src_lab[:, :, i] - src_mean[i]) * (tgt_std[i] / src_std[i]) + tgt_mean[i]
    
    # Clip values to valid range
    src_lab[:, :, 0] = np.clip(src_lab[:, :, 0], 0, 255)  # L: 0-255 in OpenCV
    src_lab[:, :, 1] = np.clip(src_lab[:, :, 1], 0, 255)  # a: 0-255
    src_lab[:, :, 2] = np.clip(src_lab[:, :, 2], 0, 255)  # b: 0-255
    
    # Convert back to BGR
    corrected = cv2.cvtColor(src_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # Blend with original based on strength
    if strength < 1.0:
        corrected = cv2.addWeighted(corrected, strength, source_img, 1 - strength, 0)
    
    return corrected


def apply_face_color_correction(frame: np.ndarray, original_frame: np.ndarray,
                                 bbox: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """
    Apply color correction to the face region in frame.
    
    Matches the color/lighting of the swapped face to the original frame's face.
    
    Args:
        frame: Frame with swapped face
        original_frame: Original frame before swap
        bbox: Face bounding box [x1, y1, x2, y2]
        strength: Color correction strength (0.0-1.0)
    
    Returns:
        Frame with color-corrected face region
    """
    if strength <= 0:
        return frame
        
    # Extract face regions
    swapped_face, coords = get_face_region(frame, bbox, padding=0.05)
    original_face, _ = get_face_region(original_frame, bbox, padding=0.05)
    
    if swapped_face.size == 0 or original_face.size == 0:
        return frame
    
    # Resize original face to match swapped face size if needed
    if swapped_face.shape[:2] != original_face.shape[:2]:
        original_face = cv2.resize(original_face, (swapped_face.shape[1], swapped_face.shape[0]))
    
    # Apply color transfer
    corrected_face = transfer_color(swapped_face, original_face, strength)
    
    # Paste back
    x1, y1, x2, y2 = coords
    result = frame.copy()
    result[y1:y2, x1:x2] = corrected_face
    
    return result

