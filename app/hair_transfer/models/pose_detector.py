"""
Pose Detection for ControlNet

Uses OpenPose or MediaPipe for detecting body/face pose.
"""

import cv2
import numpy as np
from PIL import Image
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class PoseDetector:
    """
    Detects pose for ControlNet conditioning.
    
    Uses controlnet_aux OpenPose or falls back to MediaPipe.
    """
    
    def __init__(self, use_openpose: bool = True):
        """
        Args:
            use_openpose: Use OpenPose (better) or MediaPipe (lighter)
        """
        self.openpose = None
        self.mediapipe_pose = None
        self.use_openpose = use_openpose
        
        self._load_detector()
        
    def _load_detector(self) -> None:
        """Load pose detection model."""
        if self.use_openpose:
            try:
                from controlnet_aux import OpenposeDetector
                self.openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
                logger.info("OpenPose detector loaded")
                return
            except Exception as e:
                logger.warning(f"OpenPose failed to load: {e}")
                logger.warning("Falling back to MediaPipe")
        
        # Fallback to MediaPipe
        try:
            import mediapipe as mp
            self.mediapipe_pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_pose = mp.solutions.pose
            logger.info("MediaPipe pose detector loaded")
        except Exception as e:
            logger.error(f"MediaPipe pose also failed: {e}")
            raise RuntimeError("No pose detector available")
    
    def detect(self, image: np.ndarray) -> Image.Image:
        """
        Detect pose and return control image.
        
        Args:
            image: BGR numpy array
            
        Returns:
            PIL Image with pose visualization (for ControlNet)
        """
        if self.openpose is not None:
            return self._detect_openpose(image)
        else:
            return self._detect_mediapipe(image)
    
    def _detect_openpose(self, image: np.ndarray) -> Image.Image:
        """Use OpenPose for detection."""
        # OpenPose expects RGB PIL Image
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        
        # Detect pose - returns PIL Image with skeleton overlay
        pose_image = self.openpose(pil_image, hand_and_face=True)
        
        return pose_image
    
    def _detect_mediapipe(self, image: np.ndarray) -> Image.Image:
        """Use MediaPipe for detection (fallback)."""
        h, w = image.shape[:2]
        
        # Create blank image for drawing
        pose_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Convert to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        
        # Detect
        results = self.mediapipe_pose.process(rgb)
        
        if results.pose_landmarks:
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                pose_image,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 255, 255), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 255, 255), thickness=2
                )
            )
        
        return Image.fromarray(pose_image)
    
    def close(self) -> None:
        """Release resources."""
        if self.mediapipe_pose is not None:
            self.mediapipe_pose.close()
            self.mediapipe_pose = None

