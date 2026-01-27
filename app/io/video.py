import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class VideoReader:
    def __init__(self, path: Path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def stream(self) -> Generator[Tuple[bool, np.ndarray], None, None]:
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield ret, frame
            
    def release(self):
        self.cap.release()

class VideoWriter:
    def __init__(self, path: Path, fps: float, resolution: Tuple[int, int], codec: str = 'mp4v'):
        self.path = str(path)
        self.fps = fps
        self.resolution = resolution
        self.fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(self.path, self.fourcc, self.fps, self.resolution)
        
    def write(self, frame):
        if frame.shape[:2][::-1] != self.resolution:
            frame = cv2.resize(frame, self.resolution)
        self.writer.write(frame)
        
    def release(self):
        self.writer.release()

