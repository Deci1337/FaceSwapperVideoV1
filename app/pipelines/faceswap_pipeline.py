import cv2
import numpy as np
from tqdm import tqdm
from app.config import FaceSwapConfig
from app.io.video import VideoReader, VideoWriter
from app.io.audio import extract_audio, merge_audio_video, has_audio
from app.models.swapper import FaceSwapper
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FaceSwapPipeline:
    def __init__(self, config: FaceSwapConfig):
        self.config = config
        self.reader = VideoReader(config.input_path)
        self.writer = VideoWriter(
            config.output_path, 
            fps=config.fps if config.fps else self.reader.fps,
            resolution=(self.reader.width, self.reader.height)
        )
        self.swapper = FaceSwapper(
            cache_dir=config.model_cache_dir,
            provider=config.provider
        )
        
        self.enhancer = None
        if config.enable_enhancer:
            # Lazy import to avoid basicsr/torchvision compatibility issues
            from app.models.enhancer import FaceEnhancer
            self.enhancer = FaceEnhancer(config.model_cache_dir)
        
        # Load Source Face
        source_img = cv2.imread(str(config.source_face))
        if source_img is None:
            raise ValueError(f"Could not read source face: {config.source_face}")
            
        self.source_face = self.swapper.get_face(source_img)
        if self.source_face is None:
            raise ValueError("No face detected in source image")
            
    def run(self):
        print(f"Processing face swap: {self.config.input_path} -> {self.config.output_path}")
        
        for ret, frame in tqdm(self.reader.stream(), total=self.reader.frame_count):
            if not ret:
                break
                
            # Detect faces in target frame
            target_face = self.swapper.get_face(frame)
            
            if target_face:
                # Perform Swap
                frame = self.swapper.swap(self.source_face, target_face, frame)
                
                # GFPGAN Enhancement
                if self.enhancer:
                    frame = self.enhancer.enhance(frame, target_face.bbox)
                
            self.writer.write(frame)
            
        self.reader.release()
        self.writer.release()
        
        # Handle Audio
        if self.config.keep_audio and has_audio(self.config.input_path):
            print("Merging audio...")
            temp_audio = self.config.output_path.with_suffix('.aac')
            extract_audio(self.config.input_path, temp_audio)
            
            temp_video = self.config.output_path.with_name(f"temp_{self.config.output_path.name}")
            # Rename current output to temp
            if self.config.output_path.exists():
                self.config.output_path.rename(temp_video)
                
            merge_audio_video(temp_video, temp_audio, self.config.output_path)
            
            # Cleanup
            if temp_audio.exists(): temp_audio.unlink()
            if temp_video.exists(): temp_video.unlink()

