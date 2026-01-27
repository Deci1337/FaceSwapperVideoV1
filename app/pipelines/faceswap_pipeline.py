import cv2
import numpy as np
from tqdm import tqdm
from app.config import FaceSwapConfig
from app.io.video import VideoReader, VideoWriter
from app.io.audio import has_audio, copy_audio_to_video
from app.models.swapper import FaceSwapper
from pathlib import Path
import logging
import shutil

logger = logging.getLogger(__name__)

class FaceSwapPipeline:
    def __init__(self, config: FaceSwapConfig):
        self.config = config
        self.reader = VideoReader(config.input_path)
        
        # If keeping audio, write to temp file first
        self.temp_output = None
        if config.keep_audio and has_audio(config.input_path):
            self.temp_output = config.output_path.with_name(f"_temp_{config.output_path.name}")
            output_for_writer = self.temp_output
        else:
            output_for_writer = config.output_path
            
        self.writer = VideoWriter(
            output_for_writer, 
            fps=config.fps if config.fps else self.reader.fps,
            resolution=(self.reader.width, self.reader.height)
        )
        self.swapper = FaceSwapper(
            cache_dir=config.model_cache_dir,
            provider=config.provider
        )
        
        self.enhancer = None
        if config.enable_enhancer:
            from app.models.enhancer import FaceEnhancer
            self.enhancer = FaceEnhancer(config.model_cache_dir, blend_weight=config.enhancer_weight)
        
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
                # Perform Swap with improved blending and color correction
                frame = self.swapper.swap(
                    self.source_face, target_face, frame,
                    color_correction=self.config.color_correction
                )
                
                # GFPGAN Enhancement (optional)
                if self.enhancer:
                    frame = self.enhancer.enhance(frame, target_face.bbox)
                
            self.writer.write(frame)
            
        self.reader.release()
        self.writer.release()
        
        # Handle Audio - copy from original video
        if self.temp_output and self.temp_output.exists():
            print("Adding audio from original video...")
            success = copy_audio_to_video(
                source_video=self.config.input_path,
                target_video=self.temp_output,
                output_path=self.config.output_path
            )
            
            if success:
                # Cleanup temp file
                self.temp_output.unlink()
            else:
                # Fallback: just rename temp to output
                logger.warning("Audio merge failed, output will have no audio")
                shutil.move(str(self.temp_output), str(self.config.output_path))

