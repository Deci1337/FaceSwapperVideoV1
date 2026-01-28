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
            provider=config.provider,
            mask_mode=config.mask_mode
        )
        
        self.enhancer = None
        if config.enable_enhancer:
            from app.models.enhancer import FaceEnhancer
            self.enhancer = FaceEnhancer(config.model_cache_dir, blend_weight=config.enhancer_weight)
        
        # Background Replacement (optional)
        self.bg_remover = None
        self.background = None
        if config.background_image:
            from app.models.background import BackgroundRemover
            self.bg_remover = BackgroundRemover(model_selection=1)
            
            # Load and resize background to match video resolution
            bg_img = cv2.imread(str(config.background_image))
            if bg_img is None:
                raise ValueError(f"Could not read background image: {config.background_image}")
            
            # Resize to match video dimensions
            self.background = cv2.resize(
                bg_img, 
                (self.reader.width, self.reader.height),
                interpolation=cv2.INTER_AREA
            )
            logger.info(f"Background loaded and resized to {self.reader.width}x{self.reader.height}")

        # Long Hair Transfer (optional)
        self.hair_applier = None
        if config.enable_long_hair:
            if not config.reference_hair or not config.reference_hair.exists():
                raise ValueError("reference_hair is required when enable_long_hair is True")
            from app.hair_transfer.video_pipeline import HairVideoApplier, HairTransferConfig
            hair_cfg = HairTransferConfig(
                reference_image=config.reference_hair,
                backend=config.hair_backend,
                backend_cmd=config.hair_backend_cmd,
                anchor_strategy=config.hair_anchor,
                blend_strength=config.hair_blend,
                mask_blur=config.hair_mask_blur,
                mask_ema=config.hair_mask_ema,
                mask_stride=config.hair_mask_stride,
                mask_scale=config.hair_mask_scale,
                face_exclusion=config.hair_face_exclusion,
                head_extend=config.hair_head_extend,
                seg_mode=config.hair_seg,
                seg_model=config.hair_seg_model,
                seg_provider=config.hair_seg_provider,
                hair_mask_mode=config.hair_mask_mode,
                multi_anchor=config.hair_multi_anchor,
                anchor_stride=config.hair_anchor_stride,
                anchor_yaw=config.hair_anchor_yaw,
                flow_enabled=config.hair_flow,
                flow_alpha=config.hair_flow_alpha,
                flow_pyr_scale=config.hair_flow_pyr_scale,
                flow_levels=config.hair_flow_levels,
                flow_winsize=config.hair_flow_winsize,
                flow_iterations=config.hair_flow_iterations,
                flow_poly_n=config.hair_flow_poly_n,
                flow_poly_sigma=config.hair_flow_poly_sigma,
                flow_flags=config.hair_flow_flags,
                hair_color_match=config.hair_color_match,
            )
            self.hair_applier = HairVideoApplier(config.input_path, hair_cfg)
        
        # Load Source Face
        source_img = cv2.imread(str(config.source_face))
        if source_img is None:
            raise ValueError(f"Could not read source face: {config.source_face}")
            
        self.source_face = self.swapper.get_face(source_img)
        if self.source_face is None:
            raise ValueError("No face detected in source image")
        
        # Face tracking state for stability
        self.last_face = None
        self.face_miss_count = 0
        self.max_face_miss = 5  # Use cached face for up to 5 frames
        self.bbox_smooth_alpha = 0.7  # Smoothing factor for bbox
        self.last_bbox = None
        self.last_yaw = 0.0
        self.yaw_threshold = 0.35  # Skip swap if yaw > this (strong profile)
            
    def _smooth_bbox(self, bbox):
        """Smooth bbox between frames to reduce jitter."""
        bbox_arr = np.array(bbox, dtype=np.float32)
        if self.last_bbox is None:
            self.last_bbox = bbox_arr.copy()
            return bbox_arr
        
        smoothed = self.bbox_smooth_alpha * bbox_arr + (1 - self.bbox_smooth_alpha) * self.last_bbox
        self.last_bbox = smoothed.copy()
        return smoothed
    
    def _estimate_yaw(self, face) -> float:
        """Estimate head yaw (left/right rotation) from landmarks."""
        try:
            # Try to use 2d106 landmarks
            if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                lmk = face.landmark_2d_106
                # Use eyes and nose for yaw estimation
                left_eye = lmk[33]  # Left eye center
                right_eye = lmk[87]  # Right eye center
                nose = lmk[86]  # Nose tip
            elif hasattr(face, 'kps') and face.kps is not None:
                # Fallback to 5-point landmarks
                lmk = face.kps
                left_eye = lmk[0]
                right_eye = lmk[1]
                nose = lmk[2]
            else:
                return 0.0
            
            # Calculate yaw based on nose position relative to eye midpoint
            eye_mid = (left_eye + right_eye) * 0.5
            eye_dist = np.linalg.norm(right_eye - left_eye) + 1e-6
            yaw = (nose[0] - eye_mid[0]) / eye_dist
            
            # Smooth yaw
            self.last_yaw = 0.7 * yaw + 0.3 * self.last_yaw
            return abs(self.last_yaw)
        except:
            return 0.0
    
    def _get_stable_face(self, frame):
        """Get face with stability - use cached face if detection fails."""
        target_face = self.swapper.get_face(frame)
        
        if target_face is not None:
            # Face detected - update cache and smooth bbox
            smoothed = self._smooth_bbox(target_face.bbox)
            target_face.bbox = smoothed
            
            # Estimate yaw for quality control
            yaw = self._estimate_yaw(target_face)
            target_face._yaw = yaw  # Store for later use
            
            self.last_face = target_face
            self.face_miss_count = 0
            return target_face
        
        # Face not detected - try to use cached face
        self.face_miss_count += 1
        if self.last_face is not None and self.face_miss_count <= self.max_face_miss:
            logger.debug(f"Using cached face (miss {self.face_miss_count}/{self.max_face_miss})")
            return self.last_face
        
        return None
            
    def run(self):
        print(f"Processing face swap: {self.config.input_path} -> {self.config.output_path}")
        if self.bg_remover:
            print("Background replacement enabled")
        
        for ret, frame in tqdm(self.reader.stream(), total=self.reader.frame_count):
            if not ret:
                break
                
            # Detect faces with stability
            target_face = self._get_stable_face(frame)
            
            if target_face:
                # Check face angle - reduce swap intensity for profile views
                yaw = getattr(target_face, '_yaw', 0.0)
                
                # Skip swap entirely for very strong profiles (> 45 degrees)
                if yaw > self.yaw_threshold:
                    logger.debug(f"Skipping swap - face turned too much (yaw={yaw:.2f})")
                else:
                    # Adjust color correction based on yaw (less correction for profiles)
                    adjusted_cc = self.config.color_correction * max(0.3, 1.0 - yaw)
                    
                    # 1. Perform Face Swap (first, needs original context)
                    frame = self.swapper.swap(
                        self.source_face, target_face, frame,
                        color_correction=adjusted_cc,
                        include_hair=self.config.include_hair
                    )
                    
                    # 2. GFPGAN Enhancement (optional)
                    if self.enhancer:
                        frame = self.enhancer.enhance(frame, target_face.bbox)

            # 3. Long Hair Transfer (optional, after face swap)
            if self.hair_applier:
                frame = self.hair_applier.apply(frame)
            
            # 4. Background Replacement (last, to cut out the swapped person)
            if self.bg_remover and self.background is not None:
                frame = self.bg_remover.replace_background(
                    frame, self.background, 
                    threshold=self.config.bg_threshold
                )
                
            self.writer.write(frame)
            
        self.reader.release()
        self.writer.release()
        
        # Cleanup background remover
        if self.bg_remover:
            self.bg_remover.close()

        if self.hair_applier:
            self.hair_applier.close()
        
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

