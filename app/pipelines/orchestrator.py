from pathlib import Path
from app.config import LandmarksConfig, FaceSwapConfig
from app.pipelines.landmarks_pipeline import LandmarksPipeline
from app.pipelines.faceswap_pipeline import FaceSwapPipeline
import logging

logger = logging.getLogger(__name__)

class Orchestrator:
    def __init__(self, provider: str = 'cuda', enable_enhancer: bool = False,
                 enhancer_weight: float = 0.7, color_correction: float = 0.5):
        self.provider = provider
        self.enable_enhancer = enable_enhancer
        self.enhancer_weight = enhancer_weight
        self.color_correction = color_correction
        
    def run_all(self, input_path: Path, source_face: Path, 
                output_landmarks: Path = Path("debug_landmarks.mp4"),
                output_swap: Path = Path("result_faceswap.mp4")):
        
        print("=" * 50)
        print("Starting Full Pipeline")
        print("=" * 50)
        
        # 1. Landmarks Debug
        print("\n[1/2] Generating Landmarks Debug Video...")
        landmarks_config = LandmarksConfig(
            input_path=input_path,
            output_path=output_landmarks,
            face_mode='mesh',
            draw_hands=True
        )
        lm_pipeline = LandmarksPipeline(landmarks_config)
        lm_pipeline.run()
        print(f"Landmarks saved: {output_landmarks}")
        
        # 2. Face Swap
        print("\n[2/2] Performing Face Swap...")
        swap_config = FaceSwapConfig(
            input_path=input_path,
            output_path=output_swap,
            source_face=source_face,
            quality='high',
            enable_enhancer=self.enable_enhancer,
            enhancer_weight=self.enhancer_weight,
            color_correction=self.color_correction,
            provider=self.provider,
            keep_audio=True
        )
        swap_pipeline = FaceSwapPipeline(swap_config)
        swap_pipeline.run()
        print(f"Face swap saved: {output_swap}")
        
        print("\n" + "=" * 50)
        print("All Done!")
        print("=" * 50)
        print(f"Debug output:  {output_landmarks}")
        print(f"Swap output:   {output_swap}")

