from pydantic import BaseModel, Field
from pathlib import Path
from typing import Literal, Optional

class VideoConfig(BaseModel):
    input_path: Path
    output_path: Path
    fps: Optional[float] = None
    keep_audio: bool = True

class LandmarksConfig(VideoConfig):
    face_mode: Literal['bbox', 'mesh'] = 'mesh'
    draw_hands: bool = True

class FaceSwapConfig(VideoConfig):
    source_face: Path
    quality: Literal['low', 'medium', 'high'] = 'high'
    enable_enhancer: bool = False
    enhancer_weight: float = Field(default=0.7, ge=0.0, le=1.0)  # GFPGAN blend weight
    color_correction: float = Field(default=0.5, ge=0.0, le=1.0)  # Color matching strength
    mask_mode: Literal['bbox', 'parsing'] = 'bbox'  # Masking mode: ellipse or face parsing
    include_hair: bool = False  # Include hair in parsing mask (for gender swap)
    background_image: Optional[Path] = None  # Custom background image for replacement
    bg_threshold: float = Field(default=0.6, ge=0.3, le=0.9)  # Background segmentation threshold
    provider: Literal['cuda', 'cpu'] = 'cuda'
    model_cache_dir: Path = Field(default_factory=lambda: Path.home() / ".cache" / "offline-faceswap")

class AppConfig(BaseModel):
    model_cache_dir: Path = Field(default_factory=lambda: Path.home() / ".cache" / "offline-faceswap")
    
    def ensure_dirs(self):
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

config = AppConfig()
config.ensure_dirs()

