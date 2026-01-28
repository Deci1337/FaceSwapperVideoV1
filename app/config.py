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
    provider: Literal['cuda', 'cpu', 'dml'] = 'cuda'
    model_cache_dir: Path = Field(default_factory=lambda: Path.home() / ".cache" / "offline-faceswap")
    enable_long_hair: bool = False
    reference_hair: Optional[Path] = None
    hair_backend: Literal['none', 'simple', 'hairfastgan', 'style-your-hair', 'stable-hair'] = 'none'
    hair_backend_cmd: Optional[str] = None
    hair_anchor: Literal['middle', 'first'] = 'middle'
    hair_blend: float = Field(default=0.8, ge=0.0, le=1.0)
    hair_mask_blur: int = Field(default=21, ge=3, le=61)
    hair_mask_ema: float = Field(default=0.7, ge=0.0, le=1.0)
    hair_mask_stride: int = Field(default=1, ge=1, le=10)
    hair_mask_scale: float = Field(default=1.0, ge=0.3, le=1.0)
    hair_face_exclusion: float = Field(default=0.9, ge=0.0, le=1.0)
    hair_head_extend: float = Field(default=2.0, ge=1.0, le=3.0)
    hair_seg: Literal['mediapipe', 'bisenet'] = 'mediapipe'
    hair_seg_model: Optional[Path] = None
    hair_seg_provider: Literal['cuda', 'cpu', 'dml'] = 'cuda'
    hair_mask_mode: Literal['head', 'hair'] = 'head'
    hair_multi_anchor: bool = False
    hair_anchor_stride: int = Field(default=10, ge=1, le=120)
    hair_anchor_yaw: float = Field(default=0.12, ge=0.05, le=0.6)
    hair_flow: bool = False
    hair_flow_alpha: float = Field(default=0.7, ge=0.0, le=1.0)
    hair_flow_pyr_scale: float = Field(default=0.5, ge=0.1, le=0.9)
    hair_flow_levels: int = Field(default=3, ge=1, le=6)
    hair_flow_winsize: int = Field(default=25, ge=5, le=51)
    hair_flow_iterations: int = Field(default=3, ge=1, le=10)
    hair_flow_poly_n: int = Field(default=5, ge=3, le=9)
    hair_flow_poly_sigma: float = Field(default=1.2, ge=0.5, le=2.0)
    hair_flow_flags: int = Field(default=0, ge=0, le=10)
    hair_color_match: bool = False

class AppConfig(BaseModel):
    model_cache_dir: Path = Field(default_factory=lambda: Path.home() / ".cache" / "offline-faceswap")
    
    def ensure_dirs(self):
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

config = AppConfig()
config.ensure_dirs()

