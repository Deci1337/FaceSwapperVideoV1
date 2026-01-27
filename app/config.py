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
    provider: Literal['cuda', 'cpu'] = 'cuda'
    model_cache_dir: Path = Field(default_factory=lambda: Path.home() / ".cache" / "offline-faceswap")

class AppConfig(BaseModel):
    model_cache_dir: Path = Field(default_factory=lambda: Path.home() / ".cache" / "offline-faceswap")
    
    def ensure_dirs(self):
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)

config = AppConfig()
config.ensure_dirs()

