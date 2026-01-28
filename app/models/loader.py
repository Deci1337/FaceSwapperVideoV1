import os
import urllib.request
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

MODELS = {
    "inswapper_128.onnx": "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
    "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth",
    "face_parsing.onnx": "https://huggingface.co/jonathandinu/face-parsing/resolve/main/model.onnx"
}

def download_model(model_name: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_path = cache_dir / model_name
    
    if not model_path.exists():
        url = MODELS.get(model_name)
        if not url:
            raise ValueError(f"Unknown model: {model_name}")
            
        logger.info(f"Downloading {model_name} to {model_path}...")
        urllib.request.urlretrieve(url, model_path)
        logger.info("Download complete.")
        
    return model_path

