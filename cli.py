import typer
from pathlib import Path
from typing import Optional
from typing_extensions import Annotated
import logging
import sys

app = typer.Typer(help="Offline Video Face Swap + Landmarks Debug Tool")
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def validate_input_file(path: Path, extensions: list = None) -> Path:
    """Validate that input file exists."""
    if not path.exists():
        raise typer.BadParameter(f"File not found: {path}")
    if extensions and path.suffix.lower() not in extensions:
        raise typer.BadParameter(f"Invalid file type. Expected: {extensions}")
    return path

@app.command()
def landmarks(
    input: Annotated[Path, typer.Option(help="Input video file")] = ...,
    output: Annotated[Path, typer.Option(help="Output debug video with landmarks")] = ...,
    face_mode: Annotated[str, typer.Option(help="'bbox' or 'mesh' for face landmarks")] = "mesh",
    fps: Annotated[Optional[float], typer.Option(help="Output FPS (default: same as input)")] = None,
):
    """
    Generate debug video with hand/face skeleton overlay.
    """
    # Lazy imports to speed up CLI help
    from app.config import LandmarksConfig
    from app.pipelines.landmarks_pipeline import LandmarksPipeline
    
    validate_input_file(input, ['.mp4', '.avi', '.mov', '.mkv'])
    
    if face_mode not in ['bbox', 'mesh']:
        raise typer.BadParameter("face_mode must be 'bbox' or 'mesh'")
    
    config = LandmarksConfig(
        input_path=input,
        output_path=output,
        face_mode=face_mode,
        fps=fps
    )
    
    try:
        pipeline = LandmarksPipeline(config)
        pipeline.run()
        typer.echo(f"Done! Output saved to: {output}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise typer.Exit(1)

@app.command()
def swap(
    input: Annotated[Path, typer.Option(help="Target video file")] = ...,
    source_face: Annotated[Path, typer.Option(help="Source face image (jpg/png)")] = ...,
    output: Annotated[Path, typer.Option(help="Output face-swap video")] = ...,
    quality: Annotated[str, typer.Option(help="'low', 'medium', or 'high'")] = "high",
    enable_enhancer: Annotated[bool, typer.Option(help="Enable GFPGAN enhancement")] = False,
    keep_audio: Annotated[bool, typer.Option(help="Preserve original audio")] = True,
    provider: Annotated[str, typer.Option(help="'cuda' or 'cpu'")] = "cuda",
):
    """
    Replace faces in video with source face image.
    """
    from app.config import FaceSwapConfig
    from app.pipelines.faceswap_pipeline import FaceSwapPipeline
    
    validate_input_file(input, ['.mp4', '.avi', '.mov', '.mkv'])
    validate_input_file(source_face, ['.jpg', '.jpeg', '.png'])
    
    if quality not in ['low', 'medium', 'high']:
        raise typer.BadParameter("quality must be 'low', 'medium', or 'high'")
    if provider not in ['cuda', 'cpu']:
        raise typer.BadParameter("provider must be 'cuda' or 'cpu'")
    
    config = FaceSwapConfig(
        input_path=input,
        output_path=output,
        source_face=source_face,
        quality=quality,
        enable_enhancer=enable_enhancer,
        keep_audio=keep_audio,
        provider=provider
    )
    
    try:
        pipeline = FaceSwapPipeline(config)
        pipeline.run()
        typer.echo(f"Done! Output saved to: {output}")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise typer.Exit(1)

@app.command("all")
def run_all(
    input: Annotated[Path, typer.Option(help="Input video")] = ...,
    source_face: Annotated[Path, typer.Option(help="Source face image")] = ...,
    output_dir: Annotated[Path, typer.Option(help="Output directory")] = Path("."),
):
    """
    Run both landmarks and face-swap pipelines.
    """
    from app.pipelines.orchestrator import Orchestrator
    
    validate_input_file(input, ['.mp4', '.avi', '.mov', '.mkv'])
    validate_input_file(source_face, ['.jpg', '.jpeg', '.png'])
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        orch = Orchestrator()
        orch.run_all(
            input, 
            source_face,
            output_landmarks=output_dir / "debug_landmarks.mp4",
            output_swap=output_dir / "result_faceswap.mp4"
        )
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()

