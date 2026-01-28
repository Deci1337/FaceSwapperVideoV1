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
    enable_enhancer: Annotated[bool, typer.Option(help="Enable GFPGAN face enhancement")] = False,
    enhancer_weight: Annotated[float, typer.Option(help="GFPGAN blend weight 0.0-1.0 (lower=natural)")] = 0.7,
    color_correction: Annotated[float, typer.Option(help="Color matching strength 0.0-1.0")] = 0.5,
    mask_mode: Annotated[str, typer.Option(help="'bbox' (ellipse) or 'parsing' (smart face parsing)")] = "bbox",
    include_hair: Annotated[bool, typer.Option(help="Include hair in mask (for gender swap, use with --mask-mode parsing)")] = False,
    background: Annotated[Optional[Path], typer.Option("--background", "-bg", help="Background image for replacement")] = None,
    bg_threshold: Annotated[float, typer.Option(help="Background threshold 0.3-0.9 (higher=stricter cut)")] = 0.6,
    long_hair: Annotated[bool, typer.Option(help="Enable long-hair transfer module")] = False,
    reference_hair: Annotated[Optional[Path], typer.Option(help="Reference hair image (long hair)")] = None,
    hair_backend: Annotated[str, typer.Option(help="Hair transfer backend: none|simple|hairfastgan|style-your-hair|stable-hair")] = "none",
    hair_backend_cmd: Annotated[Optional[str], typer.Option(help="External command for hair backend (uses {anchor} {reference} {output})")] = None,
    hair_anchor: Annotated[str, typer.Option(help="Anchor frame: middle or first")] = "middle",
    hair_blend: Annotated[float, typer.Option(help="Hair blend strength 0.0-1.0")] = 0.8,
    hair_mask_blur: Annotated[int, typer.Option(help="Hair mask blur kernel (odd, 3-61)")] = 21,
    hair_mask_ema: Annotated[float, typer.Option(help="Hair mask temporal smoothing 0.0-1.0")] = 0.7,
    hair_mask_stride: Annotated[int, typer.Option(help="Recompute mask every N frames (speed)")] = 1,
    hair_mask_scale: Annotated[float, typer.Option(help="Mask compute scale 0.3-1.0 (speed)")] = 1.0,
    hair_face_exclusion: Annotated[float, typer.Option(help="Face exclusion strength 0.0-1.0")] = 0.9,
    hair_head_extend: Annotated[float, typer.Option(help="Extend hair area downwards (1.0-3.0)")] = 2.0,
    hair_seg: Annotated[str, typer.Option(help="Hair segmentation: mediapipe|bisenet")] = "mediapipe",
    hair_seg_model: Annotated[Optional[Path], typer.Option(help="BiSeNet ONNX model path")] = None,
    hair_seg_provider: Annotated[str, typer.Option(help="BiSeNet provider: cuda|cpu|dml")] = "cuda",
    hair_mask_mode: Annotated[str, typer.Option(help="Hair mask mode: head|hair")] = "head",
    hair_multi_anchor: Annotated[bool, typer.Option(help="Enable multi-anchor (front/left/right)")] = False,
    hair_anchor_stride: Annotated[int, typer.Option(help="Frame stride for anchor selection")] = 10,
    hair_anchor_yaw: Annotated[float, typer.Option(help="Yaw threshold for anchors (0.05-0.6)")] = 0.12,
    hair_flow: Annotated[bool, typer.Option(help="Enable optical flow for hair")] = False,
    hair_flow_alpha: Annotated[float, typer.Option(help="Flow blend alpha 0.0-1.0")] = 0.7,
    hair_flow_pyr_scale: Annotated[float, typer.Option(help="Flow pyr_scale 0.1-0.9")] = 0.5,
    hair_flow_levels: Annotated[int, typer.Option(help="Flow pyramid levels 1-6")] = 3,
    hair_flow_winsize: Annotated[int, typer.Option(help="Flow window size 5-51")] = 25,
    hair_flow_iterations: Annotated[int, typer.Option(help="Flow iterations 1-10")] = 3,
    hair_flow_poly_n: Annotated[int, typer.Option(help="Flow poly_n 3-9")] = 5,
    hair_flow_poly_sigma: Annotated[float, typer.Option(help="Flow poly_sigma 0.5-2.0")] = 1.2,
    hair_flow_flags: Annotated[int, typer.Option(help="Flow flags (0-10)")] = 0,
    hair_color_match: Annotated[bool, typer.Option(help="Match hair color to reference")] = False,
    keep_audio: Annotated[bool, typer.Option(help="Preserve original audio")] = True,
    provider: Annotated[str, typer.Option(help="'cuda' (NVIDIA), 'dml' (AMD/Intel), or 'cpu'")] = "cuda",
):
    """
    Replace faces in video with source face image.
    
    Quality tips:
    - Use --enable-enhancer for clearer face details
    - Adjust --color-correction 0.3-0.7 for natural lighting match
    - Use --provider cpu if no NVIDIA GPU (slower)
    - Use --background to replace video background
    
    Gender swap (male to female):
    - Use --mask-mode parsing --include-hair for better results
    - This uses face parsing to include more of the source face shape
    """
    from app.config import FaceSwapConfig
    from app.pipelines.faceswap_pipeline import FaceSwapPipeline
    
    validate_input_file(input, ['.mp4', '.avi', '.mov', '.mkv'])
    validate_input_file(source_face, ['.jpg', '.jpeg', '.png'])
    
    if background:
        validate_input_file(background, ['.jpg', '.jpeg', '.png', '.webp'])

    if long_hair and reference_hair:
        validate_input_file(reference_hair, ['.jpg', '.jpeg', '.png'])
    
    if quality not in ['low', 'medium', 'high']:
        raise typer.BadParameter("quality must be 'low', 'medium', or 'high'")
    if provider not in ['cuda', 'cpu', 'dml']:
        raise typer.BadParameter("provider must be 'cuda', 'dml', or 'cpu'")
    if mask_mode not in ['bbox', 'parsing']:
        raise typer.BadParameter("mask_mode must be 'bbox' or 'parsing'")
    if hair_backend not in ['none', 'simple', 'hairfastgan', 'style-your-hair', 'stable-hair']:
        raise typer.BadParameter("hair_backend must be one of: none, simple, hairfastgan, style-your-hair, stable-hair")
    if hair_anchor not in ['middle', 'first']:
        raise typer.BadParameter("hair_anchor must be 'middle' or 'first'")
    if not 0.0 <= color_correction <= 1.0:
        raise typer.BadParameter("color_correction must be between 0.0 and 1.0")
    if not 0.0 <= enhancer_weight <= 1.0:
        raise typer.BadParameter("enhancer_weight must be between 0.0 and 1.0")
    if not 0.3 <= bg_threshold <= 0.9:
        raise typer.BadParameter("bg_threshold must be between 0.3 and 0.9")
    if not 0.0 <= hair_blend <= 1.0:
        raise typer.BadParameter("hair_blend must be between 0.0 and 1.0")
    if not 3 <= hair_mask_blur <= 61:
        raise typer.BadParameter("hair_mask_blur must be between 3 and 61")
    if not 0.0 <= hair_mask_ema <= 1.0:
        raise typer.BadParameter("hair_mask_ema must be between 0.0 and 1.0")
    if not 1 <= hair_mask_stride <= 10:
        raise typer.BadParameter("hair_mask_stride must be between 1 and 10")
    if not 0.3 <= hair_mask_scale <= 1.0:
        raise typer.BadParameter("hair_mask_scale must be between 0.3 and 1.0")
    if not 0.0 <= hair_face_exclusion <= 1.0:
        raise typer.BadParameter("hair_face_exclusion must be between 0.0 and 1.0")
    if not 1.0 <= hair_head_extend <= 3.0:
        raise typer.BadParameter("hair_head_extend must be between 1.0 and 3.0")
    if hair_seg not in ["mediapipe", "bisenet"]:
        raise typer.BadParameter("hair_seg must be mediapipe or bisenet")
    if hair_seg_provider not in ["cuda", "cpu", "dml"]:
        raise typer.BadParameter("hair_seg_provider must be cuda, cpu, or dml")
    if hair_seg == "bisenet" and not hair_seg_model:
        raise typer.BadParameter("--hair-seg-model is required for bisenet")
    if hair_mask_mode not in ["head", "hair"]:
        raise typer.BadParameter("hair_mask_mode must be head or hair")
    if not 1 <= hair_anchor_stride <= 120:
        raise typer.BadParameter("hair_anchor_stride must be between 1 and 120")
    if not 0.05 <= hair_anchor_yaw <= 0.6:
        raise typer.BadParameter("hair_anchor_yaw must be between 0.05 and 0.6")
    if not 0.0 <= hair_flow_alpha <= 1.0:
        raise typer.BadParameter("hair_flow_alpha must be between 0.0 and 1.0")
    if not 0.1 <= hair_flow_pyr_scale <= 0.9:
        raise typer.BadParameter("hair_flow_pyr_scale must be between 0.1 and 0.9")
    if not 1 <= hair_flow_levels <= 6:
        raise typer.BadParameter("hair_flow_levels must be between 1 and 6")
    if not 5 <= hair_flow_winsize <= 51:
        raise typer.BadParameter("hair_flow_winsize must be between 5 and 51")
    if not 1 <= hair_flow_iterations <= 10:
        raise typer.BadParameter("hair_flow_iterations must be between 1 and 10")
    if not 3 <= hair_flow_poly_n <= 9:
        raise typer.BadParameter("hair_flow_poly_n must be between 3 and 9")
    if not 0.5 <= hair_flow_poly_sigma <= 2.0:
        raise typer.BadParameter("hair_flow_poly_sigma must be between 0.5 and 2.0")
    if not 0 <= hair_flow_flags <= 10:
        raise typer.BadParameter("hair_flow_flags must be between 0 and 10")
    if long_hair and not reference_hair:
        raise typer.BadParameter("reference_hair is required when long_hair is enabled")
    
    config = FaceSwapConfig(
        input_path=input,
        output_path=output,
        source_face=source_face,
        quality=quality,
        enable_enhancer=enable_enhancer,
        enhancer_weight=enhancer_weight,
        color_correction=color_correction,
        mask_mode=mask_mode,
        include_hair=include_hair,
        background_image=background,
        bg_threshold=bg_threshold,
        enable_long_hair=long_hair,
        reference_hair=reference_hair,
        hair_backend=hair_backend,
        hair_backend_cmd=hair_backend_cmd,
        hair_anchor=hair_anchor,
        hair_blend=hair_blend,
        hair_mask_blur=hair_mask_blur,
        hair_mask_ema=hair_mask_ema,
        hair_mask_stride=hair_mask_stride,
        hair_mask_scale=hair_mask_scale,
        hair_face_exclusion=hair_face_exclusion,
        hair_head_extend=hair_head_extend,
        hair_seg=hair_seg,
        hair_seg_model=hair_seg_model,
        hair_seg_provider=hair_seg_provider,
        hair_mask_mode=hair_mask_mode,
        hair_multi_anchor=hair_multi_anchor,
        hair_anchor_stride=hair_anchor_stride,
        hair_anchor_yaw=hair_anchor_yaw,
        hair_flow=hair_flow,
        hair_flow_alpha=hair_flow_alpha,
        hair_flow_pyr_scale=hair_flow_pyr_scale,
        hair_flow_levels=hair_flow_levels,
        hair_flow_winsize=hair_flow_winsize,
        hair_flow_iterations=hair_flow_iterations,
        hair_flow_poly_n=hair_flow_poly_n,
        hair_flow_poly_sigma=hair_flow_poly_sigma,
        hair_flow_flags=hair_flow_flags,
        hair_color_match=hair_color_match,
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

