import subprocess
import shutil
from pathlib import Path
import logging
import json
import os

logger = logging.getLogger(__name__)

def find_ffmpeg() -> str:
    """Find FFmpeg executable path."""
    # Check if in PATH
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg:
        return ffmpeg
    
    # Check common Windows locations
    common_paths = [
        Path.home() / "AppData/Local/Microsoft/WinGet/Packages",
        Path("C:/ffmpeg/bin"),
        Path("C:/Program Files/ffmpeg/bin"),
    ]
    
    for base_path in common_paths:
        if base_path.exists():
            # Search for ffmpeg.exe
            for ffmpeg_exe in base_path.rglob("ffmpeg.exe"):
                return str(ffmpeg_exe)
    
    return 'ffmpeg'  # Fallback to hoping it's in PATH

def find_ffprobe() -> str:
    """Find FFprobe executable path."""
    ffprobe = shutil.which('ffprobe')
    if ffprobe:
        return ffprobe
    
    # Try same directory as ffmpeg
    ffmpeg_path = find_ffmpeg()
    if ffmpeg_path != 'ffmpeg':
        ffprobe_path = Path(ffmpeg_path).parent / 'ffprobe.exe'
        if ffprobe_path.exists():
            return str(ffprobe_path)
    
    return 'ffprobe'

# Cache the paths
FFMPEG_PATH = find_ffmpeg()
FFPROBE_PATH = find_ffprobe()

def run_ffmpeg(cmd: list, description: str = "FFmpeg") -> bool:
    """Run ffmpeg command and return success status."""
    # Replace 'ffmpeg' with actual path
    if cmd[0] == 'ffmpeg':
        cmd[0] = FFMPEG_PATH
    elif cmd[0] == 'ffprobe':
        cmd[0] = FFPROBE_PATH
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        if result.returncode != 0:
            logger.error(f"{description} failed: {result.stderr}")
            return False
        return True
    except Exception as e:
        logger.error(f"{description} exception: {e}")
        return False

def has_audio(video_path: Path) -> bool:
    """Check if video has audio stream using ffprobe."""
    try:
        cmd = [
            FFPROBE_PATH, '-v', 'quiet', '-print_format', 'json',
            '-show_streams', str(video_path)
        ]
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return any(s.get('codec_type') == 'audio' for s in data.get('streams', []))
    except Exception as e:
        logger.warning(f"Could not probe audio: {e}")
    return False

def extract_audio(video_path: Path, audio_path: Path) -> bool:
    """Extract audio from video file."""
    cmd = [
        'ffmpeg', '-y', '-i', str(video_path),
        '-vn', '-acodec', 'copy',
        str(audio_path)
    ]
    return run_ffmpeg(cmd, "Extract audio")

def merge_audio_video(video_path: Path, audio_path: Path, output_path: Path) -> bool:
    """Merge video and audio streams."""
    cmd = [
        'ffmpeg', '-y',
        '-i', str(video_path),
        '-i', str(audio_path),
        '-c:v', 'copy', '-c:a', 'aac',
        '-map', '0:v:0', '-map', '1:a:0',
        '-shortest',
        str(output_path)
    ]
    return run_ffmpeg(cmd, "Merge audio")

def copy_audio_to_video(source_video: Path, target_video: Path, output_path: Path) -> bool:
    """
    Copy audio from source_video and merge with target_video (no audio).
    This is the most reliable method - takes audio directly from original.
    """
    cmd = [
        'ffmpeg', '-y',
        '-i', str(target_video),   # Video without audio
        '-i', str(source_video),   # Original video with audio
        '-c:v', 'copy',            # Copy video stream as-is
        '-c:a', 'aac',             # Re-encode audio to AAC
        '-map', '0:v:0',           # Take video from first input
        '-map', '1:a:0?',          # Take audio from second input (optional)
        '-shortest',
        str(output_path)
    ]
    return run_ffmpeg(cmd, "Copy audio to video")

