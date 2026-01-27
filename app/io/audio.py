import ffmpeg
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

def extract_audio(video_path: Path, audio_path: Path):
    """Extract audio from video file."""
    try:
        (
            ffmpeg
            .input(str(video_path))
            .output(str(audio_path), acodec='aac', loglevel='quiet')
            .overwrite_output()
            .run()
        )
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error extracting audio: {e}")
        # It's possible the video has no audio, which is fine.

def merge_audio_video(video_path: Path, audio_path: Path, output_path: Path):
    """Merge video and audio streams."""
    try:
        video = ffmpeg.input(str(video_path))
        audio = ffmpeg.input(str(audio_path))
        
        (
            ffmpeg
            .output(video, audio, str(output_path), vcodec='copy', acodec='aac', loglevel='quiet')
            .overwrite_output()
            .run()
        )
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg error merging audio: {e}")
        # Fallback: just rename video if merge fails (e.g. no audio)
        if not output_path.exists():
            import shutil
            shutil.copy(video_path, output_path)

def has_audio(video_path: Path) -> bool:
    try:
        probe = ffmpeg.probe(str(video_path))
        return any(stream['codec_type'] == 'audio' for stream in probe['streams'])
    except Exception:
        return False

