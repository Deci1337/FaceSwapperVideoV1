"""
Test script for face swap functionality.
Usage: python scripts/test_faceswap.py --input video.mp4 --face face.jpg --output result.mp4
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Test face swap")
    parser.add_argument("--input", "-i", type=Path, required=True, help="Input video")
    parser.add_argument("--face", "-f", type=Path, required=True, help="Source face image")
    parser.add_argument("--output", "-o", type=Path, default=Path("output_faceswap.mp4"), help="Output video")
    parser.add_argument("--enhance", action="store_true", help="Enable GFPGAN enhancement")
    args = parser.parse_args()
    
    root = Path(__file__).resolve().parents[1]
    
    if not args.input.exists():
        print(f"Error: Input video not found: {args.input}")
        return 1
    if not args.face.exists():
        print(f"Error: Face image not found: {args.face}")
        return 1
    
    cmd = [
        sys.executable,
        str(root / "cli.py"),
        "swap",
        "--input", str(args.input),
        "--source-face", str(args.face),
        "--output", str(args.output),
        "--quality", "high",
        "--color-correction", "0.5",
    ]
    
    if args.enhance:
        cmd.extend(["--enable-enhancer", "--enhancer-weight", "0.6"])
    
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Done: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

