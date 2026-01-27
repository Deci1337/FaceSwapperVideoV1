# Offline Video Face Swap + Skeleton Debug

A CLI tool for offline video face swapping and landmarks visualization.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Requirements

### Hardware
- **GPU (recommended):** NVIDIA GPU with CUDA support (RTX 3060+ recommended)
- **RAM:** 8GB+ (16GB recommended for HD video)
- **Disk:** ~2GB for models

### Software
- Python 3.9 - 3.11 (3.10 recommended)
- CUDA Toolkit 11.8+ (for GPU acceleration)
- FFmpeg (for audio handling)

---

## Installation

### Step 1: Install FFmpeg

**Windows (using Chocolatey):**
```powershell
choco install ffmpeg
```

**Windows (manual):**
1. Download from https://ffmpeg.org/download.html
2. Extract to `C:\ffmpeg`
3. Add `C:\ffmpeg\bin` to PATH

**Linux (Ubuntu/Debian):**
```bash
sudo apt update && sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

Verify installation:
```bash
ffmpeg -version
```

---

### Step 2: Create Virtual Environment

```bash
cd offline-faceswap

# Create venv
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Windows CMD)
venv\Scripts\activate.bat

# Activate (Linux/macOS)
source venv/bin/activate
```

---

### Step 3: Install PyTorch with CUDA

**For NVIDIA GPU (CUDA 11.8):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For NVIDIA GPU (CUDA 12.1):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only (slow, not recommended):**
```bash
pip install torch torchvision
```

Verify CUDA:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** First run will download models (~500MB):
- `buffalo_l` (InsightFace face detection/recognition)
- `inswapper_128.onnx` (face swapping model)
- `GFPGANv1.4.pth` (face enhancement, only if --enable-enhancer)

---

## Usage

### 1. Landmarks Debug Mode

Visualizes hand skeleton (21 points) and face mesh (468 points):

```bash
python cli.py landmarks --input video.mp4 --output debug.mp4
```

Options:
- `--face-mode mesh` (default) - Full face mesh visualization
- `--face-mode bbox` - Simple bounding box
- `--fps 30` - Override output FPS

Example:
```bash
python cli.py landmarks --input input.mp4 --output debug_landmarks.mp4 --face-mode mesh
```

---

### 2. Face Swap Mode

Replaces faces in video with a source face image:

```bash
python cli.py swap --input video.mp4 --source-face face.jpg --output result.mp4
```

Options:
- `--quality low|medium|high` (default: high)
- `--enable-enhancer` - Use GFPGAN for face enhancement
- `--keep-audio` (default: True) - Preserve original audio
- `--provider cuda|cpu` (default: cuda)

Example (basic):
```bash
python cli.py swap --input input.mp4 --source-face myface.jpg --output swapped.mp4
```

Example (with enhancement):
```bash
python cli.py swap --input input.mp4 --source-face myface.jpg --output swapped_hq.mp4 --enable-enhancer --quality high
```

---

### 3. Run Both Pipelines

```bash
python cli.py all --input video.mp4 --source-face face.jpg --output-dir ./output
```

This creates:
- `output/debug_landmarks.mp4` - Landmarks visualization
- `output/result_faceswap.mp4` - Face-swapped video

---

## Testing

### Test 1: Verify Installation

```bash
# Check Python version
python --version  # Should be 3.9-3.11

# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check ONNX providers
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"

# Check MediaPipe
python -c "import mediapipe; print('MediaPipe OK')"

# Check FFmpeg
ffmpeg -version
```

---

### Test 2: Quick Landmarks Test

Create a short test video (5 seconds recommended):

```bash
python cli.py landmarks --input test_video.mp4 --output test_debug.mp4
```

Expected output:
- Progress bar showing frame processing
- Output file with hand skeletons and face mesh overlay
- Processing time: ~10-30 sec for 5 sec video

---

### Test 3: Face Swap Test

Prepare:
1. `test_video.mp4` - Video with a clear face
2. `source_face.jpg` - Clear frontal face photo

```bash
python cli.py swap --input test_video.mp4 --source-face source_face.jpg --output test_swap.mp4 --provider cuda
```

Expected output:
- Model download on first run (~500MB)
- Progress bar showing frame processing
- Output video with swapped face
- Processing time: ~1-2 min for 30 sec 720p video on RTX 3060

---

### Test 4: Full Pipeline Test

```bash
python cli.py all --input test_video.mp4 --source-face source_face.jpg --output-dir ./test_output
```

---

### Test 5: CPU Fallback Test

```bash
python cli.py swap --input test_video.mp4 --source-face source_face.jpg --output test_cpu.mp4 --provider cpu
```

**Warning:** CPU mode is 10-50x slower than GPU.

---

## Troubleshooting

### Error: "CUDA not available"

1. Check NVIDIA driver: `nvidia-smi`
2. Reinstall PyTorch with CUDA:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

### Error: "No face detected in source image"

- Use a clear, frontal face photo
- Ensure face is well-lit and not occluded
- Try a higher resolution image (512x512+)

### Error: "Could not open video file"

- Check file path and extension
- Ensure video is not corrupted: `ffprobe video.mp4`
- Supported formats: MP4, AVI, MOV, MKV

### Error: "FFmpeg not found"

- Install FFmpeg (see Installation section)
- Add to PATH and restart terminal

### GFPGAN creates "plastic" faces

- Reduce enhancement by not using `--enable-enhancer`
- Or wait for future weight parameter tuning

### Out of Memory (OOM)

- Use lower resolution input video
- Use `--provider cpu` (slower but uses less VRAM)
- Close other GPU applications

---

## Performance Benchmarks

| Video | Resolution | Duration | GPU | Time (swap) | Time (landmarks) |
|-------|------------|----------|-----|-------------|------------------|
| Test  | 720p       | 30 sec   | RTX 3060 | ~45 sec | ~15 sec |
| Test  | 1080p      | 60 sec   | RTX 3060 | ~3 min | ~45 sec |
| Test  | 720p       | 30 sec   | CPU | ~10 min | ~2 min |

---

## License

MIT License - See LICENSE file.
