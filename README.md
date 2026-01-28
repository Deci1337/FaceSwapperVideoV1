# Offline Video Face Swap

CLI tool for offline video face swapping with landmarks visualization.

## Core Features

1. **Face Swap** - Replace faces in video using a source face image
2. **Landmarks Debug** - Visualize hand skeleton (21 points) and face mesh (468 points)
3. **Audio Preservation** - Keep original video audio in output
4. **Color Correction** - Match lighting between source face and target video
5. **Face Enhancement** - Optional GFPGAN for improved face quality

---

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1060 6GB | RTX 3060+ 12GB |
| RAM | 8 GB | 16 GB |
| Python | 3.9-3.11 | 3.10 |
| FFmpeg | Required | Required |

---

## Installation

```bash
# 1. Create virtual environment
cd offline-faceswap
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell

# 2. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify
python cli.py --help
```

---

## Usage

### 1. Face Swap

Replace faces in video:

```bash
python cli.py swap --input video.mp4 --source-face face.jpg --output result.mp4
```

**Key Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--quality` | low/medium/high | high |
| `--enable-enhancer` | Use GFPGAN | False |
| `--enhancer-weight` | GFPGAN strength (0.0-1.0) | 0.7 |
| `--color-correction` | Lighting match (0.0-1.0) | 0.5 |
| `--keep-audio` | Preserve original audio | True |
| `--provider` | cuda/dml/cpu | cuda |

**Example with enhancement:**
```bash
python cli.py swap --input video.mp4 --source-face face.jpg --output result.mp4 \
  --enable-enhancer --enhancer-weight 0.6 --color-correction 0.5
```

### 2. Landmarks Debug

Visualize face mesh and hand skeleton:

```bash
python cli.py landmarks --input video.mp4 --output debug.mp4
```

**Options:**
| Option | Description | Default |
|--------|-------------|---------|
| `--face-mode` | mesh/bbox | mesh |
| `--fps` | Output FPS | same as input |

### 3. Run Both

```bash
python cli.py all --input video.mp4 --source-face face.jpg --output-dir ./output
```

Creates:
- `output/debug_landmarks.mp4`
- `output/result_faceswap.mp4`

---

## Project Structure

```
offline-faceswap/
  cli.py                # Main CLI entry point
  requirements.txt      # Python dependencies
  README.md             # This file
  INSTALL_RU.md         # Russian installation guide
  CONTRIBUTORS.md       # Credits and licenses
  
  app/
    config.py           # Configuration classes
    io/                 # Video/audio I/O
      video.py
      audio.py
    models/             # ML models
      swapper.py        # Face swap model
      enhancer.py       # GFPGAN enhancement
      background.py     # Background removal
      parsing.py        # Face parsing
      loader.py         # Model loading
    pipelines/          # Processing pipelines
      faceswap_pipeline.py
      landmarks_pipeline.py
      orchestrator.py
    utils/              # Utilities
      visualization.py
      color_transfer.py
    hair_transfer/      # Optional hair transfer module
  
  scripts/              # Test scripts
    test_faceswap.py
  
  gfpgan/weights/       # Model weights (auto-downloaded)
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| CUDA not available | Reinstall PyTorch: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` |
| No face detected | Use clear frontal face photo (512x512+) |
| FFmpeg not found | Install FFmpeg and add to PATH |
| Out of Memory | Use lower resolution or `--provider cpu` |

---

## Performance

| Video | Resolution | GPU | Face Swap | Landmarks |
|-------|------------|-----|-----------|-----------|
| 30s | 720p | RTX 3060 | ~45s | ~15s |
| 60s | 1080p | RTX 3060 | ~3min | ~45s |

---

## License

MIT License. See [CONTRIBUTORS.md](CONTRIBUTORS.md) for third-party licenses.
