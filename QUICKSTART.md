# Quick Start Guide

## Installation

1. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

2. **Dependencies are already installed!** ✅
   All required packages have been installed in the virtual environment.

## Usage

### Option 1: Run with Webcam (Default)
```bash
python run.py
```
Press **Q** to quit.

### Option 2: Run with Video File
```bash
VIDEO_SOURCE=/path/to/video.mp4 python run.py
```

### Option 3: Run with RTSP Stream
```bash
VIDEO_SOURCE=rtsp://user:pass@ip:port/stream python run.py
```

### Option 4: Start API Server
```bash
python -m uvicorn src.api.server:app --reload
```
Then visit: http://localhost:8000/docs

## Configuration

Copy and edit the environment file:
```bash
cp .env.example .env
nano .env  # or your preferred editor
```

Key settings:
- `VIDEO_SOURCE`: Video source (0=webcam, path, or RTSP URL)
- `USE_GPU`: Enable GPU acceleration (requires onnxruntime-gpu)
- `DET_THRESH`: Detection confidence threshold (0.0-1.0)
- `ENABLE_TRACKING`: Enable face tracking
- `ENABLE_REID`: Enable re-identification

## What You'll See

The display shows:
- 🟢 **Green boxes** around detected faces
- **0.95** - Detection confidence score
- **T1** - Track ID (temporal identity)
- **P1** - Person ID (persistent identity across tracks)
- **FPS** counter in top-left

## Testing

Run the demo to verify everything works:
```bash
python demo.py
```

## Need Help?

See the full [README.md](README.md) for detailed documentation.
