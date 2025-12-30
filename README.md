<<<<<<< HEAD
# Face Detection Pipeline

A complete face detection and tracking pipeline with re-identification capabilities using InsightFace and OpenCV.

## Features

- 🎥 **Video Capture**: Support for webcam, RTSP streams, and video files with auto-reconnect
- 👤 **Face Detection**: High-accuracy face detection using InsightFace
- 🎯 **Face Tracking**: IOU-based tracker to maintain temporal identity across frames
- 🔍 **Re-Identification**: Match faces across different tracks using embedding similarity
- ⚡ **GPU Support**: Optional GPU acceleration with CUDA
- 🌐 **API Server**: FastAPI-based REST API for integration

## Architecture

```
Video Source → VideoReader → FaceDetector → IouTracker → ReIdentifier → Display/API
```

**Pipeline Stages:**
1. **Video Capture**: Reads frames from various sources
2. **Face Detection**: Detects faces and extracts 512-dim embeddings
3. **Tracking**: Maintains track IDs across frames using IoU matching
4. **Re-Identification**: Assigns persistent person IDs using embedding similarity

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for acceleration

### Setup

1. Clone or download this repository

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

For GPU support:
```bash
pip install onnxruntime-gpu
```

4. Download InsightFace models (automatic on first run):
The models will be downloaded to `~/.insightface/models/` on first execution.

## Usage

### Basic Usage (Webcam)

```bash
python run.py
```

Or using the module:
```bash
python -m src.main
```

### Configuration

Copy the example environment file and customize:
```bash
cp .env.example .env
```

Edit `.env` to configure:
- Video source (webcam, RTSP, file)
- Detection parameters
- Tracking settings
- Re-identification thresholds

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_SOURCE` | `0` | Video source (0=webcam, path to file, RTSP URL) |
| `FRAME_W` | `1280` | Frame width for processing |
| `FRAME_H` | `720` | Frame height for processing |
| `USE_GPU` | `false` | Enable GPU acceleration |
| `DET_THRESH` | `0.6` | Face detection confidence threshold |
| `ENABLE_TRACKING` | `true` | Enable face tracking |
| `ENABLE_REID` | `true` | Enable re-identification |

See `.env.example` for all available options.

### Running with Different Sources

**Webcam:**
```bash
VIDEO_SOURCE=0 python run.py
```

**Video file:**
```bash
VIDEO_SOURCE=/path/to/video.mp4 python run.py
```

**RTSP stream:**
```bash
VIDEO_SOURCE=rtsp://username:password@192.168.1.100:554/stream python run.py
```

### API Server

Start the FastAPI server:
```bash
python -m uvicorn src.api.server:app --reload
```

Access the API documentation at: `http://localhost:8000/docs`

## Controls

- **Q**: Quit the application
- The display shows:
  - Green bounding boxes around detected faces
  - Detection confidence score
  - Track ID (T#)
  - Person ID (P#) for re-identification
  - FPS counter

## Project Structure

```
proyecto_visual/
├── src/
│   ├── api/
│   │   └── server.py          # FastAPI server
│   ├── video/
│   │   └── capture.py         # Video capture with reconnect
│   ├── vision/
│   │   ├── detect.py          # Face detection (InsightFace)
│   │   ├── embed.py           # Embedding utilities
│   │   ├── reid.py            # Re-identification
│   │   └── track.py           # IOU tracker
│   ├── config.py              # Configuration management
│   └── main.py                # Main pipeline
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project metadata
├── .env.example              # Example configuration
└── README.md                 # This file
```

## How It Works

### Face Detection
Uses InsightFace's `buffalo_l` model to detect faces and extract 512-dimensional embeddings. The detector can be configured for different input sizes and confidence thresholds.

### Tracking
A simple but effective IOU (Intersection over Union) tracker maintains track IDs across frames by matching bounding boxes. Tracks are dropped after `TRACKER_MAX_MISSED` frames without detection.

### Re-Identification
Maintains persistent person IDs by:
1. Comparing face embeddings using cosine similarity
2. Matching new tracks to known persons using a similarity threshold
3. Updating person centroids with exponential moving average
4. Creating new person IDs for unmatched faces

## Performance Tips

1. **GPU Acceleration**: Install `onnxruntime-gpu` and set `USE_GPU=true`
2. **Detection Size**: Smaller `DET_W` and `DET_H` = faster but less accurate
3. **Max Faces**: Set `MAX_FACES` to limit detections in crowded scenes
4. **Disable Features**: Turn off tracking or re-ID if not needed

## Troubleshooting

**Models not downloading:**
- Ensure internet connection
- Check `~/.insightface/models/` directory
- Try manual download from InsightFace repository

**Camera not opening:**
- Check `VIDEO_SOURCE` is correct
- Verify camera permissions
- Try different camera indices (0, 1, 2...)

**Low FPS:**
- Reduce `DET_W` and `DET_H`
- Enable GPU acceleration
- Set `MAX_FACES` to limit detections

## License

MIT License - See LICENSE file for details

## Credits

- [InsightFace](https://github.com/deepinsight/insightface) for face detection models
- [OpenCV](https://opencv.org/) for video processing
- [FastAPI](https://fastapi.tiangolo.com/) for API framework
=======
# Deteccion_Facial
>>>>>>> 18b1cbe79673e91ef442e571ed6ca7819ee46327
