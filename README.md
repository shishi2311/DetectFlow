# DetectFlow — Multi-Object Detection & Persistent ID Tracking

A modular computer vision pipeline for detecting and tracking multiple objects (primarily people/players) in sports and event footage with persistent unique ID assignment.

## Features

- **YOLOv8 Object Detection** — State-of-the-art real-time detection using Ultralytics YOLOv8
- **ByteTrack / BoT-SORT Tracking** — Robust multi-object tracking with persistent ID assignment
- **Annotated Output Video** — Bounding boxes, unique IDs, confidence scores, and trajectory trails
- **Analytics** — Trajectory plots, position heatmaps, object count over time
- **Handles Real-World Challenges** — Occlusion, motion blur, camera motion, similar-looking subjects

## Project Structure

```
DetectFlow/
├── main.py                  # Entry point with CLI argument parsing
├── config.py                # All configuration dataclasses
├── detector.py              # YOLOv8 detection module
├── tracker.py               # Multi-object tracking with ID persistence
├── pipeline.py              # Main pipeline orchestrating all modules
├── visualizer.py            # Drawing bounding boxes, IDs, trails
├── analytics.py             # Trajectory plots, heatmaps, count graphs
├── download_and_run.py      # Helper: download video + run pipeline
├── setup.sh                 # Virtual environment setup script
├── requirements.txt         # Python dependencies
├── TECHNICAL_REPORT.md      # Detailed technical report
├── input_video.mp4          # Original source video
├── output/                  # Generated output files
│   ├── annotated_output.mp4 # Annotated output video with tracking
│   ├── screenshots/         # Sample screenshots at various frames
│   ├── trajectories.png     # Object trajectory plot
│   ├── heatmap.png          # Position heatmap
│   └── count_over_time.png  # Object count over time
└── README.md
```

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup (Recommended — Virtual Environment)

```bash
# Clone the repository
git clone https://github.com/shishi2311/DetectFlow.git
cd DetectFlow

# Option A: One-command setup (creates venv + installs everything)
chmod +x setup.sh
./setup.sh
source venv/bin/activate

# Option B: Manual setup
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
pip install yt-dlp              # for video downloading
```

To deactivate the virtual environment when done:
```bash
deactivate
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `ultralytics` | YOLOv8 model and built-in tracker (ByteTrack/BoT-SORT) |
| `opencv-python-headless` | Video I/O and image processing |
| `numpy` | Numerical operations |
| `scipy` | Gaussian smoothing for heatmaps |
| `lapx` | Linear assignment (Hungarian algorithm) for tracking |
| `matplotlib` | Analytics visualization |

## How to Run

### One-Command Download & Run

The easiest way to get started — downloads a public video and runs the full pipeline:

```bash
# Install yt-dlp (needed once)
pip install yt-dlp

# Download a video and run the pipeline in one command
python download_and_run.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"

# With extra options passed through to the pipeline
python download_and_run.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --max-frames 500 --device cuda

# Skip re-downloading if video already exists
python download_and_run.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --skip-download
```

### Basic Usage

```bash
# Run with default settings (YOLOv8m, ByteTrack, person detection)
python main.py --input path/to/video.mp4

# Specify output directory
python main.py --input video.mp4 --output results/
```

### Advanced Options

```bash
# Use a larger model for better accuracy
python main.py --input video.mp4 --model yolov8x.pt

# Use GPU acceleration
python main.py --input video.mp4 --device cuda

# Use BoT-SORT tracker instead of ByteTrack
python main.py --input video.mp4 --tracker botsort

# Detect all COCO classes (not just people)
python main.py --input video.mp4 --classes -1

# Process every 2nd frame for faster processing
python main.py --input video.mp4 --frame-skip 2

# Limit to first 500 frames
python main.py --input video.mp4 --max-frames 500

# Disable trajectory trails
python main.py --input video.mp4 --no-trails

# Disable analytics generation
python main.py --input video.mp4 --no-analytics

# Full example with all options
python main.py \
    --input match.mp4 \
    --output results/ \
    --model yolov8m.pt \
    --tracker bytetrack \
    --confidence 0.3 \
    --imgsz 1280 \
    --device cpu \
    --trail-length 50 \
    --source-url "https://example.com/video"
```

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--input, -i` | (required) | Path to input video |
| `--output, -o` | `output` | Output directory |
| `--model` | `yolov8m.pt` | YOLOv8 model variant |
| `--confidence` | `0.3` | Detection confidence threshold |
| `--imgsz` | `1280` | Input image size |
| `--classes` | `[0]` | COCO class IDs (`-1` for all) |
| `--device` | `cpu` | Inference device |
| `--tracker` | `bytetrack` | Tracker type (`bytetrack` / `botsort`) |
| `--track-buffer` | `30` | Frames to keep lost tracks |
| `--frame-skip` | `1` | Process every Nth frame |
| `--max-frames` | `None` | Max frames to process |
| `--no-trails` | `False` | Disable trajectory trails |
| `--trail-length` | `50` | Trail points per track |
| `--no-analytics` | `False` | Disable analytics plots |
| `--source-url` | `""` | Original video URL |

## Pipeline Architecture

```
Input Video
    │
    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  YOLOv8      │───▶│  ByteTrack/  │───▶│  Visualizer  │──▶ Annotated Video
│  Detector    │    │  BoT-SORT    │    │  (Draw)      │
└──────────────┘    └──────────────┘    └──────────────┘
                          │
                          ▼
                    ┌──────────────┐
                    │  Analytics   │──▶ Trajectory Plot
                    │  Module      │──▶ Heatmap
                    │              │──▶ Count Graph
                    └──────────────┘
```

1. **Detection**: YOLOv8 processes each frame to detect objects (filtered by class, e.g., person)
2. **Tracking**: ByteTrack associates detections across frames using IoU-based matching with Kalman filter motion prediction, assigning persistent unique IDs
3. **Visualization**: Bounding boxes, IDs, confidence scores, and trajectory trails are drawn on each frame
4. **Analytics**: Post-processing generates trajectory plots, position heatmaps, and object count over time

## Assumptions

- The primary detection target is people/players (COCO class 0)
- Input video is in a standard format readable by OpenCV (MP4, AVI, MOV, etc.)
- The YOLOv8 model weights are automatically downloaded on first run
- CPU inference is used by default; GPU requires CUDA-compatible hardware

## Limitations

- **ID Switches**: Long occlusions or very similar appearances may cause ID reassignment
- **Crowded Scenes**: Very dense crowds can lead to merged/missed detections
- **Small Objects**: Very distant or small subjects may fall below the detection threshold
- **Camera Cuts**: Abrupt scene changes will break track continuity
- **Processing Speed**: CPU inference is slow (~2-5 FPS); GPU recommended for real-time

## Model & Tracker Choices

See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for detailed rationale on model and tracker selection.

## Video Source

**Original Video**: [Football/Soccer Match — YouTube](https://youtu.be/KgQquW68E-Q)

The video is publicly available football match footage. It was downloaded at 480p resolution for processing.

## Results Summary

| Metric | Value |
|--------|-------|
| Video resolution | 640x360 |
| Frames processed | 500 (first 20 seconds) |
| Processing time | ~435 seconds (CPU) |
| Processing speed | ~1.1 FPS |
| Unique IDs assigned | 29 |
| Valid tracks (>= 3 frames) | 29 |
| Players tracked per frame | 8–12 |

### Sample Output

The `output/` directory contains:
- **`annotated_output.mp4`** — Full annotated video with bounding boxes, unique IDs, and trajectory trails
- **`screenshots/`** — Sample frames captured at 10%, 25%, 50%, 75%, and 90% progress
- **`trajectories.png`** — All player movement paths plotted on a single figure
- **`heatmap.png`** — Spatial density map showing where players spent the most time
- **`count_over_time.png`** — Number of actively tracked players per frame

## Live Demo (Gradio Web App)

Run the pipeline as an interactive web app:

```bash
pip install gradio
python app.py
```

Opens at `http://localhost:7860`. Upload a video, adjust settings, and get annotated output + analytics.

**Deploy to Hugging Face Spaces (free public hosting):**

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Gradio** as the SDK
3. Push the repo to the Space:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/DetectFlow
   git push hf main
   ```
4. The app will be live at `https://huggingface.co/spaces/YOUR_USERNAME/DetectFlow`
