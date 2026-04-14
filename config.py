"""
Configuration parameters for the DetectFlow pipeline.

Centralizes all tunable parameters for detection, tracking, visualization,
and analytics so they can be adjusted without modifying module code.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DetectorConfig:
    """Configuration for the YOLOv8 object detector."""

    # Model variant: "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"
    model_name: str = "yolov8m.pt"

    # COCO class IDs to detect (0 = person). Set to None for all classes.
    target_classes: Optional[list[int]] = field(default_factory=lambda: [0])

    # Minimum confidence threshold for detections
    confidence_threshold: float = 0.3

    # IoU threshold for NMS
    iou_threshold: float = 0.5

    # Input image size for the model
    imgsz: int = 1280

    # Device: "cpu", "cuda", "0", "1", etc.
    device: str = "cpu"


@dataclass
class TrackerConfig:
    """Configuration for the multi-object tracker."""

    # Tracker type: "bytetrack" or "botsort"
    tracker_type: str = "bytetrack"

    # Maximum number of frames a track can be lost before removal
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8

    # Minimum track length to consider valid (filters spurious detections)
    min_track_length: int = 3


@dataclass
class VisualizerConfig:
    """Configuration for the output video visualization."""

    # Bounding box line thickness
    bbox_thickness: int = 2

    # Font scale for ID labels
    font_scale: float = 0.7

    # Font thickness
    font_thickness: int = 2

    # Whether to draw trajectory trails
    draw_trails: bool = True

    # Maximum number of trail points to draw per track
    max_trail_length: int = 50

    # Trail line thickness
    trail_thickness: int = 2

    # Output video codec
    codec: str = "mp4v"

    # Output video FPS (None = match source)
    output_fps: Optional[float] = None


@dataclass
class AnalyticsConfig:
    """Configuration for optional analytics."""

    # Generate trajectory plot
    generate_trajectories: bool = True

    # Generate heatmap
    generate_heatmap: bool = True

    # Generate object count over time plot
    generate_count_plot: bool = True

    # Heatmap resolution (bins)
    heatmap_bins: int = 100

    # Heatmap colormap
    heatmap_colormap: str = "hot"


@dataclass
class PipelineConfig:
    """Top-level configuration for the entire pipeline."""

    # Input video path
    input_video: str = ""

    # Output directory
    output_dir: str = "output"

    # Process every Nth frame (1 = every frame)
    frame_skip: int = 1

    # Maximum frames to process (None = all)
    max_frames: Optional[int] = None

    # Sub-configs
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    visualizer: VisualizerConfig = field(default_factory=VisualizerConfig)
    analytics: AnalyticsConfig = field(default_factory=AnalyticsConfig)

    # Source URL of the video
    video_source_url: str = ""

    def __post_init__(self):
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
