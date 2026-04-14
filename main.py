#!/usr/bin/env python3
"""
DetectFlow - Multi-Object Detection and Persistent ID Tracking

Entry point for the detection and tracking pipeline.
Supports both command-line arguments and programmatic usage.

Usage:
    python main.py --input video.mp4
    python main.py --input video.mp4 --output results/ --model yolov8m.pt
    python main.py --input video.mp4 --tracker botsort --max-frames 500
"""

import argparse
import sys

from config import (
    PipelineConfig,
    DetectorConfig,
    TrackerConfig,
    VisualizerConfig,
    AnalyticsConfig,
)
from pipeline import Pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DetectFlow: Multi-Object Detection & Tracking Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the input video file.",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
        help="Output directory for annotated video and analytics.",
    )

    # Detection
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8m.pt",
        help="YOLOv8 model variant (e.g., yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Minimum detection confidence threshold.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Input image size for the detector.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[0],
        help="COCO class IDs to detect (0=person). Use -1 for all classes.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference (cpu, cuda, 0, 1, etc.).",
    )

    # Tracking
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack",
        choices=["bytetrack", "botsort"],
        help="Tracking algorithm.",
    )
    parser.add_argument(
        "--track-buffer",
        type=int,
        default=30,
        help="Number of frames to keep lost tracks before deletion.",
    )

    # Processing
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame (1 = every frame).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process.",
    )

    # Visualization
    parser.add_argument(
        "--no-trails",
        action="store_true",
        help="Disable trajectory trails in output video.",
    )
    parser.add_argument(
        "--trail-length",
        type=int,
        default=50,
        help="Maximum number of trail points per track.",
    )

    # Analytics
    parser.add_argument(
        "--no-analytics",
        action="store_true",
        help="Disable analytics generation (trajectories, heatmap, counts).",
    )

    # Video source
    parser.add_argument(
        "--source-url",
        type=str,
        default="",
        help="URL of the original public video source.",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Build a PipelineConfig from parsed command-line arguments."""
    target_classes = args.classes if args.classes != [-1] else None

    detector = DetectorConfig(
        model_name=args.model,
        target_classes=target_classes,
        confidence_threshold=args.confidence,
        imgsz=args.imgsz,
        device=args.device,
    )

    tracker = TrackerConfig(
        tracker_type=args.tracker,
        track_buffer=args.track_buffer,
    )

    visualizer = VisualizerConfig(
        draw_trails=not args.no_trails,
        max_trail_length=args.trail_length,
    )

    analytics = AnalyticsConfig(
        generate_trajectories=not args.no_analytics,
        generate_heatmap=not args.no_analytics,
        generate_count_plot=not args.no_analytics,
    )

    return PipelineConfig(
        input_video=args.input,
        output_dir=args.output,
        frame_skip=args.frame_skip,
        max_frames=args.max_frames,
        detector=detector,
        tracker=tracker,
        visualizer=visualizer,
        analytics=analytics,
        video_source_url=args.source_url,
    )


def main():
    args = parse_args()
    config = build_config(args)

    print("=" * 60)
    print("  DetectFlow - Multi-Object Detection & Tracking")
    print("=" * 60)
    print(f"  Input:    {config.input_video}")
    print(f"  Output:   {config.output_dir}")
    print(f"  Model:    {config.detector.model_name}")
    print(f"  Tracker:  {config.tracker.tracker_type}")
    print(f"  Device:   {config.detector.device}")
    print(f"  Classes:  {config.detector.target_classes or 'all'}")
    print("=" * 60)

    pipeline = Pipeline(config)
    output_path = pipeline.run()

    print("\n" + "=" * 60)
    print(f"  Output video: {output_path}")
    print(f"  Analytics:    {config.output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
