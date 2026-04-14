"""
Main pipeline module that orchestrates detection, tracking,
visualization, and analytics.

The pipeline reads video frames, runs multi-object tracking,
produces an annotated output video, captures screenshots, and
optionally generates analytics.
"""

import time
from pathlib import Path

import cv2
import numpy as np

from config import PipelineConfig
from tracker import MultiObjectTracker
from visualizer import Visualizer
from analytics import Analytics


class Pipeline:
    """End-to-end multi-object detection and tracking pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.tracker = MultiObjectTracker(config.tracker, config.detector)
        self.visualizer = Visualizer(config.visualizer)
        self.analytics = Analytics(config.analytics)

        # Per-frame object counts for analytics
        self.frame_counts: dict[int, int] = {}

    def run(self) -> str:
        """
        Execute the full pipeline on the configured input video.

        Returns:
            Path to the output annotated video.
        """
        input_path = self.config.input_video
        if not Path(input_path).exists():
            raise FileNotFoundError(f"Input video not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        # Video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_fps = self.config.visualizer.output_fps or src_fps
        print(f"[Pipeline] Input: {input_path}")
        print(f"[Pipeline] Resolution: {width}x{height}, FPS: {src_fps:.1f}, Frames: {total_frames}")

        # Setup output video writer
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "annotated_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*self.config.visualizer.codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, out_fps, (width, height))

        # Screenshot capture: save frames at 25%, 50%, 75% progress
        screenshot_dir = output_dir / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        screenshot_frames = set()
        if total_frames > 0:
            for pct in [0.1, 0.25, 0.5, 0.75, 0.9]:
                screenshot_frames.add(int(total_frames * pct))

        frame_idx = 0
        processed = 0
        t_start = time.time()
        fps_avg = 0.0

        max_frames = self.config.max_frames or float("inf")

        print(f"[Pipeline] Processing started (frame_skip={self.config.frame_skip})...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx >= max_frames:
                break

            # Frame skipping
            if frame_idx % self.config.frame_skip != 0:
                frame_idx += 1
                continue

            t_frame = time.time()

            # Run tracking
            tracks = self.tracker.update(frame, frame_idx)
            self.frame_counts[frame_idx] = len(tracks)

            # Build trails for visualization
            trails = {}
            if self.config.visualizer.draw_trails:
                for track in tracks:
                    trail = self.tracker.get_trail(
                        track.track_id,
                        max_length=self.config.visualizer.max_trail_length,
                    )
                    if trail:
                        trails[track.track_id] = trail

            # Draw annotations
            annotated = self.visualizer.draw_tracks(frame, tracks, trails)

            # Frame info overlay
            dt = time.time() - t_frame
            fps_inst = 1.0 / dt if dt > 0 else 0
            fps_avg = 0.9 * fps_avg + 0.1 * fps_inst if fps_avg > 0 else fps_inst
            annotated = self.visualizer.draw_frame_info(
                annotated, frame_idx, len(tracks), fps_avg
            )

            # Write annotated frame
            writer.write(annotated)

            # Save screenshots at milestone frames
            if frame_idx in screenshot_frames:
                ss_path = screenshot_dir / f"frame_{frame_idx:06d}.png"
                cv2.imwrite(str(ss_path), annotated)
                print(f"[Pipeline] Screenshot saved: {ss_path}")

            processed += 1
            if processed % 100 == 0:
                elapsed = time.time() - t_start
                overall_fps = processed / elapsed if elapsed > 0 else 0
                pct = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                print(
                    f"[Pipeline] Frame {frame_idx}/{total_frames} "
                    f"({pct:.1f}%) | {overall_fps:.1f} FPS | "
                    f"Tracks: {len(tracks)}"
                )

            frame_idx += 1

        cap.release()
        writer.release()

        elapsed = time.time() - t_start
        print(f"[Pipeline] Done! Processed {processed} frames in {elapsed:.1f}s")
        print(f"[Pipeline] Output video: {output_path}")

        # Print tracking summary
        valid_ids = self.tracker.get_valid_track_ids()
        all_ids = list(self.tracker.track_histories.keys())
        print(f"[Pipeline] Total unique IDs assigned: {len(all_ids)}")
        print(f"[Pipeline] Valid tracks (>= {self.config.tracker.min_track_length} frames): {len(valid_ids)}")

        # Generate analytics
        self._generate_analytics((width, height), processed)

        return str(output_path)

    def _generate_analytics(self, frame_size: tuple[int, int], total_frames: int):
        """Generate all analytics outputs."""
        histories = self.tracker.get_all_histories()
        # Filter to valid tracks only for cleaner analytics
        valid_ids = set(self.tracker.get_valid_track_ids())
        filtered = {k: v for k, v in histories.items() if k in valid_ids}

        self.analytics.generate_all(
            track_histories=filtered,
            frame_size=frame_size,
            output_dir=self.config.output_dir,
            frame_counts=self.frame_counts,
            total_frames=total_frames,
        )
