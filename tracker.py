"""
Multi-object tracking module.

Uses Ultralytics' built-in ByteTrack / BoT-SORT tracker for robust
multi-object tracking with persistent ID assignment. The tracker handles:
- Motion prediction via Kalman filtering
- Hungarian algorithm-based association
- Low-confidence detection recovery (ByteTrack's key innovation)
- Track lifecycle management (creation, update, deletion)
"""

from dataclasses import dataclass, field

import numpy as np
from ultralytics import YOLO

from config import TrackerConfig, DetectorConfig


@dataclass
class Track:
    """A tracked object with persistent identity."""

    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    frame_idx: int  # Frame where this observation was made
    history: list[tuple[int, np.ndarray]] = field(default_factory=list)

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )


class MultiObjectTracker:
    """
    Multi-object tracker using Ultralytics' integrated tracking.

    Leverages model.track() which internally uses ByteTrack or BoT-SORT
    for robust association and ID persistence. This approach is preferred
    over separate detect-then-track because:
    1. The tracker has direct access to the model's multi-scale features
    2. ByteTrack's two-stage association uses both high and low confidence
       detections, recovering objects that might be missed at high thresholds
    3. BoT-SORT adds camera-motion compensation and Re-ID features
    """

    def __init__(self, tracker_config: TrackerConfig, detector_config: DetectorConfig):
        self.tracker_config = tracker_config
        self.detector_config = detector_config
        self.model = YOLO(detector_config.model_name)

        # Track histories: {track_id: [(frame_idx, center_x, center_y, bbox)]}
        self.track_histories: dict[int, list[tuple[int, float, float, np.ndarray]]] = {}

        # Track first/last seen frames
        self.track_first_seen: dict[int, int] = {}
        self.track_last_seen: dict[int, int] = {}

        # Count of valid frames per track (for filtering short-lived tracks)
        self.track_frame_count: dict[int, int] = {}

        self._write_tracker_config()
        print(f"[Tracker] Initialized {tracker_config.tracker_type} tracker")

    def _write_tracker_config(self):
        """Write a custom tracker YAML config for ultralytics."""
        import yaml
        from pathlib import Path

        cfg = {
            "tracker_type": self.tracker_config.tracker_type,
            "track_high_thresh": self.tracker_config.track_high_thresh,
            "track_low_thresh": self.tracker_config.track_low_thresh,
            "new_track_thresh": self.tracker_config.new_track_thresh,
            "track_buffer": self.tracker_config.track_buffer,
            "match_thresh": self.tracker_config.match_thresh,
            "fuse_score": True,
        }

        if self.tracker_config.tracker_type == "botsort":
            cfg.update({
                "gmc_method": "sparseOptFlow",
                "proximity_thresh": 0.5,
                "appearance_thresh": 0.8,
                "with_reid": False,
                "model": "auto",
            })

        self._tracker_yaml = Path("output") / "tracker_config.yaml"
        self._tracker_yaml.parent.mkdir(parents=True, exist_ok=True)
        with open(self._tracker_yaml, "w") as f:
            yaml.dump(cfg, f)

    def update(self, frame: np.ndarray, frame_idx: int) -> list[Track]:
        """
        Run detection + tracking on a single frame.

        Args:
            frame: BGR image (H, W, 3).
            frame_idx: Current frame index in the video.

        Returns:
            List of Track objects with persistent IDs.
        """
        results = self.model.track(
            frame,
            persist=True,
            conf=self.detector_config.confidence_threshold,
            iou=self.detector_config.iou_threshold,
            imgsz=self.detector_config.imgsz,
            device=self.detector_config.device,
            tracker=str(self._tracker_yaml),
            verbose=False,
            classes=self.detector_config.target_classes,
        )

        tracks = []
        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                track_id = int(boxes.id[i].cpu().numpy())
                bbox = boxes.xyxy[i].cpu().numpy().astype(np.float32)
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())

                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2

                # Update track history
                if track_id not in self.track_histories:
                    self.track_histories[track_id] = []
                    self.track_first_seen[track_id] = frame_idx
                    self.track_frame_count[track_id] = 0

                self.track_histories[track_id].append(
                    (frame_idx, float(cx), float(cy), bbox.copy())
                )
                self.track_last_seen[track_id] = frame_idx
                self.track_frame_count[track_id] += 1

                track = Track(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=conf,
                    class_id=cls_id,
                    frame_idx=frame_idx,
                )
                tracks.append(track)

        return tracks

    def get_trail(self, track_id: int, max_length: int = 50) -> list[tuple[float, float]]:
        """Get the recent trajectory trail for a track as (x, y) points."""
        if track_id not in self.track_histories:
            return []
        history = self.track_histories[track_id][-max_length:]
        return [(h[1], h[2]) for h in history]

    def get_all_histories(self) -> dict[int, list[tuple[int, float, float, np.ndarray]]]:
        """Get the full history of all tracks."""
        return self.track_histories

    def get_valid_track_ids(self) -> list[int]:
        """Return track IDs that have been seen for enough frames."""
        min_len = self.tracker_config.min_track_length
        return [
            tid for tid, count in self.track_frame_count.items()
            if count >= min_len
        ]

    def reset(self):
        """Reset tracker state for a new video."""
        self.model = YOLO(self.detector_config.model_name)
        self.track_histories.clear()
        self.track_first_seen.clear()
        self.track_last_seen.clear()
        self.track_frame_count.clear()
