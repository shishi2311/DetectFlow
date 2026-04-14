"""
Visualization module for drawing detections, tracks, and annotations.

Draws bounding boxes, unique IDs, confidence scores, and trajectory
trails on video frames for the annotated output.
"""

import cv2
import numpy as np

from config import VisualizerConfig
from tracker import Track

# Distinct colors for up to 80 tracked objects (BGR format).
# Uses a perceptually distinct palette to differentiate nearby IDs.
_PALETTE = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128),
    (60, 20, 220), (80, 127, 255), (34, 139, 34), (0, 215, 255),
    (192, 192, 192), (255, 99, 71), (46, 139, 87), (238, 130, 238),
    (85, 107, 47), (255, 165, 0), (72, 61, 139), (0, 250, 154),
]


def _get_color(track_id: int) -> tuple[int, int, int]:
    """Get a consistent color for a given track ID."""
    return _PALETTE[track_id % len(_PALETTE)]


class Visualizer:
    """Draws tracking annotations on video frames."""

    def __init__(self, config: VisualizerConfig):
        self.config = config

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        trails: dict[int, list[tuple[float, float]]] | None = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes, IDs, and optional trails on a frame.

        Args:
            frame: BGR image to annotate (modified in-place and returned).
            tracks: Current frame's tracked objects.
            trails: Optional dict mapping track_id -> list of (x, y) trail points.

        Returns:
            Annotated frame.
        """
        annotated = frame.copy()

        # Draw trails first (under bounding boxes)
        if self.config.draw_trails and trails:
            for track_id, points in trails.items():
                if len(points) < 2:
                    continue
                color = _get_color(track_id)
                for j in range(1, len(points)):
                    # Fade older points
                    alpha = j / len(points)
                    thickness = max(1, int(self.config.trail_thickness * alpha))
                    pt1 = (int(points[j - 1][0]), int(points[j - 1][1]))
                    pt2 = (int(points[j][0]), int(points[j][1]))
                    cv2.line(annotated, pt1, pt2, color, thickness)

        # Draw bounding boxes and labels
        for track in tracks:
            color = _get_color(track.track_id)
            x1, y1, x2, y2 = track.bbox.astype(int)

            # Bounding box
            cv2.rectangle(
                annotated, (x1, y1), (x2, y2), color, self.config.bbox_thickness
            )

            # Label with ID and confidence
            label = f"ID:{track.track_id} {track.confidence:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), baseline = cv2.getTextSize(
                label, font, self.config.font_scale, self.config.font_thickness
            )

            # Label background
            label_y = max(y1 - 6, th + 6)
            cv2.rectangle(
                annotated,
                (x1, label_y - th - 6),
                (x1 + tw + 4, label_y + baseline - 4),
                color,
                -1,
            )

            # Label text (white on colored background)
            cv2.putText(
                annotated,
                label,
                (x1 + 2, label_y - 4),
                font,
                self.config.font_scale,
                (255, 255, 255),
                self.config.font_thickness,
            )

        return annotated

    def draw_frame_info(
        self,
        frame: np.ndarray,
        frame_idx: int,
        num_tracks: int,
        fps: float = 0.0,
    ) -> np.ndarray:
        """Draw frame number, active track count, and processing FPS."""
        h, w = frame.shape[:2]
        info_lines = [
            f"Frame: {frame_idx}",
            f"Tracked: {num_tracks}",
        ]
        if fps > 0:
            info_lines.append(f"FPS: {fps:.1f}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30
        for line in info_lines:
            (tw, th), _ = cv2.getTextSize(line, font, 0.6, 2)
            cv2.rectangle(frame, (w - tw - 15, y_offset - th - 5), (w - 5, y_offset + 5), (0, 0, 0), -1)
            cv2.putText(frame, line, (w - tw - 10, y_offset), font, 0.6, (255, 255, 255), 2)
            y_offset += th + 15

        return frame
