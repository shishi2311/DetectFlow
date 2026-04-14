"""
Analytics module for generating trajectory plots, heatmaps,
and object count over time graphs.

These are optional enhancements that provide deeper insight
into the tracked objects' behavior.
"""

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from config import AnalyticsConfig

# Re-use the same color palette as the visualizer
_PALETTE_NORM = [
    (0.90, 0.10, 0.29), (0.24, 0.71, 0.29), (1.00, 0.88, 0.10),
    (0.00, 0.51, 0.78), (0.96, 0.51, 0.19), (0.57, 0.12, 0.71),
    (0.27, 0.94, 0.94), (0.94, 0.20, 0.90), (0.82, 0.96, 0.24),
    (0.98, 0.75, 0.83), (0.00, 0.50, 0.50), (0.86, 0.75, 1.00),
    (0.67, 0.43, 0.16), (1.00, 0.98, 0.78), (0.50, 0.00, 0.00),
    (0.67, 1.00, 0.76), (0.50, 0.50, 0.00), (1.00, 0.84, 0.71),
]


class Analytics:
    """Generates post-processing analytics from tracking data."""

    def __init__(self, config: AnalyticsConfig):
        self.config = config

    def generate_all(
        self,
        track_histories: dict[int, list[tuple[int, float, float, np.ndarray]]],
        frame_size: tuple[int, int],
        output_dir: str,
        frame_counts: dict[int, int] | None = None,
        total_frames: int = 0,
    ):
        """
        Generate all configured analytics outputs.

        Args:
            track_histories: {track_id: [(frame_idx, cx, cy, bbox), ...]}
            frame_size: (width, height) of the video frames.
            output_dir: Directory to save outputs.
            frame_counts: {frame_idx: num_tracks} for count-over-time plot.
            total_frames: Total number of frames processed.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if self.config.generate_trajectories:
            self._plot_trajectories(track_histories, frame_size, out)

        if self.config.generate_heatmap:
            self._plot_heatmap(track_histories, frame_size, out)

        if self.config.generate_count_plot and frame_counts:
            self._plot_count_over_time(frame_counts, total_frames, out)

        print(f"[Analytics] All outputs saved to {out}")

    def _plot_trajectories(
        self,
        track_histories: dict[int, list[tuple[int, float, float, np.ndarray]]],
        frame_size: tuple[int, int],
        output_dir: Path,
    ):
        """Plot all object trajectories on a single figure."""
        w, h = frame_size
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)  # Invert y-axis to match image coordinates
        ax.set_aspect("equal")
        ax.set_title("Object Trajectories", fontsize=14)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

        for track_id, history in track_histories.items():
            if len(history) < 3:
                continue
            color = _PALETTE_NORM[track_id % len(_PALETTE_NORM)]
            xs = [h[1] for h in history]
            ys = [h[2] for h in history]
            ax.plot(xs, ys, color=color, linewidth=1.0, alpha=0.7, label=f"ID {track_id}")
            # Mark start and end
            ax.scatter(xs[0], ys[0], color=color, marker="o", s=30, zorder=5)
            ax.scatter(xs[-1], ys[-1], color=color, marker="x", s=30, zorder=5)

        # Only show legend if not too many tracks
        if len(track_histories) <= 20:
            ax.legend(fontsize=7, loc="upper right", ncol=2)

        plt.tight_layout()
        path = output_dir / "trajectories.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[Analytics] Trajectory plot saved: {path}")

    def _plot_heatmap(
        self,
        track_histories: dict[int, list[tuple[int, float, float, np.ndarray]]],
        frame_size: tuple[int, int],
        output_dir: Path,
    ):
        """Generate a spatial heatmap of object positions."""
        w, h = frame_size
        all_x = []
        all_y = []
        for history in track_histories.values():
            for entry in history:
                all_x.append(entry[1])
                all_y.append(entry[2])

        if not all_x:
            return

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        bins = self.config.heatmap_bins
        heatmap, xedges, yedges = np.histogram2d(
            all_x, all_y, bins=bins, range=[[0, w], [0, h]]
        )

        # Apply Gaussian smoothing for smoother visualization
        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(heatmap.T, sigma=2)

        ax.imshow(
            heatmap,
            extent=[0, w, h, 0],
            cmap=self.config.heatmap_colormap,
            aspect="equal",
            interpolation="bilinear",
        )
        ax.set_title("Position Heatmap", fontsize=14)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        plt.colorbar(ax.images[0], ax=ax, label="Density")

        plt.tight_layout()
        path = output_dir / "heatmap.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[Analytics] Heatmap saved: {path}")

    def _plot_count_over_time(
        self,
        frame_counts: dict[int, int],
        total_frames: int,
        output_dir: Path,
    ):
        """Plot the number of tracked objects over time."""
        frames = sorted(frame_counts.keys())
        counts = [frame_counts[f] for f in frames]

        fig, ax = plt.subplots(1, 1, figsize=(12, 4))
        ax.fill_between(frames, counts, alpha=0.3, color="steelblue")
        ax.plot(frames, counts, color="steelblue", linewidth=1.0)
        ax.set_title("Number of Tracked Objects Over Time", fontsize=14)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Active Tracks")
        ax.set_xlim(0, max(frames) if frames else total_frames)
        ax.set_ylim(0, max(counts) + 2 if counts else 10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = output_dir / "count_over_time.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[Analytics] Count plot saved: {path}")
