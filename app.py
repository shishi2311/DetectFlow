#!/usr/bin/env python3
"""
DetectFlow — Gradio Web App

Upload a video or provide a URL, and the pipeline will detect and track
all people with persistent IDs, then return the annotated video and analytics.

Usage:
    pip install gradio
    python app.py

Deploys locally at http://localhost:7860
Can be deployed to Hugging Face Spaces for free public hosting.
"""

import shutil
import tempfile
from pathlib import Path

import gradio as gr

from config import (
    PipelineConfig,
    DetectorConfig,
    TrackerConfig,
    VisualizerConfig,
    AnalyticsConfig,
)
from pipeline import Pipeline


def process_video(
    video_file,
    model_name: str,
    tracker_type: str,
    confidence: float,
    max_frames: int,
    draw_trails: bool,
):
    """Run the DetectFlow pipeline on an uploaded video."""
    if video_file is None:
        raise gr.Error("Please upload a video file.")

    # Create a temporary output directory
    tmp_dir = tempfile.mkdtemp(prefix="detectflow_")
    output_dir = Path(tmp_dir) / "output"
    output_dir.mkdir()

    detector = DetectorConfig(
        model_name=model_name,
        confidence_threshold=confidence,
        imgsz=1280,
        device="cpu",
    )

    tracker = TrackerConfig(
        tracker_type=tracker_type,
    )

    visualizer = VisualizerConfig(
        draw_trails=draw_trails,
        max_trail_length=50,
    )

    analytics = AnalyticsConfig(
        generate_trajectories=True,
        generate_heatmap=True,
        generate_count_plot=True,
    )

    config = PipelineConfig(
        input_video=video_file,
        output_dir=str(output_dir),
        max_frames=max_frames if max_frames > 0 else None,
        detector=detector,
        tracker=tracker,
        visualizer=visualizer,
        analytics=analytics,
    )

    pipe = Pipeline(config)
    output_path = pipe.run()

    # Collect outputs
    trajectories = str(output_dir / "trajectories.png")
    heatmap = str(output_dir / "heatmap.png")
    count_plot = str(output_dir / "count_over_time.png")

    # Build summary
    valid_tracks = len(pipe.tracker.get_valid_track_ids())
    total_ids = len(pipe.tracker.track_histories)
    summary = (
        f"**Processing Complete**\n\n"
        f"| Metric | Value |\n"
        f"|--------|-------|\n"
        f"| Unique IDs assigned | {total_ids} |\n"
        f"| Valid tracks (>= 3 frames) | {valid_tracks} |\n"
        f"| Frames processed | {len(pipe.frame_counts)} |\n"
    )

    traj_out = trajectories if Path(trajectories).exists() else None
    heat_out = heatmap if Path(heatmap).exists() else None
    count_out = count_plot if Path(count_plot).exists() else None

    return output_path, summary, traj_out, heat_out, count_out


# Build Gradio interface
with gr.Blocks(
    title="DetectFlow — Multi-Object Detection & Tracking",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        """
        # DetectFlow — Multi-Object Detection & Tracking
        Upload a sports or event video to detect and track all people with persistent unique IDs.
        The pipeline uses **YOLOv8** for detection and **ByteTrack/BoT-SORT** for tracking.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload Video")

            with gr.Accordion("Settings", open=False):
                model_name = gr.Dropdown(
                    choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
                    value="yolov8m.pt",
                    label="Model",
                    info="Larger models are more accurate but slower",
                )
                tracker_type = gr.Dropdown(
                    choices=["bytetrack", "botsort"],
                    value="bytetrack",
                    label="Tracker",
                )
                confidence = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.3, step=0.05,
                    label="Confidence Threshold",
                )
                max_frames = gr.Slider(
                    minimum=0, maximum=2000, value=300, step=50,
                    label="Max Frames (0 = all)",
                    info="Limit frames to process for faster results",
                )
                draw_trails = gr.Checkbox(value=True, label="Draw Trajectory Trails")

            run_btn = gr.Button("Run Detection & Tracking", variant="primary", size="lg")

        with gr.Column(scale=1):
            video_output = gr.Video(label="Annotated Output")
            summary_output = gr.Markdown(label="Results Summary")

    with gr.Row():
        traj_output = gr.Image(label="Trajectories", type="filepath")
        heat_output = gr.Image(label="Heatmap", type="filepath")
        count_output = gr.Image(label="Object Count", type="filepath")

    run_btn.click(
        fn=process_video,
        inputs=[video_input, model_name, tracker_type, confidence, max_frames, draw_trails],
        outputs=[video_output, summary_output, traj_output, heat_output, count_output],
    )

    gr.Markdown(
        """
        ---
        **How it works:** YOLOv8 detects all people in each frame. ByteTrack/BoT-SORT assigns
        persistent IDs using Kalman filter motion prediction and IoU-based Hungarian matching.
        Each person keeps their ID across frames, even through brief occlusions.

        [GitHub Repository](https://github.com/shishi2311/DetectFlow) |
        [Technical Report](https://github.com/shishi2311/DetectFlow/blob/main/TECHNICAL_REPORT.md)
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
