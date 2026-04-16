#!/usr/bin/env python3
"""
Download a public video and run the DetectFlow pipeline in one command.

Usage:
    python download_and_run.py --url "https://www.youtube.com/watch?v=VIDEO_ID"
    python download_and_run.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --max-frames 300
    python download_and_run.py --url "https://www.youtube.com/watch?v=VIDEO_ID" --tracker botsort --device cuda

Requirements:
    pip install yt-dlp
"""

import argparse
import subprocess
import sys
from pathlib import Path


def check_ytdlp():
    """Check if yt-dlp is installed."""
    try:
        subprocess.run(["yt-dlp", "--version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def download_video(url: str, output_path: str, max_height: int = 720) -> str:
    """Download video using yt-dlp."""
    print(f"Downloading video from: {url}")
    print(f"Max resolution: {max_height}p")

    cmd = [
        "yt-dlp",
        "-f", f"best[height<={max_height}][ext=mp4]/best[height<={max_height}]/best",
        "-o", output_path,
        "--no-playlist",
        url,
    ]

    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print("yt-dlp: first attempt failed. Trying alternative format selection...")
        cmd_alt = [
            "yt-dlp",
            "-f", "best",
            "-o", output_path,
            "--no-playlist",
            url,
        ]
        result = subprocess.run(cmd_alt, capture_output=False)
        if result.returncode != 0:
            sys.exit("ERROR: Failed to download video. Check the URL and try again.")

    if not Path(output_path).exists():
        # yt-dlp may add extension, find the file
        for ext in [".mp4", ".webm", ".mkv"]:
            candidate = output_path.rsplit(".", 1)[0] + ext
            if Path(candidate).exists():
                return candidate
        sys.exit(f"ERROR: Downloaded file not found at {output_path}")

    return output_path


def run_pipeline(video_path: str, source_url: str, extra_args: list[str]):
    """Run the DetectFlow pipeline."""
    cmd = [
        sys.executable, "main.py",
        "--input", video_path,
        "--source-url", source_url,
    ] + extra_args

    print("\n" + "=" * 60)
    print("  Starting DetectFlow Pipeline")
    print("=" * 60)
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Download a public video and run DetectFlow pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url", "-u",
        required=True,
        help="Public video URL (YouTube, Vimeo, etc.)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=720,
        help="Max video resolution to download (height in pixels).",
    )
    parser.add_argument(
        "--video-path",
        default="input_video.mp4",
        help="Path to save the downloaded video.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download if video already exists at --video-path.",
    )

    # Pass remaining args to main.py
    args, remaining = parser.parse_known_args()

    # Check yt-dlp
    if not args.skip_download:
        if not check_ytdlp():
            print("yt-dlp is not installed. Installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "yt-dlp"], check=True)

    # Download
    video_path = args.video_path
    if args.skip_download and Path(video_path).exists():
        print(f"Using existing video: {video_path}")
    else:
        video_path = download_video(args.url, args.video_path, args.resolution)
        print(f"Video downloaded: {video_path}")

    # Run pipeline
    rc = run_pipeline(video_path, args.url, remaining)

    if rc == 0:
        print("\n" + "=" * 60)
        print("  Done! Check the output/ directory for results:")
        print("    - output/annotated_output.mp4  (annotated video)")
        print("    - output/screenshots/          (sample frames)")
        print("    - output/trajectories.png      (trajectory plot)")
        print("    - output/heatmap.png           (position heatmap)")
        print("    - output/count_over_time.png   (object count graph)")
        print("=" * 60)
    else:
        print(f"\nPipeline exited with code {rc}")

    sys.exit(rc)


if __name__ == "__main__":
    main()
