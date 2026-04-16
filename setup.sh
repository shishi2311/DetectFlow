#!/bin/bash
# DetectFlow - Environment Setup Script
# Creates a Python virtual environment and installs all dependencies.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#   source venv/bin/activate
#   python main.py --input video.mp4

set -e

echo "============================================"
echo "  DetectFlow - Environment Setup"
echo "============================================"

# Check Python version
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &> /dev/null; then
        version=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        major=$("$cmd" -c "import sys; print(sys.version_info.major)" 2>/dev/null)
        minor=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$cmd"
            echo "  Found $cmd ($version)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.10+ is required but not found."
    echo "Please install Python 3.10 or newer and try again."
    exit 1
fi

# Create virtual environment
if [ -d "venv" ]; then
    echo "  Virtual environment already exists. Reinstalling dependencies..."
else
    echo "  Creating virtual environment..."
    "$PYTHON" -m venv venv
fi

# Activate
source venv/bin/activate

# Upgrade pip
echo "  Upgrading pip..."
pip install --upgrade pip --quiet

# Install dependencies
echo "  Installing project dependencies..."
pip install -r requirements.txt --quiet

# Install yt-dlp for video downloading
echo "  Installing yt-dlp (video downloader)..."
pip install yt-dlp --quiet

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "  To activate the environment:"
echo "    source venv/bin/activate"
echo ""
echo "  To run the pipeline:"
echo "    python main.py --input video.mp4"
echo ""
echo "  To download + run in one command:"
echo "    python download_and_run.py --url 'https://youtube.com/watch?v=...'"
echo ""
echo "  To deactivate when done:"
echo "    deactivate"
echo "============================================"
