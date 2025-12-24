#!/bin/bash
# OPU Launcher Script
# Sets environment variables needed for macOS compatibility before running
# This MUST be used on macOS with Python 3.13+ to prevent tkinter crashes

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate the Python 3.12 virtual environment
if [ -d "$SCRIPT_DIR/venv_python312" ]; then
    source "$SCRIPT_DIR/venv_python312/bin/activate"
else
    echo "Error: venv_python312 not found. Please run setup_python312.sh first."
    exit 1
fi

# Set tkinter deprecation warning suppression for macOS (Python 3.13+)
# This must be set BEFORE Python starts, not just before importing tkinter
export TK_SILENCE_DEPRECATION=1

# Also set for matplotlib
export MPLBACKEND=TkAgg

# Run the OPU
python3 main.py "$@"

