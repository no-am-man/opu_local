#!/bin/bash
# OPU Launcher Script
# Sets environment variables needed for macOS compatibility before running
# This MUST be used on macOS with Python 3.13+ to prevent tkinter crashes

# Set tkinter deprecation warning suppression for macOS (Python 3.13+)
# This must be set BEFORE Python starts, not just before importing tkinter
export TK_SILENCE_DEPRECATION=1

# Also set for matplotlib
export MPLBACKEND=TkAgg

# Run the OPU
python3 main.py "$@"

