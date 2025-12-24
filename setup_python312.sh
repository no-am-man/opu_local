#!/bin/bash
# Setup script for Python 3.12 virtual environment with DeepFace support

set -e

echo "=== Python 3.12 Setup for OPU with DeepFace ==="
echo ""

# Check if Python 3.12 is available
if ! command -v python3.12 &> /dev/null; then
    echo "❌ Python 3.12 not found!"
    echo ""
    echo "Please install Python 3.12 first:"
    echo "1. Download from: https://www.python.org/downloads/release/python-3120/"
    echo "2. Or run: open /tmp/python3.12.pkg"
    echo "3. Complete the installation wizard"
    echo ""
    exit 1
fi

echo "✅ Python 3.12 found: $(python3.12 --version)"
echo ""

# Create virtual environment
VENV_DIR="venv_python312"
if [ -d "$VENV_DIR" ]; then
    echo "⚠️  Virtual environment already exists at $VENV_DIR"
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
    else
        echo "Using existing virtual environment..."
        source "$VENV_DIR/bin/activate"
        python --version
        exit 0
    fi
fi

echo "Creating virtual environment with Python 3.12..."
python3.12 -m venv "$VENV_DIR"

echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements (including DeepFace)..."
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To use this environment:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "To run OPU with DeepFace:"
echo "  source $VENV_DIR/bin/activate"
echo "  ./run_opu.sh"
echo ""

