#!/bin/bash
# Disk Space Cleanup Script for OPU Setup
# Safely removes cache files to free up space for Python 3.12 and DeepFace installation

set -e

echo "=== Disk Space Cleanup Script ==="
echo ""
echo "This script will clean up cache directories to free up disk space."
echo "All caches can be safely removed - they will be regenerated as needed."
echo ""

# Check current disk space
echo "Current disk usage:"
df -h . | tail -1
echo ""

# Calculate space before cleanup
SPACE_BEFORE=$(df . | tail -1 | awk '{print $4}')
echo "Space available before cleanup: ${SPACE_BEFORE}KB"
echo ""

# Function to safely remove directory with size reporting
cleanup_dir() {
    local dir=$1
    local name=$2
    
    if [ -d "$dir" ]; then
        local size=$(du -sh "$dir" 2>/dev/null | cut -f1)
        echo "Cleaning $name ($size)..."
        rm -rf "$dir"/*
        echo "✅ Cleaned $name"
    else
        echo "⚠️  $name not found, skipping..."
    fi
}

# Clean Homebrew cache
if command -v brew &> /dev/null; then
    echo "Cleaning Homebrew cache..."
    brew cleanup --prune=all 2>/dev/null || echo "⚠️  Homebrew cleanup had issues (may need sudo)"
    echo "✅ Homebrew cache cleaned"
else
    echo "⚠️  Homebrew not found, skipping..."
fi
echo ""

# Clean pip cache (all Python versions)
echo "Cleaning pip cache..."
if command -v pip3 &> /dev/null; then
    pip3 cache purge 2>/dev/null || echo "⚠️  pip3 cache purge had issues"
fi
if command -v pip &> /dev/null; then
    pip cache purge 2>/dev/null || echo "⚠️  pip cache purge had issues"
fi
# Also clean the cache directory directly
cleanup_dir "$HOME/Library/Caches/pip" "pip cache"
echo ""

# Clean Google Chrome cache
cleanup_dir "$HOME/Library/Caches/Google" "Google Chrome cache"
echo ""

# Clean Spotify cache
cleanup_dir "$HOME/Library/Caches/com.spotify.client" "Spotify cache"
echo ""

# Clean TypeScript cache
cleanup_dir "$HOME/Library/Caches/typescript" "TypeScript cache"
echo ""

# Clean node-gyp cache
cleanup_dir "$HOME/Library/Caches/node-gyp" "node-gyp cache"
echo ""

# Clean temporary files
echo "Cleaning temporary files..."
rm -rf /tmp/python3.12.pkg 2>/dev/null || true
rm -rf /tmp/get-pip.py 2>/dev/null || true
rm -rf /tmp/*.pkg 2>/dev/null || true
echo "✅ Temporary files cleaned"
echo ""

# Calculate space after cleanup
SPACE_AFTER=$(df . | tail -1 | awk '{print $4}')
FREED=$((SPACE_AFTER - SPACE_BEFORE))

echo "=== Cleanup Complete ==="
echo ""
echo "Space available after cleanup: ${SPACE_AFTER}KB"
if [ $FREED -gt 0 ]; then
    echo "✅ Freed approximately: ${FREED}KB"
else
    echo "⚠️  No significant space freed (may need manual cleanup)"
fi
echo ""
echo "Current disk usage:"
df -h . | tail -1
echo ""

# Check if we have enough space (need at least 2GB for TensorFlow + DeepFace)
AVAILABLE_GB=$(df . | tail -1 | awk '{print int($4/1024/1024)}')
if [ $AVAILABLE_GB -lt 2 ]; then
    echo "⚠️  WARNING: Still low on disk space (${AVAILABLE_GB}GB available)"
    echo "   TensorFlow and DeepFace require ~2GB+ of free space."
    echo "   You may need to free up more space manually."
    echo ""
    echo "   Suggestions:"
    echo "   - Empty Trash"
    echo "   - Remove old downloads"
    echo "   - Uninstall unused applications"
    echo "   - Use 'du -sh ~/*' to find large directories"
else
    echo "✅ Sufficient space available (${AVAILABLE_GB}GB) for installation!"
fi
echo ""

