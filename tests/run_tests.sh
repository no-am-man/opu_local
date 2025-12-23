#!/bin/bash
# Test runner script for OPU v3.0.0
# Runs all tests with coverage reporting

set -e

echo "============================================================"
echo "OPU v3.0.0 Test Suite"
echo "============================================================"
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "Error: pytest is not installed."
    echo "Please install test dependencies: pip install -r requirements.txt"
    exit 1
fi

# Run tests with coverage
echo "Running tests with coverage..."
pytest tests/ -v --cov=core --cov=utils --cov=main --cov-report=term-missing --cov-report=html

echo ""
echo "============================================================"
echo "Coverage report generated in htmlcov/index.html"
echo "============================================================"

