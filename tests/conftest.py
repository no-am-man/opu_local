"""
Pytest configuration and shared fixtures for OPU tests.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path


@pytest.fixture
def sample_audio_vector():
    """Generate a sample audio input vector for testing."""
    return np.random.randn(1024).astype(np.float32)


@pytest.fixture
def sample_audio_vector_empty():
    """Empty audio vector."""
    return np.array([])


@pytest.fixture
def sample_audio_vector_constant():
    """Constant audio vector (zero std dev)."""
    return np.ones(100) * 0.5


@pytest.fixture
def sample_genomic_bit():
    """Sample genomic bit value."""
    return 0.5


@pytest.fixture
def sample_s_score():
    """Sample surprise score."""
    return 2.5


@pytest.fixture
def temp_state_file():
    """Create a temporary state file for persistence tests."""
    fd, path = tempfile.mkstemp(suffix='.json')
    os.close(fd)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def temp_state_dir():
    """Create a temporary directory for state files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_audio_stream(mocker):
    """Mock audio stream to avoid hardware dependencies."""
    mock_stream = mocker.MagicMock()
    mock_stream.start = mocker.MagicMock()
    mock_stream.stop = mocker.MagicMock()
    mock_stream.close = mocker.MagicMock()
    return mock_stream

