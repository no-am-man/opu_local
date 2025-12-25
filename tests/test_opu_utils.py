"""
Tests for utils/opu_utils.py - Common OPU utility functions.
"""

import pytest
import sys
import time
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from utils.opu_utils import (
    setup_file_logging,
    cleanup_file_logging,
    calculate_ethical_veto,
    extract_emotion_from_detections,
    get_cycle_timestamp
)
from utils.file_logger import FileLogger


class TestSetupFileLogging:
    """Tests for setup_file_logging function."""
    
    def test_setup_file_logging_with_default(self):
        """Test setup_file_logging with default log file name."""
        logger, orig_stdout, orig_stderr = setup_file_logging()
        
        assert logger is not None
        assert orig_stdout is not None
        assert orig_stderr is not None
        assert sys.stdout == logger
        assert sys.stderr == logger
        
        cleanup_file_logging(logger, orig_stdout, orig_stderr)
        assert sys.stdout == orig_stdout
        assert sys.stderr == orig_stderr
    
    def test_setup_file_logging_with_custom_name(self):
        """Test setup_file_logging with custom log file name."""
        logger, orig_stdout, orig_stderr = setup_file_logging(log_file="test_custom.log")
        
        assert logger is not None
        assert logger.log_file_path == "test_custom.log"
        
        cleanup_file_logging(logger, orig_stdout, orig_stderr)
    
    def test_setup_file_logging_with_none(self):
        """Test setup_file_logging with None uses default."""
        logger, orig_stdout, orig_stderr = setup_file_logging(log_file=None, default_name="test_default.log")
        
        assert logger is not None
        assert logger.log_file_path == "test_default.log"
        
        cleanup_file_logging(logger, orig_stdout, orig_stderr)


class TestCleanupFileLogging:
    """Tests for cleanup_file_logging function."""
    
    def test_cleanup_file_logging_restores_stdout_stderr(self):
        """Test that cleanup restores original stdout/stderr."""
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        
        logger, _, _ = setup_file_logging()
        assert sys.stdout == logger
        assert sys.stderr == logger
        
        cleanup_file_logging(logger, orig_stdout, orig_stderr)
        assert sys.stdout == orig_stdout
        assert sys.stderr == orig_stderr
    
    def test_cleanup_file_logging_with_none(self):
        """Test cleanup with None logger doesn't crash."""
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        
        cleanup_file_logging(None, orig_stdout, orig_stderr)
        assert sys.stdout == orig_stdout
        assert sys.stderr == orig_stderr


class TestCalculateEthicalVeto:
    """Tests for calculate_ethical_veto function."""
    
    def test_calculate_ethical_veto_with_action(self):
        """Test ethical veto calculation when action is returned."""
        mock_genesis = Mock()
        mock_genesis.ethical_veto.return_value = np.array([0.5, 0.3])
        
        result = calculate_ethical_veto(mock_genesis, 0.8, 0.6)
        
        assert result == 0.5
        mock_genesis.ethical_veto.assert_called_once()
        call_args = mock_genesis.ethical_veto.call_args[0][0]
        np.testing.assert_array_equal(call_args, np.array([0.8, 0.6]))
    
    def test_calculate_ethical_veto_empty_action(self):
        """Test ethical veto when action is empty (returns fused_score)."""
        mock_genesis = Mock()
        mock_genesis.ethical_veto.return_value = np.array([])
        
        result = calculate_ethical_veto(mock_genesis, 0.8, 0.6)
        
        assert result == 0.8  # Returns fused_score when action is empty
        mock_genesis.ethical_veto.assert_called_once()


class TestExtractEmotionFromDetections:
    """Tests for extract_emotion_from_detections function."""
    
    def test_extract_emotion_with_face_and_emotion(self):
        """Test extracting emotion from detection with face and emotion."""
        detections = [
            {'label': 'person', 'confidence': 0.9},
            {'label': 'face', 'emotion': {'emotion': 'happy', 'confidence': 0.8}},
            {'label': 'car', 'confidence': 0.7}
        ]
        
        result = extract_emotion_from_detections(detections)
        
        assert result is not None
        assert result['emotion'] == 'happy'
        assert result['confidence'] == 0.8
    
    def test_extract_emotion_no_face(self):
        """Test extracting emotion when no face detected."""
        detections = [
            {'label': 'person', 'confidence': 0.9},
            {'label': 'car', 'confidence': 0.7}
        ]
        
        result = extract_emotion_from_detections(detections)
        
        assert result is None
    
    def test_extract_emotion_face_no_emotion(self):
        """Test extracting emotion when face detected but no emotion."""
        detections = [
            {'label': 'face', 'confidence': 0.9}
        ]
        
        result = extract_emotion_from_detections(detections)
        
        assert result is None
    
    def test_extract_emotion_empty_list(self):
        """Test extracting emotion from empty detection list."""
        result = extract_emotion_from_detections([])
        assert result is None
    
    def test_extract_emotion_none(self):
        """Test extracting emotion from None."""
        result = extract_emotion_from_detections(None)
        assert result is None


class TestGetCycleTimestamp:
    """Tests for get_cycle_timestamp function."""
    
    def test_get_cycle_timestamp_returns_float(self):
        """Test that get_cycle_timestamp returns a float."""
        timestamp = get_cycle_timestamp()
        
        assert isinstance(timestamp, float)
        assert timestamp > 0
    
    def test_get_cycle_timestamp_increases(self):
        """Test that timestamps increase over time."""
        timestamp1 = get_cycle_timestamp()
        time.sleep(0.01)  # Small delay
        timestamp2 = get_cycle_timestamp()
        
        assert timestamp2 >= timestamp1
    
    def test_get_cycle_timestamp_reasonable_value(self):
        """Test that timestamp is a reasonable epoch time."""
        timestamp = get_cycle_timestamp()
        current_time = time.time()
        
        # Should be within 1 second of current time
        assert abs(timestamp - current_time) < 1.0

