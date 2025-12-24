"""
Tests for Object Detection Module (core/object_detection.py)
OPU v3.2 - Visual Object Recognition + Emotion Detection with DeepFace
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
import sys


# Mock cv2 at module level before importing object_detection
mock_cv2 = MagicMock()
mock_cv2.data.haarcascades = '/fake/path/'
sys.modules['cv2'] = mock_cv2

from core.object_detection import ObjectDetector, DetectionConfig


class TestObjectDetector:
    """Test suite for ObjectDetector class."""
    
    def test_init_without_cv2(self):
        """Test initialization when cv2 is not available."""
        with patch.dict('sys.modules', {'cv2': None}):
            import importlib
            import core.object_detection
            importlib.reload(core.object_detection)
            from core.object_detection import ObjectDetector
            
            detector = ObjectDetector()
            assert detector.active is False
    
    def test_detect_objects_no_frame(self):
        """Test detect_objects with None frame."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                detector = ObjectDetector()
                result = detector.detect_objects(None)
                
                assert result == []
    
    def test_detect_objects_inactive(self):
        """Test detect_objects when detector is inactive."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                detector = ObjectDetector()
                detector.active = False
                
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                result = detector.detect_objects(frame)
                
                assert result == []
    
    def test_draw_detections_no_frame(self):
        """Test draw_detections with None frame."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                detector = ObjectDetector()
                result = detector.draw_detections(None, [])
                
                assert result is None
    
    def test_draw_detections_inactive(self):
        """Test draw_detections when inactive."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                detector = ObjectDetector()
                detector.active = False
                
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                result = detector.draw_detections(frame, [])
                
                assert result is frame
    
    def test_draw_detections_exception(self):
        """Test draw_detections with exception handling."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                detector = ObjectDetector()
                
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                detections = [{'label': 'face', 'bbox': (10, 20, 30, 40)}]
                
                with patch('cv2.rectangle', side_effect=Exception("OpenCV error")):
                    result = detector.draw_detections(frame, detections)
                    
                    assert result is frame
    
    def test_emotions_constant(self):
        """Test EMOTIONS constant in DetectionConfig."""
        config = DetectionConfig()
        assert config.EMOTIONS == ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        assert len(config.EMOTIONS) == 7
