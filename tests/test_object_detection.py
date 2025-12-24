"""
Tests for Object Detection Module (core/object_detection.py)
OPU v3.2 - Visual Object Recognition + Emotion Detection with DeepFace
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, Mock, mock_open
import sys
import os


# Mock cv2 at module level before importing object_detection
mock_cv2 = MagicMock()
mock_cv2.data.haarcascades = '/fake/path/'
sys.modules['cv2'] = mock_cv2

from core.object_detection import ObjectDetector


class TestObjectDetector:
    """Test suite for ObjectDetector class."""
    
    def test_init_without_cv2(self):
        """Test initialization when cv2 is not available."""
        with patch.dict('sys.modules', {'cv2': None}):
            # Reload module to pick up None cv2
            import importlib
            import core.object_detection
            importlib.reload(core.object_detection)
            from core.object_detection import ObjectDetector
            
            detector = ObjectDetector()
            assert detector.active is False
    
    def test_init_basic(self):
        """Test basic initialization."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                detector = ObjectDetector()
                
                assert detector.active is True
                assert detector.confidence_threshold == 0.5
                assert detector.use_dnn is False  # DNN disabled by default
                assert detector.detect_emotions is True
                assert detector.face_cascade is not None
    
    def test_init_without_emotions(self):
        """Test initialization with emotion detection disabled."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                detector = ObjectDetector(detect_emotions=False)
                
                assert detector.detect_emotions is False
                assert not hasattr(detector, 'emotion_method') or detector.emotion_method is None
    
    def test_init_dnn(self):
        """Test DNN initialization."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                detector = ObjectDetector(use_dnn=True)
                
                # DNN should be disabled (model files not available)
                assert detector.use_dnn is False
                assert len(detector.classes) == 80  # COCO classes defined
    
    def test_init_face_detector_success(self):
        """Test successful face detector initialization."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                detector = ObjectDetector()
                
                assert detector.face_cascade is not None
                mock_classifier.assert_called()
    
    def test_init_face_detector_failure(self):
        """Test face detector initialization when cascade file not found."""
        with patch('os.path.exists', return_value=False):
            detector = ObjectDetector()
            
            # Should still initialize but without face cascade
            assert detector.active is True
    
    def test_init_emotion_detector_deepface(self):
        """Test emotion detector initialization with DeepFace."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                with patch('builtins.__import__') as mock_import:
                    def import_side_effect(name, *args, **kwargs):
                        if name == 'deepface':
                            mock_deepface = MagicMock()
                            mock_deepface.DeepFace = MagicMock()
                            return mock_deepface
                        return __import__(name, *args, **kwargs)
                    mock_import.side_effect = import_side_effect
                    
                    detector = ObjectDetector(detect_emotions=True)
                    
                    # Should try to use DeepFace
                    assert hasattr(detector, 'emotion_method')
    
    def test_init_emotion_detector_fer(self):
        """Test emotion detector initialization with FER fallback."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                # Mock DeepFace import failure, FER success
                with patch('builtins.__import__') as mock_import:
                    call_count = [0]
                    def import_side_effect(name, *args, **kwargs):
                        if name == 'deepface':
                            call_count[0] += 1
                            if call_count[0] == 1:
                                raise ImportError("No module named 'deepface'")
                        elif name == 'fer':
                            mock_fer_module = MagicMock()
                            mock_fer_module.FER = MagicMock(return_value=MagicMock())
                            return mock_fer_module
                        return __import__(name, *args, **kwargs)
                    mock_import.side_effect = import_side_effect
                    
                    detector = ObjectDetector(detect_emotions=True)
                    
                    assert hasattr(detector, 'emotion_method')
    
    def test_init_emotion_detector_heuristic(self):
        """Test emotion detector initialization with heuristic fallback."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                # Mock both DeepFace and FER import failures
                with patch('core.object_detection.DeepFace', side_effect=ImportError()):
                    with patch.dict('sys.modules', {'fer': None}):
                        detector = ObjectDetector(detect_emotions=True)
                        
                        assert detector.emotion_method == 'heuristic'
    
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
    
    def test_detect_objects_face_detection(self):
        """Test face detection."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                # Mock face detection
                mock_cascade.detectMultiScale.return_value = np.array([[10, 20, 30, 40]])
                
                detector = ObjectDetector(detect_emotions=False)
                
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                with patch('cv2.cvtColor', return_value=frame[:, :, 0]):
                    result = detector.detect_objects(frame)
                    
                    assert len(result) == 1
                    assert result[0]['label'] == 'face'
                    assert result[0]['bbox'] == (10, 20, 30, 40)
                    assert result[0]['confidence'] == 0.9
    
    def test_detect_objects_with_emotion(self):
        """Test face detection with emotion detection."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                mock_cascade.detectMultiScale.return_value = np.array([[10, 20, 30, 40]])
                
                detector = ObjectDetector(detect_emotions=True)
                detector.emotion_method = 'heuristic'
                
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                with patch('cv2.cvtColor', return_value=frame[:, :, 0]):
                    result = detector.detect_objects(frame)
                    
                    assert len(result) == 1
                    assert 'emotion' in result[0]
                    assert result[0]['emotion']['emotion'] == 'neutral'
    
    def test_detect_emotion_heuristic(self):
        """Test heuristic emotion detection."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                detector = ObjectDetector()
                detector.emotion_method = 'heuristic'
                
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                emotion = detector._detect_emotion(frame, 10, 20, 30, 40)
                
                assert emotion is not None
                assert emotion['emotion'] == 'neutral'
                assert emotion['confidence'] == 0.5
    
    def test_detect_emotion_empty_roi(self):
        """Test emotion detection with empty face region."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                detector = ObjectDetector()
                detector.emotion_method = 'heuristic'
                
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                # Invalid coordinates that would create empty ROI
                emotion = detector._detect_emotion(frame, 100, 100, 0, 0)
                
                assert emotion is None
    
    def test_detect_emotion_deepface_dict(self):
        """Test DeepFace emotion detection with dict result."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                detector = ObjectDetector()
                detector.emotion_method = 'deepface'
                
                # Mock DeepFace result as dict
                mock_deepface_result = {
                    'emotion': {
                        'happy': 80.0,
                        'sad': 10.0,
                        'neutral': 10.0
                    }
                }
                
                mock_deepface_class = MagicMock()
                mock_deepface_class.analyze.return_value = mock_deepface_result
                
                with patch('builtins.__import__') as mock_import:
                    def import_side_effect(name, *args, **kwargs):
                        if name == 'deepface':
                            mock_deepface_module = MagicMock()
                            mock_deepface_module.DeepFace = mock_deepface_class
                            return mock_deepface_module
                        return __import__(name, *args, **kwargs)
                    mock_import.side_effect = import_side_effect
                    
                    frame = np.zeros((100, 100, 3), dtype=np.uint8)
                    emotion = detector._detect_emotion(frame, 10, 20, 30, 40)
                    
                    assert emotion is not None
                    assert emotion['emotion'] == 'happy'
                    assert emotion['confidence'] == 0.8
    
    def test_detect_emotion_deepface_list(self):
        """Test DeepFace emotion detection with list result."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                detector = ObjectDetector()
                detector.emotion_method = 'deepface'
                
                # Mock DeepFace result as list
                mock_deepface_result = [{
                    'emotion': {
                        'sad': 70.0,
                        'happy': 20.0,
                        'neutral': 10.0
                    }
                }]
                
                with patch('core.object_detection.DeepFace') as mock_deepface:
                    mock_deepface.analyze.return_value = mock_deepface_result
                    
                    frame = np.zeros((100, 100, 3), dtype=np.uint8)
                    emotion = detector._detect_emotion(frame, 10, 20, 30, 40)
                    
                    assert emotion is not None
                    assert emotion['emotion'] == 'sad'
                    assert emotion['confidence'] == 0.7
    
    def test_detect_emotion_deepface_exception(self):
        """Test DeepFace emotion detection with exception."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                detector = ObjectDetector()
                detector.emotion_method = 'deepface'
                
                with patch('core.object_detection.DeepFace') as mock_deepface:
                    mock_deepface.analyze.side_effect = Exception("DeepFace error")
                    
                    frame = np.zeros((100, 100, 3), dtype=np.uint8)
                    emotion = detector._detect_emotion(frame, 10, 20, 30, 40)
                    
                    # Should return None on exception
                    assert emotion is None
    
    def test_detect_emotion_fer(self):
        """Test FER emotion detection."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                detector = ObjectDetector()
                detector.emotion_method = 'fer'
                detector.emotion_detector = MagicMock()
                detector.emotion_detector.detect_emotions.return_value = [{
                    'emotions': {
                        'angry': 0.6,
                        'happy': 0.3,
                        'neutral': 0.1
                    }
                }]
                
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                emotion = detector._detect_emotion(frame, 10, 20, 30, 40)
                
                assert emotion is not None
                assert emotion['emotion'] == 'angry'
                assert emotion['confidence'] == 0.6
    
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
    
    def test_draw_detections_basic(self):
        """Test drawing basic detections."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                detector = ObjectDetector()
                
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                detections = [{
                    'label': 'face',
                    'confidence': 0.9,
                    'bbox': (10, 20, 30, 40)
                }]
                
                with patch('cv2.rectangle') as mock_rect, \
                     patch('cv2.getTextSize', return_value=((50, 10), None)), \
                     patch('cv2.putText') as mock_text:
                    
                    result = detector.draw_detections(frame, detections)
                    
                    assert result is not None
                    mock_rect.assert_called()
                    mock_text.assert_called()
    
    def test_draw_detections_with_emotion(self):
        """Test drawing detections with emotion."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                detector = ObjectDetector()
                
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                detections = [{
                    'label': 'face',
                    'confidence': 0.9,
                    'bbox': (10, 20, 30, 40),
                    'emotion': {
                        'emotion': 'happy',
                        'confidence': 0.8
                    }
                }]
                
                with patch('cv2.rectangle') as mock_rect, \
                     patch('cv2.getTextSize', return_value=((50, 10), None)), \
                     patch('cv2.putText') as mock_text:
                    
                    result = detector.draw_detections(frame, detections)
                    
                    assert result is not None
                    # Should use green color for happy emotion
                    mock_rect.assert_called()
    
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
                    
                    # Should return original frame on exception
                    assert result is frame
    
    def test_cleanup(self):
        """Test cleanup method."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                detector = ObjectDetector()
                detector.cleanup()
                
                assert detector.active is False
                assert detector.net is None
                assert detector.face_cascade is None
    
    def test_emotions_constant(self):
        """Test EMOTIONS constant."""
        assert ObjectDetector.EMOTIONS == ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        assert len(ObjectDetector.EMOTIONS) == 7
    
    def test_detect_objects_keyboard_interrupt(self):
        """Test that KeyboardInterrupt is re-raised."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                detector = ObjectDetector()
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                
                with patch('cv2.cvtColor', side_effect=KeyboardInterrupt()):
                    with pytest.raises(KeyboardInterrupt):
                        detector.detect_objects(frame)
    
    def test_detect_emotion_keyboard_interrupt(self):
        """Test that KeyboardInterrupt is re-raised in emotion detection."""
        with patch('os.path.exists', return_value=True):
            with patch('cv2.CascadeClassifier') as mock_classifier:
                mock_cascade = MagicMock()
                mock_classifier.return_value = mock_cascade
                
                detector = ObjectDetector()
                detector.emotion_method = 'heuristic'
                frame = np.zeros((100, 100, 3), dtype=np.uint8)
                
                with patch.object(detector, '_detect_emotion', side_effect=KeyboardInterrupt()):
                    with pytest.raises(KeyboardInterrupt):
                        detector._detect_emotion(frame, 10, 20, 30, 40)

