"""
Tests for Camera Capture (core/camera.py)
OPU v3.1 - Multi-Modal Integration
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch, Mock

# Mock cv2 at module level before importing camera
import sys
mock_cv2 = MagicMock()
sys.modules['cv2'] = mock_cv2

from core.camera import VisualPerception


class TestVisualPerception:
    """Test suite for VisualPerception class."""
    
    def test_init_with_camera(self):
        """Test VisualPerception initialization with camera."""
        with patch('cv2.VideoCapture') as mock_cap:
            mock_cap_instance = MagicMock()
            mock_cap_instance.isOpened.return_value = True
            mock_cap.return_value = mock_cap_instance
            
            vc = VisualPerception(camera_index=0)
            
            assert vc.active is True
            assert vc.cap is not None
            mock_cap_instance.set.assert_called()
    
    def test_init_without_camera(self):
        """Test VisualPerception initialization without camera."""
        with patch('cv2.VideoCapture') as mock_cap:
            mock_cap_instance = MagicMock()
            mock_cap_instance.isOpened.return_value = False
            mock_cap.return_value = mock_cap_instance
            
            vc = VisualPerception(camera_index=0)
            
            assert vc.active is False
    
    def test_get_visual_input_success(self):
        """Test successful visual input capture."""
        with patch('cv2.VideoCapture') as mock_cap, patch('cv2.split') as mock_split:
            mock_cap_instance = MagicMock()
            mock_cap_instance.isOpened.return_value = True
            mock_cap.return_value = mock_cap_instance
            
            # Create a test frame (320x240, 3 channels)
            test_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            mock_cap_instance.read.return_value = (True, test_frame)
            
            # Mock cv2.split to return B, G, R channels
            b_channel = test_frame[:, :, 0]
            g_channel = test_frame[:, :, 1]
            r_channel = test_frame[:, :, 2]
            mock_split.return_value = (b_channel, g_channel, r_channel)
            
            vc = VisualPerception(camera_index=0)
            visual_vector, frame = vc.get_visual_input()
            
            assert visual_vector is not None
            assert len(visual_vector) == 3
            assert all(isinstance(x, (float, np.floating)) for x in visual_vector)
            assert all(x >= 0 for x in visual_vector)  # Std dev is always >= 0
            assert frame is not None
            assert frame.shape == (240, 320, 3)
    
    def test_get_visual_input_no_camera(self):
        """Test visual input when camera is not active."""
        with patch('cv2.VideoCapture') as mock_cap:
            mock_cap_instance = MagicMock()
            mock_cap_instance.isOpened.return_value = False
            mock_cap.return_value = mock_cap_instance
            
            vc = VisualPerception(camera_index=0)
            visual_vector, frame = vc.get_visual_input()
            
            assert np.array_equal(visual_vector, np.array([0.0, 0.0, 0.0]))
            assert frame is None
    
    def test_get_visual_input_read_failure(self):
        """Test visual input when frame read fails."""
        with patch('cv2.VideoCapture') as mock_cap:
            mock_cap_instance = MagicMock()
            mock_cap_instance.isOpened.return_value = True
            mock_cap_instance.read.return_value = (False, None)
            mock_cap.return_value = mock_cap_instance
            
            vc = VisualPerception(camera_index=0)
            visual_vector, frame = vc.get_visual_input()
            
            assert np.array_equal(visual_vector, np.array([0.0, 0.0, 0.0]))
            assert frame is None
    
    def test_get_visual_input_channel_separation(self):
        """Test that R, G, B channels are correctly separated."""
        with patch('cv2.VideoCapture') as mock_cap, patch('cv2.split') as mock_split:
            mock_cap_instance = MagicMock()
            mock_cap_instance.isOpened.return_value = True
            mock_cap.return_value = mock_cap_instance
            
            # Create a test frame with known channel values
            # OpenCV uses BGR format, so we need to set channels accordingly
            test_frame = np.zeros((240, 320, 3), dtype=np.uint8)
            test_frame[:, :, 0] = 100  # B channel
            test_frame[:, :, 1] = 150  # G channel
            test_frame[:, :, 2] = 200  # R channel
            
            mock_cap_instance.read.return_value = (True, test_frame)
            
            # Mock cv2.split to return B, G, R channels
            b_channel = test_frame[:, :, 0]
            g_channel = test_frame[:, :, 1]
            r_channel = test_frame[:, :, 2]
            mock_split.return_value = (b_channel, g_channel, r_channel)
            
            vc = VisualPerception(camera_index=0)
            visual_vector, frame = vc.get_visual_input()
            
            # All channels should have std dev of 0 (uniform values)
            assert visual_vector[0] == 0.0  # R channel std dev
            assert visual_vector[1] == 0.0  # G channel std dev
            assert visual_vector[2] == 0.0  # B channel std dev
    
    def test_get_visual_input_high_entropy(self):
        """Test visual input with high entropy (random frame)."""
        with patch('cv2.VideoCapture') as mock_cap, patch('cv2.split') as mock_split:
            mock_cap_instance = MagicMock()
            mock_cap_instance.isOpened.return_value = True
            mock_cap.return_value = mock_cap_instance
            
            # Create a high-entropy frame (random values)
            test_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            mock_cap_instance.read.return_value = (True, test_frame)
            
            # Mock cv2.split to return B, G, R channels
            b_channel = test_frame[:, :, 0]
            g_channel = test_frame[:, :, 1]
            r_channel = test_frame[:, :, 2]
            mock_split.return_value = (b_channel, g_channel, r_channel)
            
            vc = VisualPerception(camera_index=0)
            visual_vector, frame = vc.get_visual_input()
            
            # High entropy should produce high std dev values
            assert all(x > 0 for x in visual_vector)
            # Random frame should have std dev around 70-80
            assert all(50 < x < 100 for x in visual_vector)
    
    def test_is_active(self):
        """Test is_active method."""
        with patch('cv2.VideoCapture') as mock_cap:
            mock_cap_instance = MagicMock()
            mock_cap_instance.isOpened.return_value = True
            mock_cap.return_value = mock_cap_instance
            
            vc = VisualPerception(camera_index=0)
            assert vc.is_active() is True
            
            vc.active = False
            assert vc.is_active() is False
    
    def test_cleanup(self):
        """Test cleanup method."""
        with patch('cv2.VideoCapture') as mock_cap:
            mock_cap_instance = MagicMock()
            mock_cap_instance.isOpened.return_value = True
            mock_cap.return_value = mock_cap_instance
            
            vc = VisualPerception(camera_index=0)
            vc.cleanup()
            
            mock_cap_instance.release.assert_called_once()
            assert vc.active is False
    
    def test_cleanup_inactive(self):
        """Test cleanup when already inactive."""
        with patch('cv2.VideoCapture') as mock_cap:
            mock_cap_instance = MagicMock()
            mock_cap_instance.isOpened.return_value = False
            mock_cap.return_value = mock_cap_instance
            
            vc = VisualPerception(camera_index=0)
            vc.cleanup()
            
            # Should not raise exception
            assert vc.active is False
    
    def test_get_visual_input_returns_float32(self):
        """Test that visual vector is returned as float32."""
        with patch('cv2.VideoCapture') as mock_cap, patch('cv2.split') as mock_split:
            mock_cap_instance = MagicMock()
            mock_cap_instance.isOpened.return_value = True
            mock_cap.return_value = mock_cap_instance
            
            test_frame = np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8)
            mock_cap_instance.read.return_value = (True, test_frame)
            
            # Mock cv2.split to return B, G, R channels
            b_channel = test_frame[:, :, 0]
            g_channel = test_frame[:, :, 1]
            r_channel = test_frame[:, :, 2]
            mock_split.return_value = (b_channel, g_channel, r_channel)
            
            vc = VisualPerception(camera_index=0)
            visual_vector, frame = vc.get_visual_input()
            
            assert visual_vector.dtype == np.float32

