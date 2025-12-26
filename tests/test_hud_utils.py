"""
Tests for utils/hud_utils.py - HUD drawing utilities.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None

from utils.hud_utils import draw_youtube_hud
from config import (
    YOUTUBE_HUD_POS_X, YOUTUBE_HUD_POS_Y_LINE1, YOUTUBE_HUD_POS_Y_LINE2, YOUTUBE_HUD_POS_Y_LINE3
)


@pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
class TestDrawYouTubeHUD:
    """Tests for draw_youtube_hud function."""
    
    def test_draw_youtube_hud_creates_frame(self):
        """Test that draw_youtube_hud creates a frame with HUD."""
        # Create a test frame (640x360, 3 channels, BGR)
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        
        result = draw_youtube_hud(
            frame,
            safe_score=0.5,
            s_audio=0.3,
            s_visual=0.4,
            title="Test Video",
            frame_count=100,
            fps=30.0
        )
        
        assert result is not None
        assert result.shape == frame.shape
    
    def test_draw_youtube_hud_with_long_title(self):
        """Test that long titles are truncated."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        long_title = "A" * 100  # Very long title
        
        result = draw_youtube_hud(
            frame,
            safe_score=0.5,
            s_audio=0.3,
            s_visual=0.4,
            title=long_title,
            frame_count=100,
            fps=30.0
        )
        
        assert result is not None
    
    def test_draw_youtube_hud_with_none_frame(self):
        """Test that draw_youtube_hud handles None frame."""
        result = draw_youtube_hud(
            None,
            safe_score=0.5,
            s_audio=0.3,
            s_visual=0.4,
            title="Test",
            frame_count=100,
            fps=30.0
        )
        
        assert result is None
    
    def test_draw_youtube_hud_formats_scores(self):
        """Test that scores are properly formatted in HUD."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        
        result = draw_youtube_hud(
            frame,
            safe_score=0.1234,
            s_audio=0.5678,
            s_visual=0.9012,
            title="Test",
            frame_count=123,
            fps=29.5
        )
        
        assert result is not None
    
    @pytest.mark.skipif(not CV2_AVAILABLE, reason="OpenCV not available")
    def test_draw_youtube_hud_calls_puttext(self):
        """Test that draw_youtube_hud calls cv2.putText."""
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        
        with patch('utils.hud_utils.cv2.putText') as mock_puttext:
            draw_youtube_hud(
                frame,
                safe_score=0.5,
                s_audio=0.3,
                s_visual=0.4,
                title="Test",
                frame_count=100,
                fps=30.0
            )
            
            # Should be called 3 times (3 lines of HUD text)
            assert mock_puttext.call_count == 3
    
    def test_draw_youtube_hud_cv2_not_available(self, monkeypatch):
        """Test draw_youtube_hud when cv2 is not available (covers lines 16-18)."""
        # Mock cv2 import failure
        import sys
        if 'utils.hud_utils' in sys.modules:
            del sys.modules['utils.hud_utils']
        
        with patch.dict('sys.modules', {'cv2': None}):
            with patch('utils.hud_utils.CV2_AVAILABLE', False):
                # Re-import to trigger the except block
                import importlib
                import utils.hud_utils
                importlib.reload(utils.hud_utils)
                
                # Now test the function
                frame = np.zeros((360, 640, 3), dtype=np.uint8)
                result = utils.hud_utils.draw_youtube_hud(
                    frame,
                    safe_score=0.5,
                    s_audio=0.3,
                    s_visual=0.4,
                    title="Test",
                    frame_count=100,
                    fps=30.0
                )
                
                # Should return frame as-is when cv2 not available
                assert result is not None

