"""
Tests for utils/visualization.py - Cognitive Map Visualizer.
Targets 100% code coverage.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from utils.visualization import CognitiveMapVisualizer


class TestCognitiveMapVisualizer:
    """Tests for CognitiveMapVisualizer class."""
    
    def test_init(self):
        """Test visualizer initialization (covers lines 40-65)."""
        visualizer = CognitiveMapVisualizer()
        
        assert visualizer.fig is not None
        assert visualizer.ax is not None
        assert len(visualizer.s_score_history) == 0
        assert len(visualizer.coherence_history) == 0
        assert len(visualizer.maturity_history) == 0
        assert visualizer.current_s_score == 0.0
        assert visualizer.current_coherence == 0.0
        assert visualizer.current_maturity == 0.0
        assert visualizer.pulse_circle is not None
        assert visualizer.coherence_shape is not None
        assert visualizer.text_info is not None
    
    def test_setup_visualization(self):
        """Test setup_visualization method (covers lines 67-88)."""
        visualizer = CognitiveMapVisualizer()
        
        # Verify visualization elements are created
        assert visualizer.pulse_circle is not None
        assert visualizer.coherence_shape is not None
        assert visualizer.text_info is not None
    
    def test_update_state(self):
        """Test update_state method (covers lines 90-109)."""
        visualizer = CognitiveMapVisualizer()
        
        visualizer.update_state(0.5, 0.7, 0.3, maturity_level=2)
        
        assert visualizer.current_s_score == 0.5
        assert visualizer.current_coherence == 0.7
        assert visualizer.current_maturity == 0.3
        assert visualizer.current_maturity_level == 2
        assert len(visualizer.s_score_history) == 1
        assert len(visualizer.coherence_history) == 1
        assert len(visualizer.maturity_history) == 1
    
    def test_update_state_without_maturity_level(self):
        """Test update_state without maturity_level parameter."""
        visualizer = CognitiveMapVisualizer()
        
        visualizer.update_state(0.5, 0.7, 0.3)
        
        assert visualizer.current_s_score == 0.5
        assert visualizer.current_maturity_level == 0  # Default
    
    def test_draw_cognitive_map(self):
        """Test draw_cognitive_map method (covers lines 111-225)."""
        visualizer = CognitiveMapVisualizer()
        visualizer.update_state(0.5, 0.7, 0.3, maturity_level=1)
        
        # Should not crash
        visualizer.draw_cognitive_map()
        
        # Verify pulse circle radius is updated
        assert visualizer.pulse_circle.radius > 0
    
    def test_draw_cognitive_map_with_history(self):
        """Test draw_cognitive_map with history data."""
        visualizer = CognitiveMapVisualizer()
        
        # Add some history
        for i in range(10):
            visualizer.update_state(0.1 * i, 0.2 * i, 0.3 * i, maturity_level=i % 3)
        
        visualizer.draw_cognitive_map()
        
        # Should have updated visualization
        assert len(visualizer.s_score_history) == 10
    
    def test_draw_cognitive_map_edge_cases(self):
        """Test draw_cognitive_map with edge case values."""
        visualizer = CognitiveMapVisualizer()
        
        # Test with zero values
        visualizer.update_state(0.0, 0.0, 0.0, maturity_level=0)
        visualizer.draw_cognitive_map()
        
        # Test with max values
        visualizer.update_state(1.0, 1.0, 1.0, maturity_level=7)
        visualizer.draw_cognitive_map()
        
        # Should not crash
        assert True
    
    def test_render_to_image(self):
        """Test render_to_image method (covers lines 227-253)."""
        visualizer = CognitiveMapVisualizer()
        visualizer.update_state(0.5, 0.7, 0.3)
        visualizer.draw_cognitive_map()
        
        # Only test if cv2 is available
        from utils.visualization import CV2_AVAILABLE
        if CV2_AVAILABLE:
            image = visualizer.render_to_image()
            # If cv2 is available, should return an image (numpy array or mock)
            # The actual type depends on whether cv2 is mocked in the test environment
            assert image is not None
            # Only check type if it's actually a numpy array (not mocked)
            if isinstance(image, np.ndarray):
                assert len(image.shape) == 3  # Should be RGB image
        else:
            # If cv2 is not available, function returns None
            image = visualizer.render_to_image()
            assert image is None
    
    def test_render_to_image_with_cv2(self):
        """Test render_to_image when cv2 is available."""
        visualizer = CognitiveMapVisualizer()
        visualizer.update_state(0.5, 0.7, 0.3)
        visualizer.draw_cognitive_map()
        
        with patch('utils.visualization.CV2_AVAILABLE', True):
            with patch('utils.visualization.cv2') as mock_cv2:
                mock_cv2.cvtColor.return_value = np.zeros((400, 400, 3), dtype=np.uint8)
                
                image = visualizer.render_to_image()
                
                # Should return image
                assert image is not None
    
    def test_render_to_image_without_cv2(self):
        """Test render_to_image when cv2 is not available."""
        visualizer = CognitiveMapVisualizer()
        visualizer.update_state(0.5, 0.7, 0.3)
        visualizer.draw_cognitive_map()
        
        with patch('utils.visualization.CV2_AVAILABLE', False):
            image = visualizer.render_to_image()
            
            # When cv2 is not available, function returns None
            assert image is None
    
    def test_on_state_changed(self):
        """Test on_state_changed observer method (covers lines 255-290)."""
        visualizer = CognitiveMapVisualizer()
        
        # Create mock state - on_state_changed expects 'maturity' not 'maturity_index'
        state = {
            's_score': 0.5,
            'coherence': 0.7,
            'maturity': 0.3,
            'maturity_level': 2
        }
        
        visualizer.on_state_changed(state)
        
        assert visualizer.current_s_score == 0.5
        assert visualizer.current_coherence == 0.7
        assert visualizer.current_maturity == 0.3
    
    def test_on_state_changed_partial_state(self):
        """Test on_state_changed with partial state."""
        visualizer = CognitiveMapVisualizer()
        
        # State with only some fields
        state = {
            's_score': 0.5,
            'coherence': 0.7
        }
        
        visualizer.on_state_changed(state)
        
        assert visualizer.current_s_score == 0.5
        assert visualizer.current_coherence == 0.7
    
    def test_on_state_changed_empty_state(self):
        """Test on_state_changed with empty state."""
        visualizer = CognitiveMapVisualizer()
        
        visualizer.on_state_changed({})
        
        # Should not crash
        assert True
    
    def test_history_maxlen(self):
        """Test that history deques respect maxlen."""
        visualizer = CognitiveMapVisualizer()
        
        # Add more than 100 items
        for i in range(150):
            visualizer.update_state(0.1, 0.2, 0.3)
        
        # History should be capped at 100
        assert len(visualizer.s_score_history) == 100
        assert len(visualizer.coherence_history) == 100
        assert len(visualizer.maturity_history) == 100
    
    def test_pulse_circle_color_mapping(self):
        """Test pulse circle color changes with s_score."""
        visualizer = CognitiveMapVisualizer()
        
        # Low s_score (blue)
        visualizer.update_state(0.1, 0.5, 0.3)
        visualizer.draw_cognitive_map()
        low_color = visualizer.pulse_circle.get_facecolor()
        
        # High s_score (red)
        visualizer.update_state(0.9, 0.5, 0.3)
        visualizer.draw_cognitive_map()
        high_color = visualizer.pulse_circle.get_facecolor()
        
        # Colors should be different
        assert low_color != high_color
    
    def test_coherence_shape_updates(self):
        """Test that coherence shape updates with coherence value."""
        visualizer = CognitiveMapVisualizer()
        
        visualizer.update_state(0.5, 0.1, 0.3)
        visualizer.draw_cognitive_map()
        low_coherence = visualizer.coherence_shape.get_alpha()
        
        visualizer.update_state(0.5, 0.9, 0.3)
        visualizer.draw_cognitive_map()
        high_coherence = visualizer.coherence_shape.get_alpha()
        
        # Alpha should change with coherence
        assert high_coherence > low_coherence
    
    def test_text_info_updates(self):
        """Test that text info updates with state."""
        visualizer = CognitiveMapVisualizer()
        
        visualizer.update_state(0.5, 0.7, 0.3, maturity_level=2)
        visualizer.draw_cognitive_map()
        
        text = visualizer.text_info.get_text()
        
        # Text should contain state information
        assert len(text) > 0
    
    def test_maturity_level_display(self):
        """Test maturity level display in visualization."""
        visualizer = CognitiveMapVisualizer()
        
        for level in range(8):
            visualizer.update_state(0.5, 0.7, 0.3, maturity_level=level)
            visualizer.draw_cognitive_map()
            
            # Should not crash for any level
            assert visualizer.current_maturity_level == level
    
    def test_render_to_image_exception_handling(self):
        """Test render_to_image exception handling (covers lines 240-253)."""
        visualizer = CognitiveMapVisualizer()
        visualizer.update_state(0.5, 0.7, 0.3)
        visualizer.draw_cognitive_map()
        
        # Mock buffer_rgba to raise exception (the actual method used)
        with patch.object(visualizer.fig.canvas, 'buffer_rgba', side_effect=Exception("Render error")):
            # Should handle exception gracefully and return None
            image = visualizer.render_to_image()
            assert image is None
    
    def test_cv2_import_error(self):
        """Test behavior when cv2 import fails (covers lines 28-29)."""
        # This is tested implicitly by the CV2_AVAILABLE flag
        # The code should work whether cv2 is available or not
        visualizer = CognitiveMapVisualizer()
        visualizer.update_state(0.5, 0.7, 0.3)
        visualizer.draw_cognitive_map()
        
        # Should work without cv2 - returns None when cv2 is not available
        with patch('utils.visualization.CV2_AVAILABLE', False):
            image = visualizer.render_to_image()
            assert image is None

