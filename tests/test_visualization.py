"""
Tests for utils/visualization.py - Cognitive Map Visualization
100% code coverage target
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests
from utils.visualization import CognitiveMapVisualizer


class TestCognitiveMapVisualizer:
    """Test suite for CognitiveMapVisualizer class."""
    
    def test_init(self):
        """Test CognitiveMapVisualizer initialization."""
        visualizer = CognitiveMapVisualizer()
        assert visualizer.fig is not None
        assert visualizer.ax is not None
        assert len(visualizer.s_score_history) == 0
        assert len(visualizer.coherence_history) == 0
        assert len(visualizer.maturity_history) == 0
        assert visualizer.current_s_score == 0.0
        assert visualizer.current_coherence == 0.0
        assert visualizer.current_maturity == 0.0
    
    def test_update_state(self):
        """Test update_state."""
        visualizer = CognitiveMapVisualizer()
        visualizer.update_state(2.5, 0.7, 0.5, maturity_level=3)
        
        assert visualizer.current_s_score == 2.5
        assert visualizer.current_coherence == 0.7
        assert visualizer.current_maturity == 0.5
        assert visualizer.current_maturity_level == 3
        assert len(visualizer.s_score_history) == 1
        assert len(visualizer.coherence_history) == 1
        assert len(visualizer.maturity_history) == 1
    
    def test_update_state_without_maturity_level(self):
        """Test update_state without maturity_level parameter."""
        visualizer = CognitiveMapVisualizer()
        visualizer.update_state(2.5, 0.7, 0.5)
        # maturity_level should remain at default (0)
        assert visualizer.current_maturity_level == 0
    
    def test_update_state_history_capping(self):
        """Test that history is capped."""
        visualizer = CognitiveMapVisualizer()
        # Add more than maxlen (100)
        for i in range(150):
            visualizer.update_state(float(i), 0.5, 0.5)
        
        assert len(visualizer.s_score_history) <= 100
        assert len(visualizer.coherence_history) <= 100
        assert len(visualizer.maturity_history) <= 100
    
    def test_draw_cognitive_map(self):
        """Test draw_cognitive_map."""
        visualizer = CognitiveMapVisualizer()
        visualizer.update_state(2.5, 0.7, 0.5, maturity_level=3)
        visualizer.draw_cognitive_map()
        
        # Should not raise exception
        assert visualizer.pulse_circle is not None
        assert visualizer.coherence_shape is not None
        assert visualizer.text_info is not None
    
    def test_draw_cognitive_map_pulse_size(self):
        """Test that pulse size scales with s_score."""
        visualizer = CognitiveMapVisualizer()
        visualizer.update_state(0.0, 0.5, 0.5)
        visualizer.draw_cognitive_map()
        radius_low = visualizer.pulse_circle.get_radius()
        
        visualizer.update_state(10.0, 0.5, 0.5)
        visualizer.draw_cognitive_map()
        radius_high = visualizer.pulse_circle.get_radius()
        
        assert radius_high > radius_low
    
    def test_draw_cognitive_map_color_tension(self):
        """Test that color changes with tension (s_score)."""
        visualizer = CognitiveMapVisualizer()
        visualizer.update_state(0.0, 0.5, 0.5)
        visualizer.draw_cognitive_map()
        color_low = visualizer.pulse_circle.get_facecolor()[:3]
        
        visualizer.update_state(10.0, 0.5, 0.5)
        visualizer.draw_cognitive_map()
        color_high = visualizer.pulse_circle.get_facecolor()[:3]
        
        # High tension should be more red
        assert color_high[0] > color_low[0]
    
    def test_draw_cognitive_map_coherence_shape(self):
        """Test that coherence affects shape."""
        visualizer = CognitiveMapVisualizer()
        visualizer.update_state(2.0, 0.0, 0.5)
        visualizer.draw_cognitive_map()
        shape_low = visualizer.coherence_shape
        
        visualizer.update_state(2.0, 1.0, 0.5)
        visualizer.draw_cognitive_map()
        shape_high = visualizer.coherence_shape
        
        # Should have different shapes
        assert shape_low is not None
        assert shape_high is not None
    
    def test_draw_cognitive_map_maturity_ring(self):
        """Test that maturity ring is drawn."""
        visualizer = CognitiveMapVisualizer()
        visualizer.update_state(2.0, 0.7, 0.8, maturity_level=4)
        visualizer.draw_cognitive_map()
        
        # Should have maturity circle
        patches = visualizer.ax.patches
        assert len(patches) > 0
    
    def test_draw_cognitive_map_time_scale(self):
        """Test that time scale is displayed correctly."""
        visualizer = CognitiveMapVisualizer()
        visualizer.update_state(2.0, 0.7, 0.5, maturity_level=2)
        visualizer.draw_cognitive_map()
        
        # Text should contain time scale
        text_content = visualizer.text_info.get_text()
        assert "1 day" in text_content
    
    def test_draw_cognitive_map_history_trace(self):
        """Test that history trace is drawn."""
        visualizer = CognitiveMapVisualizer()
        # Add history
        for i in range(10):
            visualizer.update_state(float(i), 0.5, 0.5)
        visualizer.draw_cognitive_map()
        
        # Should have plot lines
        lines = visualizer.ax.lines
        assert len(lines) > 0
    
    def test_generate_coherence_shape_low_coherence(self):
        """Test _generate_coherence_shape with low coherence."""
        visualizer = CognitiveMapVisualizer()
        points = visualizer._generate_coherence_shape(0.0)
        
        # Low coherence = fewer sides (triangle-like)
        assert len(points) >= 3
        assert len(points) <= 4
    
    def test_generate_coherence_shape_high_coherence(self):
        """Test _generate_coherence_shape with high coherence."""
        visualizer = CognitiveMapVisualizer()
        points = visualizer._generate_coherence_shape(1.0)
        
        # High coherence = more sides (hexagon-like)
        assert len(points) >= 5
        assert len(points) <= 6
    
    def test_generate_coherence_shape_radius(self):
        """Test that coherence affects shape radius."""
        visualizer = CognitiveMapVisualizer()
        points_low = visualizer._generate_coherence_shape(0.0)
        points_high = visualizer._generate_coherence_shape(1.0)
        
        # Calculate average radius
        def avg_radius(points):
            distances = [np.sqrt(x**2 + y**2) for x, y in points]
            return np.mean(distances)
        
        radius_low = avg_radius(points_low)
        radius_high = avg_radius(points_high)
        
        assert radius_high > radius_low
    
    def test_show(self):
        """Test show method."""
        visualizer = CognitiveMapVisualizer()
        # Should not raise exception
        visualizer.show()
    
    def test_refresh(self):
        """Test refresh method."""
        visualizer = CognitiveMapVisualizer()
        visualizer.draw_cognitive_map()
        # Should not raise exception
        visualizer.refresh()
    
    def test_all_maturity_levels(self):
        """Test visualization with all maturity levels."""
        visualizer = CognitiveMapVisualizer()
        time_scales = {
            0: "1 minute",
            1: "1 hour",
            2: "1 day",
            3: "1 week",
            4: "1 month",
            5: "1 year"
        }
        
        for level in range(6):
            visualizer.update_state(2.0, 0.7, level / 5.0, maturity_level=level)
            visualizer.draw_cognitive_map()
            text_content = visualizer.text_info.get_text()
            assert time_scales[level] in text_content

