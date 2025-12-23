"""
Tests for Visual Cortex (core/vision_cortex.py)
OPU v3.1 - Multi-Modal Integration
"""

import pytest
import numpy as np
from core.vision_cortex import VisualCortex
from core.opu import OrthogonalProcessingUnit


class TestVisualCortex:
    """Test suite for VisualCortex introspection."""
    
    def test_init(self):
        """Test VisualCortex initialization."""
        vc = VisualCortex()
        
        assert 'R' in vc.visual_memory
        assert 'G' in vc.visual_memory
        assert 'B' in vc.visual_memory
        assert vc.max_visual_history == 100
    
    def test_introspect_visual_initialization(self):
        """Test that visual memory is initialized correctly in OPU."""
        cortex = OrthogonalProcessingUnit()
        
        assert 'R' in cortex.visual_memory
        assert 'G' in cortex.visual_memory
        assert 'B' in cortex.visual_memory
        assert cortex.max_visual_history == 100
    
    def test_introspect_insufficient_history(self):
        """Test visual introspection with insufficient history."""
        vc = VisualCortex()
        
        # Not enough history yet
        visual_vector = np.array([10.0, 20.0, 30.0])
        s_visual, channel_scores = vc.introspect(visual_vector)
        
        assert s_visual == 0.0
        assert channel_scores['R'] == 0.0
        assert channel_scores['G'] == 0.0
        assert channel_scores['B'] == 0.0
    
    def test_introspect_builds_history(self):
        """Test that visual introspection builds history."""
        vc = VisualCortex()
        
        visual_vector = np.array([10.0, 20.0, 30.0])
        
        # Add enough frames to build history
        for _ in range(15):
            vc.introspect(visual_vector)
        
        assert len(vc.visual_memory['R']) == 15
        assert len(vc.visual_memory['G']) == 15
        assert len(vc.visual_memory['B']) == 15
    
    def test_introspect_visual_insufficient_history(self):
        """Test visual introspection with insufficient history (OPU wrapper)."""
        cortex = OrthogonalProcessingUnit()
        
        # Not enough history yet
        visual_vector = np.array([10.0, 20.0, 30.0])
        s_visual, channel_scores = cortex.introspect_visual(visual_vector)
        
        assert s_visual == 0.0
        assert channel_scores['R'] == 0.0
        assert channel_scores['G'] == 0.0
        assert channel_scores['B'] == 0.0
    
    def test_introspect_visual_builds_history(self):
        """Test that visual introspection builds history (OPU wrapper)."""
        cortex = OrthogonalProcessingUnit()
        
        visual_vector = np.array([10.0, 20.0, 30.0])
        
        # Add enough frames to build history
        for _ in range(15):
            cortex.introspect_visual(visual_vector)
        
        assert len(cortex.visual_memory['R']) == 15
        assert len(cortex.visual_memory['G']) == 15
        assert len(cortex.visual_memory['B']) == 15
    
    def test_introspect_visual_calculates_surprise(self):
        """Test visual introspection calculates surprise correctly."""
        cortex = OrthogonalProcessingUnit()
        
        # Build baseline history with consistent values
        baseline_vector = np.array([10.0, 20.0, 30.0])
        for _ in range(20):
            cortex.introspect_visual(baseline_vector)
        
        # Now introduce a surprising value (much higher)
        surprising_vector = np.array([100.0, 20.0, 30.0])  # High R channel
        s_visual, channel_scores = cortex.introspect_visual(surprising_vector)
        
        # R channel should have high surprise (10x increase from baseline)
        assert channel_scores['R'] > 3.0
        # G and B should have low surprise (unchanged)
        assert channel_scores['G'] < 1.0
        assert channel_scores['B'] < 1.0
        # Global surprise should be max (R channel)
        assert s_visual == channel_scores['R']
    
    def test_introspect_visual_max_surprise(self):
        """Test that visual introspection returns max surprise across channels."""
        cortex = OrthogonalProcessingUnit()
        
        # Build baseline
        baseline = np.array([10.0, 20.0, 30.0])
        for _ in range(20):
            cortex.introspect_visual(baseline)
        
        # Create surprise in G channel (highest)
        surprise = np.array([10.0, 200.0, 30.0])
        s_visual, channel_scores = cortex.introspect_visual(surprise)
        
        # Global should be max of all channels
        assert s_visual == max(channel_scores.values())
        assert s_visual == channel_scores['G']
    
    def test_introspect_visual_memory_capping(self):
        """Test that visual memory is capped to prevent unbounded growth."""
        cortex = OrthogonalProcessingUnit()
        
        visual_vector = np.array([10.0, 20.0, 30.0])
        
        # Add more frames than max_visual_history
        for _ in range(cortex.max_visual_history + 50):
            cortex.introspect_visual(visual_vector)
        
        # Memory should be capped
        assert len(cortex.visual_memory['R']) <= cortex.max_visual_history
        assert len(cortex.visual_memory['G']) <= cortex.max_visual_history
        assert len(cortex.visual_memory['B']) <= cortex.max_visual_history
    
    def test_introspect_visual_zero_sigma(self):
        """Test visual introspection handles zero sigma (constant values)."""
        cortex = OrthogonalProcessingUnit()
        
        # Build history with constant values (zero variance)
        constant_vector = np.array([10.0, 20.0, 30.0])
        for _ in range(20):
            cortex.introspect_visual(constant_vector)
        
        # Add another constant value (should not crash)
        s_visual, channel_scores = cortex.introspect_visual(constant_vector)
        
        # Should handle gracefully (sigma replaced with 0.1)
        assert s_visual >= 0.0
        assert all(score >= 0.0 for score in channel_scores.values())
    
    def test_introspect_visual_multiple_channels_surprise(self):
        """Test visual introspection with surprise in multiple channels."""
        cortex = OrthogonalProcessingUnit()
        
        # Build baseline
        baseline = np.array([10.0, 20.0, 30.0])
        for _ in range(20):
            cortex.introspect_visual(baseline)
        
        # Create surprise in R and B channels
        surprise = np.array([100.0, 20.0, 200.0])
        s_visual, channel_scores = cortex.introspect_visual(surprise)
        
        # Both R and B should have high surprise (10x and 6.67x increase)
        assert channel_scores['R'] > 3.0
        assert channel_scores['B'] > 3.0
        # G should be low
        assert channel_scores['G'] < 1.0
        # Global should be max
        assert s_visual == max(channel_scores.values())
    
    def test_introspect_visual_negative_values(self):
        """Test visual introspection with negative values (shouldn't happen but test anyway)."""
        cortex = OrthogonalProcessingUnit()
        
        # Build baseline
        baseline = np.array([10.0, 20.0, 30.0])
        for _ in range(20):
            cortex.introspect_visual(baseline)
        
        # Negative values (std dev should always be positive, but test abs())
        negative_vector = np.array([-50.0, 20.0, 30.0])
        s_visual, channel_scores = cortex.introspect_visual(negative_vector)
        
        # Should handle gracefully (abs() in calculation)
        assert s_visual >= 0.0
        assert channel_scores['R'] >= 0.0

