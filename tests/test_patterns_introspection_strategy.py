"""
Tests for core/patterns/introspection_strategy.py - Strategy Pattern
100% code coverage target
"""

import pytest
import numpy as np
from core.patterns.introspection_strategy import (
    IntrospectionStrategy,
    AudioIntrospectionStrategy,
    VisualIntrospectionStrategy
)


class TestIntrospectionStrategy:
    """Test suite for abstract IntrospectionStrategy class."""
    
    def test_strategy_is_abstract(self):
        """Test that IntrospectionStrategy cannot be instantiated."""
        from abc import ABC
        assert issubclass(IntrospectionStrategy, ABC)
        # Cannot instantiate abstract class
        with pytest.raises(TypeError):
            IntrospectionStrategy()


class TestAudioIntrospectionStrategy:
    """Test suite for AudioIntrospectionStrategy."""
    
    def test_init(self):
        """Test AudioIntrospectionStrategy initialization."""
        strategy = AudioIntrospectionStrategy(max_history_size=1000)
        assert strategy.max_history_size == 1000
        assert len(strategy.genomic_bits_history) == 0
        assert strategy.s_score == 0.0
        assert strategy.coherence == 1.0  # Perfect coherence when no history
        assert strategy.g_now is None
    
    def test_introspect_single_value(self):
        """Test introspect with single value (no history)."""
        strategy = AudioIntrospectionStrategy()
        result = strategy.introspect(0.5)
        assert result == 0.0  # No surprise with only one data point
        assert strategy.g_now == 0.5
        assert len(strategy.genomic_bits_history) == 1
    
    def test_introspect_two_values(self):
        """Test introspect with two values."""
        strategy = AudioIntrospectionStrategy()
        strategy.introspect(0.5)
        result = strategy.introspect(1.5)
        assert result >= 0.0
        assert strategy.s_score >= 0.0
        assert strategy.coherence > 0.0
        assert len(strategy.genomic_bits_history) == 2
    
    def test_introspect_zero_sigma(self):
        """Test introspect when sigma is zero."""
        strategy = AudioIntrospectionStrategy()
        strategy.introspect(1.0)
        result = strategy.introspect(1.0)  # Same value
        assert result >= 0.0
        assert strategy.coherence > 0.0
    
    def test_introspect_history_capping(self):
        """Test that history is capped at max_history_size."""
        strategy = AudioIntrospectionStrategy(max_history_size=5)
        for i in range(10):
            strategy.introspect(float(i))
        assert len(strategy.genomic_bits_history) == 5
        # mu_history and sigma_history are only appended when there are >= 2 values
        # So with 10 calls, we get 9 mu_history entries, but capped at 5
        assert len(strategy.mu_history) <= 5
        assert len(strategy.sigma_history) <= 5
    
    def test_introspect_mu_history_zero(self):
        """Test introspect when mu_history is zero."""
        strategy = AudioIntrospectionStrategy()
        strategy.introspect(0.0)
        result = strategy.introspect(0.5)
        assert result >= 0.0
    
    def test_get_state(self):
        """Test get_state method."""
        strategy = AudioIntrospectionStrategy()
        strategy.introspect(0.5)
        state = strategy.get_state()
        assert 's_score' in state
        assert 'coherence' in state
        assert 'g_now' in state
        assert state['g_now'] == 0.5


class TestVisualIntrospectionStrategy:
    """Test suite for VisualIntrospectionStrategy."""
    
    def test_init(self):
        """Test VisualIntrospectionStrategy initialization."""
        strategy = VisualIntrospectionStrategy(max_history=10)
        assert strategy.max_visual_history == 10
        assert 'R' in strategy.visual_memory
        assert 'G' in strategy.visual_memory
        assert 'B' in strategy.visual_memory
    
    def test_introspect_insufficient_history(self):
        """Test introspect with insufficient history."""
        strategy = VisualIntrospectionStrategy()
        visual_vector = np.array([0.3, 0.4, 0.5])
        s_visual, channel_surprises = strategy.introspect(visual_vector)
        # Should return 0.0 with insufficient history
        assert s_visual == 0.0
        assert isinstance(channel_surprises, dict)
    
    def test_introspect_builds_history(self):
        """Test that introspect builds history."""
        strategy = VisualIntrospectionStrategy()
        visual_vector = np.array([0.3, 0.4, 0.5])
        for _ in range(10):  # Min is 5, but we'll use 10 to test capping
            strategy.introspect(visual_vector)
        assert len(strategy.visual_memory['R']) == 10
        assert len(strategy.visual_memory['G']) == 10
        assert len(strategy.visual_memory['B']) == 10
    
    def test_introspect_calculates_surprise(self):
        """Test that introspect calculates surprise correctly."""
        strategy = VisualIntrospectionStrategy()
        # Build history
        for i in range(15):
            strategy.introspect(np.array([0.3, 0.4, 0.5]))
        # Now introduce surprise
        s_visual, channel_surprises = strategy.introspect(np.array([1.0, 1.0, 1.0]))
        assert s_visual >= 0.0
        assert isinstance(channel_surprises, dict)
    
    def test_introspect_max_surprise(self):
        """Test that introspect returns max surprise across channels."""
        strategy = VisualIntrospectionStrategy()
        # Build history
        for i in range(15):
            strategy.introspect(np.array([0.3, 0.4, 0.5]))
        # R channel has high surprise, G and B are normal
        s_visual, channel_surprises = strategy.introspect(np.array([2.0, 0.4, 0.5]))
        assert s_visual > 0.0
        assert isinstance(channel_surprises, dict)
    
    def test_introspect_memory_capping(self):
        """Test that visual memory is capped."""
        strategy = VisualIntrospectionStrategy(max_history=5)
        visual_vector = np.array([0.3, 0.4, 0.5])
        for _ in range(10):
            strategy.introspect(visual_vector)
        assert len(strategy.visual_memory['R']) == 5
        assert len(strategy.visual_memory['G']) == 5
        assert len(strategy.visual_memory['B']) == 5
    
    def test_introspect_zero_sigma(self):
        """Test introspect when sigma is zero."""
        strategy = VisualIntrospectionStrategy()
        # Build history with constant values
        for _ in range(15):
            strategy.introspect(np.array([0.5, 0.5, 0.5]))
        # Should handle zero sigma gracefully
        s_visual, channel_surprises = strategy.introspect(np.array([0.5, 0.5, 0.5]))
        assert s_visual >= 0.0
        assert isinstance(channel_surprises, dict)
    
    def test_introspect_with_channels(self):
        """Test introspect_with_channels method."""
        strategy = VisualIntrospectionStrategy()
        # Build history
        for i in range(15):
            strategy.introspect(np.array([0.3, 0.4, 0.5]))
        s_visual, channel_scores = strategy.introspect_with_channels(np.array([1.0, 0.4, 0.5]))
        assert s_visual >= 0.0
        assert 'R' in channel_scores
        assert 'G' in channel_scores
        assert 'B' in channel_scores
        assert channel_scores['R'] >= 0.0
    
    def test_introspect_with_channels_insufficient_history(self):
        """Test introspect_with_channels with insufficient history."""
        strategy = VisualIntrospectionStrategy()
        s_visual, channel_scores = strategy.introspect_with_channels(np.array([0.3, 0.4, 0.5]))
        assert s_visual == 0.0
        assert channel_scores['R'] == 0.0
    
    def test_get_state(self):
        """Test get_state method."""
        strategy = VisualIntrospectionStrategy()
        state = strategy.get_state()
        assert 'visual_memory' in state
        assert 'visual_stats' in state

