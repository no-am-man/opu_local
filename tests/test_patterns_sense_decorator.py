"""
Tests for core/patterns/sense_decorator.py - Decorator Pattern
100% code coverage target
"""

import pytest
import numpy as np
from core.patterns.sense_decorator import (
    SenseDecorator,
    NoiseGateDecorator,
    NormalizationDecorator,
    HighPassFilterDecorator,
    AmplificationDecorator
)
from core.patterns.sense_factory import AudioSense


class TestSenseDecorator:
    """Test suite for SenseDecorator base class."""
    
    def test_init(self):
        """Test SenseDecorator initialization."""
        base_sense = AudioSense()
        decorator = SenseDecorator(base_sense)
        assert decorator._wrapped == base_sense
    
    def test_perceive_delegates(self):
        """Test that perceive delegates to wrapped sense."""
        base_sense = AudioSense()
        decorator = SenseDecorator(base_sense)
        audio_input = np.random.randn(1024).astype(np.float32)
        result = decorator.perceive(audio_input)
        assert 'genomic_bit' in result
    
    def test_get_label_delegates(self):
        """Test that get_label delegates to wrapped sense."""
        base_sense = AudioSense()
        decorator = SenseDecorator(base_sense)
        assert decorator.get_label() == base_sense.get_label()


class TestNoiseGateDecorator:
    """Test suite for NoiseGateDecorator."""
    
    def test_init(self):
        """Test NoiseGateDecorator initialization."""
        base_sense = AudioSense()
        decorator = NoiseGateDecorator(base_sense, threshold=0.1)
        assert decorator.threshold == 0.1
    
    def test_preprocess_below_threshold(self):
        """Test preprocessing with amplitude below threshold."""
        base_sense = AudioSense()
        decorator = NoiseGateDecorator(base_sense, threshold=0.1)
        # Low amplitude input - ensure max is definitely below threshold
        low_input = np.ones(1024, dtype=np.float32) * 0.05
        result = decorator.preprocess(low_input)
        # Should be silenced (all zeros) since max amplitude (0.05) < threshold (0.1)
        assert np.allclose(result, np.zeros_like(low_input))
    
    def test_preprocess_above_threshold(self):
        """Test preprocessing with amplitude above threshold."""
        base_sense = AudioSense()
        decorator = NoiseGateDecorator(base_sense, threshold=0.1)
        # High amplitude input
        high_input = np.random.randn(1024).astype(np.float32) * 0.5
        result = decorator.preprocess(high_input)
        # Should pass through
        np.testing.assert_array_equal(result, high_input)
    
    def test_preprocess_non_array(self):
        """Test preprocessing with non-array input."""
        base_sense = AudioSense()
        decorator = NoiseGateDecorator(base_sense, threshold=0.1)
        result = decorator.preprocess("not an array")
        assert result == "not an array"


class TestNormalizationDecorator:
    """Test suite for NormalizationDecorator."""
    
    def test_preprocess_normalizes(self):
        """Test that preprocessing normalizes input."""
        base_sense = AudioSense()
        decorator = NormalizationDecorator(base_sense)
        # Input with varying amplitude
        input_data = np.random.randn(1024).astype(np.float32) * 2.0
        result = decorator.preprocess(input_data)
        # Max absolute value should be 1.0
        assert np.max(np.abs(result)) <= 1.0 + 1e-6
    
    def test_preprocess_zero_max(self):
        """Test preprocessing with zero max value."""
        base_sense = AudioSense()
        decorator = NormalizationDecorator(base_sense)
        zero_input = np.zeros(1024, dtype=np.float32)
        result = decorator.preprocess(zero_input)
        np.testing.assert_array_equal(result, zero_input)
    
    def test_preprocess_non_array(self):
        """Test preprocessing with non-array input."""
        base_sense = AudioSense()
        decorator = NormalizationDecorator(base_sense)
        result = decorator.preprocess("not an array")
        assert result == "not an array"


class TestHighPassFilterDecorator:
    """Test suite for HighPassFilterDecorator."""
    
    def test_init(self):
        """Test HighPassFilterDecorator initialization."""
        base_sense = AudioSense()
        decorator = HighPassFilterDecorator(base_sense, alpha=0.95)
        assert decorator.alpha == 0.95
        assert decorator._prev_output is None
    
    def test_preprocess_first_call(self):
        """Test preprocessing on first call."""
        base_sense = AudioSense()
        decorator = HighPassFilterDecorator(base_sense, alpha=0.95)
        input_data = np.random.randn(1024).astype(np.float32)
        result = decorator.preprocess(input_data)
        assert decorator._prev_output is not None
        assert len(result) == len(input_data)
    
    def test_preprocess_subsequent_calls(self):
        """Test preprocessing on subsequent calls."""
        base_sense = AudioSense()
        decorator = HighPassFilterDecorator(base_sense, alpha=0.95)
        input_data = np.random.randn(1024).astype(np.float32)
        result1 = decorator.preprocess(input_data)
        result2 = decorator.preprocess(input_data)
        # Results should be different due to filtering
        assert not np.allclose(result1, result2)
    
    def test_preprocess_non_array(self):
        """Test preprocessing with non-array input."""
        base_sense = AudioSense()
        decorator = HighPassFilterDecorator(base_sense, alpha=0.95)
        result = decorator.preprocess("not an array")
        assert result == "not an array"


class TestAmplificationDecorator:
    """Test suite for AmplificationDecorator."""
    
    def test_init(self):
        """Test AmplificationDecorator initialization."""
        base_sense = AudioSense()
        decorator = AmplificationDecorator(base_sense, gain=2.0)
        assert decorator.gain == 2.0
    
    def test_preprocess_amplifies(self):
        """Test that preprocessing amplifies input."""
        base_sense = AudioSense()
        decorator = AmplificationDecorator(base_sense, gain=2.0)
        input_data = np.random.randn(1024).astype(np.float32) * 0.5
        result = decorator.preprocess(input_data)
        # Should be amplified by gain
        np.testing.assert_array_almost_equal(result, input_data * 2.0)
    
    def test_preprocess_attenuates(self):
        """Test that preprocessing can attenuate input."""
        base_sense = AudioSense()
        decorator = AmplificationDecorator(base_sense, gain=0.5)
        input_data = np.random.randn(1024).astype(np.float32)
        result = decorator.preprocess(input_data)
        # Should be attenuated by gain
        np.testing.assert_array_almost_equal(result, input_data * 0.5)
    
    def test_preprocess_non_array(self):
        """Test preprocessing with non-array input."""
        base_sense = AudioSense()
        decorator = AmplificationDecorator(base_sense, gain=2.0)
        result = decorator.preprocess("not an array")
        assert result == "not an array"


class TestDecoratorChaining:
    """Test suite for chaining decorators."""
    
    def test_chain_decorators(self):
        """Test chaining multiple decorators."""
        base_sense = AudioSense()
        decorated = AmplificationDecorator(
            NormalizationDecorator(
                NoiseGateDecorator(base_sense, threshold=0.1)
            )
        )
        audio_input = np.random.randn(1024).astype(np.float32) * 0.5
        result = decorated.perceive(audio_input)
        assert 'genomic_bit' in result
    
    def test_decorator_label_preserved(self):
        """Test that decorator preserves sense label."""
        base_sense = AudioSense()
        decorated = NoiseGateDecorator(base_sense, threshold=0.1)
        assert decorated.get_label() == base_sense.get_label()

