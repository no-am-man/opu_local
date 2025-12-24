"""
Decorator Pattern: Sense Processing Layers.

Allows composable preprocessing/filtering of sense input through decorators.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import numpy as np
from core.patterns.sense_factory import Sense
from config import (
    DECORATOR_NOISE_GATE_THRESHOLD, DECORATOR_HIGHPASS_ALPHA,
    DECORATOR_AMPLIFICATION_GAIN
)


class SenseDecorator(Sense):
    """Decorator base class for sense processing."""
    
    def __init__(self, wrapped_sense: Sense):
        """
        Initialize sense decorator.
        
        Args:
            wrapped_sense: The sense to wrap
        """
        self._wrapped = wrapped_sense
    
    def perceive(self, raw_input: Any) -> Dict[str, Any]:
        """
        Perceive with preprocessing and postprocessing.
        
        Args:
            raw_input: Raw input data
            
        Returns:
            dict with perception results
        """
        # Preprocess
        processed = self.preprocess(raw_input)
        
        # Delegate to wrapped sense
        result = self._wrapped.perceive(processed)
        
        # Postprocess
        return self.postprocess(result)
    
    def preprocess(self, raw_input: Any) -> Any:
        """
        Override to add preprocessing.
        
        Args:
            raw_input: Raw input
            
        Returns:
            Preprocessed input
        """
        return raw_input
    
    def postprocess(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override to add postprocessing.
        
        Args:
            result: Perception result
            
        Returns:
            Postprocessed result
        """
        return result
    
    def get_label(self) -> str:
        """Get sense label from wrapped sense."""
        return self._wrapped.get_label()


class NoiseGateDecorator(SenseDecorator):
    """Filters out low-amplitude noise."""
    
    def __init__(self, wrapped_sense: Sense, threshold: float = DECORATOR_NOISE_GATE_THRESHOLD):
        """
        Initialize noise gate decorator.
        
        Args:
            wrapped_sense: Sense to wrap
            threshold: Amplitude threshold below which input is silenced
        """
        super().__init__(wrapped_sense)
        self.threshold = threshold
    
    def preprocess(self, raw_input):
        """Apply noise gate."""
        if isinstance(raw_input, np.ndarray):
            max_amplitude = np.max(np.abs(raw_input))
            if max_amplitude < self.threshold:
                return np.zeros_like(raw_input)
        return raw_input


class NormalizationDecorator(SenseDecorator):
    """Normalizes input amplitude."""
    
    def preprocess(self, raw_input):
        """Normalize input amplitude."""
        if isinstance(raw_input, np.ndarray):
            max_val = np.max(np.abs(raw_input))
            if max_val > 0:
                return raw_input / max_val
        return raw_input


class HighPassFilterDecorator(SenseDecorator):
    """Simple high-pass filter to remove DC offset."""
    
    def __init__(self, wrapped_sense: Sense, alpha: float = DECORATOR_HIGHPASS_ALPHA):
        """
        Initialize high-pass filter.
        
        Args:
            wrapped_sense: Sense to wrap
            alpha: Filter coefficient (0-1, higher = more filtering)
        """
        super().__init__(wrapped_sense)
        self.alpha = alpha
        self._prev_output = None
    
    def preprocess(self, raw_input):
        """Apply high-pass filter."""
        if isinstance(raw_input, np.ndarray):
            if self._prev_output is None:
                self._prev_output = np.zeros_like(raw_input)
            
            # Simple high-pass: y[n] = alpha * (y[n-1] + x[n] - x[n-1])
            filtered = self.alpha * (self._prev_output + raw_input - np.roll(raw_input, 1))
            self._prev_output = filtered
            return filtered
        return raw_input


class AmplificationDecorator(SenseDecorator):
    """Amplifies input signal."""
    
    def __init__(self, wrapped_sense: Sense, gain: float = DECORATOR_AMPLIFICATION_GAIN):
        """
        Initialize amplification decorator.
        
        Args:
            wrapped_sense: Sense to wrap
            gain: Amplification gain (>1.0 = amplify, <1.0 = attenuate)
        """
        super().__init__(wrapped_sense)
        self.gain = gain
    
    def preprocess(self, raw_input):
        """Apply amplification."""
        if isinstance(raw_input, np.ndarray):
            return raw_input * self.gain
        return raw_input

