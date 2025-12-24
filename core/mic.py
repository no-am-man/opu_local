"""
The Ear: Microphone Capture Module.
Normalizes input to create a "Scale Invariant" signature.
Uses Standard Deviation as the "Genomic Bit".

Updates:
- Added Soft Limiter (Tanh) to prevent digital clipping distortion.

This is PERCEPTION (like camera.py for visual).
For INTROSPECTION, see audio_cortex.py.
"""

import numpy as np
from config import SCALE_INVARIANCE_ENABLED, MIC_DEFAULT_GENOMIC_BIT, MIC_DEFAULT_MAGNITUDE


def perceive(input_vector):
    """
    Perceives input vector with scale invariance and soft limiting.
    
    Logic: 
    1. Apply Soft Limiter (Tanh) to fix audio clipping.
    2. Normalize to create a "Scale Invariant" signature.
    3. Use Standard Deviation as the "Genomic Bit".
    
    Args:
        input_vector: numpy array of raw input data
        
    Returns:
        dict containing:
            - normalized: scale-invariant signature
            - genomic_bit: standard deviation (the "Genomic Bit")
            - magnitude: original magnitude
            - raw: soft-limited input vector
    """
    if _is_empty_input(input_vector):
        return _create_empty_result()
    
    input_vector = np.asarray(input_vector)
    soft_limited = _apply_soft_limiter(input_vector)
    genomic_bit = _calculate_genomic_bit(soft_limited)
    magnitude = _calculate_magnitude(soft_limited)
    normalized = _create_normalized_vector(soft_limited, magnitude)
    
    return {
        'normalized': normalized,
        'genomic_bit': genomic_bit,
        'magnitude': magnitude,
        'raw': soft_limited
    }


def _is_empty_input(input_vector):
    """Check if input vector is None or empty."""
    return input_vector is None or len(input_vector) == 0


def _create_empty_result():
    """Create empty result dictionary with default values."""
    return {
        'normalized': np.array([]),
        'genomic_bit': MIC_DEFAULT_GENOMIC_BIT,
        'magnitude': MIC_DEFAULT_MAGNITUDE,
        'raw': np.array([])
    }


def _apply_soft_limiter(input_vector):
    """
    Apply soft limiter (tanh) to prevent digital clipping.
    
    The hyperbolic tangent squashes values > 1.0 or < -1.0 smoothly,
    turning hard digital clipping into smooth analog saturation.
    This models biological eardrum behavior.
    
    Args:
        input_vector: numpy array of raw input data
        
    Returns:
        Soft-limited input vector (all values in [-1, 1] range)
    """
    return np.tanh(input_vector)


def _calculate_genomic_bit(input_vector):
    """
    Calculate the Genomic Bit (Standard Deviation).
    
    The Genomic Bit measures the variance/entropy of the input signal.
    
    Args:
        input_vector: numpy array (should be soft-limited)
        
    Returns:
        Standard deviation of the input vector
    """
    return np.std(input_vector)


def _calculate_magnitude(input_vector):
    """
    Calculate the magnitude (L2 norm) of the input vector.
    
    Args:
        input_vector: numpy array
        
    Returns:
        Magnitude (L2 norm) of the input vector
    """
    return np.linalg.norm(input_vector)


def _create_normalized_vector(input_vector, magnitude):
    """
    Create scale-invariant normalized vector.
    
    If scale invariance is enabled and magnitude > 0, normalizes to unit vector.
    Otherwise, returns a copy of the input vector.
    
    Args:
        input_vector: numpy array (should be soft-limited)
        magnitude: magnitude of the input vector
        
    Returns:
        Normalized vector (unit vector if scale invariance enabled, else copy)
    """
    if _should_normalize(magnitude):
        return input_vector / magnitude
    return input_vector.copy()


def _should_normalize(magnitude):
    """Check if normalization should be applied."""
    return SCALE_INVARIANCE_ENABLED and magnitude > 0

