"""
The Ear: Microphone Capture Module.
Normalizes input to create a "Scale Invariant" signature.
Uses Standard Deviation as the "Genomic Bit".

This is PERCEPTION (like camera.py for visual).
For INTROSPECTION, see audio_cortex.py.
"""

import numpy as np
from config import SCALE_INVARIANCE_ENABLED


def perceive(input_vector):
    """
    Perceives input vector with scale invariance.
    
    Logic: Normalize the input vector to create a "Scale Invariant" signature.
    Use Standard Deviation as the "Genomic Bit".
    
    Args:
        input_vector: numpy array of raw input data
        
    Returns:
        dict containing:
            - normalized: scale-invariant signature
            - genomic_bit: standard deviation (the "Genomic Bit")
            - magnitude: original magnitude
    """
    if input_vector is None or len(input_vector) == 0:
        return {
            'normalized': np.array([]),
            'genomic_bit': 0.0,
            'magnitude': 0.0
        }
    
    input_vector = np.asarray(input_vector)
    
    # Calculate the "Genomic Bit" (Standard Deviation)
    genomic_bit = np.std(input_vector)
    
    # Calculate magnitude
    magnitude = np.linalg.norm(input_vector)
    
    # Create scale-invariant signature
    if SCALE_INVARIANCE_ENABLED and magnitude > 0:
        # Normalize to unit vector (scale invariant)
        normalized = input_vector / magnitude
    else:
        normalized = input_vector.copy()
    
    return {
        'normalized': normalized,
        'genomic_bit': genomic_bit,
        'magnitude': magnitude,
        'raw': input_vector
    }

