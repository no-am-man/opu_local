"""
The Visual Cortex: Visual Introspection.
Calculates surprise scores from visual genomic vectors (R, G, B channels).

This is INTROSPECTION (like audio_cortex.py for audio).
For PERCEPTION (camera capture), see vision.py.

Now uses Strategy Pattern internally for better extensibility.
"""

import numpy as np
from core.patterns.introspection_strategy import VisualIntrospectionStrategy
from config import INTROSPECTION_VISUAL_MAX_HISTORY


class VisualCortex:
    """
    The Visual Cortex: Visual Introspection.
    
    This module handles INTROSPECTION - calculating surprise scores from
    visual genomic vectors. It's the visual equivalent of audio_cortex.py.
    
    For PERCEPTION (camera capture), see vision.py (VisualPerception).
    
    Processes visual genomic vectors and calculates surprise.
    Tracks R, G, B channel history independently and computes visual surprise.
    """
    
    def __init__(self, max_history=INTROSPECTION_VISUAL_MAX_HISTORY):
        """
        Initialize Visual Cortex Introspection.
        
        Uses Strategy Pattern internally for introspection.
        
        Args:
            max_history: Maximum number of frames to keep in history per channel
            (Reduced from 100 to 50 for higher sensitivity - shorter memory = more reactive)
        """
        # Use Strategy Pattern for introspection
        self._strategy = VisualIntrospectionStrategy(max_history=max_history)
        
        # Expose for backward compatibility
        self.max_visual_history = max_history
    
    def introspect(self, visual_vector):
        """
        Calculates Visual Surprise (S_visual) for R, G, B channels.
        
        Delegates to VisualIntrospectionStrategy (Strategy Pattern).
        
        Logic:
        1. Compare current Channel Entropy against historical baseline.
        2. S_visual = Max(Z-Score of R, G, B).
        
        The OPU is "visually surprised" if any color channel deviates
        significantly from its learned baseline.
        
        Args:
            visual_vector: np.array([sigma_r, sigma_g, sigma_b])
            
        Returns:
            tuple: (s_visual, channel_surprises)
                - s_visual: The highest surprise score found across the 3 channels
                - channel_surprises: dict with individual channel scores {'R': float, 'G': float, 'B': float}
        """
        # Delegate to strategy
        return self._strategy.introspect(visual_vector)
    
    # Properties for backward compatibility (delegate to strategy)
    @property
    def visual_memory(self):
        """Visual memory (delegates to strategy)."""
        return self._strategy.visual_memory
    
    @property
    def visual_stats(self):
        """Visual statistics (delegates to strategy)."""
        return self._strategy.visual_stats

