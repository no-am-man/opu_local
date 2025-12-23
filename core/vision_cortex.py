"""
The Visual Cortex: Visual Introspection.
Calculates surprise scores from visual genomic vectors (R, G, B channels).

This is INTROSPECTION (like audio_cortex.py for audio).
For PERCEPTION (camera capture), see vision.py.
"""

import numpy as np


class VisualCortex:
    """
    The Visual Cortex: Visual Introspection.
    
    This module handles INTROSPECTION - calculating surprise scores from
    visual genomic vectors. It's the visual equivalent of audio_cortex.py.
    
    For PERCEPTION (camera capture), see vision.py (VisualPerception).
    
    Processes visual genomic vectors and calculates surprise.
    Tracks R, G, B channel history independently and computes visual surprise.
    """
    
    def __init__(self, max_history=50):
        """
        Initialize Visual Cortex Introspection.
        
        Args:
            max_history: Maximum number of frames to keep in history per channel
            (Reduced from 100 to 50 for higher sensitivity - shorter memory = more reactive)
        """
        # Visual memory for each channel independently
        # R, G, B are treated as orthogonal information streams
        self.visual_memory = {
            'R': [],
            'G': [],
            'B': []
        }
        self.max_visual_history = max_history
        self.visual_stats = {
            'mu': np.zeros(3),  # Mean for [R, G, B]
            'sigma': np.ones(3)  # Std dev for [R, G, B]
        }
    
    def introspect(self, visual_vector):
        """
        Calculates Visual Surprise (S_visual) for R, G, B channels.
        
        Logic:
        1. Compare current Channel Entropy against historical baseline.
        2. S_visual = Max(Z-Score of R, G, B).
        
        The OPU is "visually surprised" if any color channel deviates
        significantly from its learned baseline.
        
        Args:
            visual_vector: np.array([sigma_r, sigma_g, sigma_b])
            
        Returns:
            s_visual: The highest surprise score found across the 3 channels
            channel_surprises: dict with individual channel scores {'R': float, 'G': float, 'B': float}
        """
        channels = ['R', 'G', 'B']
        channel_surprises = {}
        
        for i, channel in enumerate(channels):
            g_now = visual_vector[i]
            mem = self.visual_memory[channel]
            
            # 1. Add to Short Term Memory
            mem.append(g_now)
            if len(mem) > self.max_visual_history:
                mem.pop(0)
            
            # Need history to judge surprise (at least 10 frames)
            if len(mem) < 10:
                channel_surprises[channel] = 0.0
                continue
                
            # 2. Calculate Baseline (Normalcy)
            # What does "Red" usually look like in this room?
            mu_history = np.mean(mem)
            sigma_history = np.std(mem)
            
            # Prevent divide by zero
            if sigma_history == 0:
                sigma_history = 0.1
            
            # 3. Calculate Z-Score (Surprise)
            # Same formula as audio introspection: |g_now - mu| / sigma
            s_channel = abs(g_now - mu_history) / sigma_history
            channel_surprises[channel] = s_channel

        # 4. SENSORY FUSION
        # The "Visual Score" is the maximum surprise found in any channel.
        # If the scene is mostly static (Low G, Low B) but a red laser appears (High R),
        # the OPU should be Surprised.
        if not channel_surprises:
            return 0.0, {'R': 0.0, 'G': 0.0, 'B': 0.0}
            
        s_visual = max(channel_surprises.values())
        return s_visual, channel_surprises

