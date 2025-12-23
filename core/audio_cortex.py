"""
The Audio Cortex: Audio Introspection.
Calculates surprise scores from audio genomic bits.
"""

import numpy as np


class AudioCortex:
    """
    The Audio Cortex: Processes audio genomic bits and calculates surprise.
    Tracks audio history and computes s_score (surprise) through introspection.
    """
    
    def __init__(self, max_history_size=50):
        """
        Initialize Audio Cortex.
        
        Args:
            max_history_size: Maximum number of genomic bits to keep in history
            (Reduced from 10000 to 50 for higher sensitivity - shorter memory = more reactive)
        """
        # History for introspection (capped to prevent unbounded growth)
        self.max_history_size = max_history_size
        self.mu_history = []  # Mean of historical genomic bits
        self.sigma_history = []  # Std dev of historical genomic bits
        self.genomic_bits_history = []  # Raw genomic bits
        
        # Current state
        self.g_now = None
        self.s_score = 0.0
        self.coherence = 0.0
    
    def introspect(self, genomic_bit):
        """
        Calculates surprise score (s_score) through introspection.
        
        Formula: s_score = |g_now - mu_history| / sigma_history
        
        Args:
            genomic_bit: current genomic bit (standard deviation from perception)
            
        Returns:
            s_score: surprise score (higher = more surprising)
        """
        # Update current genomic bit immediately
        self.g_now = float(genomic_bit) if genomic_bit is not None else 0.0
        
        # Cap histories BEFORE appending to ensure we never exceed max_history_size
        if len(self.genomic_bits_history) >= self.max_history_size:
            # Remove oldest entry to make room
            self.genomic_bits_history.pop(0)
            if len(self.mu_history) > 0:
                self.mu_history.pop(0)
            if len(self.sigma_history) > 0:
                self.sigma_history.pop(0)
        
        # Append current genomic bit
        self.genomic_bits_history.append(self.g_now)
        
        # Need at least 2 data points for meaningful introspection
        if len(self.genomic_bits_history) < 2:
            self.s_score = 0.0
            self.coherence = 1.0  # Perfect coherence when no history
            return self.s_score
        
        # Calculate historical statistics from ALL history (not just mu/sigma history)
        history_array = np.array(self.genomic_bits_history, dtype=np.float64)
        mu_history = float(np.mean(history_array))
        sigma_history = float(np.std(history_array))
        
        # --- FIX: RAISE NOISE FLOOR TO PREVENT "TRAUMA LEARNING" ---
        # A perfectly silent room has sigma=0. We enforce a minimum noise floor.
        # 0.0001 was too sensitive - it caused tiny glitches to generate s_score > 5.0,
        # which triggered "Trauma Evolution" (jumping directly to Level 5).
        # 0.01 prevents false high scores from silence while still allowing real surprises.
        if sigma_history < 0.01:
            sigma_history = 0.01
        
        # Store for later use (these are for backward compatibility)
        self.mu_history.append(mu_history)
        self.sigma_history.append(sigma_history)
        
        # Calculate surprise score (Z-score formula)
        # Now safe because sigma_history is guaranteed >= 0.0001
        self.s_score = float(abs(self.g_now - mu_history) / sigma_history)
        
        # Calculate coherence (inverse of surprise, normalized)
        self.coherence = float(1.0 / (1.0 + self.s_score))
        
        # Ensure s_score is never negative (safety check)
        if self.s_score < 0:
            self.s_score = 0.0
        
        return self.s_score
    
    def get_state(self):
        """
        Returns current audio cortex state.
        
        Returns:
            dict with s_score, coherence, g_now
        """
        return {
            's_score': self.s_score,
            'coherence': self.coherence,
            'g_now': self.g_now
        }

