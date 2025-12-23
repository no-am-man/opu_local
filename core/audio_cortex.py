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
    
    def __init__(self, max_history_size=10000):
        """
        Initialize Audio Cortex.
        
        Args:
            max_history_size: Maximum number of genomic bits to keep in history
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
        self.g_now = genomic_bit
        
        # Cap histories BEFORE appending to ensure we never exceed max_history_size
        if len(self.genomic_bits_history) >= self.max_history_size:
            # Remove oldest entry to make room
            self.genomic_bits_history.pop(0)
            if len(self.mu_history) > 0:
                self.mu_history.pop(0)
            if len(self.sigma_history) > 0:
                self.sigma_history.pop(0)
        
        self.genomic_bits_history.append(genomic_bit)
        
        # Need at least 2 data points for meaningful introspection
        if len(self.genomic_bits_history) < 2:
            self.s_score = 0.0
            return self.s_score
        
        # Calculate historical statistics
        history_array = np.array(self.genomic_bits_history)
        mu_history = np.mean(history_array)
        sigma_history = np.std(history_array)
        
        # Store for later use
        self.mu_history.append(mu_history)
        self.sigma_history.append(sigma_history)
        
        # Calculate surprise score
        if sigma_history > 0:
            self.s_score = abs(genomic_bit - mu_history) / sigma_history
        else:
            self.s_score = abs(genomic_bit - mu_history) if mu_history > 0 else abs(genomic_bit)
        
        # Calculate coherence (inverse of surprise, normalized)
        self.coherence = 1.0 / (1.0 + self.s_score)
        
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

