"""
The Audio Cortex: Audio Introspection.
Calculates surprise scores from audio genomic bits.

Now uses Strategy Pattern internally for better extensibility.
"""

import numpy as np
from core.patterns.introspection_strategy import AudioIntrospectionStrategy


class AudioCortex:
    """
    The Audio Cortex: Processes audio genomic bits and calculates surprise.
    Tracks audio history and computes s_score (surprise) through introspection.
    """
    
    def __init__(self, max_history_size=50):
        """
        Initialize Audio Cortex.
        
        Uses Strategy Pattern internally for introspection.
        
        Args:
            max_history_size: Maximum number of genomic bits to keep in history
            (Reduced from 10000 to 50 for higher sensitivity - shorter memory = more reactive)
        """
        # Use Strategy Pattern for introspection
        self._strategy = AudioIntrospectionStrategy(max_history_size=max_history_size)
        
        # Expose strategy properties for backward compatibility
        self.max_history_size = max_history_size
    
    def introspect(self, genomic_bit):
        """
        Calculates surprise score (s_score) through introspection.
        
        Delegates to AudioIntrospectionStrategy (Strategy Pattern).
        
        Formula: s_score = |g_now - mu_history| / sigma_history
        
        Args:
            genomic_bit: current genomic bit (standard deviation from perception)
            
        Returns:
            s_score: surprise score (higher = more surprising)
        """
        # Delegate to strategy
        return self._strategy.introspect(genomic_bit)
    
    def get_state(self):
        """
        Returns current audio cortex state.
        
        Delegates to strategy.
        
        Returns:
            dict with s_score, coherence, g_now
        """
        return self._strategy.get_state()
    
    # Properties for backward compatibility (delegate to strategy)
    @property
    def g_now(self):
        """Current genomic bit (delegates to strategy)."""
        return self._strategy.g_now
    
    @g_now.setter
    def g_now(self, value):
        """Set current genomic bit (for persistence)."""
        self._strategy.g_now = value
    
    @property
    def s_score(self):
        """Current surprise score (delegates to strategy)."""
        return self._strategy.s_score
    
    @s_score.setter
    def s_score(self, value):
        """Set surprise score (for persistence)."""
        self._strategy.s_score = value
    
    @property
    def coherence(self):
        """Current coherence (delegates to strategy)."""
        return self._strategy.coherence
    
    @coherence.setter
    def coherence(self, value):
        """Set coherence (for persistence)."""
        self._strategy.coherence = value
    
    @property
    def mu_history(self):
        """Mu history (delegates to strategy)."""
        return self._strategy.mu_history
    
    @mu_history.setter
    def mu_history(self, value):
        """Set mu history (for persistence)."""
        self._strategy.mu_history = value
    
    @property
    def sigma_history(self):
        """Sigma history (delegates to strategy)."""
        return self._strategy.sigma_history
    
    @sigma_history.setter
    def sigma_history(self, value):
        """Set sigma history (for persistence)."""
        self._strategy.sigma_history = value
    
    @property
    def genomic_bits_history(self):
        """Genomic bits history (delegates to strategy)."""
        return self._strategy.genomic_bits_history
    
    @genomic_bits_history.setter
    def genomic_bits_history(self, value):
        """Set genomic bits history (for persistence)."""
        self._strategy.genomic_bits_history = value

