"""
Strategy Pattern: Introspection Strategies.

Abstract strategy interface for calculating surprise scores from genomic input.
Allows easy extension to new introspection types (tactile, temperature, etc.).
"""

from abc import ABC, abstractmethod
import numpy as np


class IntrospectionStrategy(ABC):
    """Abstract strategy for calculating surprise scores."""
    
    @abstractmethod
    def introspect(self, genomic_input):
        """
        Calculate surprise score from genomic input.
        
        Args:
            genomic_input: The genomic bit/vector to introspect on
            
        Returns:
            surprise_score: The calculated surprise score
        """
        pass
    
    @abstractmethod
    def get_state(self):
        """
        Get current introspection state.
        
        Returns:
            dict with current state information
        """
        pass


class AudioIntrospectionStrategy(IntrospectionStrategy):
    """Strategy for audio introspection."""
    
    def __init__(self, max_history_size=50):
        """Initialize audio introspection strategy.
        
        Default reduced from 10000 to 50 for higher sensitivity.
        """
        self.max_history_size = max_history_size
        self.mu_history = []
        self.sigma_history = []
        self.genomic_bits_history = []
        self.g_now = None
        self.s_score = 0.0
        self.coherence = 0.0
    
    def introspect(self, genomic_bit):
        """
        Calculate audio surprise score.
        
        Matches AudioCortex implementation exactly for backward compatibility.
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
        # Now safe because sigma_history is guaranteed >= 0.01
        self.s_score = float(abs(self.g_now - mu_history) / sigma_history)
        
        # Calculate coherence (inverse of surprise, normalized)
        self.coherence = float(1.0 / (1.0 + self.s_score))
        
        # Ensure s_score is never negative (safety check)
        if self.s_score < 0:
            self.s_score = 0.0
        
        return self.s_score
    
    def get_state(self):
        """Get current audio introspection state."""
        return {
            's_score': self.s_score,
            'coherence': self.coherence,
            'g_now': self.g_now
        }


class VisualIntrospectionStrategy(IntrospectionStrategy):
    """Strategy for visual introspection."""
    
    def __init__(self, max_history=50):
        """Initialize visual introspection strategy.
        
        Default reduced from 100 to 50 for higher sensitivity.
        """
        self.visual_memory = {
            'R': [],
            'G': [],
            'B': []
        }
        self.max_visual_history = max_history
        self.visual_stats = {
            'mu': np.zeros(3),
            'sigma': np.ones(3)
        }
    
    def introspect(self, visual_vector):
        """
        Calculate visual surprise score.
        
        Matches VisualCortex implementation exactly for backward compatibility.
        Returns tuple (s_visual, channel_surprises) to match VisualCortex API.
        
        Args:
            visual_vector: np.array([sigma_r, sigma_g, sigma_b])
            
        Returns:
            tuple: (s_visual, channel_surprises_dict)
                - s_visual: The highest surprise score found across the 3 channels
                - channel_surprises: dict with individual channel scores {'R': float, 'G': float, 'B': float}
        """
        return self.introspect_with_channels(visual_vector)
    
    def introspect_with_channels(self, visual_vector):
        """
        Calculate visual surprise with channel breakdown.
        
        Matches VisualCortex implementation exactly for backward compatibility.
        
        Returns:
            tuple: (s_visual, channel_surprises_dict)
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
    
    def get_state(self):
        """Get current visual introspection state."""
        return {
            'visual_memory': self.visual_memory,
            'visual_stats': self.visual_stats
        }

