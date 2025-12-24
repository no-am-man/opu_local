"""
Strategy Pattern: Introspection Strategies.

Abstract strategy interface for calculating surprise scores from genomic input.
Allows easy extension to new introspection types (tactile, temperature, etc.).
"""

from abc import ABC, abstractmethod
import numpy as np
from config import (
    INTROSPECTION_AUDIO_MAX_HISTORY, INTROSPECTION_VISUAL_MAX_HISTORY,
    INTROSPECTION_MIN_DATA_POINTS, INTROSPECTION_VISUAL_MIN_FRAMES,
    INTROSPECTION_NOISE_FLOOR, INTROSPECTION_SIGMA_DEFAULT,
    INTROSPECTION_DEFAULT_S_SCORE, INTROSPECTION_DEFAULT_COHERENCE,
    INTROSPECTION_DEFAULT_G_NOW
)


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
    
    def __init__(self, max_history_size=INTROSPECTION_AUDIO_MAX_HISTORY):
        """Initialize audio introspection strategy.
        
        Default reduced from 10000 to 50 for higher sensitivity.
        """
        self.max_history_size = max_history_size
        self.mu_history = []
        self.sigma_history = []
        self.genomic_bits_history = []
        self.g_now = None
        self.s_score = INTROSPECTION_DEFAULT_S_SCORE
        self.coherence = INTROSPECTION_DEFAULT_COHERENCE
    
    def introspect(self, genomic_bit):
        """
        Calculate audio surprise score.
        
        Matches AudioCortex implementation exactly for backward compatibility.
        """
        self.g_now = self._update_genomic_bit(genomic_bit)
        self._cap_and_append_history()
        
        if not self._has_sufficient_data():
            return self._get_default_surprise_score()
        
        mu_history, sigma_history = self._calculate_historical_statistics()
        sigma_history = self._apply_noise_floor(sigma_history)
        self._store_statistics(mu_history, sigma_history)
        self.s_score = self._calculate_surprise_score(mu_history, sigma_history)
        self.coherence = self._calculate_coherence()
        
        return self.s_score
    
    def _update_genomic_bit(self, genomic_bit):
        """Update current genomic bit from input."""
        return float(genomic_bit) if genomic_bit is not None else INTROSPECTION_DEFAULT_G_NOW
    
    def _cap_and_append_history(self):
        """Cap histories if needed and append current genomic bit."""
        if len(self.genomic_bits_history) >= self.max_history_size:
            self._remove_oldest_entries()
        self.genomic_bits_history.append(self.g_now)
    
    def _remove_oldest_entries(self):
        """Remove oldest entries from all history lists."""
        self.genomic_bits_history.pop(0)
        if len(self.mu_history) > 0:
            self.mu_history.pop(0)
        if len(self.sigma_history) > 0:
            self.sigma_history.pop(0)
    
    def _has_sufficient_data(self):
        """Check if there's enough data for meaningful introspection."""
        if len(self.genomic_bits_history) < INTROSPECTION_MIN_DATA_POINTS:
            self.s_score = INTROSPECTION_DEFAULT_S_SCORE
            self.coherence = INTROSPECTION_DEFAULT_COHERENCE
            return False
        return True
    
    def _get_default_surprise_score(self):
        """Return default surprise score when insufficient data."""
        return self.s_score
    
    def _calculate_historical_statistics(self):
        """Calculate mean and standard deviation from genomic bits history."""
        history_array = np.array(self.genomic_bits_history, dtype=np.float64)
        mu_history = float(np.mean(history_array))
        sigma_history = float(np.std(history_array))
        return mu_history, sigma_history
    
    def _apply_noise_floor(self, sigma_history):
        """
        Apply noise floor to prevent trauma learning.
        
        A perfectly silent room has sigma=0. We enforce a minimum noise floor
        to prevent false high scores from silence while still allowing real surprises.
        """
        if sigma_history < INTROSPECTION_NOISE_FLOOR:
            return INTROSPECTION_NOISE_FLOOR
        return sigma_history
    
    def _store_statistics(self, mu_history, sigma_history):
        """Store statistics for backward compatibility."""
        self.mu_history.append(mu_history)
        self.sigma_history.append(sigma_history)
    
    def _calculate_surprise_score(self, mu_history, sigma_history):
        """Calculate surprise score (Z-score) from historical statistics."""
        s_score = float(abs(self.g_now - mu_history) / sigma_history)
        if s_score < 0:
            return INTROSPECTION_DEFAULT_S_SCORE
        return s_score
    
    def _calculate_coherence(self):
        """Calculate coherence (inverse of surprise, normalized)."""
        return float(1.0 / (1.0 + self.s_score))
    
    def get_state(self):
        """Get current audio introspection state."""
        return {
            's_score': self.s_score,
            'coherence': self.coherence,
            'g_now': self.g_now
        }


class VisualIntrospectionStrategy(IntrospectionStrategy):
    """Strategy for visual introspection."""
    
    def __init__(self, max_history=INTROSPECTION_VISUAL_MAX_HISTORY):
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
        channel_surprises = {}
        
        for i, channel in enumerate(['R', 'G', 'B']):
            g_now = visual_vector[i]
            self._update_channel_memory(channel, g_now)
            
            if not self._has_sufficient_channel_frames(channel):
                channel_surprises[channel] = INTROSPECTION_DEFAULT_S_SCORE
                continue
            
            surprise = self._calculate_channel_surprise(channel, g_now)
            channel_surprises[channel] = surprise

        return self._fuse_visual_scores(channel_surprises)
    
    def _update_channel_memory(self, channel, g_now):
        """Update channel memory with new value, capping size if needed."""
        mem = self.visual_memory[channel]
        mem.append(g_now)
        if len(mem) > self.max_visual_history:
            mem.pop(0)
    
    def _has_sufficient_channel_frames(self, channel):
        """Check if channel has enough frames for meaningful introspection."""
        return len(self.visual_memory[channel]) >= INTROSPECTION_VISUAL_MIN_FRAMES
    
    def _calculate_channel_surprise(self, channel, g_now):
        """Calculate surprise score for a single channel."""
        mem = self.visual_memory[channel]
        mu_history = np.mean(mem)
        sigma_history = np.std(mem)
        sigma_history = self._apply_visual_noise_floor(sigma_history)
        
        return abs(g_now - mu_history) / sigma_history
    
    def _apply_visual_noise_floor(self, sigma_history):
        """Apply noise floor to prevent divide by zero."""
        if sigma_history == 0:
            return INTROSPECTION_SIGMA_DEFAULT
        return sigma_history
    
    def _fuse_visual_scores(self, channel_surprises):
        """
        Fuse channel scores into overall visual surprise.
        
        The visual score is the maximum surprise found in any channel.
        If the scene is mostly static (Low G, Low B) but a red laser appears (High R),
        the OPU should be Surprised.
        """
        if not channel_surprises:
            default_scores = {
                'R': INTROSPECTION_DEFAULT_S_SCORE,
                'G': INTROSPECTION_DEFAULT_S_SCORE,
                'B': INTROSPECTION_DEFAULT_S_SCORE
            }
            return INTROSPECTION_DEFAULT_S_SCORE, default_scores
        
        s_visual = max(channel_surprises.values())
        return s_visual, channel_surprises
    
    def get_state(self):
        """Get current visual introspection state."""
        return {
            'visual_memory': self.visual_memory,
            'visual_stats': self.visual_stats
        }

