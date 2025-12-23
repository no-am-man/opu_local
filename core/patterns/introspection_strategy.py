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
        """Calculate audio surprise score."""
        self.g_now = genomic_bit
        
        # Cap histories BEFORE appending
        if len(self.genomic_bits_history) >= self.max_history_size:
            self.genomic_bits_history.pop(0)
            if len(self.mu_history) > 0:
                self.mu_history.pop(0)
            if len(self.sigma_history) > 0:
                self.sigma_history.pop(0)
        
        self.genomic_bits_history.append(genomic_bit)
        
        # Need at least 2 data points
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
        
        # Calculate coherence
        self.coherence = 1.0 / (1.0 + self.s_score)
        
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
        
        Args:
            visual_vector: np.array([sigma_r, sigma_g, sigma_b])
            
        Returns:
            s_visual: The highest surprise score found across the 3 channels
        """
        channels = ['R', 'G', 'B']
        channel_surprises = {}
        
        for i, channel in enumerate(channels):
            g_now = visual_vector[i]
            mem = self.visual_memory[channel]
            
            # Add to memory
            mem.append(g_now)
            if len(mem) > self.max_visual_history:
                mem.pop(0)
            
            # Need history to judge surprise
            if len(mem) < 10:
                channel_surprises[channel] = 0.0
                continue
            
            # Calculate baseline
            mu_history = np.mean(mem)
            sigma_history = np.std(mem)
            
            if sigma_history == 0:
                sigma_history = 0.1
            
            # Calculate Z-Score (Surprise)
            s_channel = abs(g_now - mu_history) / sigma_history
            channel_surprises[channel] = s_channel
        
        # Sensory fusion: max surprise across channels
        if not channel_surprises:
            return 0.0
        
        s_visual = max(channel_surprises.values())
        return s_visual
    
    def introspect_with_channels(self, visual_vector):
        """
        Calculate visual surprise with channel breakdown.
        
        Returns:
            tuple: (s_visual, channel_surprises_dict)
        """
        channels = ['R', 'G', 'B']
        channel_surprises = {}
        
        for i, channel in enumerate(channels):
            g_now = visual_vector[i]
            mem = self.visual_memory[channel]
            
            mem.append(g_now)
            if len(mem) > self.max_visual_history:
                mem.pop(0)
            
            if len(mem) < 10:
                channel_surprises[channel] = 0.0
                continue
            
            mu_history = np.mean(mem)
            sigma_history = np.std(mem)
            
            if sigma_history == 0:
                sigma_history = 0.1
            
            s_channel = abs(g_now - mu_history) / sigma_history
            channel_surprises[channel] = s_channel
        
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

