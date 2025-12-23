"""
The Orthogonal Processing Unit (OPU): Main Processing Unit.

This module provides the OrthogonalProcessingUnit class which combines:
- Brain: Memory abstraction and character evolution
- AudioCortex: Audio introspection
- VisualCortex: Visual introspection

This is the main entry point for the OPU. For direct access to subsystems,
use the individual modules (brain.py, audio_cortex.py, vision_cortex.py).
"""

import numpy as np
from core.brain import Brain
from core.audio_cortex import AudioCortex
from core.vision_cortex import VisualCortex


class OrthogonalProcessingUnit:
    """
    The core processing unit with introspection and memory abstraction.
    Evolves from a noisy child to a deep-voiced sage through memory consolidation.
    
    This class acts as a facade that combines:
    - Brain: Memory abstraction and character evolution
    - AudioCortex: Audio introspection
    - VisualCortex: Visual introspection
    
    For direct access to subsystems, use:
    - opu.brain
    - opu.audio_cortex
    - opu.vision_cortex
    """
    
    def __init__(self):
        # Initialize subsystems
        self.brain = Brain()
        self.audio_cortex = AudioCortex()
        self.vision_cortex = VisualCortex()
        
        # Expose memory_levels for backward compatibility
        self.memory_levels = self.brain.memory_levels
        self.character_profile = self.brain.character_profile
        
        # Expose visual memory for backward compatibility
        self.visual_memory = self.vision_cortex.visual_memory
        self.max_visual_history = self.vision_cortex.max_visual_history
        self.visual_stats = self.vision_cortex.visual_stats
        
        # Expose audio cortex attributes for backward compatibility
        self.genomic_bits_history = self.audio_cortex.genomic_bits_history
        self.mu_history = self.audio_cortex.mu_history
        self.sigma_history = self.audio_cortex.sigma_history
        self.max_history_size = self.audio_cortex.max_history_size
    
    def introspect(self, genomic_bit):
        """
        Audio introspection (delegates to AudioCortex).
        
        Args:
            genomic_bit: current genomic bit (standard deviation from perception)
            
        Returns:
            s_score: surprise score (higher = more surprising)
        """
        return self.audio_cortex.introspect(genomic_bit)
    
    def introspect_visual(self, visual_vector):
        """
        Visual introspection (delegates to VisualCortex).
        
        Args:
            visual_vector: np.array([sigma_r, sigma_g, sigma_b])
            
        Returns:
            s_visual: The highest surprise score found across the 3 channels
            channel_surprises: dict with individual channel scores
        """
        return self.vision_cortex.introspect(visual_vector)
    
    def store_memory(self, genomic_bit, s_score):
        """
        Store memory (delegates to Brain).
        
        Args:
            genomic_bit: the genomic bit to store
            s_score: the surprise score (determines level)
        """
        # Calculate timestamp based on genomic bits history length
        # This matches the original behavior where timestamp reflected processing count
        timestamp = len(self.genomic_bits_history)
        self.brain.store_memory(genomic_bit, s_score, timestamp=timestamp)
    
    def consolidate_memory(self, level):
        """
        Consolidate memory (delegates to Brain).
        
        Args:
            level: abstraction level to consolidate
        """
        self.brain.consolidate_memory(level)
    
    def evolve_character(self, level=None):
        """
        Evolve character (delegates to Brain).
        
        Args:
            level: The abstraction level that triggered evolution (0-5)
        """
        self.brain.evolve_character(level)
    
    def get_character_state(self):
        """
        Get character state (delegates to Brain).
        
        Returns:
            dict with maturity_index, base_pitch, stability_threshold
        """
        return self.brain.get_character_state()
    
    def get_current_state(self):
        """
        Returns current cognitive state (combines audio and brain state).
        
        Returns:
            dict with s_score, coherence, g_now, maturity
        """
        audio_state = self.audio_cortex.get_state()
        return {
            's_score': audio_state['s_score'],
            'coherence': audio_state['coherence'],
            'g_now': audio_state['g_now'],
            'maturity': self.brain.character_profile['maturity_index']
        }
    
    # Expose properties for backward compatibility
    @property
    def s_score(self):
        """Current audio surprise score."""
        return self.audio_cortex.s_score
    
    @s_score.setter
    def s_score(self, value):
        """Set audio surprise score (for persistence)."""
        self.audio_cortex.s_score = value
    
    @property
    def coherence(self):
        """Current coherence score."""
        return self.audio_cortex.coherence
    
    @coherence.setter
    def coherence(self, value):
        """Set coherence score (for persistence)."""
        self.audio_cortex.coherence = value
    
    @property
    def g_now(self):
        """Current genomic bit."""
        return self.audio_cortex.g_now
    
    @g_now.setter
    def g_now(self, value):
        """Set current genomic bit (for persistence)."""
        self.audio_cortex.g_now = value
