"""
The Orthogonal Processing Unit (OPU): Main Processing Unit.

This module provides the OrthogonalProcessingUnit class which combines:
- Brain: Memory abstraction and character evolution
- AudioCortex: Audio introspection
- VisualCortex: Visual introspection

This is the main entry point for the OPU. For direct access to subsystems,
use the individual modules (brain.py, audio_cortex.py, vision_cortex.py).

Now supports Observer Pattern for state change notifications.
"""

import numpy as np
import time  # For EPOCH timestamps
from core.brain import Brain
from core.audio_cortex import AudioCortex
from core.vision_cortex import VisualCortex
from core.patterns.observer import ObservableOPU


class OrthogonalProcessingUnit(ObservableOPU):
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
        # Initialize observer functionality
        super().__init__()
        
        # Initialize subsystems
        self.brain = Brain()
        self.audio_cortex = AudioCortex()
        self.vision_cortex = VisualCortex()
        
        # DO NOT create shadow copies - use properties/delegation instead
        # This prevents synchronization issues when cortex objects update their internal state
    
    def introspect(self, genomic_bit):
        """
        Audio introspection (delegates to AudioCortex).
        Ensures state is properly updated and synchronized.
        
        Args:
            genomic_bit: current genomic bit (standard deviation from perception)
            
        Returns:
            s_score: surprise score (higher = more surprising)
        """
        # Delegate to audio cortex - this updates self.audio_cortex.s_score
        s_score = self.audio_cortex.introspect(genomic_bit)
        
        # Notify observers of state change (reads fresh state)
        self._notify_state_change()
        
        return s_score
    
    def introspect_visual(self, visual_vector):
        """
        Visual introspection (delegates to VisualCortex).
        
        Args:
            visual_vector: np.array([sigma_r, sigma_g, sigma_b])
            
        Returns:
            s_visual: The highest surprise score found across the 3 channels
            channel_surprises: dict with individual channel scores
        """
        result = self.vision_cortex.introspect(visual_vector)
        # Notify observers of state change
        self._notify_state_change()
        return result
    
    def store_memory(self, genomic_bit, s_score, sense_label="UNKNOWN"):
        """
        Store memory (delegates to Brain).
        Uses EPOCH time for temporal synchronization across all senses.
        
        Args:
            genomic_bit: the genomic bit to store
            s_score: the surprise score (determines level)
            sense_label: label identifying the input sense (e.g., "AUDIO_V1", "VIDEO_V1")
        """
        # --- FIX: USE EPOCH TIME FOR GLOBAL SYNC ---
        # EPOCH time (time.time()) creates a universal "Wall Clock" that forces
        # Audio and Video to align perfectly on the timeline, regardless of
        # processing rates (FPS vs Sample Rate).
        # Old logical time: timestamp = len(self.audio_cortex.genomic_bits_history)
        timestamp = time.time()
        
        self.brain.store_memory(genomic_bit, s_score, sense_label=sense_label, timestamp=timestamp)
    
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
        Always reads fresh state from subsystems to avoid stale data.
        
        Returns:
            dict with s_score, coherence, g_now, maturity
        """
        # Always read fresh state directly from audio_cortex
        audio_state = self.audio_cortex.get_state()
        return {
            's_score': audio_state['s_score'],
            'coherence': audio_state['coherence'],
            'g_now': audio_state['g_now'],
            'maturity': self.brain.character_profile['maturity_index']
        }
    
    def _notify_state_change(self):
        """Notify all observers of state change."""
        state = self.get_current_state()
        self.notify_observers(state)
    
    # --- PROPERTIES FOR BACKWARD COMPATIBILITY ---
    # These delegate directly to subsystems (no shadow copies)
    
    @property
    def memory_levels(self):
        """Memory levels (delegates to Brain)."""
        return self.brain.memory_levels
    
    @property
    def character_profile(self):
        """Character profile (delegates to Brain)."""
        return self.brain.character_profile
    
    @property
    def visual_memory(self):
        """Visual memory (delegates to VisualCortex)."""
        return self.vision_cortex.visual_memory
    
    @property
    def max_visual_history(self):
        """Max visual history (delegates to VisualCortex)."""
        return self.vision_cortex.max_visual_history
    
    @property
    def visual_stats(self):
        """Visual stats (delegates to VisualCortex)."""
        return self.vision_cortex.visual_stats
    
    @property
    def genomic_bits_history(self):
        """Genomic bits history (delegates to AudioCortex)."""
        return self.audio_cortex.genomic_bits_history
    
    @property
    def mu_history(self):
        """Mean history (delegates to AudioCortex)."""
        return self.audio_cortex.mu_history
    
    @property
    def sigma_history(self):
        """Sigma history (delegates to AudioCortex)."""
        return self.audio_cortex.sigma_history
    
    @property
    def max_history_size(self):
        """Max history size (delegates to AudioCortex)."""
        return self.audio_cortex.max_history_size
    
    @property
    def s_score(self):
        """Current audio surprise score (delegates to AudioCortex)."""
        return self.audio_cortex.s_score
    
    @s_score.setter
    def s_score(self, value):
        """Set audio surprise score (for persistence)."""
        self.audio_cortex.s_score = value
    
    @property
    def coherence(self):
        """Current coherence score (delegates to AudioCortex)."""
        return self.audio_cortex.coherence
    
    @coherence.setter
    def coherence(self, value):
        """Set coherence score (for persistence)."""
        self.audio_cortex.coherence = value
    
    @property
    def g_now(self):
        """Current genomic bit (delegates to AudioCortex)."""
        return self.audio_cortex.g_now
    
    @g_now.setter
    def g_now(self, value):
        """Set current genomic bit (for persistence)."""
        self.audio_cortex.g_now = value
