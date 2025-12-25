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
from dataclasses import dataclass
from core.brain import Brain
from core.audio_cortex import AudioCortex
from core.vision_cortex import VisualCortex
from core.patterns.observer import ObservableOPU
from config import (
    OPU_EMOTION_HISTORY_MAX_SIZE, OPU_EMOTION_DEFAULT_CONFIDENCE,
    OPU_EMOTION_DEFAULT_TOTAL, OPU_EMOTION_UNKNOWN_LABEL,
    BRAIN_DEFAULT_SENSE_LABEL
)


@dataclass
class EmotionMetrics:
    """Metrics calculated from emotion history."""
    emotion_counts: dict
    total_confidence: float
    emotion_count: int


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
        
        # Emotion tracking: history of detected emotions for learning
        self.emotion_history = []  # List of emotion dicts with timestamp
        
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
    
    def store_memory(self, genomic_bit, s_score, sense_label=BRAIN_DEFAULT_SENSE_LABEL, emotion=None, timestamp=None):
        """
        Store memory (delegates to Brain).
        Uses EPOCH time for temporal synchronization across all senses.
        
        Args:
            genomic_bit: the genomic bit to store
            s_score: the surprise score (determines level)
            sense_label: label identifying the input sense (e.g., "AUDIO_V1", "VIDEO_V1", "AUDIO_V2", "VIDEO_V2")
            emotion: optional emotion dict with 'emotion' (str) and 'confidence' (float) from face detection
            timestamp: optional timestamp for temporal sync (if None, uses current time)
                      Use same timestamp for all 4 channels (VIDEO_V1, AUDIO_V1, VIDEO_V2, AUDIO_V2) in a cycle
        """
        # --- TEMPORAL SYNC: Use provided timestamp or current time ---
        # For 4-channel sync (VIDEO_V1, AUDIO_V1, VIDEO_V2, AUDIO_V2), capture timestamp
        # once at start of cycle and pass it to all store_memory() calls
        # EPOCH time (time.time()) creates a universal "Wall Clock" that forces
        # all channels to align perfectly on the timeline, regardless of processing rates
        if timestamp is None:
            timestamp = time.time()
        
        self.brain.store_memory(genomic_bit, s_score, sense_label=sense_label, timestamp=timestamp, emotion=emotion)
        
        # Track emotion in history if available
        if emotion is not None:
            self._add_emotion_to_history(emotion, timestamp, sense_label)
    
    def consolidate_memory(self, level):
        """
        Consolidate memory (delegates to Brain).
        
        Args:
            level: abstraction level to consolidate
        """
        self.brain.consolidate_memory(level)
    
    def evolve_character(self):
        """
        Evolve character (delegates to Brain).
        """
        self.brain.evolve_character()
    
    def get_character_state(self):
        """
        Get character state (delegates to Brain).
        
        Returns:
            dict with maturity_index, base_pitch, stability_threshold
        """
        return self.brain.get_character_state()
    
    def get_emotion_statistics(self):
        """
        Get statistics about detected emotions.
        
        Returns:
            dict with emotion counts, most common emotion, and average confidence
        """
        if not self.emotion_history:
            return self._create_empty_emotion_statistics()
        
        metrics = self._calculate_emotion_metrics()
        most_common = self._find_most_common_emotion(metrics.emotion_counts)
        avg_confidence = self._calculate_average_confidence(metrics.total_confidence, metrics.emotion_count)
        
        return {
            'total_emotions': len(self.emotion_history),
            'emotion_counts': metrics.emotion_counts,
            'most_common': most_common,
            'average_confidence': avg_confidence
        }
    
    def _add_emotion_to_history(self, emotion, timestamp, sense_label):
        """Add emotion to history with size cap."""
        emotion_entry = self._create_emotion_entry(emotion, timestamp, sense_label)
        self.emotion_history.append(emotion_entry)
        self._cap_emotion_history_size()
    
    def _create_emotion_entry(self, emotion, timestamp, sense_label):
        """Create emotion entry dictionary."""
        return {
            'emotion': emotion,
            'timestamp': timestamp,
            'sense': sense_label
        }
    
    def _cap_emotion_history_size(self):
        """Cap emotion history size to maximum allowed."""
        if len(self.emotion_history) > OPU_EMOTION_HISTORY_MAX_SIZE:
            self.emotion_history = self.emotion_history[-OPU_EMOTION_HISTORY_MAX_SIZE:]
    
    def _create_empty_emotion_statistics(self):
        """Create empty emotion statistics when no history exists."""
        return {
            'total_emotions': OPU_EMOTION_DEFAULT_TOTAL,
            'emotion_counts': {},
            'most_common': None,
            'average_confidence': OPU_EMOTION_DEFAULT_CONFIDENCE
        }
    
    def _calculate_emotion_metrics(self):
        """Calculate emotion counts and total confidence from history."""
        emotion_counts = {}
        total_confidence = OPU_EMOTION_DEFAULT_CONFIDENCE
        emotion_count = 0
        
        for entry in self.emotion_history:
            if self._is_valid_emotion_entry(entry):
                em_name = self._extract_emotion_name(entry)
                em_conf = self._extract_emotion_confidence(entry)
                self._update_emotion_counts(emotion_counts, em_name)
                total_confidence += em_conf
                emotion_count += 1
        
        return EmotionMetrics(
            emotion_counts=emotion_counts,
            total_confidence=total_confidence,
            emotion_count=emotion_count
        )
    
    def _is_valid_emotion_entry(self, entry):
        """Check if entry contains a valid emotion dictionary."""
        return isinstance(entry.get('emotion'), dict)
    
    def _extract_emotion_name(self, entry):
        """Extract emotion name from entry, defaulting to unknown if missing."""
        return entry['emotion'].get('emotion', OPU_EMOTION_UNKNOWN_LABEL)
    
    def _extract_emotion_confidence(self, entry):
        """Extract emotion confidence from entry, defaulting to default if missing."""
        return entry['emotion'].get('confidence', OPU_EMOTION_DEFAULT_CONFIDENCE)
    
    def _update_emotion_counts(self, emotion_counts, emotion_name):
        """Update emotion count dictionary with new emotion."""
        emotion_counts[emotion_name] = emotion_counts.get(emotion_name, 0) + 1
    
    def _find_most_common_emotion(self, emotion_counts):
        """Find the most common emotion from counts."""
        return max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else None
    
    def _calculate_average_confidence(self, total_confidence, emotion_count):
        """Calculate average confidence from total and count."""
        return total_confidence / emotion_count if emotion_count > 0 else OPU_EMOTION_DEFAULT_CONFIDENCE
    
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
