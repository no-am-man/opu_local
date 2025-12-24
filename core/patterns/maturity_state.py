"""
State Pattern: Maturity Levels.

Models maturity levels as states, encapsulating state-specific behavior
(pitch, stability threshold, time scales).
"""

from abc import ABC, abstractmethod
from typing import Dict
from config import (
    MATURITY_INSTANT_PITCH_MULTIPLIER, MATURITY_INSTANT_STABILITY,
    MATURITY_CHILD_PITCH_MULTIPLIER, MATURITY_CHILD_STABILITY,
    MATURITY_INFANT_PITCH_MULTIPLIER, MATURITY_INFANT_STABILITY,
    MATURITY_ADOLESCENT_PITCH_MULTIPLIER, MATURITY_ADOLESCENT_STABILITY,
    MATURITY_ADULT_PITCH_MULTIPLIER, MATURITY_ADULT_STABILITY,
    MATURITY_ELDER_PITCH_MULTIPLIER, MATURITY_ELDER_STABILITY,
    MATURITY_SAGE_PITCH_MULTIPLIER, MATURITY_SAGE_STABILITY,
    MATURITY_SCIRE_PITCH_MULTIPLIER, MATURITY_SCIRE_STABILITY,
    MATURITY_BASE_PITCH, MATURITY_PITCH_RANGE,
    MATURITY_STABILITY_BASE, MATURITY_STABILITY_RANGE
)


class MaturityState(ABC):
    """Abstract maturity state."""
    
    @abstractmethod
    def get_pitch_multiplier(self) -> float:
        """
        Return pitch multiplier for this maturity level.
        
        Returns:
            float: Multiplier (1.0 = 440Hz, 0.25 = 110Hz)
        """
        pass
    
    @abstractmethod
    def get_stability_threshold(self) -> float:
        """
        Return stability threshold for this level.
        
        Returns:
            float: Threshold value (higher = harder to surprise)
        """
        pass
    
    @abstractmethod
    def get_time_scale(self) -> str:
        """
        Return time scale name.
        
        Returns:
            str: Human-readable time scale (e.g., "1 minute")
        """
        pass
    
    @abstractmethod
    def get_level(self) -> int:
        """
        Return maturity level number (0-7).
        
        Returns:
            int: Level number
        """
        pass


class InstantState(MaturityState):
    """Level 0: Instant state (1 second) - Immediate Sensation (The "Now")."""
    
    def get_pitch_multiplier(self):
        return MATURITY_INSTANT_PITCH_MULTIPLIER  # 440Hz
    
    def get_stability_threshold(self):
        return MATURITY_INSTANT_STABILITY  # Very reactive
    
    def get_time_scale(self):
        return "1 second"
    
    def get_level(self):
        return 0


class ChildState(MaturityState):
    """Level 1: Child state (1 minute) - Working Memory."""
    
    def get_pitch_multiplier(self):
        return MATURITY_CHILD_PITCH_MULTIPLIER  # ~418Hz
    
    def get_stability_threshold(self):
        return MATURITY_CHILD_STABILITY
    
    def get_time_scale(self):
        return "1 minute"
    
    def get_level(self):
        return 1


class InfantState(MaturityState):
    """Level 2: Infant state (1 hour) - Episode / Situation."""
    
    def get_pitch_multiplier(self):
        return MATURITY_INFANT_PITCH_MULTIPLIER  # ~396Hz
    
    def get_stability_threshold(self):
        return MATURITY_INFANT_STABILITY
    
    def get_time_scale(self):
        return "1 hour"
    
    def get_level(self):
        return 2


class AdolescentState(MaturityState):
    """Level 3: Adolescent state (1 day) - Circadian / Sleep Consolidation."""
    
    def get_pitch_multiplier(self):
        return 0.7  # ~308Hz
    
    def get_stability_threshold(self):
        return 4.5
    
    def get_time_scale(self):
        return "1 day"
    
    def get_level(self):
        return 3


class AdultState(MaturityState):
    """Level 4: Adult state (1 week) - Trend."""
    
    def get_pitch_multiplier(self):
        return MATURITY_ADULT_PITCH_MULTIPLIER  # ~220Hz
    
    def get_stability_threshold(self):
        return MATURITY_ADULT_STABILITY
    
    def get_time_scale(self):
        return "1 week"
    
    def get_level(self):
        return 4


class ElderState(MaturityState):
    """Level 5: Elder state (1 month) - Season."""
    
    def get_pitch_multiplier(self):
        return MATURITY_ELDER_PITCH_MULTIPLIER  # ~154Hz
    
    def get_stability_threshold(self):
        return MATURITY_ELDER_STABILITY
    
    def get_time_scale(self):
        return "1 month"
    
    def get_level(self):
        return 5


class SageState(MaturityState):
    """Level 6: Sage state (1 year) - Epoch."""
    
    def get_pitch_multiplier(self):
        return MATURITY_SAGE_PITCH_MULTIPLIER  # 110Hz
    
    def get_stability_threshold(self):
        return MATURITY_SAGE_STABILITY
    
    def get_time_scale(self):
        return "1 year"
    
    def get_level(self):
        return 6


class ScireState(MaturityState):
    """Level 7: Scire state (10 years) - Core Identity / Deep Wisdom."""
    
    def get_pitch_multiplier(self):
        return MATURITY_SCIRE_PITCH_MULTIPLIER  # ~88Hz (even deeper, approaching fundamental)
    
    def get_stability_threshold(self):
        return MATURITY_SCIRE_STABILITY  # Very hard to surprise - true wisdom
    
    def get_time_scale(self):
        return "10 years"
    
    def get_level(self):
        return 7


class MaturityContext:
    """Context that maintains current maturity state."""
    
    _state_map = {
        0: InstantState(),
        1: ChildState(),
        2: InfantState(),
        3: AdolescentState(),
        4: AdultState(),
        5: ElderState(),
        6: SageState(),
        7: ScireState(),
    }
    
    def __init__(self):
        """Initialize with child state."""
        self._state = self._state_map[0]
        self._base_pitch = 440.0
    
    def transition_to_level(self, level: int):
        """
        Transition to a specific maturity level.
        
        Args:
            level: Maturity level (0-7)
        """
        if 0 <= level <= 7:
            self._state = self._state_map[level]
    
    def get_current_state(self) -> MaturityState:
        """Get current maturity state."""
        return self._state
    
    def get_pitch(self, maturity_index: float = None) -> float:
        """
        Get current pitch based on state, with optional continuous interpolation.
        
        Args:
            maturity_index: Optional continuous maturity index (0.0-1.0) for interpolation.
                           If None, uses discrete state multiplier.
        
        Returns:
            float: Pitch in Hz
        """
        if maturity_index is not None:
            # Interpolate between states based on continuous maturity_index
            # Drops from MATURITY_BASE_PITCH (A4) to 110Hz (A2) as maturity increases
            return MATURITY_BASE_PITCH - (maturity_index * MATURITY_PITCH_RANGE)
        return self._base_pitch * self._state.get_pitch_multiplier()
    
    def set_base_pitch(self, base_pitch: float):
        """
        Set base pitch.
        
        Args:
            base_pitch: Base pitch in Hz (default: MATURITY_BASE_PITCH)
        """
        self._base_pitch = base_pitch
    
    def get_stability_threshold(self, maturity_index: float = None) -> float:
        """
        Get stability threshold for current state, with optional continuous interpolation.
        
        Args:
            maturity_index: Optional continuous maturity index (0.0-1.0) for interpolation.
                           If None, uses discrete state threshold.
        
        Returns:
            float: Stability threshold
        """
        if maturity_index is not None:
            # Interpolate: Threshold moves from MATURITY_STABILITY_BASE to MATURITY_STABILITY_BASE + MATURITY_STABILITY_RANGE as maturity increases
            return MATURITY_STABILITY_BASE + (maturity_index * MATURITY_STABILITY_RANGE)
        return self._state.get_stability_threshold()
    
    def get_time_scale(self) -> str:
        """Get time scale name for current state."""
        return self._state.get_time_scale()
    
    def get_level(self) -> int:
        """Get current maturity level."""
        return self._state.get_level()
    
    def update_from_memory_levels(self, memory_levels: Dict[int, list]) -> Dict[str, float]:
        """
        Update maturity state based on memory levels and return character profile.
        
        This matches Brain.evolve_character() logic but uses State Pattern.
        
        Args:
            memory_levels: Dict mapping level (0-7) to list of memories/abstractions
        
        Returns:
            dict with maturity_level, maturity_index, base_pitch, stability_threshold
        """
        # Find highest level with consolidated memories
        highest_level = 0
        for lvl in range(7, -1, -1):
            if len(memory_levels.get(lvl, [])) > 0:
                highest_level = lvl
                break
        
        # Transition to highest level reached
        self.transition_to_level(highest_level)
        
        # Calculate continuous maturity_index
        level_progress = min(1.0, len(memory_levels.get(highest_level, [])) / 10.0) if highest_level > 0 else 0.0
        base_maturity = highest_level / 7.0  # 0.0, 0.143, 0.286, 0.429, 0.571, 0.714, 0.857, 1.0
        maturity_index = min(1.0, base_maturity + (level_progress * 0.143))
        
        return {
            'maturity_level': highest_level,
            'maturity_index': maturity_index,
            'base_pitch': self.get_pitch(maturity_index),
            'stability_threshold': self.get_stability_threshold(maturity_index)
        }

