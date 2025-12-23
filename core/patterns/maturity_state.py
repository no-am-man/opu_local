"""
State Pattern: Maturity Levels.

Models maturity levels as states, encapsulating state-specific behavior
(pitch, stability threshold, time scales).
"""

from abc import ABC, abstractmethod
from typing import Dict


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
        Return maturity level number (0-5).
        
        Returns:
            int: Level number
        """
        pass


class ChildState(MaturityState):
    """Level 0: Child state (1 minute)."""
    
    def get_pitch_multiplier(self):
        return 1.0  # 440Hz
    
    def get_stability_threshold(self):
        return 3.0
    
    def get_time_scale(self):
        return "1 minute"
    
    def get_level(self):
        return 0


class InfantState(MaturityState):
    """Level 1: Infant state (1 hour)."""
    
    def get_pitch_multiplier(self):
        return 0.9  # ~396Hz
    
    def get_stability_threshold(self):
        return 3.5
    
    def get_time_scale(self):
        return "1 hour"
    
    def get_level(self):
        return 1


class AdolescentState(MaturityState):
    """Level 2: Adolescent state (1 day)."""
    
    def get_pitch_multiplier(self):
        return 0.7  # ~308Hz
    
    def get_stability_threshold(self):
        return 4.5
    
    def get_time_scale(self):
        return "1 day"
    
    def get_level(self):
        return 2


class AdultState(MaturityState):
    """Level 3: Adult state (1 week)."""
    
    def get_pitch_multiplier(self):
        return 0.5  # ~220Hz
    
    def get_stability_threshold(self):
        return 5.5
    
    def get_time_scale(self):
        return "1 week"
    
    def get_level(self):
        return 3


class ElderState(MaturityState):
    """Level 4: Elder state (1 month)."""
    
    def get_pitch_multiplier(self):
        return 0.35  # ~154Hz
    
    def get_stability_threshold(self):
        return 6.5
    
    def get_time_scale(self):
        return "1 month"
    
    def get_level(self):
        return 4


class SageState(MaturityState):
    """Level 5: Sage state (1 year)."""
    
    def get_pitch_multiplier(self):
        return 0.25  # 110Hz
    
    def get_stability_threshold(self):
        return 8.0
    
    def get_time_scale(self):
        return "1 year"
    
    def get_level(self):
        return 5


class MaturityContext:
    """Context that maintains current maturity state."""
    
    _state_map = {
        0: ChildState(),
        1: InfantState(),
        2: AdolescentState(),
        3: AdultState(),
        4: ElderState(),
        5: SageState(),
    }
    
    def __init__(self):
        """Initialize with child state."""
        self._state = self._state_map[0]
        self._base_pitch = 440.0
    
    def transition_to_level(self, level: int):
        """
        Transition to a specific maturity level.
        
        Args:
            level: Maturity level (0-5)
        """
        if 0 <= level <= 5:
            self._state = self._state_map[level]
    
    def get_current_state(self) -> MaturityState:
        """Get current maturity state."""
        return self._state
    
    def get_pitch(self) -> float:
        """
        Get current pitch based on state.
        
        Returns:
            float: Pitch in Hz
        """
        return self._base_pitch * self._state.get_pitch_multiplier()
    
    def set_base_pitch(self, base_pitch: float):
        """
        Set base pitch.
        
        Args:
            base_pitch: Base pitch in Hz (default: 440.0)
        """
        self._base_pitch = base_pitch
    
    def get_stability_threshold(self) -> float:
        """Get stability threshold for current state."""
        return self._state.get_stability_threshold()
    
    def get_time_scale(self) -> str:
        """Get time scale name for current state."""
        return self._state.get_time_scale()
    
    def get_level(self) -> int:
        """Get current maturity level."""
        return self._state.get_level()

