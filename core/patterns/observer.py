"""
Observer Pattern: OPU State Change Notifications.

Allows subsystems (visualization, expression, logging) to react to OPU state changes
without tight coupling to the core processing logic.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List


class OPUObserver(ABC):
    """Observer interface for OPU state changes."""
    
    @abstractmethod
    def on_state_changed(self, state: Dict[str, Any]):
        """
        Called when OPU state changes.
        
        Args:
            state: Dictionary containing current OPU state:
                - s_score: audio surprise score
                - s_visual: visual surprise score
                - s_global: fused surprise score
                - coherence: coherence score
                - maturity: maturity index
                - genomic_bit: current genomic bit
                - channel_scores: visual channel scores (if applicable)
        """
        pass


class ObservableOPU:
    """Mixin class that adds observer functionality to OPU."""
    
    def __init__(self):
        """Initialize observer list."""
        self._observers: List[OPUObserver] = []
    
    def attach_observer(self, observer: OPUObserver):
        """
        Attach an observer to receive state change notifications.
        
        Args:
            observer: Observer instance to attach
        """
        if observer not in self._observers:
            self._observers.append(observer)
    
    def detach_observer(self, observer: OPUObserver):
        """
        Detach an observer from notifications.
        
        Args:
            observer: Observer instance to detach
        """
        if observer in self._observers:
            self._observers.remove(observer)
    
    def notify_observers(self, state: Dict[str, Any]):
        """
        Notify all observers of state change.
        
        Args:
            state: State dictionary to broadcast
        """
        for observer in self._observers:
            try:
                observer.on_state_changed(state)
            except Exception as e:
                # Don't let observer errors break the OPU
                print(f"[OBSERVER] Error notifying {observer.__class__.__name__}: {e}")

