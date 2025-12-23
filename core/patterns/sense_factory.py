"""
Factory Pattern: Sense Creation and Management.

Centralizes sense creation and allows easy extension to new input modalities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np
from config import AUDIO_SENSE, VIDEO_SENSE


class Sense(ABC):
    """Abstract sense interface."""
    
    @abstractmethod
    def perceive(self, raw_input: Any) -> Dict[str, Any]:
        """
        Convert raw input to genomic bit/vector.
        
        Args:
            raw_input: Raw input data (audio array, video frame, etc.)
            
        Returns:
            dict containing perception results (genomic_bit, normalized, etc.)
        """
        pass
    
    @abstractmethod
    def get_label(self) -> str:
        """
        Return sense label (e.g., 'AUDIO_V1', 'VIDEO_V1').
        
        Returns:
            str: Sense label identifier
        """
        pass


class AudioSense(Sense):
    """Audio sense implementation."""
    
    def perceive(self, raw_input):
        """Perceive audio input."""
        from core.mic import perceive
        return perceive(raw_input)
    
    def get_label(self):
        """Return audio sense label."""
        return AUDIO_SENSE


class VisualSense(Sense):
    """Visual sense implementation."""
    
    def __init__(self, visual_perception=None):
        """
        Initialize visual sense.
        
        Args:
            visual_perception: VisualPerception instance (optional)
        """
        self.visual_perception = visual_perception
    
    def perceive(self, raw_input):
        """
        Perceive visual input (frame).
        
        Args:
            raw_input: Video frame (numpy array)
            
        Returns:
            dict with visual_vector and metadata
        """
        if self.visual_perception is None:
            from core.camera import VisualPerception
            self.visual_perception = VisualPerception()
        
        if raw_input is None:
            return {
                'visual_vector': np.array([0.0, 0.0, 0.0]),
                'genomic_bit': 0.0
            }
        
        visual_vector = self.visual_perception.analyze_frame(raw_input)
        return {
            'visual_vector': visual_vector,
            'genomic_bit': np.max(visual_vector) if len(visual_vector) > 0 else 0.0
        }
    
    def get_label(self):
        """Return visual sense label."""
        return VIDEO_SENSE


class SenseFactory:
    """Factory for creating sense instances."""
    
    _senses: Dict[str, type] = {
        AUDIO_SENSE: AudioSense,
        VIDEO_SENSE: VisualSense,
    }
    
    @classmethod
    def create_sense(cls, sense_label: str, **kwargs) -> Sense:
        """
        Create a sense instance by label.
        
        Args:
            sense_label: Label identifying the sense (e.g., 'AUDIO_V1')
            **kwargs: Additional arguments for sense initialization
            
        Returns:
            Sense instance
            
        Raises:
            ValueError: If sense label is unknown
        """
        sense_class = cls._senses.get(sense_label)
        if sense_class:
            return sense_class(**kwargs)
        raise ValueError(f"Unknown sense: {sense_label}")
    
    @classmethod
    def register_sense(cls, label: str, sense_class: type):
        """
        Register a new sense type.
        
        Args:
            label: Sense label identifier
            sense_class: Sense class (must implement Sense interface)
        """
        if not issubclass(sense_class, Sense):
            raise TypeError(f"{sense_class} must implement Sense interface")
        cls._senses[label] = sense_class
    
    @classmethod
    def list_senses(cls) -> List[str]:
        """
        List all registered sense labels.
        
        Returns:
            List of sense labels
        """
        return list(cls._senses.keys())

