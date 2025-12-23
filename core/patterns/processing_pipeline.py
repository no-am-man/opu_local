"""
Template Method Pattern: Processing Pipeline.

Defines the skeleton of the OPU processing algorithm, allowing subclasses
to override specific steps while maintaining the overall structure.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class ProcessingPipeline(ABC):
    """Template method for OPU processing cycle."""
    
    def process(self) -> Dict[str, Any]:
        """
        Template method defining the processing steps.
        
        Returns:
            dict with processing results
        """
        # 1. Perception
        audio_data = self.capture_audio()
        visual_data = self.capture_visual()
        
        # 2. Genomic extraction
        audio_genomic = self.extract_audio_genomic(audio_data)
        visual_genomic = self.extract_visual_genomic(visual_data)
        
        # 3. Introspection
        s_audio = self.introspect_audio(audio_genomic)
        s_visual, channel_scores = self.introspect_visual(visual_genomic)
        
        # 4. Fusion (hook point - can be overridden)
        s_global = self.fuse_scores(s_audio, s_visual)
        
        # 5. Safety
        safe_score = self.apply_safety(s_global, audio_genomic)
        
        # 6. Memory (hook point)
        self.store_memory(audio_genomic, safe_score, visual_genomic, s_visual)
        
        # 7. Expression (hook point)
        self.generate_expression(safe_score)
        
        return self.create_result(s_audio, s_visual, s_global, safe_score, channel_scores)
    
    # Hook methods - can be overridden
    def fuse_scores(self, s_audio: float, s_visual: float) -> float:
        """
        Default fusion: max of both scores.
        
        Args:
            s_audio: Audio surprise score
            s_visual: Visual surprise score
            
        Returns:
            float: Fused surprise score
        """
        return max(s_audio, s_visual)
    
    def create_result(self, s_audio: float, s_visual: float, s_global: float,
                     safe_score: float, channel_scores: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Create result dictionary.
        
        Args:
            s_audio: Audio surprise score
            s_visual: Visual surprise score
            s_global: Global fused score
            safe_score: Safety-clamped score
            channel_scores: Visual channel scores (optional)
            
        Returns:
            dict with processing results
        """
        return {
            's_audio': s_audio,
            's_visual': s_visual,
            's_global': s_global,
            'safe_score': safe_score,
            'channel_scores': channel_scores or {}
        }
    
    # Abstract methods - must be implemented
    @abstractmethod
    def capture_audio(self) -> np.ndarray:
        """Capture audio input."""
        pass
    
    @abstractmethod
    def capture_visual(self) -> Optional[np.ndarray]:
        """Capture visual input."""
        pass
    
    @abstractmethod
    def extract_audio_genomic(self, audio_data: np.ndarray) -> float:
        """Extract genomic bit from audio."""
        pass
    
    @abstractmethod
    def extract_visual_genomic(self, visual_data: Optional[np.ndarray]) -> np.ndarray:
        """Extract genomic vector from visual."""
        pass
    
    @abstractmethod
    def introspect_audio(self, genomic_bit: float) -> float:
        """Calculate audio surprise score."""
        pass
    
    @abstractmethod
    def introspect_visual(self, visual_vector: np.ndarray) -> tuple:
        """Calculate visual surprise score."""
        pass
    
    @abstractmethod
    def apply_safety(self, s_score: float, genomic_bit: float) -> float:
        """Apply safety kernel."""
        pass
    
    @abstractmethod
    def store_memory(self, audio_genomic: float, s_score: float,
                    visual_genomic: np.ndarray, s_visual: float):
        """Store memory."""
        pass
    
    @abstractmethod
    def generate_expression(self, s_score: float):
        """Generate expression (audio/phonemes)."""
        pass

