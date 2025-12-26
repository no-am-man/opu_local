"""
Cycle Processing Utilities: Common utilities for OPU processing cycles.
Extracts shared logic from main.py and youtube_processor.py.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np


def fuse_scores(s_audio: float, s_visual: float) -> float:
    """
    Fuse audio and visual surprise scores.
    
    Args:
        s_audio: Audio surprise score
        s_visual: Visual surprise score
        
    Returns:
        Fused surprise score (max of both)
    """
    return max(s_audio, s_visual)


def apply_ethical_veto(genesis, fused_score: float, genomic_bit: float) -> float:
    """
    Apply ethical veto (safety kernel) to fused score.
    
    Args:
        genesis: GenesisKernel instance
        fused_score: Fused surprise score
        genomic_bit: Genomic bit value
        
    Returns:
        Safe score after ethical veto
    """
    action = genesis.ethical_veto(np.array([fused_score, genomic_bit]))
    return action[0] if len(action) > 0 else fused_score


def extract_visual_bit(visual_vector: np.ndarray) -> float:
    """
    Extract visual genomic bit from visual vector.
    
    Args:
        visual_vector: Visual vector array
        
    Returns:
        Maximum value from vector (visual genomic bit)
    """
    return max(visual_vector) if len(visual_vector) > 0 else 0.0


def extract_emotion_from_detections(detections: list) -> Optional[Dict[str, Any]]:
    """
    Extract emotion from detection list.
    
    Args:
        detections: List of detection dictionaries
        
    Returns:
        Emotion dict with 'emotion' and 'confidence' or None
    """
    if not detections:
        return None
    
    for det in detections:
        if det.get('label') == 'face' and 'emotion' in det:
            return det['emotion']
    
    return None


def format_emotion_string(emotion: Optional[Dict[str, Any]]) -> str:
    """
    Format emotion for logging.
    
    Args:
        emotion: Emotion dict or None
        
    Returns:
        Formatted emotion string
    """
    if emotion:
        return f" | Emotion: {emotion['emotion']} ({emotion['confidence']:.2f})"
    return ""


class LogCounter:
    """
    Counter utility for periodic logging.
    Reduces code duplication for log frequency management.
    """
    
    def __init__(self, interval: int = 100):
        """
        Initialize log counter.
        
        Args:
            interval: Logging interval (log every N calls)
        """
        self.interval = interval
        self.count = 0
    
    def should_log(self) -> bool:
        """
        Check if should log this cycle.
        Increments counter automatically.
        
        Returns:
            True if should log, False otherwise
        """
        self.count += 1
        return self.count % self.interval == 0
    
    def reset(self):
        """Reset counter to 0."""
        self.count = 0


def create_processing_result(
    s_audio: float,
    s_visual: float,
    safe_score: float,
    channel_scores: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Create standardized processing result dictionary.
    
    Args:
        s_audio: Audio surprise score
        s_visual: Visual surprise score
        safe_score: Safe score after ethical veto
        channel_scores: Optional channel scores dict
        
    Returns:
        Processing result dictionary
    """
    result = {
        's_audio': s_audio,
        's_visual': s_visual,
        'safe_score': safe_score,
        'fused_score': fuse_scores(s_audio, s_visual)
    }
    
    if channel_scores:
        result['channel_scores'] = channel_scores
    
    return result

