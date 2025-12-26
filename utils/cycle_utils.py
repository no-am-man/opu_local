"""
Cycle Processing Utilities: Common utilities for OPU processing cycles.
Extracts shared logic from main.py and youtube_processor.py.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from config import (
    BRAIN_CONSOLIDATION_RATIO_L0, BRAIN_CONSOLIDATION_RATIO_L1,
    BRAIN_CONSOLIDATION_RATIO_L2, BRAIN_CONSOLIDATION_RATIO_L3,
    BRAIN_CONSOLIDATION_RATIO_L4, BRAIN_CONSOLIDATION_RATIO_L5,
    BRAIN_CONSOLIDATION_RATIO_L6, BRAIN_CONSOLIDATION_RATIO_L7,
    BRAIN_DEFAULT_CONSOLIDATION_RATIO
)


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


# Consolidation ratio mapping (shared across main.py and core/brain.py)
_CONSOLIDATION_RATIOS = {
    0: BRAIN_CONSOLIDATION_RATIO_L0,
    1: BRAIN_CONSOLIDATION_RATIO_L1,
    2: BRAIN_CONSOLIDATION_RATIO_L2,
    3: BRAIN_CONSOLIDATION_RATIO_L3,
    4: BRAIN_CONSOLIDATION_RATIO_L4,
    5: BRAIN_CONSOLIDATION_RATIO_L5,
    6: BRAIN_CONSOLIDATION_RATIO_L6,
    7: BRAIN_CONSOLIDATION_RATIO_L7
}


def get_consolidation_ratio(level: int) -> int:
    """
    Get consolidation ratio for a given memory level.
    
    Args:
        level: Memory level (0-7)
        
    Returns:
        Consolidation ratio (number of items needed to consolidate)
    """
    return _CONSOLIDATION_RATIOS.get(level, BRAIN_DEFAULT_CONSOLIDATION_RATIO)


def can_consolidate_at_level(level: int, memory_count: int) -> bool:
    """
    Check if consolidation can happen at given level with current memory count.
    
    Args:
        level: Memory level (0-7)
        memory_count: Current number of memories at this level
        
    Returns:
        True if consolidation can happen, False otherwise
    """
    required = get_consolidation_ratio(level)
    return memory_count >= required


# Timestamp approximation utilities (for memory consolidation)
def approximate_timestamp(timestamp: float, time_window: float) -> float:
    """
    Approximate a timestamp to a time window.
    This mimics how real memories are grouped by approximate time periods.
    
    Args:
        timestamp: Exact timestamp
        time_window: Size of time window in seconds
        
    Returns:
        Approximate timestamp (rounded to time window)
    """
    # Round timestamp to nearest time window
    # This creates "fuzzy" time grouping like in real memory
    return int(timestamp / time_window) * time_window


def get_time_window_for_level(level: int) -> float:
    """
    Get approximate time window for grouping memories at this level.
    In real life, memories are grouped by approximate time periods, not exact timestamps.
    
    Args:
        level: Memory level (0-7)
        
    Returns:
        Time window in seconds for this level
    """
    from config import MATURITY_LEVEL_TIMES
    return MATURITY_LEVEL_TIMES.get(level, 1.0)


def group_memories_by_time_window(memories: list, target_size: int, time_window: float) -> list:
    """
    Group memories by approximate time windows.
    This mimics real-life memory consolidation where memories are grouped
    by approximate time periods rather than exact timestamps.
    
    Args:
        memories: List of memory dictionaries with 'timestamp' key
        target_size: Target number of memories to group
        time_window: Time window size in seconds
        
    Returns:
        List of memories grouped by approximate time window
    """
    if not memories:
        return []
    
    # Sort memories by timestamp (oldest first)
    sorted_memories = sorted(memories, key=lambda m: m.get('timestamp', 0))
    
    if not sorted_memories:
        return []
    
    # Start with the oldest memory
    chunk = []
    base_timestamp = sorted_memories[0].get('timestamp', 0)
    
    # Approximate timestamp: round to time window
    # This makes memories "fuzzy" like in real life
    base_time_window = approximate_timestamp(base_timestamp, time_window)
    
    for mem in sorted_memories:
        if len(chunk) >= target_size:
            break
        
        mem_timestamp = mem.get('timestamp', 0)
        mem_time_window = approximate_timestamp(mem_timestamp, time_window)
        
        # Group memories within the same approximate time window
        if mem_time_window == base_time_window or len(chunk) < target_size:
            chunk.append(mem)
        else:
            # If we have enough items from this time window, stop
            if len(chunk) >= target_size:
                break
    
    return chunk[:target_size] if len(chunk) > target_size else chunk

