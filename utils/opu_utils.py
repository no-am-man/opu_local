"""
OPU Utility Functions: Common operations shared across main.py and youtube_opu.py
"""

import sys
import time
from typing import Optional, Tuple, Dict, Any
import numpy as np
from utils.file_logger import FileLogger


def setup_file_logging(log_file: Optional[str] = None, default_name: str = "opu.log") -> Tuple[FileLogger, Any, Any]:
    """
    Setup file logging for OPU processes.
    
    Args:
        log_file: Path to log file (if None, uses default_name)
        default_name: Default log file name if log_file is None
        
    Returns:
        Tuple of (file_logger, original_stdout, original_stderr)
    """
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    if log_file is None:
        log_file = default_name
    
    file_logger = FileLogger(log_file, chain_to=original_stdout)
    sys.stdout = file_logger
    sys.stderr = file_logger
    
    return file_logger, original_stdout, original_stderr


def cleanup_file_logging(file_logger: Optional[FileLogger], original_stdout: Any, original_stderr: Any):
    """
    Cleanup file logging and restore original stdout/stderr.
    
    Args:
        file_logger: FileLogger instance to close
        original_stdout: Original stdout to restore
        original_stderr: Original stderr to restore
    """
    if file_logger:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        file_logger.close()


def calculate_ethical_veto(genesis, fused_score: float, genomic_bit: float) -> float:
    """
    Calculate safe score using ethical veto.
    
    Args:
        genesis: GenesisKernel instance
        fused_score: Fused surprise score
        genomic_bit: Genomic bit value
        
    Returns:
        Safe score after ethical veto
    """
    action = genesis.ethical_veto(np.array([fused_score, genomic_bit]))
    return action[0] if len(action) > 0 else fused_score


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


def get_cycle_timestamp() -> float:
    """
    Get current timestamp for temporal synchronization.
    
    Returns:
        Current epoch time
    """
    return time.time()

