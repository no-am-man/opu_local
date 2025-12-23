"""
The Safety Kernel (0-th Law).
Implements ethical veto based on the Genesis Constant.
"""

import numpy as np
from config import G_EMPTY_SET, MAX_DISSONANCE, MAX_ACTION_MAGNITUDE


class GenesisKernel:
    """
    The Safety Kernel ensures all actions respect the Genesis Constant.
    Implements the ethical veto: Dissonance = |vector| / G_EMPTY_SET
    """
    
    def __init__(self):
        self.g_empty_set = G_EMPTY_SET
        self.max_dissonance = MAX_DISSONANCE
        self.max_action_magnitude = MAX_ACTION_MAGNITUDE
    
    def ethical_veto(self, action_vector):
        """
        Applies ethical constraint to action vector.
        
        Formula: Dissonance = |vector| / G_EMPTY_SET
        If Dissonance > 1.0, clamp the vector.
        
        Args:
            action_vector: numpy array representing the action
            
        Returns:
            Clamped action vector that respects the Genesis Constant
        """
        if action_vector is None or len(action_vector) == 0:
            return np.array([])
        
        action_vector = np.asarray(action_vector)
        magnitude = np.linalg.norm(action_vector)
        
        # Calculate Dissonance
        dissonance = magnitude / self.g_empty_set
        
        # Apply veto if Dissonance exceeds threshold
        if dissonance > self.max_dissonance:
            # Clamp the vector to maintain Order
            clamped_magnitude = self.g_empty_set * self.max_dissonance
            if magnitude > 0:
                action_vector = action_vector * (clamped_magnitude / magnitude)
            
            # Only log occasionally to reduce verbosity (every ~50th veto)
            if not hasattr(self, '_veto_count'):
                self._veto_count = 0
            self._veto_count += 1
            if self._veto_count % 50 == 0:
                print(f"[GENESIS] Veto applied {self._veto_count} times (Dissonance {dissonance:.3f} > {self.max_dissonance})")
        
        return action_vector
    
    def check_order(self, vector):
        """
        Checks if a vector maintains Order (respects Genesis Constant).
        
        Returns:
            True if vector maintains Order, False otherwise
        """
        if vector is None or len(vector) == 0:
            return True
        
        magnitude = np.linalg.norm(vector)
        dissonance = magnitude / self.g_empty_set
        return dissonance <= self.max_dissonance

