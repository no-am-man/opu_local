"""
The Safety Kernel (0-th Law).
Implements ethical veto based on the Genesis Constant.
"""

import numpy as np
from config import G_EMPTY_SET, MAX_DISSONANCE, MAX_ACTION_MAGNITUDE, GENESIS_VETO_LOG_INTERVAL


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
        if self._is_empty_vector(action_vector):
            return np.array([])
        
        action_vector = np.asarray(action_vector)
        magnitude = np.linalg.norm(action_vector)
        
        dissonance = self._calculate_dissonance(magnitude)
        
        if dissonance > self.max_dissonance:
            action_vector = self._apply_veto_clamping(action_vector, magnitude, dissonance)
        
        return action_vector
    
    def check_order(self, vector):
        """
        Checks if a vector maintains Order (respects Genesis Constant).
        
        Returns:
            True if vector maintains Order, False otherwise
        """
        if self._is_empty_vector(vector):
            return True
        
        magnitude = np.linalg.norm(vector)
        dissonance = self._calculate_dissonance(magnitude)
        return dissonance <= self.max_dissonance
    
    def _is_empty_vector(self, vector):
        """Check if vector is None or empty."""
        return vector is None or len(vector) == 0
    
    def _calculate_dissonance(self, magnitude):
        """Calculate dissonance from vector magnitude."""
        return magnitude / self.g_empty_set
    
    def _apply_veto_clamping(self, action_vector, magnitude, dissonance):
        """Apply veto clamping and logging."""
        clamped_magnitude = self.g_empty_set * self.max_dissonance
        if magnitude > 0:
            action_vector = action_vector * (clamped_magnitude / magnitude)
        
        self._log_veto_if_needed(dissonance)
        return action_vector
    
    def _log_veto_if_needed(self, dissonance):
        """Log veto application at specified intervals."""
        if not hasattr(self, '_veto_count'):
            self._veto_count = 0
        self._veto_count += 1
        if self._veto_count % GENESIS_VETO_LOG_INTERVAL == 0:
            print(f"[GENESIS] Veto applied {self._veto_count} times (Dissonance {dissonance:.3f} > {self.max_dissonance})")

