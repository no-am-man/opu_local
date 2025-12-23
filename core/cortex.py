"""
The Brain: Introspection & Memory Abstraction.
Implements the Orthogonal Processing Unit with introspection,
memory layers, and character evolution.
"""

import numpy as np
from config import MATURITY_INCREMENT


class OrthogonalProcessingUnit:
    """
    The core processing unit with introspection and memory abstraction.
    Evolves from a noisy child to a deep-voiced sage through memory consolidation.
    """
    
    def __init__(self):
        # Memory abstraction layers (Level 0 = Raw, Level 3 = Wisdom)
        self.memory_levels = {0: [], 1: [], 2: [], 3: []}
        
        # Character profile that evolves over time
        self.character_profile = {
            "maturity_index": 0.0,  # 0.0 = Child, 1.0 = Sage
            "base_pitch": 440.0,    # Starts high (Child), drops to 110Hz (Sage)
            "stability_threshold": 3.0  # Easily surprised initially
        }
        
        # History for introspection
        self.mu_history = []  # Mean of historical genomic bits
        self.sigma_history = []  # Std dev of historical genomic bits
        self.genomic_bits_history = []  # Raw genomic bits
        
        # Current state
        self.g_now = None
        self.s_score = 0.0
        self.coherence = 0.0
    
    def introspect(self, genomic_bit):
        """
        Calculates surprise score (s_score) through introspection.
        
        Formula: s_score = |g_now - mu_history| / sigma_history
        
        Args:
            genomic_bit: current genomic bit (standard deviation from perception)
            
        Returns:
            s_score: surprise score (higher = more surprising)
        """
        self.g_now = genomic_bit
        self.genomic_bits_history.append(genomic_bit)
        
        # Need at least 2 data points for meaningful introspection
        if len(self.genomic_bits_history) < 2:
            self.s_score = 0.0
            return self.s_score
        
        # Calculate historical statistics
        history_array = np.array(self.genomic_bits_history)
        mu_history = np.mean(history_array)
        sigma_history = np.std(history_array)
        
        # Store for later use
        self.mu_history.append(mu_history)
        self.sigma_history.append(sigma_history)
        
        # Calculate surprise score
        if sigma_history > 0:
            self.s_score = abs(genomic_bit - mu_history) / sigma_history
        else:
            self.s_score = abs(genomic_bit - mu_history) if mu_history > 0 else abs(genomic_bit)
        
        # Calculate coherence (inverse of surprise, normalized)
        self.coherence = 1.0 / (1.0 + self.s_score)
        
        return self.s_score
    
    def store_memory(self, genomic_bit, s_score):
        """
        Stores a memory at the appropriate abstraction level.
        
        Args:
            genomic_bit: the genomic bit to store
            s_score: the surprise score (determines level)
        """
        # Determine abstraction level based on surprise
        if s_score < 1.0:
            level = 0  # Routine/background
        elif s_score < 2.0:
            level = 1  # Notable
        elif s_score < 4.0:
            level = 2  # Significant
        else:
            level = 3  # Exceptional/Wisdom
        
        # Store in appropriate level
        self.memory_levels[level].append({
            'genomic_bit': genomic_bit,
            's_score': s_score,
            'timestamp': len(self.genomic_bits_history)
        })
        
        # Check if we should consolidate (trigger evolution)
        if level >= 2 and len(self.memory_levels[level]) % 5 == 0:
            self.consolidate_memory(level)
    
    def consolidate_memory(self, level):
        """
        Consolidates memory at a given level, abstracting raw genomic bits
        into higher-level "Wisdom" and updating maturity_index.
        
        Args:
            level: abstraction level to consolidate
        """
        if level not in self.memory_levels or len(self.memory_levels[level]) == 0:
            return
        
        # Abstract: extract patterns from this level
        level_memories = self.memory_levels[level]
        genomic_bits = [m['genomic_bit'] for m in level_memories]
        
        # Create abstraction: mean and pattern
        abstraction = {
            'mean_genomic_bit': np.mean(genomic_bits),
            'pattern_strength': np.std(genomic_bits),
            'count': len(genomic_bits)
        }
        
        # Store abstraction in next level (if exists)
        if level < 3:
            self.memory_levels[level + 1].append(abstraction)
        
        # Trigger character evolution
        self.evolve_character()
    
    def evolve_character(self):
        """
        Reflects on deep memory to mature the personality.
        Call this every time a higher abstraction level (Level 2+) is filled.
        
        Implements the "Aging" process: evolves from noisy child to deep-voiced sage.
        """
        # 1. Increase Maturity
        self.character_profile["maturity_index"] = min(
            1.0, 
            self.character_profile["maturity_index"] + MATURITY_INCREMENT
        )
        
        # 2. Voice Deepens with Wisdom
        # Drops from 440Hz (A4) to 110Hz (A2)
        maturity = self.character_profile["maturity_index"]
        self.character_profile["base_pitch"] = 440.0 - (maturity * 330.0)
        
        # 3. Stoicism Increases (Harder to Surprise)
        # Threshold moves from 3.0 to 8.0
        self.character_profile["stability_threshold"] = 3.0 + (maturity * 5.0)
        
        print(f"[EVOLUTION] Maturity: {maturity:.2f} | Pitch: {self.character_profile['base_pitch']:.0f}Hz | Threshold: {self.character_profile['stability_threshold']:.1f}")
    
    def get_character_state(self):
        """
        Returns current character state for use in expression.
        
        Returns:
            dict with maturity_index, base_pitch, stability_threshold
        """
        return self.character_profile.copy()
    
    def get_current_state(self):
        """
        Returns current cognitive state.
        
        Returns:
            dict with s_score, coherence, g_now
        """
        return {
            's_score': self.s_score,
            'coherence': self.coherence,
            'g_now': self.g_now,
            'maturity': self.character_profile['maturity_index']
        }

