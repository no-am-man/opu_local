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
        # Memory abstraction layers (6 levels: 0 = 1 minute, 5 = 1 year)
        # Level 0: 1 minute - Immediate/short-term memory
        # Level 1: 1 hour - Short-term patterns
        # Level 2: 1 day - Daily patterns
        # Level 3: 1 week - Weekly patterns
        # Level 4: 1 month - Monthly patterns
        # Level 5: 1 year - Yearly patterns/wisdom
        self.memory_levels = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        
        # Character profile that evolves over time
        self.character_profile = {
            "maturity_index": 0.0,  # 0.0 = Child, 1.0 = Sage
            "maturity_level": 0,    # Current maturity level (0-5)
            "base_pitch": 440.0,     # Starts high (Child), drops to 110Hz (Sage)
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
        # Maps to 6 maturity levels (0-5)
        if s_score < 0.5:
            level = 0  # 1 minute - Routine/background
        elif s_score < 1.0:
            level = 1  # 1 hour - Notable
        elif s_score < 2.0:
            level = 2  # 1 day - Significant
        elif s_score < 3.5:
            level = 3  # 1 week - Important
        elif s_score < 5.0:
            level = 4  # 1 month - Exceptional
        else:
            level = 5  # 1 year - Wisdom/Transcendent
        
        # Store in appropriate level
        self.memory_levels[level].append({
            'genomic_bit': genomic_bit,
            's_score': s_score,
            'timestamp': len(self.genomic_bits_history)
        })
        
        # Check if we should consolidate (trigger evolution)
        # Higher levels trigger consolidation more frequently
        consolidation_thresholds = {
            0: 100,  # Level 0: every 100 items (1 minute scale)
            1: 50,   # Level 1: every 50 items (1 hour scale)
            2: 20,   # Level 2: every 20 items (1 day scale)
            3: 10,   # Level 3: every 10 items (1 week scale)
            4: 5,    # Level 4: every 5 items (1 month scale)
            5: 3     # Level 5: every 3 items (1 year scale - wisdom)
        }
        
        threshold = consolidation_thresholds.get(level, 10)
        if len(self.memory_levels[level]) % threshold == 0 and len(self.memory_levels[level]) > 0:
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
        
        # Handle both raw memories (with 'genomic_bit') and abstractions (with 'mean_genomic_bit')
        genomic_bits = []
        for m in level_memories:
            if 'genomic_bit' in m:
                # Raw memory
                genomic_bits.append(m['genomic_bit'])
            elif 'mean_genomic_bit' in m:
                # Abstraction - use mean_genomic_bit as the representative value
                genomic_bits.append(m['mean_genomic_bit'])
            else:
                # Skip if neither key exists
                continue
        
        if len(genomic_bits) == 0:
            return  # No valid genomic bits to consolidate
        
        # Create abstraction: mean and pattern
        abstraction = {
            'mean_genomic_bit': np.mean(genomic_bits),
            'pattern_strength': np.std(genomic_bits) if len(genomic_bits) > 1 else 0.0,
            'count': len(genomic_bits)
        }
        
        # Store abstraction in next level (if exists)
        if level < 5:  # Now we have 6 levels (0-5)
            self.memory_levels[level + 1].append(abstraction)
        
        # Trigger character evolution (only for higher levels)
        if level >= 2:  # Level 2+ (1 day and above) triggers evolution
            self.evolve_character(level)
    
    def evolve_character(self, level=None):
        """
        Reflects on deep memory to mature the personality.
        Call this every time a higher abstraction level (Level 2+) is filled.
        
        Implements the "Aging" process: evolves from noisy child to deep-voiced sage.
        Now supports 6 maturity levels (1 minute to 1 year).
        
        Args:
            level: The abstraction level that triggered evolution (0-5)
        """
        # Update maturity level based on highest level with consolidated memories
        highest_level = 0
        for lvl in range(5, -1, -1):
            if len(self.memory_levels[lvl]) > 0:
                highest_level = lvl
                break
        
        # Maturity level is the highest level reached (0-5)
        self.character_profile["maturity_level"] = highest_level
        
        # Maturity index is a continuous value from 0.0 to 1.0
        # Based on both the level reached and how much of that level is filled
        level_progress = min(1.0, len(self.memory_levels[highest_level]) / 10.0) if highest_level > 0 else 0.0
        base_maturity = highest_level / 5.0  # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0
        self.character_profile["maturity_index"] = min(1.0, base_maturity + (level_progress * 0.2))
        
        # 2. Voice Deepens with Wisdom
        # Drops from 440Hz (A4) to 110Hz (A2) as maturity increases
        maturity = self.character_profile["maturity_index"]
        self.character_profile["base_pitch"] = 440.0 - (maturity * 330.0)
        
        # 3. Stoicism Increases (Harder to Surprise)
        # Threshold moves from 3.0 to 8.0 as maturity increases
        self.character_profile["stability_threshold"] = 3.0 + (maturity * 5.0)
        
        # Get time scale name for the current maturity level
        time_scales = {
            0: "1 minute",
            1: "1 hour",
            2: "1 day",
            3: "1 week",
            4: "1 month",
            5: "1 year"
        }
        time_scale = time_scales.get(highest_level, "unknown")
        
        print(f"[EVOLUTION] Level {highest_level} ({time_scale}) | Maturity: {maturity:.2f} | Pitch: {self.character_profile['base_pitch']:.0f}Hz | Threshold: {self.character_profile['stability_threshold']:.1f}")
    
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

