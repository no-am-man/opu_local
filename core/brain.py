"""
The Brain: Core Cognitive Processing (v3.4).
Implements 8-Layer Fractal Memory (1s to 10y) and Emotional Persistence.
"""

import numpy as np
import time
from config import MATURITY_INCREMENT
from core.patterns.maturity_state import MaturityContext


class Brain:
    """
    The core brain: Memory abstraction and character evolution.
    Manages 8-level memory hierarchy (1s to 10y) and Emotional Memory.
    """
    
    def __init__(self):
        # 8 Memory Layers
        # L0=1s, L1=1m, L2=1h, L3=1d, L4=1w, L5=1mo, L6=1y, L7=10y
        self.memory_levels = {i: [] for i in range(8)}
        
        # Use State Pattern for maturity management
        self.maturity_context = MaturityContext()
        
        # Character profile that evolves over time (updated by maturity_context)
        self.character_profile = {
            "maturity_index": 0.0,  # 0.0 = Child, 1.0 = Sage
            "maturity_level": 0,    # Current highest active level (0-7)
            "base_pitch": 440.0,    # Child (440Hz) -> Sage (110Hz)
            "stability_threshold": 3.0
        }
        
        # Consolidation Ratios (How many items of Level N make 1 item of Level N+1)
        # Based on roughly converting seconds -> minutes -> hours etc.
        # Assuming input rate of ~20Hz (50ms per cycle)
        self.consolidation_ratios = {
            0: 20,  # 20 raw inputs (~1s) -> 1 L1 item
            1: 60,  # 60 L1 items (1m) -> 1 L2 item
            2: 60,  # 60 L2 items (1h) -> 1 L3 item
            3: 24,  # 24 L3 items (1d) -> 1 L4 item
            4: 7,   # 7  L4 items (1w) -> 1 L5 item
            5: 4,   # 4  L5 items (1mo)-> 1 L6 item
            6: 12,  # 12 L6 items (1y) -> 1 L7 item
            7: 10   # 10 L7 items -> Full Wisdom
        }
    
    def store_memory(self, genomic_bit, s_score, sense_label="UNKNOWN", emotion=None, timestamp=None):
        """
        Stores a raw memory at Level 0 (1 Second).
        Includes v3.4 Emotional Vector support.
        
        CRITICAL: All raw sensory data MUST enter at Level 0.
        Wisdom cannot be "jumped" to via high surprise scores.
        Evolution must be earned through time-based consolidation, not trauma.
        """
        # --- LAW OF STRICT ENTRY (v3.3) ---
        # All inputs MUST start at Level 0. No skipping to wisdom.
        level = 0
        
        if timestamp is None:
            timestamp = time.time()
            
        # Default emotion if none provided (Neutral)
        emotion_vector = None
        if emotion is not None:
            # Support both old format (dict with 'emotion' and 'confidence')
            # and new format (dict with 'intensity' and 'label')
            if 'emotion' in emotion and 'confidence' in emotion:
                # Old format - convert to new format
                emotion_vector = {
                    'intensity': emotion.get('confidence', 0.0),
                    'label': emotion.get('emotion', 'neutral')
                }
            else:
                # New format or partial format
                emotion_vector = {
                    'intensity': emotion.get('intensity', emotion.get('confidence', 0.0)),
                    'label': emotion.get('label', emotion.get('emotion', 'neutral'))
                }
        else:
            # Default emotion (Neutral)
            emotion_vector = {'intensity': 0.0, 'label': 'neutral'}

        # Store Memory Object
        memory_item = {
            'genomic_bit': genomic_bit,
            's_score': s_score,
            'sense': sense_label,
            'emotion': emotion_vector,  # v3.4 Feature
            'timestamp': timestamp
        }
        
        self.memory_levels[level].append(memory_item)
        
        # Check Consolidation
        ratio = self.consolidation_ratios.get(level, 20)
        if len(self.memory_levels[level]) >= ratio:
            self.consolidate_memory(level)

    def consolidate_memory(self, level):
        """
        Compresses lower-level memories into a higher-level abstraction.
        Preserves Emotional Context (v3.4).
        """
        # Safety check
        if level >= 7:
            return  # Max level reached
        
        # Get the chunk to consolidate
        chunk_size = self.consolidation_ratios.get(level, 10)
        if len(self.memory_levels[level]) < chunk_size:
            return

        # Extract chunk and remove from current level
        chunk = self.memory_levels[level][:chunk_size]
        self.memory_levels[level] = self.memory_levels[level][chunk_size:]
        
        # --- ABSTRACTION LOGIC ---
        
        # 1. Structural Data (Genomic Bits)
        bits = []
        for m in chunk:
            if 'genomic_bit' in m:
                bits.append(m['genomic_bit'])
            elif 'mean_genomic_bit' in m:
                bits.append(m['mean_genomic_bit'])
        
        if len(bits) == 0:
            return  # No valid genomic bits to consolidate
        
        mean_bit = np.mean(bits)
        pattern_strength = np.std(bits) if len(bits) > 1 else 0.0
        
        # 2. Emotional Data (v3.4 Consolidation)
        # We calculate the "Average Mood" of this time block
        emotions = [m.get('emotion') for m in chunk if 'emotion' in m]
        
        if emotions:
            # Calculate average intensity
            intensities = []
            labels = []
            for em in emotions:
                if isinstance(em, dict):
                    intensities.append(em.get('intensity', 0.0))
                    labels.append(em.get('label', 'neutral'))
            
            avg_intensity = np.mean(intensities) if intensities else 0.0
            
            # Determine dominant emotion label (simple voting)
            dominant_label = max(set(labels), key=labels.count) if labels else 'neutral'
            
            consolidated_emotion = {
                'intensity': float(avg_intensity),
                'label': dominant_label
            }
        else:
            consolidated_emotion = {'intensity': 0.0, 'label': 'neutral'}

        # 3. Collect sense labels
        sense_labels = []
        for m in chunk:
            if 'sense' in m:
                sense_labels.append(m['sense'])

        # 4. Create the Abstracted Token
        abstraction = {
            'mean_genomic_bit': float(mean_bit),
            'pattern_strength': float(pattern_strength),  # How chaotic was this period?
            'emotion': consolidated_emotion,  # The "Vibe" of the period
            'count': len(chunk),
            'senses': list(set(sense_labels)) if sense_labels else ['UNKNOWN'],
            'timestamp': time.time()  # Timestamp of consolidation
        }
        
        # Push to next level
        self.memory_levels[level + 1].append(abstraction)
        
        # Recursive check (Cascade effect: L0->L1 might trigger L1->L2)
        next_ratio = self.consolidation_ratios.get(level + 1, 10)
        if len(self.memory_levels[level + 1]) >= next_ratio:
            self.consolidate_memory(level + 1)
            
        # Trigger Evolution (Character Updates)
        # We only evolve if we reach Level 3 (1 Day+) to prevent jitters
        if level + 1 >= 3:
            self.evolve_character()

    def evolve_character(self):
        """
        Updates Maturity Index based on the depth of the stack.
        Uses State Pattern (MaturityContext) for maturity management.
        """
        # Use State Pattern to update character profile
        profile = self.maturity_context.update_from_memory_levels(self.memory_levels)
        self.character_profile.update(profile)
        
        # Get time scale from current state
        time_scale = self.maturity_context.get_time_scale()
        maturity = self.character_profile["maturity_index"]
        
        print(f"[EVOLUTION] Level {profile['maturity_level']} ({time_scale}) | "
              f"Maturity: {maturity:.3f} | "
              f"Voice: {profile['base_pitch']:.1f}Hz | "
              f"Threshold: {profile['stability_threshold']:.1f}")
    
    def get_character_state(self):
        """
        Returns current character state for use in expression.
        
        Returns:
            dict with maturity_index, base_pitch, stability_threshold, maturity_level
        """
        return self.character_profile.copy()
