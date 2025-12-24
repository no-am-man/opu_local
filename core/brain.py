"""
The Brain: Core Cognitive Processing (v3.4).
Implements 8-Layer Fractal Memory (1s to 10y) and Emotional Persistence.
"""

import numpy as np
import time
from dataclasses import dataclass
from config import (
    MATURITY_INCREMENT,
    BRAIN_CHILD_PITCH, BRAIN_SAGE_PITCH, BRAIN_STABILITY_THRESHOLD,
    BRAIN_EMOTION_DEFAULT_INTENSITY, BRAIN_EMOTION_DEFAULT_LABEL,
    BRAIN_MAX_MEMORY_LEVEL, BRAIN_EVOLUTION_MIN_LEVEL,
    BRAIN_DEFAULT_CONSOLIDATION_RATIO, BRAIN_PATTERN_STRENGTH_MIN_SAMPLES,
    BRAIN_CONSOLIDATION_RATIO_L0, BRAIN_CONSOLIDATION_RATIO_L1,
    BRAIN_CONSOLIDATION_RATIO_L2, BRAIN_CONSOLIDATION_RATIO_L3,
    BRAIN_CONSOLIDATION_RATIO_L4, BRAIN_CONSOLIDATION_RATIO_L5,
    BRAIN_CONSOLIDATION_RATIO_L6, BRAIN_CONSOLIDATION_RATIO_L7,
    BRAIN_DEFAULT_SENSE_LABEL
)
from core.patterns.maturity_state import MaturityContext


@dataclass
class AbstractionData:
    """Data for creating a memory abstraction."""
    mean_bit: float
    pattern_strength: float
    emotion: dict
    count: int
    sense_labels: list


@dataclass
class MemoryItemData:
    """Data for creating a memory item."""
    genomic_bit: float
    s_score: float
    sense_label: str
    emotion_vector: dict
    timestamp: float


@dataclass
class StructuralAbstraction:
    """Result of structural abstraction calculation."""
    mean_bit: float
    pattern_strength: float


@dataclass
class EmotionData:
    """Extracted emotion data from emotion list."""
    intensities: list
    labels: list


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
            "base_pitch": BRAIN_CHILD_PITCH,
            "stability_threshold": BRAIN_STABILITY_THRESHOLD
        }
        
        # Consolidation Ratios (How many items of Level N make 1 item of Level N+1)
        # Based on roughly converting seconds -> minutes -> hours etc.
        # Assuming input rate of ~20Hz (50ms per cycle)
        self.consolidation_ratios = {
            0: BRAIN_CONSOLIDATION_RATIO_L0,  # 20 raw inputs (~1s) -> 1 L1 item
            1: BRAIN_CONSOLIDATION_RATIO_L1,  # 60 L1 items (1m) -> 1 L2 item
            2: BRAIN_CONSOLIDATION_RATIO_L2,  # 60 L2 items (1h) -> 1 L3 item
            3: BRAIN_CONSOLIDATION_RATIO_L3,  # 24 L3 items (1d) -> 1 L4 item
            4: BRAIN_CONSOLIDATION_RATIO_L4,  # 7  L4 items (1w) -> 1 L5 item
            5: BRAIN_CONSOLIDATION_RATIO_L5,  # 4  L5 items (1mo)-> 1 L6 item
            6: BRAIN_CONSOLIDATION_RATIO_L6,  # 12 L6 items (1y) -> 1 L7 item
            7: BRAIN_CONSOLIDATION_RATIO_L7   # 10 L7 items -> Full Wisdom
        }
    
    def store_memory(self, genomic_bit, s_score, sense_label=BRAIN_DEFAULT_SENSE_LABEL, emotion=None, timestamp=None):
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
        
        timestamp = self._get_timestamp(timestamp)
        emotion_vector = self._normalize_emotion(emotion)
        memory_data = MemoryItemData(
            genomic_bit=genomic_bit,
            s_score=s_score,
            sense_label=sense_label,
            emotion_vector=emotion_vector,
            timestamp=timestamp
        )
        memory_item = self._create_memory_item(memory_data)
        
        self.memory_levels[level].append(memory_item)
        self._check_and_consolidate_if_needed(level)
    
    def _get_timestamp(self, timestamp):
        """Get timestamp, defaulting to current time if None."""
        return timestamp if timestamp is not None else time.time()
    
    def _create_memory_item(self, data: MemoryItemData):
        """Create memory item dictionary from memory item data."""
        return {
            'genomic_bit': data.genomic_bit,
            's_score': data.s_score,
            'sense': data.sense_label,
            'emotion': data.emotion_vector,
            'timestamp': data.timestamp
        }
    
    def _check_and_consolidate_if_needed(self, level):
        """Check if consolidation threshold reached and consolidate if needed."""
        ratio = self._get_consolidation_ratio(level)
        if len(self.memory_levels[level]) >= ratio:
            self.consolidate_memory(level)
    
    def consolidate_memory(self, level):
        """
        Compresses lower-level memories into a higher-level abstraction.
        Preserves Emotional Context (v3.4).
        """
        if not self._can_consolidate(level):
            return

        chunk = self._extract_consolidation_chunk(level)
        if not chunk:
            return
        
        bits = self._extract_genomic_bits(chunk)
        if not bits:
            return
        
        abstraction_result = self._calculate_structural_abstraction(bits)
        consolidated_emotion = self._consolidate_emotions(chunk)
        sense_labels = self._collect_sense_labels(chunk)
        abstraction_data = AbstractionData(
            mean_bit=abstraction_result.mean_bit,
            pattern_strength=abstraction_result.pattern_strength,
            emotion=consolidated_emotion,
            count=len(chunk),
            sense_labels=sense_labels
        )
        abstraction = self._create_abstraction(abstraction_data)
        
        self.memory_levels[level + 1].append(abstraction)
        self._check_recursive_consolidation(level + 1)
        self._trigger_evolution_if_needed(level + 1)
    
    def _can_consolidate(self, level):
        """Check if consolidation is possible for the given level."""
        if level >= BRAIN_MAX_MEMORY_LEVEL:
            return False
        chunk_size = self._get_consolidation_ratio(level)
        return len(self.memory_levels[level]) >= chunk_size
    
    def _extract_consolidation_chunk(self, level):
        """Extract and remove chunk from current level."""
        chunk_size = self._get_consolidation_ratio(level)
        chunk = self.memory_levels[level][:chunk_size]
        self.memory_levels[level] = self.memory_levels[level][chunk_size:]
        return chunk
    
    def _get_consolidation_ratio(self, level):
        """Get consolidation ratio for given level."""
        return self.consolidation_ratios.get(level, BRAIN_DEFAULT_CONSOLIDATION_RATIO)
    
    def _extract_genomic_bits(self, chunk):
        """Extract genomic bits from memory chunk."""
        bits = []
        for m in chunk:
            bit = self._get_genomic_bit_from_memory(m)
            if bit is not None:
                bits.append(bit)
        return bits
    
    def _get_genomic_bit_from_memory(self, memory):
        """Extract genomic bit from memory item, checking both formats."""
        if 'genomic_bit' in memory:
            return memory['genomic_bit']
        if 'mean_genomic_bit' in memory:
            return memory['mean_genomic_bit']
        return None
    
    def _calculate_structural_abstraction(self, bits):
        """Calculate mean and pattern strength from genomic bits."""
        mean_bit = np.mean(bits)
        pattern_strength = np.std(bits) if len(bits) > BRAIN_PATTERN_STRENGTH_MIN_SAMPLES else 0.0
        return StructuralAbstraction(mean_bit=mean_bit, pattern_strength=pattern_strength)
    
    def _consolidate_emotions(self, chunk):
        """Consolidate emotions from memory chunk into single emotion vector."""
        emotions = self._extract_emotions_from_chunk(chunk)
        
        if not emotions:
            return self._create_default_emotion()
        
        emotion_data = self._extract_emotion_data(emotions)
        avg_intensity = self._calculate_average_intensity(emotion_data.intensities)
        dominant_label = self._find_dominant_emotion_label(emotion_data.labels)
        
        return {
            'intensity': float(avg_intensity),
            'label': dominant_label
        }
    
    def _extract_emotions_from_chunk(self, chunk):
        """Extract all emotions from memory chunk."""
        return [m.get('emotion') for m in chunk if 'emotion' in m]
    
    def _create_default_emotion(self):
        """Create default emotion vector when no emotions found."""
        return {
            'intensity': BRAIN_EMOTION_DEFAULT_INTENSITY,
            'label': BRAIN_EMOTION_DEFAULT_LABEL
        }
    
    def _extract_emotion_data(self, emotions):
        """Extract intensity and label data from emotion list."""
        intensities = []
        labels = []
        for em in emotions:
            if isinstance(em, dict):
                intensities.append(em.get('intensity', BRAIN_EMOTION_DEFAULT_INTENSITY))
                labels.append(em.get('label', BRAIN_EMOTION_DEFAULT_LABEL))
        return EmotionData(intensities=intensities, labels=labels)
    
    def _calculate_average_intensity(self, intensities):
        """Calculate average intensity from intensity list."""
        return np.mean(intensities) if intensities else BRAIN_EMOTION_DEFAULT_INTENSITY
    
    def _find_dominant_emotion_label(self, labels):
        """Find the most common emotion label from label list."""
        if not labels:
            return BRAIN_EMOTION_DEFAULT_LABEL
        return self._get_most_frequent_label(labels)
    
    def _get_most_frequent_label(self, labels):
        """Get the most frequently occurring label from the list."""
        unique_labels = set(labels)
        return max(unique_labels, key=labels.count)
    
    def _collect_sense_labels(self, chunk):
        """Collect unique sense labels from memory chunk."""
        sense_labels = self._extract_sense_labels_from_chunk(chunk)
        return self._get_unique_sense_labels(sense_labels)
    
    def _extract_sense_labels_from_chunk(self, chunk):
        """Extract all sense labels from memory chunk."""
        return [m['sense'] for m in chunk if 'sense' in m]
    
    def _get_unique_sense_labels(self, sense_labels):
        """Get unique sense labels, defaulting to default label if empty."""
        return list(set(sense_labels)) if sense_labels else [BRAIN_DEFAULT_SENSE_LABEL]
    
    def _create_abstraction(self, data: AbstractionData):
        """Create abstracted memory token from abstraction data."""
        return {
            'mean_genomic_bit': float(data.mean_bit),
            'pattern_strength': float(data.pattern_strength),
            'emotion': data.emotion,
            'count': data.count,
            'senses': data.sense_labels,
            'timestamp': time.time()
        }
    
    def _check_recursive_consolidation(self, level):
        """Check if next level should also consolidate (cascade effect)."""
        next_ratio = self._get_consolidation_ratio(level)
        if len(self.memory_levels[level]) >= next_ratio:
            self.consolidate_memory(level)
    
    def _trigger_evolution_if_needed(self, level):
        """Trigger character evolution if minimum level reached."""
        if level >= BRAIN_EVOLUTION_MIN_LEVEL:
            self.evolve_character()
    
    def _normalize_emotion(self, emotion):
        """Normalize emotion to standard format (intensity + label)."""
        if emotion is None:
            return self._create_default_emotion()
        
        if self._is_emotion_confidence_format(emotion):
            return self._normalize_confidence_format(emotion)
        
        return self._normalize_standard_format(emotion)
    
    def _is_emotion_confidence_format(self, emotion):
        """Check if emotion uses confidence format (emotion + confidence keys)."""
        return 'emotion' in emotion and 'confidence' in emotion
    
    def _normalize_confidence_format(self, emotion):
        """Normalize emotion from confidence format (emotion + confidence)."""
        return {
            'intensity': emotion.get('confidence', BRAIN_EMOTION_DEFAULT_INTENSITY),
            'label': emotion.get('emotion', BRAIN_EMOTION_DEFAULT_LABEL)
        }
    
    def _normalize_standard_format(self, emotion):
        """Normalize emotion from standard format (intensity + label)."""
        return {
            'intensity': self._extract_intensity(emotion),
            'label': self._extract_label(emotion)
        }
    
    def _extract_intensity(self, emotion):
        """Extract intensity from emotion, with fallback to confidence."""
        return emotion.get('intensity') or emotion.get('confidence', BRAIN_EMOTION_DEFAULT_INTENSITY)
    
    def _extract_label(self, emotion):
        """Extract label from emotion, with fallback to emotion key."""
        return emotion.get('label') or emotion.get('emotion', BRAIN_EMOTION_DEFAULT_LABEL)

    def evolve_character(self):
        """
        Updates Maturity Index based on the depth of the stack.
        Uses State Pattern (MaturityContext) for maturity management.
        """
        profile = self._update_character_profile()
        self._log_evolution(profile)
    
    def _update_character_profile(self):
        """Update character profile using State Pattern."""
        profile = self.maturity_context.update_from_memory_levels(self.memory_levels)
        self.character_profile.update(profile)
        return profile
    
    def _log_evolution(self, profile):
        """Log character evolution details."""
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
