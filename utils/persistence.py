"""
Persistence module for saving and loading OPU state.
Allows the OPU to resume learning from where it left off.
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
from config import (
    PERSISTENCE_DEFAULT_DAY_COUNTER, PERSISTENCE_DEFAULT_S_SCORE,
    PERSISTENCE_STATE_VERSION, PERSISTENCE_TEMP_FILE_SUFFIX,
    BRAIN_MAX_MEMORY_LEVEL, BRAIN_DEFAULT_SENSE_LABEL
)

# Constants
DEFAULT_STATE_FILE = "opu_state.json"
EMPTY_LIST = []


class OPUPersistence:
    """
    Handles saving and loading of OPU state to/from disk.
    """
    
    def __init__(self, state_file: Union[str, Path] = DEFAULT_STATE_FILE):
        """
        Initialize persistence manager.
        
        Args:
            state_file: Path to the state file (default: opu_state.json in current directory)
        """
        self.state_file = Path(state_file)
        self.state_dir = self.state_file.parent
        # Ensure directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)
    
    def save_state(self, cortex, phoneme_analyzer, day_counter=None, last_abstraction_times=None):
        """
        Save OPU state to disk.
        
        Args:
            cortex: OrthogonalProcessingUnit instance
            phoneme_analyzer: PhonemeAnalyzer instance
            day_counter: Current day counter
            last_abstraction_times: Dict of last abstraction times per level (optional)
        """
        try:
            state = self._build_state_dict(cortex, phoneme_analyzer, day_counter, last_abstraction_times)
            self._write_state_to_file(state)
            print(f"[PERSISTENCE] State saved to {self.state_file}")
            return True
            
        except Exception as e:
            self._handle_save_error(e)
            return False
    
    def _build_state_dict(self, cortex, phoneme_analyzer, day_counter, last_abstraction_times):
        """Build the state dictionary to save."""
        return {
            'version': PERSISTENCE_STATE_VERSION,
            'day_counter': day_counter if day_counter is not None else PERSISTENCE_DEFAULT_DAY_COUNTER,
            'cortex': self._serialize_cortex_state(cortex),
            'phonemes': self._serialize_phoneme_history(phoneme_analyzer),
            'abstraction_timers': self._serialize_abstraction_timers(last_abstraction_times)
        }
    
    def _serialize_cortex_state(self, cortex):
        """Serialize cortex state."""
        return {
            'character_profile': self._convert_numpy_types_to_native(cortex.character_profile),
            'memory_levels': self._serialize_memory_levels(cortex.memory_levels),
            'genomic_bits_history': self._serialize_array(cortex.genomic_bits_history),
            'mu_history': self._serialize_array(cortex.mu_history),
            'sigma_history': self._serialize_array(cortex.sigma_history),
            'current_state': self._serialize_current_state(cortex),
            'emotion_history': self._convert_numpy_types_to_native(getattr(cortex, 'emotion_history', []))
        }
    
    def _serialize_current_state(self, cortex):
        """Serialize current cognitive state."""
        return self._convert_numpy_types_to_native({
            'g_now': cortex.g_now,
            's_score': cortex.s_score,
            'coherence': cortex.coherence
        })
    
    def _serialize_abstraction_timers(self, last_abstraction_times):
        """Serialize abstraction timers."""
        return self._convert_numpy_types_to_native(last_abstraction_times) if last_abstraction_times else None
    
    def _write_state_to_file(self, state):
        """Write state to file atomically."""
        temp_file = self.state_file.with_suffix(PERSISTENCE_TEMP_FILE_SUFFIX)
        with open(temp_file, 'w') as f:
            json.dump(state, f, indent=2)
        temp_file.replace(self.state_file)
    
    def _handle_save_error(self, error):
        """Handle save error with traceback."""
        import traceback
        print(f"[PERSISTENCE] Error saving state: {error}")
        print(f"[PERSISTENCE] Traceback:")
        traceback.print_exc()
    
    def load_state(self, cortex, phoneme_analyzer) -> Tuple[bool, int, Optional[Dict[int, float]]]:
        """
        Load OPU state from disk.
        
        Args:
            cortex: OrthogonalProcessingUnit instance to populate
            phoneme_analyzer: PhonemeAnalyzer instance to populate
            
        Returns:
            tuple: (success: bool, day_counter: int, last_abstraction_times: dict or None)
        """
        if not self.state_file.exists():
            print(f"[PERSISTENCE] No saved state found at {self.state_file}")
            return False, PERSISTENCE_DEFAULT_DAY_COUNTER, None
        
        try:
            state = self._read_state_from_file()
            self._load_cortex_state(state, cortex)
            self._load_phoneme_state(state, phoneme_analyzer)
            day_counter = state.get('day_counter', PERSISTENCE_DEFAULT_DAY_COUNTER)
            last_abstraction_times = self._deserialize_abstraction_timers(state)
            self._print_load_summary(cortex, phoneme_analyzer, day_counter, last_abstraction_times)
            return True, day_counter, last_abstraction_times
            
        except Exception as e:
            self._handle_load_error(e)
            return False, PERSISTENCE_DEFAULT_DAY_COUNTER, None
    
    def _handle_load_error(self, error: Exception) -> None:
        """Handle load error with message."""
        print(f"[PERSISTENCE] Error loading state: {error}")
    
    def _read_state_from_file(self):
        """Read state from JSON file."""
        with open(self.state_file, 'r') as f:
            return json.load(f)
    
    def _load_cortex_state(self, state, cortex):
        """Load cortex state from state dictionary."""
        if 'cortex' not in state:
            return
        
        cortex_data = state['cortex']
        # Restore memory levels FIRST, then recalculate character profile
        # This ensures maturity is calculated based on actual memory state
        self._restore_memory_levels(cortex_data, cortex)
        # Restore character profile (maturity fields will be recalculated below)
        self._restore_character_profile(cortex_data, cortex)
        # Recalculate maturity based on loaded memory levels
        # This fixes the issue where maturity_index stays at 0.0 after loading state
        cortex.brain.evolve_character()
        self._restore_history(cortex_data, cortex)
        self._restore_current_state(cortex_data, cortex)
        self._restore_emotion_history(cortex_data, cortex)
    
    def _restore_character_profile(self, cortex_data, cortex):
        """Restore character profile."""
        if 'character_profile' in cortex_data:
            cortex.character_profile.update(cortex_data['character_profile'])
    
    def _restore_memory_levels(self, cortex_data, cortex):
        """Restore memory levels."""
        if 'memory_levels' in cortex_data:
            cortex.brain.memory_levels = self._deserialize_memory_levels(cortex_data['memory_levels'])
    
    def _restore_history(self, cortex_data, cortex):
        """Restore introspection history."""
        if 'genomic_bits_history' in cortex_data:
            cortex.audio_cortex.genomic_bits_history = self._deserialize_array(cortex_data['genomic_bits_history'])
        if 'mu_history' in cortex_data:
            cortex.audio_cortex.mu_history = self._deserialize_array(cortex_data['mu_history'])
        if 'sigma_history' in cortex_data:
            cortex.audio_cortex.sigma_history = self._deserialize_array(cortex_data['sigma_history'])
    
    def _restore_current_state(self, cortex_data, cortex):
        """Restore current cognitive state."""
        if 'current_state' in cortex_data:
            cs = cortex_data['current_state']
            cortex.g_now = cs.get('g_now')
            cortex.s_score = cs.get('s_score', PERSISTENCE_DEFAULT_S_SCORE)
            cortex.coherence = cs.get('coherence', PERSISTENCE_DEFAULT_S_SCORE)
    
    def _restore_emotion_history(self, cortex_data, cortex):
        """Restore emotion history."""
        if 'emotion_history' in cortex_data:
            cortex.emotion_history = cortex_data['emotion_history']
        else:
            cortex.emotion_history = []
    
    def _load_phoneme_state(self, state, phoneme_analyzer):
        """Load phoneme analyzer state."""
        if 'phonemes' not in state:
            return
        
        phoneme_data = state['phonemes']
        if 'history' in phoneme_data:
            phoneme_analyzer.phoneme_history = phoneme_data['history']
        if 'speech_threshold' in phoneme_data:
            phoneme_analyzer.speech_threshold = phoneme_data['speech_threshold']
    
    def _deserialize_abstraction_timers(self, state):
        """Deserialize abstraction timers."""
        if 'abstraction_timers' not in state or not state['abstraction_timers']:
            return None
        
        last_abstraction_times = {}
        for level_str, timestamp in state['abstraction_timers'].items():
            level = int(level_str)
            if 0 <= level <= BRAIN_MAX_MEMORY_LEVEL:
                last_abstraction_times[level] = float(timestamp)
        return last_abstraction_times
    
    def _print_load_summary(self, cortex, phoneme_analyzer, day_counter: int, 
                           last_abstraction_times: Optional[Dict[int, float]]) -> None:
        """Print summary of loaded state."""
        print(f"[PERSISTENCE] State loaded from {self.state_file}")
        self._print_maturity_info(cortex)
        self._print_memory_info(cortex)
        self._print_phoneme_info(phoneme_analyzer)
        self._print_emotion_info(cortex)
        print(f"  Day: {day_counter}")
        self._print_abstraction_timers(last_abstraction_times)
    
    def _print_maturity_info(self, cortex) -> None:
        """Print maturity level information."""
        maturity_level = cortex.character_profile.get('maturity_level', 0)
        maturity_index = cortex.character_profile['maturity_index']
        print(f"  Maturity Level: {maturity_level} | Index: {maturity_index:.2f}")
    
    def _print_memory_info(self, cortex) -> None:
        """Print memory level information."""
        memory_counts = [f"L{i}={len(cortex.memory_levels.get(i, []))}" 
                        for i in range(BRAIN_MAX_MEMORY_LEVEL + 1)]
        print(f"  Memory: " + " | ".join(memory_counts))
    
    def _print_phoneme_info(self, phoneme_analyzer) -> None:
        """Print phoneme history information."""
        print(f"  Phonemes: {len(phoneme_analyzer.phoneme_history)}")
    
    def _print_emotion_info(self, cortex) -> None:
        """Print emotion history information."""
        emotion_history = getattr(cortex, 'emotion_history', [])
        print(f"  Emotions: {len(emotion_history)} detected emotions")
    
    def _print_abstraction_timers(self, last_abstraction_times: Optional[Dict[int, float]]) -> None:
        """Print abstraction timers information."""
        if last_abstraction_times:
            print(f"  Abstraction Timers: Restored for {len(last_abstraction_times)} levels")
    
    def _serialize_phoneme_history(self, phoneme_analyzer):
        """
        Serialize phoneme history, handling numpy types in s_score and pitch.
        
        Args:
            phoneme_analyzer: PhonemeAnalyzer instance
            
        Returns:
            dict with serialized phoneme data
        """
        # Use robust conversion function for phoneme history
        phoneme_history_serialized = self._convert_numpy_types_to_native(phoneme_analyzer.phoneme_history)
        
        return {
            'history': phoneme_history_serialized,
            'speech_threshold': self._convert_numpy_types_to_native(phoneme_analyzer.speech_threshold)
        }
    
    def _convert_numpy_types_to_native(self, obj: Any) -> Any:
        """
        Recursively convert numpy types to native Python types.
        Handles all numpy scalar types including float32, float64, int32, int64, etc.
        NumPy 2.0 compatible (doesn't use deprecated np.float_ or np.int_).
        
        Args:
            obj: Any object that might contain numpy types
            
        Returns:
            Object with all numpy types converted to native Python types
        """
        if obj is None:
            return None
        
        if isinstance(obj, np.ndarray):
            return self._convert_array(obj)
        
        if isinstance(obj, (list, tuple)):
            return self._convert_sequence(obj)
        
        if isinstance(obj, dict):
            return self._convert_dict(obj)
        
        if isinstance(obj, np.generic):
            return self._convert_numpy_scalar(obj)
        
        # Handle numpy number types (additional check for defensive programming)
        if isinstance(obj, (np.integer, np.floating)):  # pragma: no cover
            return float(obj) if np.issubdtype(type(obj), np.floating) else int(obj)
        
        return obj
    
    def _convert_array(self, arr: np.ndarray) -> list:
        """Convert numpy array to list with recursive conversion."""
        return [self._convert_numpy_types_to_native(x) for x in arr.tolist()]
    
    def _convert_sequence(self, seq: Union[list, tuple]) -> list:
        """Convert list or tuple to list with recursive conversion."""
        return [self._convert_numpy_types_to_native(x) for x in seq]
    
    def _convert_dict(self, dct: dict) -> dict:
        """Convert dictionary with recursive conversion."""
        return {k: self._convert_numpy_types_to_native(v) for k, v in dct.items()}
    
    def _convert_numpy_scalar(self, scalar: np.generic) -> Union[int, float, str]:
        """Convert numpy scalar to native Python type."""
        if np.issubdtype(type(scalar), np.integer):
            return int(scalar)
        elif np.issubdtype(type(scalar), np.floating):
            return float(scalar)
        else:
            # Fallback conversion chain
            return self._convert_scalar_fallback(scalar)
    
    def _convert_scalar_fallback(self, scalar: np.generic) -> Union[float, int, str]:
        """Fallback conversion for numpy scalars that aren't integer/floating."""
        try:
            return float(scalar)
        except (ValueError, TypeError):
            try:
                return int(scalar)
            except (ValueError, TypeError):
                return str(scalar)  # Last resort: convert to string
    
    def _serialize_memory_levels(self, memory_levels):
        """
        Serialize memory levels to JSON-serializable format.
        
        Args:
            memory_levels: dict of memory levels
            
        Returns:
            dict with serialized memory levels
        """
        serialized = {}
        for level, memories in memory_levels.items():
            serialized[str(level)] = []
            for mem in memories:
                serialized_mem = {}
                for key, value in mem.items():
                    # Handle all numpy types (NumPy 2.0 compatible)
                    serialized_mem[key] = self._convert_numpy_types_to_native(value)
                serialized[str(level)].append(serialized_mem)
        return serialized
    
    def _deserialize_memory_levels(self, serialized):
        """
        Deserialize memory levels from JSON format.
        
        Args:
            serialized: dict with serialized memory levels
            
        Returns:
            dict with deserialized memory levels
        """
        # Support old format (4 levels), new format (6 levels), and current format (7 levels)
        memory_levels = {i: [] for i in range(8)}  # 8 levels (0-7)
        for level_str, memories in serialized.items():
            level = int(level_str)
            if level in memory_levels:
                # Add backward compatibility: if memories don't have 'sense' field,
                # add default sense label for old state files
                memories = self._add_default_sense_to_memories(memories)
                memory_levels[level] = memories
        return memory_levels
    
    def _add_default_sense_to_memories(self, memories: list) -> list:
        """Add default sense label to memories missing it (backward compatibility)."""
        for mem in memories:
            if isinstance(mem, dict) and 'sense' not in mem:
                mem['sense'] = BRAIN_DEFAULT_SENSE_LABEL
        return memories
    
    def _serialize_array(self, arr):
        """
        Serialize numpy array or list to JSON-serializable format.
        
        Args:
            arr: numpy array or list
            
        Returns:
            list
        """
        # Use the robust conversion function
        return self._convert_numpy_types_to_native(arr)
    
    def _deserialize_array(self, data: Any) -> list:
        """
        Deserialize array from JSON format.
        
        Args:
            data: list or array data
            
        Returns:
            list (empty list if data is falsy or not a list)
        """
        if isinstance(data, list):
            return [float(x) for x in data]
        return data if data else EMPTY_LIST

