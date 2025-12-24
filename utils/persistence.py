"""
Persistence module for saving and loading OPU state.
Allows the OPU to resume learning from where it left off.
"""

import json
import os
import numpy as np
from pathlib import Path


class OPUPersistence:
    """
    Handles saving and loading of OPU state to/from disk.
    """
    
    def __init__(self, state_file="opu_state.json"):
        """
        Initialize persistence manager.
        
        Args:
            state_file: Path to the state file (default: opu_state.json in current directory)
        """
        self.state_file = Path(state_file)
        self.state_dir = self.state_file.parent
        # Ensure directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)
    
    def save_state(self, cortex, phoneme_analyzer, day_counter=0, last_abstraction_times=None):
        """
        Save OPU state to disk.
        
        Args:
            cortex: OrthogonalProcessingUnit instance
            phoneme_analyzer: PhonemeAnalyzer instance
            day_counter: Current day counter
            last_abstraction_times: Dict of last abstraction times per level (optional)
        """
        try:
            # Serialize character profile (handle numpy types)
            # Use robust conversion function (NumPy 2.0 compatible)
            character_profile = self._convert_numpy_types_to_native(cortex.character_profile)
            
            state = {
                'version': '1.0',
                'day_counter': day_counter,
                
                # Cortex state
                'cortex': {
                    'character_profile': character_profile,
                    'memory_levels': self._serialize_memory_levels(cortex.memory_levels),
                    'genomic_bits_history': self._serialize_array(cortex.genomic_bits_history),
                    'mu_history': self._serialize_array(cortex.mu_history),
                    'sigma_history': self._serialize_array(cortex.sigma_history),
                    'current_state': self._convert_numpy_types_to_native({
                        'g_now': cortex.g_now,
                        's_score': cortex.s_score,
                        'coherence': cortex.coherence
                    }),
                    # Emotion history (NEW: persist detected emotions)
                    'emotion_history': self._convert_numpy_types_to_native(getattr(cortex, 'emotion_history', []))
                },
                
                # Phoneme analyzer state
                'phonemes': self._serialize_phoneme_history(phoneme_analyzer),
                
                # Abstraction cycle timers (NEW: persist timing state)
                'abstraction_timers': self._convert_numpy_types_to_native(last_abstraction_times) if last_abstraction_times else None
            }
            
            # Write to file atomically
            temp_file = self.state_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=2)
            
            # Atomic replace
            temp_file.replace(self.state_file)
            
            print(f"[PERSISTENCE] State saved to {self.state_file}")
            return True
            
        except Exception as e:
            import traceback
            print(f"[PERSISTENCE] Error saving state: {e}")
            print(f"[PERSISTENCE] Traceback:")
            traceback.print_exc()
            return False
    
    def load_state(self, cortex, phoneme_analyzer):
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
            return False, 0, None
        
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            
            # Load cortex state
            if 'cortex' in state:
                cortex_data = state['cortex']
                
                # Restore character profile
                if 'character_profile' in cortex_data:
                    cortex.character_profile.update(cortex_data['character_profile'])
                
                # Restore memory levels
                # FIX: Set on brain.memory_levels since memory_levels is now a property
                if 'memory_levels' in cortex_data:
                    cortex.brain.memory_levels = self._deserialize_memory_levels(cortex_data['memory_levels'])
                
                # Restore history
                # FIX: Set on audio_cortex since these are now properties
                if 'genomic_bits_history' in cortex_data:
                    cortex.audio_cortex.genomic_bits_history = self._deserialize_array(cortex_data['genomic_bits_history'])
                
                if 'mu_history' in cortex_data:
                    cortex.audio_cortex.mu_history = self._deserialize_array(cortex_data['mu_history'])
                
                if 'sigma_history' in cortex_data:
                    cortex.audio_cortex.sigma_history = self._deserialize_array(cortex_data['sigma_history'])
                
                # Restore current state
                if 'current_state' in cortex_data:
                    cs = cortex_data['current_state']
                    cortex.g_now = cs.get('g_now')
                    cortex.s_score = cs.get('s_score', 0.0)
                    cortex.coherence = cs.get('coherence', 0.0)
                
                # Restore emotion history (NEW: load persisted emotions)
                if 'emotion_history' in cortex_data:
                    cortex.emotion_history = cortex_data['emotion_history']
                else:
                    # Initialize empty emotion history if not present (backward compatibility)
                    cortex.emotion_history = []
            
            # Load phoneme analyzer state
            if 'phonemes' in state:
                phoneme_data = state['phonemes']
                if 'history' in phoneme_data:
                    phoneme_analyzer.phoneme_history = phoneme_data['history']
                if 'speech_threshold' in phoneme_data:
                    phoneme_analyzer.speech_threshold = phoneme_data['speech_threshold']
            
            day_counter = state.get('day_counter', 0)
            
            # Load abstraction cycle timers (NEW: restore timing state)
            last_abstraction_times = None
            if 'abstraction_timers' in state and state['abstraction_timers']:
                last_abstraction_times = {}
                for level_str, timestamp in state['abstraction_timers'].items():
                    level = int(level_str)
                    if 0 <= level <= 7:  # Support all 8 levels
                        last_abstraction_times[level] = float(timestamp)
            
            print(f"[PERSISTENCE] State loaded from {self.state_file}")
            print(f"  Maturity Level: {cortex.character_profile.get('maturity_level', 0)} | Index: {cortex.character_profile['maturity_index']:.2f}")
            print(f"  Memory: " + " | ".join([f"L{i}={len(cortex.memory_levels.get(i, []))}" for i in range(8)]))
            print(f"  Phonemes: {len(phoneme_analyzer.phoneme_history)}")
            print(f"  Emotions: {len(getattr(cortex, 'emotion_history', []))} detected emotions")
            print(f"  Day: {day_counter}")
            if last_abstraction_times:
                print(f"  Abstraction Timers: Restored for {len(last_abstraction_times)} levels")
            
            return True, day_counter, last_abstraction_times
            
        except Exception as e:
            print(f"[PERSISTENCE] Error loading state: {e}")
            return False, 0, None
    
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
    
    def _convert_numpy_types_to_native(self, obj):
        """
        Recursively convert numpy types to native Python types.
        Handles all numpy scalar types including float32, float64, int32, int64, etc.
        NumPy 2.0 compatible (doesn't use deprecated np.float_ or np.int_).
        
        Args:
            obj: Any object that might contain numpy types
            
        Returns:
            Object with all numpy types converted to native Python types
        """
        # Handle None values
        if obj is None:
            return None
        
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return [self._convert_numpy_types_to_native(x) for x in obj.tolist()]
        
        # Handle lists and tuples
        if isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types_to_native(x) for x in obj]
        
        # Handle dictionaries
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types_to_native(v) for k, v in obj.items()}
        
        # Handle numpy scalar types (NumPy 2.0 compatible)
        # np.generic is the base class for all numpy scalars
        if isinstance(obj, np.generic):
            if np.issubdtype(type(obj), np.integer):
                return int(obj)
            elif np.issubdtype(type(obj), np.floating):
                return float(obj)
            else:
                # Fallback: try to convert to float, then int if that fails
                try:
                    return float(obj)
                except (ValueError, TypeError):
                    try:
                        return int(obj)
                    except (ValueError, TypeError):
                        return str(obj)  # Last resort: convert to string
        
        # Handle numpy number types (additional check)
        # Note: This is effectively unreachable because all np.integer/np.floating
        # are also np.generic, so they're caught by the check above.
        # Kept for defensive programming and potential future numpy versions.
        if isinstance(obj, (np.integer, np.floating)):  # pragma: no cover
            return float(obj) if np.issubdtype(type(obj), np.floating) else int(obj)
        
        # Return as-is for native Python types
        return obj
    
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
                # add default 'UNKNOWN' sense label for old state files
                for mem in memories:
                    if isinstance(mem, dict) and 'sense' not in mem:
                        mem['sense'] = 'UNKNOWN'
                memory_levels[level] = memories
        return memory_levels
    
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
    
    def _deserialize_array(self, data):
        """
        Deserialize array from JSON format.
        
        Args:
            data: list or array data
            
        Returns:
            list
        """
        if isinstance(data, list):
            return [float(x) for x in data]
        return data if data else []

