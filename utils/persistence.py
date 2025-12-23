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
    
    def save_state(self, cortex, phoneme_analyzer, day_counter=0):
        """
        Save OPU state to disk.
        
        Args:
            cortex: OrthogonalProcessingUnit instance
            phoneme_analyzer: PhonemeAnalyzer instance
            day_counter: Current day counter
        """
        try:
            # Serialize character profile (handle numpy types)
            character_profile = {}
            for key, value in cortex.character_profile.items():
                if isinstance(value, (np.integer, np.floating, np.float32, np.float64, 
                                     np.int32, np.int64, np.int_, np.float_)):
                    character_profile[key] = float(value)
                else:
                    character_profile[key] = value
            
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
                    'current_state': {
                        'g_now': float(cortex.g_now) if cortex.g_now is not None and not isinstance(cortex.g_now, (str, type(None))) else None,
                        's_score': float(cortex.s_score) if not isinstance(cortex.s_score, (str, type(None))) else 0.0,
                        'coherence': float(cortex.coherence) if not isinstance(cortex.coherence, (str, type(None))) else 0.0
                    }
                },
                
                # Phoneme analyzer state
                'phonemes': {
                    'history': phoneme_analyzer.phoneme_history.copy(),
                    'speech_threshold': float(phoneme_analyzer.speech_threshold) if isinstance(phoneme_analyzer.speech_threshold, (np.integer, np.floating, np.float32, np.float64)) else phoneme_analyzer.speech_threshold
                }
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
            tuple: (success: bool, day_counter: int)
        """
        if not self.state_file.exists():
            print(f"[PERSISTENCE] No saved state found at {self.state_file}")
            return False, 0
        
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
                if 'memory_levels' in cortex_data:
                    cortex.memory_levels = self._deserialize_memory_levels(cortex_data['memory_levels'])
                
                # Restore history
                if 'genomic_bits_history' in cortex_data:
                    cortex.genomic_bits_history = self._deserialize_array(cortex_data['genomic_bits_history'])
                
                if 'mu_history' in cortex_data:
                    cortex.mu_history = self._deserialize_array(cortex_data['mu_history'])
                
                if 'sigma_history' in cortex_data:
                    cortex.sigma_history = self._deserialize_array(cortex_data['sigma_history'])
                
                # Restore current state
                if 'current_state' in cortex_data:
                    cs = cortex_data['current_state']
                    cortex.g_now = cs.get('g_now')
                    cortex.s_score = cs.get('s_score', 0.0)
                    cortex.coherence = cs.get('coherence', 0.0)
            
            # Load phoneme analyzer state
            if 'phonemes' in state:
                phoneme_data = state['phonemes']
                if 'history' in phoneme_data:
                    phoneme_analyzer.phoneme_history = phoneme_data['history']
                if 'speech_threshold' in phoneme_data:
                    phoneme_analyzer.speech_threshold = phoneme_data['speech_threshold']
            
            day_counter = state.get('day_counter', 0)
            
            print(f"[PERSISTENCE] State loaded from {self.state_file}")
            print(f"  Maturity Level: {cortex.character_profile.get('maturity_level', 0)} | Index: {cortex.character_profile['maturity_index']:.2f}")
            print(f"  Memory: " + " | ".join([f"L{i}={len(cortex.memory_levels.get(i, []))}" for i in range(6)]))
            print(f"  Phonemes: {len(phoneme_analyzer.phoneme_history)}")
            print(f"  Day: {day_counter}")
            
            return True, day_counter
            
        except Exception as e:
            print(f"[PERSISTENCE] Error loading state: {e}")
            return False, 0
    
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
                    # Handle all numpy types
                    if isinstance(value, (np.integer, np.floating, np.float32, np.float64, 
                                         np.int32, np.int64, np.int_, np.float_)):
                        serialized_mem[key] = float(value)
                    elif isinstance(value, np.ndarray):
                        serialized_mem[key] = [float(x) for x in value.tolist()]
                    elif isinstance(value, (list, tuple)):
                        # Recursively handle lists/tuples that might contain numpy types
                        serialized_mem[key] = [float(x) if isinstance(x, (np.integer, np.floating, 
                                                                          np.float32, np.float64,
                                                                          np.int32, np.int64)) 
                                              else x for x in value]
                    else:
                        serialized_mem[key] = value
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
        # Support both old format (4 levels) and new format (6 levels)
        memory_levels = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        for level_str, memories in serialized.items():
            level = int(level_str)
            if level in memory_levels:
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
        if isinstance(arr, np.ndarray):
            return [float(x) for x in arr.tolist()]
        elif isinstance(arr, list):
            # Convert any numpy types in list (handle all numpy scalar types)
            result = []
            for x in arr:
                if isinstance(x, (np.integer, np.floating, np.float32, np.float64, 
                                 np.int32, np.int64, np.int_, np.float_)):
                    result.append(float(x))
                elif isinstance(x, np.ndarray):
                    result.append([float(y) for y in x.tolist()])
                else:
                    result.append(x)
            return result
        return arr
    
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

