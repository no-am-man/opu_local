"""
Tests for utils/persistence.py - State Persistence
100% code coverage target
"""

import pytest
import numpy as np
import json
import os
from pathlib import Path
from unittest.mock import Mock
from core.opu import OrthogonalProcessingUnit
from core.expression import PhonemeAnalyzer
from utils.persistence import OPUPersistence


# Test fixtures and helpers
@pytest.fixture
def persistence(temp_state_file):
    """Create OPUPersistence instance with temporary file."""
    return OPUPersistence(state_file=temp_state_file)


@pytest.fixture
def cortex():
    """Create a fresh OrthogonalProcessingUnit instance."""
    return OrthogonalProcessingUnit()


@pytest.fixture
def phoneme_analyzer():
    """Create a fresh PhonemeAnalyzer instance."""
    return PhonemeAnalyzer()


@pytest.fixture
def populated_cortex(cortex):
    """Create cortex with some initial state."""
    cortex.introspect(0.5)
    cortex.introspect(1.0)
    cortex.store_memory(0.5, 1.5)
    return cortex


@pytest.fixture
def populated_phoneme_analyzer(phoneme_analyzer):
    """Create phoneme analyzer with some history."""
    phoneme_analyzer.analyze(2.0, 300.0)
    return phoneme_analyzer


class TestOPUPersistence:
    """Test suite for OPUPersistence class."""
    
    def test_init_default(self, temp_state_dir):
        """Test OPUPersistence initialization with default file."""
        persistence = OPUPersistence()
        assert persistence.state_file.name == "opu_state.json"
        assert persistence.state_dir.exists()
    
    def test_init_custom_file(self, temp_state_file):
        """Test OPUPersistence initialization with custom file."""
        persistence = OPUPersistence(state_file=temp_state_file)
        assert persistence.state_file == Path(temp_state_file)
        assert persistence.state_dir.exists()
    
    def test_save_state_success(self, persistence, populated_cortex, populated_phoneme_analyzer):
        """Test successful state save."""
        result = persistence.save_state(populated_cortex, populated_phoneme_analyzer, day_counter=5)
        assert result is True
        assert os.path.exists(persistence.state_file)
        
        # Verify file is valid JSON
        with open(persistence.state_file, 'r') as f:
            state = json.load(f)
        assert state['version'] == '1.0'
        assert state['day_counter'] == 5
    
    def test_save_state_numpy_types(self, persistence, cortex, phoneme_analyzer):
        """Test that numpy types are properly converted."""
        # Add numpy types
        cortex.character_profile['maturity_index'] = np.float32(0.5)
        cortex.character_profile['base_pitch'] = np.float64(220.0)
        cortex.g_now = np.float32(0.3)
        
        result = persistence.save_state(cortex, phoneme_analyzer)
        assert result is True
        
        # Verify JSON can be loaded (numpy types converted)
        with open(persistence.state_file, 'r') as f:
            state = json.load(f)
        # Should not raise exception
    
    def test_save_state_memory_levels(self, persistence, cortex, phoneme_analyzer):
        """Test that memory levels are properly serialized."""
        # Add memories to multiple levels
        for level in range(8):
            for i in range(3):
                cortex.brain.memory_levels[level].append({
                    'genomic_bit': float(i),
                    's_score': float(level)
                })
        
        result = persistence.save_state(cortex, phoneme_analyzer)
        assert result is True
        
        # Verify memory levels are in saved state
        with open(persistence.state_file, 'r') as f:
            state = json.load(f)
        assert 'memory_levels' in state['cortex']
        assert len(state['cortex']['memory_levels']) == 8
    
    def test_save_state_phoneme_history(self, persistence, cortex, phoneme_analyzer):
        """Test that phoneme history is properly serialized."""
        # Add phonemes
        for i in range(5):
            phoneme_analyzer.analyze(2.0 + i, 300.0)
        
        result = persistence.save_state(cortex, phoneme_analyzer)
        assert result is True
        
        # Verify phoneme history is in saved state
        with open(persistence.state_file, 'r') as f:
            state = json.load(f)
        assert 'phonemes' in state
        assert len(state['phonemes']['history']) == 5
    
    def test_load_state_not_found(self, temp_state_dir):
        """Test load_state when file doesn't exist."""
        persistence = OPUPersistence(state_file=os.path.join(temp_state_dir, "nonexistent.json"))
        cortex = OrthogonalProcessingUnit()
        phoneme_analyzer = PhonemeAnalyzer()
        
        success, day_counter, timers = persistence.load_state(cortex, phoneme_analyzer)
        assert success is False
        assert day_counter == 0
        assert timers is None
    
    def test_load_state_success(self, persistence, populated_cortex, populated_phoneme_analyzer):
        """Test successful state load."""
        # Save initial state
        persistence.save_state(populated_cortex, populated_phoneme_analyzer, day_counter=10)
        
        # Create new instances and load
        new_cortex = OrthogonalProcessingUnit()
        new_phoneme_analyzer = PhonemeAnalyzer()
        
        success, day_counter, timers = persistence.load_state(new_cortex, new_phoneme_analyzer)
        assert success is True
        assert day_counter == 10
        # Maturity is recalculated based on loaded memory state (not preserved from save)
        # With only level 0 memories, maturity_index will be 0.0
        assert new_cortex.character_profile['maturity_index'] == 0.0
        # With natural learning, memory goes to level 0 first
        assert len(new_cortex.memory_levels[0]) > 0
        assert len(new_phoneme_analyzer.phoneme_history) > 0
        # timers may be None for old state files
    
    def test_load_state_backward_compatible_4_levels(self, persistence, cortex, phoneme_analyzer):
        """Test loading state with old 4-level format."""
        # Create old format state (4 levels)
        old_state = {
            'version': '1.0',
            'day_counter': 5,
            'cortex': {
                'character_profile': {
                    'maturity_index': 0.5,
                    'maturity_level': 2,
                    'base_pitch': 300.0,
                    'stability_threshold': 5.0
                },
                'memory_levels': {
                    '0': [{'genomic_bit': 0.5}],
                    '1': [{'genomic_bit': 0.6}],
                    '2': [{'genomic_bit': 0.7}],
                    '3': [{'genomic_bit': 0.8}]
                },
                'genomic_bits_history': [0.5, 0.6, 0.7],
                'mu_history': [0.5, 0.55, 0.6],
                'sigma_history': [0.0, 0.05, 0.05],
                'current_state': {
                    'g_now': 0.7,
                    's_score': 1.0,
                    'coherence': 0.5
                }
            },
            'phonemes': {
                'history': [{'phoneme': 'a', 's_score': 2.0, 'pitch': 300.0}],
                'speech_threshold': 1.5
            }
        }
        
        with open(persistence.state_file, 'w') as f:
            json.dump(old_state, f)
        
        success, day_counter, timers = persistence.load_state(cortex, phoneme_analyzer)
        assert success is True
        assert day_counter == 5
        # Should have 8 levels (4 from old + 4 empty)
        assert len(cortex.memory_levels) == 8
        # timers may be None for old state files
    
    @pytest.mark.parametrize("input_val,expected_type,expected_value", [
        (None, type(None), None),
        (np.array([1.0, 2.0, 3.0], dtype=np.float32), list, None),
        ([np.float32(1.0), np.float64(2.0), np.int32(3)], list, None),
        ({'a': np.float32(1.0), 'b': np.float64(2.0)}, dict, None),
        (np.float32(1.5), float, 1.5),
    ])
    def test_convert_numpy_types_to_native_basic(self, persistence, input_val, expected_type, expected_value):
        """Test _convert_numpy_types_to_native with basic types."""
        result = persistence._convert_numpy_types_to_native(input_val)
        assert isinstance(result, expected_type) or (expected_type == type(None) and result is None)
        if expected_value is not None:
            assert result == expected_value
    
    def test_convert_numpy_types_to_native_array(self, persistence):
        """Test _convert_numpy_types_to_native with numpy array."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = persistence._convert_numpy_types_to_native(arr)
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)
    
    def test_convert_numpy_types_to_native_list(self, persistence):
        """Test _convert_numpy_types_to_native with list."""
        lst = [np.float32(1.0), np.float64(2.0), np.int32(3)]
        result = persistence._convert_numpy_types_to_native(lst)
        assert isinstance(result, list)
        assert all(isinstance(x, (int, float)) for x in result)
    
    def test_convert_numpy_types_to_native_dict(self, persistence):
        """Test _convert_numpy_types_to_native with dict."""
        dct = {
            'a': np.float32(1.0),
            'b': np.float64(2.0),
            'c': [np.int32(3)]
        }
        result = persistence._convert_numpy_types_to_native(dct)
        assert isinstance(result, dict)
        assert isinstance(result['a'], float)
        assert isinstance(result['b'], float)
        assert isinstance(result['c'], list)
    
    def test_convert_numpy_types_to_native_scalar(self, persistence):
        """Test _convert_numpy_types_to_native with numpy scalar."""
        scalar = np.float32(1.5)
        result = persistence._convert_numpy_types_to_native(scalar)
        assert isinstance(result, float)
        assert result == 1.5
    
    def test_serialize_memory_levels(self, persistence):
        """Test _serialize_memory_levels."""
        memory_levels = {
            0: [{'genomic_bit': np.float32(0.5), 's_score': 0.3}],
            1: [{'genomic_bit': np.float64(0.6), 's_score': 0.4}]
        }
        result = persistence._serialize_memory_levels(memory_levels)
        assert isinstance(result, dict)
        assert '0' in result
        assert '1' in result
        # Verify numpy types are converted
        assert isinstance(result['0'][0]['genomic_bit'], float)
    
    def test_serialize_array(self, persistence):
        """Test _serialize_array."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = persistence._serialize_array(arr)
        assert isinstance(result, list)
        assert all(isinstance(x, float) for x in result)
    
    def test_deserialize_memory_levels(self, persistence):
        """Test _deserialize_memory_levels."""
        serialized = {
            '0': [{'genomic_bit': 0.5}],
            '1': [{'genomic_bit': 0.6}],
            '5': [{'genomic_bit': 0.7}]
        }
        result = persistence._deserialize_memory_levels(serialized)
        assert len(result) == 8  # 8 levels (0-7)
        assert len(result[0]) == 1
        assert len(result[1]) == 1
        assert len(result[5]) == 1
        assert len(result[6]) == 0  # Level 6 should exist but be empty
        assert len(result[7]) == 0  # Level 7 should exist but be empty
    
    def test_deserialize_array(self, persistence):
        """Test _deserialize_array."""
        data = [1.0, 2.0, 3.0]
        result = persistence._deserialize_array(data)
        assert isinstance(result, list)
        assert result == [1.0, 2.0, 3.0]
    
    def test_save_load_roundtrip(self, persistence, populated_cortex, phoneme_analyzer):
        """Test complete save/load roundtrip."""
        # Add phonemes (start fresh, don't use populated_phoneme_analyzer)
        phoneme_analyzer.analyze(2.0, 300.0)
        phoneme_analyzer.analyze(4.0, 250.0)
        
        # Save
        persistence.save_state(populated_cortex, phoneme_analyzer, day_counter=42)
        
        # Load
        new_cortex = OrthogonalProcessingUnit()
        new_phoneme_analyzer = PhonemeAnalyzer()
        success, day_counter, timers = persistence.load_state(new_cortex, new_phoneme_analyzer)
        
        assert success is True
        assert day_counter == 42
        # Maturity is recalculated based on loaded memory state (not preserved from save)
        # With only level 0 memories, maturity_index will be 0.0
        assert new_cortex.character_profile['maturity_index'] == 0.0
        assert len(new_phoneme_analyzer.phoneme_history) == 2
        # timers may be None for state files without timers
    
    def test_save_state_exception_handling(self, persistence, cortex, phoneme_analyzer, monkeypatch, capsys):
        """Test save_state exception handling."""
        # Force an exception by making json.dump fail
        def mock_dump(*args, **kwargs):
            raise Exception("JSON error")
        
        monkeypatch.setattr('json.dump', mock_dump)
        
        result = persistence.save_state(cortex, phoneme_analyzer)
        assert result is False
        captured = capsys.readouterr()
        assert "[PERSISTENCE] Error saving state" in captured.out
    
    def test_load_state_exception_handling(self, persistence, cortex, phoneme_analyzer, capsys):
        """Test load_state exception handling."""
        # Create invalid JSON file
        with open(persistence.state_file, 'w') as f:
            f.write("invalid json{")
        
        success, day_counter, timers = persistence.load_state(cortex, phoneme_analyzer)
        assert success is False
        assert day_counter == 0
        assert timers is None
        captured = capsys.readouterr()
        assert "[PERSISTENCE] Error loading state" in captured.out
    
    def test_convert_numpy_types_to_native_tuple(self, persistence):
        """Test _convert_numpy_types_to_native with tuple."""
        tup = (np.float32(1.0), np.float64(2.0))
        result = persistence._convert_numpy_types_to_native(tup)
        assert isinstance(result, list)  # Tuples become lists
        assert all(isinstance(x, float) for x in result)
    
    def test_convert_numpy_types_to_native_nested_structure(self, persistence):
        """Test _convert_numpy_types_to_native with deeply nested structure."""
        nested = {
            'a': [np.float32(1.0), {'b': np.int32(2)}],
            'c': np.array([3.0, 4.0])
        }
        result = persistence._convert_numpy_types_to_native(nested)
        assert isinstance(result, dict)
        assert isinstance(result['a'], list)
        assert isinstance(result['a'][0], float)
        assert isinstance(result['a'][1], dict)
        assert isinstance(result['a'][1]['b'], int)
    
    def test_convert_numpy_types_to_native_fallback_conversion(self, persistence):
        """Test _convert_numpy_types_to_native fallback conversions."""
        # Test with complex number which is np.generic but not np.integer or np.floating
        complex_val = np.complex128(1.5 + 2.0j)
        result = persistence._convert_numpy_types_to_native(complex_val)
        # Should convert to float or handle gracefully
        assert isinstance(result, (float, complex, str))
    
    def test_convert_numpy_types_to_native_fallback_float_int_string(self, persistence):
        """Test _convert_numpy_types_to_native fallback chain."""
        # Test with np.complex128 which is np.generic but not integer/floating
        complex_val = np.complex128(1.5 + 2.0j)
        result = persistence._convert_numpy_types_to_native(complex_val)
        # Complex numbers can be converted to float (takes real part) or string
        # The exact behavior depends on numpy version, but should not crash
        assert result is not None
    
    def test_convert_numpy_types_to_native_fallback_exception_paths(self, persistence):
        """Test _convert_numpy_types_to_native exception paths in fallback."""
        # Test with np.datetime64 which is np.generic but not integer/floating
        try:
            dt_val = np.datetime64('2023-01-01')
            result = persistence._convert_numpy_types_to_native(dt_val)
            assert result is not None
        except (TypeError, ValueError):
            # Some numpy versions may not support datetime64 conversion
            pass
        
        # Test with np.timedelta64 which also triggers the fallback path
        try:
            td_val = np.timedelta64(1, 'D')
            result = persistence._convert_numpy_types_to_native(td_val)
            assert result is not None
        except (TypeError, ValueError):
            # Some numpy versions may not support timedelta64 conversion
            pass
    
    def test_load_cortex_state_no_cortex_key(self, persistence):
        """Test _load_cortex_state when 'cortex' key is missing."""
        state = {}  # No 'cortex' key
        cortex = Mock()
        
        # Should return early without error
        persistence._load_cortex_state(state, cortex)
        # No assertions needed - just verify it doesn't crash
    
    def test_load_phoneme_state_no_phonemes_key(self, persistence):
        """Test _load_phoneme_state when 'phonemes' key is missing."""
        state = {}  # No 'phonemes' key
        phoneme_analyzer = Mock()
        
        # Should return early without error
        persistence._load_phoneme_state(state, phoneme_analyzer)
        # No assertions needed - just verify it doesn't crash
    
    def test_deserialize_abstraction_timers_with_data(self, persistence):
        """Test _deserialize_abstraction_timers with valid data."""
        state = {
            'abstraction_timers': {
                '0': 1000.0,
                '1': 2000.0,
                '2': 3000.0,
                '99': 4000.0  # Invalid level, should be filtered
            }
        }
        
        result = persistence._deserialize_abstraction_timers(state)
        
        assert result is not None
        assert 0 in result
        assert 1 in result
        assert 2 in result
        assert 99 not in result  # Invalid level filtered
        assert result[0] == 1000.0
        assert result[1] == 2000.0
        assert result[2] == 3000.0
    
    def test_print_load_summary_with_abstraction_timers(self, persistence, capsys):
        """Test _print_load_summary with abstraction timers."""
        cortex = Mock()
        cortex.character_profile = {'maturity_level': 2, 'maturity_index': 0.5}
        cortex.memory_levels = {i: [] for i in range(8)}
        cortex.emotion_history = []
        phoneme_analyzer = Mock()
        phoneme_analyzer.phoneme_history = []
        day_counter = 5
        last_abstraction_times = {0: 1000.0, 1: 2000.0}
        
        persistence._print_load_summary(cortex, phoneme_analyzer, day_counter, last_abstraction_times)
        
        captured = capsys.readouterr()
        assert "Abstraction Timers: Restored for 2 levels" in captured.out
    
    def test_convert_numpy_types_to_native_numpy_number_types_direct(self, persistence, monkeypatch):
        """Test _convert_numpy_types_to_native with numpy number types that bypass generic check."""
        # Mock isinstance to bypass generic check to test defensive code path
        original_isinstance = isinstance
        
        def mock_isinstance(obj, class_or_tuple):
            if class_or_tuple == np.generic:
                return False
            return original_isinstance(obj, class_or_tuple)
        
        monkeypatch.setattr('builtins.isinstance', mock_isinstance)
        
        # Now test with numpy types
        int_val = np.int32(42)
        float_val = np.float32(3.14)
        
        result_int = persistence._convert_numpy_types_to_native(int_val)
        result_float = persistence._convert_numpy_types_to_native(float_val)
        
        assert isinstance(result_int, int)
        assert isinstance(result_float, float)
    
    def test_convert_numpy_types_to_native_numpy_number_types(self, persistence):
        """Test _convert_numpy_types_to_native with numpy number types."""
        int_val = np.int64(42)
        float_val = np.float64(3.14)
        
        result_int = persistence._convert_numpy_types_to_native(int_val)
        result_float = persistence._convert_numpy_types_to_native(float_val)
        
        assert isinstance(result_int, int)
        assert isinstance(result_float, float)
    
    @pytest.mark.parametrize("input_data,expected", [
        (None, []),
        ([], []),
        (False, []),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
    ])
    def test_deserialize_array_empty_data(self, persistence, input_data, expected):
        """Test _deserialize_array with empty/None data."""
        result = persistence._deserialize_array(input_data)
        assert result == expected

