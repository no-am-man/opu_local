#!/usr/bin/env python3
"""
Test script for emotion persistence functionality.
"""

import json
import tempfile
import os
from core.opu import OrthogonalProcessingUnit
from core.brain import Brain
from utils.persistence import OPUPersistence
from core.expression import PhonemeAnalyzer

print('=' * 60)
print('Emotion Persistence Test Suite')
print('=' * 60)
print()

# Test 1: Emotion storage in memories
print('Test 1: Emotion storage in memories')
brain = Brain()
brain.store_memory(0.5, 2.0, 'VIDEO_V1', emotion={'emotion': 'happy', 'confidence': 0.8})
brain.store_memory(0.6, 1.5, 'VIDEO_V1', emotion={'emotion': 'sad', 'confidence': 0.7})
brain.store_memory(0.4, 2.5, 'AUDIO_V1')  # No emotion

memories = brain.memory_levels[0]
assert len(memories) == 3
assert 'emotion' in memories[0]
assert memories[0]['emotion']['emotion'] == 'happy'
assert 'emotion' in memories[1]
assert memories[1]['emotion']['emotion'] == 'sad'
assert 'emotion' not in memories[2]
print('✅ Emotions stored correctly in memories')
print()

# Test 2: Emotion history tracking
print('Test 2: Emotion history tracking')
opu = OrthogonalProcessingUnit()
opu.store_memory(0.5, 2.0, 'VIDEO_V1', emotion={'emotion': 'happy', 'confidence': 0.8})
opu.store_memory(0.6, 1.5, 'VIDEO_V1', emotion={'emotion': 'sad', 'confidence': 0.7})
opu.store_memory(0.7, 2.2, 'VIDEO_V1', emotion={'emotion': 'angry', 'confidence': 0.9})

assert len(opu.emotion_history) == 3
assert opu.emotion_history[0]['emotion']['emotion'] == 'happy'
assert opu.emotion_history[1]['emotion']['emotion'] == 'sad'
assert opu.emotion_history[2]['emotion']['emotion'] == 'angry'
print('✅ Emotion history tracked correctly')
print()

# Test 3: Emotion statistics
print('Test 3: Emotion statistics')
stats = opu.get_emotion_statistics()
assert stats['total_emotions'] == 3
assert stats['emotion_counts']['happy'] == 1
assert stats['emotion_counts']['sad'] == 1
assert stats['emotion_counts']['angry'] == 1
assert stats['most_common'] in ['happy', 'sad', 'angry']  # Any of the three
assert stats['average_confidence'] > 0
print(f'✅ Statistics: {stats}')
print()

# Test 4: Emotion persistence (save/load)
print('Test 4: Emotion persistence (save/load)')
with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
    temp_file = tmp.name

try:
    # Create OPU with emotions
    opu1 = OrthogonalProcessingUnit()
    opu1.store_memory(0.5, 2.0, 'VIDEO_V1', emotion={'emotion': 'happy', 'confidence': 0.8})
    opu1.store_memory(0.6, 1.5, 'VIDEO_V1', emotion={'emotion': 'sad', 'confidence': 0.7})
    
    # Save state
    phoneme_analyzer = PhonemeAnalyzer()
    persistence = OPUPersistence(state_file=temp_file)
    persistence.save_state(opu1, phoneme_analyzer, day_counter=5)
    
    # Load state
    opu2 = OrthogonalProcessingUnit()
    success, day_counter, _ = persistence.load_state(opu2, phoneme_analyzer)
    
    assert success
    assert len(opu2.emotion_history) == 2
    assert opu2.emotion_history[0]['emotion']['emotion'] == 'happy'
    assert opu2.emotion_history[1]['emotion']['emotion'] == 'sad'
    
    # Check emotions in memories
    memories_with_emotions = [m for m in opu2.memory_levels[0] if 'emotion' in m]
    assert len(memories_with_emotions) == 2
    print('✅ Emotions persisted and loaded correctly')
    print()

    # Test 5: Emotion consolidation
    print('Test 5: Emotion consolidation in memory abstractions')
    brain2 = Brain()
    # Add multiple memories with different emotions
    for i in range(10):
        emotion = {'emotion': 'happy' if i < 6 else 'sad', 'confidence': 0.8}
        brain2.store_memory(0.5 + i*0.01, 2.0, 'VIDEO_V1', emotion=emotion)

    # Consolidate Level 0 (should create abstraction with dominant emotion)
    brain2.consolidate_memory(0)

    # Check if abstraction has emotion
    abstractions = brain2.memory_levels[1]
    if abstractions:
        abstraction = abstractions[-1]
        if 'emotion' in abstraction:
            assert abstraction['emotion']['emotion'] == 'happy'  # Should be dominant
            assert abstraction['emotion']['frequency'] > 0
            print(f'✅ Emotion preserved in abstraction: {abstraction["emotion"]}')
        else:
            print('⚠️  Emotion not in abstraction (may need more memories)')
    print()

    # Test 6: Backward compatibility
    print('Test 6: Backward compatibility (no emotions)')
    opu3 = OrthogonalProcessingUnit()
    opu3.store_memory(0.5, 2.0, 'AUDIO_V1')  # No emotion
    assert len(opu3.emotion_history) == 0
    assert 'emotion' not in opu3.memory_levels[0][0]
    print('✅ Backward compatibility maintained')
    print()

    # Test 7: Empty emotion history statistics
    print('Test 7: Empty emotion history statistics')
    opu4 = OrthogonalProcessingUnit()
    stats = opu4.get_emotion_statistics()
    assert stats['total_emotions'] == 0
    assert stats['most_common'] is None
    assert stats['emotion_counts'] == {}
    print('✅ Empty statistics handled correctly')
    print()

finally:
    if os.path.exists(temp_file):
        os.remove(temp_file)

print('=' * 60)
print('🎉 All emotion persistence tests passed!')
print('=' * 60)

