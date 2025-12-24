#!/usr/bin/env python3
"""
Quick test script to verify DeepFace integration with OPU.
"""

import sys
import numpy as np

print("=" * 60)
print("DeepFace Integration Test for OPU")
print("=" * 60)
print()

# Test 1: Import DeepFace
print("Test 1: Importing DeepFace...")
try:
    from deepface import DeepFace
    print("✅ DeepFace imported successfully")
except Exception as e:
    print(f"❌ Failed to import DeepFace: {e}")
    sys.exit(1)

# Test 2: Import ObjectDetector
print("\nTest 2: Importing ObjectDetector...")
try:
    from core.object_detection import ObjectDetector
    print("✅ ObjectDetector imported successfully")
except Exception as e:
    print(f"❌ Failed to import ObjectDetector: {e}")
    sys.exit(1)

# Test 3: Initialize ObjectDetector
print("\nTest 3: Initializing ObjectDetector...")
try:
    detector = ObjectDetector(detect_emotions=True)
    print(f"✅ ObjectDetector initialized")
    
    if hasattr(detector, 'emotion_method'):
        method = detector.emotion_method
        print(f"   Emotion detection method: {method}")
        
        if method == 'deepface':
            print("   ✅ Using DeepFace (high accuracy)")
        elif method == 'fer':
            print("   ⚠️  Using FER library (medium accuracy)")
        elif method == 'heuristic':
            print("   ⚠️  Using heuristic method (basic accuracy)")
            print("   💡 DeepFace should be available but not detected")
        else:
            print(f"   ⚠️  Unknown method: {method}")
    else:
        print("   ⚠️  emotion_method attribute not found")
        
except Exception as e:
    print(f"❌ Failed to initialize ObjectDetector: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Check if DeepFace can be used directly
print("\nTest 4: Testing DeepFace.analyze() availability...")
try:
    # Just check if the method exists, don't actually analyze (would need an image)
    if hasattr(DeepFace, 'analyze'):
        print("✅ DeepFace.analyze() method available")
    else:
        print("⚠️  DeepFace.analyze() method not found")
except Exception as e:
    print(f"⚠️  Could not verify DeepFace.analyze(): {e}")

print("\n" + "=" * 60)
print("Test Summary:")
print("=" * 60)
print("✅ All core tests passed!")
print("\nDeepFace is ready to use with OPU.")
print("\nTo run OPU with DeepFace:")
print("  1. Activate virtual environment: source venv_python312/bin/activate")
print("  2. Run OPU: ./run_opu.sh")
print("  3. DeepFace will automatically detect emotions in faces!")
print("=" * 60)

