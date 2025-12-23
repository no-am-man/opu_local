# OPU Architecture Overview

## Module Organization

The OPU is organized into clear **Perception** and **Cortex** layers:

### Perception Layer (Input Capture)
- **`core/mic.py`** - Microphone capture (`perceive()` function)
  - Captures audio input
  - Extracts genomic bit (standard deviation)
  
- **`core/camera.py`** - Camera capture (`VisualPerception` class)
  - Captures camera frames
  - Extracts R, G, B genomic vectors (standard deviation per channel)

### Cortex Layer (Introspection & Processing)
- **`core/audio_cortex.py`** - Audio introspection (`AudioCortex` class)
  - Calculates audio surprise (`s_score`)
  - Tracks audio history
  
- **`core/vision_cortex.py`** - Visual introspection (`VisualCortex` class)
  - Calculates visual surprise (`s_visual`)
  - Tracks R, G, B channel history independently

- **`core/brain.py`** - Core cognitive processing (`Brain` class)
  - Memory abstraction (6 levels)
  - Character evolution
  - Memory consolidation

### Main OPU Layer
- **`core/opu.py`** - Main OPU facade (`OrthogonalProcessingUnit` class)
  - Combines Brain + AudioCortex + VisualCortex
  - Provides unified API
  - Main entry point for the OPU

## Data Flow

```
Audio Input → mic.py → AudioCortex → Brain → Expression
                                              ↓
Visual Input → camera.py → VisualCortex ─────┘
```

## Naming Convention

- **`mic.py`** = Microphone capture (audio input)
- **`camera.py`** = Camera capture (visual input)
- **`*_cortex.py`** = Introspection (surprise calculation)
- **`brain.py`** = Memory & character (cognitive core)
- **`opu.py`** = Main OPU facade (unified API)

## Why `opu.py`?

`opu.py` provides:
1. **Convenience** - Single import for the full OPU
2. **Unified API** - One class that combines all subsystems
3. **Clear naming** - The file name matches what it contains (OPU)

## Usage

### Recommended (Main Entry Point)
```python
from core.opu import OrthogonalProcessingUnit

opu = OrthogonalProcessingUnit()
s_score = opu.introspect(genomic_bit)
s_visual, channels = opu.introspect_visual(visual_vector)
opu.store_memory(genomic_bit, s_score)
```

### Direct Access (Advanced)
```python
from core.brain import Brain
from core.audio_cortex import AudioCortex
from core.vision_cortex import VisualCortex
from core.camera import VisualPerception
from core.mic import perceive

# Use subsystems directly
brain = Brain()
audio = AudioCortex()
vision = VisualCortex()
camera = VisualPerception()
audio_input = perceive(mic_data)
```

