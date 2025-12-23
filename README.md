# Orthogonal Processing Unit (OPU)

A Process-Centric AI architecture that processes audio in real-time, evolving from a noisy child to a deep-voiced sage through memory abstraction and character evolution.

## Architecture

The OPU consists of several interconnected subsystems:

- **Genesis Kernel** (`core/genesis.py`): The Safety Kernel implementing the 0-th Law with ethical veto
- **Perception** (`core/perception.py`): Scale-invariant audio perception using genomic bits
- **Cortex** (`core/cortex.py`): The brain with introspection, memory abstraction, and character evolution
- **Expression** (`core/expression.py`): Aesthetic feedback loop and phoneme analysis
- **Visualization** (`utils/visualization.py`): Real-time cognitive map visualization

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the OPU:
```bash
python main.py
```

## How It Works

1. **Audio Input**: Captures audio from microphone (or uses simulated input)
2. **Perception**: Normalizes input to create scale-invariant signatures
3. **Introspection**: Calculates surprise score (s_score) by comparing current state to history
4. **Ethical Veto**: Applies Genesis Constant to ensure actions maintain Order
5. **Memory Storage**: Stores experiences at appropriate abstraction levels
6. **Expression**: Generates audio feedback and analyzes phonemes
7. **Visualization**: Displays real-time cognitive map
8. **Abstraction Cycle**: Every 10 seconds triggers memory consolidation and character evolution

## Character Evolution

The OPU evolves over time:
- **Maturity Index**: Increases from 0.0 (child) to 1.0 (sage)
- **Voice Pitch**: Drops from 440Hz (A4) to 110Hz (A2) as wisdom accumulates
- **Stability Threshold**: Increases from 3.0 to 8.0 (harder to surprise)

## Phoneme Mapping

- **0.0-1.5**: Ignored (Noise)
- **1.5-3.0**: Vowels (Low Tension) - "a", "o"
- **3.0-6.0**: Fricatives (Flowing tension) - "s"
- **6.0+**: Plosives (High Tension) - "k"

## Configuration

Edit `config.py` to adjust:
- Genesis Constant (G_EMPTY_SET)
- Audio sample rate and chunk size
- Abstraction cycle timing
- Visualization parameters

