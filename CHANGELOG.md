# Changelog

All notable changes to the Orthogonal Processing Unit (OPU) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.4.0] - 2024-12-24

### Added
- **Enhanced Emotion Persistence**: Improved emotional memory consolidation and statistics
  - Added `get_emotion_statistics()` method for analyzing emotional history
  - Emotions now properly preserved through all memory abstraction levels
  - Enhanced emotion tracking with confidence scores and timestamps
- **Monograph Update**: Updated OPU Monograph to v3.4 with formal mathematical treatment of emotional memory persistence ($\Psi_{emotion}$)

### Changed
- **Emotion Memory**: Emotions are now stored alongside sensory memories and persist across learning phases
- **Emotional Consolidation**: Dominant emotions are preserved in memory abstractions, allowing the OPU to build emotional associations over time

### Technical Details
- Emotional memory vector: $M(t) = \langle g_{bit}, S_{score}, E, \tau \rangle$
- Emotional consolidation: $E_{abstraction} = \arg\max_{e \in E_{set}} \text{Frequency}(e) \cdot \text{Confidence}(e)$
- Emotion history tracked with timestamps and confidence scores
- Backward compatible with v3.3.0 state files

---

## [3.3.0] - 2024-12-24

### Added
- **Color Constancy (Shadow Invariance)**: Visual perception now uses normalized chromaticity (R+G+B normalization)
  - OPU ignores lighting changes (shadows, flickers) and only responds to actual color/structure changes
  - Implements biological color constancy (lateral inhibition)
  - Configurable via `USE_COLOR_CONSTANCY` in `config.py` (default: True)
- **Emotion Persistence**: Detected emotions are now stored in memories and persist across sessions
  - Emotions preserved through memory consolidation
  - Emotion history tracked and saved to disk
  - OPU builds emotional memory over time

### Technical Details
- Chromaticity vector: $\vec{C} = \left\langle \frac{R}{\Sigma}, \frac{G}{\Sigma}, \frac{B}{\Sigma} \right\rangle$ where $\Sigma = R + G + B + \epsilon$
- Luminance isolation for energetic events vs. structural changes
- Backward compatible with v3.2.0 state files

---

## [3.2.0] - 2024-12-24

### Added
- **Natural Learning Process**: Enforced "Cognitive Sedimentation" - all memories must start at Level 0 and progress through time-based consolidation
- **EPOCH Timestamps**: Switched from logical timestamps to EPOCH time (`time.time()`) for perfect temporal synchronization across all senses
- **File Logging**: Command-line option `--log-file` to save all OPU output for debugging
- **Log Window Copy Features**: Copy All, Copy Selected, and Save to File buttons in the log window

### Changed
- **Memory Hierarchy**: All raw sensory data now enters at Level 0, regardless of surprise score (prevents "Trauma Evolution")
- **Noise Floor**: Epsilon increased from 0.0001 to 0.01 to prevent false high s_score values from silence
- **Audio Input**: Returns dithering noise instead of pure zeros when buffer is empty (prevents divide-by-zero)
- **State Delegation**: Removed shadow copies in `opu.py`, now uses property-based delegation for proper state synchronization

### Fixed
- **Trauma Evolution Bug**: Fixed instant maturity jump to Level 6 caused by high s_score values bypassing the hierarchy
- **Audio Dead Zone**: Fixed audio flatlining to 0.0 by returning dithering noise instead of zeros
- **Temporal Sync Error**: Fixed desynchronization between audio and video by using EPOCH timestamps
- **State Synchronization**: Fixed s_score always showing 0.00 by removing stale shadow copies
- **Divide-by-Zero**: Enhanced epsilon floor to prevent s_score explosion in silent environments
- **Persistence**: Fixed property setter issues when loading state files

### Breaking Changes
- **State File Reset Required**: Existing `opu_state.json` files contain false memories from trauma evolution and must be deleted
- **Memory Storage**: Memories no longer jump to higher levels based on s_score - all start at Level 0

### Technical Details
- Natural learning enforces: 100 Level 0 → 1 Level 1 → 50 Level 1 → 1 Level 2, etc.
- EPOCH timestamps ensure audio (sample rate) and video (FPS) align on universal timeline
- Dithering noise (σ=0.0001) prevents audio dead zone while maintaining signal integrity
- Cascade consolidation: when Level N consolidates, it checks if Level N+1 should also consolidate

---

## [3.1.0] - 2024-12-23

### Added
- **Level 7 "Scire"**: New maturity level with 10-year cycle (transcendent knowledge state)
- **Enhanced Persistence**: Abstraction cycle timers now saved/restored for timing continuity
- **Emotion Detection**: Real-time facial emotion recognition (7 emotions: angry, disgust, fear, happy, neutral, sad, surprise)
- **Visual Cortex**: Multi-modal integration with recursive perceptual loops
- **Acoustic Feedback Prevention**: Automatic microphone muting when OPU is speaking
- **Design Patterns**: 8 Gang of Four (GOF) patterns implemented (Strategy, Observer, Factory, State, Template Method, Command, Decorator, Builder)
- **Sense Labeling**: Extensible input system (AUDIO_V1, VIDEO_V1) for future sensor integration
- **Log Window**: Dedicated GUI window for real-time OPU logs (macOS compatible)

### Changed
- **Maturity Levels**: Extended from 6 to 7 levels (1 minute to 10 years)
- **Sensitivity Tuning**: Lower thresholds for more responsive, "chatty" behavior
  - Speech threshold: 1.5 → 0.6
  - Noise gate: 0.2 → 0.1
  - Memory history: 10000 → 50 (more reactive)
- **Persistence Format**: Added abstraction timers to state file
- **Memory Hierarchy**: Now supports 7 abstraction layers (L0-L6)

### Fixed
- Audio buffer overflow handling (aggressive draining, low latency)
- macOS tkinter crash (requires `run_opu.sh` launcher script)
- Matplotlib threading issues (graceful shutdown on Ctrl+C)
- OpenCV threading issues (exception handling for interrupts)
- Test coverage updated for 7-level system

### Technical Details
- Backward compatible with v3.0.0 state files
- All 7 maturity levels fully persistent
- Abstraction cycle timing preserved across restarts

---

## [3.0.0] - 2024

### Added
- Initial open source release
- Complete OPU architecture implementation
- Real-time audio processing with microphone input
- Scale-invariant perception system
- Introspection and surprise score calculation
- Memory abstraction across 4 levels (L0-L3)
- Character evolution system (maturity, voice pitch, stability)
- Phoneme learning and analysis
- Aesthetic feedback loop (audio output)
- Cognitive map visualization
- State persistence (save/load learned state)
- Genesis Kernel with ethical veto
- Abstraction cycles (simulated "Days")
- Comprehensive documentation

### Technical Details
- Python 3.8+ support
- NumPy for numerical processing
- Matplotlib for visualization
- sounddevice for audio I/O
- JSON-based state persistence
- Modular architecture (core, utils)

### Known Issues
- Audio buffer overflow warnings (mitigated with aggressive reading)
- Phoneme detection requires s_score >= 1.5

---

[3.4.0]: https://github.com/no-am-man/opu_local/releases/tag/v3.4.0
[3.3.0]: https://github.com/no-am-man/opu_local/releases/tag/v3.3.0
[3.2.0]: https://github.com/no-am-man/opu_local/releases/tag/v3.2.0
[3.1.0]: https://github.com/no-am-man/opu_local/releases/tag/v3.1.0
[3.0.0]: https://github.com/no-am-man/opu_local/releases/tag/v3.0.0

