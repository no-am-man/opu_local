# Changelog

All notable changes to the Orthogonal Processing Unit (OPU) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[3.1.0]: https://github.com/no-am-man/opu_local/releases/tag/v3.1.0
[3.0.0]: https://github.com/no-am-man/opu_local/releases/tag/v3.0.0

