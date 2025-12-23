# Changelog

All notable changes to the Orthogonal Processing Unit (OPU) project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[3.0.0]: https://github.com/yourusername/opu/releases/tag/v3.0.0

