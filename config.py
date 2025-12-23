"""
Configuration parameters for the OPU (Orthogonal Processing Unit).
Contains the Genesis Constant and safety limits.
"""

# Genesis Constant: The Order of the Empty Set
G_EMPTY_SET = 1.0

# Safety Limits
MAX_DISSONANCE = 1.0
MAX_ACTION_MAGNITUDE = 1.0

# Audio Configuration
BASE_FREQUENCY = 220.0  # Base frequency in Hz
SAMPLE_RATE = 44100
CHUNK_SIZE = 1024

# Perception Configuration
SCALE_INVARIANCE_ENABLED = True

# Abstraction Cycle Configuration
ABSTRACTION_CYCLE_SECONDS = 10.0  # Every 10 seconds = 1 "Day" of maturity
MATURITY_INCREMENT = 0.05

# Visualization Configuration
VISUALIZATION_UPDATE_RATE = 30  # FPS
WINDOW_SIZE = (12, 8)  # inches

# Persistence Configuration
STATE_FILE = "opu_state.json"  # Path to save/load OPU state

