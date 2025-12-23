"""
Configuration parameters for the OPU (Orthogonal Processing Unit).
Contains the Genesis Constant and safety limits.

OPU v3.0.0 - MIT License
"""

# Version
OPU_VERSION = "3.0.0"

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
# 6 Maturity Levels with time-scaled abstraction cycles:
# Level 0: 1 minute (60 seconds) - Immediate/short-term memory
# Level 1: 1 hour (3600 seconds) - Short-term patterns
# Level 2: 1 day (86400 seconds) - Daily patterns
# Level 3: 1 week (604800 seconds) - Weekly patterns
# Level 4: 1 month (2592000 seconds) - Monthly patterns
# Level 5: 1 year (31536000 seconds) - Yearly patterns/wisdom

# Time scale multipliers (for simulation - can be adjusted for faster/slower progression)
TIME_SCALE_MULTIPLIER = 1.0  # 1.0 = real time, 10.0 = 10x faster, etc.

MATURITY_LEVEL_TIMES = {
    0: 60.0,           # 1 minute
    1: 3600.0,         # 1 hour
    2: 86400.0,        # 1 day
    3: 604800.0,       # 1 week
    4: 2592000.0,      # 1 month
    5: 31536000.0      # 1 year
}

# Apply time scale multiplier
MATURITY_LEVEL_TIMES = {k: v / TIME_SCALE_MULTIPLIER for k, v in MATURITY_LEVEL_TIMES.items()}

# Legacy support (for backward compatibility)
ABSTRACTION_CYCLE_SECONDS = MATURITY_LEVEL_TIMES[2]  # 1 day
MATURITY_INCREMENT = 0.0167  # Smaller increment for 6 levels (1.0 / 60 steps to reach full maturity)

# Visualization Configuration
VISUALIZATION_UPDATE_RATE = 30  # FPS
WINDOW_SIZE = (12, 8)  # inches

# Persistence Configuration
STATE_FILE = "opu_state.json"  # Path to save/load OPU state

# Sense Labels (for extensible input system)
# Each input source is labeled with a sense identifier
# This allows the OPU to track and learn from different input modalities
# Future senses can be added: TOUCH_V1, TEMPERATURE_V1, etc.
AUDIO_SENSE = "AUDIO_V1"
VIDEO_SENSE = "VIDEO_V1"

