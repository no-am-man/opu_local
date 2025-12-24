"""
Configuration parameters for the OPU (Orthogonal Processing Unit).
Contains the Genesis Constant and safety limits.

OPU v3.4.0 - MIT License
"""

# Version
OPU_VERSION = "3.4.0"  # v3.4: Enhanced Emotion Persistence + Emotional Memory Statistics

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

# Visual Perception Configuration
USE_COLOR_CONSTANCY = True  # If True, uses normalized chromaticity (shadow-invariant)
                            # If False, uses raw RGB channels (legacy mode)
                            # Color constancy makes OPU ignore lighting changes (shadows, flickers)
                            # and only respond to actual color/structure changes

# Abstraction Cycle Configuration
# 8 Maturity Levels with time-scaled abstraction cycles (Fractal Time Architecture):
# Level 0: 1 second (1s) - Immediate Sensation (The "Now")
# Level 1: 1 minute (60s) - Working Memory (Short-term buffer)
# Level 2: 1 hour (3600s) - Episode / Situation
# Level 3: 1 day (86400s) - Circadian / Sleep Consolidation
# Level 4: 1 week (604800s) - Trend
# Level 5: 1 month (2592000s) - Season
# Level 6: 1 year (31536000s) - Epoch
# Level 7: 10 years (315360000s) - Core Identity / Deep Wisdom

# Time scale multipliers (for simulation - can be adjusted for faster/slower progression)
TIME_SCALE_MULTIPLIER = 1.0  # 1.0 = real time, 10.0 = 10x faster, etc.

MATURITY_LEVEL_TIMES = {
    0: 1.0,            # 1 second
    1: 60.0,           # 1 minute
    2: 3600.0,         # 1 hour
    3: 86400.0,        # 1 day
    4: 604800.0,       # 1 week
    5: 2592000.0,      # 1 month
    6: 31536000.0,     # 1 year
    7: 315360000.0     # 10 years (Deep Wisdom)
}

# Apply time scale multiplier
MATURITY_LEVEL_TIMES = {k: v / TIME_SCALE_MULTIPLIER for k, v in MATURITY_LEVEL_TIMES.items()}

# Legacy support (for backward compatibility)
ABSTRACTION_CYCLE_SECONDS = MATURITY_LEVEL_TIMES[3]  # 1 day
MATURITY_INCREMENT = 0.0125  # Smaller increment for 8 levels (1.0 / 80 steps to reach full maturity)

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

# Processing Thresholds
VISUAL_SURPRISE_THRESHOLD = 0.5  # Minimum visual surprise to store memory
AUDIO_TONE_DURATION_SECONDS = 0.05  # Duration of audio feedback tone

# Audio Input Configuration
DITHERING_NOISE_SIGMA = 0.0001  # Standard deviation for dithering noise (prevents divide-by-zero)
BUFFER_DRAIN_MULTIPLIER = 8  # Multiplier for aggressive buffer draining (CHUNK_SIZE * 8)
BUFFER_READ_MULTIPLIER = 2  # Multiplier for normal buffer reading (CHUNK_SIZE * 2)
OVERFLOW_WARNING_INTERVAL_SECONDS = 5.0  # Minimum seconds between overflow warnings
BUFFER_DRAIN_WARNING_INTERVAL_SECONDS = 2.0  # Minimum seconds between drain warnings
BUFFER_FULL_THRESHOLD_MULTIPLIER = 8  # Buffer considered full when available > CHUNK_SIZE * 8

# Visual Display Configuration
PREVIEW_SCALE = 0.75  # Preview window scale (75% of original size)
MAX_BAR_SCORE = 5.0  # Maximum score for bar chart scaling
BAR_SCALE_FACTOR = 50  # Pixels per score unit for bar charts
BAR_SPACING = 30  # Vertical spacing between bars (pixels)
BAR_LABEL_X = 10  # X position for bar labels
BAR_START_X = 100  # X position where bars start
BAR_HEIGHT = 20  # Height of each bar (pixels)
ALERT_THRESHOLD = 3.0  # Surprise score threshold for RED alert status
INTEREST_THRESHOLD = 1.5  # Surprise score threshold for YELLOW interest status
FONT_SCALE_SMALL = 0.6  # OpenCV font scale for small text
FONT_SCALE_MEDIUM = 0.7  # OpenCV font scale for medium text
FONT_THICKNESS_THIN = 1  # OpenCV font thickness for thin text
FONT_THICKNESS_THICK = 2  # OpenCV font thickness for thick text
TEXT_COLOR_GRAY = (200, 200, 200)  # Gray color for text overlay
STATUS_COLOR_RED = (0, 0, 255)  # BGR color for RED alert
STATUS_COLOR_YELLOW = (0, 255, 255)  # BGR color for YELLOW interest
STATUS_COLOR_GREEN = (0, 255, 0)  # BGR color for GREEN calm
CHANNEL_COLOR_RED = (0, 0, 255)  # BGR color for R channel
CHANNEL_COLOR_GREEN = (0, 255, 0)  # BGR color for G channel
CHANNEL_COLOR_BLUE = (255, 0, 0)  # BGR color for B channel

# Simulated Audio Input Configuration
SIMULATED_FREQ_BASE_1 = 440.0  # Base frequency 1 (Hz)
SIMULATED_FREQ_BASE_2 = 220.0  # Base frequency 2 (Hz)
SIMULATED_FREQ_BASE_3 = 880.0  # Base frequency 3 (Hz)
SIMULATED_FREQ_WALK_STD_1 = 20.0  # Random walk std dev for freq 1
SIMULATED_FREQ_WALK_STD_2 = 15.0  # Random walk std dev for freq 2
SIMULATED_FREQ_WALK_STD_3 = 30.0  # Random walk std dev for freq 3
SIMULATED_FREQ_MIN_1 = 200.0  # Minimum frequency 1
SIMULATED_FREQ_MAX_1 = 1000.0  # Maximum frequency 1
SIMULATED_FREQ_MIN_2 = 100.0  # Minimum frequency 2
SIMULATED_FREQ_MAX_2 = 500.0  # Maximum frequency 2
SIMULATED_FREQ_MIN_3 = 400.0  # Minimum frequency 3
SIMULATED_FREQ_MAX_3 = 2000.0  # Maximum frequency 3
SIMULATED_AMP_BASE = 0.5  # Base amplitude
SIMULATED_AMP_RANGE = 0.5  # Amplitude range
SIMULATED_AMP_FREQ = 0.3  # Amplitude modulation frequency
SIMULATED_SIGNAL_AMP_1 = 0.4  # Signal amplitude for freq 1
SIMULATED_SIGNAL_AMP_2 = 0.3  # Signal amplitude for freq 2
SIMULATED_SIGNAL_AMP_3 = 0.2  # Signal amplitude for freq 3
SIMULATED_NOISE_BASE = 0.05  # Base noise level
SIMULATED_NOISE_RANGE = 0.15  # Noise level range
SIMULATED_NOISE_FREQ = 0.7  # Noise modulation frequency
SIMULATED_SPIKE_PROBABILITY = 0.15  # Probability of spike event
SIMULATED_SPIKE_MAGNITUDE_MIN = 1.5  # Minimum spike magnitude
SIMULATED_SPIKE_MAGNITUDE_MAX = 4.0  # Maximum spike magnitude
SIMULATED_SPIKE_LENGTH_MIN = 50  # Minimum spike length (samples)
SIMULATED_SPIKE_LENGTH_MAX = 200  # Maximum spike length (samples)
SIMULATED_SILENCE_PROBABILITY = 0.03  # Probability of silence event
SIMULATED_SILENCE_START_RATIO = 0.5  # Silence start position ratio
SIMULATED_SILENCE_LENGTH_MIN_RATIO = 0.25  # Minimum silence length ratio
SIMULATED_SILENCE_LENGTH_MAX_RATIO = 0.5  # Maximum silence length ratio
SIMULATED_SILENCE_ATTENUATION = 0.1  # Silence attenuation factor

# Abstraction Cycle Configuration
MATURITY_TIME_SCALES = {
    0: "1 second",
    1: "1 minute",
    2: "1 hour",
    3: "1 day",
    4: "1 week",
    5: "1 month",
    6: "1 year",
    7: "10 years"
}
DAY_COUNTER_LEVEL = 3  # Level 3 (1 day) increments day counter

