"""
Configuration parameters for the OPU (Orthogonal Processing Unit).
Contains the Genesis Constant and safety limits.

OPU v3.4.3 - MIT License
"""

# Version
OPU_VERSION = "3.4.3"  # v3.4.3: Language system refactoring - extracted common utilities, reduced duplication, improved maintainability

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
# Level 7: 4 years (126230400s) - Core Identity / Deep Wisdom

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
    7: 126230400.0     # 4 years (Deep Wisdom)
}

# Apply time scale multiplier
MATURITY_LEVEL_TIMES = {k: v / TIME_SCALE_MULTIPLIER for k, v in MATURITY_LEVEL_TIMES.items()}

# Legacy support (for backward compatibility)
ABSTRACTION_CYCLE_SECONDS = MATURITY_LEVEL_TIMES[3]  # 1 day
MATURITY_INCREMENT = 0.0125  # Smaller increment for 8 levels (1.0 / 80 steps to reach full maturity)

# Visualization Configuration
VISUALIZATION_UPDATE_RATE = 30  # FPS
WINDOW_SIZE = (6, 4)  # inches (reduced from 12x8 for smaller window)

# Persistence Configuration
STATE_FILE = "opu_state.json"  # Path to save/load OPU state

# Sense Labels (for extensible input system)
# Each input source is labeled with a sense identifier
# This allows the OPU to track and learn from different input modalities
# Future senses can be added: TOUCH_V1, TEMPERATURE_V1, etc.
AUDIO_SENSE = "AUDIO_V1"  # Microphone input
VIDEO_SENSE = "VIDEO_V1"  # Webcam input
AUDIO_SENSE_YOUTUBE = "AUDIO_V2"  # YouTube audio stream
VIDEO_SENSE_YOUTUBE = "VIDEO_V2"  # YouTube video stream
BRAIN_DEFAULT_SENSE_LABEL = "UNKNOWN"  # Default sense label for unknown input sources

# Processing Thresholds
VISUAL_SURPRISE_THRESHOLD = 0.2  # Minimum visual surprise to store memory (tunable: start at 0.2, raise to 0.3-0.4 if too spammy)
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
TEXT_COLOR_WHITE = (255, 255, 255)  # White color for text overlay
STATUS_COLOR_RED = (0, 0, 255)  # BGR color for RED alert
STATUS_COLOR_YELLOW = (0, 255, 255)  # BGR color for YELLOW interest
STATUS_COLOR_GREEN = (0, 255, 0)  # BGR color for GREEN calm

# YouTube OPU Configuration
YOUTUBE_VIDEO_RESIZE_DIM = (640, 360)  # Resize video frames for faster processing
YOUTUBE_AUDIO_VOLUME_MULTIPLIER = 0.5  # Reduce YouTube audio volume

# Auto-start YouTube mode when main.py starts (set to None to disable)
# If set to a YouTube URL, the app will automatically launch YouTube mode instead of regular mode
# YouTube Auto-Start Configuration
# Set to None to disable auto-start (default: normal OPU mode)
# Set to a URL string to auto-start YouTube mode
YOUTUBE_AUTO_START_URL = None  # Default: None (normal OPU mode)
# YOUTUBE_AUTO_START_URL = "https://www.youtube.com/watch?v=jfKfPfyJRdk"  # Uncomment to enable auto-start
YOUTUBE_HUD_POS_X = 10  # X position for HUD text
YOUTUBE_HUD_POS_Y_LINE1 = 30  # Y position for first HUD line
YOUTUBE_HUD_POS_Y_LINE2 = 60  # Y position for second HUD line
YOUTUBE_HUD_POS_Y_LINE3 = 90  # Y position for third HUD line
YOUTUBE_HUD_FONT_SCALE_LARGE = 0.6  # Font scale for main HUD text
YOUTUBE_HUD_FONT_SCALE_SMALL = 0.5  # Font scale for secondary HUD text
YOUTUBE_HUD_FONT_THICKNESS = 2  # Font thickness for main HUD text
YOUTUBE_HUD_FONT_THICKNESS_THIN = 1  # Font thickness for secondary HUD text
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
    7: "4 years"
}
DAY_COUNTER_LEVEL = 3  # Level 3 (1 day) increments day counter

# Aesthetic Feedback Loop Configuration
AFL_BASE_BREATH_RATE = 0.2  # Hz (Slow breath = ~0.2Hz = 12 breaths/min)
AFL_BREATH_RATE_MULTIPLIER = 2.0  # Breath rate multiplier for stress response
AFL_BREATH_SMOOTHING = 0.05  # Smoothing factor for breath rate transitions
AFL_BREATH_BASE_LEVEL = 0.4  # Base breath envelope level (never goes to zero)
AFL_BREATH_CAPACITY = 0.6  # Breath envelope capacity (0.4 + 0.6 = 1.0 max)
AFL_PITCH_SMOOTHING = 0.1  # Smoothing factor for pitch glide (portamento)
AFL_AMPLITUDE_SMOOTHING = 0.1  # Smoothing factor for amplitude transitions
AFL_SYLLABLE_RATE = 8.0  # Hz (LFO for syllable rate when speaking)
AFL_ARTICULATION_MIN = 0.6  # Minimum articulation amplitude (wah-wah envelope)
AFL_ARTICULATION_RANGE = 0.4  # Articulation range (0.6 to 1.0)
AFL_MASTER_GAIN = 0.5  # Master gain for final signal mix
AFL_NOISE_GATE_THRESHOLD = 0.1  # Minimum s_score to generate audio
AFL_VOLUME_DIVISOR = 2.0  # Divisor for volume calculation (s_score / 2.0)
AFL_PITCH_DIVISOR = 5.0  # Divisor for pitch calculation (s_score / 5.0)
AFL_MIN_FREQUENCY = 50.0  # Minimum frequency in Hz
AFL_MAX_FREQUENCY = 2000.0  # Maximum frequency in Hz
AFL_SPEAKING_THRESHOLD = 0.6  # Minimum s_score to trigger speaking/articulation
AFL_ACTIVE_THRESHOLD = 0.05  # Minimum amplitude to consider OPU active
AFL_AUDIO_BLOCKSIZE = 1024  # Audio buffer blocksize for stability

# Phoneme Analyzer Configuration
PHONEME_SPEECH_THRESHOLD = 1.5  # Minimum s_score to recognize as speech (not noise)
PHONEME_VOWEL_BOUNDARY = 3.0  # Boundary between vowels and fricatives
PHONEME_FRICATIVE_BOUNDARY = 6.0  # Boundary between fricatives and plosives
PHONEME_PITCH_THRESHOLD = 200.0  # Pitch threshold for vowel selection (a vs o)
PHONEME_MAX_HISTORY = 10000  # Maximum phoneme history entries (prevents memory leak)
PHONEME_USE_FULL_INVENTORY = True  # Use full IPA inventory (~44 phonemes) instead of basic set
PHONEME_USE_UNIVERSAL_INVENTORY = True  # Use universal inventory supporting all languages (~150+ phonemes)
PHONEME_LANGUAGE_FAMILIES = None  # Set of language families to include (None = all families)
# Options: 'romance', 'germanic', 'slavic', 'semitic', 'sino-tibetan', 'dravidian',
#          'japonic', 'koreanic', 'african', 'polynesian', 'native_american', etc.

# Speech Synthesis Configuration
SPEECH_USE_TTS = True  # Use TTS library (pyttsx3) for word-level synthesis
SPEECH_TTS_RATE = 150  # TTS speech rate (words per minute)
SPEECH_TTS_VOLUME = 0.8  # TTS volume (0.0 to 1.0)
SPEECH_FORMANT_ENABLED = True  # Enable formant synthesis for phoneme-level generation

# Speech Recognition Configuration
SPEECH_RECOGNITION_ENABLED = False  # Enable speech recognition for comprehension
SPEECH_USE_WHISPER = True  # Use Whisper for recognition (requires openai-whisper)
SPEECH_WHISPER_MODEL = "base"  # Whisper model size: tiny, base, small, medium, large
SPEECH_RECOGNITION_TIMEOUT = 1.0  # Recognition timeout in seconds
SPEECH_PHRASE_TIME_LIMIT = 5.0  # Maximum phrase length in seconds

# Language Memory Configuration
LANGUAGE_MEMORY_MAX_WORDS = 10000  # Maximum number of words to store
LANGUAGE_MEMORY_MAX_SEQUENCES = 1000  # Maximum number of word sequences (phrases)
LANGUAGE_MEMORY_ENABLED = True  # Enable language learning and word association

# Brain Configuration
BRAIN_CHILD_PITCH = 440.0  # Base pitch for child state (Hz)
BRAIN_SAGE_PITCH = 110.0  # Base pitch for sage state (Hz)
BRAIN_STABILITY_THRESHOLD = 3.0  # Stability threshold for character profile
BRAIN_EMOTION_DEFAULT_INTENSITY = 0.0  # Default emotion intensity
BRAIN_EMOTION_DEFAULT_LABEL = 'neutral'  # Default emotion label
BRAIN_MAX_MEMORY_LEVEL = 7  # Maximum memory level (0-7 for 8 layers)
BRAIN_EVOLUTION_MIN_LEVEL = 3  # Minimum level to trigger character evolution (Level 3 = 1 day)
BRAIN_DEFAULT_CONSOLIDATION_RATIO = 20  # Default consolidation ratio if level not found
BRAIN_PATTERN_STRENGTH_MIN_SAMPLES = 1  # Minimum samples to calculate pattern strength std dev

# Brain Consolidation Ratios (8-Layer Architecture)
# How many items of Level N make 1 item of Level N+1
# Based on roughly converting seconds -> minutes -> hours etc.
# Assuming input rate of ~20Hz (50ms per cycle)
BRAIN_CONSOLIDATION_RATIO_L0 = 20  # 20 raw inputs (~1s) -> 1 L1 item
BRAIN_CONSOLIDATION_RATIO_L1 = 60  # 60 L1 items (1m) -> 1 L2 item
BRAIN_CONSOLIDATION_RATIO_L2 = 60  # 60 L2 items (1h) -> 1 L3 item
BRAIN_CONSOLIDATION_RATIO_L3 = 24  # 24 L3 items (1d) -> 1 L4 item
BRAIN_CONSOLIDATION_RATIO_L4 = 7   # 7  L4 items (1w) -> 1 L5 item
BRAIN_CONSOLIDATION_RATIO_L5 = 4   # 4  L5 items (1mo)-> 1 L6 item
BRAIN_CONSOLIDATION_RATIO_L6 = 12  # 12 L6 items (1y) -> 1 L7 item
BRAIN_CONSOLIDATION_RATIO_L7 = 10  # 10 L7 items -> Full Wisdom

# Visual Perception Configuration
VISUAL_EPSILON = 0.001  # Epsilon to prevent division by zero in color constancy
VISUAL_COLOR_CONSTANCY_SCALE = 100.0  # Scale factor for normalized chromaticity (0-1 to 0-255 range)
VISUAL_CAMERA_WIDTH = 640  # Camera capture width (pixels)
VISUAL_CAMERA_HEIGHT = 480  # Camera capture height (pixels)
VISUAL_CAMERA_FPS = 15  # Camera frame rate (frames per second)

# Persistence Configuration
PERSISTENCE_DEFAULT_DAY_COUNTER = 0  # Default day counter value
PERSISTENCE_DEFAULT_S_SCORE = 0.0  # Default surprise score
PERSISTENCE_STATE_VERSION = "1.0"  # State file version
PERSISTENCE_TEMP_FILE_SUFFIX = ".tmp"  # Temporary file suffix for atomic writes

# Main Event Loop Configuration
MAIN_DEFAULT_CONFIDENCE_THRESHOLD = 0.5  # Default confidence threshold for object detection
MAIN_DEFAULT_SURPRISE_SCORE = 0.0  # Default surprise score value
MAIN_STATE_VIEWER_UPDATE_INTERVAL = 0.1  # State viewer update interval (seconds)
MAIN_EMPTY_VISUAL_VECTOR = [0.0, 0.0, 0.0]  # Empty visual vector (3 channels)

# OPU Configuration
OPU_EMOTION_HISTORY_MAX_SIZE = 1000  # Maximum emotion history entries (prevents unbounded growth)
OPU_EMOTION_DEFAULT_CONFIDENCE = 0.0  # Default emotion confidence value
OPU_EMOTION_DEFAULT_TOTAL = 0  # Default total emotions count
OPU_EMOTION_UNKNOWN_LABEL = 'unknown'  # Default unknown emotion label

# State Viewer Configuration
STATE_VIEWER_EMOTION_HISTORY_MAX_DISPLAY = 100  # Maximum emotions to display in real-time panel (prevents UI lag)
STATE_VIEWER_EMOTION_TIMESTAMP_FORMAT = "%H:%M:%S"  # Timestamp format for emotion history display
STATE_VIEWER_EMOTION_UNKNOWN_TIMESTAMP = "??:??:??"  # Placeholder for invalid timestamps

# Emotion Visualization Configuration
EMOTION_VIZ_DPI = 100  # DPI for matplotlib figures
EMOTION_VIZ_SECONDS_TO_MINUTES = 60.0  # Conversion factor for time normalization
EMOTION_VIZ_SCATTER_ALPHA = 0.6  # Transparency for scatter plot points
EMOTION_VIZ_SCATTER_SIZE = 30  # Size of scatter plot points
EMOTION_VIZ_BOX_ALPHA = 0.7  # Transparency for box plot boxes
EMOTION_VIZ_GRID_ALPHA = 0.3  # Transparency for grid lines
EMOTION_VIZ_CONFIDENCE_MAX = 1.1  # Maximum y-axis value for confidence plots
EMOTION_VIZ_PIE_START_ANGLE = 90  # Starting angle for pie charts (degrees)
EMOTION_VIZ_PIE_TITLE_PAD = 10  # Padding for pie chart title
EMOTION_VIZ_TITLE_FONTSIZE = 12  # Font size for chart titles
EMOTION_VIZ_AXIS_FONTSIZE = 10  # Font size for axis labels
EMOTION_VIZ_LEGEND_FONTSIZE = 8  # Font size for legends
EMOTION_VIZ_TICK_FONTSIZE = 8  # Font size for tick labels

# Object Detection Emotion Colors (BGR format for OpenCV)
DETECTION_EMOTION_COLOR_HAPPY = (0, 255, 0)      # Green
DETECTION_EMOTION_COLOR_SAD = (255, 0, 0)        # Blue
DETECTION_EMOTION_COLOR_ANGRY = (0, 0, 255)      # Red
DETECTION_EMOTION_COLOR_SURPRISE = (0, 255, 255) # Yellow
DETECTION_EMOTION_COLOR_FEAR = (128, 0, 128)     # Purple
DETECTION_EMOTION_COLOR_DISGUST = (0, 128, 128)  # Teal
DETECTION_EMOTION_COLOR_NEUTRAL = (128, 128, 128) # Gray
DETECTION_EMOTION_COLOR_DEFAULT = (128, 128, 128) # Default gray
DETECTION_FACE_COLOR_NO_EMOTION = (0, 255, 0)    # Green for faces without emotion
DETECTION_OBJECT_COLOR = (255, 0, 0)             # Blue for other objects

# State Viewer Configuration
STATE_VIEWER_DEFAULT_MATURITY_INDEX = 0.0  # Default maturity index value
STATE_VIEWER_DEFAULT_PITCH = 440.0  # Default base pitch (Hz)
STATE_VIEWER_DEFAULT_STABILITY_THRESHOLD = 3.0  # Default stability threshold
STATE_VIEWER_DEFAULT_S_SCORE = 0.0  # Default surprise score
STATE_VIEWER_DEFAULT_COHERENCE = 0.0  # Default coherence value
STATE_VIEWER_DEFAULT_G_NOW = 0.0  # Default genomic bit value
STATE_VIEWER_DEFAULT_CONFIDENCE = 0.0  # Default emotion confidence
STATE_VIEWER_DEFAULT_SPEECH_THRESHOLD = 0.0  # Default speech threshold
STATE_VIEWER_MEMORY_LEVELS_COUNT = 8  # Number of memory levels (0-7)

# Command Pattern Configuration
COMMAND_MAX_HISTORY_DEFAULT = 100  # Default maximum command history size
COMMAND_UNDO_SEARCH_MAX_LEVEL = 7  # Maximum level to search when undoing (0-7 for 8-layer architecture)

# Introspection Strategy Configuration
INTROSPECTION_AUDIO_MAX_HISTORY = 10  # Maximum history size for audio introspection (baby-like: extremely reactive, very short memory)
INTROSPECTION_VISUAL_MAX_HISTORY = 10  # Maximum history size for visual introspection (baby-like: extremely reactive, very short memory)
INTROSPECTION_MIN_DATA_POINTS = 2  # Minimum data points needed for meaningful introspection
INTROSPECTION_VISUAL_MIN_FRAMES = 5  # Minimum frames needed for visual introspection (baby-like: reacts faster)
INTROSPECTION_NOISE_FLOOR = 0.0001  # Minimum sigma to prevent false high scores from silence (baby-like: extremely sensitive, 100x more than default)
INTROSPECTION_SIGMA_DEFAULT = 0.05  # Default sigma when history is zero (baby-like: very sensitive, narrow expectation curve)
INTROSPECTION_DEFAULT_S_SCORE = 0.0  # Default surprise score
INTROSPECTION_DEFAULT_COHERENCE = 1.0  # Default coherence (perfect when no history)
INTROSPECTION_DEFAULT_G_NOW = 0.0  # Default genomic bit value

# Maturity State Configuration
MATURITY_INSTANT_PITCH_MULTIPLIER = 1.0  # Level 0: 440Hz
MATURITY_INSTANT_STABILITY = 2.5  # Level 0: Very reactive
MATURITY_CHILD_PITCH_MULTIPLIER = 0.95  # Level 1: ~418Hz
MATURITY_CHILD_STABILITY = 3.0  # Level 1
MATURITY_INFANT_PITCH_MULTIPLIER = 0.9  # Level 2: ~396Hz
MATURITY_INFANT_STABILITY = 3.5  # Level 2
MATURITY_ADOLESCENT_PITCH_MULTIPLIER = 0.7  # Level 3: ~308Hz
MATURITY_ADOLESCENT_STABILITY = 4.5  # Level 3
MATURITY_ADULT_PITCH_MULTIPLIER = 0.5  # Level 4: ~220Hz
MATURITY_ADULT_STABILITY = 5.5  # Level 4
MATURITY_ELDER_PITCH_MULTIPLIER = 0.35  # Level 5: ~154Hz
MATURITY_ELDER_STABILITY = 6.5  # Level 5
MATURITY_SAGE_PITCH_MULTIPLIER = 0.25  # Level 6: 110Hz
MATURITY_SAGE_STABILITY = 8.0  # Level 6
MATURITY_SCIRE_PITCH_MULTIPLIER = 0.2  # Level 7: ~88Hz
MATURITY_SCIRE_STABILITY = 10.0  # Level 7: Very hard to surprise
MATURITY_BASE_PITCH = 440.0  # Base pitch in Hz (A4)
MATURITY_PITCH_RANGE = 330.0  # Pitch range (440 - 110)
MATURITY_STABILITY_BASE = 3.0  # Base stability threshold for interpolation
MATURITY_STABILITY_RANGE = 5.0  # Stability range for interpolation (3.0 to 8.0)

# Sense Decorator Configuration
DECORATOR_NOISE_GATE_THRESHOLD = 0.1  # Default noise gate threshold (amplitude below which input is silenced)
DECORATOR_HIGHPASS_ALPHA = 0.95  # High-pass filter coefficient (0-1, higher = more filtering)
DECORATOR_AMPLIFICATION_GAIN = 1.0  # Default amplification gain (>1.0 = amplify, <1.0 = attenuate)

# Sense Factory Configuration
SENSE_FACTORY_DEFAULT_GENOMIC_BIT = 0.0  # Default genomic bit value for empty visual input
SENSE_FACTORY_EMPTY_VISUAL_VECTOR = [0.0, 0.0, 0.0]  # Empty visual vector (3 channels)

# Genesis Kernel Configuration
GENESIS_VETO_LOG_INTERVAL = 50  # Log veto message every N times (reduces verbosity)

# OPU Builder Configuration
OPU_BUILDER_DEFAULT_AUDIO_HISTORY = 50  # Default audio history size for builder
OPU_BUILDER_DEFAULT_VISUAL_HISTORY = 50  # Default visual history size for builder

# Mic/Perception Configuration
MIC_DEFAULT_GENOMIC_BIT = 0.0  # Default genomic bit for empty input
MIC_DEFAULT_MAGNITUDE = 0.0  # Default magnitude for empty input
