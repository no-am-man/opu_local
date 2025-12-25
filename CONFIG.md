# OPU Configuration Guide

This document explains all configuration parameters in `config.py` and how they affect the OPU's behavior.

## Table of Contents

1. [Surprise Sensitivity](#surprise-sensitivity)
2. [Audio Configuration](#audio-configuration)
3. [Visual Configuration](#visual-configuration)
4. [Memory & Introspection](#memory--introspection)
5. [Maturity & Evolution](#maturity--evolution)
6. [Display & Visualization](#display--visualization)
7. [State & Persistence](#state--persistence)
8. [YouTube Mode Configuration](#youtube-mode-configuration)
9. [Quick Reference: Making OPU More Sensitive](#quick-reference-making-opu-more-sensitive)

---

## Surprise Sensitivity

These parameters control how sensitive the OPU is to detecting surprises (changes in input).

### `INTROSPECTION_NOISE_FLOOR` (Default: `0.0001`)

**What it does:** Minimum sigma (standard deviation) value used in s_score calculation.

**Formula:** `s_score = |g_now - mu_history| / max(sigma_history, NOISE_FLOOR)`

**Effect:**
- **Lower value** = More sensitive (higher s_score for same input differences)
- **Higher value** = Less sensitive (lower s_score, filters out small changes)

**Example:**
- With `0.01`: Small audio change → s_score = 5.0
- With `0.001`: Same change → s_score = 50.0 (10x higher!)
- With `0.0001`: Same change → s_score = 500.0 (100x higher! - baby-like sensitivity)

**Recommended values:**
- `0.0001` - Extremely sensitive, baby-like (current default)
- `0.001` - Very sensitive
- `0.01` - Moderate sensitivity
- `0.1` - Low sensitivity (only large changes detected)

---

### `VISUAL_SURPRISE_THRESHOLD` (Default: `0.2`)

**What it does:** Minimum visual surprise score required to store a visual memory.

**Calibration Note:** Start at `0.2` for YouTube testing. If memory storage is too frequent (spammy), raise to `0.3` or `0.4`. If too quiet, lower to `0.1` or `0.15`.

**Effect:**
- **Lower value** = More visual memories stored (OPU remembers more visual events)
- **Higher value** = Fewer visual memories (only significant visual changes stored)

**Note:** This does NOT directly affect s_score calculation, only memory storage decisions.

**Recommended values:**
- `0.05` - Baby-like: very sensitive, remembers almost everything (current default)
- `0.1` - Store almost all visual changes
- `0.2` - Store moderate visual changes
- `0.5` - Store only significant visual changes

---

### `INTROSPECTION_AUDIO_MAX_HISTORY` (Default: `10`)

**What it does:** Maximum number of audio genomic bits kept in history for introspection.

**Effect:**
- **Smaller value** = Faster reaction to changes (less historical averaging, more reactive)
- **Larger value** = Slower reaction (more stable, but less reactive)

**Recommended values:**
- `10` - Baby-like: extremely reactive, very short memory (current default)
- `20-30` - Very reactive
- `50` - Moderate reactivity
- `100+` - Very stable, slow to adapt

---

### `INTROSPECTION_VISUAL_MAX_HISTORY` (Default: `10`)

**What it does:** Maximum number of visual frames kept in history for visual introspection.

**Effect:** Similar to audio history, but for visual processing.
- **Smaller value** = Faster reaction to visual changes (less historical averaging)
- **Larger value** = Slower reaction (more stable, but less reactive)

**Recommended values:**
- `10` - Baby-like: extremely reactive, very short memory (current default)
- `30` - More reactive to visual changes
- `50` - Balanced
- `100` - More stable visual processing

---

### `INTROSPECTION_MIN_DATA_POINTS` (Default: `2`)

**What it does:** Minimum number of data points needed before s_score can be calculated.

**Effect:**
- Until this many genomic bits are collected, s_score stays at 0.0
- After reaching this threshold, s_score calculation begins

**Recommended values:**
- `2` - Start calculating immediately (current default)
- `5` - Wait for more data before calculating
- `10` - Very conservative, needs more history

---

### `INTROSPECTION_VISUAL_MIN_FRAMES` (Default: `5`)

**What it does:** Minimum number of visual frames needed before visual introspection can calculate surprise scores.

**Effect:**
- Until this many frames are collected, visual s_score stays at 0.0
- After reaching this threshold, visual surprise calculation begins
- Lower values = Faster reaction to visual changes

**Recommended values:**
- `5` - Baby-like: reacts faster (current default)
- `10` - Moderate reaction speed
- `15` - Slower reaction, needs more history

---

## Audio Configuration

### `SAMPLE_RATE` (Default: `44100`)

**What it does:** Audio sample rate in Hz (samples per second).

**Effect:** Higher = better audio quality, but more CPU usage.

**Common values:**
- `44100` - CD quality (current default)
- `48000` - Professional audio
- `22050` - Lower quality, less CPU

---

### `CHUNK_SIZE` (Default: `1024`)

**What it does:** Number of audio samples processed per cycle.

**Effect:**
- **Smaller** = Lower latency, more CPU cycles
- **Larger** = Higher latency, fewer CPU cycles

**Recommended values:**
- `512` - Very low latency
- `1024` - Balanced (current default)
- `2048` - Lower CPU, higher latency

---

### `BASE_FREQUENCY` (Default: `220.0`)

**What it does:** Base frequency in Hz for audio output.

**Effect:** The fundamental frequency the OPU uses for audio feedback.

---

### `DITHERING_NOISE_SIGMA` (Default: `0.0001`)

**What it does:** Standard deviation of dithering noise added to prevent audio dead zones.

**Effect:** Prevents audio from flatlining to 0.0, which would cause divide-by-zero errors.

---

## Visual Configuration

### `USE_COLOR_CONSTANCY` (Default: `True`)

**What it does:** Enables shadow-invariant visual perception using normalized chromaticity.

**Effect:**
- **True** = OPU ignores lighting changes (shadows, flickers), only detects actual color/structure changes
- **False** = OPU responds to all changes including lighting (legacy mode)

**Recommended:** Keep `True` for better performance.

---

### `VISUAL_SURPRISE_THRESHOLD` (Default: `0.2`)

See [Surprise Sensitivity](#visual_surprise_threshold-default-02) section above.

---

### `PREVIEW_SCALE` (Default: `0.75`)

**What it does:** Scale factor for webcam preview window (75% of original size).

**Effect:** Adjusts the size of the visual preview display.

---

## Memory & Introspection

### `INTROSPECTION_NOISE_FLOOR` (Default: `0.001`)

See [Surprise Sensitivity](#introspection_noise_floor-default-0001) section above.

---

### `INTROSPECTION_SIGMA_DEFAULT` (Default: `0.05`)

**What it does:** Default sigma value used when visual history is zero. Controls the width of the expectation bell curve - narrower curve = more sensitive to deviations.

**Calibration Note:** Lower values (0.02-0.05) make OPU more "jumpy" and reactive. Higher values (0.08-0.1) make it more stable and less reactive.

**Effect:** Prevents divide-by-zero errors in visual introspection calculations.

---

### `BRAIN_CONSOLIDATION_RATIO_L0` through `BRAIN_CONSOLIDATION_RATIO_L7`

**What it does:** Defines how many items at Level N are needed to create 1 item at Level N+1.

**Effect:** Controls the memory consolidation rate through the 8-layer hierarchy.

**Example:** If `BRAIN_CONSOLIDATION_RATIO_L0 = 20`, then 20 Level 0 memories → 1 Level 1 memory.

---

## Maturity & Evolution

### `MATURITY_LEVEL_TIMES`

**What it does:** Time scales for each maturity level (in seconds).

**Levels:**
- Level 0: 1 second
- Level 1: 1 minute (60s)
- Level 2: 1 hour (3600s)
- Level 3: 1 day (86400s)
- Level 4: 1 week (604800s)
- Level 5: 1 month (2592000s)
- Level 6: 1 year (31536000s)
- Level 7: 4 years (126230400s)

**Effect:** Controls how long each abstraction cycle takes.

---

### `TIME_SCALE_MULTIPLIER` (Default: `1.0`)

**What it does:** Multiplier for time scales (for simulation speed).

**Effect:**
- `1.0` = Real time (current default)
- `10.0` = 10x faster progression
- `0.1` = 10x slower progression

---

### `BRAIN_STABILITY_THRESHOLD` (Default: `3.0`)

**What it does:** Base stability threshold for character profile.

**Effect:** Higher = harder to surprise the OPU (more stable/calm).

---

## Display & Visualization

### `WINDOW_SIZE` (Default: `(6, 4)`)

**What it does:** Cognitive Map window size in inches.

**Effect:** Adjusts the size of the visualization window.

---

### `VISUALIZATION_UPDATE_RATE` (Default: `30`)

**What it does:** Target frames per second for visualization updates.

**Effect:** Higher = smoother animation, more CPU usage.

---

### `ALERT_THRESHOLD` (Default: `3.0`)

**What it does:** s_score threshold for RED alert status in visual display.

**Effect:** When s_score >= this value, display shows red alert.

---

### `INTEREST_THRESHOLD` (Default: `1.5`)

**What it does:** s_score threshold for YELLOW interest status in visual display.

**Effect:** When s_score >= this value but < ALERT_THRESHOLD, display shows yellow interest.

---

## State & Persistence

### `STATE_FILE` (Default: `"opu_state.json"`)

**What it does:** Path to the file where OPU state is saved/loaded.

**Effect:** Change this to use a different state file or location.

---

### `OPU_VERSION` (Default: `"3.4.3"`)

**What it does:** Current OPU version number.

**Effect:** Used for version checking and display.

---

## Expression & Audio Output

### `AFL_NOISE_GATE_THRESHOLD` (Default: `0.1`)

**What it does:** Minimum s_score to generate audio output.

**Effect:** Filters out very low s_score values from producing sound.

---

### `AFL_SPEAKING_THRESHOLD` (Default: `0.6`)

**What it does:** Minimum s_score to trigger speaking/articulation.

**Effect:** Higher = OPU speaks less frequently.

---

### `PHONEME_SPEECH_THRESHOLD` (Default: `1.5`)

**What it does:** Minimum s_score to recognize as speech (not noise).

**Effect:** Filters phoneme analysis to only process significant audio events.

---

## YouTube Mode Configuration

These parameters control YouTube video/audio streaming functionality.

### `AUDIO_SENSE_YOUTUBE` (Default: `"AUDIO_V2"`)

**What it does:** Sense label for YouTube audio stream input.

**Effect:** Used to distinguish YouTube audio from microphone audio (`AUDIO_V1`) in memory storage.

---

### `VIDEO_SENSE_YOUTUBE` (Default: `"VIDEO_V2"`)

**What it does:** Sense label for YouTube video stream input.

**Effect:** Used to distinguish YouTube video from webcam video (`VIDEO_V1`) in memory storage.

---

### `YOUTUBE_VIDEO_RESIZE_DIM` (Default: `(640, 360)`)

**What it does:** Target resolution for YouTube video frames (width, height in pixels).

**Effect:**
- **Smaller dimensions** = Faster processing, lower CPU usage
- **Larger dimensions** = Better quality, higher CPU usage

**Recommended values:**
- `(640, 360)` - Balanced (current default)
- `(320, 180)` - Faster processing
- `(1280, 720)` - Higher quality

---

### `YOUTUBE_AUDIO_VOLUME_MULTIPLIER` (Default: `0.5`)

**What it does:** Multiplier to reduce YouTube audio volume before processing.

**Effect:**
- **Lower value** = Quieter audio (prevents clipping)
- **Higher value** = Louder audio (may cause clipping)

**Recommended values:**
- `0.5` - Reduces volume by 50% (current default)
- `0.3` - Quieter, safer
- `1.0` - Full volume (may clip)

---

### YouTube HUD Display Parameters

These control the HUD overlay displayed on YouTube video frames:

- **`YOUTUBE_HUD_POS_X`** (Default: `10`) - X position for HUD text
- **`YOUTUBE_HUD_POS_Y_LINE1`** (Default: `30`) - Y position for first HUD line (scores)
- **`YOUTUBE_HUD_POS_Y_LINE2`** (Default: `60`) - Y position for second HUD line (title)
- **`YOUTUBE_HUD_POS_Y_LINE3`** (Default: `90`) - Y position for third HUD line (frame/FPS)
- **`YOUTUBE_HUD_FONT_SCALE_LARGE`** (Default: `0.6`) - Font scale for main HUD text
- **`YOUTUBE_HUD_FONT_SCALE_SMALL`** (Default: `0.5`) - Font scale for secondary HUD text
- **`YOUTUBE_HUD_FONT_THICKNESS`** (Default: `2`) - Font thickness for main HUD text
- **`YOUTUBE_HUD_FONT_THICKNESS_THIN`** (Default: `1`) - Font thickness for secondary HUD text

**Effect:** Adjusts the position and appearance of the HUD overlay on YouTube video frames.

---

## Quick Reference: Making OPU More Sensitive

To make the OPU more sensitive to surprises (higher s_score more frequently):

1. **Lower `INTROSPECTION_NOISE_FLOOR`**: `0.001` → `0.0001` (10x more sensitive)
2. **Lower `VISUAL_SURPRISE_THRESHOLD`**: `0.2` → `0.05` (stores more memories)
3. **Reduce `INTROSPECTION_AUDIO_MAX_HISTORY`**: `50` → `10` (faster reaction)
4. **Reduce `INTROSPECTION_VISUAL_MAX_HISTORY`**: `50` → `10` (faster visual reaction)
5. **Lower `INTROSPECTION_VISUAL_MIN_FRAMES`**: `10` → `5` (reacts faster)

To make the OPU less sensitive (calmer, fewer surprises):

1. **Raise `INTROSPECTION_NOISE_FLOOR`**: `0.0001` → `0.001` or `0.01`
2. **Raise `VISUAL_SURPRISE_THRESHOLD`**: `0.05` → `0.2` or `0.5`
3. **Increase `INTROSPECTION_AUDIO_MAX_HISTORY`**: `10` → `30` or `50`
4. **Increase `INTROSPECTION_VISUAL_MAX_HISTORY`**: `10` → `30` or `50`
5. **Raise `INTROSPECTION_VISUAL_MIN_FRAMES`**: `5` → `10` or `15`

---

## Notes

- Most parameters can be adjusted while the OPU is running, but some require a restart
- Changes to `config.py` take effect on the next OPU startup
- Be careful with very low `INTROSPECTION_NOISE_FLOOR` values - they can cause false positives from noise
- The OPU is designed to work well with default values, but tuning can optimize for specific use cases

---

For more information, see the main [README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

