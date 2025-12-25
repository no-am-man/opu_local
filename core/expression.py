"""
The Voice: Aesthetic Feedback Loop & Phoneme Analysis.
Maps surprise scores to audio frequencies and phonemes.
"""

import numpy as np
import sounddevice as sd
from typing import Optional
from config import (
    BASE_FREQUENCY, SAMPLE_RATE,
    AFL_BASE_BREATH_RATE, AFL_BREATH_RATE_MULTIPLIER, AFL_BREATH_SMOOTHING,
    AFL_BREATH_BASE_LEVEL, AFL_BREATH_CAPACITY, AFL_PITCH_SMOOTHING,
    AFL_AMPLITUDE_SMOOTHING, AFL_SYLLABLE_RATE, AFL_ARTICULATION_MIN,
    AFL_ARTICULATION_RANGE, AFL_MASTER_GAIN, AFL_NOISE_GATE_THRESHOLD,
    AFL_VOLUME_DIVISOR, AFL_PITCH_DIVISOR, AFL_MIN_FREQUENCY, AFL_MAX_FREQUENCY,
    AFL_SPEAKING_THRESHOLD, AFL_ACTIVE_THRESHOLD, AFL_AUDIO_BLOCKSIZE,
    PHONEME_SPEECH_THRESHOLD, PHONEME_VOWEL_BOUNDARY, PHONEME_FRICATIVE_BOUNDARY,
    PHONEME_PITCH_THRESHOLD, PHONEME_MAX_HISTORY
)
from core.phoneme_inventory import PHONEME_INVENTORY
from core.universal_phoneme_inventory import UNIVERSAL_PHONEME_INVENTORY
from config import PHONEME_USE_UNIVERSAL_INVENTORY, PHONEME_LANGUAGE_FAMILIES
from core.formant_synthesizer import FORMANT_SYNTHESIZER


class AestheticFeedbackLoop:
    """
    Continuous Phase Oscillator with Syllabic Articulation AND Respiratory Cycle.
    The breath cycle makes the OPU feel truly "biological" rather than robotic.
    """
    
    def __init__(self, base_pitch=440.0):
        self.base_pitch = base_pitch
        self.sample_rate = SAMPLE_RATE
        
        # Oscillator State
        self.current_frequency = base_pitch
        self.target_frequency = base_pitch
        self.phase = 0.0
        
        # Amplitude State
        self.current_amp = 0.0
        self.target_amp = 0.0
        
        # THE LUNGS - Respiratory Cycle
        self.breath_phase = 0.0
        self.breath_rate = AFL_BASE_BREATH_RATE
        self.current_breath_rate = AFL_BASE_BREATH_RATE
        
        # Articulation (The "Talking" Rhythm)
        self.syllable_phase = 0.0
        self.is_speaking = False
        
        # Audio Stream State
        self.stream = None
        self.running = False
        
        self.start()
    
    def callback(self, outdata, frames, time_info, status):
        """Audio callback - generates continuous waveform with phase continuity and breathing."""
        if status:
            print(f"[AFL] Audio output status: {status}")
        
        self._update_breath_rate()
        breath_envelope = self._generate_breath_envelope(frames)
        self._smooth_pitch_and_amplitude()
        carrier = self._generate_carrier_wave(frames)
        articulation = self._apply_syllabic_rhythm(frames)
        signal = self._mix_final_signal(carrier, articulation, breath_envelope)
        
        outdata[:] = signal.reshape(-1, 1).astype(np.float32)
    
    def _update_breath_rate(self):
        """Update breath rate based on stress/attention level."""
        target_breath_rate = AFL_BASE_BREATH_RATE + (self.target_amp * AFL_BREATH_RATE_MULTIPLIER)
        self.current_breath_rate += (target_breath_rate - self.current_breath_rate) * AFL_BREATH_SMOOTHING
    
    def _generate_breath_envelope(self, frames):
        """Generate breath envelope (respiratory cycle modulation)."""
        breath_inc = 2 * np.pi * self.current_breath_rate / self.sample_rate
        breath_phases = self.breath_phase + np.arange(frames) * breath_inc
        self.breath_phase = breath_phases[-1] % (2 * np.pi)
        return AFL_BREATH_BASE_LEVEL + (AFL_BREATH_CAPACITY * (0.5 + 0.5 * np.sin(breath_phases)))
    
    def _smooth_pitch_and_amplitude(self):
        """Apply smoothing (portamento) to pitch and amplitude transitions."""
        self.current_frequency += (self.target_frequency - self.current_frequency) * AFL_PITCH_SMOOTHING
        self.current_amp += (self.target_amp - self.current_amp) * AFL_AMPLITUDE_SMOOTHING
    
    def _generate_carrier_wave(self, frames):
        """Generate carrier wave (the voice)."""
        phase_increment = 2 * np.pi * self.current_frequency / self.sample_rate
        phases = self.phase + np.arange(frames) * phase_increment
        self.phase = phases[-1] % (2 * np.pi)
        return np.sin(phases)
    
    def _apply_syllabic_rhythm(self, frames):
        """Apply syllabic rhythm (articulation) when speaking."""
        if self.is_speaking:
            syllable_inc = 2 * np.pi * AFL_SYLLABLE_RATE / self.sample_rate
            syllable_phases = self.syllable_phase + np.arange(frames) * syllable_inc
            self.syllable_phase = syllable_phases[-1] % (2 * np.pi)
            return AFL_ARTICULATION_MIN + (AFL_ARTICULATION_RANGE * np.sin(syllable_phases))
        return 1.0
    
    def _mix_final_signal(self, carrier, articulation, breath_envelope):
        """Mix all components into final audio signal."""
        return carrier * self.current_amp * articulation * breath_envelope * AFL_MASTER_GAIN
    
    def start(self):
        try:
            self.stream = sd.OutputStream(
                channels=1,
                callback=self.callback,
                samplerate=self.sample_rate,
                dtype=np.float32,
                latency='low',
                blocksize=AFL_AUDIO_BLOCKSIZE
            )
            self.stream.start()
            self.running = True
            print("[AFL] Continuous phase oscillator initialized.")
        except Exception as e:
            print(f"[AFL] Error: {e}")
            self.running = False
            
    def update_pitch(self, base_pitch):
        self.base_pitch = base_pitch

    def play_tone(self, s_score, duration=None):
        """
        Updates the continuous oscillator based on surprise.
        
        Tuned for higher sensitivity - makes OPU more "chatty" and responsive
        to small changes in the environment.
        """
        if self._apply_noise_gate(s_score):
            return
        
        self._calculate_volume(s_score)
        self._calculate_pitch(s_score)
        self._determine_speaking_state(s_score)
    
    def _apply_noise_gate(self, s_score):
        """Apply noise gate - silence output if below threshold."""
        if s_score < AFL_NOISE_GATE_THRESHOLD:
            self.target_amp = 0.0
            self.is_speaking = False
            return True
        return False
    
    def _calculate_volume(self, s_score):
        """Calculate target amplitude based on surprise score."""
        self.target_amp = min(1.0, s_score / AFL_VOLUME_DIVISOR)
    
    def _calculate_pitch(self, s_score):
        """Calculate target frequency based on surprise score."""
        new_freq = self.base_pitch * (1.0 + s_score / AFL_PITCH_DIVISOR)
        self.target_frequency = np.clip(new_freq, AFL_MIN_FREQUENCY, AFL_MAX_FREQUENCY)
    
    def _determine_speaking_state(self, s_score):
        """Determine if OPU should be speaking (articulating)."""
        self.is_speaking = s_score > AFL_SPEAKING_THRESHOLD

    def is_active(self):
        """
        Returns True if the OPU is currently generating audio output.
        Used for feedback prevention - mute microphone when OPU is speaking.
        """
        return self._is_target_amplitude_active() or self._is_current_amplitude_active()
    
    def _is_target_amplitude_active(self):
        """Check if target amplitude exceeds active threshold."""
        return self.target_amp > AFL_ACTIVE_THRESHOLD
    
    def _is_current_amplitude_active(self):
        """Check if current amplitude exceeds active threshold."""
        return self.current_amp > AFL_ACTIVE_THRESHOLD
    
    def cleanup(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()


class PhonemeAnalyzer:
    """
    Maps s_score ranges to phonemes using full English IPA inventory.
    Filters out noise and only recognizes "Spoken" sounds with Structural Intent.
    Now supports ~44 English phonemes with intelligent selection based on
    s_score, pitch, and spectral features.
    """
    
    def __init__(self, speech_threshold=None, max_history=None, use_full_inventory=True, use_universal_inventory=None):
        self.speech_threshold = speech_threshold or PHONEME_SPEECH_THRESHOLD
        self.phoneme_history = []
        self.max_history = max_history or PHONEME_MAX_HISTORY
        self.use_full_inventory = use_full_inventory
        
        # Initialize inventory based on configuration
        self.inventory = self._initialize_inventory(use_universal_inventory)
    
    def _initialize_inventory(self, use_universal_inventory: Optional[bool]) -> Optional[object]:
        """
        Initialize phoneme inventory based on configuration.
        
        Args:
            use_universal_inventory: Whether to use universal inventory (None = from config)
            
        Returns:
            Phoneme inventory instance or None
        """
        if not self.use_full_inventory:
            return None
        
        # Use universal inventory if enabled (default from config)
        if use_universal_inventory is None:
            use_universal_inventory = PHONEME_USE_UNIVERSAL_INVENTORY
        
        if use_universal_inventory:
            # Import here to avoid circular dependencies
            from core.universal_phoneme_inventory import UniversalPhonemeInventory
            if PHONEME_LANGUAGE_FAMILIES is not None:
                return UniversalPhonemeInventory(enabled_families=set(PHONEME_LANGUAGE_FAMILIES))
            else:
                return UNIVERSAL_PHONEME_INVENTORY
        else:
            return PHONEME_INVENTORY 
    
    def analyze(self, s_score, pitch):
        """
        Returns a phoneme only if the sound has Structural Intent.
        
        Phoneme Map:
        - Below threshold: Ignored (Noise)
        - Threshold-3.0: Vowels (Low Tension)
        - 3.0-6.0: Fricatives (Flowing tension)
        - 6.0+: Plosives (High Tension)
        
        Args:
            s_score: surprise score
            pitch: current audio pitch
            
        Returns:
            phoneme string or None if below speech threshold
        """
        if not self._is_speech(s_score):
            return None
        
        phoneme = self._map_tension_to_phoneme(s_score, pitch)
        self._store_phoneme(phoneme, s_score, pitch)
        return phoneme
    
    def _is_speech(self, s_score):
        """Check if s_score indicates speech (not noise)."""
        return s_score >= self.speech_threshold
    
    def _map_tension_to_phoneme(self, s_score, pitch):
        """Map tension level (s_score) to phoneme using full IPA inventory."""
        if not self.use_full_inventory or self.inventory is None:
            # Fallback to simple mapping for backward compatibility
            return self._map_tension_to_phoneme_simple(s_score, pitch)
        
        # Use full inventory with intelligent selection
        if s_score < PHONEME_VOWEL_BOUNDARY:
            return self._get_vowel_phoneme_advanced(s_score, pitch)
        elif s_score < PHONEME_FRICATIVE_BOUNDARY:
            return self._get_fricative_phoneme_advanced(s_score, pitch)
        else:
            return self._get_plosive_phoneme_advanced(s_score, pitch)
    
    def _map_tension_to_phoneme_simple(self, s_score, pitch):
        """Simple mapping for backward compatibility."""
        method_map = self._build_phoneme_method_map()
        for boundary, method in method_map:
            if s_score < boundary:
                return method(s_score, pitch)
        return self._get_plosive_phoneme_simple(s_score, pitch)
    
    def _build_phoneme_method_map(self):
        """Build method dispatch map for phoneme selection (legacy)."""
        return [
            (PHONEME_VOWEL_BOUNDARY, self._get_vowel_phoneme_simple),
            (PHONEME_FRICATIVE_BOUNDARY, self._get_fricative_phoneme_simple)
        ]
    
    def _get_vowel_phoneme_advanced(self, s_score, pitch):
        """Get vowel phoneme from full inventory based on pitch and s_score."""
        vowels = self.inventory.get_vowels()
        
        # Map pitch to vowel height (high pitch = high vowels, low pitch = low vowels)
        if pitch > 400:  # Very high pitch
            candidates = [v for v in vowels if v.symbol in ["/i/", "/ɪ/", "/e/"]]
        elif pitch > 300:  # High pitch
            candidates = [v for v in vowels if v.symbol in ["/e/", "/ɛ/", "/æ/"]]
        elif pitch > 200:  # Mid pitch
            candidates = [v for v in vowels if v.symbol in ["/ɛ/", "/æ/", "/ʌ/", "/ə/"]]
        elif pitch > 150:  # Low-mid pitch
            candidates = [v for v in vowels if v.symbol in ["/ɑ/", "/ɔ/", "/o/", "/ʌ/"]]
        else:  # Very low pitch
            candidates = [v for v in vowels if v.symbol in ["/ɔ/", "/o/", "/ʊ/", "/u/"]]
        
        if not candidates:
            candidates = vowels
        
        # Select based on s_score (higher s_score = more open/back vowels)
        index = min(int(s_score * len(candidates) / PHONEME_VOWEL_BOUNDARY), len(candidates) - 1)
        selected = candidates[index]
        return selected.symbol
    
    def _get_fricative_phoneme_advanced(self, s_score, pitch):
        """Get fricative phoneme from full inventory."""
        fricatives = self.inventory.get_by_articulation("fricative")
        
        if not fricatives:
            return "/s/"
        
        # Map s_score to fricative type (higher = more intense)
        # Common fricatives: /s/, /z/, /f/, /v/, /θ/, /ð/, /ʃ/, /ʒ/, /h/
        if s_score < 4.0:
            candidates = [f for f in fricatives if f.symbol in ["/f/", "/v/", "/h/"]]
        elif s_score < 5.0:
            candidates = [f for f in fricatives if f.symbol in ["/θ/", "/ð/", "/s/", "/z/"]]
        else:
            candidates = [f for f in fricatives if f.symbol in ["/ʃ/", "/ʒ/"]]
        
        if not candidates:
            candidates = fricatives
        
        # Select based on voicing (pitch correlates with voicing)
        if pitch > 250:
            voiced = [f for f in candidates if f.voiced]
            if voiced:
                candidates = voiced
        
        index = min(int((s_score - PHONEME_VOWEL_BOUNDARY) * len(candidates) / 
                       (PHONEME_FRICATIVE_BOUNDARY - PHONEME_VOWEL_BOUNDARY)), len(candidates) - 1)
        selected = candidates[index]
        return selected.symbol
    
    def _get_plosive_phoneme_advanced(self, s_score, pitch):
        """Get plosive phoneme from full inventory."""
        plosives = self.inventory.get_by_articulation("plosive")
        
        if not plosives:
            return "/k/"
        
        # Map s_score to plosive type
        # Common plosives: /p/, /b/, /t/, /d/, /k/, /g/
        if s_score < 7.0:
            candidates = [p for p in plosives if p.symbol in ["/p/", "/b/", "/t/", "/d/"]]
        else:
            candidates = [p for p in plosives if p.symbol in ["/k/", "/g/"]]
        
        if not candidates:
            candidates = plosives
        
        # Select based on voicing
        if pitch > 250:
            voiced = [p for p in candidates if p.voiced]
            if voiced:
                candidates = voiced
        
        index = min(int((s_score - PHONEME_FRICATIVE_BOUNDARY) * len(candidates) / 5.0), len(candidates) - 1)
        selected = candidates[index]
        return selected.symbol
    
    def _get_vowel_phoneme_simple(self, s_score, pitch):
        """Get vowel phoneme (legacy simple version)."""
        return "a" if pitch > PHONEME_PITCH_THRESHOLD else "o"
    
    def _get_fricative_phoneme_simple(self, s_score, pitch):
        """Get fricative phoneme (legacy simple version)."""
        return "s"
    
    def _get_plosive_phoneme_simple(self, s_score, pitch):
        """Get plosive phoneme (legacy simple version)."""
        return "k"
    
    def _store_phoneme(self, phoneme, s_score, pitch):
        """Store phoneme in history with memory cap."""
        self.phoneme_history.append({
            'phoneme': phoneme,
            's_score': s_score,
            'pitch': pitch
        })
        if len(self.phoneme_history) > self.max_history:
            self.phoneme_history = self.phoneme_history[-self.max_history:]
    
    def get_recent_phonemes(self, count=10):
        """
        Returns recent phonemes for analysis.
        
        Args:
            count: number of recent phonemes to return
            
        Returns:
            list of phoneme strings
        """
        recent = self.phoneme_history[-count:]
        return [p['phoneme'] for p in recent if p['phoneme'] is not None]
    
    def get_phoneme_statistics(self):
        """
        Returns statistics about learned phonemes.
        
        Returns:
            dict with counts and distribution
        """
        phonemes = self._extract_valid_phonemes()
        
        if not phonemes:
            return self._create_empty_phoneme_statistics()
        
        return self._calculate_phoneme_statistics(phonemes)
    
    def _extract_valid_phonemes(self):
        """Extract valid phonemes from history."""
        return [p['phoneme'] for p in self.phoneme_history if p['phoneme'] is not None]
    
    def _create_empty_phoneme_statistics(self):
        """Create empty phoneme statistics."""
        return {
            'total': 0,
            'vowels': 0,
            'fricatives': 0,
            'plosives': 0,
            'distribution': {
                'vowels': 0,
                'fricatives': 0,
                'plosives': 0
            }
        }
    
    def _calculate_phoneme_statistics(self, phonemes):
        """Calculate statistics from phoneme list."""
        vowels = self._count_vowels(phonemes)
        fricatives = self._count_fricatives(phonemes)
        plosives = self._count_plosives(phonemes)
        distribution = self._calculate_distribution(phonemes, vowels, fricatives, plosives)
        
        return {
            'total': len(phonemes),
            'vowels': vowels,
            'fricatives': fricatives,
            'plosives': plosives,
            'distribution': distribution
        }
    
    def _count_vowels(self, phonemes):
        """Count vowel phonemes."""
        if self.use_full_inventory and self.inventory:
            vowel_symbols = {v.symbol for v in self.inventory.get_vowels()}
            return sum(1 for p in phonemes if p in vowel_symbols)
        else:
            # Legacy: simple vowel set
            vowel_set = {'a', 'o', 'e', 'i', 'u'}
            return sum(1 for p in phonemes if p in vowel_set)
    
    def _count_fricatives(self, phonemes):
        """Count fricative phonemes."""
        if self.use_full_inventory and self.inventory:
            fricative_symbols = {f.symbol for f in self.inventory.get_by_articulation("fricative")}
            return sum(1 for p in phonemes if p in fricative_symbols)
        else:
            # Legacy: simple fricative set
            fricative_set = {'s', 'f', 'h'}
            return sum(1 for p in phonemes if p in fricative_set)
    
    def _count_plosives(self, phonemes):
        """Count plosive phonemes."""
        if self.use_full_inventory and self.inventory:
            plosive_symbols = {p.symbol for p in self.inventory.get_by_articulation("plosive")}
            return sum(1 for p in phonemes if p in plosive_symbols)
        else:
            # Legacy: simple plosive set
            plosive_set = {'k', 'p', 't', 'b', 'd', 'g'}
            return sum(1 for p in phonemes if p in plosive_set)
    
    def _calculate_distribution(self, phonemes, vowels, fricatives, plosives):
        """Calculate phoneme distribution percentages."""
        total = len(phonemes)
        if total == 0:
            return {'vowels': 0, 'fricatives': 0, 'plosives': 0}
        return {
            'vowels': vowels / total,
            'fricatives': fricatives / total,
            'plosives': plosives / total
        }
