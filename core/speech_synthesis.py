"""
Speech Synthesis: Text-to-Speech integration for word-level synthesis.
Uses pyttsx3 for TTS and formant synthesis for phoneme-level generation.
"""

import numpy as np
from typing import Optional, List
import threading
import queue

from core.formant_synthesizer import FORMANT_SYNTHESIZER
from core.phoneme_inventory import PHONEME_INVENTORY
from core.language_utils import (
    TTS_AVAILABLE, pyttsx3, check_dependency, safe_initialize,
    convert_audio_bytes_to_array, resample_audio
)
from config import SAMPLE_RATE


class SpeechSynthesizer:
    """
    Hybrid speech synthesizer: formant synthesis for phonemes, TTS for words.
    Integrates with AestheticFeedbackLoop for real-time output.
    """
    
    def __init__(self, use_tts: bool = True):
        self.use_tts = use_tts and check_dependency("pyttsx3", TTS_AVAILABLE)
        self.formant_synthesizer = FORMANT_SYNTHESIZER
        self.inventory = PHONEME_INVENTORY
        self.sample_rate = SAMPLE_RATE
        
        # TTS engine (if available)
        self.tts_engine = None
        if self.use_tts:
            self.tts_engine = safe_initialize(
                self._init_tts_engine,
                fallback_value=None,
                error_message="TTS initialization failed"
            )
            if not self.tts_engine:
                self.use_tts = False
    
    def _init_tts_engine(self):
        """Initialize TTS engine with configuration."""
        engine = pyttsx3.init()
        
        # Configure TTS voice
        voices = engine.getProperty('voices')
        if voices:
            # Prefer female voice if available
            for voice in voices:
                if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            else:
                # Use first available voice
                engine.setProperty('voice', voices[0].id)
        
        # Set speech rate (words per minute)
        engine.setProperty('rate', 150)
        # Set volume (0.0 to 1.0)
        engine.setProperty('volume', 0.8)
        
        return engine
    
    def synthesize_phoneme_sequence(self, phoneme_symbols: List[str]) -> np.ndarray:
        """
        Synthesize a sequence of phonemes using formant synthesis.
        
        Args:
            phoneme_symbols: List of IPA phoneme symbols (e.g., ["/h/", "/ɛ/", "/l/", "/oʊ/"])
            
        Returns:
            numpy array of concatenated audio samples
        """
        audio_segments = []
        
        for symbol in phoneme_symbols:
            phoneme_def = self.inventory.get_phoneme(symbol)
            if phoneme_def:
                audio = self.formant_synthesizer.synthesize_phoneme(phoneme_def)
                audio_segments.append(audio)
            else:
                # Unknown phoneme: generate silence
                silence = np.zeros(int(self.sample_rate * 0.05))  # 50ms silence
                audio_segments.append(silence)
        
        if audio_segments:
            return np.concatenate(audio_segments)
        else:
            return np.array([])
    
    def synthesize_word(self, word: str, use_tts: Optional[bool] = None) -> Optional[np.ndarray]:
        """
        Synthesize a word using TTS (if available) or formant synthesis.
        
        Args:
            word: Word to synthesize (e.g., "hello")
            use_tts: Override default TTS setting
            
        Returns:
            numpy array of audio samples, or None if synthesis fails
        """
        use_tts_final = use_tts if use_tts is not None else self.use_tts
        
        if use_tts_final and self.tts_engine:
            return self._synthesize_word_tts(word)
        else:
            # Fallback: convert word to phonemes and use formant synthesis
            return self._synthesize_word_formant(word)
    
    def _synthesize_word_tts(self, word: str) -> Optional[np.ndarray]:
        """Synthesize word using TTS engine."""
        if not self.tts_engine:
            return None
        
        try:
            import tempfile
            import os
            import wave
            
            temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_path = temp_file.name
            temp_file.close()
            
            # Use TTS to generate audio file
            self.tts_engine.save_to_file(word, temp_path)
            self.tts_engine.runAndWait()
            
            # Read audio file
            try:
                with wave.open(temp_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    sample_rate = wav_file.getframerate()
                    audio_bytes = wav_file.readframes(frames)
                    sample_width = wav_file.getsampwidth()
                    
                    # Convert to numpy array
                    audio = convert_audio_bytes_to_array(audio_bytes, sample_width)
                    
                    # Resample if needed
                    if sample_rate != self.sample_rate:
                        audio = resample_audio(audio, sample_rate, self.sample_rate)
                    
                    # Clean up
                    os.unlink(temp_path)
                    return audio
            except Exception as e:
                print(f"[SPEECH] Error reading TTS audio: {e}")
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                return None
                
        except Exception as e:
            print(f"[SPEECH] TTS synthesis failed: {e}")
            return None
    
    def _synthesize_word_formant(self, word: str) -> Optional[np.ndarray]:
        """Synthesize word using formant synthesis (fallback)."""
        # Simple phoneme mapping (would need proper grapheme-to-phoneme conversion)
        # For now, just return None - this would require a pronunciation dictionary
        return None
    
    def speak_text(self, text: str, async_mode: bool = False):
        """
        Speak text using TTS engine.
        
        Args:
            text: Text to speak
            async_mode: If True, speak in background thread
        """
        if not self.use_tts or not self.tts_engine:
            print(f"[SPEECH] TTS not available, would speak: {text}")
            return
        
        if async_mode:
            thread = threading.Thread(target=self._speak_sync, args=(text,), daemon=True)
            thread.start()
        else:
            self._speak_sync(text)
    
    def _speak_sync(self, text: str):
        """Synchronous speech (internal)."""
        try:
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"[SPEECH] Error speaking: {e}")
    
    def cleanup(self):
        """Cleanup TTS engine."""
        if self.tts_engine:
            try:
                self.tts_engine.stop()
            except:
                pass
            self.tts_engine = None

