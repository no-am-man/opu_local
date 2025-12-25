"""
Language System: Integration module for full English language capabilities.
Combines phoneme analysis, speech synthesis, recognition, and language memory.
"""

from typing import Optional, Callable
from core.expression import PhonemeAnalyzer
from core.speech_synthesis import SpeechSynthesizer
from core.speech_recognition import SpeechRecognizer
from core.language_memory import LanguageMemory
from config import (
    PHONEME_USE_FULL_INVENTORY, SPEECH_USE_TTS, SPEECH_RECOGNITION_ENABLED,
    SPEECH_USE_WHISPER, LANGUAGE_MEMORY_ENABLED
)


class LanguageSystem:
    """
    Complete language system integrating all language capabilities.
    Provides phoneme analysis, speech synthesis, recognition, and word learning.
    """
    
    def __init__(self, use_full_inventory: bool = None, use_tts: bool = None,
                 enable_recognition: bool = None, use_whisper: bool = None,
                 enable_memory: bool = None):
        """
        Initialize language system with all components.
        
        Args:
            use_full_inventory: Use full IPA inventory (default: from config)
            use_tts: Enable TTS for word synthesis (default: from config)
            enable_recognition: Enable speech recognition (default: from config)
            use_whisper: Use Whisper for recognition (default: from config)
            enable_memory: Enable language memory (default: from config)
        """
        # Configuration with defaults from config
        config_defaults = {
            'use_full_inventory': PHONEME_USE_FULL_INVENTORY,
            'use_tts': SPEECH_USE_TTS,
            'enable_recognition': SPEECH_RECOGNITION_ENABLED,
            'use_whisper': SPEECH_USE_WHISPER,
            'enable_memory': LANGUAGE_MEMORY_ENABLED
        }
        
        self.use_full_inventory = use_full_inventory if use_full_inventory is not None else config_defaults['use_full_inventory']
        self.use_tts = use_tts if use_tts is not None else config_defaults['use_tts']
        self.enable_recognition = enable_recognition if enable_recognition is not None else config_defaults['enable_recognition']
        self.use_whisper = use_whisper if use_whisper is not None else config_defaults['use_whisper']
        self.enable_memory = enable_memory if enable_memory is not None else config_defaults['enable_memory']
        
        # Initialize components
        self.phoneme_analyzer = PhonemeAnalyzer(use_full_inventory=self.use_full_inventory)
        self.speech_synthesizer = SpeechSynthesizer(use_tts=self.use_tts) if self.use_tts else None
        self.speech_recognizer = SpeechRecognizer(use_whisper=self.use_whisper) if self.enable_recognition else None
        self.language_memory = LanguageMemory() if self.enable_memory else None
        
        # Callbacks
        self.on_word_recognized: Optional[Callable[[str], None]] = None
        self.on_word_learned: Optional[Callable[[str], None]] = None
        
        # Setup recognition callback if enabled
        if self.enable_recognition and self.speech_recognizer:
            self.speech_recognizer.on_text_recognized = self._handle_recognized_text
    
    def analyze_phoneme(self, s_score: float, pitch: float) -> Optional[str]:
        """
        Analyze surprise score and pitch to generate phoneme.
        
        Args:
            s_score: Surprise score
            pitch: Current pitch
            
        Returns:
            IPA phoneme symbol or None
        """
        return self.phoneme_analyzer.analyze(s_score, pitch)
    
    def synthesize_phoneme(self, phoneme_symbol: str) -> Optional[any]:
        """
        Synthesize a phoneme using formant synthesis.
        
        Args:
            phoneme_symbol: IPA phoneme symbol
            
        Returns:
            numpy array of audio samples or None
        """
        from core.phoneme_inventory import PHONEME_INVENTORY
        phoneme_def = PHONEME_INVENTORY.get_phoneme(phoneme_symbol)
        if phoneme_def:
            from core.formant_synthesizer import FORMANT_SYNTHESIZER
            return FORMANT_SYNTHESIZER.synthesize_phoneme(phoneme_def)
        return None
    
    def synthesize_word(self, word: str) -> Optional[any]:
        """
        Synthesize a word using TTS or formant synthesis.
        
        Args:
            word: Word to synthesize
            
        Returns:
            numpy array of audio samples or None
        """
        if self.speech_synthesizer:
            return self.speech_synthesizer.synthesize_word(word)
        return None
    
    def speak_text(self, text: str, async_mode: bool = False):
        """
        Speak text using TTS.
        
        Args:
            text: Text to speak
            async_mode: Speak in background thread
        """
        if self.speech_synthesizer:
            self.speech_synthesizer.speak_text(text, async_mode=async_mode)
    
    def recognize_speech(self, audio_data: bytes = None) -> Optional[str]:
        """
        Recognize speech from audio or microphone.
        
        Args:
            audio_data: Audio bytes (if None, uses microphone)
            
        Returns:
            Recognized text or None
        """
        if not self.speech_recognizer:
            return None
        
        if audio_data:
            return self.speech_recognizer.recognize_audio(audio_data)
        else:
            return self.speech_recognizer.recognize_from_microphone()
    
    def start_continuous_recognition(self, callback: Optional[Callable[[str], None]] = None):
        """Start continuous speech recognition."""
        if self.speech_recognizer:
            if callback:
                self.on_word_recognized = callback
            self.speech_recognizer.start_continuous_recognition()
    
    def stop_continuous_recognition(self):
        """Stop continuous speech recognition."""
        if self.speech_recognizer:
            self.speech_recognizer.stop_continuous_recognition()
    
    def learn_word(self, word: str, phonemes: Optional[list] = None, 
                   context: Optional[str] = None, emotion: Optional[str] = None,
                   s_score: float = 0.0):
        """
        Learn a word and store in language memory.
        
        Args:
            word: Word text
            phonemes: IPA phoneme sequence
            context: Context where word was encountered
            emotion: Associated emotion
            s_score: Surprise score
        """
        if self.language_memory:
            entry = self.language_memory.learn_word(word, phonemes, context, emotion, s_score)
            if self.on_word_learned:
                self.on_word_learned(word)
            return entry
        return None
    
    def learn_phrase(self, words: list, s_score: float = 0.0):
        """
        Learn a phrase (sequence of words).
        
        Args:
            words: List of words
            s_score: Surprise score
        """
        if self.language_memory:
            self.language_memory.learn_phrase(words, s_score)
    
    def get_word(self, word: str):
        """Get word entry from language memory."""
        if self.language_memory:
            return self.language_memory.get_word(word)
        return None
    
    def get_statistics(self) -> dict:
        """Get language system statistics."""
        stats = {
            'phoneme_inventory_size': len(self.phoneme_analyzer.inventory.phonemes) if self.phoneme_analyzer.inventory else 0,
            'phoneme_history_size': len(self.phoneme_analyzer.phoneme_history),
            'tts_available': self.speech_synthesizer is not None and self.speech_synthesizer.use_tts,
            'recognition_available': self.speech_recognizer is not None,
            'memory_enabled': self.language_memory is not None
        }
        
        if self.language_memory:
            stats.update(self.language_memory.get_statistics())
        
        return stats
    
    def _handle_recognized_text(self, text: str):
        """Handle recognized text from speech recognition."""
        # Learn words from recognized text
        words = text.lower().split()
        for word in words:
            # Clean word (remove punctuation)
            word_clean = ''.join(c for c in word if c.isalnum())
            if word_clean:
                self.learn_word(word_clean, context=text)
        
        # Learn phrase
        if len(words) > 1:
            self.learn_phrase(words)
        
        # Call user callback
        if self.on_word_recognized:
            self.on_word_recognized(text)
    
    def cleanup(self):
        """Cleanup all language system components."""
        if self.speech_synthesizer:
            self.speech_synthesizer.cleanup()
        if self.speech_recognizer:
            self.speech_recognizer.cleanup()

