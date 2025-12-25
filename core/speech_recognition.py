"""
Speech Recognition: Speech-to-text for comprehension.
Uses speech recognition libraries to convert audio to text.
"""

from typing import Optional, Callable, List
import threading
import queue

from core.language_utils import (
    SPEECH_RECOGNITION_AVAILABLE, sr, WHISPER_AVAILABLE, whisper,
    check_dependency, safe_initialize, convert_audio_bytes_to_array
)
from config import SAMPLE_RATE


class SpeechRecognizer:
    """
    Speech recognition for OPU comprehension.
    Converts audio input to text for language understanding.
    """
    
    def __init__(self, use_whisper: bool = True):
        self.use_whisper = use_whisper and check_dependency("whisper", WHISPER_AVAILABLE)
        self.use_sr = not self.use_whisper and check_dependency("speech_recognition", SPEECH_RECOGNITION_AVAILABLE)
        
        # Whisper model (if available)
        self.whisper_model = None
        if self.use_whisper:
            self.whisper_model = safe_initialize(
                lambda: whisper.load_model("base"),
                fallback_value=None,
                error_message="Whisper initialization failed"
            )
            if self.whisper_model:
                print("[SPEECH] Whisper model loaded")
            else:
                self.use_whisper = False
                self.use_sr = check_dependency("speech_recognition", SPEECH_RECOGNITION_AVAILABLE)
        
        # SpeechRecognition engine (fallback)
        self.recognizer = None
        if self.use_sr:
            self.recognizer = safe_initialize(
                lambda: sr.Recognizer(),
                fallback_value=None,
                error_message="SpeechRecognition initialization failed"
            )
            if self.recognizer:
                print("[SPEECH] SpeechRecognition initialized")
            else:
                self.use_sr = False
        
        # Callback for recognized text
        self.on_text_recognized: Optional[Callable[[str], None]] = None
        
        # Recognition queue for async processing
        self.recognition_queue = queue.Queue()
        self.recognition_thread = None
        self.running = False
    
    def recognize_audio(self, audio_data: bytes, sample_rate: int = SAMPLE_RATE) -> Optional[str]:
        """
        Recognize speech from audio data.
        
        Args:
            audio_data: Raw audio bytes
            sample_rate: Sample rate of audio
            
        Returns:
            Recognized text or None if recognition fails
        """
        if self.use_whisper and self.whisper_model:
            return self._recognize_whisper(audio_data, sample_rate)
        elif self.use_sr and self.recognizer:
            return self._recognize_sr(audio_data, sample_rate)
        else:
            return None
    
    def _recognize_whisper(self, audio_data: bytes, sample_rate: int) -> Optional[str]:
        """Recognize using Whisper."""
        try:
            # Convert bytes to numpy array
            audio_array = convert_audio_bytes_to_array(audio_data, sample_width=2)
            
            # Whisper expects float32 audio
            result = self.whisper_model.transcribe(audio_array, language="en")
            text = result.get("text", "").strip()
            
            if text:
                return text
            return None
        except Exception as e:
            print(f"[SPEECH] Whisper recognition error: {e}")
            return None
    
    def _recognize_sr(self, audio_data: bytes, sample_rate: int) -> Optional[str]:
        """Recognize using SpeechRecognition library."""
        try:
            import io
            
            # Create AudioData object
            audio_source = sr.AudioData(audio_data, sample_rate, 2)  # 2 bytes per sample
            
            # Try Google Speech Recognition first
            try:
                text = self.recognizer.recognize_google(audio_source)
                return text
            except sr.UnknownValueError:
                return None
            except sr.RequestError as e:
                print(f"[SPEECH] Recognition service error: {e}")
                return None
        except Exception as e:
            print(f"[SPEECH] SpeechRecognition error: {e}")
            return None
    
    def recognize_from_microphone(self, timeout: float = 1.0, phrase_time_limit: float = 5.0) -> Optional[str]:
        """
        Recognize speech from microphone input.
        
        Args:
            timeout: Timeout in seconds
            phrase_time_limit: Maximum phrase length in seconds
            
        Returns:
            Recognized text or None
        """
        if not self.use_sr or not self.recognizer:
            return None
        
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                # Listen for audio
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
                
                # Recognize
                try:
                    text = self.recognizer.recognize_google(audio)
                    return text
                except sr.UnknownValueError:
                    return None
                except sr.RequestError as e:
                    print(f"[SPEECH] Recognition service error: {e}")
                    return None
        except Exception as e:
            print(f"[SPEECH] Microphone recognition error: {e}")
            return None
    
    def start_continuous_recognition(self, callback: Optional[Callable[[str], None]] = None):
        """Start continuous recognition in background thread."""
        if callback:
            self.on_text_recognized = callback
        
        self.running = True
        self.recognition_thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self.recognition_thread.start()
        print("[SPEECH] Continuous recognition started")
    
    def stop_continuous_recognition(self):
        """Stop continuous recognition."""
        self.running = False
        if self.recognition_thread:
            self.recognition_thread.join(timeout=2.0)
        print("[SPEECH] Continuous recognition stopped")
    
    def _recognition_loop(self):
        """Background recognition loop."""
        while self.running:
            try:
                text = self.recognize_from_microphone(timeout=1.0)
                if text and self.on_text_recognized:
                    self.on_text_recognized(text)
            except Exception as e:
                if self.running:
                    print(f"[SPEECH] Recognition loop error: {e}")
                break
    
    def cleanup(self):
        """Cleanup recognition resources."""
        self.stop_continuous_recognition()
        self.whisper_model = None
        self.recognizer = None

