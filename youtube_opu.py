"""
YouTube OPU: Feed YouTube video/audio streams into the OPU.

This script extracts video and audio from YouTube streams and feeds them
directly into the OPU's processing pipeline, allowing the OPU to "watch"
and "listen" to YouTube content.

Requirements:
    pip install yt-dlp opencv-python numpy sounddevice

    Also requires ffmpeg installed on system:
        macOS: brew install ffmpeg
        Linux: apt-get install ffmpeg
        Windows: Download from https://ffmpeg.org/
"""

import sys
import os
import time
import numpy as np
import threading
import queue
import argparse
import subprocess
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Optional imports
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[YOUTUBE] Warning: opencv-python not installed. Video disabled.")

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    print("[YOUTUBE] Error: yt-dlp not installed. Install with: pip install yt-dlp")

try:
    import subprocess
    # Check if ffmpeg is actually available in PATH
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      timeout=2)
        FFMPEG_AVAILABLE = True
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        FFMPEG_AVAILABLE = False
except ImportError:
    FFMPEG_AVAILABLE = False

from config import (
    SAMPLE_RATE, CHUNK_SIZE, AUDIO_SENSE_YOUTUBE, VIDEO_SENSE_YOUTUBE, OPU_VERSION,
    YOUTUBE_VIDEO_RESIZE_DIM, YOUTUBE_AUDIO_VOLUME_MULTIPLIER, VISUAL_SURPRISE_THRESHOLD,
    BASE_FREQUENCY
)
from core.opu import OrthogonalProcessingUnit
from core.mic import perceive
from core.camera import VisualPerception
from core.object_detection import ObjectDetector
from core.genesis import GenesisKernel
from core.expression import AestheticFeedbackLoop, PhonemeAnalyzer
from utils.visualization import CognitiveMapVisualizer
from utils.opu_utils import (
    setup_file_logging, cleanup_file_logging
)
from utils.youtube_processor import YouTubeOPUProcessor


class YouTubeStreamer:
    """
    Extracts and streams video/audio from YouTube URLs.
    Uses yt-dlp to get stream URLs and ffmpeg to decode them.
    """
    
    def __init__(self, youtube_url, video_enabled=True, audio_enabled=True):
        """
        Initialize YouTube streamer.
        
        Args:
            youtube_url: YouTube video URL
            video_enabled: Whether to extract video stream
            audio_enabled: Whether to extract audio stream
        """
        if not YT_DLP_AVAILABLE:
            raise ImportError("yt-dlp is required. Install with: pip install yt-dlp")
        
        self.youtube_url = youtube_url
        self.video_enabled = video_enabled and CV2_AVAILABLE
        self.audio_enabled = audio_enabled
        self.running = False
        
        # Stream URLs (will be extracted)
        self.video_url = None
        self.audio_url = None
        self.video_info = None
        
        # Video capture
        self.video_cap = None
        
        # Audio streaming
        self.audio_queue = queue.Queue(maxsize=10)
        self.audio_process = None
        self.audio_thread = None
        
        # Extract stream info
        self._extract_stream_info()
        
        # Setup streams
        if self.video_enabled:
            self._setup_video_stream()
        if self.audio_enabled:
            self._setup_audio_stream()
    
    def _extract_stream_info(self):
        """Extract stream URLs and video info using yt-dlp."""
        print(f"[YOUTUBE] Extracting stream info from: {self.youtube_url}")
        
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.youtube_url, download=False)
                
                self.video_info = {
                    'title': info.get('title', 'Unknown'),
                    'duration': info.get('duration', 0),
                    'view_count': info.get('view_count', 0),
                }
                
                # Try to get separate video and audio URLs
                formats = info.get('formats', [])
                video_format = None
                audio_format = None
                
                for fmt in formats:
                    if fmt.get('vcodec') != 'none' and video_format is None:
                        video_format = fmt
                    if fmt.get('acodec') != 'none' and audio_format is None:
                        audio_format = fmt
                
                if video_format:
                    self.video_url = video_format.get('url')
                if audio_format:
                    self.audio_url = audio_format.get('url')
                
                # Fallback: use best combined format
                if not self.video_url and not self.audio_url:
                    self.video_url = info.get('url')
                    self.audio_url = info.get('url')
                
                print(f"[YOUTUBE] Stream found: {self.video_info['title']}")
                print(f"[YOUTUBE] Duration: {self.video_info['duration']}s")
                
        except Exception as e:
            print(f"[YOUTUBE] Error extracting stream info: {e}")
            raise
    
    def _setup_video_stream(self):
        """Setup video stream using OpenCV."""
        if not self.video_url:
            print("[YOUTUBE] No video URL available, video disabled")
            self.video_enabled = False
            return
        
        try:
            self.video_cap = cv2.VideoCapture(self.video_url)
            if not self.video_cap.isOpened():
                print("[YOUTUBE] Failed to open video stream")
                self.video_enabled = False
            else:
                print("[YOUTUBE] Video stream ready")
        except Exception as e:
            print(f"[YOUTUBE] Error setting up video: {e}")
            self.video_enabled = False
    
    def _setup_audio_stream(self):
        """Setup audio stream using ffmpeg subprocess."""
        if not self.audio_url:
            print("[YOUTUBE] No audio URL available, audio disabled")
            self.audio_enabled = False
            return
        
        if not FFMPEG_AVAILABLE:
            print("[YOUTUBE] ffmpeg not available, audio disabled")
            print("[YOUTUBE] Install ffmpeg: brew install ffmpeg (macOS) or apt-get install ffmpeg (Linux)")
            self.audio_enabled = False
            return
        
        # Start audio streaming thread
        self.running = True
        self.audio_thread = threading.Thread(target=self._stream_audio, daemon=True)
        self.audio_thread.start()
        print("[YOUTUBE] Audio stream ready")
    
    def _stream_audio(self):
        """Stream audio from YouTube using ffmpeg."""
        try:
            # Use ffmpeg to extract audio as raw PCM
            cmd = [
                'ffmpeg',
                '-i', self.audio_url,
                '-f', 'f32le',  # 32-bit float little-endian
                '-acodec', 'pcm_f32le',
                '-ac', '1',  # Mono
                '-ar', str(SAMPLE_RATE),  # Sample rate
                '-loglevel', 'quiet',  # Suppress ffmpeg output
                'pipe:1'  # Output to stdout
            ]
            
            self.audio_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=CHUNK_SIZE * 4  # 4 bytes per float32
            )
            
            bytes_per_chunk = CHUNK_SIZE * 4  # 4 bytes per float32
            
            while self.running:
                try:
                    audio_bytes = self.audio_process.stdout.read(bytes_per_chunk)
                    if not audio_bytes:
                        break
                    
                    if len(audio_bytes) < bytes_per_chunk:
                        # Pad with zeros if incomplete
                        audio_bytes += b'\x00' * (bytes_per_chunk - len(audio_bytes))
                    
                    # Convert bytes to numpy array
                    audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
                    
                    # Normalize volume (YouTube can be loud)
                    audio_data = audio_data * YOUTUBE_AUDIO_VOLUME_MULTIPLIER
                    
                    # Put in queue (non-blocking, drop if full)
                    try:
                        self.audio_queue.put_nowait(audio_data)
                    except queue.Full:
                        # Drop frame if queue is full (maintain real-time)
                        pass
                        
                except Exception as e:
                    if self.running:
                        print(f"[YOUTUBE] Audio streaming error: {e}")
                    break
                    
        except Exception as e:
            print(f"[YOUTUBE] Error starting audio stream: {e}")
            self.audio_enabled = False
    
    def get_video_frame(self):
        """
        Get next video frame from YouTube stream.
        
        Returns:
            numpy array (BGR format) or None if no frame available
        """
        if not self.video_enabled or self.video_cap is None:
            return None
        
        ret, frame = self.video_cap.read()
        if ret:
            return frame
        return None
    
    def get_audio_chunk(self):
        """
        Get next audio chunk from YouTube stream.
        
        Returns:
            numpy array of shape (CHUNK_SIZE,) or zeros if no audio available
        """
        if not self.audio_enabled:
            return np.zeros(CHUNK_SIZE, dtype=np.float32)
        
        try:
            return self.audio_queue.get_nowait()
        except queue.Empty:
            # Return silence if buffering
            return np.zeros(CHUNK_SIZE, dtype=np.float32)
    
    def close(self):
        """Close all streams and cleanup."""
        self.running = False
        
        if self.video_cap is not None:
            self.video_cap.release()
        
        if self.audio_process is not None:
            self.audio_process.terminate()
            self.audio_process.wait(timeout=2)
        
        if self.audio_thread is not None:
            self.audio_thread.join(timeout=1)


def run_youtube_opu(youtube_url, enable_state_viewer=True, log_file=None):
    """
    Run OPU with YouTube stream as input.
    
    Args:
        youtube_url: YouTube video URL
        enable_state_viewer: Whether to enable the state viewer GUI
        log_file: Path to log file (if None, uses default "youtube_opu.log")
    """
    # Setup file logging using utility function
    file_logger, original_stdout, original_stderr = setup_file_logging(
        log_file, default_name="youtube_opu.log"
    )
    
    print("=" * 70)
    print("OPU YouTube Mode")
    print("=" * 70)
    print(f"Version: {OPU_VERSION}")
    print(f"URL: {youtube_url}")
    print(f"Logging active: {file_logger.log_file_path if file_logger else 'None'}")
    print()
    
    # Initialize YouTube streamer
    try:
        yt = YouTubeStreamer(youtube_url)
    except Exception as e:
        print(f"[ERROR] Failed to initialize YouTube streamer: {e}")
        cleanup_file_logging(file_logger, original_stdout, original_stderr)
        return
    
    # Initialize OPU components
    print("[OPU] Initializing...")
    cortex = OrthogonalProcessingUnit()
    genesis = GenesisKernel()
    afl = AestheticFeedbackLoop(base_pitch=BASE_FREQUENCY)
    phoneme_analyzer = PhonemeAnalyzer()
    visualizer = CognitiveMapVisualizer()
    visual_perception = VisualPerception(camera_index=0, use_color_constancy=True)
    object_detector = ObjectDetector(use_dnn=False, confidence_threshold=0.5)
    
    # Create processor to encapsulate processing logic
    processor = YouTubeOPUProcessor(
        yt, cortex, genesis, afl, phoneme_analyzer,
        visualizer, visual_perception, object_detector
    )
    
    print("[OPU] Ready. Processing YouTube stream...")
    print("[OPU] Press 'q' to quit, 'p' to pause")
    print()
    
    paused = False
    
    try:
        while True:
            if paused:
                time.sleep(0.1)
                continue
            
            # Process one cycle (handles all 4-channel temporal sync internally)
            result = processor.process_cycle()
            
            # Keyboard input
            if CV2_AVAILABLE:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"[OPU] {'Paused' if paused else 'Resumed'}")
            
            # Status update is now handled in YouTubeOPUProcessor.process_cycle()
            # This keeps logging consistent and centralized
            
            # Sync to ~30 FPS
            time.sleep(1.0 / 30.0)
    
    except KeyboardInterrupt:
        print("\n[OPU] Interrupted by user")
    except Exception as e:
        print(f"\n[OPU] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[OPU] Cleaning up...")
        yt.close()
        afl.cleanup()
        if CV2_AVAILABLE:
            cv2.destroyAllWindows()
        
        # Restore stdout/stderr and close log file
        cleanup_file_logging(file_logger, original_stdout, original_stderr)
        
        print("[OPU] Done.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='OPU YouTube Mode - Feed YouTube streams into the OPU',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python youtube_opu.py "https://www.youtube.com/watch?v=jfKfPfyJRdk"
  python youtube_opu.py "https://youtu.be/jfKfPfyJRdk" --no-state-viewer
        """
    )
    parser.add_argument(
        'url',
        type=str,
        nargs='?',
        default=None,
        help='YouTube video URL (optional, can use default)'
    )
    parser.add_argument(
        '--no-state-viewer',
        action='store_true',
        help='Disable state viewer GUI'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='Path to log file (enables file logging). Default: youtube_opu.log'
    )
    
    args = parser.parse_args()
    
    if not YT_DLP_AVAILABLE:
        print("[ERROR] yt-dlp is required. Install with: pip install yt-dlp")
        sys.exit(1)
    
    if not CV2_AVAILABLE:
        print("[ERROR] opencv-python is required. Install with: pip install opencv-python")
        sys.exit(1)
    
    # Use default URL if not provided
    url = args.url if args.url else "https://www.youtube.com/watch?v=jfKfPfyJRdk"
    run_youtube_opu(url, enable_state_viewer=not args.no_state_viewer, log_file=args.log_file)


if __name__ == "__main__":
    main()

