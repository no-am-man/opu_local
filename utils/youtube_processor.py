"""
YouTube OPU Processor: Encapsulates the processing logic for YouTube streams.
"""

import time
import numpy as np
from typing import Dict, Any, Optional, Tuple
from config import (
    CHUNK_SIZE, AUDIO_SENSE_YOUTUBE, VIDEO_SENSE_YOUTUBE,
    YOUTUBE_VIDEO_RESIZE_DIM, VISUAL_SURPRISE_THRESHOLD, AUDIO_TONE_DURATION_SECONDS
)
from core.opu import OrthogonalProcessingUnit
from core.mic import perceive
from core.camera import VisualPerception
from core.object_detection import ObjectDetector
from core.genesis import GenesisKernel
from core.expression import AestheticFeedbackLoop, PhonemeAnalyzer
from utils.visualization import CognitiveMapVisualizer
from utils.opu_utils import (
    calculate_ethical_veto, extract_emotion_from_detections, get_cycle_timestamp
)
from utils.hud_utils import draw_youtube_hud

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    cv2 = None


class YouTubeOPUProcessor:
    """
    Processes YouTube streams through the OPU pipeline.
    Encapsulates all processing logic for better organization.
    """
    
    def __init__(self, youtube_streamer, cortex, genesis, afl, phoneme_analyzer, 
                 visualizer, visual_perception, object_detector, image_queue=None):
        """
        Initialize the processor.
        
        Args:
            youtube_streamer: YouTubeStreamer instance
            cortex: OrthogonalProcessingUnit instance
            genesis: GenesisKernel instance
            afl: AestheticFeedbackLoop instance
            phoneme_analyzer: PhonemeAnalyzer instance
            visualizer: CognitiveMapVisualizer instance
            visual_perception: VisualPerception instance
            object_detector: ObjectDetector instance
            image_queue: multiprocessing.Queue for sending cognitive map images to State Viewer
        """
        self.yt = youtube_streamer
        self.cortex = cortex
        self.genesis = genesis
        self.afl = afl
        self.phoneme_analyzer = phoneme_analyzer
        self.visualizer = visualizer
        self.visual_perception = visual_perception
        self.object_detector = object_detector
        self.image_queue = image_queue  # Queue for State Viewer
        
        # State
        self.safe_score = 0.0
        self.frame_count = 0
        self.start_time = time.time()
    
    def process_audio_channel(self) -> Tuple[float, Dict[str, float]]:
        """
        Process audio channel (AUDIO_V2).
        
        Returns:
            Tuple of (s_audio, audio_perception dict)
        """
        audio_chunk = self.yt.get_audio_chunk()
        if len(audio_chunk) == CHUNK_SIZE:
            audio_perception = perceive(audio_chunk)
            s_audio = self.cortex.introspect(audio_perception['genomic_bit'])
        else:
            s_audio = 0.0
            audio_perception = {'genomic_bit': 0.0}
        
        return s_audio, audio_perception
    
    def process_video_channel(self, s_audio: float, audio_perception: Dict[str, float]) -> Dict[str, Any]:
        """
        Process video channel (VIDEO_V2) with object detection and HUD.
        
        Args:
            s_audio: Audio surprise score
            audio_perception: Audio perception dict
            
        Returns:
            Dict with processing results
        """
        frame = self.yt.get_video_frame()
        if frame is None:
            return self._create_empty_video_result()
        
        # Resize for performance
        frame = cv2.resize(frame, YOUTUBE_VIDEO_RESIZE_DIM) if CV2_AVAILABLE else frame
        
        # Object detection
        detections = self.object_detector.detect_objects(frame)
        processed_frame = self.object_detector.draw_detections(frame.copy(), detections)
        
        # Calculate baseline visual surprise (without HUD)
        visual_vector_baseline = self.visual_perception.analyze_frame(processed_frame)
        s_visual_baseline, _ = self.cortex.introspect_visual(visual_vector_baseline)
        
        # Calculate safe_score for HUD
        fused_score = max(s_audio, s_visual_baseline)
        safe_score_current = calculate_ethical_veto(
            self.genesis, fused_score, audio_perception['genomic_bit']
        )
        
        # Add HUD overlay
        fps = self.frame_count / max(time.time() - self.start_time, 0.1)
        draw_youtube_hud(
            processed_frame,
            safe_score_current,
            s_audio,
            s_visual_baseline,
            self.yt.video_info['title'],
            self.frame_count,
            fps
        )
        
        # Analyze fully annotated frame (with HUD) for recursive perception
        visual_vector_annotated = self.visual_perception.analyze_frame(processed_frame)
        s_visual, channel_scores = self.cortex.introspect_visual(visual_vector_annotated)
        
        # Extract emotion
        emotion = extract_emotion_from_detections(detections)
        
        # Update state
        self.safe_score = safe_score_current
        
        return {
            's_visual': s_visual,
            's_visual_baseline': s_visual_baseline,
            'visual_vector_annotated': visual_vector_annotated,
            'processed_frame': processed_frame,
            'detections': detections,
            'emotion': emotion,
            'channel_scores': channel_scores,
            'safe_score_current': safe_score_current
        }
    
    def _create_empty_video_result(self) -> Dict[str, Any]:
        """Create empty video result when no frame is available."""
        return {
            's_visual': 0.0,
            's_visual_baseline': 0.0,
            'visual_vector_annotated': np.array([0.0, 0.0, 0.0]),
            'processed_frame': None,
            'detections': [],
            'emotion': None,
            'channel_scores': {},
            'safe_score_current': self.safe_score
        }
    
    def process_fusion_and_veto(self, s_audio: float, s_visual: float, 
                                audio_perception: Dict[str, float]) -> float:
        """
        Fuse scores and apply ethical veto.
        
        Args:
            s_audio: Audio surprise score
            s_visual: Visual surprise score
            audio_perception: Audio perception dict
            
        Returns:
            Safe score after ethical veto
        """
        fused_score = max(s_audio, s_visual)
        return calculate_ethical_veto(self.genesis, fused_score, audio_perception['genomic_bit'])
    
    def store_memories(self, audio_perception: Dict[str, float], safe_score: float,
                      video_result: Dict[str, Any], cycle_timestamp: float):
        """
        Store memories for both audio and video channels.
        
        Args:
            audio_perception: Audio perception dict
            safe_score: Safe score after ethical veto
            video_result: Video processing result dict
            cycle_timestamp: Synchronized timestamp
        """
        # Store audio memory (AUDIO_V2)
        self.cortex.store_memory(
            audio_perception['genomic_bit'],
            safe_score,
            sense_label=AUDIO_SENSE_YOUTUBE,
            timestamp=cycle_timestamp
        )
        
        # Store visual memory if surprise threshold met (VIDEO_V2)
        if video_result['s_visual'] > VISUAL_SURPRISE_THRESHOLD:
            v_bit = max(video_result['visual_vector_annotated']) if len(video_result['visual_vector_annotated']) > 0 else 0
            emotion = video_result.get('emotion')
            self.cortex.store_memory(
                v_bit,
                video_result['s_visual'],
                sense_label=VIDEO_SENSE_YOUTUBE,
                emotion=emotion,
                timestamp=cycle_timestamp
            )
            
            # Log memory storage for debugging
            emotion_str = f" | Emotion: {emotion['emotion']} ({emotion['confidence']:.2f})" if emotion else ""
            print(f"[YOUTUBE] Stored VIDEO_V2 memory: s_visual={video_result['s_visual']:.4f}{emotion_str}")
    
    def update_expression(self, safe_score: float):
        """
        Update expression (audio feedback and phonemes).
        
        Args:
            safe_score: Safe score for expression
        """
        char = self.cortex.get_character_state()
        self.afl.update_pitch(char['base_pitch'])
        try:
            self.afl.play_tone(safe_score, duration=AUDIO_TONE_DURATION_SECONDS)
        except:
            pass
        
        # Phoneme analysis
        phoneme = self.phoneme_analyzer.analyze(safe_score, self.afl.current_frequency)
        if phoneme:
            print(f"[PHONEME] {phoneme} (s_score: {safe_score:.2f})")
    
    def update_visualization(self, safe_score: float):
        """
        Update cognitive map visualization.
        
        Args:
            safe_score: Safe score for visualization
        """
        state = self.cortex.get_current_state()
        char = self.cortex.get_character_state()
        self.visualizer.update_state(
            safe_score, state['coherence'],
            char['maturity_index'], char.get('maturity_level', 0)
        )
        self.visualizer.draw_cognitive_map()
        viz_image = self.visualizer.render_to_image()
        
        # Send to State Viewer if available (as tuple: ('cognitive_map', image))
        if self.image_queue is not None and viz_image is not None:
            try:
                # Convert BGR to RGB for State Viewer
                if CV2_AVAILABLE:
                    viz_rgb = cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB)
                    # Send as tuple: ('cognitive_map', image) - State Viewer expects this format
                    if not self.image_queue.full():
                        self.image_queue.put_nowait(('cognitive_map', viz_rgb))
            except Exception:
                pass  # Don't block if queue is full or viewer closed
        
        return viz_image
    
    def display_frames(self, processed_frame, viz_image):
        """
        Display processed video frame and cognitive map.
        Also sends video frame to State Viewer if available.
        
        Args:
            processed_frame: Processed video frame with annotations
            viz_image: Cognitive map visualization image
        """
        if not CV2_AVAILABLE:
            return
        
        # Send processed video frame to State Viewer (as 'webcam' for compatibility)
        if self.image_queue is not None and processed_frame is not None:
            try:
                # Convert BGR to RGB for State Viewer
                display_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                # Send as tuple: ('webcam', image) - State Viewer expects this format
                if not self.image_queue.full():
                    self.image_queue.put_nowait(('webcam', display_rgb))
            except Exception:
                pass  # Don't block if queue is full or viewer closed
        
        # Also show in OpenCV windows (optional, for debugging)
        if processed_frame is not None:
            cv2.imshow("OPU YouTube Viewer", processed_frame)
        
        if viz_image is not None:
            viz_rgb = cv2.cvtColor(viz_image, cv2.COLOR_BGR2RGB)
            cv2.imshow("OPU Cognitive Map", viz_rgb)
    
    def process_cycle(self) -> Dict[str, Any]:
        """
        Process one complete cycle of audio and video.
        
        Returns:
            Dict with cycle results
        """
        # Capture timestamp for temporal sync
        cycle_timestamp = get_cycle_timestamp()
        
        # Process audio channel
        s_audio, audio_perception = self.process_audio_channel()
        
        # Process video channel
        video_result = self.process_video_channel(s_audio, audio_perception)
        
        # Fusion and ethical veto
        safe_score = self.process_fusion_and_veto(
            s_audio, video_result['s_visual'], audio_perception
        )
        
        # Store memories
        self.store_memories(audio_perception, safe_score, video_result, cycle_timestamp)
        
        # Update expression
        self.update_expression(safe_score)
        
        # Update visualization
        viz_image = self.update_visualization(safe_score)
        
        # Display frames
        self.display_frames(video_result['processed_frame'], viz_image)
        
        # Increment frame count
        self.frame_count += 1
        
        # Log cycle activity every 100 frames for debugging
        if self.frame_count % 100 == 0:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / max(elapsed, 0.1)
            char = self.cortex.get_character_state()
            print(f"[YOUTUBE] Cycle {self.frame_count} | FPS: {fps:.1f} | "
                  f"s_audio: {s_audio:.4f} | s_visual: {video_result['s_visual']:.4f} | "
                  f"safe_score: {safe_score:.4f} | Maturity: {char['maturity_index']:.3f} | "
                  f"Memories: L0={len(self.cortex.memory_levels[0])} L1={len(self.cortex.memory_levels[1])}")
        
        return {
            's_audio': s_audio,
            's_visual': video_result['s_visual'],
            'safe_score': safe_score,
            'frame_count': self.frame_count
        }

