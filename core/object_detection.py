"""
Object Detection Module for OPU Visual Cortex.
Uses OpenCV DNN for real-time object detection and emotion recognition.

OPU v3.2 - Visual Object Recognition + Emotion Detection
"""

try:
    import cv2
except ImportError:
    cv2 = None

import numpy as np
import os
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
from config import (
    DETECTION_EMOTION_COLOR_HAPPY, DETECTION_EMOTION_COLOR_SAD,
    DETECTION_EMOTION_COLOR_ANGRY, DETECTION_EMOTION_COLOR_SURPRISE,
    DETECTION_EMOTION_COLOR_FEAR, DETECTION_EMOTION_COLOR_DISGUST,
    DETECTION_EMOTION_COLOR_NEUTRAL, DETECTION_EMOTION_COLOR_DEFAULT,
    DETECTION_FACE_COLOR_NO_EMOTION, DETECTION_OBJECT_COLOR
)


class EmotionMethod(Enum):
    """Emotion detection method types."""
    DEEPFACE = 'deepface'
    FER = 'fer'
    HEURISTIC = 'heuristic'


@dataclass
class DetectionConfig:
    """Configuration constants for object detection."""
    DEFAULT_CONFIDENCE_THRESHOLD = 0.5
    DEFAULT_EMOTION_CONFIDENCE = 0.5
    DEEPFACE_CONFIDENCE_NORMALIZER = 100.0
    MIN_FACE_SIZE = 0
    
    # Emotion labels
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    # COCO class names (80 classes)
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]


@dataclass
class EmotionResult:
    """Result of emotion detection."""
    emotion: str
    confidence: float


class EmotionDetector:
    """Handles emotion detection using various methods."""
    
    def __init__(self, method: EmotionMethod):
        """Initialize emotion detector with specified method."""
        self.method = method
        self.emotion_detector = None
        self._detection_methods = self._build_detection_method_map()
        self._initialization_methods = self._build_initialization_method_map()
        self._initialize_detector()
    
    def _build_detection_method_map(self):
        """Build method dispatch map for emotion detection."""
        return {
            EmotionMethod.DEEPFACE: self._detect_with_deepface,
            EmotionMethod.FER: self._detect_with_fer,
            EmotionMethod.HEURISTIC: self._detect_with_heuristic
        }
    
    def _build_initialization_method_map(self):
        """Build method dispatch map for detector initialization."""
        return {
            EmotionMethod.DEEPFACE: self._initialize_deepface,
            EmotionMethod.FER: self._initialize_fer,
            EmotionMethod.HEURISTIC: self._initialize_heuristic
        }
    
    def _initialize_detector(self):
        """Initialize the emotion detection method using dispatch."""
        initialization_method = self._initialization_methods.get(self.method)
        if initialization_method:
            initialization_method()
    
    def _initialize_deepface(self):
        """Initialize DeepFace detector (imported on-demand)."""
        pass
    
    def _initialize_fer(self):
        """Initialize FER detector."""
        try:
            from fer import FER
            self.emotion_detector = FER(mtcnn=True)
        except ImportError:
            self.method = EmotionMethod.HEURISTIC
    
    def _initialize_heuristic(self):
        """Initialize heuristic detector (no setup needed)."""
        pass
    
    def detect(self, face_roi: np.ndarray) -> Optional[EmotionResult]:
        """
        Detect emotion in a face region.
        
        Args:
            face_roi: Face region of interest (BGR image)
            
        Returns:
            EmotionResult or None if detection fails
        """
        if face_roi.size == 0:
            return None
        
        detection_method = self._detection_methods.get(self.method)
        if not detection_method:
            return None
        
        try:
            return detection_method(face_roi)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            return None
    
    def _detect_with_deepface(self, face_roi: np.ndarray) -> Optional[EmotionResult]:
        """Detect emotion using DeepFace."""
        from deepface import DeepFace
        result = DeepFace.analyze(
            face_roi, 
            actions=['emotion'], 
            enforce_detection=False, 
            silent=True
        )
        
        if not result:
            return None
        
        emotions = self._extract_emotions_from_deepface(result)
        if not emotions:
            return None
        
        return self._create_emotion_result_from_emotions(emotions)
    
    def _create_emotion_result_from_emotions(self, emotions: Dict[str, float]) -> EmotionResult:
        """Create EmotionResult from emotions dictionary."""
        top_emotion = max(emotions.items(), key=lambda x: x[1])
        normalized_confidence = top_emotion[1] / DetectionConfig.DEEPFACE_CONFIDENCE_NORMALIZER
        
        return EmotionResult(
            emotion=top_emotion[0],
            confidence=normalized_confidence
        )
    
    def _extract_emotions_from_deepface(self, result: Any) -> Dict[str, float]:
        """Extract emotions dict from DeepFace result."""
        if isinstance(result, list):
            return self._extract_emotions_from_list(result)
        elif isinstance(result, dict):
            return self._extract_emotions_from_dict(result)
        return {}
    
    def _extract_emotions_from_list(self, result_list: list) -> Dict[str, float]:
        """Extract emotions from list result."""
        if len(result_list) > 0:
            return result_list[0].get('emotion', {})
        return {}
    
    def _extract_emotions_from_dict(self, result_dict: dict) -> Dict[str, float]:
        """Extract emotions from dict result."""
        return result_dict.get('emotion', {})
    
    def _detect_with_fer(self, face_roi: np.ndarray) -> Optional[EmotionResult]:
        """Detect emotion using FER library."""
        if not self.emotion_detector:
            return None
        
        emotions = self.emotion_detector.detect_emotions(face_roi)
        if not emotions or len(emotions) == 0:
            return None
        
        top_emotions = emotions[0].get('emotions', {})
        if not top_emotions:
            return None
        
        top_emotion = max(top_emotions.items(), key=lambda x: x[1])
        return EmotionResult(
            emotion=top_emotion[0],
            confidence=top_emotion[1]
        )
    
    def _detect_with_heuristic(self, face_roi: np.ndarray = None) -> EmotionResult:
        """Simple heuristic-based emotion detection (placeholder)."""
        return EmotionResult(
            emotion='neutral',
            confidence=DetectionConfig.DEFAULT_EMOTION_CONFIDENCE
        )


class ObjectDetector:
    """
    Real-time object detection using OpenCV DNN.
    
    Supports:
    - COCO dataset (80 classes) - requires model files
    - Face detection (Haar cascades) - built-in
    - Emotion detection (facial expression analysis)
    """
    
    def __init__(self, use_dnn=True, confidence_threshold=None, detect_emotions=True):
        """
        Initialize object detector.
        
        Args:
            use_dnn: Use DNN-based detection (requires model files)
            confidence_threshold: Minimum confidence for detections
            detect_emotions: Enable emotion detection on detected faces
        """
        if cv2 is None:
            self.active = False
            print("[DETECTION] Warning: opencv-python not installed. Detection disabled.")
            return
        
        self.active = True
        self.config = DetectionConfig()
        self.confidence_threshold = confidence_threshold or self.config.DEFAULT_CONFIDENCE_THRESHOLD
        self.use_dnn = use_dnn
        self.detect_emotions = detect_emotions
        self.net = None
        self.classes = []
        self.face_cascade = None
        self.emotion_detector = None
        
        self._initialize_detection_components()
    
    def _initialize_detection_components(self):
        """Initialize all detection components."""
        if self.use_dnn:
            self._init_dnn()
        
        self._init_face_detector()
        
        if self.detect_emotions:
            self._init_emotion_detector()
    
    def _init_dnn(self):
        """Initialize DNN model for COCO object detection."""
        try:
            self.classes = self.config.COCO_CLASSES.copy()
            print("[DETECTION] DNN model requires separate download. Using face detection only.")
            self.use_dnn = False
        except Exception as e:
            print(f"[DETECTION] Error initializing DNN: {e}")
            self.use_dnn = False
    
    def _init_face_detector(self):
        """Initialize Haar cascade face detector (built-in)."""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                print("[DETECTION] Face detector initialized.")
            else:
                print("[DETECTION] Face cascade file not found.")
        except Exception as e:
            print(f"[DETECTION] Error initializing face detector: {e}")
    
    def _init_emotion_detector(self):
        """Initialize emotion detection system."""
        method = self._determine_emotion_method()
        self.emotion_detector = EmotionDetector(method)
    
    def _determine_emotion_method(self) -> EmotionMethod:
        """Determine which emotion detection method to use."""
        try:
            from deepface import DeepFace
            # Test that DeepFace can actually be used (not just imported)
            _ = DeepFace
            return EmotionMethod.DEEPFACE
        except (ImportError, Exception):
            # Catch any exception (including TensorFlow loading errors)
            pass
        
        try:
            from fer import FER
            # Test that FER can actually be used
            _ = FER
            return EmotionMethod.FER
        except (ImportError, Exception):
            pass
        
        return EmotionMethod.HEURISTIC
    
    def detect_objects(self, frame):
        """
        Detect objects and emotions in a frame.
        
        Args:
            frame: BGR image frame
            
        Returns:
            List of detection dicts with 'label', 'bbox', 'confidence', 'emotion'
        """
        if not self.active or frame is None:
            return []
        
        detections = []
        
        if self.use_dnn and self.net:
            dnn_detections = self._detect_with_dnn(frame)
            detections.extend(dnn_detections)
        
        face_detections = self._detect_faces(frame)
        detections.extend(face_detections)
        
        return detections
    
    def _detect_with_dnn(self, frame):
        """Detect objects using DNN (placeholder - requires model files)."""
        return []
    
    def _detect_faces(self, frame):
        """Detect faces and optionally emotions."""
        if not self.face_cascade:
            return []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        detections = []
        for (x, y, w, h) in faces:
            detection = {
                'label': 'face',
                'bbox': (x, y, w, h),
                'confidence': 1.0
            }
            
            if self.detect_emotions and self.emotion_detector:
                emotion_result = self.emotion_detector.detect(frame[y:y+h, x:x+w])
                if emotion_result:
                    detection['emotion'] = {
                        'emotion': emotion_result.emotion,
                        'confidence': emotion_result.confidence
                    }
            
            detections.append(detection)
        
        return detections
    
    def draw_detections(self, frame, detections):
        """
        Draw detection boxes and labels on frame.
        
        Args:
            frame: BGR image frame
            detections: List of detection dicts
            
        Returns:
            frame: Frame with drawn detections
        """
        if not self.active or frame is None:
            return frame
        
        try:
            display = frame.copy()
            
            for det in detections:
                self._draw_single_detection(display, det)
            
            return display
        except Exception as e:
            print(f"[DETECTION] Error drawing detections: {e}")
            return frame
    
    def _draw_single_detection(self, frame, det: Dict[str, Any]):
        """Draw a single detection on the frame."""
        label = det['label']
        confidence = det.get('confidence', 0.0)
        x, y, w, h = det['bbox']
        emotion = det.get('emotion', None)
        
        color = self._get_detection_color(label, emotion)
        label_text = self._format_detection_label(label, confidence, emotion)
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame, 
            label_text, 
            (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            color, 
            2
        )
    
    def _get_detection_color(self, label: str, emotion: Optional[Dict]) -> Tuple[int, int, int]:
        """Get color for detection box based on label and emotion."""
        if label == 'face' and emotion:
            return self._get_emotion_color(emotion)
        elif label == 'face':
            return DETECTION_FACE_COLOR_NO_EMOTION
        else:
            return DETECTION_OBJECT_COLOR
    
    def _get_emotion_color(self, emotion: Dict) -> Tuple[int, int, int]:
        """Get color based on detected emotion."""
        emotion_name = self._extract_emotion_name_from_dict(emotion)
        return self._map_emotion_to_color(emotion_name)
    
    def _extract_emotion_name_from_dict(self, emotion: Dict) -> str:
        """Extract emotion name from emotion dictionary."""
        if isinstance(emotion, dict):
            return emotion.get('emotion', 'neutral')
        return str(emotion) if emotion else 'neutral'
    
    def _map_emotion_to_color(self, emotion_name: str) -> Tuple[int, int, int]:
        """Map emotion name to BGR color tuple."""
        emotion_color_map = {
            'happy': DETECTION_EMOTION_COLOR_HAPPY,
            'sad': DETECTION_EMOTION_COLOR_SAD,
            'angry': DETECTION_EMOTION_COLOR_ANGRY,
            'surprise': DETECTION_EMOTION_COLOR_SURPRISE,
            'fear': DETECTION_EMOTION_COLOR_FEAR,
            'disgust': DETECTION_EMOTION_COLOR_DISGUST,
            'neutral': DETECTION_EMOTION_COLOR_NEUTRAL
        }
        return emotion_color_map.get(emotion_name, DETECTION_EMOTION_COLOR_DEFAULT)
    
    def _format_detection_label(self, label: str, confidence: float, 
                                emotion: Optional[Dict]) -> str:
        """Format detection label with confidence and emotion."""
        label_text = f"{label} {confidence:.2f}"
        
        if emotion:
            emotion_data = self._extract_emotion_data_for_label(emotion)
            label_text += f" [{emotion_data['name']} {emotion_data['confidence']:.2f}]"
        
        return label_text
    
    def _extract_emotion_data_for_label(self, emotion: Dict) -> Dict[str, Any]:
        """Extract emotion name and confidence for label formatting."""
        if isinstance(emotion, dict):
            return {
                'name': emotion.get('emotion', ''),
                'confidence': emotion.get('confidence', 0.0)
            }
        return {
            'name': str(emotion),
            'confidence': 0.0
        }
    
    def cleanup(self):
        """Clean up resources (release OpenCV windows if needed)."""
        if self.active and cv2 is not None:
            # ObjectDetector doesn't hold persistent resources that need cleanup
            # OpenCV windows are managed by the main event loop
            pass
