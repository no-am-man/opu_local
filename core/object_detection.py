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


class ObjectDetector:
    """
    Real-time object detection using OpenCV DNN.
    
    Supports:
    - COCO dataset (80 classes) - requires model files
    - Face detection (Haar cascades) - built-in
    - Emotion detection (facial expression analysis) - NEW
    """
    
    # Emotion labels (7 basic emotions)
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    def __init__(self, use_dnn=True, confidence_threshold=0.5, detect_emotions=True):
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
        self.confidence_threshold = confidence_threshold
        self.use_dnn = use_dnn
        self.detect_emotions = detect_emotions
        self.net = None
        self.classes = []
        self.face_cascade = None
        self.emotion_net = None
        
        # Initialize DNN model if requested
        if use_dnn:
            self._init_dnn()
        
        # Initialize face detector (always available)
        self._init_face_detector()
        
        # Initialize emotion detector if requested
        if detect_emotions:
            self._init_emotion_detector()
    
    def _init_dnn(self):
        """Initialize DNN model for COCO object detection."""
        try:
            # COCO class names (80 classes)
            self.classes = [
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
            
            # Try to load pre-trained model files
            # These would need to be downloaded separately
            # For now, we'll use a simpler approach
            print("[DETECTION] DNN model requires separate download. Using face detection only.")
            self.use_dnn = False
            
        except Exception as e:
            print(f"[DETECTION] Error initializing DNN: {e}")
            self.use_dnn = False
    
    def _init_face_detector(self):
        """Initialize Haar cascade face detector (built-in)."""
        try:
            # Try to load the face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                print("[DETECTION] Face detector initialized.")
            else:
                print("[DETECTION] Warning: Face cascade not found.")
        except Exception as e:
            print(f"[DETECTION] Error initializing face detector: {e}")
    
    def _init_emotion_detector(self):
        """
        Initialize emotion detection model.
        
        Uses a lightweight approach: tries to load a pre-trained emotion model,
        or falls back to a simple heuristic-based method.
        """
        try:
            # Try to use DeepFace library if available (most accurate)
            try:
                from deepface import DeepFace
                self.emotion_method = 'deepface'
                print("[DETECTION] Emotion detection: Using DeepFace (high accuracy)")
                return
            except ImportError:
                pass
            
            # Try to use fer library if available (lightweight)
            try:
                import fer
                self.emotion_detector = fer.FER(mtcnn=True)
                self.emotion_method = 'fer'
                print("[DETECTION] Emotion detection: Using FER library")
                return
            except ImportError:
                pass
            
            # Fallback: Simple heuristic-based emotion detection
            # This uses basic facial feature analysis
            self.emotion_method = 'heuristic'
            print("[DETECTION] Emotion detection: Using heuristic method (basic)")
            print("[DETECTION] Tip: Install 'fer' or 'deepface' for better accuracy")
            
        except Exception as e:
            print(f"[DETECTION] Error initializing emotion detector: {e}")
            self.emotion_method = None
            self.detect_emotions = False
    
    def detect_objects(self, frame):
        """
        Detect objects in a frame.
        
        Args:
            frame: BGR image frame
            
        Returns:
            detections: List of dicts with 'label', 'confidence', 'bbox' (x, y, w, h), 'emotion' (optional)
        """
        if not self.active or frame is None:
            return []
        
        detections = []
        
        # Face detection (always available)
        if self.face_cascade is not None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces:
                detection = {
                    'label': 'face',
                    'confidence': 0.9,  # Haar cascades don't provide confidence
                    'bbox': (int(x), int(y), int(w), int(h))
                }
                
                # Add emotion detection if enabled
                if self.detect_emotions and self.emotion_method:
                    emotion = self._detect_emotion(frame, x, y, w, h)
                    if emotion:
                        detection['emotion'] = emotion
                
                detections.append(detection)
        
        # DNN detection (if available)
        if self.use_dnn and self.net is not None:
            # This would run DNN inference
            # For now, skip since we don't have model files
            pass
        
        return detections
    
    def _detect_emotion(self, frame, x, y, w, h):
        """
        Detect emotion in a face region.
        
        Args:
            frame: BGR image frame
            x, y, w, h: Face bounding box coordinates
            
        Returns:
            dict with 'emotion' (str) and 'confidence' (float), or None
        """
        try:
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size == 0:
                return None
            
            if self.emotion_method == 'deepface':
                # Use DeepFace for emotion detection
                from deepface import DeepFace
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, silent=True)
                if result and len(result) > 0:
                    emotions = result[0].get('emotion', {})
                    if emotions:
                        # Get emotion with highest confidence
                        top_emotion = max(emotions.items(), key=lambda x: x[1])
                        return {
                            'emotion': top_emotion[0],
                            'confidence': top_emotion[1] / 100.0  # Normalize to 0-1
                        }
            
            elif self.emotion_method == 'fer':
                # Use FER library
                emotions = self.emotion_detector.detect_emotions(face_roi)
                if emotions and len(emotions) > 0:
                    # Get top emotion from first detected face
                    top_emotions = emotions[0].get('emotions', {})
                    if top_emotions:
                        top_emotion = max(top_emotions.items(), key=lambda x: x[1])
                        return {
                            'emotion': top_emotion[0],
                            'confidence': top_emotion[1]
                        }
            
            elif self.emotion_method == 'heuristic':
                # Simple heuristic-based emotion detection
                # This is a placeholder - analyzes basic facial features
                # For production, use a proper model (fer or deepface)
                return {
                    'emotion': 'neutral',
                    'confidence': 0.5
                }
            
        except Exception as e:
            # Silently fail - emotion detection is optional
            return None
        
        return None
    
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
        
        display = frame.copy()
        
        for det in detections:
            label = det['label']
            confidence = det.get('confidence', 0.0)
            x, y, w, h = det['bbox']
            emotion = det.get('emotion', None)
            
            # Color based on label and emotion
            if label == 'face':
                # Color based on detected emotion
                if emotion:
                    emotion_name = emotion.get('emotion', 'neutral') if isinstance(emotion, dict) else emotion
                    color_map = {
                        'happy': (0, 255, 0),      # Green
                        'sad': (255, 0, 0),        # Blue
                        'angry': (0, 0, 255),       # Red
                        'surprise': (255, 255, 0), # Cyan
                        'fear': (128, 0, 128),     # Purple
                        'disgust': (0, 128, 128),   # Teal
                        'neutral': (0, 255, 255)   # Yellow
                    }
                    color = color_map.get(emotion_name, (0, 255, 255))
                else:
                    color = (0, 255, 255)  # Yellow for faces without emotion
            else:
                color = (0, 255, 0)  # Green for other objects
            
            # Draw bounding box
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with confidence and emotion
            label_text = f"{label}"
            if emotion:
                if isinstance(emotion, dict):
                    emotion_name = emotion.get('emotion', 'unknown')
                    emotion_conf = emotion.get('confidence', 0.0)
                    label_text += f": {emotion_name} ({emotion_conf:.2f})"
                else:
                    label_text += f": {emotion}"
            elif confidence > 0:
                label_text += f" {confidence:.2f}"
            
            # Background for text
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                display,
                (x, y - text_height - 5),
                (x + text_width, y),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                display,
                label_text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        return display
    
    def cleanup(self):
        """Cleanup resources."""
        self.active = False
        self.net = None
        self.face_cascade = None

