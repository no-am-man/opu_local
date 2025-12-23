"""
Object Detection Module for OPU Visual Cortex.
Uses OpenCV DNN for real-time object detection.

OPU v3.2 - Visual Object Recognition
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
    """
    
    def __init__(self, use_dnn=True, confidence_threshold=0.5):
        """
        Initialize object detector.
        
        Args:
            use_dnn: Use DNN-based detection (requires model files)
            confidence_threshold: Minimum confidence for detections
        """
        if cv2 is None:
            self.active = False
            print("[DETECTION] Warning: opencv-python not installed. Detection disabled.")
            return
        
        self.active = True
        self.confidence_threshold = confidence_threshold
        self.use_dnn = use_dnn
        self.net = None
        self.classes = []
        self.face_cascade = None
        
        # Initialize DNN model if requested
        if use_dnn:
            self._init_dnn()
        
        # Initialize face detector (always available)
        self._init_face_detector()
    
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
    
    def detect_objects(self, frame):
        """
        Detect objects in a frame.
        
        Args:
            frame: BGR image frame
            
        Returns:
            detections: List of dicts with 'label', 'confidence', 'bbox' (x, y, w, h)
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
                detections.append({
                    'label': 'face',
                    'confidence': 0.9,  # Haar cascades don't provide confidence
                    'bbox': (int(x), int(y), int(w), int(h))
                })
        
        # DNN detection (if available)
        if self.use_dnn and self.net is not None:
            # This would run DNN inference
            # For now, skip since we don't have model files
            pass
        
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
        
        display = frame.copy()
        
        for det in detections:
            label = det['label']
            confidence = det.get('confidence', 0.0)
            x, y, w, h = det['bbox']
            
            # Color based on label
            if label == 'face':
                color = (0, 255, 255)  # Yellow for faces
            else:
                color = (0, 255, 0)  # Green for other objects
            
            # Draw bounding box
            cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
            
            # Draw label with confidence
            label_text = f"{label}"
            if confidence > 0:
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

