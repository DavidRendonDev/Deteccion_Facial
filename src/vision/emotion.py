"""
Emotion detection module using DeepFace.
Analyzes facial expressions to detect emotions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict
import numpy as np
from loguru import logger

try:
    from deepface import DeepFace
except ImportError as e:
    raise ImportError(
        "No se pudo importar deepface. Instala con: pip install deepface"
    ) from e


@dataclass
class EmotionResult:
    """Result of emotion detection"""
    dominant_emotion: str  # The emotion with highest probability
    confidence: float  # Confidence of the dominant emotion (0-1)
    all_emotions: Dict[str, float]  # All emotion probabilities


class EmotionDetector:
    """
    Detects emotions from face images using DeepFace.
    
    Supports 7 basic emotions: angry, disgust, fear, happy, sad, surprise, neutral
    """
    
    def __init__(self, backend: str = 'opencv', enforce_detection: bool = False):
        """
        Initialize emotion detector.
        
        Args:
            backend: DeepFace backend for face detection ('opencv', 'ssd', 'mtcnn', etc.)
            enforce_detection: If True, raise error when face not detected
        """
        self.backend = backend
        self.enforce_detection = enforce_detection
        logger.info(f"EmotionDetector initialized | backend={backend}")
    
    def detect_emotion(self, face_img: np.ndarray) -> Optional[EmotionResult]:
        """
        Detect emotion from a face image.
        
        Args:
            face_img: BGR image containing a face (numpy array)
            
        Returns:
            EmotionResult if successful, None if detection fails
        """
        try:
            # DeepFace expects RGB, but we work with BGR in OpenCV
            # DeepFace.analyze handles the conversion internally
            result = DeepFace.analyze(
                img_path=face_img,
                actions=['emotion'],
                enforce_detection=self.enforce_detection,
                detector_backend=self.backend,
                silent=True
            )
            
            # DeepFace returns a list of results (one per face)
            if isinstance(result, list) and len(result) > 0:
                result = result[0]
            
            # Extract emotion data
            emotions = result.get('emotion', {})
            dominant = result.get('dominant_emotion', 'neutral')
            
            # Get confidence of dominant emotion
            confidence = emotions.get(dominant, 0.0) / 100.0  # Convert to 0-1 range
            
            # Normalize all emotions to 0-1 range
            all_emotions = {k: v / 100.0 for k, v in emotions.items()}
            
            return EmotionResult(
                dominant_emotion=dominant,
                confidence=confidence,
                all_emotions=all_emotions
            )
            
        except Exception as e:
            logger.debug(f"Emotion detection failed: {e}")
            return None
    
    def detect_emotions_batch(self, face_imgs: list[np.ndarray]) -> list[Optional[EmotionResult]]:
        """
        Detect emotions for multiple face images.
        
        Args:
            face_imgs: List of BGR face images
            
        Returns:
            List of EmotionResults (None for failed detections)
        """
        results = []
        for face_img in face_imgs:
            result = self.detect_emotion(face_img)
            results.append(result)
        return results
