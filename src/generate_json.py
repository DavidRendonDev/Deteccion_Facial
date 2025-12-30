"""
Headless script to generate emotion results JSON from video.
Run this if you just want the JSON data without the visualization window.
"""
from __future__ import annotations
import json
import time
from datetime import datetime
import cv2
import numpy as np
from loguru import logger

from .config import load_settings
from .video.capture import VideoReader
from .vision.detect import FaceDetector
from .vision.track import IouTracker
from .vision.reid import ReIdentifier
from .vision.emotion import EmotionDetector
from .main import extract_face_crop, save_results_to_json

def generate():
    s = load_settings()
    logger.info(f"Starting headless generation. Settings: {s}")

    reader = VideoReader(s.video_source, frame_size=(s.frame_width, s.frame_height))
    detector = FaceDetector(
        det_size=s.det_size,
        det_thresh=s.det_thresh,
        max_faces=s.max_faces,
        use_gpu=s.use_gpu,
    )

    tracker = IouTracker(iou_thresh=s.tracker_iou_thresh, max_missed=s.tracker_max_missed) if s.enable_tracking else None
    reid = ReIdentifier(
        similarity_thresh=s.reid_similarity_thresh,
        update_alpha=s.reid_update_alpha,
        min_det_score=s.reid_min_det_score,
    ) if s.enable_reid else None
    
    emotion_detector = EmotionDetector(backend=s.emotion_backend) if s.enable_emotion else None
    
    json_results = []
    frame_number = 0
    start_time = time.time()
    
    # Process max 50 frames (enough for testing)
    MAX_FRAMES = 50 

    while frame_number < MAX_FRAMES:
        ok, frame = reader.read()
        if not ok or frame is None:
            break

        frame_number += 1
        if frame_number % 10 == 0:
            logger.info(f"Processing frame {frame_number}...")

        faces = detector.detect(frame)
        
        # Emotion detection
        if emotion_detector is not None and len(faces) > 0:
            for face in faces:
                try:
                    face_crop = extract_face_crop(frame, face.bbox_xyxy)
                    if face_crop.size > 0:
                        face.emotion = emotion_detector.detect_emotion(face_crop)
                except Exception:
                    pass

        # Tracking
        det_to_track = {}
        if tracker is not None:
            det_bboxes = [f.bbox_xyxy for f in faces]
            det_to_track = tracker.update(det_bboxes)
            if reid is not None:
                reid.cleanup_tracks(tracker.active_track_ids())

        # Collect results
        if len(faces) > 0:
            frame_data = {
                "timestamp": datetime.now().isoformat(),
                "frame_number": frame_number,
                "faces": []
            }
            
            for i, face in enumerate(faces):
                face_data = {
                    "bbox": face.bbox_xyxy.tolist(),
                    "confidence": float(face.det_score),
                }
                
                track_id = det_to_track.get(i) if tracker is not None else None
                if track_id is not None:
                    face_data["track_id"] = track_id
                
                if reid is not None and track_id is not None:
                    pid = reid.assign_person_id(track_id, face.embedding, face.det_score)
                    if pid is not None:
                        face_data["person_id"] = pid
                
                if face.emotion is not None:
                    face_data["emotion"] = {
                        "dominant": face.emotion.dominant_emotion,
                        "confidence": float(face.emotion.confidence),
                        "all_emotions": {k: float(v) for k, v in face.emotion.all_emotions.items()}
                    }
                
                frame_data["faces"].append(face_data)
            
            json_results.append(frame_data)

    reader.release()
    
    if len(json_results) > 0:
        save_results_to_json(json_results, s.json_output_path)
        logger.info(f"SUCCESS: Saved {len(json_results)} frames to {s.json_output_path}")
    else:
        logger.warning("No results to save.")

if __name__ == "__main__":
    generate()
