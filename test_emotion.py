#!/usr/bin/env python
"""
Quick test script to verify emotion detection and JSON output.
Processes a few frames from video and saves results.
"""

import cv2
import json
from datetime import datetime
from src.config import load_settings
from src.vision.detect import FaceDetector
from src.vision.emotion import EmotionDetector

def main():
    # Load settings
    s = load_settings()
    
    # Initialize detectors
    detector = FaceDetector(
        det_size=s.det_size,
        det_thresh=s.det_thresh,
        max_faces=s.max_faces,
        use_gpu=s.use_gpu,
    )
    
    emotion_detector = EmotionDetector(backend=s.emotion_backend)
    
    # Open video
    cap = cv2.VideoCapture("video.mp4")
    
    results = []
    frame_count = 0
    max_frames = 30  # Process only 30 frames for quick test
    
    print("Processing video frames...")
    
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect faces
        faces = detector.detect(frame)
        
        if len(faces) == 0:
            continue
        
        # Detect emotions
        for face in faces:
            try:
                x1, y1, x2, y2 = face.bbox_xyxy.astype(int)
                h, w = frame.shape[:2]
                pad = 10
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
                face_crop = frame[y1:y2, x1:x2]
                
                if face_crop.size > 0:
                    emotion_result = emotion_detector.detect_emotion(face_crop)
                    face.emotion = emotion_result
            except Exception as e:
                print(f"Emotion detection failed: {e}")
        
        # Collect results
        frame_data = {
            "timestamp": datetime.now().isoformat(),
            "frame_number": frame_count,
            "faces": []
        }
        
        for face in faces:
            face_data = {
                "bbox": face.bbox_xyxy.tolist(),
                "confidence": float(face.det_score),
            }
            
            if face.emotion is not None:
                face_data["emotion"] = {
                    "dominant": face.emotion.dominant_emotion,
                    "confidence": float(face.emotion.confidence),
                    "all_emotions": {k: float(v) for k, v in face.emotion.all_emotions.items()}
                }
            
            frame_data["faces"].append(face_data)
        
        results.append(frame_data)
        print(f"Frame {frame_count}: {len(faces)} face(s) detected")
    
    cap.release()
    
    # Save results
    output_path = "emotion_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved {len(results)} frames to {output_path}")
    print(f"Total faces detected: {sum(len(r['faces']) for r in results)}")
    
    # Show sample result
    if results and results[0]["faces"]:
        print("\n📊 Sample result (first frame with faces):")
        print(json.dumps(results[0], indent=2))

if __name__ == "__main__":
    main()
