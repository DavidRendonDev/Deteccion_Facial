
from __future__ import annotations

import time
import json
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from loguru import logger

from .config import load_settings
from .video.capture import VideoReader
from .vision.detect import FaceDetector, FaceDet
from .vision.track import IouTracker
from .vision.reid import ReIdentifier
from .vision.emotion import EmotionDetector


def _color_from_id(n: int) -> tuple[int, int, int]:
    # BGR (OpenCV)
    return (int((n * 37) % 255), int((n * 17) % 255), int((n * 29) % 255))


def draw_face(frame: np.ndarray, face: FaceDet, label: str = "") -> None:
    x1, y1, x2, y2 = face.bbox_xyxy.astype(int).tolist()
    color = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    if face.kps is not None:
        for (px, py) in face.kps.astype(int):
            cv2.circle(frame, (px, py), 2, (0, 255, 255), -1)

    if label:
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )


def extract_face_crop(frame: np.ndarray, bbox: np.ndarray, padding: float = 0.1) -> np.ndarray:
    """Extract face region from frame with optional padding"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    
    # Add padding
    pad_w = int((x2 - x1) * padding)
    pad_h = int((y2 - y1) * padding)
    
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)
    
    return frame[y1:y2, x1:x2]


def save_results_to_json(results: list, output_path: str) -> None:
    """Save detection results to JSON file"""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.debug(f"Saved {len(results)} results to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON: {e}")


def main() -> None:
    s = load_settings()
    logger.info(f"Settings: {s}")

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
    
    # Initialize emotion detector if enabled
    emotion_detector = EmotionDetector(backend=s.emotion_backend) if s.enable_emotion else None
    
    # JSON results storage
    json_results = []
    frame_number = 0

    fps = 0.0
    last_t = time.time()

    while True:
        ok, frame = reader.read()
        if not ok or frame is None:
            continue

        frame_number += 1
        t0 = time.time()
        faces = detector.detect(frame)
        
        # Emotion detection
        if emotion_detector is not None and len(faces) > 0:
            for face in faces:
                try:
                    face_crop = extract_face_crop(frame, face.bbox_xyxy)
                    if face_crop.size > 0:
                        emotion_result = emotion_detector.detect_emotion(face_crop)
                        face.emotion = emotion_result
                except Exception as e:
                    logger.debug(f"Emotion detection failed for face: {e}")

        # Tracking
        det_to_track = {}
        if tracker is not None:
            det_bboxes = [f.bbox_xyxy for f in faces]
            det_to_track = tracker.update(det_bboxes)

            # Limpia tracks muertos en reid
            if reid is not None:
                reid.cleanup_tracks(tracker.active_track_ids())

        # Collect results for JSON
        if s.save_results_json and len(faces) > 0:
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
                
                # Add tracking info
                track_id = det_to_track.get(i) if tracker is not None else None
                if track_id is not None:
                    face_data["track_id"] = track_id
                
                # Add person ID
                if reid is not None and track_id is not None:
                    pid = reid.assign_person_id(track_id, face.embedding, face.det_score)
                    if pid is not None:
                        face_data["person_id"] = pid
                
                # Add emotion data
                if face.emotion is not None:
                    face_data["emotion"] = {
                        "dominant": face.emotion.dominant_emotion,
                        "confidence": float(face.emotion.confidence),
                        "all_emotions": {k: float(v) for k, v in face.emotion.all_emotions.items()}
                    }
                
                frame_data["faces"].append(face_data)
            
            json_results.append(frame_data)

        # Dibujo
        for i, face in enumerate(faces):
            label_parts = [f"{face.det_score:.2f}"]

            track_id = None
            if tracker is not None:
                track_id = det_to_track.get(i)
                if track_id is not None:
                    label_parts.append(f"T{track_id}")

            if reid is not None and track_id is not None:
                pid = reid.assign_person_id(track_id, face.embedding, face.det_score)
                if pid is not None:
                    label_parts.append(f"P{pid}")
            
            # Add emotion to label
            if face.emotion is not None:
                emotion_label = f"{face.emotion.dominant_emotion}"
                label_parts.append(emotion_label)

            draw_face(frame, face, " | ".join(label_parts))

        # FPS
        dt = time.time() - last_t
        last_t = time.time()
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | faces: {len(faces)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(s.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
            
        # Periodic save (every 30 frames ~ 1 sec)
        if s.save_results_json and len(json_results) > 0 and frame_number % 30 == 0:
            save_results_to_json(json_results, s.json_output_path)

    # Save final results before exiting
    if s.save_results_json and len(json_results) > 0:
        save_results_to_json(json_results, s.json_output_path)
        logger.info(f"Saved {len(json_results)} frames to {s.json_output_path}")

    reader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
