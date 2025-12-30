
from __future__ import annotations

import time
import cv2
import numpy as np
from loguru import logger

from .config import load_settings
from .video.capture import VideoReader
from .vision.detect import FaceDetector, FaceDet
from .vision.track import IouTracker
from .vision.reid import ReIdentifier


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

    fps = 0.0
    last_t = time.time()

    while True:
        ok, frame = reader.read()
        if not ok or frame is None:
            continue

        t0 = time.time()
        faces = detector.detect(frame)
        
        


        # Tracking
        det_to_track = {}
        if tracker is not None:
            det_bboxes = [f.bbox_xyxy for f in faces]
            det_to_track = tracker.update(det_bboxes)

            # Limpia tracks muertos en reid
            if reid is not None:
                reid.cleanup_tracks(tracker.active_track_ids())

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

    reader.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
