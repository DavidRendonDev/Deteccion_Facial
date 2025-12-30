# src/config.py
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Union, Tuple


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_video_source(raw: str) -> Union[int, str]:
    raw = raw.strip()
    # Si es "0", "1", "2" => webcam
    if raw.isdigit():
        return int(raw)
    # Si no, puede ser RTSP/archivo
    return raw


@dataclass(frozen=True)
class Settings:
    # Video
    video_source: Union[int, str]
    frame_width: int
    frame_height: int
    window_name: str

    # Runtime toggles (para ir paso a paso sin romperte la cabeza)
    enable_tracking: bool
    enable_reid: bool
    use_gpu: bool

    # Detección (InsightFace)
    det_size: Tuple[int, int]
    det_thresh: float
    max_faces: int

    # Tracking (IOU tracker simple)
    tracker_iou_thresh: float
    tracker_max_missed: int

    # Re-ID (persistencia)
    reid_similarity_thresh: float
    reid_update_alpha: float
    reid_min_det_score: float


def load_settings() -> Settings:
    video_source = _parse_video_source(os.getenv("VIDEO_SOURCE", "0"))

    det_w = _env_int("DET_W", 640)
    det_h = _env_int("DET_H", 640)

    return Settings(
        video_source=video_source,
        frame_width=_env_int("FRAME_W", 1280),
        frame_height=_env_int("FRAME_H", 720),
        window_name=os.getenv("WINDOW_NAME", "Face Pipeline"),
        enable_tracking=_env_bool("ENABLE_TRACKING", True),
        enable_reid=_env_bool("ENABLE_REID", True),
        use_gpu=_env_bool("USE_GPU", False),  # arranca en CPU, luego optimizamos
        det_size=(det_w, det_h),
        det_thresh=_env_float("DET_THRESH", 0.6),
        max_faces=_env_int("MAX_FACES", 0),  # 0 = sin límite (ojo rendimiento)
        tracker_iou_thresh=_env_float("TRACKER_IOU_THRESH", 0.3),
        tracker_max_missed=_env_int("TRACKER_MAX_MISSED", 25),
        reid_similarity_thresh=_env_float("REID_SIM_THRESH", 0.45),
        reid_update_alpha=_env_float("REID_ALPHA", 0.9),
        reid_min_det_score=_env_float("REID_MIN_DET_SCORE", 0.65),
    )
