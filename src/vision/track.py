
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """
    a, b: [x1,y1,x2,y2]
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 1e-9:
        return 0.0
    return float(inter_area / union)


@dataclass
class Track:
    track_id: int
    bbox_xyxy: np.ndarray
    hits: int = 1
    missed: int = 0


class IouTracker:
    """
    Tracker básico: asocia detecciones a tracks por IoU.
    Es simple pero funcional para comenzar.
    """

    def __init__(self, iou_thresh: float = 0.3, max_missed: int = 25) -> None:
        self.iou_thresh = iou_thresh
        self.max_missed = max_missed
        self._next_id = 1
        self.tracks: Dict[int, Track] = {}

    def _new_track(self, det_bbox: np.ndarray) -> int:
        tid = self._next_id
        self._next_id += 1
        self.tracks[tid] = Track(track_id=tid, bbox_xyxy=det_bbox.copy())
        return tid

    def update(self, det_bboxes: List[np.ndarray]) -> Dict[int, int]:
        """
        det_bboxes: lista de bbox [x1,y1,x2,y2]
        retorna: mapping {det_idx -> track_id}
        """
        assignments: Dict[int, int] = {}

        # No hay detecciones
        if len(det_bboxes) == 0:
            # todos los tracks se “pierden” un frame
            to_delete = []
            for tid, tr in self.tracks.items():
                tr.missed += 1
                if tr.missed > self.max_missed:
                    to_delete.append(tid)
            for tid in to_delete:
                del self.tracks[tid]
            return assignments

        # No hay tracks: crear uno por detección
        if len(self.tracks) == 0:
            for i, bb in enumerate(det_bboxes):
                tid = self._new_track(bb)
                assignments[i] = tid
            return assignments

        track_ids = list(self.tracks.keys())
        unmatched_tracks = set(track_ids)
        unmatched_dets = set(range(len(det_bboxes)))

        # Greedy matching por mejor IoU
        while unmatched_tracks and unmatched_dets:
            best_iou = 0.0
            best_pair: Optional[Tuple[int, int]] = None

            for tid in unmatched_tracks:
                tr = self.tracks[tid]
                for di in unmatched_dets:
                    val = iou_xyxy(tr.bbox_xyxy, det_bboxes[di])
                    if val > best_iou:
                        best_iou = val
                        best_pair = (tid, di)

            if best_pair is None or best_iou < self.iou_thresh:
                break

            tid, di = best_pair
            # asignar
            assignments[di] = tid
            unmatched_tracks.remove(tid)
            unmatched_dets.remove(di)

            # actualizar track
            tr = self.tracks[tid]
            tr.bbox_xyxy = det_bboxes[di].copy()
            tr.hits += 1
            tr.missed = 0

        # Tracks no asignados: missed++
        to_delete = []
        for tid in list(unmatched_tracks):
            tr = self.tracks[tid]
            tr.missed += 1
            if tr.missed > self.max_missed:
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        # Detecciones no asignadas: nuevos tracks
        for di in list(unmatched_dets):
            tid = self._new_track(det_bboxes[di])
            assignments[di] = tid

        return assignments

    def active_track_ids(self) -> set[int]:
        return set(self.tracks.keys())
