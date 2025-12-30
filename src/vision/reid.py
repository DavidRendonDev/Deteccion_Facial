# src/vision/reid.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .embed import l2_normalize, cosine_similarity


@dataclass
class Person:
    person_id: int
    centroid: np.ndarray  # embedding promedio (normalizado)
    seen_count: int = 1


class ReIdentifier:
    """
    Maneja person_id persistente (re-ID) usando embeddings faciales.
    - Mantiene track_id -> person_id mientras el track exista
    - Si un track nuevo aparece, intenta matchear con centroides previos
    """

    def __init__(
        self,
        similarity_thresh: float = 0.45,
        update_alpha: float = 0.9,
        min_det_score: float = 0.65,
    ) -> None:
        self.similarity_thresh = similarity_thresh
        self.update_alpha = update_alpha
        self.min_det_score = min_det_score

        self._next_person_id = 1
        self.people: Dict[int, Person] = {}         # person_id -> Person
        self.track_to_person: Dict[int, int] = {}   # track_id -> person_id

    def cleanup_tracks(self, active_track_ids: set[int]) -> None:
        dead = [tid for tid in self.track_to_person.keys() if tid not in active_track_ids]
        for tid in dead:
            del self.track_to_person[tid]

    def _create_person(self, emb: np.ndarray) -> int:
        pid = self._next_person_id
        self._next_person_id += 1
        self.people[pid] = Person(person_id=pid, centroid=emb.copy(), seen_count=1)
        return pid

    def _best_match(self, emb: np.ndarray) -> tuple[Optional[int], float]:
        best_pid = None
        best_sim = -1.0
        for pid, person in self.people.items():
            sim = cosine_similarity(emb, person.centroid)
            if sim > best_sim:
                best_sim = sim
                best_pid = pid
        return best_pid, best_sim

    def _update_person(self, pid: int, emb: np.ndarray) -> None:
        person = self.people[pid]
        # EMA: centroid = alpha*centroid + (1-alpha)*emb
        new_centroid = self.update_alpha * person.centroid + (1.0 - self.update_alpha) * emb
        person.centroid = l2_normalize(new_centroid)
        person.seen_count += 1

    def assign_person_id(
        self,
        track_id: int,
        embedding: Optional[np.ndarray],
        det_score: float,
    ) -> Optional[int]:
        """
        Retorna person_id o None si no hay embedding confiable todavía.
        """
        # Si ya está asignado, mantenlo (estabilidad mientras está en cámara)
        if track_id in self.track_to_person:
            pid = self.track_to_person[track_id]
            if embedding is not None and det_score >= self.min_det_score and pid in self.people:
                self._update_person(pid, l2_normalize(embedding))
            return pid

        # Si no tenemos embedding bueno, no inventamos ID
        if embedding is None or det_score < self.min_det_score:
            return None

        emb = l2_normalize(embedding)

        # Intentar match con personas conocidas
        if len(self.people) > 0:
            best_pid, best_sim = self._best_match(emb)
            if best_pid is not None and best_sim >= self.similarity_thresh:
                self.track_to_person[track_id] = best_pid
                self._update_person(best_pid, emb)
                return best_pid

        # Si no matchea: crear persona nueva
        new_pid = self._create_person(emb)
        self.track_to_person[track_id] = new_pid
        return new_pid
