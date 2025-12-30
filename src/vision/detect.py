
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
from loguru import logger

from .embed import l2_normalize

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    from insightface.app import FaceAnalysis
except ImportError as e:
    raise ImportError(
        "No se pudo importar insightface. Instala con: pip install insightface"
    ) from e


@dataclass
class FaceDet:
    bbox_xyxy: np.ndarray              # shape (4,) float32  [x1,y1,x2,y2]
    kps: Optional[np.ndarray]          # shape (5,2) si existe
    det_score: float
    embedding: Optional[np.ndarray]    # shape (512,) float32 L2-normalized


class FaceDetector:
    def __init__(
        self,
        det_size: Tuple[int, int] = (640, 640),
        det_thresh: float = 0.6,
        max_faces: int = 0,
        use_gpu: bool = False,
        model_name: str = "buffalo_l",
    ) -> None:
        self.det_size = det_size
        self.det_thresh = det_thresh
        self.max_faces = max_faces
        self.use_gpu = use_gpu
        self.model_name = model_name

        providers = None
        if ort is not None:
            avail = ort.get_available_providers()
            if use_gpu and "CUDAExecutionProvider" in avail:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                logger.info("ONNXRuntime providers: CUDA + CPU")
            else:
                providers = ["CPUExecutionProvider"]
                logger.info("ONNXRuntime providers: CPU")

        # Algunos builds aceptan providers, otros no. Lo intentamos y fallback.
        try:
            self.app = FaceAnalysis(name=self.model_name, providers=providers)
        except TypeError:
            self.app = FaceAnalysis(name=self.model_name)

        ctx_id = 0 if use_gpu else -1
        self.app.prepare(ctx_id=ctx_id, det_size=self.det_size)
        # det_thresh suele ser atributo (más compatible)
        try:
            self.app.det_thresh = self.det_thresh
        except Exception:
            pass

        logger.info(
            f"FaceDetector listo | model={self.model_name} | det_size={self.det_size} | "
            f"det_thresh={self.det_thresh} | use_gpu={self.use_gpu}"
        )

    def detect(self, frame_bgr: np.ndarray) -> List[FaceDet]:
        faces = self.app.get(frame_bgr)
        out: List[FaceDet] = []

        for f in faces:
            bbox = np.array(getattr(f, "bbox", [0, 0, 0, 0]), dtype=np.float32)
            score = float(getattr(f, "det_score", 0.0))
            kps = getattr(f, "kps", None)

            emb = None
            if hasattr(f, "normed_embedding") and f.normed_embedding is not None:
                emb = np.asarray(f.normed_embedding, dtype=np.float32)
            elif hasattr(f, "embedding") and f.embedding is not None:
                emb = l2_normalize(np.asarray(f.embedding, dtype=np.float32))

            out.append(
                FaceDet(
                    bbox_xyxy=bbox,
                    kps=None if kps is None else np.asarray(kps, dtype=np.float32),
                    det_score=score,
                    embedding=emb,
                )
            )

        # Opcional: limitar a los mejores por score (si MAX_FACES > 0)
        if self.max_faces and self.max_faces > 0:
            out.sort(key=lambda x: x.det_score, reverse=True)
            out = out[: self.max_faces]

        return out
