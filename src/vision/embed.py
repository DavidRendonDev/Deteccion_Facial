
from __future__ import annotations

import numpy as np


def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = v.astype(np.float32, copy=False)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Asume a y b ya normalizados L2. Entonces coseno = dot(a, b).
    """
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)
    return float(np.dot(a, b))
