from __future__ import annotations

import math
from typing import Tuple

import numpy as np


Point = Tuple[float, float]


def angle_between_points(a: Point, b: Point, c: Point) -> float:
    """Ángulo ABC en grados usando producto punto."""
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)

    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return float("nan")

    cos_theta = float(np.dot(ba, bc) / denom)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def euclidean_distance(p1: Point, p2: Point) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def normalize_vector(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm
