from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from chains_analysis.models import LandmarkPoint


Point = Tuple[float, float]


def point_xy(point: Optional[LandmarkPoint], fallback: Optional[Point] = None) -> Optional[Point]:
    if point is None:
        return fallback
    return float(point.x), float(point.y)


def midpoint(p1: Point, p2: Point) -> Point:
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def lerp(p1: Point, p2: Point, t: float) -> Point:
    return (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t)


def distance(p1: Point, p2: Point) -> float:
    return float(np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float)))


def angle(a: Point, b: Point, c: Point) -> float:
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)

    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return float("nan")

    cos_theta = float(np.dot(ba, bc) / denom)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def line_angle(a: Point, b: Point) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))


def signed_horizontal_offset(point: Point, reference_x: float, mm_per_px: float, anterior_sign: int = 1) -> float:
    return (point[0] - reference_x) * mm_per_px * float(anterior_sign)


def to_mm(value_px: float, mm_per_px: float) -> float:
    return value_px * mm_per_px


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))