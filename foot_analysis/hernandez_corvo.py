from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from utils.geometry import normalize_vector


@dataclass
class HCResult:
    index: float
    x_width: float
    y_width: float
    classification: str
    anterior_point: Tuple[int, int]
    posterior_point: Tuple[int, int]


def classify_plantar_index(index_value: float) -> str:
    if index_value < 30:
        return "Pie cavo"
    if index_value <= 45:
        return "Pie normal"
    return "Pie plano"


def _principal_axis_from_contour(contour: np.ndarray):
    pts = contour.reshape(-1, 2).astype(np.float32)
    mean, eigenvectors, _ = cv2.PCACompute2(pts, mean=None)
    center = mean[0]
    axis = normalize_vector(eigenvectors[0])

    projections = np.dot(pts - center, axis)
    idx_min = int(np.argmin(projections))
    idx_max = int(np.argmax(projections))

    posterior = tuple(pts[idx_min].astype(int))
    anterior = tuple(pts[idx_max].astype(int))

    return center, axis, posterior, anterior


def _measure_widths_rotated(mask: np.ndarray, center: np.ndarray, axis: np.ndarray):
    angle = np.degrees(np.arctan2(axis[1], axis[0]))

    h, w = mask.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle - 90.0, 1.0)
    rotated = cv2.warpAffine(mask, rot_mat, (w, h), flags=cv2.INTER_NEAREST)

    ys, xs = np.where(rotated > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No hay píxeles válidos tras la rotación")

    y_min, y_max = int(np.min(ys)), int(np.max(ys))
    x_min, x_max = int(np.min(xs)), int(np.max(xs))

    roi = rotated[y_min : y_max + 1, x_min : x_max + 1]

    widths = np.sum(roi > 0, axis=1).astype(float)
    n_rows = len(widths)
    if n_rows < 10:
        raise ValueError("Máscara insuficiente para medir anchos")

    fore_start, fore_end = 0, max(1, int(0.35 * n_rows))
    arch_start, arch_end = int(0.35 * n_rows), max(int(0.7 * n_rows), int(0.35 * n_rows) + 1)

    fore_region = widths[fore_start:fore_end]
    arch_region = widths[arch_start:arch_end]

    x_width = float(np.max(fore_region))
    y_width = float(np.min(arch_region))

    x_row_local = fore_start + int(np.argmax(fore_region))
    y_row_local = arch_start + int(np.argmin(arch_region))

    x_row = y_min + x_row_local
    y_row = y_min + y_row_local

    return {
        "rotated": rotated,
        "rot_mat": rot_mat,
        "x_width": x_width,
        "y_width": y_width,
        "x_row": x_row,
        "y_row": y_row,
        "x_min": x_min,
        "x_max": x_max,
    }


def apply_hernandez_corvo(mask: np.ndarray, contour: np.ndarray):
    center, axis, posterior, anterior = _principal_axis_from_contour(contour)
    widths_info = _measure_widths_rotated(mask, center, axis)

    x_width = widths_info["x_width"]
    y_width = widths_info["y_width"]
    index_value = (y_width / x_width) * 100.0 if x_width > 0 else float("nan")
    classification = classify_plantar_index(index_value)

    return HCResult(
        index=index_value,
        x_width=x_width,
        y_width=y_width,
        classification=classification,
        anterior_point=anterior,
        posterior_point=posterior,
    ), widths_info
