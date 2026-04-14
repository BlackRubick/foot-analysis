from __future__ import annotations

import cv2
import numpy as np


def preprocess_foot_image(image: np.ndarray):
    """Retorna etapas de procesamiento para huella plantar."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Asegura pie en blanco sobre negro
    if np.sum(binary == 255) > np.sum(binary == 0):
        binary = cv2.bitwise_not(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel, iterations=2)

    edges = cv2.Canny(clean, 50, 150)

    return {
        "gray": gray,
        "binary": binary,
        "clean": clean,
        "edges": edges,
    }


def largest_contour(mask: np.ndarray, min_area_ratio: float = 0.01, max_area_ratio: float = 0.8):
    """
    Retorna el mayor contorno dentro de un rango de área razonable para un pie.
    min_area_ratio: proporción mínima del área de la imagen (descarta ruido pequeño)
    max_area_ratio: proporción máxima del área de la imagen (descarta objetos grandes)
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    img_h, img_w = mask.shape[:2]
    img_area = img_h * img_w
    min_area = img_area * min_area_ratio
    max_area = img_area * max_area_ratio

    def is_reasonable_foot_shape(cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = h / (w + 1e-5)
        # Pie: relación alto/ancho entre 1.2 y 3.5 (ajustable)
        if not (1.2 < aspect < 3.5):
            return False
        # No estar pegado a los bordes (10px margen)
        margin = 10
        if x < margin or y < margin or (x + w) > (img_w - margin) or (y + h) > (img_h - margin):
            return False
        return True

    # Filtra contornos por área, forma y posición
    filtered = [c for c in contours if min_area < cv2.contourArea(c) < max_area and is_reasonable_foot_shape(c)]
    if not filtered:
        return None
    # Selecciona el contorno más alargado (mayor relación alto/ancho)
    def aspect_ratio(cnt):
        x, y, w, h = cv2.boundingRect(cnt)
        return h / (w + 1e-5)
    return max(filtered, key=aspect_ratio)
