from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from foot_analysis.hernandez_corvo import apply_hernandez_corvo
from foot_analysis.preprocessing import largest_contour, preprocess_foot_image


class FootAnalyzer:
    def analyze(self, image: np.ndarray) -> Dict:
        steps = preprocess_foot_image(image)
        contour = largest_contour(steps["clean"])

        if contour is None:
            raise ValueError("No se detectó contorno de pie")

        hc_result, widths_info = apply_hernandez_corvo(steps["clean"], contour)

        annotated = image.copy()
        cv2.drawContours(annotated, [contour], -1, (0, 255, 0), 2)
        cv2.line(
            annotated,
            tuple(map(int, hc_result.posterior_point)),
            tuple(map(int, hc_result.anterior_point)),
            (255, 0, 0),
            2,
        )
        cv2.circle(annotated, tuple(map(int, hc_result.posterior_point)), 6, (0, 0, 255), -1)
        cv2.circle(annotated, tuple(map(int, hc_result.anterior_point)), 6, (255, 255, 0), -1)

        text = (
            f"Indice plantar={hc_result.index:.2f} | "
            f"X={hc_result.x_width:.1f}px Y={hc_result.y_width:.1f}px | "
            f"{hc_result.classification}"
        )
        cv2.putText(annotated, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        rotated_bgr = cv2.cvtColor(widths_info["rotated"], cv2.COLOR_GRAY2BGR)
        x_row = int(widths_info["x_row"])
        y_row = int(widths_info["y_row"])
        x_min, x_max = int(widths_info["x_min"]), int(widths_info["x_max"])

        cv2.line(rotated_bgr, (x_min, x_row), (x_max, x_row), (0, 255, 0), 2)
        cv2.line(rotated_bgr, (x_min, y_row), (x_max, y_row), (0, 165, 255), 2)
        cv2.putText(rotated_bgr, "X (antepie)", (x_min + 5, max(15, x_row - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(rotated_bgr, "Y (arco)", (x_min + 5, max(15, y_row - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        return {
            "metrics": {
                "plantar_index": hc_result.index,
                "x_width_px": hc_result.x_width,
                "y_width_px": hc_result.y_width,
                "classification": hc_result.classification,
            },
            "images": {
                "gray": steps["gray"],
                "binary": steps["binary"],
                "clean": steps["clean"],
                "edges": steps["edges"],
                "annotated": annotated,
                "rotated_widths": rotated_bgr,
            },
        }
