from __future__ import annotations

from typing import Dict

import cv2
import numpy as np

from utils.pose_detector import PoseDetector


class PostureAnalyzer:
    def __init__(self):
        self.detector = PoseDetector()

    def _select_side(self, landmarks: Dict[str, object]):
        left_vis = (
            landmarks["left_ear"].visibility
            + landmarks["left_shoulder"].visibility
            + landmarks["left_hip"].visibility
        )
        right_vis = (
            landmarks["right_ear"].visibility
            + landmarks["right_shoulder"].visibility
            + landmarks["right_hip"].visibility
        )
        return "left" if left_vis >= right_vis else "right"

    @staticmethod
    def _classify(mean_dev_px: float, tol_px: float) -> str:
        if abs(mean_dev_px) <= tol_px:
            return "Postura alineada"
        if mean_dev_px > tol_px:
            return "Cuerpo anteriorizado"
        return "Cuerpo posteriorizado"

    def analyze(self, image: np.ndarray) -> Dict:
        h, w = image.shape[:2]
        landmarks = self.detector.detect(image)
        side = self._select_side(landmarks)

        if side == "left":
            ear_lm, sh_lm, hip_lm = (
                landmarks["left_ear"],
                landmarks["left_shoulder"],
                landmarks["left_hip"],
            )
        else:
            ear_lm, sh_lm, hip_lm = (
                landmarks["right_ear"],
                landmarks["right_shoulder"],
                landmarks["right_hip"],
            )

        ear = (int(ear_lm.x), int(ear_lm.y))
        shoulder = (int(sh_lm.x), int(sh_lm.y))
        hip = (int(hip_lm.x), int(hip_lm.y))

        plumb_x = shoulder[0]
        deviations = np.array([ear[0] - plumb_x, shoulder[0] - plumb_x, hip[0] - plumb_x], dtype=float)
        mean_dev = float(np.mean(deviations))

        tol_px = max(8.0, 0.015 * w)
        classification = self._classify(mean_dev, tol_px)

        annotated = image.copy()
        cv2.line(annotated, (plumb_x, 0), (plumb_x, h - 1), (255, 255, 0), 2)

        for p, color, name in [
            (ear, (0, 255, 0), "Oreja"),
            (shoulder, (0, 255, 255), "Hombro"),
            (hip, (0, 128, 255), "Cadera"),
        ]:
            cv2.circle(annotated, p, 6, color, -1)
            cv2.putText(annotated, name, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.putText(
            annotated,
            f"Desv media: {mean_dev:.2f}px",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            annotated,
            classification,
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 0),
            2,
        )

        return {
            "metrics": {
                "side": side,
                "mean_deviation_px": mean_dev,
                "classification": classification,
            },
            "images": {
                "annotated": annotated,
            },
        }
