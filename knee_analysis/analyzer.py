from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from utils.geometry import angle_between_points
from utils.pose_detector import PoseDetector


class KneeAnalyzer:
    def __init__(self):
        self.detector = PoseDetector()

    def _select_side(self, landmarks: Dict[str, object]):
        left_vis = (
            landmarks["left_hip"].visibility
            + landmarks["left_knee"].visibility
            + landmarks["left_ankle"].visibility
        )
        right_vis = (
            landmarks["right_hip"].visibility
            + landmarks["right_knee"].visibility
            + landmarks["right_ankle"].visibility
        )
        return "left" if left_vis >= right_vis else "right"

    @staticmethod
    def classify(angle: float, plane: str) -> str:
        plane = plane.lower()

        if plane == "frontal":
            if 170 <= angle <= 175:
                return "Normal"
            if angle < 170:
                return "Genu Valgo"
            if angle > 180:
                return "Genu Varo"
            return "Límite / indeterminado"

        if plane == "sagital":
            if 175 <= angle <= 185:
                return "Normal"
            if angle < 175:
                return "Genu Flexum"
            if angle > 185:
                return "Genu Recurvatum"
            return "Límite / indeterminado"

        return "Plano no válido"

    def analyze(self, image: np.ndarray, plane: str = "frontal") -> Dict:
        try:
            landmarks = self.detector.detect(image)
        except Exception as e:
            # No se detectaron landmarks, devolver imagen con mensaje
            annotated = image.copy()
            cv2.putText(
                annotated,
                "No se detectaron puntos de referencia (landmarks)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            return {
                "metrics": {
                    "plane": plane,
                    "side": "-",
                    "knee_angle_deg": 0.0,
                    "classification": "No detectado",
                },
                "images": {
                    "annotated": annotated,
                },
            }

        side = self._select_side(landmarks)

        if side == "left":
            hip_lm, knee_lm, ankle_lm = (
                landmarks["left_hip"],
                landmarks["left_knee"],
                landmarks["left_ankle"],
            )
        else:
            hip_lm, knee_lm, ankle_lm = (
                landmarks["right_hip"],
                landmarks["right_knee"],
                landmarks["right_ankle"],
            )

        hip = (int(hip_lm.x), int(hip_lm.y))
        knee = (int(knee_lm.x), int(knee_lm.y))
        ankle = (int(ankle_lm.x), int(ankle_lm.y))

        angle = angle_between_points(hip, knee, ankle)
        classification = self.classify(angle, plane)

        annotated = image.copy()
        cv2.line(annotated, hip, knee, (255, 0, 0), 3)
        cv2.line(annotated, knee, ankle, (255, 0, 0), 3)
        cv2.circle(annotated, hip, 6, (0, 255, 0), -1)
        cv2.circle(annotated, knee, 6, (0, 255, 255), -1)
        cv2.circle(annotated, ankle, 6, (0, 0, 255), -1)

        cv2.putText(
            annotated,
            f"Angulo rodilla: {angle:.2f} deg",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            annotated,
            f"Plano: {plane} | {classification}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        return {
            "metrics": {
                "plane": plane,
                "side": side,
                "knee_angle_deg": angle,
                "classification": classification,
            },
            "images": {
                "annotated": annotated,
            },
        }
