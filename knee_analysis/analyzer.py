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

        # Hombros, caderas y ojos
        left_shoulder = (int(landmarks["left_shoulder"].x), int(landmarks["left_shoulder"].y)) if "left_shoulder" in landmarks else None
        right_shoulder = (int(landmarks["right_shoulder"].x), int(landmarks["right_shoulder"].y)) if "right_shoulder" in landmarks else None
        left_hip = (int(landmarks["left_hip"].x), int(landmarks["left_hip"].y)) if "left_hip" in landmarks else None
        right_hip = (int(landmarks["right_hip"].x), int(landmarks["right_hip"].y)) if "right_hip" in landmarks else None
        left_eye = (int(landmarks["left_ear"].x), int(landmarks["left_ear"].y)) if "left_ear" in landmarks else None
        right_eye = (int(landmarks["right_ear"].x), int(landmarks["right_ear"].y)) if "right_ear" in landmarks else None

        hip = (int(hip_lm.x), int(hip_lm.y))
        knee = (int(knee_lm.x), int(knee_lm.y))
        ankle = (int(ankle_lm.x), int(ankle_lm.y))

        angle = angle_between_points(hip, knee, ankle)
        classification = self.classify(angle, plane)


        annotated = image.copy()
        h, w = annotated.shape[:2]

        # Línea vertical central (azul)
        center_x = w // 2
        cv2.line(annotated, (center_x, 0), (center_x, h), (255, 0, 0), 2)

        # Línea hombro a hombro (amarillo)
        if left_shoulder and right_shoulder:
            cv2.line(annotated, left_shoulder, right_shoulder, (0, 255, 255), 3)

        # Línea cadera a cadera (magenta)
        if left_hip and right_hip:
            cv2.line(annotated, left_hip, right_hip, (255, 0, 255), 3)

        # Línea ojo a ojo (cyan)
        if left_eye and right_eye:
            cv2.line(annotated, left_eye, right_eye, (255, 255, 0), 3)

        # Puntos anatómicos extra
        if left_shoulder:
            cv2.circle(annotated, left_shoulder, 7, (0, 255, 255), -1)
        if right_shoulder:
            cv2.circle(annotated, right_shoulder, 7, (0, 255, 255), -1)
        if left_hip:
            cv2.circle(annotated, left_hip, 7, (255, 0, 255), -1)
        if right_hip:
            cv2.circle(annotated, right_hip, 7, (255, 0, 255), -1)
        if left_eye:
            cv2.circle(annotated, left_eye, 7, (255, 255, 0), -1)
        if right_eye:
            cv2.circle(annotated, right_eye, 7, (255, 255, 0), -1)

        # Línea de cadera a rodilla y rodilla a tobillo (verde)
        cv2.line(annotated, hip, knee, (0, 255, 0), 3)
        cv2.line(annotated, knee, ankle, (0, 255, 0), 3)

        # Línea horizontal por la cadera (roja)
        cv2.line(annotated, (0, hip[1]), (w, hip[1]), (0, 0, 255), 2)

        # Puntos anatómicos
        cv2.circle(annotated, hip, 7, (0, 255, 0), -1)
        cv2.circle(annotated, knee, 7, (0, 255, 255), -1)
        cv2.circle(annotated, ankle, 7, (0, 0, 255), -1)

        # Medidas y ángulos en recuadros verdes
        box_color = (36, 180, 80)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        # Ángulo de rodilla
        angle_text = f"{angle:.1f}°"
        (tw, th), _ = cv2.getTextSize(angle_text, font, font_scale, thickness)
        cv2.rectangle(annotated, (knee[0] - tw//2 - 8, knee[1] - th - 18), (knee[0] + tw//2 + 8, knee[1] - 8), box_color, -1)
        cv2.putText(annotated, angle_text, (knee[0] - tw//2, knee[1] - 12), font, font_scale, (255,255,255), 2)

        # Distancia cadera-rodilla
        dist_hip_knee = np.linalg.norm(np.array(hip) - np.array(knee))
        dist_text = f"{dist_hip_knee:.1f} px"
        (tw, th), _ = cv2.getTextSize(dist_text, font, font_scale, thickness)
        mid_hip_knee = ((hip[0] + knee[0]) // 2, (hip[1] + knee[1]) // 2)
        cv2.rectangle(annotated, (mid_hip_knee[0] - tw//2 - 8, mid_hip_knee[1] - th - 18), (mid_hip_knee[0] + tw//2 + 8, mid_hip_knee[1] - 8), box_color, -1)
        cv2.putText(annotated, dist_text, (mid_hip_knee[0] - tw//2, mid_hip_knee[1] - 12), font, font_scale, (255,255,255), 2)

        # Distancia rodilla-tobillo
        dist_knee_ankle = np.linalg.norm(np.array(knee) - np.array(ankle))
        dist_text2 = f"{dist_knee_ankle:.1f} px"
        (tw, th), _ = cv2.getTextSize(dist_text2, font, font_scale, thickness)
        mid_knee_ankle = ((knee[0] + ankle[0]) // 2, (knee[1] + ankle[1]) // 2)
        cv2.rectangle(annotated, (mid_knee_ankle[0] - tw//2 - 8, mid_knee_ankle[1] - th - 18), (mid_knee_ankle[0] + tw//2 + 8, mid_knee_ankle[1] - 8), box_color, -1)
        cv2.putText(annotated, dist_text2, (mid_knee_ankle[0] - tw//2, mid_knee_ankle[1] - 12), font, font_scale, (255,255,255), 2)

        # Clasificación y plano
        cv2.putText(
            annotated,
            f"Plano: {plane} | {classification}",
            (20, 30),
            font,
            0.8,
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
