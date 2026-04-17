from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from utils.geometry import angle_between_points
from utils.pose_detector import PoseDetector




class KneeAnalyzer:
    def __init__(self):
        self.detector = PoseDetector()

    def draw_q_angle(self, image: np.ndarray) -> dict:
        """
        Dibuja el ángulo Q en ambas piernas (izquierda y derecha) sobre la imagen,
        usando los landmarks de MediaPipe (hip ≈ EIAS, knee ≈ patela, ankle ≈ tuberosidad tibial).
        Devuelve imagen anotada y ángulos Q.
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]
        try:
            landmarks = self.detector.detect(image)
        except Exception as e:
            cv2.putText(
                annotated,
                "No se detectaron puntos de referencia",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            return {"annotated": annotated, "q_angle_left": None, "q_angle_right": None}


        # Anatomical keypoints
        left_hip = (int(landmarks["left_hip"].x), int(landmarks["left_hip"].y))
        right_hip = (int(landmarks["right_hip"].x), int(landmarks["right_hip"].y))
        left_knee = (int(landmarks["left_knee"].x), int(landmarks["left_knee"].y))
        right_knee = (int(landmarks["right_knee"].x), int(landmarks["right_knee"].y))
        left_ankle = (int(landmarks["left_ankle"].x), int(landmarks["left_ankle"].y))
        right_ankle = (int(landmarks["right_ankle"].x), int(landmarks["right_ankle"].y))

        # Groin (entrepierna):
        # X = punto medio entre rodillas (alineado con las piernas)
        crotch_x = int((left_knee[0] + right_knee[0]) / 2)
        # Y = 70% hacia la rodilla desde la cadera (más realista clínicamente)
        mid_hip_y = (left_hip[1] + right_hip[1]) / 2
        mid_knee_y = (left_knee[1] + right_knee[1]) / 2
        crotch_y = int(mid_hip_y + (mid_knee_y - mid_hip_y) * 0.7)
        crotch = (crotch_x, crotch_y)

        # Colors
        blue = (255, 102, 0)  # Azul fuerte para ejes de miembros
        orange = (0, 140, 255) # Naranja para referencia
        yellow = (0, 255, 255)
        green = (36, 180, 80)

        # Draw anatomical keypoints
        for pt in [left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle, crotch]:
            cv2.circle(annotated, pt, 8, (0,0,0), -1)
            cv2.circle(annotated, pt, 5, (255,255,255), -1)

        # Draw limb axes (blue): groin to knees, knees to ankles
        cv2.line(annotated, crotch, left_knee, blue, 4)
        cv2.line(annotated, crotch, right_knee, blue, 4)
        cv2.line(annotated, left_knee, left_ankle, blue, 4)
        cv2.line(annotated, right_knee, right_ankle, blue, 4)

        # Draw reference alignment (orange): hip to knee
        cv2.line(annotated, left_hip, left_knee, orange, 3)
        cv2.line(annotated, right_hip, right_knee, orange, 3)

        # Línea amarilla central eliminada (no se dibuja)


        # Ángulos clínicos de rodilla (valgo/varo): entre eje fémur (hip-knee) y eje tibia (knee-ankle)
        from utils.geometry import angle_between_points
        results = {}
        for side, hip, knee, ankle in [
            ("left", left_hip, left_knee, left_ankle),
            ("right", right_hip, right_knee, right_ankle)
        ]:
            angle = angle_between_points(hip, knee, ankle)
            angle_clinical = 180 - angle
            results[f"knee_angle_{side}"] = angle_clinical
            # Draw angle label near knee
            angle_text = f"{angle_clinical:.1f}°"
            (tw, th), _ = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            text_pos = (knee[0] - tw//2, knee[1] + 40)
            cv2.rectangle(annotated, (text_pos[0] - 8, text_pos[1] - th - 8), (text_pos[0] + tw + 8, text_pos[1] + 8), green, -1)
            cv2.putText(annotated, angle_text, (text_pos[0], text_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)


        # Ángulo de la entrepierna (apertura): entre ambas rodillas, vértice en la entrepierna
        groin_angle = angle_between_points(left_knee, crotch, right_knee)
        groin_angle = 180 - groin_angle
        results["groin_angle"] = groin_angle
        angle_text = f"{groin_angle:.1f}°"
        (tw, th), _ = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        text_pos = (crotch[0] - tw//2, crotch[1] - 24)
        cv2.rectangle(annotated, (text_pos[0] - 10, text_pos[1] - th - 10), (text_pos[0] + tw + 10, text_pos[1] + 10), yellow, -1)
        cv2.putText(annotated, angle_text, (text_pos[0], text_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3)


        # (Ya no hay bloque duplicado, solo la versión correcta arriba)
        return {"annotated": annotated, **results}

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

        if plane == "sagital":
            # Solo análisis limpio de rodilla sagital
            result = self.draw_sagittal_knee_angle(image, side)
            return {
                "metrics": {
                    "plane": plane,
                    "side": side,
                    "knee_angle_deg": result["sagittal_angle"],
                    "classification": self.classify(result["sagittal_angle"], plane) if result["sagittal_angle"] is not None else "No detectado",
                },
                "images": {
                    "annotated": result["annotated"],
                },
            }

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

        # --- Q angle overlay ---
        q_result = self.draw_q_angle(image)


        # Línea ojo a ojo (cyan)
        if left_eye and right_eye:
            cv2.line(annotated, left_eye, right_eye, (255, 255, 0), 3)

        # Puntos anatómicos extra
        if left_shoulder:
            cv2.circle(annotated, left_shoulder, 7, (0, 255, 255), -1)
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
                "q_angle_left": q_result.get("q_angle_left"),
                "q_angle_right": q_result.get("q_angle_right"),
            },
            "images": {
                "annotated": q_result["annotated"],
            },
        }

    def draw_sagittal_knee_angle(self, image: np.ndarray, side: str = "right") -> dict:
        """
        Dibuja el ángulo sagital de la rodilla (vista lateral) sobre la imagen.
        Solo usa cadera, rodilla y tobillo del lado especificado ("right" o "left").
        Devuelve imagen anotada y ángulo.
        """
        annotated = image.copy()

        try:
            landmarks = self.detector.detect(image)
        except Exception:
            cv2.putText(
                annotated,
                "No landmarks detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            return {"annotated": annotated, "sagittal_angle": None}

        # Obtener puntos
        hip = (int(landmarks[f"{side}_hip"].x), int(landmarks[f"{side}_hip"].y))
        knee = (int(landmarks[f"{side}_knee"].x), int(landmarks[f"{side}_knee"].y))
        ankle = (int(landmarks[f"{side}_ankle"].x), int(landmarks[f"{side}_ankle"].y))

        # Colores
        line_color = (255, 140, 0)   # naranja
        point_color = (0, 0, 0)
        angle_bg = (0, 255, 255)     # amarillo fuerte para máximo contraste

        # SOLO 3 puntos
        for pt in [hip, knee, ankle]:
            cv2.circle(annotated, pt, 6, point_color, -1)

        # SOLO 2 líneas
        cv2.line(annotated, hip, knee, line_color, 3)
        cv2.line(annotated, knee, ankle, line_color, 3)

        # Cálculo del ángulo
        from utils.geometry import angle_between_points
        angle = angle_between_points(hip, knee, ankle)

        # Dibujo del arco del ángulo (media luna) corregido
        import math
        def vector_angle_deg(v):
            return math.degrees(math.atan2(v[1], v[0]))

        v1 = (hip[0] - knee[0], hip[1] - knee[1])
        v2 = (ankle[0] - knee[0], ankle[1] - knee[1])

        a1 = (vector_angle_deg(v1) + 360) % 360
        a2 = (vector_angle_deg(v2) + 360) % 360

        diff = abs(a1 - a2)

        # Elegir ángulo interno
        if diff > 180:
            start_angle = max(a1, a2)
            end_angle = min(a1, a2) + 360
        else:
            start_angle = min(a1, a2)
            end_angle = max(a1, a2)

        arc_radius = 35
        arc_color = (255, 180, 0)  # azulito/amarillo
        thickness = 3

        cv2.ellipse(
            annotated,
            (knee[0], knee[1]),
            (arc_radius, arc_radius),
            0,
            start_angle,
            end_angle,
            arc_color,
            thickness,
            lineType=cv2.LINE_AA
        )

        # Posición y fondo del texto (centrado sobre la rodilla, fondo más pequeño y discreto)
        angle_text = f"{angle:.1f}°"
        (tw, th), _ = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
        text_pos = (knee[0] - tw // 2, knee[1] - 24)
        cv2.rectangle(
            annotated,
            (text_pos[0] - 4, text_pos[1] - th - 4),
            (text_pos[0] + tw + 4, text_pos[1] + th + 4),
            angle_bg,
            -1
        )
        cv2.putText(
            annotated,
            angle_text,
            (text_pos[0], text_pos[1] + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            1
        )

        try:
            landmarks = self.detector.detect(image)
        except Exception:
            cv2.putText(
                annotated,
                "No landmarks detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            return {"annotated": annotated, "sagittal_angle": None}

        # Obtener puntos
        hip = (int(landmarks[f"{side}_hip"].x), int(landmarks[f"{side}_hip"].y))
        knee = (int(landmarks[f"{side}_knee"].x), int(landmarks[f"{side}_knee"].y))
        ankle = (int(landmarks[f"{side}_ankle"].x), int(landmarks[f"{side}_ankle"].y))

        # Colores
        line_color = (255, 140, 0)   # naranja
        point_color = (0, 0, 0)
        angle_bg = (0, 255, 255)     # amarillo fuerte para máximo contraste

        # SOLO 3 puntos
        for pt in [hip, knee, ankle]:
            cv2.circle(annotated, pt, 6, point_color, -1)

        # SOLO 2 líneas
        cv2.line(annotated, hip, knee, line_color, 3)
        cv2.line(annotated, knee, ankle, line_color, 3)

        # Cálculo del ángulo
        from utils.geometry import angle_between_points
        angle = angle_between_points(hip, knee, ankle)

        # Posición y fondo del texto (centrado sobre la rodilla, fondo más grande y visible)
        angle_text = f"{angle:.1f}°"
        (tw, th), _ = cv2.getTextSize(angle_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        text_pos = (knee[0] - tw // 2, knee[1] - 40)
        # Fondo más grande y visible
        cv2.rectangle(
            annotated,
            (text_pos[0] - 16, text_pos[1] - th - 16),
            (text_pos[0] + tw + 16, text_pos[1] + th + 16),
            angle_bg,
            -1
        )
        # Texto negro, grande y centrado
        cv2.putText(
            annotated,
            angle_text,
            (text_pos[0], text_pos[1] + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            3
        )

        return {"annotated": annotated, "sagittal_angle": angle}
