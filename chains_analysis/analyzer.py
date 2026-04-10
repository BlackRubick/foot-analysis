from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from chains_analysis.detector import BodyDetection, BodyLandmarkDetector
from chains_analysis.measurements import angle, distance, lerp, midpoint, point_xy, signed_horizontal_offset, to_mm
from chains_analysis.models import AnalysisResult, Calibration, ChainSummary, FeatureResult, LandmarkPoint
from chains_analysis.rules import CHAIN_LABELS, CHAIN_ORDER, FeatureRule, evaluate_rule, rules_for_plane


def _safe_point(values: Dict[str, LandmarkPoint], key: str) -> Tuple[float, float]:
    point = point_xy(values.get(key))
    if point is None:
        raise ValueError(f"Falta el landmark requerido: {key}")
    return point


def _fallback_midpoint(values: Dict[str, LandmarkPoint], left_key: str, right_key: str) -> Tuple[float, float]:
    left = point_xy(values.get(left_key))
    right = point_xy(values.get(right_key))
    if left is None and right is None:
        raise ValueError(f"No se pudo obtener el par {left_key}/{right_key}")
    if left is None:
        return right
    if right is None:
        return left
    return midpoint(left, right)


@dataclass(frozen=True)
class AnatomicalProxies:
    side: str
    anterior_sign: int
    c7: Tuple[float, float]
    xiphoid: Tuple[float, float]
    d7: Tuple[float, float]
    l1: Tuple[float, float]
    sacrum: Tuple[float, float]
    plumb_x: float
    mid_shoulder: Tuple[float, float]
    mid_hip: Tuple[float, float]
    mid_head: Tuple[float, float]
    left_asis: Tuple[float, float]
    right_asis: Tuple[float, float]
    left_psis: Tuple[float, float]
    right_psis: Tuple[float, float]


class MuscleChainAnalyzer:
    def __init__(self, calibration: Calibration | None = None):
        self.detector = BodyLandmarkDetector()
        self.calibration = calibration or Calibration()

    @staticmethod
    def _select_side(pose: Dict[str, LandmarkPoint]) -> str:
        left_vis = sum(pose[name].visibility for name in ("left_ear", "left_shoulder", "left_hip") if name in pose)
        right_vis = sum(pose[name].visibility for name in ("right_ear", "right_shoulder", "right_hip") if name in pose)
        return "left" if left_vis >= right_vis else "right"

    @staticmethod
    def _infer_anterior_sign(pose: Dict[str, LandmarkPoint], side: str) -> int:
        ear = point_xy(pose.get(f"{side}_ear"))
        nose = point_xy(pose.get("nose"))
        if ear is None or nose is None:
            return 1
        if side == "left":
            return 1 if nose[0] >= ear[0] else -1
        return 1 if nose[0] <= ear[0] else -1

    def _build_proxies(self, detection: BodyDetection, plane: str, profile_side: str) -> AnatomicalProxies:
        pose = detection.pose
        side = profile_side if profile_side in {"left", "right"} else self._select_side(pose)
        anterior_sign = self._infer_anterior_sign(pose, side)

        left_shoulder = _safe_point(pose, "left_shoulder")
        right_shoulder = _safe_point(pose, "right_shoulder")
        left_hip = _safe_point(pose, "left_hip")
        right_hip = _safe_point(pose, "right_hip")
        left_ankle = _safe_point(pose, "left_ankle")
        right_ankle = _safe_point(pose, "right_ankle")

        mid_shoulder = midpoint(left_shoulder, right_shoulder)
        mid_hip = midpoint(left_hip, right_hip)
        mid_head = _fallback_midpoint(pose, "left_ear", "right_ear") if "left_ear" in pose and "right_ear" in pose else _safe_point(pose, "nose")

        torso = max(distance(mid_shoulder, mid_hip), 1.0)
        body_width = max(distance(left_shoulder, right_shoulder), distance(left_hip, right_hip), 1.0)

        c7 = lerp(mid_shoulder, mid_head, 0.45)
        xiphoid = lerp(mid_shoulder, mid_hip, 0.28)
        d7 = lerp(mid_shoulder, mid_hip, 0.42)
        l1 = lerp(mid_shoulder, mid_hip, 0.70)
        sacrum = lerp(mid_hip, mid_shoulder, -0.12)

        if plane.lower() == "sagittal":
            ankle_ref = midpoint(left_ankle, right_ankle)
            plumb_x = ankle_ref[0]
        else:
            plumb_x = mid_shoulder[0]

        posterior_shift = -0.04 * body_width * anterior_sign
        anterior_shift = 0.04 * body_width * anterior_sign

        left_asis = (left_hip[0] + anterior_shift, left_hip[1] + 0.02 * torso)
        right_asis = (right_hip[0] + anterior_shift, right_hip[1] + 0.02 * torso)
        left_psis = (left_hip[0] + posterior_shift, left_hip[1] - 0.02 * torso)
        right_psis = (right_hip[0] + posterior_shift, right_hip[1] - 0.02 * torso)

        return AnatomicalProxies(
            side=side,
            anterior_sign=anterior_sign,
            c7=c7,
            xiphoid=xiphoid,
            d7=d7,
            l1=l1,
            sacrum=sacrum,
            plumb_x=plumb_x,
            mid_shoulder=mid_shoulder,
            mid_hip=mid_hip,
            mid_head=mid_head,
            left_asis=left_asis,
            right_asis=right_asis,
            left_psis=left_psis,
            right_psis=right_psis,
        )

    def _hand_rotation_score(self, detection: BodyDetection, side: str) -> float:
        hand = detection.left_hand if side == "left" else detection.right_hand
        if not hand:
            return 0.0
        thumb_tip = point_xy(hand.get("thumb_tip"))
        pinky_tip = point_xy(hand.get("pinky_tip"))
        if thumb_tip is None or pinky_tip is None:
            return 0.0
        return thumb_tip[0] - pinky_tip[0]

    def _lower_limb_rotation_score(self, pose: Dict[str, LandmarkPoint], side: str) -> float:
        heel = point_xy(pose.get(f"{side}_heel"))
        toe = point_xy(pose.get(f"{side}_foot_index"))
        if heel is None or toe is None:
            return 0.0
        return toe[0] - heel[0]

    @staticmethod
    def estimate_aruco_calibration(image: np.ndarray, marker_length_mm: float, dictionary_name: str = "DICT_4X4_50") -> Calibration:
        if marker_length_mm <= 0:
            raise ValueError("La longitud real del marcador ArUco debe ser mayor que cero")

        if not hasattr(cv2, "aruco"):
            raise RuntimeError("Esta instalación de OpenCV no incluye el módulo aruco")

        aruco = cv2.aruco
        dictionary_id = getattr(aruco, dictionary_name, aruco.DICT_4X4_50)
        dictionary = aruco.getPredefinedDictionary(dictionary_id)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if hasattr(aruco, "ArucoDetector"):
            detector = aruco.ArucoDetector(dictionary)
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = aruco.detectMarkers(gray, dictionary)

        if ids is None or len(corners) == 0:
            raise ValueError("No se detectó un marcador ArUco en la imagen")

        best_corners = max(corners, key=lambda c: cv2.arcLength(c.reshape(-1, 2).astype(np.float32), True))
        pts = best_corners.reshape(-1, 2)
        side_lengths = [
            np.linalg.norm(pts[0] - pts[1]),
            np.linalg.norm(pts[1] - pts[2]),
            np.linalg.norm(pts[2] - pts[3]),
            np.linalg.norm(pts[3] - pts[0]),
        ]
        marker_side_px = float(np.mean(side_lengths))
        return Calibration(mm_per_px=marker_length_mm / marker_side_px, source="aruco")

    def _compute_metrics(self, detection: BodyDetection, proxies: AnatomicalProxies, plane: str) -> Dict[str, float]:
        pose = detection.pose
        side = proxies.side

        hip = _safe_point(pose, f"{side}_hip")
        knee = _safe_point(pose, f"{side}_knee")
        ankle = _safe_point(pose, f"{side}_ankle")
        shoulder = _safe_point(pose, f"{side}_shoulder")
        ear = _safe_point(pose, f"{side}_ear")
        mouth_left = point_xy(pose.get("mouth_left")) or ear
        mouth_right = point_xy(pose.get("mouth_right")) or ear

        left_shoulder = _safe_point(pose, "left_shoulder")
        right_shoulder = _safe_point(pose, "right_shoulder")
        left_wrist = point_xy(pose.get("left_wrist"))
        right_wrist = point_xy(pose.get("right_wrist"))
        left_foot = point_xy(pose.get("left_foot_index"))
        right_foot = point_xy(pose.get("right_foot_index"))

        knee_angle_deg = angle(hip, knee, ankle)
        q_angle_deg = angle(_safe_point(pose, f"{side}_hip"), knee, ankle)

        charpy_angle_deg = angle(left_shoulder, proxies.xiphoid, right_shoulder)
        clavicular_opening_angle_deg = angle(left_shoulder, proxies.c7, right_shoulder)
        cervical_angle_deg = angle(ear, proxies.c7, proxies.mid_hip)
        craniovertebral_angle_deg = angle((proxies.c7[0] + 100.0, proxies.c7[1]), proxies.c7, ear)
        mandibular_angle_deg = angle(mouth_left, point_xy(pose.get("nose")) or proxies.mid_head, mouth_right)

        d7_offset_mm = signed_horizontal_offset(proxies.d7, proxies.plumb_x, self.calibration.mm_per_px, proxies.anterior_sign)
        l1_offset_mm = signed_horizontal_offset(proxies.l1, proxies.plumb_x, self.calibration.mm_per_px, proxies.anterior_sign)
        sacrum_offset_mm = signed_horizontal_offset(proxies.sacrum, proxies.plumb_x, self.calibration.mm_per_px, proxies.anterior_sign)
        thoracic_curve_mm = d7_offset_mm - l1_offset_mm
        lumbar_curve_mm = l1_offset_mm - sacrum_offset_mm

        shoulder_drop_mm = to_mm(max(0.0, proxies.mid_shoulder[1] - proxies.mid_head[1]), self.calibration.mm_per_px)
        head_barre_offset_mm = signed_horizontal_offset(proxies.mid_head, proxies.plumb_x, self.calibration.mm_per_px, proxies.anterior_sign)
        shoulder_barre_offset_mm = signed_horizontal_offset(proxies.mid_shoulder, proxies.plumb_x, self.calibration.mm_per_px, proxies.anterior_sign)
        hip_barre_offset_mm = signed_horizontal_offset(proxies.mid_hip, proxies.plumb_x, self.calibration.mm_per_px, proxies.anterior_sign)
        knee_barre_offset_mm = signed_horizontal_offset(knee, proxies.plumb_x, self.calibration.mm_per_px, proxies.anterior_sign)
        ankle_barre_offset_mm = signed_horizontal_offset(ankle, proxies.plumb_x, self.calibration.mm_per_px, proxies.anterior_sign)
        arm_midline_distance_mm = 0.0
        counted = 0
        if left_wrist is not None:
            arm_midline_distance_mm += abs(left_wrist[0] - proxies.mid_shoulder[0])
            counted += 1
        if right_wrist is not None:
            arm_midline_distance_mm += abs(right_wrist[0] - proxies.mid_shoulder[0])
            counted += 1
        if counted:
            arm_midline_distance_mm = to_mm(arm_midline_distance_mm / counted, self.calibration.mm_per_px)

        arm_rotation_score = self._hand_rotation_score(detection, side)
        hip_rotation_score = self._lower_limb_rotation_score(pose, side)

        pelvic_tilt_mm = sacrum_offset_mm
        foot_orientation_score = 0.0
        if left_foot is not None and right_foot is not None:
            foot_orientation_score = (left_foot[0] - right_foot[0]) * self.calibration.mm_per_px

        return {
            "knee_angle_deg": knee_angle_deg,
            "q_angle_deg": q_angle_deg,
            "charpy_angle_deg": charpy_angle_deg,
            "clavicular_opening_angle_deg": clavicular_opening_angle_deg,
            "cervical_angle_deg": cervical_angle_deg,
            "craniovertebral_angle_deg": craniovertebral_angle_deg,
            "mandibular_angle_deg": mandibular_angle_deg,
            "d7_offset_mm": d7_offset_mm,
            "l1_offset_mm": l1_offset_mm,
            "sacrum_offset_mm": sacrum_offset_mm,
            "thoracic_curve_mm": thoracic_curve_mm,
            "lumbar_curve_mm": lumbar_curve_mm,
            "pelvic_tilt_mm": pelvic_tilt_mm,
            "shoulder_drop_mm": shoulder_drop_mm,
            "head_barre_offset_mm": head_barre_offset_mm,
            "shoulder_barre_offset_mm": shoulder_barre_offset_mm,
            "hip_barre_offset_mm": hip_barre_offset_mm,
            "knee_barre_offset_mm": knee_barre_offset_mm,
            "ankle_barre_offset_mm": ankle_barre_offset_mm,
            "arm_midline_distance_mm": arm_midline_distance_mm,
            "arm_rotation_score": arm_rotation_score,
            "hip_rotation_score": hip_rotation_score,
            "foot_orientation_score": foot_orientation_score,
        }

    def _evaluate(self, metrics: Dict[str, float], rules: List[FeatureRule]) -> List[FeatureResult]:
        results: List[FeatureResult] = []
        for rule in rules:
            value = float(metrics.get(rule.metric, 0.0))
            present = evaluate_rule(value, rule)
            results.append(
                FeatureResult(
                    code=rule.code,
                    label=rule.label,
                    chain=rule.chain,
                    present=present,
                    value=value,
                    unit="deg" if "angle" in rule.metric else "mm",
                    rule=f"{rule.metric} {rule.operator} {rule.threshold}",
                    note=rule.note,
                )
            )
        return results

    def _summarize(self, feature_results: List[FeatureResult]) -> Dict[str, ChainSummary]:
        by_chain: Dict[str, List[FeatureResult]] = {}
        for result in feature_results:
            by_chain.setdefault(result.chain, []).append(result)

        activation_by_chain: Dict[str, float] = {}
        for chain, items in by_chain.items():
            positives = sum(1 for item in items if item.present)
            total = len(items)
            activation_by_chain[chain] = (positives / total * 100.0) if total else 0.0

        activation_sum = float(sum(activation_by_chain.values()))
        summaries: Dict[str, ChainSummary] = {}
        for chain in CHAIN_ORDER:
            items = by_chain.get(chain, [])
            positives = sum(1 for item in items if item.present)
            total = len(items)
            activation_percentage = activation_by_chain.get(chain, 0.0)
            prevalence_percentage = (activation_percentage / activation_sum * 100.0) if activation_sum > 0 else 0.0
            summaries[chain] = ChainSummary(
                name=CHAIN_LABELS.get(chain, chain),
                positives=positives,
                total=total,
                activation_percentage=activation_percentage,
                percentage=prevalence_percentage,
            )
        return summaries

    def _draw_point(self, image: np.ndarray, point: Tuple[float, float], color: Tuple[int, int, int], label: str) -> None:
        cv2.circle(image, (int(point[0]), int(point[1])), 5, color, -1)
        cv2.putText(image, label, (int(point[0]) + 6, int(point[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    def _render_overlay(self, image: np.ndarray, detection: BodyDetection, proxies: AnatomicalProxies, metrics: Dict[str, float], feature_results: List[FeatureResult], chain_summaries: Dict[str, ChainSummary], plane: str) -> np.ndarray:
        annotated = image.copy()
        pose = detection.pose

        skeleton_pairs = [
            ("left_ear", "left_shoulder"),
            ("right_ear", "right_shoulder"),
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            ("left_hip", "left_knee"),
            ("right_hip", "right_knee"),
            ("left_knee", "left_ankle"),
            ("right_knee", "right_ankle"),
        ]
        for a_name, b_name in skeleton_pairs:
            a = point_xy(pose.get(a_name))
            b = point_xy(pose.get(b_name))
            if a is None or b is None:
                continue
            cv2.line(annotated, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (0, 165, 255), 2)

        key_points = {
            "C7": proxies.c7,
            "Xif": proxies.xiphoid,
            "D7": proxies.d7,
            "L1": proxies.l1,
            "S": proxies.sacrum,
            "ASIS-I": proxies.left_asis,
            "ASIS-D": proxies.right_asis,
            "PSIS-I": proxies.left_psis,
            "PSIS-D": proxies.right_psis,
        }
        for label, point in key_points.items():
            self._draw_point(annotated, point, (0, 255, 0), label)

        cv2.line(annotated, (int(proxies.plumb_x), 0), (int(proxies.plumb_x), annotated.shape[0] - 1), (255, 255, 0), 2)

        y = 28
        cv2.putText(annotated, f"Plano: {plane} | lado: {proxies.side}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 28
        for metric_name in ["knee_angle_deg", "q_angle_deg", "charpy_angle_deg", "cervical_angle_deg", "craniovertebral_angle_deg", "mandibular_angle_deg"]:
            cv2.putText(annotated, f"{metric_name}: {metrics[metric_name]:.1f}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            y += 22

        y += 6
        for chain_key in CHAIN_ORDER:
            summary = chain_summaries.get(chain_key)
            if summary is None:
                continue
            cv2.putText(
                annotated,
                f"{summary.name}: {summary.percentage:.1f}% prev | {summary.activation_percentage:.1f}% act",
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
            )
            y += 24

        present_labels = [result.label for result in feature_results if result.present]
        for index, label in enumerate(present_labels[:6]):
            cv2.putText(annotated, f"+ {label}", (20, y + index * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return annotated

    def analyze(
        self,
        image: np.ndarray,
        plane: str = "sagittal",
        profile_side: str = "auto",
        calibration: Calibration | None = None,
    ) -> AnalysisResult:
        if calibration is not None:
            self.calibration = calibration
        detection = self.detector.detect(image)
        proxies = self._build_proxies(detection, plane=plane, profile_side=profile_side)
        rules = rules_for_plane(plane)
        metrics = self._compute_metrics(detection, proxies, plane)
        feature_results = self._evaluate(metrics, rules)
        chain_summaries = self._summarize(feature_results)
        annotated = self._render_overlay(image, detection, proxies, metrics, feature_results, chain_summaries, plane)

        if chain_summaries:
            best_key = max(chain_summaries, key=lambda key: chain_summaries[key].percentage)
            best = chain_summaries[best_key]
            if all(abs(summary.percentage - best.percentage) <= 5.0 for summary in chain_summaries.values()):
                notes = ["Equilibrio corporal: las cadenas estan dentro de un margen cercano."]
            else:
                notes = [f"Cadena predominante: {best.name} ({best.percentage:.1f}%)."]
        else:
            notes = ["No se pudo calcular la predominancia de cadenas."]

        metrics.update(
            {
                "plane": plane,
                "profile_side": proxies.side,
                "calibration_mm_per_px": self.calibration.mm_per_px,
                "calibration_source": self.calibration.source,
            }
        )

        return AnalysisResult(
            metrics=metrics,
            feature_results=feature_results,
            chain_summaries=chain_summaries,
            images={"annotated": annotated},
            notes=notes,
        )