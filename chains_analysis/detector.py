from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import mediapipe as mp
import numpy as np

from chains_analysis.models import LandmarkPoint


POSE_INDEX = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

HAND_NAMES = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_mcp",
    "index_pip",
    "index_dip",
    "index_tip",
    "middle_mcp",
    "middle_pip",
    "middle_dip",
    "middle_tip",
    "ring_mcp",
    "ring_pip",
    "ring_dip",
    "ring_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]


@dataclass(frozen=True)
class BodyDetection:
    pose: Dict[str, LandmarkPoint]
    left_hand: Dict[str, LandmarkPoint]
    right_hand: Dict[str, LandmarkPoint]
    image_width: int
    image_height: int


class BodyLandmarkDetector:
    def __init__(self) -> None:
        self.backend = "holistic" if hasattr(mp.solutions, "holistic") else "pose"
        if self.backend == "holistic":
            self._detector = mp.solutions.holistic.Holistic(
                static_image_mode=True,
                model_complexity=1,
                refine_face_landmarks=False,
            )
        else:
            self._detector = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)

    def _extract_pose(self, landmarks, image_width: int, image_height: int) -> Dict[str, LandmarkPoint]:
        extracted: Dict[str, LandmarkPoint] = {}
        for name, index in POSE_INDEX.items():
            lm = landmarks[index]
            extracted[name] = LandmarkPoint(
                x=float(lm.x * image_width),
                y=float(lm.y * image_height),
                visibility=float(getattr(lm, "visibility", 1.0)),
            )
        return extracted

    def _extract_hand(self, hand_landmarks, image_width: int, image_height: int) -> Dict[str, LandmarkPoint]:
        if hand_landmarks is None:
            return {}
        extracted: Dict[str, LandmarkPoint] = {}
        for name, landmark in zip(HAND_NAMES, hand_landmarks.landmark):
            extracted[name] = LandmarkPoint(
                x=float(landmark.x * image_width),
                y=float(landmark.y * image_height),
                visibility=float(getattr(landmark, "visibility", 1.0)),
            )
        return extracted

    def detect(self, image_bgr: np.ndarray) -> BodyDetection:
        image_height, image_width = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.backend == "holistic":
            results = self._detector.process(rgb)
            if not results.pose_landmarks:
                raise ValueError("No se detectaron landmarks corporales")
            pose = self._extract_pose(results.pose_landmarks.landmark, image_width, image_height)
            left_hand = self._extract_hand(results.left_hand_landmarks, image_width, image_height)
            right_hand = self._extract_hand(results.right_hand_landmarks, image_width, image_height)
            return BodyDetection(pose=pose, left_hand=left_hand, right_hand=right_hand, image_width=image_width, image_height=image_height)

        results = self._detector.process(rgb)
        if not results.pose_landmarks:
            raise ValueError("No se detectaron landmarks corporales")
        pose = self._extract_pose(results.pose_landmarks.landmark, image_width, image_height)
        return BodyDetection(pose=pose, left_hand={}, right_hand={}, image_width=image_width, image_height=image_height)