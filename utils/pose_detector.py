from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Dict

import cv2
import mediapipe as mp
import numpy as np


POSE_LANDMARK_INDEX = {
    "left_ear": 7,
    "right_ear": 8,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


@dataclass
class SimpleLandmark:
    x: float
    y: float
    visibility: float = 1.0


class PoseDetector:
    """Compatibilidad entre MediaPipe Solutions (legacy) y Tasks API."""

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"

    def __init__(self, model_dir: str = "models"):
        self.backend = "solutions" if hasattr(mp, "solutions") else "tasks"
        self._pose = None
        self._landmarker = None

        if self.backend == "solutions":
            self._pose = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=1)
        else:
            model_path = self._ensure_task_model(model_dir)
            base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
            vision = mp.tasks.vision
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def _ensure_task_model(self, model_dir: str) -> str:
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "pose_landmarker_full.task")
        if os.path.exists(model_path):
            return model_path

        try:
            urllib.request.urlretrieve(self.MODEL_URL, model_path)
        except Exception as exc:
            raise RuntimeError(
                "No se pudo descargar el modelo de pose para MediaPipe Tasks. "
                "Verifica conexión a internet o descarga manualmente el archivo .task en la carpeta models/."
            ) from exc

        return model_path

    def detect(self, image_bgr: np.ndarray) -> Dict[str, SimpleLandmark]:
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        if self.backend == "solutions":
            results = self._pose.process(rgb)
            if not results.pose_landmarks:
                raise ValueError("No se detectaron landmarks corporales")

            lms = results.pose_landmarks.landmark
            return {
                name: SimpleLandmark(
                    x=float(lms[idx].x * w),
                    y=float(lms[idx].y * h),
                    visibility=float(getattr(lms[idx], "visibility", 1.0)),
                )
                for name, idx in POSE_LANDMARK_INDEX.items()
            }

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.pose_landmarks:
            raise ValueError("No se detectaron landmarks corporales")

        lms = result.pose_landmarks[0]
        return {
            name: SimpleLandmark(
                x=float(lms[idx].x * w),
                y=float(lms[idx].y * h),
                visibility=float(getattr(lms[idx], "visibility", 1.0)),
            )
            for name, idx in POSE_LANDMARK_INDEX.items()
        }
