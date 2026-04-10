from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class LandmarkPoint:
    x: float
    y: float
    visibility: float = 1.0


@dataclass(frozen=True)
class Calibration:
    mm_per_px: float = 1.0
    source: str = "pixel"

    @classmethod
    def from_reference(cls, reference_mm: float, reference_px: float) -> "Calibration":
        if reference_mm <= 0 or reference_px <= 0:
            raise ValueError("La calibracion por referencia requiere valores positivos")
        return cls(mm_per_px=reference_mm / reference_px, source="reference")

    @classmethod
    def from_height(cls, patient_height_mm: float, pose_height_px: float) -> "Calibration":
        if patient_height_mm <= 0 or pose_height_px <= 0:
            raise ValueError("La calibracion por altura requiere valores positivos")
        return cls(mm_per_px=patient_height_mm / pose_height_px, source="height")

    def convert(self, value_px: float) -> float:
        return value_px * self.mm_per_px


@dataclass(frozen=True)
class FeatureResult:
    code: str
    label: str
    chain: str
    present: bool
    value: float
    unit: str
    rule: str
    note: str = ""


@dataclass(frozen=True)
class ChainSummary:
    name: str
    positives: int
    total: int
    activation_percentage: float
    percentage: float


@dataclass
class AnalysisResult:
    metrics: Dict[str, Any] = field(default_factory=dict)
    feature_results: List[FeatureResult] = field(default_factory=list)
    chain_summaries: Dict[str, ChainSummary] = field(default_factory=dict)
    images: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)