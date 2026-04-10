from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class FeatureRule:
    code: str
    label: str
    chain: str
    metric: str
    operator: str
    threshold: float
    plane: str = "any"
    note: str = ""


def _compare(value: float, operator: str, threshold: float) -> bool:
    if operator == "lt":
        return value < threshold
    if operator == "le":
        return value <= threshold
    if operator == "gt":
        return value > threshold
    if operator == "ge":
        return value >= threshold
    if operator == "abs_lt":
        return abs(value) < threshold
    if operator == "abs_le":
        return abs(value) <= threshold
    if operator == "abs_gt":
        return abs(value) > threshold
    if operator == "abs_ge":
        return abs(value) >= threshold
    raise ValueError(f"Operador no soportado: {operator}")


FEATURE_RULES: List[FeatureRule] = [
    FeatureRule("genu_flexum", "Genu flexum", "flexion", "knee_angle_deg", "lt", 175.0, "sagittal"),
    FeatureRule("genu_recurvatum", "Genu recurvatum", "extension", "knee_angle_deg", "gt", 185.0, "sagittal"),
    FeatureRule("sacro_vertical", "Sacro vertical", "flexion", "sacrum_offset_mm", "abs_le", 5.0, "sagittal"),
    FeatureRule("sacro_horizontal", "Sacro horizontal", "extension", "sacrum_offset_mm", "abs_ge", 20.0, "sagittal"),
    FeatureRule("cifosis", "Cifosis", "flexion", "thoracic_curve_mm", "gt", 10.0, "sagittal"),
    FeatureRule("dorso_plano", "Dorso plano", "extension", "thoracic_curve_mm", "abs_le", 5.0, "sagittal"),
    FeatureRule("pectus_excavatum", "Pectus excavatum", "flexion", "charpy_angle_deg", "lt", 80.0, "frontal"),
    FeatureRule("cierre_costillas", "Cierre de costillas", "flexion", "charpy_angle_deg", "lt", 90.0, "frontal"),
    FeatureRule("apertura_costillas", "Apertura de costillas", "extension", "charpy_angle_deg", "gt", 90.0, "frontal"),
    FeatureRule("inversion_cervical", "Inversion cervical", "flexion", "cervical_angle_deg", "lt", 155.0, "sagittal"),
    FeatureRule("rectificacion_cervical", "Rectificacion cervical", "extension", "cervical_angle_deg", "gt", 170.0, "sagittal"),
    FeatureRule("proyeccion_anterior_cabeza", "Proyeccion anterior de cabeza", "flexion", "craniovertebral_angle_deg", "lt", 50.0, "sagittal"),
    FeatureRule("bascula_posterior_cabeza", "Bascula posterior de cabeza", "extension", "craniovertebral_angle_deg", "gt", 60.0, "sagittal"),
    FeatureRule("retroversion_pelvica", "Retroversion pelvica", "flexion", "pelvic_tilt_mm", "lt", -10.0, "sagittal"),
    FeatureRule("anteversion_pelvica", "Anteversion pelvica", "extension", "pelvic_tilt_mm", "gt", 10.0, "sagittal"),
    FeatureRule("coxis_hacia_adentro", "Coxis hacia adentro", "flexion", "sacrum_offset_mm", "lt", -6.0, "sagittal"),
    FeatureRule("rectificacion_lumbar", "Rectificacion lumbar", "flexion", "lumbar_curve_mm", "abs_le", 5.0, "sagittal"),
    FeatureRule("hiperlordosis", "Hiperlordosis", "extension", "lumbar_curve_mm", "gt", 10.0, "sagittal"),
    FeatureRule("valgo_rodilla", "Valgo de rodilla", "flexion", "q_angle_deg", "lt", 170.0, "frontal"),
    FeatureRule("varo_rodilla", "Varo de rodilla", "extension", "q_angle_deg", "gt", 180.0, "frontal"),
    FeatureRule("rotacion_interna_cadera", "Rotacion interna de cadera", "flexion", "hip_rotation_score", "lt", -5.0, "frontal"),
    FeatureRule("rotacion_externa", "Rotacion externa", "extension", "hip_rotation_score", "gt", 5.0, "frontal"),
    FeatureRule("aduccion_brazos", "Aduccion de brazos", "flexion", "arm_midline_distance_mm", "lt", 90.0, "frontal"),
    FeatureRule("abduccion_brazos", "Abduccion de brazos", "extension", "arm_midline_distance_mm", "gt", 120.0, "frontal"),
    FeatureRule("rotacion_interna_brazos", "Rotacion interna de brazos", "flexion", "arm_rotation_score", "lt", -5.0, "frontal"),
    FeatureRule("rotacion_externa_brazos", "Rotacion externa de brazos", "extension", "arm_rotation_score", "gt", 5.0, "frontal"),
    FeatureRule("descenso_hombros", "Descenso de hombros", "flexion", "shoulder_drop_mm", "gt", 20.0, "any"),
    FeatureRule("ascenso_hombros", "Ascenso de hombros", "extension", "shoulder_drop_mm", "lt", 10.0, "any"),
    FeatureRule("elevacion_esternoclavicular", "Elevacion esternoclavicular", "flexion", "clavicular_opening_angle_deg", "lt", 65.0, "frontal"),
    FeatureRule("esternon_horizontal", "Esternon horizontal", "extension", "clavicular_opening_angle_deg", "gt", 95.0, "frontal"),
    FeatureRule("apertura_mandibula", "Apertura de mandibula", "extension", "mandibular_angle_deg", "gt", 100.0, "frontal"),
    FeatureRule("cierre_mandibula", "Cierre de mandibula", "flexion", "mandibular_angle_deg", "lt", 80.0, "frontal"),
    FeatureRule("hipercifosis", "Hipercifosis", "flexion", "d7_offset_mm", "gt", 60.0, "sagittal"),
    FeatureRule("espalda_plana", "Espalda plana", "extension", "d7_offset_mm", "lt", 20.0, "sagittal"),
    FeatureRule("rectificacion_lumbar_baja", "Rectificacion lumbar", "flexion", "l1_offset_mm", "lt", 20.0, "sagittal"),
    FeatureRule("hiperlordosis_baja", "Hiperlordosis baja", "extension", "l1_offset_mm", "gt", 45.0, "sagittal"),
    FeatureRule("flexion_msls", "Flexion de MMII", "flexion", "knee_angle_deg", "lt", 175.0, "sagittal"),
    FeatureRule("knee_posteriorizada_flexion", "Rodilla posteriorizada", "flexion", "knee_barre_offset_mm", "lt", -5.0, "sagittal"),
    FeatureRule("ankle_posteriorizado_flexion", "Tobillo posteriorizado", "flexion", "ankle_barre_offset_mm", "lt", -5.0, "sagittal"),
    FeatureRule("extension_msls", "Extension de MMII", "extension", "knee_angle_deg", "gt", 185.0, "sagittal"),
    FeatureRule("knee_antepulsion_extension", "Rodilla en antepulsion", "extension", "knee_barre_offset_mm", "gt", 5.0, "sagittal"),
    FeatureRule("ankle_antepulsion_extension", "Tobillo en antepulsion", "extension", "ankle_barre_offset_mm", "gt", 5.0, "sagittal"),

    # Cadenas complementarias (arquitectura de 6 cadenas)
    # Para la cadena de apertura/cierre costal global usamos el umbral clinico de 90°:
    # - angulo Charpy > 90°  -> apertura de costillas
    # - angulo Charpy < 90°  -> cierre de costillas
    FeatureRule("apertura_costal_global", "Apertura costal global", "apertura", "charpy_angle_deg", "gt", 90.0, "frontal"),
    FeatureRule("apertura_brazos_global", "Abduccion global de MMSS", "apertura", "arm_midline_distance_mm", "gt", 130.0, "frontal"),
    FeatureRule("cierre_costal_global", "Cierre costal global", "cierre", "charpy_angle_deg", "lt", 90.0, "frontal"),
    FeatureRule("cierre_brazos_global", "Aduccion global de MMSS", "cierre", "arm_midline_distance_mm", "lt", 85.0, "frontal"),
    FeatureRule("inspiracion_global", "Patron inspiratorio", "inspiracion", "thoracic_curve_mm", "gt", 15.0, "sagittal"),
    FeatureRule("mandibula_abierta_global", "Apertura mandibular global", "inspiracion", "mandibular_angle_deg", "gt", 100.0, "frontal"),
    FeatureRule("espiracion_global", "Patron espiratorio", "espiracion", "thoracic_curve_mm", "lt", 8.0, "sagittal"),
    FeatureRule("mandibula_cerrada_global", "Cierre mandibular global", "espiracion", "mandibular_angle_deg", "lt", 80.0, "frontal"),
]

CHAIN_ORDER = ["flexion", "extension", "apertura", "cierre", "inspiracion", "espiracion"]
CHAIN_LABELS = {
    "flexion": "Cadena de flexion",
    "extension": "Cadena de extension",
    "apertura": "Cadena de apertura",
    "cierre": "Cadena de cierre",
    "inspiracion": "Cadena inspiratoria",
    "espiracion": "Cadena espiratoria",
}


def evaluate_rule(value: float, rule: FeatureRule) -> bool:
    return _compare(value, rule.operator, rule.threshold)


def rules_for_plane(plane: str) -> List[FeatureRule]:
    plane = plane.lower()
    return [rule for rule in FEATURE_RULES if rule.plane in {"any", plane}]


def chain_totals(rules: Iterable[FeatureRule]) -> Dict[str, int]:
    totals: Dict[str, int] = {name: 0 for name in CHAIN_ORDER}
    for rule in rules:
        totals[rule.chain] = totals.get(rule.chain, 0) + 1
    return totals