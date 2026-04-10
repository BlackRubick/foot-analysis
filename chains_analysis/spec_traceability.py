from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set

from chains_analysis.rules import FEATURE_RULES


@dataclass(frozen=True)
class SpecItem:
    area: str
    name: str
    expected_rule_codes: List[str]
    status: str
    notes: str


TARGET_ITEMS: List[SpecItem] = [
    SpecItem("Flexion", "Genu flexum", ["genu_flexum"], "implemented", "Regla directa por angulo de rodilla"),
    SpecItem("Flexion", "Sacro vertical", ["sacro_vertical"], "implemented", "Basado en flecha sacra"),
    SpecItem("Flexion", "Cifosis", ["cifosis", "hipercifosis"], "implemented", "Basado en D7 y curva toracica"),
    SpecItem("Flexion", "Esternon hundido / pectus", ["pectus_excavatum"], "implemented", "Aproximado por angulo de Charpy"),
    SpecItem("Flexion", "Inversion cervical", ["inversion_cervical"], "implemented", "Medicion angular cervical"),
    SpecItem("Flexion", "Retroversion pelvica", ["retroversion_pelvica"], "implemented", "Proxy pelvis/sacro"),
    SpecItem("Flexion", "Coxis hacia adentro", ["coxis_hacia_adentro"], "implemented", "Inferido por flecha sacra"),
    SpecItem("Flexion", "Cierre de costillas", ["cierre_costillas", "cierre_costal_global"], "implemented", "Charpy y patron de cierre"),
    SpecItem("Flexion", "Proyeccion anterior cabeza", ["proyeccion_anterior_cabeza"], "implemented", "Angulo craneovertebral"),
    SpecItem("Flexion", "Rectificacion lumbar", ["rectificacion_lumbar", "rectificacion_lumbar_baja"], "implemented", "Flecha lumbar"),
    SpecItem("Flexion", "Valgo de rodilla", ["valgo_rodilla"], "implemented", "Aproximacion por angulo frontal"),
    SpecItem("Flexion", "Rotacion interna cadera", ["rotacion_interna_cadera"], "implemented", "Proxy por pie-talón"),
    SpecItem("Flexion", "Aduccion/rotacion interna brazos", ["aduccion_brazos", "rotacion_interna_brazos"], "implemented", "Distancia a linea media + mano"),
    SpecItem("Flexion", "Descenso de hombros", ["descenso_hombros"], "implemented", "Aproximacion vertical hombro-cabeza"),
    SpecItem("Flexion", "Flexion MMII", ["flexion_msls", "ankle_posteriorizado_flexion", "knee_posteriorizada_flexion"], "implemented", "Integracion rodilla y Barré"),
    SpecItem("Flexion", "Cierre mandibular", ["cierre_mandibula", "mandibula_cerrada_global"], "implemented", "Angulo mandibular"),

    SpecItem("Extension", "Genu recurvatum", ["genu_recurvatum"], "implemented", "Regla directa por angulo de rodilla"),
    SpecItem("Extension", "Sacro horizontal", ["sacro_horizontal"], "implemented", "Flecha sacra elevada"),
    SpecItem("Extension", "Dorso plano", ["dorso_plano", "espalda_plana"], "implemented", "Flecha dorsal reducida"),
    SpecItem("Extension", "Rectificacion cervical", ["rectificacion_cervical"], "implemented", "Medicion angular cervical"),
    SpecItem("Extension", "Bascula posterior de cabeza", ["bascula_posterior_cabeza"], "implemented", "Angulo craneovertebral alto"),
    SpecItem("Extension", "Esternon horizontal", ["esternon_horizontal"], "implemented", "Proxy por apertura clavicular"),
    SpecItem("Extension", "Apertura mandibular", ["apertura_mandibula", "mandibula_abierta_global"], "implemented", "Angulo mandibular"),
    SpecItem("Extension", "Anteversion pelvica", ["anteversion_pelvica"], "implemented", "Proxy pelvis/sacro"),
    SpecItem("Extension", "Hiperlordosis", ["hiperlordosis", "hiperlordosis_baja"], "implemented", "Flecha lumbar alta"),
    SpecItem("Extension", "Extension MMII", ["extension_msls", "ankle_antepulsion_extension", "knee_antepulsion_extension"], "implemented", "Integracion rodilla y Barré"),
    SpecItem("Extension", "Ascenso hombros", ["ascenso_hombros"], "implemented", "Aproximacion vertical hombro-cabeza"),
    SpecItem("Extension", "Rotacion externa", ["rotacion_externa", "rotacion_externa_brazos", "varo_rodilla"], "implemented", "Versiones opuestas"),

    # MsSs: cadenas de apertura/cierre en miembros superiores segun especificacion clinica
    SpecItem(
        "Apertura",
        "MsSs: ascenso, abduccion, rotacion externa, supinacion",
        ["ascenso_hombros", "abduccion_brazos", "rotacion_externa_brazos"],
        "implemented",
        "Se combinan ascenso de hombros, separacion de brazos y rotacion externa/supinacion de manos",
    ),
    SpecItem(
        "Cierre",
        "MsSs: descenso, aduccion, rotacion interna, pronacion",
        ["descenso_hombros", "aduccion_brazos", "rotacion_interna_brazos"],
        "implemented",
        "Mismos puntos pero en sentido inverso: hombros caidos, brazos pegados y mano pronada",
    ),

    SpecItem("Pendiente", "Hallux valgus", [], "pending", "Requiere pipeline dedicado de pie frontal"),
    SpecItem("Pendiente", "Pie cavo/supino", [], "pending", "Requiere integrar huella plantar al modulo de cadenas"),
    SpecItem("Pendiente", "Dedos en garra", [], "pending", "Requiere segmentacion detallada de dedos en huella"),
    SpecItem("Pendiente", "Validacion clinica de umbrales", [], "pending", "Se necesita dataset etiquetado por especialista"),
]


def _implemented_rule_codes() -> Set[str]:
    return {rule.code for rule in FEATURE_RULES}


def build_traceability_markdown() -> str:
    implemented_codes = _implemented_rule_codes()

    rows: List[str] = []
    implemented_items = 0
    pending_items = 0

    for item in TARGET_ITEMS:
        has_coverage = bool(item.expected_rule_codes) and any(code in implemented_codes for code in item.expected_rule_codes)
        covered = "SI" if has_coverage else "NO"
        if has_coverage:
            implemented_items += 1
        elif item.status == "pending":
            pending_items += 1

        codes = ", ".join(item.expected_rule_codes) if item.expected_rule_codes else "-"
        rows.append(f"| {item.area} | {item.name} | {codes} | {covered} | {item.notes} |")

    total_items = len(TARGET_ITEMS)
    coverage_pct = (implemented_items / total_items * 100.0) if total_items else 0.0

    header = [
        "# Trazabilidad de Especificacion - Cadenas Musculares",
        "",
        f"- Items totales: {total_items}",
        f"- Items con cobertura en reglas actuales: {implemented_items}",
        f"- Items pendientes declarados: {pending_items}",
        f"- Cumplimiento global aproximado: {coverage_pct:.1f}%",
        "",
        "| Area | Rasgo | Codigos Regla | Cubierto | Nota |",
        "|---|---|---|---|---|",
    ]

    return "\n".join(header + rows) + "\n"
