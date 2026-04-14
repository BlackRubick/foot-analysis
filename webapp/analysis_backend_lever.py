from lever_analysis.mechanics import LeverMechanics
from lever_analysis.calculations import (
    mechanical_advantage, interpret_mechanical_advantage,
    calculate_alpha, round_rule, calculate_mass, calculate_force, calculate_torque
)
import json
from pathlib import Path

SEGMENTS_PATH = Path(__file__).parent.parent / "lever_analysis/data/segments.json"
ARTICULATIONS_PATH = Path(__file__).parent.parent / "lever_analysis/data/articulations.json"

with open(SEGMENTS_PATH, 'r', encoding='utf-8') as f:
    SEGMENTS = json.load(f)

with open(ARTICULATIONS_PATH, 'r', encoding='utf-8') as f:
    ARTICULATIONS = json.load(f)

def analizar_palanca(peso, articulacion, movimiento, segmento, medida_le, medida_lr, co, h):
    tipo_palanca = LeverMechanics.classify_lever(("R", "F", "E"))  # Demo: hardcoded
    vm = mechanical_advantage(medida_le, medida_lr)
    vm_interp = interpret_mechanical_advantage(vm)
    alpha = calculate_alpha(co, h)
    alpha_r = round_rule(alpha)
    masa = calculate_mass(peso, SEGMENTS[segmento])
    fuerza = calculate_force(masa)
    torque = calculate_torque(fuerza, medida_le, alpha)
    return {
        "tipo_palanca": tipo_palanca,
        "ventaja_mecanica": vm,
        "ventaja_interp": vm_interp,
        "alpha": alpha,
        "alpha_redondeado": alpha_r,
        "masa_segmento": masa,
        "fuerza": fuerza,
        "torque": torque,
    }
