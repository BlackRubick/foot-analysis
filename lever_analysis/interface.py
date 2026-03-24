import json
from pathlib import Path
from lever_analysis.articulations import ArticulationManager
from lever_analysis.mechanics import LeverMechanics
from lever_analysis.calculations import (
    mechanical_advantage, interpret_mechanical_advantage,
    calculate_alpha, round_rule, calculate_mass, calculate_force, calculate_torque
)

SEGMENTS_PATH = Path(__file__).parent / "data/segments.json"
ARTICULATIONS_PATH = Path(__file__).parent / "data/articulations.json"

class BiomechanicsCLI:
    def __init__(self):
        self.art_manager = ArticulationManager(ARTICULATIONS_PATH)
        with open(SEGMENTS_PATH, 'r', encoding='utf-8') as f:
            self.segments = json.load(f)

    def run(self):
        print("\n=== Análisis Biomecánico Dinámico ===\n")
        while True:
            try:
                peso = float(input("Peso de la persona (kg): "))
                print("Articulaciones disponibles:")
                for i, art in enumerate(self.art_manager.get_articulations()):
                    print(f"  {i+1}. {art}")
                art_idx = int(input("Selecciona articulación: ")) - 1
                articulacion = self.art_manager.get_articulations()[art_idx]
                movimientos = self.art_manager.get_movements(articulacion)
                print("Movimientos disponibles:")
                for i, mov in enumerate(movimientos):
                    print(f"  {i+1}. {mov['name']}")
                mov_idx = int(input("Selecciona movimiento: ")) - 1
                movimiento = movimientos[mov_idx]
                segmento = input("Segmento corporal (ej: 'thigh', 'forearm'): ")
                if segmento not in self.segments:
                    print("Segmento no válido. Usa uno de:", ', '.join(self.segments.keys()))
                    continue
                medida_le = float(input("Brazo de esfuerzo (LE, cm): ")) / 100
                medida_lr = float(input("Brazo de resistencia (LR, cm): ")) / 100
                co = float(input("Cateto opuesto para alfa (mm): ")) / 1000
                h = float(input("Hipotenusa para alfa (mm): ")) / 1000
                # Cálculos
                tipo_palanca = LeverMechanics.classify_lever(("R", "F", "E"))  # Demo: hardcoded
                vm = mechanical_advantage(medida_le, medida_lr)
                vm_interp = interpret_mechanical_advantage(vm)
                alpha = calculate_alpha(co, h)
                alpha_r = round_rule(alpha)
                masa = calculate_mass(peso, self.segments[segmento])
                fuerza = calculate_force(masa)
                torque = calculate_torque(fuerza, medida_le, alpha)
                print("\n--- RESULTADOS ---")
                print(f"Tipo de palanca: {tipo_palanca}")
                print(f"Ventaja mecánica: {vm:.2f} ({vm_interp})")
                print(f"Ángulo alfa: {alpha:.2f}° (redondeado: {alpha_r}°)")
                print(f"Masa segmento: {masa:.2f} kg")
                print(f"Fuerza: {fuerza:.2f} N")
                print(f"Torque: {torque:.2f} Nm")
                print("Interpretación automática: ...")
                print("-------------------\n")
                again = input("¿Otro análisis? (s/n): ").strip().lower()
                if again != 's':
                    break
            except Exception as e:
                print(f"Error: {e}\n")
