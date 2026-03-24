import json
from pathlib import Path

class ArticulationManager:
    def __init__(self, articulations_path: str):
        self.articulations_path = Path(articulations_path)
        self.articulations = self._load_articulations()

    def _load_articulations(self):
        with open(self.articulations_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_articulations(self):
        return list(self.articulations.keys())

    def get_movements(self, articulation):
        return self.articulations[articulation]["movements"]

    def get_movement_info(self, articulation, movement_name):
        for m in self.get_movements(articulation):
            if m["name"] == movement_name:
                return m
        return None
