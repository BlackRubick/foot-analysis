from typing import Tuple

class LeverMechanics:
    @staticmethod
    def classify_lever(positions: Tuple[str, str, str]) -> str:
        """
        Classify lever type based on order of (R, F, E)
        Returns: 'Primera clase', 'Segunda clase', 'Tercera clase'
        """
        # positions: (R, F, E) or (E, R, F) etc.
        if positions == ("R", "F", "E"):
            return "Primera clase"
        elif positions == ("E", "R", "F"):
            return "Segunda clase"
        elif positions == ("R", "E", "F"):
            return "Tercera clase"
        else:
            return "Desconocida"

    @staticmethod
    def auto_identify_positions(fulcrum_pos, effort_pos, resistance_pos) -> Tuple[str, str, str]:
        """
        Returns the tuple (R, F, E) in spatial order (e.g., left to right)
        """
        # For demo: assume input is already ordered
        return (resistance_pos, fulcrum_pos, effort_pos)
