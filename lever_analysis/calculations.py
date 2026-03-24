import math

def mechanical_advantage(le: float, lr: float) -> float:
    """Calcula la ventaja mecánica VM = LE / LR"""
    if lr == 0:
        raise ValueError("El brazo de resistencia no puede ser cero.")
    return le / lr

def interpret_mechanical_advantage(vm: float) -> str:
    if vm > 1:
        return "Más fuerza, menos velocidad"
    elif vm < 1:
        return "Más velocidad, menos fuerza"
    else:
        return "Equilibrio entre fuerza y velocidad"

def calculate_alpha(co: float, h: float) -> float:
    """Calcula el ángulo alfa en grados usando sin(α) = CO / H"""
    if h == 0:
        raise ValueError("La hipotenusa no puede ser cero.")
    sin_alpha = co / h
    # Limitar el dominio de arcsin
    sin_alpha = max(-1.0, min(1.0, sin_alpha))
    alpha_rad = math.asin(sin_alpha)
    alpha_deg = math.degrees(alpha_rad)
    return alpha_deg

def round_rule(val: float) -> int:
    """Redondeo especial: >0.6 hacia arriba, <=0.6 hacia abajo"""
    frac = val - int(val)
    if frac > 0.6:
        return math.ceil(val)
    else:
        return math.floor(val)

def calculate_mass(person_weight: float, segment_percent: float) -> float:
    return person_weight * segment_percent

def calculate_force(mass: float, g: float = 9.81) -> float:
    return mass * g

def calculate_torque(force: float, d: float, alpha_deg: float) -> float:
    alpha_rad = math.radians(alpha_deg)
    return force * d * math.sin(alpha_rad)
