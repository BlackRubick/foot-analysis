import numpy as np
import cv2
from chains_analysis.analyzer import MuscleChainAnalyzer

chains_analyzer = MuscleChainAnalyzer()

def analizar_cadenas(file_storage, plane="sagittal", profile_side="auto"):
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No se pudo leer la imagen")
    resultado = chains_analyzer.analyze(image, plane=plane, profile_side=profile_side)
    return resultado
