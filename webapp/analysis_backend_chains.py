import numpy as np
import cv2
from chains_analysis.analyzer import MuscleChainAnalyzer

chains_analyzer = MuscleChainAnalyzer()

def analizar_cadenas(file_storage_or_image, plane="sagittal", profile_side="auto"):
    # Si ya es un np.ndarray, úsalo directamente
    if isinstance(file_storage_or_image, np.ndarray):
        image = file_storage_or_image
    else:
        file_bytes = np.frombuffer(file_storage_or_image.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No se pudo leer la imagen")
    resultado = chains_analyzer.analyze(image, plane=plane, profile_side=profile_side)
    return resultado
