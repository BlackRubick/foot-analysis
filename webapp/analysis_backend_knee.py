import numpy as np
import cv2
from knee_analysis.analyzer import KneeAnalyzer

knee_analyzer = KneeAnalyzer()

def analizar_rodilla(file_storage, plane="frontal"):
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No se pudo leer la imagen")
    resultado = knee_analyzer.analyze(image, plane=plane)
    return resultado
