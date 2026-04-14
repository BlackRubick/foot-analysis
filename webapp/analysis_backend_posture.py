import numpy as np
import cv2
from posture_analysis.analyzer import PostureAnalyzer

posture_analyzer = PostureAnalyzer()

def analizar_postura(file_storage):
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No se pudo leer la imagen")
    resultado = posture_analyzer.analyze(image)
    return resultado
