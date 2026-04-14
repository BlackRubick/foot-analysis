import numpy as np
import cv2
from foot_analysis.analyzer import FootAnalyzer

foot_analyzer = FootAnalyzer()

def analizar_huella(file_storage):
    # Lee la imagen subida desde el formulario web
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("No se pudo leer la imagen")
    resultado = foot_analyzer.analyze(image)
    return resultado
