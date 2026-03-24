import cv2
import os
import re

def list_cameras(max_devices=20):
    """
    Devuelve una lista de tuplas (index, name) de cámaras conectadas.
    En Linux intenta leer /dev/video* y usar v4l2-ctl si está disponible.
    En otros SO prueba abriendo con OpenCV.
    """
    cameras = []
    # Linux: buscar /dev/video* y probar con OpenCV si realmente están disponibles
    if os.name == "posix":
        # Buscar hasta /dev/video20 aunque no existan todos
        for idx in range(max_devices):
            name = f"/dev/video{idx}"
            if not os.path.exists(name):
                continue
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                # Probar que realmente devuelve un frame válido y no vacío
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Considerar frame válido si no es todo negro o gris
                    if frame.sum() > 10000:
                        label = name
                        # Obtener nombre real de la cámara
                        try:
                            import subprocess
                            out = subprocess.check_output(["v4l2-ctl", "-d", name, "--info"], stderr=subprocess.DEVNULL, timeout=1)
                            card_name = None
                            for line in out.decode().splitlines():
                                if b"Card type" in line or b"Card" in line:
                                    card_name = line.split(b":", 1)[-1].strip().decode(errors="ignore")
                                    break
                            if card_name:
                                label = f"{card_name} (índice {idx})"
                            else:
                                label = f"{name} (índice {idx})"
                        except Exception:
                            label = f"{name} (índice {idx})"
                        cameras.append((idx, label))
            cap.release()
        if cameras:
            return cameras
    # Fallback: probar con OpenCV
    for idx in range(max_devices):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cameras.append((idx, f"Cámara {idx}"))
            cap.release()
    return cameras
