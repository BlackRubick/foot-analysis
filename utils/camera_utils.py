import cv2
import os
import re

def list_cameras(max_devices=10):
    """
    Devuelve una lista de tuplas (index, name) de cámaras conectadas.
    En Linux intenta leer /dev/video* y usar v4l2-ctl si está disponible.
    En otros SO prueba abriendo con OpenCV.
    """
    cameras = []
    # Linux: buscar /dev/video* y probar con OpenCV si realmente están disponibles
    if os.name == "posix":
        video_devs = [f for f in os.listdir("/dev") if re.match(r"video\\d+", f)]
        for dev in sorted(video_devs, key=lambda x: int(re.findall(r"\\d+", dev)[0])):
            idx = int(re.findall(r"\\d+", dev)[0])
            name = f"/dev/{dev}"
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                # Probar que realmente devuelve un frame válido y no vacío
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Considerar frame válido si no es todo negro o gris
                    if frame.sum() > 10000:  # Umbral simple para descartar frames vacíos
                        label = name
                        try:
                            import subprocess
                            out = subprocess.check_output(["v4l2-ctl", "-d", name, "--info"], stderr=subprocess.DEVNULL, timeout=1)
                            for line in out.decode().splitlines():
                                if "Card type" in line:
                                    label = line.split(":", 1)[-1].strip()
                                    break
                        except Exception:
                            pass
                        cameras.append((idx, f"{label} (índice {idx})"))
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
