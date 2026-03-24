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
            # Intentar abrir con OpenCV para verificar que está disponible
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
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
        # Intentar detectar la cámara CSI de Raspberry Pi si no aparece en /dev/video*
        try:
            import subprocess
            # Buscar cámaras con libcamera-hello (requiere que esté instalado)
            out = subprocess.check_output(["libcamera-hello", "--list-cameras"], stderr=subprocess.DEVNULL, timeout=3)
            for line in out.decode().splitlines():
                if line.strip().startswith("0:") or line.strip().startswith("1:"):
                    # Si la cámara CSI no está en la lista de /dev/video*, agregarla como especial
                    if not any("CSI" in c[1] for c in cameras):
                        cameras.append((-1, "Cámara CSI Raspberry Pi (usar libcamera)"))
        except Exception:
            pass
        if cameras:
            return cameras
    # Fallback: probar con OpenCV
    for idx in range(max_devices):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cameras.append((idx, f"Cámara {idx}"))
            cap.release()
    return cameras
