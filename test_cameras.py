import cv2

print("Diagnóstico de cámaras disponibles (OpenCV)")

for idx in range(0, 15):
    print(f"\nProbando cámara {idx}...")
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print("  No se pudo abrir.")
        continue
    ret, frame = cap.read()
    if not ret or frame is None:
        print("  No se pudo capturar imagen.")
        cap.release()
        continue
    # Verificar si el frame es todo negro o gris
    if frame.sum() < 10000:
        print("  Imagen vacía (negra o gris).")
    else:
        print(f"  ¡Cámara {idx} FUNCIONA! Tamaño de imagen: {frame.shape}")
    cap.release()

print("\nPrueba terminada. Si alguna cámara funciona, anota su número para usarla en tu app.")
