# Sistema de análisis biomecánico por imágenes

Proyecto modular en Python para:

- Baropodometría (método Hernández-Corvo)
- Análisis de rodilla con MediaPipe
- Análisis postural (línea de plomada de Kendall)

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso rápido

```bash
python main.py
```

Esto abre la interfaz gráfica en Tkinter, donde puedes:

- Cargar imágenes desde archivo
- Tomar fotos desde la cámara
- Ejecutar cada análisis por módulo
- Ver resultados y métricas en la misma interfaz
- Guardar resultados procesados

Si quieres usar el modo consola (CLI):

```bash
python main.py --mode cli --foot-image images/pie.jpg --knee-image images/rodilla.jpg --posture-image images/postura.jpg --knee-plane frontal --save-dir outputs --show
```

## Estructura

- main.py
- foot_analysis/
- knee_analysis/
- posture_analysis/
- utils/
- images/
