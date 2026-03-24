# Sistema de análisis biomecánico por imágenes

## ¿Cómo está hecho el código?
El código está organizado en módulos para que cada parte haga una tarea específica:

- **main.py**: Es el archivo principal. Desde aquí se inicia el programa y se conecta todo.
- **foot_analysis/**: Analiza los pies usando imágenes y el método Hernández-Corvo.
- **knee_analysis/**: Analiza la rodilla usando inteligencia artificial (MediaPipe).
- **posture_analysis/**: Analiza la postura del cuerpo (línea de plomada de Kendall).
- **utils/**: Tiene funciones útiles para manejar imágenes, la cámara y otras tareas generales.
- **ui/**: Contiene la interfaz gráfica hecha con Tkinter.
- **models/**: Guarda los modelos de IA necesarios para el análisis.
- **images/**: Carpeta donde se guardan o cargan las imágenes a analizar.
- **outputs/**: Carpeta donde se guardan los resultados del análisis.


Cada módulo tiene su propio archivo `analyzer.py` que se encarga de procesar las imágenes y calcular los resultados.

## ¿Cómo funciona la IA en este sistema?
La inteligencia artificial se usa en todos los análisis: pies, rodilla y postura. El sistema utiliza modelos de IA (como MediaPipe y otros) que detectan puntos clave en el cuerpo a partir de fotos. Estos puntos permiten medir ángulos, posiciones y zonas de presión automáticamente, sin intervención manual.

Los modelos de IA están guardados en la carpeta `models/` y son cargados por el código según el análisis que se realice. Así, el sistema puede identificar la posición de los pies, rodillas y postura, y calcular métricas biomecánicas de forma precisa y rápida.

En resumen, la IA ayuda a automatizar el análisis de imágenes en todos los módulos, haciendo el proceso más fácil, rápido y objetivo.

Esta estructura hace que el código sea fácil de entender y modificar.

## ¿Qué es este sistema?
Es un programa en Python que analiza imágenes para estudiar la biomecánica del cuerpo humano. Permite analizar pies, rodillas y postura usando fotos.

## ¿Para qué se usó?
Se usó para hacer análisis biomecánico de personas, ayudando a identificar problemas en pies, rodillas y postura de manera sencilla y visual.

## ¿Cómo se usó?
Se ejecuta el programa, se cargan o toman fotos, y el sistema muestra los resultados del análisis en la pantalla. También se pueden guardar los resultados. Todo se hace desde una interfaz gráfica fácil de usar.


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