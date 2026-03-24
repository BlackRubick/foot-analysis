from __future__ import annotations
import argparse
import os
from typing import Optional
from foot_analysis.analyzer import FootAnalyzer
from knee_analysis.analyzer import KneeAnalyzer
from posture_analysis.analyzer import PostureAnalyzer
from ui.tkinter_app import run_tkinter_app
from utils.image_io import destroy_windows, load_image, save_image, show_image


def print_block(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def run_foot(path: Optional[str], save_dir: str, show: bool):
    if not path:
        print("[Baropodometría] Omitido: no se proporcionó imagen.")
        return

    image = load_image(path)
    analyzer = FootAnalyzer()
    result = analyzer.analyze(image)

    print_block("MÓDULO 1 - BAROPODOMETRÍA")
    metrics = result["metrics"]
    print(f"Imagen: {path}")
    print(f"Índice plantar: {metrics['plantar_index']:.2f}")
    print(f"X (antepié): {metrics['x_width_px']:.2f} px")
    print(f"Y (arco plantar): {metrics['y_width_px']:.2f} px")
    print(f"Clasificación: {metrics['classification']}")

    out_prefix = os.path.join(save_dir, "foot")
    save_image(out_prefix + "_annotated.jpg", result["images"]["annotated"])
    save_image(out_prefix + "_gray.jpg", result["images"]["gray"])
    save_image(out_prefix + "_binary.jpg", result["images"]["binary"])
    save_image(out_prefix + "_clean.jpg", result["images"]["clean"])
    save_image(out_prefix + "_edges.jpg", result["images"]["edges"])
    save_image(out_prefix + "_rotated_widths.jpg", result["images"]["rotated_widths"])

    if show:
        show_image("Foot - Annotated", result["images"]["annotated"], wait=1)
        show_image("Foot - Binary", result["images"]["binary"], wait=1)
        show_image("Foot - Edges", result["images"]["edges"], wait=0)


def run_knee(path: Optional[str], plane: str, save_dir: str, show: bool):
    if not path:
        print("[Rodilla] Omitido: no se proporcionó imagen.")
        return

    image = load_image(path)
    analyzer = KneeAnalyzer()
    result = analyzer.analyze(image, plane=plane)

    print_block("MÓDULO 2 - ANÁLISIS DE RODILLA")
    metrics = result["metrics"]
    print(f"Imagen: {path}")
    print(f"Plano: {metrics['plane']}")
    print(f"Lado: {metrics['side']}")
    print(f"Ángulo de rodilla: {metrics['knee_angle_deg']:.2f}°")
    print(f"Clasificación: {metrics['classification']}")

    out_path = os.path.join(save_dir, "knee_annotated.jpg")
    save_image(out_path, result["images"]["annotated"])

    if show:
        show_image("Knee - Annotated", result["images"]["annotated"], wait=0)


def run_posture(path: Optional[str], save_dir: str, show: bool):
    if not path:
        print("[Postura] Omitido: no se proporcionó imagen.")
        return

    image = load_image(path)
    analyzer = PostureAnalyzer()
    result = analyzer.analyze(image)

    print_block("MÓDULO 3 - ANÁLISIS POSTURAL")
    metrics = result["metrics"]
    print(f"Imagen: {path}")
    print(f"Lado analizado: {metrics['side']}")
    print(f"Desviación media: {metrics['mean_deviation_px']:.2f} px")
    print(f"Clasificación: {metrics['classification']}")

    out_path = os.path.join(save_dir, "posture_annotated.jpg")
    save_image(out_path, result["images"]["annotated"])

    if show:
        show_image("Posture - Annotated", result["images"]["annotated"], wait=0)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sistema biomecánico por imágenes (pie, rodilla, postura)")
    parser.add_argument("--mode", type=str, default="tk", choices=["tk", "cli"], help="Modo de ejecución: tk (interfaz gráfica) o cli")
    parser.add_argument("--foot-image", type=str, default=None, help="Ruta de imagen de huella plantar")
    parser.add_argument("--knee-image", type=str, default=None, help="Ruta de imagen para análisis de rodilla")
    parser.add_argument("--posture-image", type=str, default=None, help="Ruta de imagen para análisis postural")
    parser.add_argument("--knee-plane", type=str, default="frontal", choices=["frontal", "sagital"], help="Plano de clasificación de rodilla")
    parser.add_argument("--save-dir", type=str, default="outputs", help="Carpeta de salida")
    parser.add_argument("--show", action="store_true", help="Muestra ventanas OpenCV")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "tk":
        run_tkinter_app()
        return

    os.makedirs(args.save_dir, exist_ok=True)

    try:
        run_foot(args.foot_image, args.save_dir, args.show)
    except Exception as e:
        print(f"Error en módulo de pie: {e}")

    try:
        run_knee(args.knee_image, args.knee_plane, args.save_dir, args.show)
    except Exception as e:
        print(f"Error en módulo de rodilla: {e}")

    try:
        run_posture(args.posture_image, args.save_dir, args.show)
    except Exception as e:
        print(f"Error en módulo de postura: {e}")

    if args.show:
        destroy_windows()

    print_block("Proceso finalizado")
    print(f"Resultados guardados en: {args.save_dir}")


if __name__ == "__main__":
    main()
