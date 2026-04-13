from __future__ import annotations
import argparse
import os
from typing import Optional
from foot_analysis.analyzer import FootAnalyzer
from chains_analysis import Calibration, MuscleChainAnalyzer
from chains_analysis.spec_traceability import build_traceability_markdown
from knee_analysis.analyzer import KneeAnalyzer
from posture_analysis.analyzer import PostureAnalyzer
from ui.tkinter_app import run_tkinter_app
from utils.db import DatabaseClient
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

    return result


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

    return result


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

    return result


def _estimate_pose_height_px(detection) -> float:
    pose = detection.pose
    y_values = []
    for key in (
        "nose",
        "left_ear",
        "right_ear",
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
    ):
        point = pose.get(key)
        if point is not None:
            y_values.append(point.y)
    if len(y_values) < 2:
        raise ValueError("No fue posible estimar la altura de la pose para la calibracion")
    return float(max(y_values) - min(y_values))


def _print_feature_summary(feature_results):
    positives = [item for item in feature_results if item.present]
    negatives = [item for item in feature_results if not item.present]

    if positives:
        print("Rasgos presentes:")
        for item in positives:
            print(f"  + {item.label}: {item.value:.2f} {item.unit}")

    if negatives:
        print("Rasgos ausentes:")
        for item in negatives[:10]:
            print(f"  - {item.label}: {item.value:.2f} {item.unit}")


def run_chains(
    path: Optional[str],
    save_dir: str,
    show: bool,
    plane: str,
    profile_side: str,
    calibration_mode: str,
    reference_mm: float,
    reference_px: float,
    height_mm: float,
    aruco_marker_mm: float,
):
    if not path:
        print("[Cadenas musculares] Omitido: no se proporciono imagen.")
        return

    image = load_image(path)
    calibration = Calibration()

    def try_aruco() -> Optional[Calibration]:
        if aruco_marker_mm <= 0:
            return None
        try:
            return MuscleChainAnalyzer.estimate_aruco_calibration(image, aruco_marker_mm)
        except Exception:
            return None

    def try_reference() -> Optional[Calibration]:
        if reference_mm <= 0 or reference_px <= 0:
            return None
        try:
            return Calibration.from_reference(reference_mm, reference_px)
        except Exception:
            return None

    def try_height() -> Optional[Calibration]:
        if height_mm <= 0:
            return None
        try:
            probe_analyzer = MuscleChainAnalyzer()
            detection = probe_analyzer.detector.detect(image)
            pose_height_px = _estimate_pose_height_px(detection)
            return Calibration.from_height(height_mm, pose_height_px)
        except Exception:
            return None

    if calibration_mode == "aruco":
        calibration = try_aruco() or Calibration()
    elif calibration_mode == "reference":
        calibration = try_reference() or Calibration()
    elif calibration_mode == "height":
        calibration = try_height() or Calibration()
    elif calibration_mode == "auto":
        calibration = try_aruco() or try_reference() or try_height() or Calibration()

    analyzer = MuscleChainAnalyzer(calibration=calibration)
    result = analyzer.analyze(image, plane=plane, profile_side=profile_side)

    print_block("MODO NUEVO - ANALISIS DE CADENAS MUSCULARES")
    print(f"Imagen: {path}")
    print(f"Plano: {result.metrics['plane']}")
    print(f"Lado/patrón detectado: {result.metrics['profile_side']}")
    print(f"Calibracion: {result.metrics['calibration_mm_per_px']:.4f} mm/px")

    for chain_key, summary in result.chain_summaries.items():
        print(
            f"{summary.name}: prevalencia {summary.percentage:.1f}% | "
            f"activacion {summary.activation_percentage:.1f}% ({summary.positives}/{summary.total})"
        )

    for note in result.notes:
        print(note)

    _print_feature_summary(result.feature_results)

    out_path = os.path.join(save_dir, "chains_annotated.jpg")
    save_image(out_path, result.images["annotated"])

    if show:
        show_image("Chains - Annotated", result.images["annotated"], wait=0)

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sistema biomecanico por imagenes (pie, rodilla, postura y cadenas musculares)")
    parser.add_argument("--mode", type=str, default="tk", choices=["tk", "cli", "web"], help="Modo de ejecución: tk (interfaz gráfica), cli o web (navegador)")
    parser.add_argument("--foot-image", type=str, default=None, help="Ruta de imagen de huella plantar")
    parser.add_argument("--knee-image", type=str, default=None, help="Ruta de imagen para análisis de rodilla")
    parser.add_argument("--posture-image", type=str, default=None, help="Ruta de imagen para análisis postural")
    parser.add_argument("--chains-image", type=str, default=None, help="Ruta de imagen para análisis de cadenas musculares")
    parser.add_argument("--knee-plane", type=str, default="frontal", choices=["frontal", "sagital"], help="Plano de clasificación de rodilla")
    parser.add_argument("--chains-plane", type=str, default="sagittal", choices=["frontal", "sagittal"], help="Plano de análisis de cadenas musculares")
    parser.add_argument("--profile-side", type=str, default="auto", choices=["auto", "left", "right"], help="Lado visible para el analisis sagital")
    parser.add_argument("--calibration-mode", type=str, default="auto", choices=["auto", "none", "reference", "height", "aruco"], help="Modo de calibracion mm/px")
    parser.add_argument("--reference-mm", type=float, default=100.0, help="Longitud real del objeto de referencia en mm")
    parser.add_argument("--reference-px", type=float, default=100.0, help="Longitud del objeto de referencia en pixeles")
    parser.add_argument("--patient-height-mm", type=float, default=1700.0, help="Altura del paciente en mm para calibracion por altura")
    parser.add_argument("--aruco-marker-mm", type=float, default=50.0, help="Tamaño real del marcador ArUco en mm")
    parser.add_argument("--chains-traceability-report", action="store_true", help="Genera reporte de trazabilidad/cumplimiento de rasgos de cadenas")
    parser.add_argument("--save-dir", type=str, default="outputs", help="Carpeta de salida")
    parser.add_argument("--show", action="store_true", help="Muestra ventanas OpenCV")

    parser.add_argument("--db-host", type=str, default="127.0.0.1", help="Host MySQL")
    parser.add_argument("--db-port", type=int, default=3306, help="Puerto MySQL")
    parser.add_argument("--db-user", type=str, default="cesar", help="Usuario MySQL")
    parser.add_argument("--db-password", type=str, default="cesar123", help="Password MySQL")
    parser.add_argument("--db-name", type=str, default="foot_analysis_db", help="Nombre de base de datos")
    parser.add_argument("--patient-uuid", type=str, default=None, help="UUID anonimo del paciente")
    parser.add_argument("--patient-fhir-id", type=str, default=None, help="ID de paciente formato patient-XXXX")
    parser.add_argument("--no-db", action="store_true", help="Desactiva el guardado automatico en MySQL")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "tk":
        run_tkinter_app()
        return

    if args.mode == "web":
        from webapp.server import run_web_app
        # Permitir configurar host/puerto por variables de entorno si se desea
        host = os.environ.get("FOOT_ANALYSIS_WEB_HOST", "0.0.0.0")
        port = int(os.environ.get("FOOT_ANALYSIS_WEB_PORT", "5000"))
        run_web_app(host=host, port=port, debug=False)
        return

    os.makedirs(args.save_dir, exist_ok=True)

    db_client = None
    patient_uuid = None
    session_id = None

    if not args.no_db:
        try:
            db_client = DatabaseClient(
                host=args.db_host,
                port=args.db_port,
                user=args.db_user,
                password=args.db_password,
                database=args.db_name,
            )
            patient_uuid, patient_fhir_id = db_client.ensure_patient(
                patient_uuid=args.patient_uuid,
                patient_fhir_id=args.patient_fhir_id,
            )
            session = db_client.create_capture_session(patient_uuid)
            session_id = session["session_id"]
            print_block("BASE DE DATOS")
            print(f"Paciente: {patient_fhir_id} ({patient_uuid})")
            print(f"Sesion: {session_id} | Study UID: {session['study_instance_uid']}")
        except Exception as e:
            print(f"[DB] No se pudo inicializar MySQL. Se continua sin guardado: {e}")
            db_client = None

    if args.chains_traceability_report:
        report = build_traceability_markdown()
        report_path = os.path.join(args.save_dir, "chains_traceability_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print_block("REPORTE DE CUMPLIMIENTO - CADENAS MUSCULARES")
        print(report)
        print(f"Reporte guardado en: {report_path}")

    try:
        foot_result = run_foot(args.foot_image, args.save_dir, args.show)
        if db_client is not None and patient_uuid and session_id and foot_result:
            db_client.save_analysis(
                patient_uuid=patient_uuid,
                session_id=session_id,
                analysis_type="foot",
                metrics=foot_result["metrics"],
                notes_text="",
            )
    except Exception as e:
        print(f"Error en módulo de pie: {e}")

    try:
        knee_result = run_knee(args.knee_image, args.knee_plane, args.save_dir, args.show)
        if db_client is not None and patient_uuid and session_id and knee_result:
            db_client.save_analysis(
                patient_uuid=patient_uuid,
                session_id=session_id,
                analysis_type="knee",
                metrics=knee_result["metrics"],
                notes_text="",
            )
    except Exception as e:
        print(f"Error en módulo de rodilla: {e}")

    try:
        posture_result = run_posture(args.posture_image, args.save_dir, args.show)
        if db_client is not None and patient_uuid and session_id and posture_result:
            db_client.save_analysis(
                patient_uuid=patient_uuid,
                session_id=session_id,
                analysis_type="posture",
                metrics=posture_result["metrics"],
                notes_text="",
            )
    except Exception as e:
        print(f"Error en módulo de postura: {e}")

    try:
        chains_result = run_chains(
            args.chains_image,
            args.save_dir,
            args.show,
            args.chains_plane,
            args.profile_side,
            args.calibration_mode,
            args.reference_mm,
            args.reference_px,
            args.patient_height_mm,
            args.aruco_marker_mm,
        )
        if db_client is not None and patient_uuid and session_id and chains_result:
            db_client.save_analysis(
                patient_uuid=patient_uuid,
                session_id=session_id,
                analysis_type="chains",
                metrics=chains_result.metrics,
                notes_text="\n".join(chains_result.notes),
            )
    except Exception as e:
        print(f"Error en módulo de cadenas musculares: {e}")

    if args.show:
        destroy_windows()

    if db_client is not None:
        db_client.close()

    print_block("Proceso finalizado")
    print(f"Resultados guardados en: {args.save_dir}")


if __name__ == "__main__":
    main()
