from __future__ import annotations

import os
import base64
import json
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Tuple

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_from_directory,
    session,
)

from foot_analysis.analyzer import FootAnalyzer
from knee_analysis.analyzer import KneeAnalyzer
from posture_analysis.analyzer import PostureAnalyzer
from chains_analysis import Calibration, MuscleChainAnalyzer
from utils.image_io import load_image, save_image
from utils.db import DatabaseClient
from utils.pdf_report import generate_consent_pdf, generate_pdf_report

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "web"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CONSENTS_DIR = OUTPUT_DIR / "consents"
CONSENTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = OUTPUT_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get("FOOT_ANALYSIS_WEB_SECRET", "dev-secret-key")


foot_analyzer = FootAnalyzer()
knee_analyzer = KneeAnalyzer()
posture_analyzer = PostureAnalyzer()
chains_analyzer = MuscleChainAnalyzer(calibration=Calibration())


_db_client: Optional[DatabaseClient] = None


def _get_db_client() -> Optional[DatabaseClient]:
    """Obtiene (y memoriza) un DatabaseClient usando variables de entorno o defaults.

    Si no se puede conectar, devuelve None y el modo web funcionará sin DB.
    """
    global _db_client
    if _db_client is not None:
        return _db_client

    host = os.environ.get("FOOT_ANALYSIS_DB_HOST", "127.0.0.1")
    port = int(os.environ.get("FOOT_ANALYSIS_DB_PORT", "3306"))
    user = os.environ.get("FOOT_ANALYSIS_DB_USER", "cesar")
    password = os.environ.get("FOOT_ANALYSIS_DB_PASSWORD", "cesar123")
    name = os.environ.get("FOOT_ANALYSIS_DB_NAME", "foot_analysis_db")
    try:
        _db_client = DatabaseClient(host=host, port=port, user=user, password=password, database=name)
    except Exception as exc:
        # No interrumpir la UI web si MySQL no está disponible
        print(f"[WEB] No se pudo inicializar MySQL: {exc}")
        _db_client = None
    return _db_client


def _ensure_patient_session() -> Optional[Tuple[DatabaseClient, str, int]]:
    """Asegura que haya paciente y sesión activos en la sesión Flask.

    Devuelve (db_client, patient_uuid, session_id) o None si no hay DB.
    """
    db = _get_db_client()
    if db is None:
        return None

    patient_uuid = session.get("patient_uuid")
    patient_fhir_id = session.get("patient_fhir_id")

    patient_uuid, patient_fhir_id = db.ensure_patient(
        patient_uuid=patient_uuid,
        patient_fhir_id=patient_fhir_id,
    )
    session["patient_uuid"] = patient_uuid
    session["patient_fhir_id"] = patient_fhir_id

    session_id = session.get("session_id")
    if session_id is None:
        sess = db.create_capture_session(patient_uuid)
        session_id = int(sess["session_id"])
        session["session_id"] = session_id

    return db, patient_uuid, int(session_id)


def _require_consent() -> bool:
    """Devuelve True si hay consentimiento registrado o no hay DB; si no, avisa."""
    db_tuple = _ensure_patient_session()
    if db_tuple is None:
        return True
    db, patient_uuid, _ = db_tuple
    try:
        if db.has_consent(patient_uuid):
            return True
    except Exception as exc:
        print(f"[WEB] Error comprobando consentimiento: {exc}")
        return True
    flash("Debes registrar consentimiento informado antes de procesar imágenes.", "error")
    return False


def _save_upload(field_name: str) -> str | None:
    """Guarda una imagen subida ya sea como archivo o como data URL base64.

    - Si viene un archivo en request.files[field_name], se guarda tal cual.
    - Si viene un campo oculto <name>_data con data:image/...;base64,...,
      se decodifica y se guarda como PNG.
    """
    # 1) Archivo normal
    file = request.files.get(field_name)
    if file and file.filename:
        ext = os.path.splitext(file.filename)[1].lower() or ".jpg"
        dest = OUTPUT_DIR / f"upload_{field_name}{ext}"
        file.save(dest)
        return str(dest)

    # 2) Data URL desde la cámara del navegador
    data_url = request.form.get(f"{field_name}_data", "").strip()
    if data_url:
        # Esperamos algo como "data:image/png;base64,AAAA..."
        if "," in data_url:
            _, b64_data = data_url.split(",", 1)
        else:
            b64_data = data_url
        try:
            img_bytes = base64.b64decode(b64_data)
        except Exception:
            return None
        dest = OUTPUT_DIR / f"capture_{field_name}.png"
        with open(dest, "wb") as f:
            f.write(img_bytes)
        return str(dest)

    return None


@app.route("/web-outputs/<path:filename>")
def web_output(filename: str):
    """Sirve las imágenes anotadas generadas en OUTPUT_DIR.

    Esto permite mostrarlas directamente en la interfaz web.
    """
    return send_from_directory(str(OUTPUT_DIR), filename)


@app.route("/paciente", methods=["GET"])
def patient_form():
    """Formulario web de datos de paciente y estado de consentimiento."""
    db = _get_db_client()
    patient_data = session.get("patient_data") or {
        "Nombre": "",
        "Fecha de nacimiento": "",
        "Edad": "",
        "Sexo": "",
        "Estatura": "",
        "Peso": "",
        "Ocupación": "",
        "Actividad física": "",
        "Antecedentes traumatológicos": "",
        "Enfermedades crónico-degenerativas": "",
        "Dolor en pies": "No",
        "Dolor en oídos": "No",
        "Alteraciones de la visión": "No",
        "Vértigos": "No",
        "Inestabilidad": "No",
        "Malestar en dientes": "No",
        "Presencia de cicatrices": "No",
    }

    has_consent = False
    if db is not None and session.get("patient_uuid"):
        try:
            has_consent = db.has_consent(session["patient_uuid"])
        except Exception:
            has_consent = False

    return render_template(
        "patient_consent.html",
        patient_data=patient_data,
        patient_uuid=session.get("patient_uuid"),
        patient_fhir_id=session.get("patient_fhir_id"),
        has_consent=has_consent,
    )


@app.route("/paciente/guardar", methods=["POST"])
def save_patient():
    """Guarda/actualiza datos de paciente y crea sesión de captura."""
    db = _get_db_client()
    if db is None:
        flash("Base de datos no disponible. Verifica MySQL.", "error")
        return redirect(url_for("patient_form"))

    form = request.form
    patient_data = {
        "Nombre": form.get("nombre", "").strip(),
        "Fecha de nacimiento": form.get("fecha_nacimiento", "").strip(),
        "Edad": form.get("edad", "").strip(),
        "Sexo": form.get("sexo", "").strip(),
        "Estatura": form.get("estatura", "").strip(),
        "Peso": form.get("peso", "").strip(),
        "Ocupación": form.get("ocupacion", "").strip(),
        "Actividad física": form.get("actividad_fisica", "").strip(),
        "Antecedentes traumatológicos": form.get("antecedentes", "").strip(),
        "Enfermedades crónico-degenerativas": form.get("cronico", "").strip(),
        "Dolor en pies": form.get("dolor_pies", "No").strip() or "No",
        "Dolor en oídos": form.get("dolor_oidos", "No").strip() or "No",
        "Alteraciones de la visión": form.get("vision", "No").strip() or "No",
        "Vértigos": form.get("vertigos", "No").strip() or "No",
        "Inestabilidad": form.get("inestabilidad", "No").strip() or "No",
        "Malestar en dientes": form.get("dientes", "No").strip() or "No",
        "Presencia de cicatrices": form.get("cicatrices", "No").strip() or "No",
    }

    # Recalcular edad a partir de la fecha de nacimiento si es posible
    birth = patient_data.get("Fecha de nacimiento")
    if birth and not patient_data.get("Edad"):
        try:
            dob = datetime.strptime(birth, "%Y-%m-%d").date()
            today = date.today()
            years = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            patient_data["Edad"] = str(max(years, 0))
        except Exception:
            pass

    # Crear/asegurar paciente y sesión
    try:
        db_tuple = _ensure_patient_session()
        if db_tuple is None:
            flash("No se pudo inicializar la sesión de paciente.", "error")
            return redirect(url_for("patient_form"))
        db, patient_uuid, _ = db_tuple
        identity_name = patient_data.get("Nombre", "").strip() or "PACIENTE SIN NOMBRE"
        db.upsert_patient_identity(
            patient_uuid,
            identity_name,
            json.dumps(patient_data, ensure_ascii=False),
        )
        session["patient_data"] = patient_data
        flash("Datos de paciente guardados.", "info")
    except Exception as exc:
        flash(f"No se pudieron guardar los datos de paciente: {exc}", "error")

    return redirect(url_for("patient_form"))


@app.route("/consentimiento/guardar", methods=["POST"])
def save_consent():
    """Guarda consentimiento informado con firma dibujada y PDF en DB."""
    db_tuple = _ensure_patient_session()
    if db_tuple is None:
        flash("Base de datos no disponible para guardar consentimiento.", "error")
        return redirect(url_for("patient_form"))
    db, patient_uuid, _ = db_tuple

    form = request.form
    signed_by = (form.get("signed_by") or "").strip()
    patient_name = (form.get("patient_name") or "").strip()
    birth = (form.get("birthdate") or "").strip()
    height = (form.get("height") or "").strip()
    weight = (form.get("weight") or "").strip()
    witness1 = (form.get("witness1") or "").strip()
    witness2 = (form.get("witness2") or "").strip()
    consent_text = (form.get("consent_text") or "").strip()
    signature_data = (form.get("signature_data") or "").strip()

    if not signed_by:
        flash("El nombre de quien firma es obligatorio.", "error")
        return redirect(url_for("patient_form"))
    if not consent_text:
        flash("El texto de consentimiento es obligatorio.", "error")
        return redirect(url_for("patient_form"))

    # Guardar imagen de firma si existe
    signature_image_path = None
    signature_bytes = None
    if signature_data:
        try:
            if "," in signature_data:
                _, b64_data = signature_data.split(",", 1)
            else:
                b64_data = signature_data
            sig_bytes = base64.b64decode(b64_data)
            sig_filename = f"firma_{patient_uuid}.png"
            signature_image_path = CONSENTS_DIR / sig_filename
            with open(signature_image_path, "wb") as f_sig:
                f_sig.write(sig_bytes)
            signature_bytes = sig_bytes
        except Exception as exc:
            print(f"[WEB] No se pudo procesar la firma dibujada: {exc}")
            signature_image_path = None
            signature_bytes = None

    # Generar PDF de consentimiento
    consent_pdf_filename = f"consentimiento_{patient_uuid}.pdf"
    consent_pdf_path = CONSENTS_DIR / consent_pdf_filename
    patient_data = session.get("patient_data") or {}
    consent_payload = {
        "project_name": "NEXO-POSTURAL: Kyene’is Pøndyam.",
        "issue_place": "Tuxtla Gutiérrez, Chiapas.",
        "issue_date": datetime.now().strftime("%d/%m/%Y"),
        "patient_name": patient_name or patient_data.get("Nombre", ""),
        "patient_sex": patient_data.get("Sexo", ""),
        "responsible_1": "Liliana Ruiz Alvarado",
        "responsible_2": "Ángel Enrique Patricio López",
        "witness_1": witness1,
        "witness_2": witness2,
        "patient_birthdate": birth or patient_data.get("Fecha de nacimiento", ""),
        "patient_height": height or patient_data.get("Estatura", ""),
        "patient_weight": weight or patient_data.get("Peso", ""),
    }
    try:
        if signature_image_path is not None:
            generate_consent_pdf(consent_payload, str(consent_pdf_path), signature_image_path=str(signature_image_path))
        else:
            generate_consent_pdf(consent_payload, str(consent_pdf_path))
    except Exception as exc:
        flash(f"No se pudo generar PDF de consentimiento: {exc}", "error")
        return redirect(url_for("patient_form"))

    # Leer bytes de documento para DB (se usa el PDF generado)
    try:
        with open(consent_pdf_path, "rb") as f_pdf:
            document_bytes = f_pdf.read()
    except Exception as exc:
        flash(f"No se pudo leer el PDF generado: {exc}", "error")
        return redirect(url_for("patient_form"))

    try:
        db.save_informed_consent(
            patient_uuid=patient_uuid,
            consent_text=consent_text,
            signed_by=signed_by,
            signature_digital_hash="",
            consent_document_bytes=document_bytes,
            signature_image_bytes=signature_bytes,
        )
        flash("Consentimiento informado registrado correctamente.", "info")
        session["has_consent"] = True
    except Exception as exc:
        flash(f"No se pudo guardar consentimiento en DB: {exc}", "error")

    return redirect(url_for("patient_form"))


@app.route("/")
def index():
    db = _get_db_client()
    has_consent = False
    if db is not None and session.get("patient_uuid"):
        try:
            has_consent = db.has_consent(session["patient_uuid"])
        except Exception:
            has_consent = False
    return render_template("index.html", has_consent=has_consent, patient_fhir_id=session.get("patient_fhir_id"))


@app.route("/analyze/foot", methods=["POST"])
def analyze_foot():
    if not _require_consent():
        return redirect(url_for("patient_form"))
    path = _save_upload("foot_image")
    if not path:
        flash("Sube una imagen de huella plantar para analizar.", "error")
        return redirect(url_for("index"))
    image = load_image(path)
    result = foot_analyzer.analyze(image)
    annotated_path = OUTPUT_DIR / "foot_annotated_web.jpg"
    save_image(str(annotated_path), result["images"]["annotated"])
    # Guardar en DB si está disponible
    db_tuple = _ensure_patient_session()
    if db_tuple is not None:
        db, patient_uuid, session_id = db_tuple
        try:
            db.save_analysis(
                patient_uuid=patient_uuid,
                session_id=session_id,
                analysis_type="foot",
                metrics=result["metrics"],
                notes_text="",
            )
        except Exception as exc:
            print(f"[WEB] No se pudo guardar analisis de pie en DB: {exc}")
    foot_image_url = url_for("web_output", filename=annotated_path.name)
    return render_template("index.html", foot_result=result, foot_image_url=foot_image_url)


@app.route("/analyze/knee", methods=["POST"])
def analyze_knee():
    if not _require_consent():
        return redirect(url_for("patient_form"))
    path = _save_upload("knee_image")
    if not path:
        flash("Sube una imagen de rodilla para analizar.", "error")
        return redirect(url_for("index"))
    plane = request.form.get("knee_plane", "frontal")
    image = load_image(path)
    result = knee_analyzer.analyze(image, plane=plane)
    annotated_path = OUTPUT_DIR / "knee_annotated_web.jpg"
    save_image(str(annotated_path), result["images"]["annotated"])
    db_tuple = _ensure_patient_session()
    if db_tuple is not None:
        db, patient_uuid, session_id = db_tuple
        try:
            db.save_analysis(
                patient_uuid=patient_uuid,
                session_id=session_id,
                analysis_type="knee",
                metrics=result["metrics"],
                notes_text="",
            )
        except Exception as exc:
            print(f"[WEB] No se pudo guardar analisis de rodilla en DB: {exc}")
    knee_image_url = url_for("web_output", filename=annotated_path.name)
    return render_template("index.html", knee_result=result, knee_image_url=knee_image_url)


@app.route("/analyze/posture", methods=["POST"])
def analyze_posture():
    if not _require_consent():
        return redirect(url_for("patient_form"))
    path = _save_upload("posture_image")
    if not path:
        flash("Sube una imagen de postura para analizar.", "error")
        return redirect(url_for("index"))
    image = load_image(path)
    result = posture_analyzer.analyze(image)
    annotated_path = OUTPUT_DIR / "posture_annotated_web.jpg"
    save_image(str(annotated_path), result["images"]["annotated"])
    db_tuple = _ensure_patient_session()
    if db_tuple is not None:
        db, patient_uuid, session_id = db_tuple
        try:
            db.save_analysis(
                patient_uuid=patient_uuid,
                session_id=session_id,
                analysis_type="posture",
                metrics=result["metrics"],
                notes_text="",
            )
        except Exception as exc:
            print(f"[WEB] No se pudo guardar analisis de postura en DB: {exc}")
    posture_image_url = url_for("web_output", filename=annotated_path.name)
    return render_template("index.html", posture_result=result, posture_image_url=posture_image_url)


@app.route("/analyze/chains", methods=["POST"])
def analyze_chains():
    if not _require_consent():
        return redirect(url_for("patient_form"))
    path = _save_upload("chains_image")
    if not path:
        flash("Sube una imagen para cadenas musculares.", "error")
        return redirect(url_for("index"))
    image = load_image(path)
    # Para la versión web usamos parámetros por defecto (sagittal, auto, lado auto)
    result = chains_analyzer.analyze(image, plane="sagittal", profile_side="auto")
    annotated_path = OUTPUT_DIR / "chains_annotated_web.jpg"
    save_image(str(annotated_path), result.images["annotated"])
    db_tuple = _ensure_patient_session()
    if db_tuple is not None:
        db, patient_uuid, session_id = db_tuple
        try:
            db.save_analysis(
                patient_uuid=patient_uuid,
                session_id=session_id,
                analysis_type="chains",
                metrics=result.metrics,
                notes_text="",
            )
        except Exception as exc:
            print(f"[WEB] No se pudo guardar analisis de cadenas en DB: {exc}")
    chains_image_url = url_for("web_output", filename=annotated_path.name)
    return render_template("index.html", chains_result=result, chains_image_url=chains_image_url)


def run_web_app(host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":  # pragma: no cover
    run_web_app()
