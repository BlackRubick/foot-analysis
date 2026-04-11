from fpdf import FPDF
import os
from datetime import datetime


def _sanitize_for_pdf(text: str) -> str:
    """Reemplaza caracteres Unicode problemáticos por equivalentes ASCII.

    Esto evita errores de codificación latin-1 en entornos donde fpdf
    no está configurado en UTF-8 (como algunas instalaciones en Raspberry Pi).
    """
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "\u2018": "'",  # ‘
        "\u2019": "'",  # ’
        "\u201c": '"',  # “
        "\u201d": '"',  # ”
        "\u2013": "-",  # –
        "\u2014": "-",  # —
        "\u2022": "-",  # •
        "\u2026": "...",  # …
        "\u00a0": " ",  # NBSP
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def _configure_unicode_font(pdf: FPDF) -> None:
    """Intenta usar DejaVu para soportar tildes/acentos.

    Si en alguna plataforma (p.ej. Raspberry) falla la carga de la fuente
    personalizada, se hace fallback silencioso a Arial para evitar errores
    de codificación/pickle en fpdf.
    """
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    setattr(pdf, "_dejavu_ok", False)
    if os.path.exists(font_path):
        try:
            fonts = getattr(pdf, "fonts", {})
            if "dejavu" not in {k.lower() for k in fonts.keys()}:
                pdf.add_font("DejaVu", "", font_path)
            if "dejavub" not in {k.lower() for k in fonts.keys()}:
                pdf.add_font("DejaVu", "B", font_path)
            if "dejavui" not in {k.lower() for k in fonts.keys()}:
                pdf.add_font("DejaVu", "I", font_path)
            pdf.set_font("DejaVu", "", 10)
            setattr(pdf, "_dejavu_ok", True)
            return
        except Exception:
            # Si por cualquier motivo (permisos, pkl corrupto, etc.) falla, usar Arial
            pass
    pdf.set_font("Arial", "", 10)
    setattr(pdf, "_dejavu_ok", False)


class ConsentPDF(FPDF):
    def header(self):
        if os.path.exists('logo_izq.png'):
            self.image('logo_izq.png', 10, 8, 25)
        if os.path.exists('logo_der.png'):
            self.image('logo_der.png', 175, 8, 25)
        _configure_unicode_font(self)
        use_dejavu = getattr(self, "_dejavu_ok", False)
        self.set_font('DejaVu' if use_dejavu else 'Arial', 'B', 13)
        self.set_fill_color(17, 24, 39)
        self.set_text_color(255, 255, 255)
        self.cell(0, 12, _sanitize_for_pdf('CARTA DE CONSENTIMIENTO INFORMADO Y AVISO DE PRIVACIDAD'), 0, 1, 'C', True)
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def section_title(self, title):
        use_dejavu = getattr(self, "_dejavu_ok", False)
        self.set_font('DejaVu' if use_dejavu else 'Arial', 'B', 11)
        self.set_fill_color(31, 41, 55)
        self.set_text_color(255, 255, 255)
        self.cell(0, 9, _sanitize_for_pdf(title), 0, 1, 'L', True)
        self.set_text_color(0, 0, 0)

    def paragraph(self, text, font_size=10):
        use_dejavu = getattr(self, "_dejavu_ok", False)
        self.set_font('DejaVu' if use_dejavu else 'Arial', '', font_size)
        self.multi_cell(0, 6, _sanitize_for_pdf(text))
        self.ln(1)


def generate_consent_pdf(consent_data, out_path, signature_image_path: str | None = None):
    pdf = ConsentPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    project_name = consent_data.get("project_name", "NEXO-POSTURAL: Kyene's Pøndyam.")
    issue_place = consent_data.get("issue_place", "Tuxtla Gutiérrez, Chiapas.")
    issue_date = consent_data.get("issue_date", datetime.now().strftime("%d/%m/%Y"))
    patient_name = consent_data.get("patient_name", "")
    patient_birthdate = consent_data.get("patient_birthdate", "")
    patient_sex = consent_data.get("patient_sex", "")
    patient_height = consent_data.get("patient_height", "")
    patient_weight = consent_data.get("patient_weight", "")
    responsible_1 = consent_data.get("responsible_1", "Liliana Ruiz Alvarado")
    responsible_2 = consent_data.get("responsible_2", "Ángel Enrique Patricio López")
    witness_1 = consent_data.get("witness_1", "")
    witness_2 = consent_data.get("witness_2", "")

    pdf.section_title(_sanitize_for_pdf("I. DATOS GENERALES"))
    pdf.paragraph(_sanitize_for_pdf(f"Proyecto: {project_name}"))
    pdf.paragraph(_sanitize_for_pdf(f"Lugar y fecha de emisión: {issue_date} {issue_place}"))

    pdf.section_title(_sanitize_for_pdf("II. IDENTIDAD Y DOMICILIO DEL RESPONSABLE"))
    pdf.paragraph(
        _sanitize_for_pdf(
            "El proyecto NEXO-POSTURAL, bajo la responsabilidad de Liliana Ruiz Alvarado y "
            "Ángel Enrique Patricio López, con domicilio en la Universidad Politécnica de Chiapas, "
            "es responsable del uso, tratamiento y protección de sus datos personales."
        )
    )

    pdf.section_title(_sanitize_for_pdf("III. DATOS PERSONALES Y SENSIBLES SOMETIDOS A TRATAMIENTO"))
    pdf.paragraph(
        _sanitize_for_pdf(
            "Para cumplir con las finalidades de este análisis, se recabarán los siguientes datos personales del paciente:"
        )
    )
    pdf.paragraph(
        _sanitize_for_pdf(
            f"- Nombre completo: {patient_name}\n"
            f"- Fecha de nacimiento: {patient_birthdate}\n"
            f"- Sexo: {patient_sex}\n"
            f"- Estatura: {patient_height}\n"
            f"- Peso: {patient_weight}"
        )
    )
    pdf.paragraph(
        _sanitize_for_pdf(
            "Asimismo, se informa que se tratarán datos personales sensibles que requieren protección especial:"
        )
    )
    pdf.paragraph(
        _sanitize_for_pdf(
            "- Imagenes Biométricas: Registro fotográfico y de video para la detección de puntos anatómicos corporales.\n"
            "- Información de Salud: Antecedentes médicos y diagnósticos derivados del estudio."
        )
    )

    pdf.section_title(_sanitize_for_pdf("IV. FINALIDADES DEL ANÁLISIS (ACTO AUTORIZADO)"))
    pdf.paragraph(
        _sanitize_for_pdf(
            "Los datos personales y sensibles recabados se utilizarán exclusivamente para los siguientes procedimientos técnicos:"
        )
    )
    pdf.paragraph(
        _sanitize_for_pdf(
            "1. Evaluación de Miembros Inferiores (Consecuencias mecánicas):\n"
            "- Podometría Digital: Análisis de la huella plantar para determinar la distribución de presiones.\n"
            "- Ángulo Tibiofemoral: Medición de la alineación y biomecánica de la rodilla.\n"
            "- Análisis de Cadena Miofascial: Identificación de la cadena causal (ascendente o descendente) que afecta la postura.\n\n"
            "2. Evaluación de Miembros Superiores y Tronco:\n"
            "- Análisis Dinámico: Determinación del Tipo de Palanca (1er, 2do o 3er genero) en las articulaciones.\n"
            "- Eficiencia Mecánica: Calculo de la Ventaja Mecánica y Momento de Fuerza (Torque) generado en extremidades y tronco.\n\n"
            "3. Gestion Clínica:\n"
            "- Integración de su expediente clínico digital.\n"
            "- Generación de reportes de resultados en formato PDF."
        )
    )

    pdf.section_title(_sanitize_for_pdf("V. DERECHOS ARCO Y CONFIDENCIALIDAD"))
    pdf.paragraph(
        _sanitize_for_pdf(
            "Usted tiene derecho a conocer que datos tenemos de usted, solicitar correcciones o la cancelación de los mismos "
            "(Derechos ARCO). Sin embargo, se hace de su conocimiento que por disposición de la NOM-004-SSA3-2012, los "
            "expedientes clínicos deben ser conservados por un periodo mínimo de 5 años tras el ultimo acto médico. Sus datos "
            "personales no serán divulgados y, en caso de usarse para fines de investigación o docencia, se garantiza que no podra ser identificado."
        )
    )

    pdf.section_title(_sanitize_for_pdf("VI. DECLARACIÓN DE CONSENTIMIENTO"))
    pdf.paragraph(
        _sanitize_for_pdf(
            "Por medio de la presente, autorizo al personal de NEXO-POSTURAL para la realización de los diagnósticos biomecánicos anteriormente descritos. "
            "He sido informado sobre los riesgos mínimos y los beneficios esperados de este análisis para mi salud postural. "
            "Otorgo mi consentimiento expreso para el tratamiento de mis datos personales y sensibles conforme a este aviso."
        )
    )

    pdf.section_title(_sanitize_for_pdf("VII. FIRMAS"))
    use_dejavu = getattr(pdf, "_dejavu_ok", False)
    pdf.set_font('DejaVu' if use_dejavu else 'Arial', '', 10)
    pdf.multi_cell(0, 6, _sanitize_for_pdf(f"Nombre del paciente: {patient_name}"))
    pdf.ln(4)
    pdf.cell(0, 8, "Nombre y Firma del Paciente (o familiar/representante legal)", 0, 1)
    # Línea de firma del paciente
    pdf.cell(0, 8, "______________________________________________", 0, 1)
    # Si hay imagen de firma dibujada, insertarla bajo la línea
    if signature_image_path and os.path.exists(signature_image_path):
        y_before = pdf.get_y()
        # Reservar un bloque de altura fija para la firma
        x_margin = 40
        pdf.ln(2)
        try:
            pdf.image(signature_image_path, x=x_margin, w=60)
        except Exception:
            # Si falla la imagen, al menos continuamos con el resto del documento
            pdf.set_y(y_before)
    pdf.ln(3)
    pdf.cell(0, 8, f"Nombre y Firma del responsable: {responsible_1}", 0, 1)
    pdf.cell(0, 8, "______________________________________________", 0, 1)
    pdf.ln(3)
    pdf.cell(0, 8, f"Nombre y Firma del responsable: {responsible_2}", 0, 1)
    pdf.cell(0, 8, "______________________________________________", 0, 1)
    pdf.ln(3)
    pdf.cell(0, 8, f"Testigo 1: {witness_1}", 0, 1)
    pdf.cell(0, 8, "______________________________________________", 0, 1)
    pdf.ln(3)
    pdf.cell(0, 8, f"Testigo 2: {witness_2}", 0, 1)
    pdf.cell(0, 8, "______________________________________________", 0, 1)

    pdf.output(out_path)
    return out_path

class PDFReport(FPDF):
    def header(self):
        # Logos y título
        if os.path.exists('logo_izq.png'):
            self.image('logo_izq.png', 10, 8, 25)
        if os.path.exists('logo_der.png'):
            self.image('logo_der.png', 175, 8, 25)
        self.set_font('Arial', 'B', 14)
        self.set_fill_color(41, 128, 185)
        self.cell(0, 12, 'EVALUACIÓN ESTÁTICA POSTURAL', 0, 1, 'C', True)
        self.ln(2)

    def section_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(41, 128, 185)
        self.set_text_color(255,255,255)
        self.cell(0, 10, title, 0, 1, 'L', True)
        self.set_text_color(0,0,0)

    def add_patient_data(self, patient):
        self.section_title('DATOS GENERALES DEL PACIENTE')
        self.set_font('Arial', '', 10)
        for k, v in patient.items():
            self.cell(50, 8, f"{k}", 1)
            self.cell(0, 8, str(v), 1, 1)
        self.ln(2)

    def add_calibration_summary(self, source, confidence, mm_per_px=None):
        self.section_title('RESUMEN DE CALIBRACION')
        self.set_font('Arial', 'B', 11)
        source_key = (source or 'pixel').lower()
        source_labels = {
            'aruco': 'ArUco',
            'reference': 'Referencia',
            'height': 'Altura',
            'pixel': 'Sin escala',
        }
        source_text = source_labels.get(source_key, str(source).upper())
        confidence_text = (confidence or 'BAJA').upper()
        line = f"Calibracion usada: {source_text} | Confianza: {confidence_text}"
        if mm_per_px is not None:
            line += f" | Escala: {mm_per_px:.4f} mm/px"
        self.multi_cell(0, 8, line)
        self.ln(2)

    def add_posture_eval(self, posture):
        self.section_title('EVALUACIÓN POSTURAL DEL PACIENTE')
        self.set_font('Arial', '', 10)
        for k, v in posture.items():
            self.cell(50, 8, f"{k}", 1)
            self.cell(0, 8, str(v), 1, 1)
        self.ln(2)

    def add_plantar_eval(self, plantar):
        self.section_title('ANÁLISIS DE HUELLA PLANTAR')
        self.set_font('Arial', '', 10)
        for k, v in plantar.items():
            self.cell(50, 8, f"{k}", 1)
            self.cell(0, 8, str(v), 1, 1)
        self.ln(2)

    def add_results(self, results):
        self.section_title('RESULTADOS E INTERPRETACIÓN')
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 8, results)
        self.ln(2)

    def add_analysis_image(self, img_path, w=80, h=60):
        if os.path.exists(img_path):
            self.image(img_path, x=65, w=w, h=h)
            self.ln(2)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Generado el {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 0, 'C')

def generate_pdf_report(patient_data, posture_data, plantar_data, results_text, img_path, out_path):
    pdf = PDFReport()
    pdf.add_page()

    calibration_source = None
    calibration_confidence = None
    calibration_mm_per_px = None
    for data in (posture_data, plantar_data):
        if isinstance(data, dict) and data:
            if calibration_source is None and data.get('calibration_source') is not None:
                calibration_source = data.get('calibration_source')
            if calibration_confidence is None and data.get('calibration_confidence') is not None:
                calibration_confidence = data.get('calibration_confidence')
            if calibration_mm_per_px is None and data.get('calibration_mm_per_px') is not None:
                try:
                    calibration_mm_per_px = float(data.get('calibration_mm_per_px'))
                except Exception:
                    calibration_mm_per_px = None

    if calibration_source and not calibration_confidence:
        source = str(calibration_source).lower()
        if source == 'aruco':
            calibration_confidence = 'ALTA'
        elif source in {'reference', 'height'}:
            calibration_confidence = 'MEDIA'
        else:
            calibration_confidence = 'BAJA'

    pdf.add_patient_data(patient_data)
    if calibration_source:
        pdf.add_calibration_summary(calibration_source, calibration_confidence, calibration_mm_per_px)
    pdf.add_posture_eval(posture_data)
    pdf.add_plantar_eval(plantar_data)
    pdf.add_analysis_image(img_path)
    pdf.add_results(results_text)
    pdf.output(out_path)
    return out_path
