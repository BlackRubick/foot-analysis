from fpdf import FPDF
import os
from datetime import datetime

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
    pdf.add_patient_data(patient_data)
    pdf.add_posture_eval(posture_data)
    pdf.add_plantar_eval(plantar_data)
    pdf.add_analysis_image(img_path)
    pdf.add_results(results_text)
    pdf.output(out_path)
    return out_path
