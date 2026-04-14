
from flask import Flask, render_template, request, redirect, url_for, send_file, session, flash
import os
import cv2
import time
from .analysis_backend import analizar_huella
from .analysis_backend_knee import analizar_rodilla
from .analysis_backend_posture import analizar_postura
from .analysis_backend_chains import analizar_cadenas
from .analysis_backend_lever import analizar_palanca
from utils.pdf_report import generate_pdf_report

app = Flask(__name__)
app.secret_key = 'biomecanica2026' 
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Endpoint para manejar el botón de generar PDF (POST) y redirigir a la descarga
@app.route('/generar_pdf', methods=['POST'])
def generar_pdf():
    return redirect(url_for('descargar_pdf'))



# Redirigir la raíz según el estado de sesión
@app.route('/')
def index():
    if not session.get('paciente'):
        return redirect(url_for('paciente'))
    if not session.get('consentimiento'):
        return redirect(url_for('consentimiento'))
    # Si ya hay datos y consentimiento, mostrar la pantalla principal
    context = {}
    # Huella plantar
    metrics = session.get('metrics')
    annotated_path = session.get('annotated_path')
    if metrics:
        context['foot_result'] = {'metrics': metrics}
    if annotated_path and os.path.exists(annotated_path):
        # Usar la ruta real guardada en session y pasar timestamp para evitar cache
        static_name = os.path.basename(annotated_path)
        static_path = os.path.join(app.static_folder, static_name)
        if not os.path.exists(static_path):
            import shutil
            shutil.copyfile(annotated_path, static_path)
        context['foot_image_url'] = url_for('static', filename=static_name)
        context['now'] = int(time.time())
    # Rodilla (frontal y sagital)
    knee_metrics_frontal = session.get('knee_metrics_frontal')
    knee_metrics_sagital = session.get('knee_metrics_sagital')
    knee_annotated_path_frontal = session.get('knee_annotated_path_frontal')
    knee_annotated_path_sagital = session.get('knee_annotated_path_sagital')
    if knee_metrics_frontal:
        context['knee_result_frontal'] = {'metrics': knee_metrics_frontal}
    if knee_metrics_sagital:
        context['knee_result_sagital'] = {'metrics': knee_metrics_sagital}
    if knee_annotated_path_frontal and os.path.exists(knee_annotated_path_frontal):
        static_name = os.path.basename(knee_annotated_path_frontal)
        static_path = os.path.join(app.static_folder, static_name)
        if not os.path.exists(static_path):
            import shutil
            shutil.copyfile(knee_annotated_path_frontal, static_path)
        context['knee_image_url_frontal'] = url_for('static', filename=static_name)
        context['now'] = int(time.time())
    if knee_annotated_path_sagital and os.path.exists(knee_annotated_path_sagital):
        static_name = os.path.basename(knee_annotated_path_sagital)
        static_path = os.path.join(app.static_folder, static_name)
        if not os.path.exists(static_path):
            import shutil
            shutil.copyfile(knee_annotated_path_sagital, static_path)
        context['knee_image_url_sagital'] = url_for('static', filename=static_name)
        context['now'] = int(time.time())
    # Postura (frontal y sagital)
    posture_metrics_frontal = session.get('posture_metrics_frontal')
    posture_metrics_sagital = session.get('posture_metrics_sagital')
    posture_annotated_path_frontal = session.get('posture_annotated_path_frontal')
    posture_annotated_path_sagital = session.get('posture_annotated_path_sagital')
    if posture_metrics_frontal:
        context['posture_result_frontal'] = {'metrics': posture_metrics_frontal}
    if posture_metrics_sagital:
        context['posture_result_sagital'] = {'metrics': posture_metrics_sagital}
    if posture_annotated_path_frontal and os.path.exists(posture_annotated_path_frontal):
        static_name = os.path.basename(posture_annotated_path_frontal)
        static_path = os.path.join(app.static_folder, static_name)
        if not os.path.exists(static_path):
            import shutil
            shutil.copyfile(posture_annotated_path_frontal, static_path)
        context['posture_image_url_frontal'] = url_for('static', filename=static_name)
    if posture_annotated_path_sagital and os.path.exists(posture_annotated_path_sagital):
        static_name = os.path.basename(posture_annotated_path_sagital)
        static_path = os.path.join(app.static_folder, static_name)
        if not os.path.exists(static_path):
            import shutil
            shutil.copyfile(posture_annotated_path_sagital, static_path)
        context['posture_image_url_sagital'] = url_for('static', filename=static_name)
    # Cadenas musculares
    chains_metrics = session.get('chains_metrics')
    chains_notes = session.get('chains_notes')
    chains_annotated_path = session.get('chains_annotated_path')
    if chains_metrics:
        context['chains_result'] = {'metrics': chains_metrics, 'notes': chains_notes}
    if chains_annotated_path and os.path.exists(chains_annotated_path):
        context['chains_image_url'] = url_for('static', filename='chains_annotated.jpg')
        static_path = os.path.join(app.static_folder, 'chains_annotated.jpg')
        if not os.path.exists(static_path):
            import shutil
            shutil.copyfile(chains_annotated_path, static_path)
    # Palancas
    lever_result = session.get('lever_result')
    if lever_result:
        context['lever_result'] = lever_result
    return render_template('index.html', **context)

@app.route('/paciente', methods=['GET', 'POST'])
def paciente():
    if request.method == 'POST':
        # Limpiar sesión y archivos de imágenes anteriores
        for key in ['metrics', 'annotated_path', 'knee_metrics', 'knee_annotated_path', 'posture_metrics', 'posture_annotated_path', 'chains_metrics', 'chains_notes', 'chains_annotated_path', 'lever_result']:
            session.pop(key, None)
        # Opcional: limpiar archivos en uploads/
        for fname in os.listdir(UPLOAD_FOLDER):
            if fname.endswith('.jpg') or fname.endswith('.png'):
                try:
                    os.remove(os.path.join(UPLOAD_FOLDER, fname))
                except Exception:
                    pass
        session['paciente'] = dict(request.form)
        return redirect(url_for('consentimiento'))
    return render_template('paciente.html')

@app.route('/consentimiento', methods=['GET', 'POST'])
def consentimiento():
    if request.method == 'POST':
        session['consentimiento'] = True
        return redirect(url_for('index'))
    return render_template('consentimiento.html')

@app.route('/analisis', methods=['POST'])
def analisis():
    imagen = request.files.get('foot_image') or request.files.get('imagen')
    if not imagen:
        flash('Debes subir una imagen')
        return redirect(url_for('index'))
    try:
        resultado = analizar_huella(imagen)
        annotated = resultado['images']['annotated']
        paciente = session.get('paciente', {})
        nombre = paciente.get('nombre', 'anon')
        timestamp = int(time.time())
        out_name = f"foot_annotated_{nombre}_{timestamp}.jpg".replace(' ', '_')
        out_path = os.path.join(UPLOAD_FOLDER, out_name)
        cv2.imwrite(out_path, annotated)
        session['metrics'] = resultado['metrics']
        session['annotated_path'] = out_path
    except Exception as e:
        return redirect(url_for('index'))
    return redirect(url_for('index'))


# Nuevo flujo: ambos planos desde el dashboard
@app.route('/analisis_rodilla', methods=['POST'])
def analisis_rodilla():
    import base64
    import numpy as np
    imagen_frontal = request.files.get('knee_image_frontal')
    imagen_frontal_data = request.form.get('knee_image_frontal_data')
    imagen_sagital = request.files.get('knee_image_sagital')
    imagen_sagital_data = request.form.get('knee_image_sagital_data')
    paciente = session.get('paciente', {})
    nombre = paciente.get('nombre', 'anon')
    timestamp = int(time.time())
    # Frontal
    if imagen_frontal or imagen_frontal_data:
        try:
            if imagen_frontal_data:
                # Procesar base64
                if ',' in imagen_frontal_data:
                    imagen_frontal_data = imagen_frontal_data.split(',')[1]
                img_bytes = base64.b64decode(imagen_frontal_data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                resultado = analizar_rodilla(img, plane='frontal')
            else:
                resultado = analizar_rodilla(imagen_frontal, plane='frontal')
            annotated = resultado['images']['annotated']
            out_name = f"knee_annotated_frontal_{nombre}_{timestamp}.jpg".replace(' ', '_')
            out_path = os.path.join(UPLOAD_FOLDER, out_name)
            cv2.imwrite(out_path, annotated)
            session['knee_metrics_frontal'] = resultado['metrics']
            session['knee_annotated_path_frontal'] = out_path
        except Exception as e:
            flash(f'Error en el análisis frontal: {e}')
    # Sagital
    if imagen_sagital or imagen_sagital_data:
        try:
            if imagen_sagital_data:
                if ',' in imagen_sagital_data:
                    imagen_sagital_data = imagen_sagital_data.split(',')[1]
                img_bytes = base64.b64decode(imagen_sagital_data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                resultado = analizar_rodilla(img, plane='sagital')
            else:
                resultado = analizar_rodilla(imagen_sagital, plane='sagital')
            annotated = resultado['images']['annotated']
            out_name = f"knee_annotated_sagital_{nombre}_{timestamp}.jpg".replace(' ', '_')
            out_path = os.path.join(UPLOAD_FOLDER, out_name)
            cv2.imwrite(out_path, annotated)
            session['knee_metrics_sagital'] = resultado['metrics']
            session['knee_annotated_path_sagital'] = out_path
        except Exception as e:
            flash(f'Error en el análisis sagital: {e}')
    return redirect(url_for('index'))



# Nuevo flujo: ambos planos desde el dashboard
@app.route('/analisis_postura', methods=['POST'])
def analisis_postura():
    import base64
    import numpy as np
    imagen_frontal = request.files.get('posture_image_frontal')
    imagen_frontal_data = request.form.get('posture_image_frontal_data')
    imagen_sagital = request.files.get('posture_image_sagital')
    imagen_sagital_data = request.form.get('posture_image_sagital_data')
    paciente = session.get('paciente', {})
    nombre = paciente.get('nombre', 'anon')
    timestamp = int(time.time())
    # Frontal
    if imagen_frontal or imagen_frontal_data:
        try:
            if imagen_frontal_data:
                if ',' in imagen_frontal_data:
                    imagen_frontal_data = imagen_frontal_data.split(',')[1]
                img_bytes = base64.b64decode(imagen_frontal_data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                resultado = analizar_postura(img)
            else:
                resultado = analizar_postura(imagen_frontal)
            annotated = resultado['images']['annotated']
            out_name = f"posture_annotated_frontal_{nombre}_{timestamp}.jpg".replace(' ', '_')
            out_path = os.path.join(UPLOAD_FOLDER, out_name)
            cv2.imwrite(out_path, annotated)
            session['posture_metrics_frontal'] = resultado['metrics']
            session['posture_annotated_path_frontal'] = out_path
        except Exception as e:
            flash(f'Error en el análisis frontal: {e}')
    # Sagital
    if imagen_sagital or imagen_sagital_data:
        try:
            if imagen_sagital_data:
                if ',' in imagen_sagital_data:
                    imagen_sagital_data = imagen_sagital_data.split(',')[1]
                img_bytes = base64.b64decode(imagen_sagital_data)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                resultado = analizar_postura(img)
            else:
                resultado = analizar_postura(imagen_sagital)
            annotated = resultado['images']['annotated']
            out_name = f"posture_annotated_sagital_{nombre}_{timestamp}.jpg".replace(' ', '_')
            out_path = os.path.join(UPLOAD_FOLDER, out_name)
            cv2.imwrite(out_path, annotated)
            session['posture_metrics_sagital'] = resultado['metrics']
            session['posture_annotated_path_sagital'] = out_path
        except Exception as e:
            flash(f'Error en el análisis sagital: {e}')
    return redirect(url_for('index'))

# --- Restaurar endpoint de cadenas musculares ---
@app.route('/analisis_cadenas', methods=['POST'])
def analisis_cadenas():
    import base64
    chains_image_data = request.form.get('chains_image_data')
    if not chains_image_data:
        flash('Debes capturar una imagen con la cámara')
        return redirect(url_for('index'))
    try:
        # Extraer base64 puro si viene como data:image/png;base64,...
        if ',' in chains_image_data:
            chains_image_data = chains_image_data.split(',')[1]
        img_bytes = base64.b64decode(chains_image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        resultado = analizar_cadenas(image)
        annotated = resultado.images['annotated']
        paciente = session.get('paciente', {})
        nombre = paciente.get('nombre', 'anon')
        timestamp = int(time.time())
        out_name = f"chains_annotated_{nombre}_{timestamp}.jpg".replace(' ', '_')
        out_path = os.path.join(UPLOAD_FOLDER, out_name)
        cv2.imwrite(out_path, annotated)
        session['chains_metrics'] = resultado.metrics
        session['chains_notes'] = resultado.notes
        session['chains_annotated_path'] = out_path
    except Exception as e:
        flash(f'Error en el análisis: {e}')
    return redirect(url_for('index'))

@app.route('/analisis_palanca', methods=['POST'])
def analisis_palanca():
    if request.method == 'POST':
        try:
            peso = float(request.form['peso'])
            articulacion = request.form['articulacion']
            movimiento = request.form['movimiento']
            segmento = request.form['segmento']
            medida_le = float(request.form['medida_le']) / 100  # cm a m
            medida_lr = float(request.form['medida_lr']) / 100
            co = float(request.form['co']) / 1000  # mm a m
            h = float(request.form['h']) / 1000
            resultado = analizar_palanca(peso, articulacion, movimiento, segmento, medida_le, medida_lr, co, h)
            session['lever_result'] = resultado
        except Exception as e:
            flash(f'Error en el análisis: {e}')
    return redirect(url_for('index'))

@app.route('/resultado')
def resultado():
    metrics = session.get('metrics')
    annotated_path = session.get('annotated_path')
    annotated_url = None
    if annotated_path and os.path.exists(annotated_path):
        annotated_url = url_for('static', filename='foot_annotated.jpg')
        # Copiar la imagen al static para servirla
        static_path = os.path.join(app.static_folder, 'foot_annotated.jpg')
        if not os.path.exists(static_path):
            import shutil
            shutil.copyfile(annotated_path, static_path)
    return render_template('resultado.html', metrics=metrics, annotated_url=annotated_url)

@app.route('/resultado_rodilla')
def resultado_rodilla():
    metrics = session.get('knee_metrics')
    annotated_path = session.get('knee_annotated_path')
    annotated_url = None
    if annotated_path and os.path.exists(annotated_path):
        annotated_url = url_for('static', filename='knee_annotated.jpg')
        static_path = os.path.join(app.static_folder, 'knee_annotated.jpg')
        if not os.path.exists(static_path):
            import shutil
            shutil.copyfile(annotated_path, static_path)
    return render_template('resultado_rodilla.html', metrics=metrics, annotated_url=annotated_url)

@app.route('/resultado_postura')
def resultado_postura():
    metrics = session.get('posture_metrics')
    annotated_path = session.get('posture_annotated_path')
    annotated_url = None
    if annotated_path and os.path.exists(annotated_path):
        annotated_url = url_for('static', filename='posture_annotated.jpg')
        static_path = os.path.join(app.static_folder, 'posture_annotated.jpg')
        if not os.path.exists(static_path):
            import shutil
            shutil.copyfile(annotated_path, static_path)
    return render_template('resultado_postura.html', metrics=metrics, annotated_url=annotated_url)

@app.route('/resultado_cadenas')
def resultado_cadenas():
    metrics = session.get('chains_metrics')
    notes = session.get('chains_notes')
    annotated_path = session.get('chains_annotated_path')
    annotated_url = None
    if annotated_path and os.path.exists(annotated_path):
        annotated_url = url_for('static', filename='chains_annotated.jpg')
        static_path = os.path.join(app.static_folder, 'chains_annotated.jpg')
        if not os.path.exists(static_path):
            import shutil
            shutil.copyfile(annotated_path, static_path)
    return render_template('resultado_cadenas.html', metrics=metrics, notes=notes, annotated_url=annotated_url)

@app.route('/descargar_pdf')
def descargar_pdf():
    paciente = session.get('paciente', {})
    plantar_metrics = session.get('metrics', {})
    plantar_img = session.get('annotated_path')
    # Rodilla frontal
    knee_metrics_frontal = session.get('knee_metrics_frontal', {})
    knee_img_frontal = session.get('knee_annotated_path_frontal')
    # Rodilla sagital
    knee_metrics_sagital = session.get('knee_metrics_sagital', {})
    knee_img_sagital = session.get('knee_annotated_path_sagital')
    out_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, 'reporte_paciente.pdf'))

    # Postura frontal y sagital
    posture_metrics_frontal = session.get('posture_metrics_frontal', {})
    posture_img_frontal = session.get('posture_annotated_path_frontal')
    posture_metrics_sagital = session.get('posture_metrics_sagital', {})
    posture_img_sagital = session.get('posture_annotated_path_sagital')

    # Texto de resultados
    results_text = ""
    tiene_datos = False
    if plantar_metrics:
        results_text += f"Índice plantar: {plantar_metrics.get('plantar_index', '')}\nX: {plantar_metrics.get('x_width_px', '')} px\nY: {plantar_metrics.get('y_width_px', '')} px\nClasificación: {plantar_metrics.get('classification', '')}\n\n"
        tiene_datos = True
    if knee_metrics_frontal:
        results_text += f"Rodilla (frontal): Ángulo: {knee_metrics_frontal.get('knee_angle_deg', '')}°, Clasificación: {knee_metrics_frontal.get('classification', '')}\n"
        tiene_datos = True
    if knee_metrics_sagital:
        results_text += f"Rodilla (sagital): Ángulo: {knee_metrics_sagital.get('knee_angle_deg', '')}°, Clasificación: {knee_metrics_sagital.get('classification', '')}\n"
        tiene_datos = True
    if posture_metrics_frontal:
        results_text += f"Postura (frontal): Desviación: {posture_metrics_frontal.get('mean_deviation_px', '')} px, Clasificación: {posture_metrics_frontal.get('classification', '')}\n"
        tiene_datos = True
    if posture_metrics_sagital:
        results_text += f"Postura (sagital): Desviación: {posture_metrics_sagital.get('mean_deviation_px', '')} px, Clasificación: {posture_metrics_sagital.get('classification', '')}\n"
        tiene_datos = True

    if not tiene_datos:
        flash('No hay datos suficientes de los módulos 1, 2 o 3 para generar el PDF. Realiza al menos un análisis clínico antes de generar el reporte.')
        return redirect(url_for('index'))

    from utils.pdf_report import PDFReport
    pdf = PDFReport()
    pdf.add_page()
    pdf.add_patient_data(paciente)
    if plantar_metrics:
        pdf.add_plantar_eval(plantar_metrics)
        if plantar_img:
            pdf.add_analysis_image(plantar_img)
    if knee_metrics_frontal or knee_metrics_sagital:
        pdf.section_title('ANÁLISIS DE RODILLA (FRONTAL)')
        pdf.add_posture_eval(knee_metrics_frontal)
        if knee_img_frontal:
            pdf.add_analysis_image(knee_img_frontal)
        pdf.section_title('ANÁLISIS DE RODILLA (SAGITAL)')
        pdf.add_posture_eval(knee_metrics_sagital)
        if knee_img_sagital:
            pdf.add_analysis_image(knee_img_sagital)
    if posture_metrics_frontal or posture_metrics_sagital:
        pdf.section_title('ANÁLISIS DE POSTURA (FRONTAL)')
        pdf.add_posture_eval(posture_metrics_frontal)
        if posture_img_frontal:
            pdf.add_analysis_image(posture_img_frontal)
        pdf.section_title('ANÁLISIS DE POSTURA (SAGITAL)')
        pdf.add_posture_eval(posture_metrics_sagital)
        if posture_img_sagital:
            pdf.add_analysis_image(posture_img_sagital)
    pdf.add_results(results_text)
    pdf.output(out_path)
    if not os.path.exists(out_path):
        flash('No se pudo generar el PDF. Verifica que haya datos suficientes y que la carpeta uploads exista.')
        return redirect(url_for('index'))
    return send_file(out_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
