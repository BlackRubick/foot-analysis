from __future__ import annotations

import os
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from foot_analysis.analyzer import FootAnalyzer
from knee_analysis.analyzer import KneeAnalyzer
from posture_analysis.analyzer import PostureAnalyzer
from utils.image_io import save_image
from utils.camera_utils import list_cameras


@dataclass
class ModuleState:
    source_image: Optional[np.ndarray] = None
    source_path: Optional[str] = None
    result: Optional[Dict] = None


class CameraCaptureDialog(tk.Toplevel):
    def __init__(self, parent: tk.Tk, camera_index: int = 0):
        super().__init__(parent)
        self.title(f"Captura desde cámara #{camera_index}")
        self.geometry("900x700")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()

        self.current_frame: Optional[np.ndarray] = None
        self.captured_frame: Optional[np.ndarray] = None
        self._tk_image = None
        self.cap = None

        if camera_index is not None and camera_index >= 0:
            # USB: usar OpenCV
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                messagebox.showerror("Cámara", f"No se pudo abrir la cámara #{camera_index}.")
                self.cap = None
                self.captured_frame = None
                self.destroy()
                return
            self.preview = ttk.Label(self)
            self.preview.pack(fill="both", expand=True, padx=12, pady=12)
            controls = ttk.Frame(self)
            controls.pack(fill="x", padx=12, pady=(0, 12))
            ttk.Button(controls, text="Capturar", command=self._capture).pack(side="left", padx=4)
            ttk.Button(controls, text="Cancelar", command=self._close).pack(side="left", padx=4)
            self.protocol("WM_DELETE_WINDOW", self._close)
            self._update_frame()

    def _update_frame(self):
        if self.cap is None:
            return

        ok, frame = self.cap.read()
        if ok:
            self.current_frame = frame.copy()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb).resize((860, 620), Image.Resampling.LANCZOS)
            self._tk_image = ImageTk.PhotoImage(image=image)
            self.preview.configure(image=self._tk_image)

        self.after(20, self._update_frame)

    def _capture(self):
        if self.current_frame is None:
            messagebox.showwarning("Captura", "Aún no hay fotograma disponible.")
            return
        self.captured_frame = self.current_frame.copy()
        self._close()

    def _close(self):
        if self.cap is not None:
            self.cap.release()
        self.destroy()


class BiomechanicsApp:
    def _load_image(self, state: ModuleState, label: ttk.Label, result_label: ttk.Label):
        from tkinter import filedialog, messagebox
        import cv2
        path = filedialog.askopenfilename(
            title="Selecciona imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Imagen", "No se pudo cargar la imagen.")
            return
        state.source_image = img
        state.source_path = path
        state.result = None
        self._set_image_on_label(label, img)
        self._clear_image_label(result_label, "Resultado no disponible")
    def _get_camera_index(self, camera_name: str) -> int:
        """
        Dado el nombre de la cámara (como aparece en el combobox), devuelve el índice correspondiente.
        Si no se encuentra, retorna 0 por defecto.
        """
        for idx, name in self._camera_options:
            if name == camera_name:
                return idx
        # Fallback: intentar extraer el índice del string
        import re
        m = re.search(r"índice (\d+)", camera_name)
        if m:
            return int(m.group(1))
        try:
            return int(camera_name)
        except Exception:
            return 0
    def _build_posture_tab(self):
        controls = ttk.Frame(self.tab_posture, style="Card.TFrame")
        controls.pack(fill="x", padx=10, pady=8)
        ttk.Label(controls, text="Cámara:", style="Body.TLabel").pack(side="left", padx=(0, 2))
        cam_combo = ttk.Combobox(controls, textvariable=self.posture_camera_var, state="readonly", width=32)
        cam_combo['values'] = [name for idx, name in self._camera_options_posture]
        cam_combo.pack(side="left", padx=(0, 8))
        if self._camera_options_posture:
            self.posture_camera_var.set(self._camera_options_posture[0][1])
        ttk.Button(controls, text="Actualizar cámaras", command=lambda: self._update_camera_combo(cam_combo, self.posture_camera_var)).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Cargar imagen", command=lambda: self._load_image(self.posture_state, self.posture_original_lbl, self.posture_result_lbl)).pack(side="left", padx=4)
        ttk.Button(controls, text="Tomar foto", command=lambda: self._capture_image(self.posture_state, self.posture_original_lbl, self.posture_result_lbl, self._get_camera_index(self.posture_camera_var.get()))).pack(side="left", padx=4)
        ttk.Button(controls, text="Analizar", style="Primary.TButton", command=self._analyze_posture).pack(side="left", padx=4)
        ttk.Button(controls, text="Guardar resultado", command=self._save_posture).pack(side="left", padx=4)

        self.posture_original_lbl, self.posture_result_lbl = self._build_common_image_area(self.tab_posture)

        metrics_frame = ttk.LabelFrame(self.tab_posture, text="Métricas", style="Card.TLabelframe")
        metrics_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.posture_metrics_text = tk.Text(metrics_frame, height=5)
        self.posture_metrics_text.pack(fill="x", padx=8, pady=8)
        self._configure_text_widget(self.posture_metrics_text)

    def _generate_pdf_report(self, analysis_type, metrics, image_path, interpretation=""):
        try:
            from utils.pdf_report import generate_pdf_report
            out_dir = self._ensure_output_dir()
            pdf_path = os.path.join(out_dir, f"reporte_{analysis_type}_{self.patient_data.get('Nombre','paciente')}.pdf")
            # Datos dummy para las secciones no usadas
            posture_data = metrics if analysis_type == "postura" else {}
            plantar_data = metrics if analysis_type == "pie" else {}
            lever_data = metrics if analysis_type == "palanca" else {}
            # El generador espera: patient_data, posture_data, plantar_data, results_text, img_path, out_path
            generate_pdf_report(
                self.patient_data,
                posture_data,
                plantar_data,
                interpretation,
                image_path,
                pdf_path
            )
            messagebox.showinfo("PDF generado", f"Reporte PDF guardado en:\n{pdf_path}")
        except Exception as e:
            messagebox.showerror("PDF", f"Error al generar PDF: {e}")

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Sistema de Análisis Biomecánico")
        self.root.geometry("1450x860")
        self.root.minsize(1280, 760)

        self.bg_main = "#0f172a"
        self.bg_card = "#111827"
        self.bg_soft = "#1f2937"
        self.text_primary = "#e5e7eb"
        self.text_muted = "#9ca3af"
        self.accent = "#22c55e"
        self.accent_alt = "#0ea5e9"

        self._setup_styles()

        self.foot_analyzer = FootAnalyzer()
        self.knee_analyzer = None
        self.posture_analyzer = None

        self.foot_state = ModuleState()
        self.knee_state = ModuleState()
        self.posture_state = ModuleState()

        self.save_dir_var = tk.StringVar(value="outputs")
        self.knee_plane_var = tk.StringVar(value="frontal")
        self.foot_stage_var = tk.StringVar(value="annotated")
        self.status_var = tk.StringVar(value="Listo")

        # Datos generales del paciente
        self.patient_data = {
            "Nombre": "",
            "Edad": "",
            "Sexo": "",
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
            "Presencia de cicatrices": "No"
        }

        # Variables para cámara por módulo
        self.foot_camera_var = tk.StringVar()
        self.knee_camera_var = tk.StringVar()
        self.posture_camera_var = tk.StringVar()
        self._camera_options = []
        self._refresh_cameras()

    def _show_patient_form(self):
        form = tk.Toplevel(self.root)
        form.title("Datos generales del paciente")
        form.geometry("500x600")
        form.transient(self.root)
        form.grab_set()
        entries = {}
        row = 0
        for k in self.patient_data:
            tk.Label(form, text=k+":").grid(row=row, column=0, sticky="w", padx=8, pady=4)
            e = tk.Entry(form, width=40)
            e.insert(0, self.patient_data[k])
            e.grid(row=row, column=1, padx=8, pady=4)
            entries[k] = e
            row += 1
        def save_and_close():
            for k in self.patient_data:
                self.patient_data[k] = entries[k].get()
            form.destroy()
        tk.Button(form, text="Guardar", command=save_and_close).grid(row=row, column=0, columnspan=2, pady=12)

    def _refresh_cameras(self):
        # Solo webcams USB
        self._camera_options = list_cameras()
        self._camera_options_knee = self._camera_options
        self._camera_options_posture = self._camera_options
        if self._camera_options:
            self.foot_camera_var.set(self._camera_options[0][1])
            self.knee_camera_var.set(self._camera_options[0][1])
            self.posture_camera_var.set(self._camera_options[0][1])
        else:
            self.foot_camera_var.set("")
            self.knee_camera_var.set("")
            self.posture_camera_var.set("")
        self._build_ui()

    def _setup_styles(self):
        self.root.configure(bg=self.bg_main)

        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("App.TFrame", background=self.bg_main)
        style.configure("Card.TFrame", background=self.bg_card)
        style.configure("Card.TLabelframe", background=self.bg_card, foreground=self.text_primary)
        style.configure("Card.TLabelframe.Label", background=self.bg_card, foreground=self.text_primary, font=("Segoe UI", 10, "bold"))

        style.configure("Title.TLabel", background=self.bg_main, foreground=self.text_primary, font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", background=self.bg_main, foreground=self.text_muted, font=("Segoe UI", 10))
        style.configure("Body.TLabel", background=self.bg_card, foreground=self.text_primary, font=("Segoe UI", 10))
        style.configure("Hint.TLabel", background=self.bg_card, foreground=self.text_muted, font=("Segoe UI", 9))
        style.configure("Status.TLabel", background=self.bg_main, foreground=self.text_muted, font=("Segoe UI", 10))

        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 6))
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 7))

        style.configure("TEntry", fieldbackground=self.bg_soft, foreground=self.text_primary)
        style.configure("TCombobox", fieldbackground=self.bg_soft, foreground=self.text_primary)

        style.configure("TNotebook", background=self.bg_main, borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8), font=("Segoe UI", 10, "bold"))

    @staticmethod
    def _configure_text_widget(widget: tk.Text):
        widget.configure(
            bg="#0b1220",
            fg="#d1d5db",
            insertbackground="#d1d5db",
            relief="flat",
            font=("Consolas", 11),
            padx=8,
            pady=8,
        )

    def _get_knee_analyzer(self) -> KneeAnalyzer:
        if self.knee_analyzer is None:
            self.knee_analyzer = KneeAnalyzer()
        return self.knee_analyzer

    def _get_posture_analyzer(self) -> PostureAnalyzer:
        if self.posture_analyzer is None:
            self.posture_analyzer = PostureAnalyzer()
        return self.posture_analyzer

    def _build_ui(self):

        top = ttk.Frame(self.root, style="App.TFrame")
        top.pack(fill="x", padx=14, pady=(12, 8))

        title_col = ttk.Frame(top, style="App.TFrame")
        title_col.pack(side="left", fill="x", expand=True)
        ttk.Label(title_col, text="Biomech Vision Suite", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            title_col,
            text="Análisis de huella plantar, rodilla y postura con OpenCV + MediaPipe",
            style="Subtitle.TLabel",
        ).pack(anchor="w", pady=(2, 0))

        output_card = ttk.Frame(top, style="Card.TFrame", padding=10)
        output_card.pack(side="right")

        ttk.Label(output_card, text="Carpeta de salida", style="Body.TLabel").pack(anchor="w")
        row = ttk.Frame(output_card, style="Card.TFrame")
        row.pack(fill="x", pady=(4, 0))
        ttk.Entry(row, textvariable=self.save_dir_var, width=32).pack(side="left", padx=(0, 6))
        ttk.Button(row, text="Elegir", command=self._ensure_output_dir).pack(side="left")
        ttk.Button(row, text="Datos paciente", command=self._show_patient_form).pack(side="left", padx=(8,0))

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=14, pady=8)

        self.tab_foot = ttk.Frame(notebook)
        self.tab_knee = ttk.Frame(notebook)
        self.tab_posture = ttk.Frame(notebook)


        notebook.add(self.tab_foot, text="Baropodometría (Pie)")
        notebook.add(self.tab_knee, text="Rodilla")
        notebook.add(self.tab_posture, text="Postura")
        # Nueva pestaña para palancas
        self.tab_lever = ttk.Frame(notebook)
        notebook.add(self.tab_lever, text="Palancas y Torque")

        self._build_foot_tab()
        self._build_knee_tab()
        self._build_posture_tab()
        self._build_lever_tab()

    def _build_lever_tab(self):
        # Limpiar la pestaña antes de crear controles
        for child in self.tab_lever.winfo_children():
            child.destroy()

        controls = ttk.Frame(self.tab_lever, style="Card.TFrame")
        controls.pack(fill="x", padx=10, pady=8)
        ttk.Label(controls, text="Peso (kg):", style="Body.TLabel").pack(side="left", padx=(0, 2))
        self.lever_weight_var = tk.StringVar()
        ttk.Entry(controls, textvariable=self.lever_weight_var, width=8).pack(side="left", padx=(0, 8))
        ttk.Label(controls, text="Segmento:", style="Body.TLabel").pack(side="left", padx=(0, 2))
        self.lever_segment_var = tk.StringVar()
        segmentos_es = ["cabeza", "tronco", "brazo_superior", "antebrazo", "mano", "muslo", "pierna", "pie"]
        self.lever_segment_combo = ttk.Combobox(controls, textvariable=self.lever_segment_var, values=segmentos_es, state="readonly", width=14)
        self.lever_segment_combo.pack(side="left", padx=(0, 8))
        ttk.Label(controls, text="LE (cm):", style="Body.TLabel").pack(side="left", padx=(0, 2))
        self.lever_le_var = tk.StringVar()
        ttk.Entry(controls, textvariable=self.lever_le_var, width=6).pack(side="left", padx=(0, 8))
        ttk.Label(controls, text="LR (cm):", style="Body.TLabel").pack(side="left", padx=(0, 2))
        self.lever_lr_var = tk.StringVar()
        ttk.Entry(controls, textvariable=self.lever_lr_var, width=6).pack(side="left", padx=(0, 8))
        ttk.Label(controls, text="CO (mm):", style="Body.TLabel").pack(side="left", padx=(0, 2))
        self.lever_co_var = tk.StringVar()
        ttk.Entry(controls, textvariable=self.lever_co_var, width=6).pack(side="left", padx=(0, 8))
        ttk.Label(controls, text="H (mm):", style="Body.TLabel").pack(side="left", padx=(0, 2))
        self.lever_h_var = tk.StringVar()
        ttk.Entry(controls, textvariable=self.lever_h_var, width=6).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Calcular", style="Primary.TButton", command=self._calculate_lever).pack(side="left", padx=8)

        # Área de resultados
        self.lever_result_text = tk.Text(self.tab_lever, height=12)
        self.lever_result_text.pack(fill="both", padx=10, pady=10)
        self._configure_text_widget(self.lever_result_text)
    def _load_lever_image(self):
        from tkinter import filedialog
        import cv2
        path = filedialog.askopenfilename(
            title="Selecciona imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Imagen", "No se pudo cargar la imagen.")
            return
        self.lever_captured_image = img
        self.lever_points = []
        self._draw_lever_image()
        self.lever_result_text.delete("1.0", tk.END)
        self.lever_result_text.insert(tk.END, "Imagen cargada. Haz clic en 4 puntos anatómicos: \n1) Fulcro, 2) Inserción esfuerzo, 3) Punto resistencia, 4) Inserción antagonista.\n")


        # (Se elimina la barra superior de controles y canvas de imagen para dejar solo los campos manuales)

        # Resultados (debe ir después de los controles y canvas)

    def _capture_lever_photo(self):
        idx = self._get_camera_index(self.lever_camera_var.get())
        if idx is None:
            messagebox.showerror("Cámara", "Selecciona una cámara válida.")
            return
        dialog = CameraCaptureDialog(self.root, camera_index=idx)
        self.root.wait_window(dialog)
        if getattr(dialog, "captured_frame", None) is None:
            return
        self.lever_captured_image = dialog.captured_frame
        self.lever_points = []
        self._draw_lever_image()
        self.lever_result_text.delete("1.0", tk.END)
        self.lever_result_text.insert(tk.END, "Imagen capturada. Haz clic en 4 puntos anatómicos: \n1) Fulcro, 2) Inserción esfuerzo, 3) Punto resistencia, 4) Inserción antagonista.\n")

    def _draw_lever_image(self):
        if self.lever_captured_image is None:
            self.lever_image_canvas.delete("all")
            return
        import cv2
        from PIL import Image, ImageTk
        img = self.lever_captured_image.copy()
        h, w = img.shape[:2]
        scale = min(640 / w, 480 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Dibujar puntos
        for i, (x, y) in enumerate(self.lever_points):
            cv2.circle(img_resized, (int(x), int(y)), 7, (0, 255, 0), -1)
            cv2.putText(img_resized, str(i+1), (int(x)+8, int(y)-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        self.lever_tk_img = ImageTk.PhotoImage(pil_img)
        self.lever_image_canvas.config(width=new_w, height=new_h)
        self.lever_image_canvas.delete("all")
        self.lever_image_canvas.create_image(0, 0, anchor="nw", image=self.lever_tk_img)
        # Redibujar puntos encima
        for i, (x, y) in enumerate(self.lever_points):
            self.lever_image_canvas.create_oval(x-7, y-7, x+7, y+7, fill="#22c55e", outline="white", width=2)
            self.lever_image_canvas.create_text(x+14, y-10, text=str(i+1), fill="yellow", font=("Segoe UI", 12, "bold"))

    def _on_lever_image_click(self, event):
        if self.lever_captured_image is None:
            return
        # Ajustar coordenadas al tamaño de la imagen mostrada
        canvas_w = self.lever_image_canvas.winfo_width()
        canvas_h = self.lever_image_canvas.winfo_height()
        img_h, img_w = self.lever_captured_image.shape[:2]
        scale = min(640 / img_w, 480 / img_h)
        x = event.x
        y = event.y
        self.lever_points.append((x, y))
        self._draw_lever_image()
        if len(self.lever_points) == 4:
            self._calculate_lever_from_points()

    def _calculate_lever_from_points(self):
        import math
        # Asignación: 0=Fulcro, 1=Esfuerzo, 2=Resistencia, 3=Antagonista
        f, e, r, a = self.lever_points
        # LE: distancia Fulcro-Esfuerzo
        le = math.dist(f, e) / 10  # px a cm (aprox, depende de escala real)
        # LR: distancia Fulcro-Resistencia
        lr = math.dist(f, r) / 10
        # CO: distancia perpendicular entre esfuerzo y antagonista
        co = math.dist(e, a)  # px
        # H: distancia esfuerzo-antagonista
        h = math.dist(e, a)  # px
        # Mostrar resultados y permitir editar escala real si se desea
        self.lever_result_text.delete("1.0", tk.END)
        self.lever_result_text.insert(tk.END, f"LE (cm): {le:.2f}\nLR (cm): {lr:.2f}\nCO (px): {co:.2f}\nH (px): {h:.2f}\n\nPuedes ajustar la escala real si tienes referencia de longitud en la imagen.\nLuego, usa estos valores para el cálculo biomecánico.")

    def _calculate_lever(self):
        if not hasattr(self, "lever_result_text") or self.lever_result_text is None:
            messagebox.showerror("Error", "El área de resultados no está disponible. Por favor, cambia de pestaña y vuelve a Palancas y Torque.")
            return
        try:
            from lever_analysis.calculations import mechanical_advantage, interpret_mechanical_advantage, calculate_alpha, round_rule, calculate_mass, calculate_force, calculate_torque
            import json
            from pathlib import Path
            segments_path = Path(__file__).parent.parent / "lever_analysis/data/segments.json"
            with open(segments_path, 'r', encoding='utf-8') as f:
                segments = json.load(f)
            peso = float(self.lever_weight_var.get())
            segmento = self.lever_segment_var.get().strip()
            if segmento not in segments:
                self.lever_result_text.delete("1.0", tk.END)
                self.lever_result_text.insert(tk.END, f"Segmento no válido. Usa uno de: {', '.join(segments.keys())}\n")
                return
            le = float(self.lever_le_var.get()) / 100
            lr = float(self.lever_lr_var.get()) / 100
            co = float(self.lever_co_var.get()) / 1000
            h = float(self.lever_h_var.get()) / 1000
            vm = mechanical_advantage(le, lr)
            vm_interp = interpret_mechanical_advantage(vm)
            alpha = calculate_alpha(co, h)
            alpha_r = round_rule(alpha)
            masa = calculate_mass(peso, segments[segmento])
            fuerza = calculate_force(masa)
            torque = calculate_torque(fuerza, le, alpha)
            result = (
                f"Ventaja mecánica: {vm:.2f} ({vm_interp})\n"
                f"Ángulo alfa: {alpha:.2f}° (redondeado: {alpha_r}°)\n"
                f"Masa segmento: {masa:.2f} kg\n"
                f"Fuerza: {fuerza:.2f} N\n"
                f"Torque: {torque:.2f} Nm\n"
            )
            self.lever_result_text.delete("1.0", tk.END)
            self.lever_result_text.insert(tk.END, result)
        except Exception as e:
            self.lever_result_text.delete("1.0", tk.END)
            self.lever_result_text.insert(tk.END, f"Error: {e}\n")

        status_frame = ttk.Frame(self.root, style="App.TFrame")
        status_frame.pack(fill="x", padx=14, pady=(0, 10))
        ttk.Label(status_frame, textvariable=self.status_var, anchor="w", style="Status.TLabel").pack(fill="x")

    def _set_status(self, text: str, busy: bool = False):
        self.status_var.set(text)
        self.root.config(cursor="watch" if busy else "")
        self.root.update_idletasks()

    def _clear_status(self):
        self._set_status("Listo", busy=False)

    def _build_common_image_area(self, tab: ttk.Frame):
        image_area = ttk.Frame(tab, style="Card.TFrame")
        image_area.pack(fill="both", expand=True, padx=10, pady=10)

        left = ttk.LabelFrame(image_area, text="Imagen original", style="Card.TLabelframe")
        right = ttk.LabelFrame(image_area, text="Resultado", style="Card.TLabelframe")

        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right.pack(side="left", fill="both", expand=True, padx=(6, 0))

        left_label = ttk.Label(left, text="Sin imagen cargada", style="Hint.TLabel", anchor="center", justify="center")
        right_label = ttk.Label(right, text="Resultado no disponible", style="Hint.TLabel", anchor="center", justify="center")
        left_label.pack(fill="both", expand=True, padx=8, pady=8)
        right_label.pack(fill="both", expand=True, padx=8, pady=8)

        return left_label, right_label


    def _build_foot_tab(self):
        controls = ttk.Frame(self.tab_foot, style="Card.TFrame")
        controls.pack(fill="x", padx=10, pady=8)
        ttk.Label(controls, text="Cámara:", style="Body.TLabel").pack(side="left", padx=(0, 2))
        cam_combo = ttk.Combobox(controls, textvariable=self.foot_camera_var, state="readonly", width=32)
        cam_combo['values'] = [name for idx, name in self._camera_options]
        cam_combo.pack(side="left", padx=(0, 2))
        ttk.Button(controls, text="Actualizar cámaras", command=lambda: self._update_camera_combo(cam_combo, self.foot_camera_var)).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Cargar imagen", command=lambda: self._load_image(self.foot_state, self.foot_original_lbl, self.foot_result_lbl)).pack(side="left", padx=4)
        ttk.Button(controls, text="Tomar foto", command=lambda: self._capture_image(self.foot_state, self.foot_original_lbl, self.foot_result_lbl, self._get_camera_index(self.foot_camera_var.get()))).pack(side="left", padx=4)
        ttk.Label(controls, text="Vista:", style="Body.TLabel").pack(side="left", padx=(15, 4))
        stage_cb = ttk.Combobox(
            controls,
            textvariable=self.foot_stage_var,
            values=["annotated", "gray", "binary", "clean", "edges", "rotated_widths"],
            state="readonly",
            width=20,
        )
        stage_cb.pack(side="left", padx=4)
        stage_cb.bind("<<ComboboxSelected>>", lambda _e: self._refresh_foot_view())
        ttk.Button(controls, text="Analizar", style="Primary.TButton", command=self._analyze_foot).pack(side="left", padx=4)
        ttk.Button(controls, text="Guardar resultados", command=self._save_foot).pack(side="left", padx=4)

        self.foot_original_lbl, self.foot_result_lbl = self._build_common_image_area(self.tab_foot)

        metrics_frame = ttk.LabelFrame(self.tab_foot, text="Métricas", style="Card.TLabelframe")
        metrics_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.foot_metrics_text = tk.Text(metrics_frame, height=5)
        self.foot_metrics_text.pack(fill="x", padx=8, pady=8)
        self._configure_text_widget(self.foot_metrics_text)


    def _build_knee_tab(self):
        controls = ttk.Frame(self.tab_knee, style="Card.TFrame")
        controls.pack(fill="x", padx=10, pady=8)

        ttk.Label(controls, text="Cámara:", style="Body.TLabel").pack(side="left", padx=(0, 2))
        cam_combo = ttk.Combobox(controls, textvariable=self.knee_camera_var, state="readonly", width=32)
        cam_combo['values'] = [name for idx, name in self._camera_options_knee]
        cam_combo.pack(side="left", padx=(0, 2))
        ttk.Button(controls, text="Actualizar cámaras", command=lambda: self._update_camera_combo(cam_combo, self.knee_camera_var)).pack(side="left", padx=(0, 8))


        ttk.Button(controls, text="Cargar imagen", command=lambda: self._load_image(self.knee_state, self.knee_original_lbl, self.knee_result_lbl)).pack(side="left", padx=4)
        ttk.Button(controls, text="Tomar foto", command=lambda: self._capture_image(self.knee_state, self.knee_original_lbl, self.knee_result_lbl, self._get_camera_index(self.knee_camera_var.get()))).pack(side="left", padx=4)

        ttk.Label(controls, text="Plano:", style="Body.TLabel").pack(side="left", padx=(12, 4))
        ttk.Combobox(controls, textvariable=self.knee_plane_var, values=["frontal", "sagital"], state="readonly", width=12).pack(side="left", padx=4)

        ttk.Button(controls, text="Analizar", style="Primary.TButton", command=self._analyze_knee).pack(side="left", padx=4)
        ttk.Button(controls, text="Guardar resultado", command=self._save_knee).pack(side="left", padx=4)

        self.knee_original_lbl, self.knee_result_lbl = self._build_common_image_area(self.tab_knee)

        metrics_frame = ttk.LabelFrame(self.tab_knee, text="Métricas", style="Card.TLabelframe")
        metrics_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.knee_metrics_text = tk.Text(metrics_frame, height=5)
        self.knee_metrics_text.pack(fill="x", padx=8, pady=8)
        self._configure_text_widget(self.knee_metrics_text)
        def _build_lever_tab(self):
            # Limpiar widgets antiguos de la pestaña antes de reconstruir
            # Limpiar widgets antiguos y referencias
            for child in self.tab_lever.winfo_children():
                child.destroy()
            self.lever_result_text = None
            self.lever_image_canvas = None
            self.lever_weight_var = None
            self.lever_segment_var = None
            self.lever_segment_combo = None
            self.lever_camera_var = None
            self.lever_le_var = None
            self.lever_lr_var = None
            self.lever_co_var = None
            self.lever_h_var = None
            self.lever_points = []
            self.lever_captured_image = None
            self.lever_tk_img = None

            # Resultados (debe ir primero para estar disponible en todos los métodos)
            self.lever_result_text = tk.Text(self.tab_lever, height=12)
            self.lever_result_text.pack(fill="both", padx=10, pady=10)
            self._configure_text_widget(self.lever_result_text)

            controls1 = ttk.Frame(self.tab_lever, style="Card.TFrame")
            controls1.pack(fill="x", padx=10, pady=(8, 2))
            ttk.Label(controls1, text="Peso de la persona (kg):", style="Body.TLabel").pack(side="left", padx=(0, 2))
            self.lever_weight_var = tk.StringVar()
            ttk.Entry(controls1, textvariable=self.lever_weight_var, width=8).pack(side="left", padx=(0, 8))
            ttk.Label(controls1, text="Segmento:", style="Body.TLabel").pack(side="left", padx=(0, 2))
            self.lever_segment_var = tk.StringVar()
            segmentos_es = ["cabeza", "tronco", "brazo_superior", "antebrazo", "mano", "muslo", "pierna", "pie"]
            self.lever_segment_combo = ttk.Combobox(controls1, textvariable=self.lever_segment_var, values=segmentos_es, state="readonly", width=14)
            self.lever_segment_combo.pack(side="left", padx=(0, 8))

            controls2 = ttk.Frame(self.tab_lever, style="Card.TFrame")
            controls2.pack(fill="x", padx=10, pady=(2, 2))
            ttk.Label(controls2, text="Cámara:", style="Body.TLabel").pack(side="left", padx=(0, 2))
            self.lever_camera_var = tk.StringVar()
            cam_combo = ttk.Combobox(controls2, textvariable=self.lever_camera_var, state="readonly", width=32)
            cam_combo['values'] = [name for idx, name in self._camera_options]
            cam_combo.pack(side="left", padx=(0, 8))
            if self._camera_options:
                self.lever_camera_var.set(self._camera_options[0][1])
            ttk.Button(controls2, text="Actualizar cámaras", command=lambda: self._update_camera_combo(cam_combo, self.lever_camera_var)).pack(side="left", padx=(0, 8))
            ttk.Button(controls2, text="Tomar foto", style="Primary.TButton", command=self._capture_lever_photo).pack(side="left", padx=8)
            ttk.Button(controls2, text="Subir imagen", command=self._load_lever_image).pack(side="left", padx=4)

            controls3 = ttk.Frame(self.tab_lever, style="Card.TFrame")
            controls3.pack(fill="x", padx=10, pady=(2, 8))
            ttk.Label(controls3, text="LE (cm):", style="Body.TLabel").pack(side="left", padx=(0, 2))
            self.lever_le_var = tk.StringVar()
            ttk.Entry(controls3, textvariable=self.lever_le_var, width=6).pack(side="left", padx=(0, 8))
            ttk.Label(controls3, text="LR (cm):", style="Body.TLabel").pack(side="left", padx=(0, 2))
            self.lever_lr_var = tk.StringVar()
            ttk.Entry(controls3, textvariable=self.lever_lr_var, width=6).pack(side="left", padx=(0, 8))
            ttk.Label(controls3, text="CO (mm):", style="Body.TLabel").pack(side="left", padx=(0, 2))
            self.lever_co_var = tk.StringVar()
            ttk.Entry(controls3, textvariable=self.lever_co_var, width=6).pack(side="left", padx=(0, 8))
            ttk.Label(controls3, text="H (mm):", style="Body.TLabel").pack(side="left", padx=(0, 2))
            self.lever_h_var = tk.StringVar()
            ttk.Entry(controls3, textvariable=self.lever_h_var, width=6).pack(side="left", padx=(0, 8))
            ttk.Button(controls3, text="Calcular", style="Primary.TButton", command=self._calculate_lever).pack(side="left", padx=8)

            # Área para mostrar la imagen capturada y seleccionar puntos
            self.lever_image_canvas = tk.Canvas(self.tab_lever, width=640, height=480, bg="#0b1220", highlightthickness=0)
            self.lever_image_canvas.pack(padx=10, pady=(10, 0))
            self.lever_image_canvas.bind("<Button-1>", self._on_lever_image_click)
            self.lever_points = []  # [(x, y), ...]
            self.lever_captured_image = None
            self.lever_tk_img = None
        # Ejecutar análisis automáticamente si se capturó imagen
        if self.posture_state.source_image is not None:
            try:
                self._set_status("Analizando postura...", busy=True)
                self.posture_state.result = self._get_posture_analyzer().analyze(self.posture_state.source_image)
                self._set_image_on_label(self.posture_result_lbl, self.posture_state.result["images"]["annotated"])
                m = self.posture_state.result["metrics"]
                text = (
                    f"Lado analizado: {m['side']}\n"
                    f"Desviación media: {m['mean_deviation_px']:.2f} px\n"
                    f"Clasificación: {m['classification']}"
                )
                self._write_metrics(self.posture_metrics_text, text)
            except Exception as e:
                messagebox.showerror("Postura", f"Error en el análisis: {e}")
            finally:
                def _build_lever_tab(self):
                    # Limpiar widgets antiguos de la pestaña antes de reconstruir
                    for child in self.tab_lever.winfo_children():
                        child.destroy()

                    # Resultados (debe ir primero para estar disponible en todos los métodos)
                    self.lever_result_text = tk.Text(self.tab_lever, height=12)
                    self.lever_result_text.pack(fill="both", padx=10, pady=10)
                    self._configure_text_widget(self.lever_result_text)

                    controls1 = ttk.Frame(self.tab_lever, style="Card.TFrame")
                    controls1.pack(fill="x", padx=10, pady=(8, 2))
                    ttk.Label(controls1, text="Peso de la persona (kg):", style="Body.TLabel").pack(side="left", padx=(0, 2))
                    self.lever_weight_var = tk.StringVar()
                    ttk.Entry(controls1, textvariable=self.lever_weight_var, width=8).pack(side="left", padx=(0, 8))
                    ttk.Label(controls1, text="Segmento:", style="Body.TLabel").pack(side="left", padx=(0, 2))
                    self.lever_segment_var = tk.StringVar()
                    segmentos_es = ["cabeza", "tronco", "brazo_superior", "antebrazo", "mano", "muslo", "pierna", "pie"]
                    self.lever_segment_combo = ttk.Combobox(controls1, textvariable=self.lever_segment_var, values=segmentos_es, state="readonly", width=14)
                    self.lever_segment_combo.pack(side="left", padx=(0, 8))

                    controls2 = ttk.Frame(self.tab_lever, style="Card.TFrame")
                    controls2.pack(fill="x", padx=10, pady=(2, 2))
                    ttk.Label(controls2, text="Cámara:", style="Body.TLabel").pack(side="left", padx=(0, 2))
                    self.lever_camera_var = tk.StringVar()
                    cam_combo = ttk.Combobox(controls2, textvariable=self.lever_camera_var, state="readonly", width=32)
                    cam_combo['values'] = [name for idx, name in self._camera_options]
                    cam_combo.pack(side="left", padx=(0, 8))
                    if self._camera_options:
                        self.lever_camera_var.set(self._camera_options[0][1])
                    ttk.Button(controls2, text="Actualizar cámaras", command=lambda: self._update_camera_combo(cam_combo, self.lever_camera_var)).pack(side="left", padx=(0, 8))
                    ttk.Button(controls2, text="Tomar foto", style="Primary.TButton", command=self._capture_lever_photo).pack(side="left", padx=8)
                    ttk.Button(controls2, text="Subir imagen", command=self._load_lever_image).pack(side="left", padx=4)

                    controls3 = ttk.Frame(self.tab_lever, style="Card.TFrame")
                    controls3.pack(fill="x", padx=10, pady=(2, 8))
                    ttk.Label(controls3, text="LE (cm):", style="Body.TLabel").pack(side="left", padx=(0, 2))
                    self.lever_le_var = tk.StringVar()
                    ttk.Entry(controls3, textvariable=self.lever_le_var, width=6).pack(side="left", padx=(0, 8))
                    ttk.Label(controls3, text="LR (cm):", style="Body.TLabel").pack(side="left", padx=(0, 2))
                    self.lever_lr_var = tk.StringVar()
                    ttk.Entry(controls3, textvariable=self.lever_lr_var, width=6).pack(side="left", padx=(0, 8))
                    ttk.Label(controls3, text="CO (mm):", style="Body.TLabel").pack(side="left", padx=(0, 2))
                    self.lever_co_var = tk.StringVar()
                    ttk.Entry(controls3, textvariable=self.lever_co_var, width=6).pack(side="left", padx=(0, 8))
                    ttk.Label(controls3, text="H (mm):", style="Body.TLabel").pack(side="left", padx=(0, 2))
                    self.lever_h_var = tk.StringVar()
                    ttk.Entry(controls3, textvariable=self.lever_h_var, width=6).pack(side="left", padx=(0, 8))
                    ttk.Button(controls3, text="Calcular", style="Primary.TButton", command=self._calculate_lever).pack(side="left", padx=8)

                    # Área para mostrar la imagen capturada y seleccionar puntos
                    self.lever_image_canvas = tk.Canvas(self.tab_lever, width=640, height=480, bg="#0b1220", highlightthickness=0)
                    self.lever_image_canvas.pack(padx=10, pady=(10, 0))
                    self.lever_image_canvas.bind("<Button-1>", self._on_lever_image_click)
                    self.lever_points = []  # [(x, y), ...]
                    self.lever_captured_image = None
                    self.lever_tk_img = None

    def _capture_image(self, state: ModuleState, label: ttk.Label, result_label: ttk.Label, camera_index: int = 0):
        dialog = CameraCaptureDialog(self.root, camera_index=camera_index)
        self.root.wait_window(dialog)
        if getattr(dialog, "captured_frame", None) is None:
            return

        state.source_image = dialog.captured_frame
        state.source_path = None
        state.result = None
        self._set_image_on_label(label, state.source_image)
        self._clear_image_label(result_label, "Resultado no disponible")

    @staticmethod
    def _to_tk_image(image_bgr: np.ndarray, max_w: int = 640, max_h: int = 460):
        h, w = image_bgr.shape[:2]
        scale = min(max_w / max(w, 1), max_h / max(h, 1))
        scale = min(scale, 1.0)

        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        return ImageTk.PhotoImage(image=image)

    def _set_image_on_label(self, label: ttk.Label, image_bgr: np.ndarray):
        tk_img = self._to_tk_image(image_bgr)
        label.configure(image=tk_img, text="")
        label.image = tk_img

    @staticmethod
    def _clear_image_label(label: ttk.Label, text: str):
        label.configure(image="", text=text)
        label.image = None

    def _analyze_foot(self):
        if self.foot_state.source_image is None:
            messagebox.showwarning("Baropodometría", "Primero carga o captura una imagen.")
            return
        try:
            self._set_status("Analizando huella plantar...", busy=True)
            self.foot_state.result = self.foot_analyzer.analyze(self.foot_state.source_image)
            self._refresh_foot_view()
            m = self.foot_state.result["metrics"]
            text = (
                f"Índice plantar: {m['plantar_index']:.2f}\n"
                f"X (antepié): {m['x_width_px']:.2f} px\n"
                f"Y (arco plantar): {m['y_width_px']:.2f} px\n"
                f"Clasificación: {m['classification']}"
            )
            self._write_metrics(self.foot_metrics_text, text)
            # Guardar imagen temporal para PDF
            out_dir = self._ensure_output_dir()
            img_path = os.path.join(out_dir, "foot_pdf_temp.jpg")
            save_image(img_path, self.foot_state.result["images"]["annotated"])
            self._generate_pdf_report("pie", m, img_path, text)
        except Exception as e:
            messagebox.showerror("Baropodometría", f"Error: {e}")
        finally:
            self._clear_status()

    def _refresh_foot_view(self):
        if not self.foot_state.result:
            return

        stage = self.foot_stage_var.get()
        image = self.foot_state.result["images"].get(stage)
        if image is None:
            return

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self._set_image_on_label(self.foot_result_lbl, image)

    def _analyze_knee(self):
        if self.knee_state.source_image is None:
            messagebox.showwarning("Rodilla", "Primero carga o captura una imagen.")
            return
        try:
            if self.knee_analyzer is None:
                self._set_status("Inicializando modelo de rodilla (primera vez puede tardar y descargar modelo)...", busy=True)
            else:
                self._set_status("Analizando rodilla...", busy=True)
            self.knee_state.result = self._get_knee_analyzer().analyze(
                self.knee_state.source_image,
                plane=self.knee_plane_var.get(),
            )
            self._set_image_on_label(self.knee_result_lbl, self.knee_state.result["images"]["annotated"])
            m = self.knee_state.result["metrics"]
            text = (
                f"Plano: {m['plane']}\n"
                f"Lado: {m['side']}\n"
                f"Ángulo de rodilla: {m['knee_angle_deg']:.2f}°\n"
                f"Clasificación: {m['classification']}"
            )
            self._write_metrics(self.knee_metrics_text, text)
            out_dir = self._ensure_output_dir()
            img_path = os.path.join(out_dir, "knee_pdf_temp.jpg")
            save_image(img_path, self.knee_state.result["images"]["annotated"])
            self._generate_pdf_report("rodilla", m, img_path, text)
        except Exception as e:
            messagebox.showerror("Rodilla", f"Error: {e}")
        finally:
            self._clear_status()

    def _analyze_posture(self):
        if self.posture_state.source_image is None:
            messagebox.showwarning("Postura", "Primero carga o captura una imagen.")
            return
        try:
            if self.posture_analyzer is None:
                self._set_status("Inicializando modelo postural (primera vez puede tardar y descargar modelo)...", busy=True)
            else:
                self._set_status("Analizando postura...", busy=True)
            self.posture_state.result = self._get_posture_analyzer().analyze(self.posture_state.source_image)
            self._set_image_on_label(self.posture_result_lbl, self.posture_state.result["images"]["annotated"])
            m = self.posture_state.result["metrics"]
            text = (
                f"Lado analizado: {m['side']}\n"
                f"Desviación media: {m['mean_deviation_px']:.2f} px\n"
                f"Clasificación: {m['classification']}"
            )
            self._write_metrics(self.posture_metrics_text, text)
            out_dir = self._ensure_output_dir()
            img_path = os.path.join(out_dir, "posture_pdf_temp.jpg")
            save_image(img_path, self.posture_state.result["images"]["annotated"])
            self._generate_pdf_report("postura", m, img_path, text)
        except Exception as e:
            messagebox.showerror("Postura", f"Error: {e}")
        finally:
            self._clear_status()

    @staticmethod
    def _write_metrics(widget: tk.Text, text: str):
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)

    def _ensure_output_dir(self):
        out_dir = self.save_dir_var.get().strip() or "outputs"
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _save_foot(self):
        if not self.foot_state.result:
            messagebox.showwarning("Guardar", "Primero ejecuta el análisis de pie.")
            return

        out_dir = self._ensure_output_dir()
        images = self.foot_state.result["images"]
        save_image(os.path.join(out_dir, "foot_annotated.jpg"), images["annotated"])
        save_image(os.path.join(out_dir, "foot_gray.jpg"), images["gray"])
        save_image(os.path.join(out_dir, "foot_binary.jpg"), images["binary"])
        save_image(os.path.join(out_dir, "foot_clean.jpg"), images["clean"])
        save_image(os.path.join(out_dir, "foot_edges.jpg"), images["edges"])
        save_image(os.path.join(out_dir, "foot_rotated_widths.jpg"), images["rotated_widths"])
        messagebox.showinfo("Guardar", f"Resultados de pie guardados en: {out_dir}")

    def _save_knee(self):
        if not self.knee_state.result:
            messagebox.showwarning("Guardar", "Primero ejecuta el análisis de rodilla.")
            return

        out_dir = self._ensure_output_dir()
        save_image(os.path.join(out_dir, "knee_annotated.jpg"), self.knee_state.result["images"]["annotated"])
        messagebox.showinfo("Guardar", f"Resultado de rodilla guardado en: {out_dir}")

    def _save_posture(self):
        if not self.posture_state.result:
            messagebox.showwarning("Guardar", "Primero ejecuta el análisis postural.")
            return

        out_dir = self._ensure_output_dir()
        save_image(os.path.join(out_dir, "posture_annotated.jpg"), self.posture_state.result["images"]["annotated"])
        messagebox.showinfo("Guardar", f"Resultado de postura guardado en: {out_dir}")

    def run(self):
        self.root.mainloop()


def run_tkinter_app():
    app = BiomechanicsApp()
    app.run()
