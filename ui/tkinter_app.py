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

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            messagebox.showerror("Cámara", f"No se pudo abrir la cámara #{camera_index}.")
            self.cap = None
            self.captured_frame = None
            self.destroy()
            return

        self.current_frame: Optional[np.ndarray] = None
        self.captured_frame: Optional[np.ndarray] = None
        self._tk_image = None

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

        # Variables para cámara por módulo
        self.foot_camera_var = tk.StringVar()
        self.knee_camera_var = tk.StringVar()
        self.posture_camera_var = tk.StringVar()
        self._camera_options = []
        self._refresh_cameras()
    def _refresh_cameras(self):
        self._camera_options = list_cameras()
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
        ttk.Entry(row, textvariable=self.save_dir_var, width=42).pack(side="left", padx=(0, 6))
        ttk.Button(row, text="Elegir", command=self._choose_output_dir).pack(side="left")

        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=14, pady=8)

        self.tab_foot = ttk.Frame(notebook)
        self.tab_knee = ttk.Frame(notebook)
        self.tab_posture = ttk.Frame(notebook)

        notebook.add(self.tab_foot, text="Baropodometría (Pie)")
        notebook.add(self.tab_knee, text="Rodilla")
        notebook.add(self.tab_posture, text="Postura")

        self._build_foot_tab()
        self._build_knee_tab()
        self._build_posture_tab()

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
        ttk.Button(controls, text="Analizar", style="Primary.TButton", command=self._analyze_foot).pack(side="left", padx=4)
        ttk.Button(controls, text="Guardar resultados", command=self._save_foot).pack(side="left", padx=4)

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
        cam_combo['values'] = [name for idx, name in self._camera_options]
        cam_combo.pack(side="left", padx=(0, 2))
        ttk.Button(controls, text="Actualizar cámaras", command=lambda: self._update_camera_combo(cam_combo, self.knee_camera_var)).pack(side="left", padx=(0, 8))


        ttk.Button(controls, text="Cargar imagen", command=lambda: self._load_image(self.knee_state, self.knee_original_lbl, self.knee_result_lbl)).pack(side="left", padx=4)
        ttk.Button(controls, text="Tomar foto", command=lambda: self._capture_image(self.knee_state, self.knee_original_lbl, self.knee_result_lbl, self._get_camera_index(self.knee_camera_var.get()))).pack(side="left", padx=4)

        ttk.Label(controls, text="Plano:", style="Body.TLabel").pack(side="left", padx=(12, 4))
        ttk.Combobox(controls, textvariable=self.knee_plane_var, values=["frontal", "sagital"], state="readonly", width=12).pack(side="left", padx=4)

        ttk.Button(controls, text="Analizar", style="Primary.TButton", command=self._analyze_knee).pack(side="left", padx=4)
        ttk.Button(controls, text="Guardar resultado", command=self._save_knee).pack(side="left", padx=4)

        self.knee_original_lbl, self.knee_result_lbl = self._build_common_image_area(self.tab_knee)
        def _capture_posture_photo(self):
            if not self._camera_options:
                messagebox.showerror("Cámara", "No hay cámaras disponibles para capturar.")
                return
            idx = self._get_camera_index(self.posture_camera_var.get())
            if idx is None:
                messagebox.showerror("Cámara", "Selecciona una cámara válida.")
                return
            self._capture_image(self.posture_state, self.posture_original_lbl, self.posture_result_lbl, idx)

        metrics_frame = ttk.LabelFrame(self.tab_knee, text="Métricas", style="Card.TLabelframe")
        metrics_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.knee_metrics_text = tk.Text(metrics_frame, height=5)
        self.knee_metrics_text.pack(fill="x", padx=8, pady=8)
        self._configure_text_widget(self.knee_metrics_text)



    def _build_posture_tab(self):
        controls = ttk.Frame(self.tab_posture, style="Card.TFrame")
        controls.pack(fill="x", padx=10, pady=8)

        ttk.Label(controls, text="Cámara:", style="Body.TLabel").pack(side="left", padx=(0, 2))
        cam_combo = ttk.Combobox(controls, textvariable=self.posture_camera_var, state="readonly", width=32)
        cam_combo['values'] = [name for idx, name in self._camera_options]
        cam_combo.pack(side="left", padx=(0, 2))
        ttk.Button(controls, text="Actualizar cámaras", command=lambda: self._update_camera_combo(cam_combo, self.posture_camera_var)).pack(side="left", padx=(0, 8))

        # Inicializa los labels ANTES de crear los botones
        self.posture_original_lbl, self.posture_result_lbl = self._build_common_image_area(self.tab_posture)


        ttk.Button(controls, text="Cargar imagen", command=lambda: self._load_image(self.posture_state, self.posture_original_lbl, self.posture_result_lbl)).pack(side="left", padx=4)

        # Botón de analizar manual
        ttk.Button(controls, text="Analizar", style="Primary.TButton", command=self._analyze_posture).pack(side="left", padx=4)

        # Botón de captura robusto: deshabilitado si no hay cámaras
        capture_btn = ttk.Button(
            controls,
            text="Tomar foto",
            command=lambda: self._capture_posture_photo()
        )
        capture_btn.pack(side="left", padx=4)
        if not self._camera_options:
            capture_btn.state(["disabled"])

    def _capture_posture_photo(self):
        if not self._camera_options:
            messagebox.showerror("Cámara", "No hay cámaras disponibles para capturar.")
            return
        idx = self._get_camera_index(self.posture_camera_var.get())
        if idx is None:
            messagebox.showerror("Cámara", "Selecciona una cámara válida.")
            return
        self._capture_image(self.posture_state, self.posture_original_lbl, self.posture_result_lbl, idx)
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
                self._clear_status()


    def _update_camera_combo(self, combo, var):
        self._refresh_cameras()
        combo['values'] = [name for idx, name in self._camera_options]
        if self._camera_options:
            var.set(self._camera_options[0][1])
            combo.state(["!disabled"])
        else:
            var.set("")
            combo.state(["disabled"])


    def _get_camera_index(self, name):
        for idx, label in self._camera_options:
            if label == name:
                return idx
        return None
        ttk.Button(controls, text="Analizar", style="Primary.TButton", command=self._analyze_posture).pack(side="left", padx=4)
        ttk.Button(controls, text="Guardar resultado", command=self._save_posture).pack(side="left", padx=4)

        self.posture_original_lbl, self.posture_result_lbl = self._build_common_image_area(self.tab_posture)

        metrics_frame = ttk.LabelFrame(self.tab_posture, text="Métricas", style="Card.TLabelframe")
        metrics_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.posture_metrics_text = tk.Text(metrics_frame, height=5)
        self.posture_metrics_text.pack(fill="x", padx=8, pady=8)
        self._configure_text_widget(self.posture_metrics_text)

    def _choose_output_dir(self):
        path = filedialog.askdirectory(title="Selecciona carpeta de salida")
        if path:
            self.save_dir_var.set(path)

    def _load_image(self, state: ModuleState, label: ttk.Label, result_label: ttk.Label):
        path = filedialog.askopenfilename(
            title="Selecciona imagen",
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
        )
        if not path:
            return

        image = cv2.imread(path)
        if image is None:
            messagebox.showerror("Imagen", "No se pudo cargar la imagen.")
            return

        state.source_image = image
        state.source_path = path
        state.result = None
        self._set_image_on_label(label, image)
        self._clear_image_label(result_label, "Resultado no disponible")

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
