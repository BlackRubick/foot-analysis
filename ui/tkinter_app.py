from __future__ import annotations

import os
import json
from datetime import datetime, date
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw

from foot_analysis.analyzer import FootAnalyzer
from knee_analysis.analyzer import KneeAnalyzer
from posture_analysis.analyzer import PostureAnalyzer
from chains_analysis import Calibration, MuscleChainAnalyzer
from utils.db import DatabaseClient
from utils.image_io import save_image
from utils.camera_utils import list_cameras
from utils.pdf_report import generate_consent_pdf


@dataclass
class ModuleState:
    source_image: Optional[np.ndarray] = None
    source_path: Optional[str] = None
    result: Optional[Dict] = None


@dataclass
class ChainsState:
    source_image: Optional[np.ndarray] = None
    source_path: Optional[str] = None
    result: Optional[Dict] = None
    calibration_points: list = None

    def __post_init__(self):
        if self.calibration_points is None:
            self.calibration_points = []


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
    @staticmethod
    def _calibration_mode_key(mode_label: str) -> str:
        mode_label = (mode_label or "").strip().lower()
        mode_map = {
            "automatico": "auto",
            "automático": "auto",
            "auto": "auto",
            "sin calibracion": "none",
            "sin calibración": "none",
            "ninguna": "none",
            "none": "none",
            "referencia": "reference",
            "reference": "reference",
            "altura": "height",
            "height": "height",
            "aruco": "aruco",
        }
        return mode_map.get(mode_label, "auto")

    @staticmethod
    def _side_key(side_label: str) -> str:
        side_label = (side_label or "").strip().lower()
        side_map = {
            "automatico": "auto",
            "automático": "auto",
            "auto": "auto",
            "izquierdo": "left",
            "left": "left",
            "derecho": "right",
            "right": "right",
        }
        return side_map.get(side_label, "auto")

    @staticmethod
    def _side_label(side_key: str) -> str:
        side_key = (side_key or "auto").strip().lower()
        labels = {
            "auto": "Automático",
            "left": "Izquierdo",
            "right": "Derecho",
        }
        return labels.get(side_key, side_key)

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

    def _load_chains_image(self):
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
        self.chains_state.source_image = img
        self.chains_state.source_path = path
        self.chains_state.result = None
        self.chains_state.calibration_points = []
        self._update_chains_calibration_indicator("pixel")
        self._update_chains_preview(img)
        self._write_metrics(self.chains_metrics_text, "Imagen cargada. Usa 'Analizar captura' o inicia el video en vivo.")

    def _get_chains_camera_index(self, camera_name: str) -> int:
        return self._get_camera_index(camera_name)


    def _get_current_chains_calibration(self, image_bgr: Optional[np.ndarray] = None) -> Calibration:
        mode = self._calibration_mode_key(self.chains_calibration_mode_var.get())
        reference_mm = float(self.chains_reference_mm_var.get() or 0)

        reference_px = float(self.chains_reference_px_var.get() or 0)
        patient_height_mm = float(self.chains_patient_height_mm_var.get() or 0)
        marker_mm = float(self.chains_aruco_marker_mm_var.get() or 0)

        def try_aruco() -> Optional[Calibration]:
            if image_bgr is None or marker_mm <= 0:
                return None
            try:
                return MuscleChainAnalyzer.estimate_aruco_calibration(image_bgr, marker_mm)
            except Exception:
                return None

        def try_reference() -> Optional[Calibration]:
            nonlocal reference_px
            if reference_mm <= 0:
                return None
            if reference_px > 0:
                return Calibration.from_reference(reference_mm, reference_px)
            if len(self.chains_state.calibration_points) == 2:
                import math
                p1, p2 = self.chains_state.calibration_points
                reference_px = math.dist(p1, p2)
                if reference_px > 0:
                    self.chains_reference_px_var.set(f"{reference_px:.2f}")
                    return Calibration.from_reference(reference_mm, reference_px)
            return None


        def try_height() -> Optional[Calibration]:
            if image_bgr is None or patient_height_mm <= 0:
                return None
            try:
                temp = MuscleChainAnalyzer()
                detection = temp.detector.detect(image_bgr)
                y_values = []
                for key in (
                    "nose", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_hip",
                    "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"
                ):
                    point = detection.pose.get(key)
                    if point is not None:
                        y_values.append(point.y)
                if len(y_values) >= 2:
                    pose_height_px = max(y_values) - min(y_values)
                    if pose_height_px > 0:
                        return Calibration.from_height(patient_height_mm, pose_height_px)
            except Exception:
                return None
            return None

        if mode == "aruco":
            return try_aruco() or Calibration()
        if mode == "reference":
            return try_reference() or Calibration()
        if mode == "height":
            return try_height() or Calibration()
        if mode == "auto":
            return try_aruco() or try_reference() or try_height() or Calibration()
        return Calibration()

    def _get_chains_analyzer(self, calibration: Optional[Calibration] = None) -> MuscleChainAnalyzer:
        return MuscleChainAnalyzer(calibration=calibration or Calibration())

    def _format_chains_metrics(self, result) -> str:
        source = result.metrics.get('calibration_source', '-')
        source_label = self._calibration_source_label(source)
        side_label = self._side_label(result.metrics.get('profile_side', '-'))
        confidence_text, _, confidence_level = self._calibration_quality(source)
        lines = [
            f"Plano: {result.metrics.get('plane', '-')}",
            f"Lado detectado: {side_label}",
            f"Calibración: {result.metrics.get('calibration_mm_per_px', 1.0):.4f} mm/px",
            f"Fuente calibración: {source_label}",
            f"Confianza calibración: [{confidence_level}] {confidence_text}",
        ]
        for key, summary in result.chain_summaries.items():
            lines.append(
                f"{summary.name}: prevalencia {summary.percentage:.1f}% | "
                f"activación {summary.activation_percentage:.1f}% ({summary.positives}/{summary.total})"
            )
        if result.notes:
            lines.extend(result.notes)
        positives = [item.label for item in result.feature_results if item.present]
        if positives:
            lines.append("Rasgos presentes:")
            lines.extend(f"- {label}" for label in positives)
        return "\n".join(lines)

    @staticmethod
    def _calibration_quality(source: str) -> tuple:
        source = (source or "pixel").lower()
        if source == "aruco":
            return "Alta (ArUco)", "#22c55e", "ALTA"
        if source == "reference":
            return "Media (Referencia)", "#f59e0b", "MEDIA"
        if source == "height":
            return "Media-baja (Altura)", "#f59e0b", "MEDIA"
        return "Baja (Sin escala)", "#ef4444", "BAJA"

    @staticmethod
    def _calibration_source_label(source: str) -> str:
        source = (source or "pixel").lower()
        labels = {
            "aruco": "ArUco",
            "reference": "Referencia",
            "height": "Altura",
            "pixel": "Sin escala",
        }
        return labels.get(source, source)

    def _update_chains_calibration_indicator(self, source: str):
        if not hasattr(self, "chains_calibration_status_var"):
            return
        text, color, level = self._calibration_quality(source)
        self.chains_calibration_status_var.set(f"Confianza calibración: [{level}] {text}")
        if hasattr(self, "chains_calibration_status_label"):
            self.chains_calibration_status_label.configure(fg=color)

    def _analyze_chains(self):
        if self.chains_state.source_image is None:
            messagebox.showwarning("Cadenas musculares", "Primero carga o captura una imagen.")
            return
        if not self._ensure_consent_or_warn():
            return
        try:
            self._set_status("Analizando cadenas musculares...", busy=True)
            calibration = self._get_current_chains_calibration(self.chains_state.source_image)
            analyzer = self._get_chains_analyzer(calibration)
            self.chains_state.result = analyzer.analyze(
                self.chains_state.source_image,
                plane=self.chains_plane_var.get(),
                profile_side=self._side_key(self.chains_profile_side_var.get()),
            )
            confidence_text, _, confidence_level = self._calibration_quality(
                self.chains_state.result.metrics.get("calibration_source", "pixel")
            )
            self.chains_state.result.metrics["calibration_confidence"] = confidence_level
            self.chains_state.result.metrics["calibration_confidence_text"] = confidence_text
            self._update_chains_calibration_indicator(self.chains_state.result.metrics.get("calibration_source", "pixel"))
            self._update_chains_preview(self.chains_state.result.images["annotated"])
            self._write_metrics(self.chains_metrics_text, self._format_chains_metrics(self.chains_state.result))
            out_dir = self._ensure_output_dir()
            img_path = os.path.join(out_dir, "chains_pdf_temp.jpg")
            save_image(img_path, self.chains_state.result.images["annotated"])
            chains_text = self._format_chains_metrics(self.chains_state.result)
            self._generate_pdf_report("cadenas", self.chains_state.result.metrics, img_path, chains_text)
            self._persist_ui_analysis("chains", self.chains_state.result.metrics, chains_text)
            self._refresh_history_view()
        except Exception as e:
            messagebox.showerror("Cadenas musculares", f"Error: {e}")
            self._update_chains_calibration_indicator("pixel")
        finally:
            self._clear_status()

    def _save_chains(self):
        if not self.chains_state.result:
            messagebox.showwarning("Guardar", "Primero ejecuta el análisis de cadenas musculares.")
            return
        out_dir = self._ensure_output_dir()
        save_image(os.path.join(out_dir, "chains_annotated.jpg"), self.chains_state.result.images["annotated"])
        messagebox.showinfo("Guardar", f"Resultado de cadenas musculares guardado en: {out_dir}")

    def _clear_chains_calibration(self):
        self.chains_state.calibration_points = []
        self._update_chains_calibration_indicator("pixel")
        self._write_metrics(self.chains_metrics_text, "Calibración reiniciada. Haz clic en dos puntos sobre la imagen para medir una referencia física.")

    def _on_chains_canvas_click(self, event):
        if self.chains_state.source_image is None and self.chains_live_frame is None:
            return
        image = self.chains_live_frame if self.chains_live_frame is not None else self.chains_state.source_image
        if image is None:
            return
        if not self.chains_calibration_pick_var.get():
            return
        h, w = image.shape[:2]
        canvas_w = max(self.chains_preview_canvas.winfo_width(), 1)
        canvas_h = max(self.chains_preview_canvas.winfo_height(), 1)
        scale = min(canvas_w / max(w, 1), canvas_h / max(h, 1), 1.0)
        disp_w = max(int(w * scale), 1)
        disp_h = max(int(h * scale), 1)
        offset_x = (canvas_w - disp_w) / 2
        offset_y = (canvas_h - disp_h) / 2
        x = (event.x - offset_x) / scale
        y = (event.y - offset_y) / scale
        if x < 0 or y < 0 or x > w or y > h:
            return
        self.chains_state.calibration_points.append((float(x), float(y)))
        if len(self.chains_state.calibration_points) > 2:
            self.chains_state.calibration_points = self.chains_state.calibration_points[-2:]
        if len(self.chains_state.calibration_points) == 2:
            import math
            mm = float(self.chains_reference_mm_var.get() or 0)
            if mm > 0:
                px = math.dist(self.chains_state.calibration_points[0], self.chains_state.calibration_points[1])
                if px > 0:
                    self.chains_reference_px_var.set(f"{px:.2f}")
                    self.chains_calibration_mode_var.set("Automático")
                    self._write_metrics(self.chains_metrics_text, f"Calibración lista: {mm:.2f} mm = {px:.2f} px\nmm/px = {mm / px:.4f}")
        self._update_chains_preview(image)

    def _update_chains_preview(self, image_bgr: np.ndarray):
        if not hasattr(self, "chains_preview_canvas") or self.chains_preview_canvas is None:
            return
        import cv2
        from PIL import Image, ImageTk

        image = image_bgr.copy()
        if self.chains_state.calibration_points:
            for idx, point in enumerate(self.chains_state.calibration_points):
                cv2.circle(image, (int(point[0]), int(point[1])), 7, (0, 255, 255), -1)
                cv2.putText(image, str(idx + 1), (int(point[0]) + 8, int(point[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        canvas_w = max(self.chains_preview_canvas.winfo_width(), 700)
        canvas_h = max(self.chains_preview_canvas.winfo_height(), 520)
        h, w = image.shape[:2]
        scale = min(canvas_w / max(w, 1), canvas_h / max(h, 1), 1.0)
        new_w, new_h = max(int(w * scale), 1), max(int(h * scale), 1)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        self.chains_tk_img = ImageTk.PhotoImage(pil_img)
        self.chains_preview_canvas.delete("all")
        self.chains_preview_canvas.create_image(canvas_w / 2, canvas_h / 2, anchor="center", image=self.chains_tk_img)

    def _chains_live_tick(self):
        if not self.chains_live_running or self.chains_live_cap is None:
            return
        ok, frame = self.chains_live_cap.read()
        if not ok or frame is None:
            self._set_status("No se pudo leer la cámara de cadenas.", busy=False)
            self._stop_chains_live()
            return
        self.chains_live_frame = frame.copy()
        try:
            calibration = self._get_current_chains_calibration(frame)
            analyzer = self._get_chains_analyzer(calibration)
            result = analyzer.analyze(
                frame,
                plane=self.chains_plane_var.get(),
                profile_side=self._side_key(self.chains_profile_side_var.get()),
            )
            confidence_text, _, confidence_level = self._calibration_quality(result.metrics.get("calibration_source", "pixel"))
            result.metrics["calibration_confidence"] = confidence_level
            result.metrics["calibration_confidence_text"] = confidence_text
            self.chains_state.result = result
            self._update_chains_calibration_indicator(result.metrics.get("calibration_source", "pixel"))
            self._update_chains_preview(result.images["annotated"])
            self._write_metrics(self.chains_metrics_text, self._format_chains_metrics(result))
        except Exception:
            self._update_chains_calibration_indicator("pixel")
            self._update_chains_preview(frame)
        self.chains_live_after_id = self.root.after(60, self._chains_live_tick)

    def _start_chains_live(self):
        if self.chains_live_running:
            return
        camera_index = self._get_chains_camera_index(self.chains_camera_var.get())
        if camera_index is None or camera_index < 0:
            messagebox.showerror("Cadenas musculares", "Selecciona una cámara válida.")
            return
        self.chains_live_cap = cv2.VideoCapture(camera_index)
        if not self.chains_live_cap.isOpened():
            self.chains_live_cap = None
            messagebox.showerror("Cadenas musculares", "No se pudo abrir la cámara.")
            return
        self.chains_live_running = True
        self._set_status("Video en vivo de cadenas musculares activo...", busy=True)
        self._chains_live_tick()

    def _stop_chains_live(self):
        self.chains_live_running = False
        if self.chains_live_after_id is not None:
            try:
                self.root.after_cancel(self.chains_live_after_id)
            except Exception:
                pass
            self.chains_live_after_id = None
        if self.chains_live_cap is not None:
            self.chains_live_cap.release()
            self.chains_live_cap = None
        self._clear_status()

    def _toggle_chains_live(self):
        if self.chains_live_running:
            self._stop_chains_live()
        else:
            self._start_chains_live()
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

    def _update_camera_combo(self, combo: ttk.Combobox, variable: tk.StringVar):
        cameras = list_cameras()
        values = [name for idx, name in cameras]
        combo['values'] = values
        if values:
            variable.set(values[0])

        self._camera_options = cameras
        self._camera_options_chains = cameras
        if hasattr(self, "_camera_options_knee"):
            self._camera_options_knee = cameras
        if hasattr(self, "_camera_options_posture"):
            self._camera_options_posture = cameras
        if hasattr(self, "chains_camera_var"):
            self.chains_camera_var.set(values[0] if values else "")
        if hasattr(self, "foot_camera_var") and not self.foot_camera_var.get() and values:
            self.foot_camera_var.set(values[0])
        if hasattr(self, "knee_camera_var") and not self.knee_camera_var.get() and values:
            self.knee_camera_var.set(values[0])
        if hasattr(self, "posture_camera_var") and not self.posture_camera_var.get() and values:
            self.posture_camera_var.set(values[0])

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
            posture_data = metrics if analysis_type in {"postura", "rodilla", "cadenas", "palanca"} else {}
            plantar_data = metrics if analysis_type == "pie" else {}
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
        self.chains_state = ChainsState()

        self.save_dir_var = tk.StringVar(value="outputs")
        self.knee_plane_var = tk.StringVar(value="frontal")
        self.foot_stage_var = tk.StringVar(value="annotated")
        self.status_var = tk.StringVar(value="Listo")
        self.chains_live_running = False
        self.chains_live_cap = None
        self.chains_live_after_id = None
        self.chains_live_frame = None
        self.chains_tk_img = None

        self.db_enabled_var = tk.BooleanVar(value=True)
        self.db_host_var = tk.StringVar(value="127.0.0.1")
        self.db_port_var = tk.StringVar(value="3306")
        self.db_user_var = tk.StringVar(value="cesar")
        self.db_password_var = tk.StringVar(value="cesar123")
        self.db_name_var = tk.StringVar(value="foot_analysis_db")
        self.db_patient_uuid_var = tk.StringVar(value="")
        self.db_patient_fhir_var = tk.StringVar(value="")
        self.db_client = None
        self.db_session_id = None

        # Datos generales del paciente
        self.patient_data = {
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
            "Presencia de cicatrices": "No"
        }

        # Estado para el modo de evaluación completa (test único)
        self.fulltest_step_index = 0
        self.fulltest_completed_steps = set()

        # Variables para cámara por módulo
        self.foot_camera_var = tk.StringVar()
        self.knee_camera_var = tk.StringVar()
        self.posture_camera_var = tk.StringVar()
        self._camera_options = []
        self._refresh_cameras()

    def _ensure_db_connection(self) -> bool:
        if not self.db_enabled_var.get():
            return False
        if self.db_client is not None and self.db_session_id is not None:
            return True

        try:
            self.db_client = DatabaseClient(
                host=self.db_host_var.get().strip() or "127.0.0.1",
                port=int(self.db_port_var.get().strip() or "3306"),
                user=self.db_user_var.get().strip() or "cesar",
                password=self.db_password_var.get(),
                database=self.db_name_var.get().strip() or "foot_analysis_db",
            )

            patient_uuid, patient_fhir = self.db_client.ensure_patient(
                patient_uuid=(self.db_patient_uuid_var.get().strip() or None),
                patient_fhir_id=(self.db_patient_fhir_var.get().strip() or None),
            )
            self.db_patient_uuid_var.set(patient_uuid)
            self.db_patient_fhir_var.set(patient_fhir)

            identity_name = self.patient_data.get("Nombre", "").strip() or "PACIENTE SIN NOMBRE"
            contact_data = json.dumps(
                {
                    "Edad": self.patient_data.get("Edad", ""),
                    "Sexo": self.patient_data.get("Sexo", ""),
                    "Ocupación": self.patient_data.get("Ocupación", ""),
                    "Actividad física": self.patient_data.get("Actividad física", ""),
                },
                ensure_ascii=False,
            )
            self.db_client.upsert_patient_identity(patient_uuid, identity_name, contact_data)

            session = self.db_client.create_capture_session(patient_uuid)
            self.db_session_id = int(session["session_id"])
            self._set_status(
                f"DB conectada | Paciente: {patient_fhir} | Sesion: {self.db_session_id}",
                busy=False,
            )
            return True
        except Exception as e:
            self.db_client = None
            self.db_session_id = None
            messagebox.showerror("Base de datos", f"No se pudo conectar/crear contexto DB: {e}")
            return False

    def _ensure_consent_or_warn(self) -> bool:
        if not self.db_enabled_var.get():
            return True
        if not self._ensure_db_connection():
            return False
        if self.db_client.has_consent(self.db_patient_uuid_var.get().strip()):
            return True
        messagebox.showwarning(
            "Consentimiento informado",
            "Debes registrar consentimiento informado antes de procesar imágenes.",
        )
        return False

    def _persist_ui_analysis(self, analysis_type: str, metrics: Dict, notes_text: str = "") -> None:
        if not self.db_enabled_var.get():
            return
        if not self._ensure_db_connection():
            return
        try:
            self.db_client.save_analysis(
                patient_uuid=self.db_patient_uuid_var.get().strip(),
                session_id=self.db_session_id,
                analysis_type=analysis_type,
                metrics=metrics,
                notes_text=notes_text,
            )
        except Exception as e:
            messagebox.showerror("Base de datos", f"No se pudo guardar analisis en DB: {e}")

    def _refresh_history_view(self):
        if not hasattr(self, "history_text"):
            return

        self.history_text.delete("1.0", tk.END)

        if not self.db_enabled_var.get():
            self.history_text.insert(tk.END, "Base de datos desactivada.\n")
            return

        if not self._ensure_db_connection():
            self.history_text.insert(tk.END, "Sin conexión a base de datos.\n")
            return

        patient_uuid = self.db_patient_uuid_var.get().strip()
        if not patient_uuid:
            self.history_text.insert(tk.END, "No hay paciente activo.\n")
            return

        try:
            history = self.db_client.fetch_patient_history(patient_uuid, limit=10)
        except Exception as e:
            self.history_text.insert(tk.END, f"Error al leer historial: {e}\n")
            return

        self.history_text.insert(tk.END, f"Paciente: {self.db_patient_fhir_var.get().strip()}\n")
        self.history_text.insert(tk.END, f"UUID: {patient_uuid}\n\n")

        self.history_text.insert(tk.END, "SESIONES\n")
        for item in history["sessions"]:
            self.history_text.insert(
                tk.END,
                f"- Sesion {item['session_id']} | {item['study_date']} {item['study_time']} | {item['study_instance_uid']}\n",
            )

        self.history_text.insert(tk.END, "\nANALISIS\n")
        for item in history["analyses"]:
            self.history_text.insert(
                tk.END,
                f"- {item['analysis_type']} | {item['study_date']} {item['study_time']} | ID {item['analysis_id']}\n",
            )

        self.history_text.insert(tk.END, "\nNOTAS / CDA\n")
        for item in history["notes"]:
            self.history_text.insert(
                tk.END,
                f"- {item['report_kind']} | {item['note_date']} {item['note_time']} | {item['cda_document_id']} | {item['clinician_full_name']}\n",
            )

    def _show_db_form(self):
        form = tk.Toplevel(self.root)
        form.title("Configuración de base de datos")
        form.geometry("560x420")
        form.transient(self.root)
        form.grab_set()

        fields = [
            ("Habilitar DB", self.db_enabled_var, True),
            ("DB Host", self.db_host_var, False),
            ("DB Port", self.db_port_var, False),
            ("DB User", self.db_user_var, False),
            ("DB Password", self.db_password_var, False),
            ("DB Name", self.db_name_var, False),
            ("Patient UUID", self.db_patient_uuid_var, False),
            ("Patient FHIR ID", self.db_patient_fhir_var, False),
        ]

        row = 0
        for label, variable, is_bool in fields:
            tk.Label(form, text=label + ":").grid(row=row, column=0, sticky="w", padx=8, pady=6)
            if is_bool:
                tk.Checkbutton(form, variable=variable).grid(row=row, column=1, sticky="w", padx=8, pady=6)
            elif label == "DB Password":
                tk.Entry(form, textvariable=variable, width=42, show="*").grid(row=row, column=1, padx=8, pady=6)
            else:
                tk.Entry(form, textvariable=variable, width=42).grid(row=row, column=1, padx=8, pady=6)
            row += 1

        def save_and_connect():
            if self.db_client is not None:
                try:
                    self.db_client.close()
                except Exception:
                    pass
            self.db_client = None
            self.db_session_id = None
            if self.db_enabled_var.get():
                if not self._ensure_db_connection():
                    return
            self._refresh_history_view()
            form.destroy()

        tk.Button(form, text="Guardar y conectar", command=save_and_connect).grid(row=row, column=0, columnspan=2, pady=14)

    def _show_consent_form(self):
        if not self._ensure_db_connection():
            return

        form = tk.Toplevel(self.root)
        form.title("Consentimiento informado")
        # Ventana algo más compacta y redimensionable, con scroll interno
        form.geometry("720x520")
        form.minsize(640, 480)
        form.resizable(True, True)
        form.transient(self.root)
        form.grab_set()

        # Contenedor con canvas + scrollbar para que todo el formulario sea desplazable
        container = ttk.Frame(form)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container, highlightthickness=0, bg=self.bg_main)
        vscroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")

        inner = ttk.Frame(canvas, style="Card.TFrame")
        canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        inner.bind("<Configure>", _on_configure)

        signed_var = tk.StringVar(value=self.patient_data.get("Nombre", ""))
        birth_var = tk.StringVar(value=self.patient_data.get("Fecha de nacimiento", ""))
        height_var = tk.StringVar(value=self.patient_data.get("Estatura", ""))
        weight_var = tk.StringVar(value=self.patient_data.get("Peso", ""))
        patient_name_var = tk.StringVar(value=self.patient_data.get("Nombre", ""))
        witness1_var = tk.StringVar(value="")
        witness2_var = tk.StringVar(value="")
        file_var = tk.StringVar(value="")

        tk.Label(inner, text="Nombre de quien firma:").pack(anchor="w", padx=10, pady=(10, 2))
        tk.Entry(inner, textvariable=signed_var, width=65).pack(anchor="w", padx=10)
        tk.Label(inner, text="Nombre del paciente:").pack(anchor="w", padx=10, pady=(10, 2))
        tk.Entry(inner, textvariable=patient_name_var, width=65).pack(anchor="w", padx=10)
        tk.Label(inner, text="Fecha de nacimiento:").pack(anchor="w", padx=10, pady=(10, 2))
        tk.Entry(inner, textvariable=birth_var, width=25).pack(anchor="w", padx=10)
        tk.Label(inner, text="Estatura:").pack(anchor="w", padx=10, pady=(10, 2))
        tk.Entry(inner, textvariable=height_var, width=25).pack(anchor="w", padx=10)
        tk.Label(inner, text="Peso:").pack(anchor="w", padx=10, pady=(10, 2))
        tk.Entry(inner, textvariable=weight_var, width=25).pack(anchor="w", padx=10)

        # Firma dibujada en pantalla
        tk.Label(inner, text="Firma dibujada (paciente o representante):").pack(anchor="w", padx=10, pady=(10, 2))
        sign_frame = tk.Frame(inner)
        sign_frame.pack(anchor="w", padx=10, pady=(0, 10))

        sign_width, sign_height = 420, 140
        sign_canvas = tk.Canvas(sign_frame, width=sign_width, height=sign_height, bg="white", highlightthickness=1, highlightbackground="#4b5563")
        sign_canvas.pack(side="left")

        signature_image = Image.new("RGB", (sign_width, sign_height), "white")
        signature_draw = ImageDraw.Draw(signature_image)
        signature_drawn = False
        last_point = {"x": None, "y": None}

        def _start_signature(event):
            last_point["x"], last_point["y"] = event.x, event.y

        def _draw_signature(event):
            nonlocal signature_drawn
            x0, y0 = last_point["x"], last_point["y"]
            x1, y1 = event.x, event.y
            if x0 is not None and y0 is not None:
                sign_canvas.create_line(x0, y0, x1, y1, fill="black", width=2, capstyle="round")
                signature_draw.line((x0, y0, x1, y1), fill="black", width=2)
                signature_drawn = True
            last_point["x"], last_point["y"] = x1, y1

        def _clear_signature():
            nonlocal signature_image, signature_draw, signature_drawn
            sign_canvas.delete("all")
            signature_image = Image.new("RGB", (sign_width, sign_height), "white")
            signature_draw = ImageDraw.Draw(signature_image)
            signature_drawn = False

        sign_canvas.bind("<Button-1>", _start_signature)
        sign_canvas.bind("<B1-Motion>", _draw_signature)

        btn_sig = tk.Frame(sign_frame)
        btn_sig.pack(side="left", padx=(8, 0))
        tk.Button(btn_sig, text="Borrar firma", command=_clear_signature).pack(anchor="n", pady=(0, 4))

        tk.Label(inner, text="Testigo 1:").pack(anchor="w", padx=10, pady=(10, 2))
        tk.Entry(inner, textvariable=witness1_var, width=65).pack(anchor="w", padx=10)
        tk.Label(inner, text="Testigo 2:").pack(anchor="w", padx=10, pady=(10, 2))
        tk.Entry(inner, textvariable=witness2_var, width=65).pack(anchor="w", padx=10)

        tk.Label(inner, text="Documento firmado (opcional):").pack(anchor="w", padx=10, pady=(10, 2))
        file_row = tk.Frame(inner)
        file_row.pack(fill="x", padx=10)
        tk.Entry(file_row, textvariable=file_var, width=62).pack(side="left")

        def pick_file():
            path = filedialog.askopenfilename(title="Selecciona consentimiento firmado")
            if path:
                file_var.set(path)

        tk.Button(file_row, text="Seleccionar", command=pick_file).pack(side="left", padx=(6, 0))

        tk.Label(inner, text="Texto de consentimiento:").pack(anchor="w", padx=10, pady=(10, 2))
        consent_text = tk.Text(inner, height=12)
        consent_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        def save_consent():
            try:
                text_value = consent_text.get("1.0", tk.END).strip()
                if not text_value:
                    raise ValueError("El texto de consentimiento es obligatorio")
                if not signed_var.get().strip():
                    raise ValueError("El nombre de quien firma es obligatorio")

                out_dir = self._ensure_output_dir()
                consent_pdf_path = os.path.join(
                    out_dir,
                    f"consentimiento_{(patient_name_var.get().strip() or 'paciente').replace(' ', '_')}.pdf",
                )

                signature_image_path = None
                signature_image_bytes = None
                if signature_drawn:
                    try:
                        sig_slug = (patient_name_var.get().strip() or self.patient_data.get("Nombre", "paciente") or "paciente").replace(" ", "_")
                        signature_image_path = os.path.join(out_dir, f"firma_{sig_slug}.png")
                        signature_image.save(signature_image_path)
                        with open(signature_image_path, "rb") as f_sig:
                            signature_image_bytes = f_sig.read()
                    except Exception:
                        signature_image_path = None
                        signature_image_bytes = None

                consent_payload = {
                    "project_name": "NEXO-POSTURAL: Kyene’is Pøndyam.",
                    "issue_place": "Tuxtla Gutiérrez, Chiapas.",
                    "issue_date": datetime.now().strftime("%d/%m/%Y"),
                    "patient_name": patient_name_var.get().strip() or self.patient_data.get("Nombre", ""),
                    "patient_sex": self.patient_data.get("Sexo", ""),
                    "responsible_1": "Liliana Ruiz Alvarado",
                    "responsible_2": "Ángel Enrique Patricio López",
                    "witness_1": witness1_var.get().strip(),
                    "witness_2": witness2_var.get().strip(),
                    "patient_birthdate": birth_var.get().strip(),
                    "patient_height": height_var.get().strip(),
                    "patient_weight": weight_var.get().strip(),
                }
                if signature_image_path:
                    generate_consent_pdf(consent_payload, consent_pdf_path, signature_image_path=signature_image_path)
                else:
                    generate_consent_pdf(consent_payload, consent_pdf_path)

                document_bytes = text_value.encode("utf-8")
                if os.path.exists(consent_pdf_path):
                    with open(consent_pdf_path, "rb") as f:
                        document_bytes = f.read()
                selected_file = file_var.get().strip()
                if selected_file:
                    with open(selected_file, "rb") as f:
                        document_bytes = f.read()

                self.db_client.save_informed_consent(
                    patient_uuid=self.db_patient_uuid_var.get().strip(),
                    consent_text=text_value,
                    signed_by=signed_var.get().strip(),
                    signature_digital_hash="",
                    consent_document_bytes=document_bytes,
                    signature_image_bytes=signature_image_bytes,
                )
                messagebox.showinfo("Consentimiento", f"Consentimiento informado registrado en DB.\nPDF guardado en:\n{consent_pdf_path}")

                # Abrir PDF con el visor por defecto del sistema (visor rápido)
                try:
                    self._open_external_file(consent_pdf_path)
                except Exception:
                    pass
                self._refresh_history_view()
                form.destroy()
            except Exception as e:
                # Imprimir traza completa en la consola para poder depurar errores de codificación/DB
                try:
                    import traceback
                    traceback.print_exc()
                except Exception:
                    pass
                messagebox.showerror("Consentimiento", f"No se pudo guardar consentimiento: {e}")

        tk.Button(inner, text="Guardar consentimiento", command=save_consent).pack(pady=8)

    def _default_cda_body(self) -> str:
        if self.chains_state.result:
            return self._format_chains_metrics(self.chains_state.result)
        if self.posture_state.result:
            m = self.posture_state.result["metrics"]
            return (
                f"Postura\n"
                f"Lado: {m['side']}\n"
                f"Desviacion media: {m['mean_deviation_px']:.2f} px\n"
                f"Clasificacion: {m['classification']}"
            )
        if self.knee_state.result:
            m = self.knee_state.result["metrics"]
            return (
                f"Rodilla\n"
                f"Plano: {m['plane']}\n"
                f"Lado: {m['side']}\n"
                f"Angulo: {m['knee_angle_deg']:.2f} deg\n"
                f"Clasificacion: {m['classification']}"
            )
        if self.foot_state.result:
            m = self.foot_state.result["metrics"]
            return (
                f"Pie\n"
                f"Indice plantar: {m['plantar_index']:.2f}\n"
                f"X: {m['x_width_px']:.2f} px\n"
                f"Y: {m['y_width_px']:.2f} px\n"
                f"Clasificacion: {m['classification']}"
            )
        return "Nota clinica sin resultados recientes adjuntos."

    def _show_cda_form(self):
        if not self._ensure_db_connection():
            return

        form = tk.Toplevel(self.root)
        form.title("Nota clínica / CDA")
        form.geometry("740x540")
        form.transient(self.root)
        form.grab_set()

        kind_var = tk.StringVar(value="clinical_note")
        clinician_var = tk.StringVar(value="")
        signature_var = tk.StringVar(value="")

        tk.Label(form, text="Tipo de reporte:").pack(anchor="w", padx=10, pady=(10, 2))
        ttk.Combobox(
            form,
            textvariable=kind_var,
            values=["clinical_note", "mechanical_advantage", "force_moment"],
            state="readonly",
            width=35,
        ).pack(anchor="w", padx=10)

        tk.Label(form, text="Nombre completo de clínico:").pack(anchor="w", padx=10, pady=(10, 2))
        tk.Entry(form, textvariable=clinician_var, width=65).pack(anchor="w", padx=10)

        tk.Label(form, text="Firma digital/hash:").pack(anchor="w", padx=10, pady=(10, 2))
        tk.Entry(form, textvariable=signature_var, width=65).pack(anchor="w", padx=10)

        tk.Label(form, text="Cuerpo de la nota:").pack(anchor="w", padx=10, pady=(10, 2))
        body_text = tk.Text(form, height=14)
        body_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        body_text.insert(tk.END, self._default_cda_body())

        def save_note():
            try:
                body = body_text.get("1.0", tk.END).strip()
                if not body:
                    raise ValueError("El cuerpo de la nota es obligatorio")
                if not clinician_var.get().strip():
                    raise ValueError("El nombre del clinico es obligatorio")
                if not signature_var.get().strip():
                    raise ValueError("La firma digital/hash es obligatoria")

                self.db_client.save_cda_report(
                    patient_uuid=self.db_patient_uuid_var.get().strip(),
                    session_id=self.db_session_id,
                    report_kind=kind_var.get().strip(),
                    clinician_full_name=clinician_var.get().strip(),
                    signature_digital_hash=signature_var.get().strip(),
                    body_text=body,
                )
                messagebox.showinfo("CDA", "Nota clínica guardada en DB.")
                self._refresh_history_view()
                form.destroy()
            except Exception as e:
                messagebox.showerror("CDA", f"No se pudo guardar nota clínica: {e}")

        tk.Button(form, text="Guardar nota clínica", command=save_note).pack(pady=8)

    def _show_patient_form(self):
        form = tk.Toplevel(self.root)
        form.title("Datos generales del paciente")
        form.geometry("500x600")
        form.transient(self.root)
        form.grab_set()
        entries: dict[str, tk.Entry] = {}
        birth_entry: Optional[tk.Entry] = None
        age_entry: Optional[tk.Entry] = None

        def _update_age_from_birth() -> None:
            if birth_entry is None or age_entry is None:
                return
            birth_str = birth_entry.get().strip()
            if not birth_str:
                return
            try:
                dob = datetime.strptime(birth_str, "%d/%m/%Y").date()
                today = date.today()
                years = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
                age_entry.delete(0, tk.END)
                age_entry.insert(0, str(max(years, 0)))
            except Exception:
                # Si el formato no es válido, no forzamos nada
                pass

        def _open_birthdate_picker() -> None:
            if birth_entry is None:
                return
            picker = tk.Toplevel(form)
            picker.title("Seleccionar fecha de nacimiento")
            picker.geometry("260x140")
            picker.transient(form)
            picker.grab_set()

            # Fecha inicial (intentar leer del campo, si no usar hoy)
            initial = date.today()
            try:
                current = birth_entry.get().strip()
                if current:
                    initial = datetime.strptime(current, "%d/%m/%Y").date()
            except Exception:
                pass

            tk.Label(picker, text="Día:").grid(row=0, column=0, padx=6, pady=4, sticky="w")
            day_var = tk.IntVar(value=initial.day)
            tk.Spinbox(picker, from_=1, to=31, textvariable=day_var, width=4).grid(row=0, column=1, padx=4, pady=4, sticky="w")

            tk.Label(picker, text="Mes:").grid(row=1, column=0, padx=6, pady=4, sticky="w")
            month_var = tk.IntVar(value=initial.month)
            tk.Spinbox(picker, from_=1, to=12, textvariable=month_var, width=4).grid(row=1, column=1, padx=4, pady=4, sticky="w")

            tk.Label(picker, text="Año:").grid(row=2, column=0, padx=6, pady=4, sticky="w")
            year_to = date.today().year
            year_from = year_to - 120
            year_var = tk.IntVar(value=initial.year)
            tk.Spinbox(picker, from_=year_from, to=year_to, textvariable=year_var, width=6).grid(row=2, column=1, padx=4, pady=4, sticky="w")

            def _apply_date() -> None:
                try:
                    chosen = date(year_var.get(), month_var.get(), day_var.get())
                except Exception:
                    messagebox.showerror("Fecha", "Fecha de nacimiento no válida")
                    return
                birth_entry.delete(0, tk.END)
                birth_entry.insert(0, chosen.strftime("%d/%m/%Y"))
                _update_age_from_birth()
                picker.destroy()

            btn_row = tk.Frame(picker)
            btn_row.grid(row=3, column=0, columnspan=2, pady=8)
            tk.Button(btn_row, text="Cancelar", command=picker.destroy).pack(side="left", padx=4)
            tk.Button(btn_row, text="Aceptar", command=_apply_date).pack(side="left", padx=4)

        row = 0
        for k in self.patient_data:
            tk.Label(form, text=k+":").grid(row=row, column=0, sticky="w", padx=8, pady=4)
            if k == "Fecha de nacimiento":
                e = tk.Entry(form, width=20)
                e.insert(0, self.patient_data[k])
                e.grid(row=row, column=1, padx=8, pady=4, sticky="w")
                birth_entry = e
                tk.Button(form, text="Elegir fecha", command=_open_birthdate_picker).grid(
                    row=row, column=2, padx=(0, 8), pady=4, sticky="w"
                )
            else:
                e = tk.Entry(form, width=40)
                e.insert(0, self.patient_data[k])
                e.grid(row=row, column=1, columnspan=2, padx=8, pady=4, sticky="w")
            entries[k] = e
            if k == "Edad":
                age_entry = e
            row += 1

        # Si ya hay una fecha cargada, intentar precalcular edad
        _update_age_from_birth()
        def save_and_close():
            # Recalcular edad por si la fecha se escribió manualmente
            _update_age_from_birth()
            for k in self.patient_data:
                self.patient_data[k] = entries[k].get()
            if self.db_client is not None and self.db_enabled_var.get() and self.db_patient_uuid_var.get().strip():
                try:
                    self.db_client.upsert_patient_identity(
                        self.db_patient_uuid_var.get().strip(),
                        self.patient_data.get("Nombre", "").strip() or "PACIENTE SIN NOMBRE",
                        json.dumps(self.patient_data, ensure_ascii=False),
                    )
                except Exception:
                    pass
            self._refresh_history_view()
            form.destroy()
        tk.Button(form, text="Guardar", command=save_and_close).grid(row=row, column=0, columnspan=2, pady=12)

    def _refresh_cameras(self):
        # Solo webcams USB
        self._camera_options = list_cameras()
        self._camera_options_knee = self._camera_options
        self._camera_options_posture = self._camera_options
        self._camera_options_chains = self._camera_options
        if self._camera_options:
            self.foot_camera_var.set(self._camera_options[0][1])
            self.knee_camera_var.set(self._camera_options[0][1])
            self.posture_camera_var.set(self._camera_options[0][1])
            if hasattr(self, "chains_camera_var"):
                self.chains_camera_var.set(self._camera_options[0][1])
        else:
            self.foot_camera_var.set("")
            self.knee_camera_var.set("")
            self.posture_camera_var.set("")
            if hasattr(self, "chains_camera_var"):
                self.chains_camera_var.set("")
        self._build_ui()

    def _open_external_file(self, path: str) -> None:
        """Abre un archivo con el visor por defecto del sistema (Linux/Windows/macOS)."""
        if not path or not os.path.exists(path):
            return
        try:
            import subprocess
            import sys

            if os.name == "nt":  # Windows
                os.startfile(path)  # type: ignore[attr-defined]
            elif sys.platform == "darwin":  # macOS
                subprocess.Popen(["open", path])
            else:  # Asumir Linux/Unix
                subprocess.Popen(["xdg-open", path])
        except Exception:
            # Si no se puede abrir, simplemente no interrumpir el flujo clínico
            pass

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
        ttk.Label(title_col, text="Suite de Visión Biomecánica", style="Title.TLabel").pack(anchor="w")
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

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill="both", expand=True, padx=14, pady=8)

        # Marcos internos para los módulos (no se muestran como pestañas)
        self.tab_foot = ttk.Frame(self.notebook)
        self.tab_knee = ttk.Frame(self.notebook)
        self.tab_posture = ttk.Frame(self.notebook)
        self.tab_chains = ttk.Frame(self.notebook)
        self.tab_history = ttk.Frame(self.notebook)
        # Pestaña visible: evaluación completa (test único guiado)
        self.tab_lever = ttk.Frame(self.notebook)
        self.tab_fulltest = ttk.Frame(self.notebook)

        # Solo mostramos la evaluación completa en la interfaz
        self.notebook.add(self.tab_fulltest, text="Evaluación completa")

        self._build_foot_tab()
        self._build_knee_tab()
        self._build_posture_tab()
        self._build_chains_tab()
        self._build_lever_tab()
        self._build_history_tab()
        self._build_fulltest_tab()

    def _build_chains_tab(self):
        for child in self.tab_chains.winfo_children():
            child.destroy()

        controls = ttk.Frame(self.tab_chains, style="Card.TFrame")
        controls.pack(fill="x", padx=10, pady=8)

        ttk.Label(controls, text="Cámara:", style="Body.TLabel").pack(side="left", padx=(0, 2))
        self.chains_camera_var = tk.StringVar()
        cam_combo = ttk.Combobox(controls, textvariable=self.chains_camera_var, state="readonly", width=30)
        cam_combo['values'] = [name for idx, name in self._camera_options]
        cam_combo.pack(side="left", padx=(0, 8))
        if self._camera_options:
            self.chains_camera_var.set(self._camera_options[0][1])
        ttk.Button(controls, text="Actualizar cámaras", command=lambda: self._update_camera_combo(cam_combo, self.chains_camera_var)).pack(side="left", padx=(0, 8))

        ttk.Label(controls, text="Plano:", style="Body.TLabel").pack(side="left", padx=(8, 2))
        self.chains_plane_var = tk.StringVar(value="sagittal")
        ttk.Combobox(controls, textvariable=self.chains_plane_var, values=["sagittal", "frontal"], state="readonly", width=12).pack(side="left", padx=(0, 8))

        ttk.Label(controls, text="Lado:", style="Body.TLabel").pack(side="left", padx=(8, 2))
        self.chains_profile_side_var = tk.StringVar(value="Automático")
        ttk.Combobox(
            controls,
            textvariable=self.chains_profile_side_var,
            values=["Automático", "Izquierdo", "Derecho"],
            state="readonly",
            width=12,
        ).pack(side="left", padx=(0, 8))

        ttk.Button(controls, text="Cargar imagen", command=self._load_chains_image).pack(side="left", padx=4)
        ttk.Button(controls, text="Analizar captura", style="Primary.TButton", command=self._analyze_chains).pack(side="left", padx=4)
        ttk.Button(controls, text="Guardar resultado", command=self._save_chains).pack(side="left", padx=4)

        live_controls = ttk.Frame(self.tab_chains, style="Card.TFrame")
        live_controls.pack(fill="x", padx=10, pady=(0, 8))

        ttk.Label(live_controls, text="Modo calibración:", style="Body.TLabel").pack(side="left", padx=(0, 2))
        self.chains_calibration_mode_var = tk.StringVar(value="Automático")
        ttk.Combobox(
            live_controls,
            textvariable=self.chains_calibration_mode_var,
            values=["Automático", "Sin calibración", "Referencia", "Altura", "ArUco"],
            state="readonly",
            width=16,
        ).pack(side="left", padx=(0, 8))
        ttk.Label(live_controls, text="Referencia real (mm):", style="Body.TLabel").pack(side="left", padx=(4, 2))
        self.chains_reference_mm_var = tk.StringVar(value="100.0")
        ttk.Entry(live_controls, textvariable=self.chains_reference_mm_var, width=8).pack(side="left", padx=(0, 8))
        ttk.Label(live_controls, text="Referencia (px):", style="Body.TLabel").pack(side="left", padx=(4, 2))
        self.chains_reference_px_var = tk.StringVar(value="100.0")
        ttk.Entry(live_controls, textvariable=self.chains_reference_px_var, width=8).pack(side="left", padx=(0, 8))
        ttk.Label(live_controls, text="ArUco (mm):", style="Body.TLabel").pack(side="left", padx=(4, 2))
        self.chains_aruco_marker_mm_var = tk.StringVar(value="50.0")
        ttk.Entry(live_controls, textvariable=self.chains_aruco_marker_mm_var, width=8).pack(side="left", padx=(0, 8))
        ttk.Label(live_controls, text="Altura paciente (mm):", style="Body.TLabel").pack(side="left", padx=(4, 2))
        self.chains_patient_height_mm_var = tk.StringVar(value="1700.0")
        ttk.Entry(live_controls, textvariable=self.chains_patient_height_mm_var, width=8).pack(side="left", padx=(0, 8))
        self.chains_calibration_pick_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(live_controls, text="Tomar 2 puntos de referencia", variable=self.chains_calibration_pick_var).pack(side="left", padx=(4, 8))
        ttk.Button(live_controls, text="Reiniciar puntos", command=self._clear_chains_calibration).pack(side="left", padx=4)
        ttk.Button(live_controls, text="Iniciar/Detener video", style="Primary.TButton", command=self._toggle_chains_live).pack(side="left", padx=4)

        self.chains_calibration_status_var = tk.StringVar(value="Confianza calibración: Baja (Sin escala)")
        self.chains_calibration_status_label = tk.Label(
            self.tab_chains,
            textvariable=self.chains_calibration_status_var,
            bg=self.bg_card,
            fg="#ef4444",
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        )
        self.chains_calibration_status_label.pack(fill="x", padx=12, pady=(0, 6))

        self.chains_preview_canvas = tk.Canvas(self.tab_chains, width=700, height=520, bg="#0b1220", highlightthickness=0)
        self.chains_preview_canvas.pack(fill="both", expand=False, padx=10, pady=(10, 0))
        self.chains_preview_canvas.bind("<Button-1>", self._on_chains_canvas_click)

        metrics_frame = ttk.LabelFrame(self.tab_chains, text="Métricas y cadenas", style="Card.TLabelframe")
        metrics_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.chains_metrics_text = tk.Text(metrics_frame, height=12)
        self.chains_metrics_text.pack(fill="both", expand=True, padx=8, pady=8)
        self._configure_text_widget(self.chains_metrics_text)

    def _build_history_tab(self):
        for child in self.tab_history.winfo_children():
            child.destroy()

        header = ttk.Frame(self.tab_history, style="Card.TFrame")
        header.pack(fill="x", padx=10, pady=(10, 8))
        ttk.Label(header, text="Historial clínico del paciente activo", style="Body.TLabel").pack(anchor="w")

        controls = ttk.Frame(header, style="Card.TFrame")
        controls.pack(fill="x", pady=(6, 0))

        ttk.Label(controls, text="UUID del paciente:", style="Hint.TLabel").pack(side="left")
        entry = ttk.Entry(controls, textvariable=self.db_patient_uuid_var, width=40)
        entry.pack(side="left", padx=(5, 10))

        ttk.Button(controls, text="Refrescar", command=self._refresh_history_view).pack(side="left")
        ttk.Button(controls, text="Abrir último consentimiento", command=self._open_last_consent).pack(side="left", padx=(10, 0))

        body = ttk.Frame(self.tab_history, style="Card.TFrame")
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.history_text = tk.Text(body, height=26)
        self._configure_text_widget(self.history_text)
        self.history_text.pack(fill="both", expand=True, padx=10, pady=10)
        self._refresh_history_view()

    # -------------------------
    # Evaluación completa (test único)
    # -------------------------

    def _build_fulltest_tab(self):
        """Construye la pestaña de flujo guiado para una evaluación completa."""
        for child in self.tab_fulltest.winfo_children():
            child.destroy()

        container = ttk.Frame(self.tab_fulltest, style="Card.TFrame")
        container.pack(fill="both", expand=True, padx=10, pady=10)

        # Panel superior con resumen de pasos
        self.fulltest_steps_panel = tk.Text(container, height=5)
        self.fulltest_steps_panel.pack(fill="x", pady=(0, 6))
        self._configure_text_widget(self.fulltest_steps_panel)
        self.fulltest_steps_panel.configure(state="disabled")

        self.fulltest_title_var = tk.StringVar()
        self.fulltest_status_var = tk.StringVar(value="Sigue los pasos para completar un solo estudio biomecánico.")

        ttk.Label(container, textvariable=self.fulltest_title_var, style="Body.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(container, textvariable=self.fulltest_status_var, style="Hint.TLabel").pack(anchor="w", pady=(2, 8))

        self.fulltest_body = ttk.Frame(container, style="Card.TFrame")
        self.fulltest_body.pack(fill="both", expand=True, pady=(4, 8))

        nav = ttk.Frame(container, style="Card.TFrame")
        nav.pack(fill="x")

        self.fulltest_prev_btn = ttk.Button(nav, text="⟵ Anterior", command=self._fulltest_prev_step)
        self.fulltest_prev_btn.pack(side="left")
        self.fulltest_next_btn = ttk.Button(nav, text="Siguiente ⟶", command=self._fulltest_next_step)
        self.fulltest_next_btn.pack(side="right")

        # Definir pasos del test único
        self.fulltest_steps = [
            ("intro", "0. Inicio"),
            ("foot", "1. Huella plantar (Baropodometría)"),
            ("knee", "2. Rodilla"),
            ("posture", "3. Postura global"),
            ("chains", "4. Cadenas musculares"),
            ("lever", "5. Palancas y torque"),
            ("summary", "6. Cierre y nota clínica"),
        ]

        # Asegurar un índice válido y renderizar
        self.fulltest_step_index = min(max(self.fulltest_step_index, 0), len(self.fulltest_steps) - 1)
        self._fulltest_render_step()

    def _fulltest_render_step(self):
        for child in self.fulltest_body.winfo_children():
            child.destroy()

        step_id, step_label = self.fulltest_steps[self.fulltest_step_index]
        self.fulltest_title_var.set(step_label)

        # Actualizar panel de pasos (checklist visual)
        if hasattr(self, "fulltest_steps_panel"):
            self.fulltest_steps_panel.configure(state="normal")
            self.fulltest_steps_panel.delete("1.0", tk.END)
            for idx, (sid, label) in enumerate(self.fulltest_steps):
                marker = "✔" if sid in self.fulltest_completed_steps else "•"
                current = " ⟵" if idx == self.fulltest_step_index else ""
                self.fulltest_steps_panel.insert(tk.END, f"{marker} {label}{current}\n")
            self.fulltest_steps_panel.configure(state="disabled")

        # Actualizar estado de navegación
        self.fulltest_prev_btn.state(["!disabled"] if self.fulltest_step_index > 0 else ["disabled"])
        self.fulltest_next_btn.state(["!disabled"] if self.fulltest_step_index < len(self.fulltest_steps) - 1 else ["disabled"])

        if step_id == "intro":
            self._fulltest_render_intro()
        elif step_id == "foot":
            self._fulltest_render_foot_step()
        elif step_id == "knee":
            self._fulltest_render_knee_step()
        elif step_id == "posture":
            self._fulltest_render_posture_step()
        elif step_id == "chains":
            self._fulltest_render_chains_step()
        elif step_id == "lever":
            self._fulltest_render_lever_step()
        elif step_id == "summary":
            self._fulltest_render_summary_step()

    def _fulltest_prev_step(self):
        if self.fulltest_step_index > 0:
            self.fulltest_step_index -= 1
            self._fulltest_render_step()

    def _fulltest_next_step(self):
        if self.fulltest_step_index < len(self.fulltest_steps) - 1:
            self.fulltest_step_index += 1
            self._fulltest_render_step()

    def _fulltest_mark_completed(self, step_id: str):
        self.fulltest_completed_steps.add(step_id)
        self.fulltest_status_var.set(f"Paso '{step_id}' marcado como completado en esta evaluación.")
        # Refrescar el panel de pasos para mostrar el check
        self._fulltest_render_step()

    # ----- Contenido de cada paso -----

    def _fulltest_render_intro(self):
        text = (
            "Este asistente ejecuta una EVALUACIÓN BIOMECÁNICA COMPLETA como una sola prueba:\n\n"
            "1) Capturar datos del paciente y configurar DB.\n"
            "2) Registrar consentimiento informado.\n"
            "3) Huella plantar (pie).\n"
            "4) Rodilla.\n"
            "5) Postura global.\n"
            "6) Cadenas musculares.\n"
            "7) Palancas y torque.\n\n"
            "Primero completa los datos clínicos del paciente, la configuración de base de datos y el consentimiento,\n"
            "y después avanza a las pruebas de imagen.\n"
        )

        txt = tk.Text(self.fulltest_body, height=10)
        txt.pack(fill="x", expand=False, padx=8, pady=(8, 4))
        self._configure_text_widget(txt)
        txt.insert(tk.END, text)
        txt.configure(state="disabled")

        controls = ttk.Frame(self.fulltest_body, style="Card.TFrame")
        controls.pack(fill="x", pady=(4, 8))

        ttk.Button(controls, text="Datos del paciente", command=self._show_patient_form).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Configurar base de datos", command=self._show_db_form).pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Registrar consentimiento", command=self._show_consent_form).pack(side="left", padx=(0, 8))

    def _fulltest_render_foot_step(self):
        ttk.Label(self.fulltest_body, text="Huella plantar (Baropodometría)", style="Body.TLabel").pack(anchor="w", pady=(0, 4))
        ttk.Label(
            self.fulltest_body,
            text="1) Captura la huella plantar desde aquí y 2) Ejecuta el análisis y guardado como parte de este test.",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(0, 6))

        cam_row = ttk.Frame(self.fulltest_body, style="Card.TFrame")
        cam_row.pack(fill="x", pady=(2, 4))
        ttk.Label(cam_row, text="Cámara:", style="Body.TLabel").pack(side="left", padx=(0, 4))
        cam_combo = ttk.Combobox(cam_row, textvariable=self.foot_camera_var, state="readonly", width=32)
        cam_combo['values'] = [name for idx, name in self._camera_options]
        cam_combo.pack(side="left", padx=(0, 4))
        if self._camera_options and not self.foot_camera_var.get():
            self.foot_camera_var.set(self._camera_options[0][1])
        ttk.Button(cam_row, text="Actualizar cámaras", command=lambda: self._update_camera_combo(cam_combo, self.foot_camera_var)).pack(side="left", padx=(4, 0))

        btns = ttk.Frame(self.fulltest_body, style="Card.TFrame")
        btns.pack(fill="x", pady=(4, 4))

        ttk.Button(btns, text="Capturar foto de pie", command=self._fulltest_capture_foot).pack(side="left", padx=(0, 8))
        ttk.Button(
            btns,
            text="Cargar imagen de pie",
            command=lambda: self._load_image(self.foot_state, self.foot_original_lbl, self.foot_result_lbl),
        ).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Analizar pie y guardar en DB", command=self._fulltest_analyze_foot).pack(side="left", padx=(0, 8))

    def _fulltest_render_knee_step(self):
        ttk.Label(self.fulltest_body, text="Rodilla", style="Body.TLabel").pack(anchor="w", pady=(0, 4))
        ttk.Label(
            self.fulltest_body,
            text="Captura y analiza la rodilla dentro de este mismo estudio.",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(0, 6))

        cam_row = ttk.Frame(self.fulltest_body, style="Card.TFrame")
        cam_row.pack(fill="x", pady=(2, 4))
        ttk.Label(cam_row, text="Cámara:", style="Body.TLabel").pack(side="left", padx=(0, 4))
        cam_combo = ttk.Combobox(cam_row, textvariable=self.knee_camera_var, state="readonly", width=32)
        cam_combo['values'] = [name for idx, name in self._camera_options_knee]
        cam_combo.pack(side="left", padx=(0, 4))
        if self._camera_options_knee and not self.knee_camera_var.get():
            self.knee_camera_var.set(self._camera_options_knee[0][1])
        ttk.Button(cam_row, text="Actualizar cámaras", command=lambda: self._update_camera_combo(cam_combo, self.knee_camera_var)).pack(side="left", padx=(4, 0))

        btns = ttk.Frame(self.fulltest_body, style="Card.TFrame")
        btns.pack(fill="x", pady=(4, 4))

        ttk.Button(btns, text="Capturar foto de rodilla", command=self._fulltest_capture_knee).pack(side="left", padx=(0, 8))
        ttk.Button(
            btns,
            text="Cargar imagen de rodilla",
            command=lambda: self._load_image(self.knee_state, self.knee_original_lbl, self.knee_result_lbl),
        ).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Analizar rodilla y guardar en DB", command=self._fulltest_analyze_knee).pack(side="left", padx=(0, 8))

    def _fulltest_render_posture_step(self):
        ttk.Label(self.fulltest_body, text="Postura global", style="Body.TLabel").pack(anchor="w", pady=(0, 4))
        ttk.Label(
            self.fulltest_body,
            text="Captura y analiza la postura en el mismo flujo de estudio.",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(0, 6))

        cam_row = ttk.Frame(self.fulltest_body, style="Card.TFrame")
        cam_row.pack(fill="x", pady=(2, 4))
        ttk.Label(cam_row, text="Cámara:", style="Body.TLabel").pack(side="left", padx=(0, 4))
        cam_combo = ttk.Combobox(cam_row, textvariable=self.posture_camera_var, state="readonly", width=32)
        cam_combo['values'] = [name for idx, name in self._camera_options_posture]
        cam_combo.pack(side="left", padx=(0, 4))
        if self._camera_options_posture and not self.posture_camera_var.get():
            self.posture_camera_var.set(self._camera_options_posture[0][1])
        ttk.Button(cam_row, text="Actualizar cámaras", command=lambda: self._update_camera_combo(cam_combo, self.posture_camera_var)).pack(side="left", padx=(4, 0))

        btns = ttk.Frame(self.fulltest_body, style="Card.TFrame")
        btns.pack(fill="x", pady=(4, 4))

        ttk.Button(btns, text="Capturar foto de postura", command=self._fulltest_capture_posture).pack(side="left", padx=(0, 8))
        ttk.Button(
            btns,
            text="Cargar imagen de postura",
            command=lambda: self._load_image(self.posture_state, self.posture_original_lbl, self.posture_result_lbl),
        ).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Analizar postura y guardar en DB", command=self._fulltest_analyze_posture).pack(side="left", padx=(0, 8))

    def _fulltest_render_chains_step(self):
        ttk.Label(self.fulltest_body, text="Cadenas musculares", style="Body.TLabel").pack(anchor="w", pady=(0, 4))
        ttk.Label(
            self.fulltest_body,
            text="Puedes analizar cadenas musculares aquí mismo usando la cámara en vivo. Elige cámara y usa 'Iniciar/Detener video'.",
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(0, 6))

        # Controles de cámara y análisis en vivo
        controls = ttk.Frame(self.fulltest_body, style="Card.TFrame")
        controls.pack(fill="x", padx=10, pady=8)

        # Selección de cámara
        ttk.Label(controls, text="Cámara:", style="Body.TLabel").pack(side="left", padx=(0, 2))
        if not hasattr(self, "chains_camera_var_fulltest"):
            self.chains_camera_var_fulltest = tk.StringVar()
            if self._camera_options:
                self.chains_camera_var_fulltest.set(self._camera_options[0][1])
        cam_combo = ttk.Combobox(controls, textvariable=self.chains_camera_var_fulltest, state="readonly", width=30)
        cam_combo['values'] = [name for idx, name in self._camera_options]
        cam_combo.pack(side="left", padx=(0, 8))
        ttk.Button(controls, text="Actualizar cámaras", command=lambda: self._update_camera_combo(cam_combo, self.chains_camera_var_fulltest)).pack(side="left", padx=(0, 8))

        # Botón para iniciar/detener video en vivo
        ttk.Button(controls, text="Iniciar/Detener video", style="Primary.TButton", command=lambda: self._toggle_chains_live_fulltest()).pack(side="left", padx=4)

        # Área de previsualización
        if not hasattr(self, "chains_preview_canvas_fulltest"):
            self.chains_preview_canvas_fulltest = tk.Canvas(self.fulltest_body, width=700, height=350, bg="#0b1220", highlightthickness=0)
        self.chains_preview_canvas_fulltest.pack(fill="x", padx=10, pady=(10, 0))

        # Métricas
        metrics_frame = ttk.LabelFrame(self.fulltest_body, text="Métricas y cadenas", style="Card.TLabelframe")
        metrics_frame.pack(fill="x", padx=10, pady=10)
        if not hasattr(self, "chains_metrics_text_fulltest"):
            self.chains_metrics_text_fulltest = tk.Text(metrics_frame, height=8)
            self._configure_text_widget(self.chains_metrics_text_fulltest)
        self.chains_metrics_text_fulltest.pack(fill="x", padx=8, pady=8)

        # Botón para marcar como completado
        ttk.Button(self.fulltest_body, text="Marcar cadenas como completadas", command=lambda: self._fulltest_mark_completed("chains")).pack(anchor="w", pady=(8, 0))

    def _fulltest_render_lever_step(self):
        ttk.Label(self.fulltest_body, text="Palancas y torque", style="Body.TLabel").pack(anchor="w", pady=(0, 4))
        ttk.Label(
            self.fulltest_body,
            text=(
                "Introduce los datos de palancas y torque en su pestaña y luego márcalo como completado "
                "para integrarlo a este mismo estudio."
            ),
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(0, 6))

        btns = ttk.Frame(self.fulltest_body, style="Card.TFrame")
        btns.pack(fill="x", pady=(4, 4))

        ttk.Button(btns, text="Marcar palancas como completadas", command=lambda: self._fulltest_mark_completed("lever")).pack(side="left", padx=(0, 8))

    def _fulltest_render_summary_step(self):
        ttk.Label(self.fulltest_body, text="Cierre del estudio", style="Body.TLabel").pack(anchor="w", pady=(0, 4))
        ttk.Label(
            self.fulltest_body,
            text=(
                "Con este paso cierras la EVALUACIÓN COMPLETA como una sola prueba clínica.\n"
                "Puedes: 1) Revisar el historial clínico, 2) Registrar una nota clínica/CDA, "
                "y 3) Generar PDFs desde cada módulo si lo deseas."
            ),
            style="Hint.TLabel",
        ).pack(anchor="w", pady=(0, 6))

        btns = ttk.Frame(self.fulltest_body, style="Card.TFrame")
        btns.pack(fill="x", pady=(4, 4))

        ttk.Button(btns, text="Registrar nota clínica (CDA)", command=self._show_cda_form).pack(side="left", padx=(0, 8))

    # ----- Acciones rápidas para capturar y analizar dentro del flujo -----

    def _fulltest_capture_foot(self):
        idx = self._get_camera_index(self.foot_camera_var.get() or "0")
        self._capture_image(self.foot_state, self.foot_original_lbl, self.foot_result_lbl, camera_index=idx)

    def _fulltest_analyze_foot(self):
        self._analyze_foot()
        self._fulltest_mark_completed("foot")

    def _fulltest_capture_knee(self):
        idx = self._get_camera_index(self.knee_camera_var.get() or "0")
        self._capture_image(self.knee_state, self.knee_original_lbl, self.knee_result_lbl, camera_index=idx)

    def _fulltest_analyze_knee(self):
        self._analyze_knee()
        self._fulltest_mark_completed("knee")

    def _fulltest_capture_posture(self):
        idx = self._get_camera_index(self.posture_camera_var.get() or "0")
        self._capture_image(self.posture_state, self.posture_original_lbl, self.posture_result_lbl, camera_index=idx)

    def _fulltest_analyze_posture(self):
        self._analyze_posture()
        self._fulltest_mark_completed("posture")

    def _open_last_consent(self):
        """Recupera y abre en PDF el último consentimiento del paciente activo."""
        if not self.db_enabled_var.get() or not self._ensure_db_connection():
            messagebox.showerror("Base de datos", "La base de datos no está configurada o conectada.")
            return

        patient_uuid = (self.db_patient_uuid_var.get() or "").strip()
        if not patient_uuid:
            messagebox.showwarning("Paciente", "Introduce el UUID del paciente para buscar su consentimiento.")
            return

        try:
            consent = self.db_client.fetch_latest_consent(patient_uuid)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Consentimiento", f"Error al recuperar el consentimiento: {exc}")
            return

        if not consent:
            messagebox.showinfo("Consentimiento", "No se encontró ningún consentimiento para este paciente.")
            return

        pdf_bytes = consent.get("consent_document") if isinstance(consent, dict) else None
        if not pdf_bytes:
            messagebox.showinfo("Consentimiento", "El consentimiento recuperado no tiene documento asociado.")
            return

        try:
            output_dir = self._ensure_output_dir()
            file_name = f"consentimiento_ultimo_{patient_uuid}.pdf"
            out_path = os.path.join(output_dir, file_name)
            with open(out_path, "wb") as f:
                f.write(pdf_bytes)

            self._open_external_file(out_path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Consentimiento", f"No se pudo abrir el PDF del consentimiento: {exc}")

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
        if not self._ensure_consent_or_warn():
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
            self._persist_ui_analysis("foot", m, text)
            self._refresh_history_view()
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
        if not self._ensure_consent_or_warn():
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
            self._persist_ui_analysis("knee", m, text)
            self._refresh_history_view()
        except Exception as e:
            messagebox.showerror("Rodilla", f"Error: {e}")
        finally:
            self._clear_status()

    def _analyze_posture(self):
        if self.posture_state.source_image is None:
            messagebox.showwarning("Postura", "Primero carga o captura una imagen.")
            return
        if not self._ensure_consent_or_warn():
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
            self._persist_ui_analysis("posture", m, text)
            self._refresh_history_view()
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
