from __future__ import annotations

from typing import Optional


def select_file_dialog(title: str = "Selecciona una imagen") -> Optional[str]:
    """Selector básico usando Tkinter. Retorna None si falla o se cancela."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title=title,
            filetypes=[("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")],
        )
        root.destroy()
        return path if path else None
    except Exception:
        return None
