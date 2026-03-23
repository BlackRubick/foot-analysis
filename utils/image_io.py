from __future__ import annotations

import os
from typing import Optional

import cv2


def load_image(path: str):
    if not path:
        raise ValueError("No se proporcionó ruta de imagen")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No existe el archivo: {path}")

    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen: {path}")
    return image


def save_image(path: str, image) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)


def show_image(window_name: str, image, wait: int = 0) -> None:
    cv2.imshow(window_name, image)
    cv2.waitKey(wait)


def destroy_windows() -> None:
    cv2.destroyAllWindows()
