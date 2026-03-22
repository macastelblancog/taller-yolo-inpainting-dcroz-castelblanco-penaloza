# src/core/masker.py

import numpy as np
import cv2
from PIL import Image

import logging
from pathlib import Path

from src.config import settings

# TOFIX: Importar numpy y PIL cuando se implemente generate_mask().
# TOFIX: Importar visualize_mask() desde src.utils cuando se agregue.

logger = logging.getLogger(__name__)


def generate_mask(
    pole_boxes: np.ndarray,
    width: int,
    height: int,
    dilate_px: int = 15,
) -> Image.Image:
    
    # Máscara vacía — todo negro, no tocar nada por defecto
    mask_np = np.zeros((height, width), dtype=np.uint8)

    if len(pole_boxes) == 0:
        logger.warning("generate_mask: no se recibieron bounding boxes. "
                       "La máscara estará vacía.")
        return Image.fromarray(mask_np)

    # Pintar de blanco cada bbox de poste con margen de dilatación
    for box in pole_boxes:
        x1, y1, x2, y2 = box
        x1 = max(0,      int(x1) - dilate_px)
        y1 = max(0,      int(y1) - dilate_px)
        x2 = min(width,  int(x2) + dilate_px)
        y2 = min(height, int(y2) + dilate_px)
        mask_np[y1:y2, x1:x2] = 255

    # Suavizado morfológico: mejora la transición en el inpainting
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_np = cv2.dilate(mask_np, kernel, iterations=2)    

    pct = np.sum(mask_np == 255) / mask_np.size * 100
    logger.info(f"Máscara generada — postes: {len(pole_boxes)}, "
                f"píxeles enmascarados: {pct:.1f}%")

    return Image.fromarray(mask_np)


def save_mask(mask: Image.Image, image_path: str | Path) -> Path:
    """
    Guarda la máscara siguiendo la convención de nombres de LaMa:
        <nombre>_mask001.png

    La máscara se guarda en el directorio definido en settings.paths.masks_dir.

    Parámetros
    ----------
    mask       : imagen PIL de la máscara (salida de generate_mask).
    image_path : ruta de la imagen original, usada para derivar el nombre.

    Retorna
    -------
    mask_path : ruta donde quedó guardada la máscara.
    """
    image_path = Path(image_path)
    masks_dir  = settings.paths.masks_dir
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Convención LaMa: <nombre>_mask001.png
    mask_filename = f"{image_path.stem}_mask001.png"
    mask_path     = masks_dir / mask_filename

    mask.save(str(mask_path))
    logger.info(f"Máscara guardada: {mask_path}")

    return mask_path