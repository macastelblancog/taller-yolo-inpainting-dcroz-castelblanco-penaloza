# src/core/masker.py

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import settings

# TOFIX: Importar visualize_mask() desde src.utils cuando se agregue.

logger = logging.getLogger(__name__)


def generate_mask(
    pole_boxes: list,
    width: int,
    height: int,
    dilate_px: int = None,
) -> Image.Image:
    """
    Genera una máscara binaria a partir de bounding boxes de postes.
    Usa dilatación adaptativa — proporcional al tamaño del bbox para
    evitar enmascarar áreas grandes innecesariamente.

    Args:
        pole_boxes: Lista de bounding boxes xyxy — salida de filter_by_class().
        width:      Ancho de la imagen original en píxeles.
        height:     Alto de la imagen original en píxeles.
        dilate_px:  Margen base de dilatación. Si None usa
                    settings.mask.dilation_px.

    Returns:
        Imagen PIL en modo L — blanco donde hay poste, negro donde no.
    """
    dilate_px = dilate_px if dilate_px is not None else settings.mask.dilation_px

    mask_np = np.zeros((height, width), dtype=np.uint8)

    if len(pole_boxes) == 0:
        logger.warning(
            "generate_mask: no se recibieron bounding boxes. "
            "La máscara estará vacía."
        )
        return Image.fromarray(mask_np)

    for box in pole_boxes:
        x1, y1, x2, y2 = box
        box_w = x2 - x1
        box_h = y2 - y1

        # Dilatación adaptativa — máximo 10% del lado menor del bbox.
        # Evita enmascarar áreas enormes cuando el bbox es muy grande.
        adaptive_dilate = min(
            dilate_px,
            int(min(box_w, box_h) * 0.1)
        )

        x1 = max(0,      int(x1) - adaptive_dilate)
        y1 = max(0,      int(y1) - adaptive_dilate)
        x2 = min(width,  int(x2) + adaptive_dilate)
        y2 = min(height, int(y2) + adaptive_dilate)
        mask_np[y1:y2, x1:x2] = settings.mask.fill_value

    pct = np.sum(mask_np == settings.mask.fill_value) / mask_np.size * 100
    logger.info(
        f"Máscara generada — postes: {len(pole_boxes)}, "
        f"píxeles enmascarados: {pct:.1f}%, "
        f"dilatación base: {dilate_px}px"
    )

    return Image.fromarray(mask_np)


def save_mask(mask: Image.Image, image_path: str | Path) -> Path:
    """
    Guarda la máscara siguiendo la convención de nombres de LaMa:
        <nombre>_mask001.png

    Args:
        mask:       Imagen PIL de la máscara — salida de generate_mask().
        image_path: Ruta de la imagen original — usada para derivar el nombre.

    Returns:
        Ruta donde quedó guardada la máscara.
    """
    image_path = Path(image_path)
    masks_dir  = settings.paths.masks_dir
    masks_dir.mkdir(parents=True, exist_ok=True)

    mask_filename = f"{image_path.stem}_mask001.png"
    mask_path     = masks_dir / mask_filename

    mask.save(str(mask_path))
    logger.info(f"Máscara guardada: {mask_path}")

    return mask_path