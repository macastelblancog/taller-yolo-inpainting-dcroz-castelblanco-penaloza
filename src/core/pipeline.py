# src/core/pipeline.py

import logging
from pathlib import Path

from PIL import Image

from src.core import detector
from src.config import settings

# TOFIX: Importar masker cuando generate_mask() esté implementado.
# TOFIX: Importar inpainter cuando run_inpainting() esté implementado.

logger = logging.getLogger(__name__)


def run(image_path: str | Path, model=None, out_path: str | Path = None):
    """
    Orquesta el pipeline completo sobre una imagen.
    Por ahora ejecuta solo detección y visualización.

    Args:
        image_path: Ruta a la imagen de entrada.
        model:      Modelo YOLO ya cargado. Si None, se carga internamente.
        out_path:   Directorio o archivo de salida para la imagen anotada.

    Returns:
        Imagen PIL anotada con las detecciones.

    # TOFIX: Agregar paso 2 — generación de máscara con masker.generate_mask()
    #        filtrando solo detecciones de clase "poste".
    # TOFIX: Agregar paso 3 — inpainting con inpainter.run_inpainting()
    #        usando la máscara generada.
    # TOFIX: Reemplazar retorno por dataclass PipelineResult cuando
    #        se integren los tres pasos completos.
    # TOFIX: Mover lógica de output path a src/utils.py save_result()
    #        cuando esté implementada.
    """

    # ── Paso 1: Cargar imagen ─────────────────────────────────────────────────
    image = detector.load_image(image_path)
    if image is None:
        logger.error(f"No se pudo cargar la imagen: {image_path}")
        return None

    image_path = Path(image_path)

    # ── Resolver ruta de salida ───────────────────────────────────────────────
    if out_path is None:
        save_path = settings.paths.results_dir / "detections" / image_path.name
    else:
        out_path = Path(out_path)
        save_path = out_path / image_path.name if out_path.is_dir() else out_path

    save_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Paso 2: Detección ─────────────────────────────────────────────────────
    results, inference_ms = detector.detect(image, model)
    logger.info(f"Detección completada en {inference_ms:.1f}ms.")

    # ── Paso 3: Visualización y guardado ──────────────────────────────────────
    # TOFIX: Mover plot_detections() a src/utils.py como visualize_detections()
    #        y llamarla desde aquí en lugar de desde detector directamente.
    annotated = detector.plot_detections(image, results, save_path)

    # ── TODO: Paso 4 — Generación de máscara (masker) ─────────────────────────
    # pole_boxes = detector.filter_by_class(results, "poste")
    # mask = masker.generate_mask(pole_boxes, image.width, image.height)
    # masker.save_mask(mask, image_path)

    # ── TODO: Paso 5 — Inpainting ─────────────────────────────────────────────
    # inpainted = inpainter.run_inpainting(image, mask)

    return annotated