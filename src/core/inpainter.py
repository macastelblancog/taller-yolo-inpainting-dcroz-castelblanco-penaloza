# src/core/inpainter.py

import numpy as np
import logging
from pathlib import Path

from src.config import settings
from PIL import Image
from simple_lama_inpainting import SimpleLama

# TOFIX: Importar dependencias de LaMa cuando se implemente run_inpainting().
# TOFIX: Importar dependencias de Inpaint-Anything como alternativa al backend.

logger = logging.getLogger(__name__)

# TODO: Funcionalidad de run_inpainting() para eliminar postes usando
#       el backend configurado en settings.inpainting.backend ("lama" | "inpaint_anything")
# TODO: Funcionalidad de load_inpainter() para cargar el modelo de inpainting
#       desde settings.paths.lama_model

logger = logging.getLogger(__name__)

# Instancia global para no recargar el modelo en cada llamada
_lama_instance: SimpleLama | None = None


def load_inpainter() -> SimpleLama:
    """
    Carga el modelo LaMa una sola vez y reutiliza la instancia
    en llamadas posteriores (patrón singleton).

    El modelo se descarga automáticamente la primera vez desde
    simple_lama_inpainting si no está en caché local.
    """
    global _lama_instance
    if _lama_instance is None:
        logger.info("Cargando modelo LaMa...")
        _lama_instance = SimpleLama()
        logger.info("Modelo LaMa listo.")
    return _lama_instance


def run_inpainting(
    image: Image.Image,
    mask: Image.Image,
    save_path: str | Path = None,
) -> Image.Image:
    """
    Elimina los postes de la imagen aplicando inpainting con LaMa
    sobre las zonas indicadas por la máscara.

    Parámetros
    ----------
    image     : imagen PIL original en RGB
                (salida de detector.load_image).
    mask      : imagen PIL de la máscara en modo 'L'
                (salida de masker.generate_mask).
                Blanco (255) = zona a reconstruir (poste).
                Negro  (0)   = zona a conservar intacta.
    save_path : ruta donde guardar el resultado final.
                Si None, usa settings.paths.results_dir.

    Retorna
    -------
    result : imagen PIL con los postes eliminados.
    """

    # ── Asegurar formatos correctos para LaMa ────────────────────────────────
    if image.mode != "RGB":
        image = image.convert("RGB")
    if mask.mode != "L":
        mask = mask.convert("L")

    # ── Verificar que la máscara no esté vacía ────────────────────────────────
    mask_np = np.array(mask)
    if np.sum(mask_np == 255) == 0:
        logger.warning("run_inpainting: la máscara está vacía, "
                       "se devuelve la imagen original sin cambios.")
        return image

    # ── Aplicar LaMa ─────────────────────────────────────────────────────────
    logger.info("Aplicando LaMa inpainting...")
    lama   = load_inpainter()
    result = lama(image, mask)
    logger.info("Inpainting completado.")

    # ── Guardar resultado ─────────────────────────────────────────────────────
    if save_path is None:
        results_dir = settings.paths.results_dir / "inpainting"
        results_dir.mkdir(parents=True, exist_ok=True)
        save_path = results_dir / "result.png"

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(save_path))
    logger.info(f"Resultado guardado: {save_path}")

    return result