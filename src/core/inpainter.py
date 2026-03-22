# src/core/inpainter.py

import logging
from pathlib import Path

from PIL import Image
from simple_lama_inpainting import SimpleLama

from src.config import settings

# TOFIX: settings.paths.lama_model no aplica para simple_lama_inpainting —
# el paquete gestiona sus propios pesos internamente en el venv.
# Relevante solo si se migra a LaMa completo o Inpaint-Anything.

logger = logging.getLogger(__name__)

# Instancia global — se carga una vez al importar el módulo.
# La primera llamada descarga los pesos automáticamente si no existen.
# TOFIX: Mover a un patrón de carga lazy o lifespan de FastAPI
# para evitar descarga en import time cuando no se necesita inpainting.
_lama = None


def _get_lama() -> SimpleLama:
    """
    Retorna la instancia de SimpleLama, creándola si no existe.
    Patrón lazy — solo descarga pesos cuando se necesita por primera vez.
    """
    global _lama
    if _lama is None:
        logger.info("Cargando modelo LaMa — primera vez puede tardar...")
        _lama = SimpleLama()
        logger.info("Modelo LaMa cargado correctamente.")
    return _lama


def run_inpainting(image: Image.Image, mask: Image.Image) -> Image.Image:
    """
    Elimina los postes de la imagen usando LaMa inpainting.

    Args:
        image: Imagen PIL original sin anotaciones.
        mask:  Imagen PIL de la máscara — blanco donde hay poste,
               negro donde no. Salida de masker.generate_mask().

    Returns:
        Imagen PIL con los postes eliminados.

    # TOFIX: Agregar soporte para Inpaint-Anything como backend alternativo
    # cuando settings.inpainting.backend == "inpaint_anything".
    # TOFIX: Agregar medición de tiempo (time.perf_counter()) y retornar
    # inpainting_ms para el response de la API.
    """
    lama = _get_lama()

    logger.info(
        f"Ejecutando inpainting — imagen: {image.size}, "
        f"máscara: {mask.size}, "
        f"backend: {settings.inpainting.backend}"
    )

    # SimpleLama espera imagen RGB y máscara L (escala de grises)
    image_rgb = image.convert("RGB")
    mask_l    = mask.convert("L")

    result = lama(image_rgb, mask_l)

    logger.info("Inpainting completado.")
    return result


def save_inpainted(image: Image.Image, source_path: str | Path) -> Path:
    """
    Guarda la imagen inpainted en results/inpainted/.

    Args:
        image:       Imagen PIL resultado del inpainting.
        source_path: Ruta de la imagen original — usada para el nombre.

    Returns:
        Ruta donde se guardó la imagen inpainted.
    """
    source_path   = Path(source_path)
    inpainted_dir = settings.paths.inpainted_dir
    inpainted_dir.mkdir(parents=True, exist_ok=True)

    out_path = inpainted_dir / f"{source_path.stem}_inpainted.jpg"
    image.save(str(out_path))
    logger.info(f"Imagen inpainted guardada: {out_path}")
    return out_path