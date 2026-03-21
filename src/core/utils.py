# src/utils.py

import logging
import zipfile
from pathlib import Path

from src.config import settings

# TOFIX: Agregar visualize_detections() migrado desde plot_detections()
# en detector.py cuando se haga el refactor de visualización.
# TOFIX: Agregar visualize_mask() y save_result() cuando se implemente
# masker.py e inpainter.py.

logger = logging.getLogger(__name__)


def unzip_dataset() -> None:
    """
    Descomprime el dataset de imágenes si aún no ha sido extraído.

    # TOFIX: Agregar parámetro opcional zip_path para permitir
    # descomprimir datasets alternativos sin modificar config.
    # TOFIX: Agregar validación de integridad del zip antes de extraer.
    """
    zip_path   = settings.paths.data_zip
    extract_to = settings.paths.images_dir

    if not zip_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo zip en: {zip_path}"
        )

    train_dir = extract_to / "train"
    if train_dir.exists():
        logger.info("Dataset ya extraído — omitiendo descompresión.")
        return

    logger.info(f"Extrayendo dataset desde {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)
    logger.info("Dataset extraído correctamente.")


def ensure_dirs() -> None:
    """
    Crea los directorios de salida requeridos si no existen.

    # TOFIX: Ampliar con results/masks/, results/inpainted/,
    # results/detections/ cuando se implementen masker.py e inpainter.py.
    """
    dirs = [
        settings.paths.models_dir,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directorio verificado: {d}")