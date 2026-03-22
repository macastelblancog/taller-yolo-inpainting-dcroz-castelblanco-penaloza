# src/training/train.py

import logging
from pathlib import Path

import albumentations as A
from ultralytics import YOLO

from src.core.config import settings
from src import utils

# TOFIX: Implementar integración correcta de albumentations mediante
# callback o dataset wrapper de Ultralytics — el parámetro augmentations=
# no existe en model.train() y era ignorado silenciosamente en Taller 1.

logger = logging.getLogger(__name__)

# Transformaciones de aumento de datos
# TOFIX: Mover a src/training/augmentation.py si el pipeline de
# augmentación crece con más transformaciones.
AUGMENTATION_TRANSFORMS = [
    A.Blur(blur_limit=7, p=0.5),
    A.CLAHE(clip_limit=4.0, p=0.5),
]


def train(data_yaml: Path = None) -> object:
    """
    Ejecuta el entrenamiento del modelo YOLO.
    Todos los hiperparámetros se leen desde settings.training.

    Args:
        data_yaml: Ruta al archivo data.yaml. Si None usa settings.paths.data_yaml.

    Returns:
        Objeto results de Ultralytics con métricas del entrenamiento.
    """
    data_yaml = data_yaml or settings.paths.data_yaml

    logger.info(f"Cargando modelo base desde {settings.paths.base_model}")
    model = YOLO(str(settings.paths.base_model))

    train_args = settings.training.to_train_args()
    logger.info(f"Iniciando entrenamiento — {train_args}")

    results = model.train(
        data = str(data_yaml),
        project = str(settings.paths.runs_dir),
        exist_ok = True,
        verbose = False,
        **train_args,
    )

    # Copiar best.pt a la ruta estable que usa inference
    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if best_weights.exists():
        target = settings.paths.detector_model
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(best_weights.read_bytes())
        logger.info(f"Mejores pesos guardados en {target}")
    else:
        logger.warning("No se encontró best.pt — revisar directorio de entrenamiento.")

    return results


def train_model(data_yaml: Path = None) -> object:
    """
    Entrypoint público — prepara el entorno y ejecuta el entrenamiento.

    Args:
        data_yaml: Ruta al archivo data.yaml. Si None usa settings.paths.data_yaml.

    Returns:
        Objeto results de Ultralytics.
    """
    logger.info("Preparando entorno de entrenamiento...")
    utils.unzip_dataset()
    utils.ensure_dirs()
    logger.info("Entorno listo. Iniciando entrenamiento.")
    results = train(data_yaml)
    return results


if __name__ == "__main__":
    train_model()