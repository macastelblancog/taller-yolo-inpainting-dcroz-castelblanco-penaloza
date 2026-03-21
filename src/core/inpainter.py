# src/core/inpainter.py

import logging
from pathlib import Path

from src.config import settings

# TOFIX: Importar dependencias de LaMa cuando se implemente run_inpainting().
# TOFIX: Importar dependencias de Inpaint-Anything como alternativa al backend.

logger = logging.getLogger(__name__)

# TODO: Funcionalidad de run_inpainting() para eliminar postes usando
#       el backend configurado en settings.inpainting.backend ("lama" | "inpaint_anything")
# TODO: Funcionalidad de load_inpainter() para cargar el modelo de inpainting
#       desde settings.paths.lama_model