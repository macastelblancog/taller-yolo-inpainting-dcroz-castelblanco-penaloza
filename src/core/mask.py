# src/core/masker.py

import logging
from pathlib import Path

from src.config import settings

# TOFIX: Importar numpy y PIL cuando se implemente generate_mask().
# TOFIX: Importar visualize_mask() desde src.utils cuando se agregue.

logger = logging.getLogger(__name__)

# TODO: Funcionalidad de generarte_mak a partir de bounding boxes de postes
# TODO: Funcionalidad de save_mask para guarder resultados