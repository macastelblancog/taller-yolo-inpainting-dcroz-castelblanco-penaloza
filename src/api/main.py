# src/api/main.py

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import router
from src.core.detector import load_model

# TOFIX: Agregar middleware de logging de requests cuando se implemente
# observabilidad — tiempo de respuesta por endpoint.

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Reemplaza @app.on_event("startup") — deprecado desde FastAPI 0.93.
    # El modelo se carga una vez aquí y queda disponible en app.state.model
    # para todas las rutas durante el ciclo de vida de la aplicación.
    logger.info("Iniciando aplicación — cargando modelo...")
    app.state.model = load_model()
    logger.info("Modelo cargado. Aplicación lista.")
    yield
    # TOFIX: Agregar limpieza de recursos aquí si se implementa
    # caché de GPU o conexiones externas.
    logger.info("Aplicación detenida.")


app = FastAPI(
    title       = "House Detection API",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.include_router(router)