# src/api/routes.py

import logging
import tempfile
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from src.core import pipeline

# TOFIX: Importar settings cuando /health esté implementado.
# TOFIX: Mover serialize_results a schemas.py cuando se implemente.
# TOFIX: Agregar endpoint GET /health.

logger = logging.getLogger(__name__)

router = APIRouter()

FORMATOS_PERMITIDOS = {
    "image/jpeg": ".jpg",
    "image/png":  ".png",
    "image/bmp":  ".bmp",
    "image/webp": ".webp",
}


def validar_archivo(archivo: UploadFile) -> str:
    if archivo.content_type not in FORMATOS_PERMITIDOS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Formato no soportado: {archivo.content_type}. "
                f"Permitidos: {sorted(FORMATOS_PERMITIDOS)}"
            ),
        )
    return FORMATOS_PERMITIDOS[archivo.content_type]


def serialize_results(results) -> list[dict]:
    # TOFIX: Mover a schemas.py como método de DetectionResult.summary()
    # cuando se implemente ese módulo.
    detections_out = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for xyxy, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            cls_id = int(cls_id.item())
            conf   = float(conf.item())
            coords = [float(v) for v in xyxy.tolist()]
            detections_out.append({
                "class_id":   cls_id,
                "class_name": result.names[cls_id],
                "confidence": round(conf, 4),
                "bbox_xyxy":  [round(v, 2) for v in coords],
            })
    return detections_out


# Endpoint raíz — renombrado de raiz() a root() para consistencia con inglés.
# La respuesta mantiene los textos originales en español.
@router.get("/")
def root():
    return {
        "api": "House Detection API",
        "endpoints": {
            "POST /predict":       "Return JSON detections",
            "POST /predict/image": "Return annotated image",
        },
    }


# TOFIX: Renombrar /predict a /process cuando masker e inpainter
# estén integrados en el pipeline completo.
@router.post("/predict")
async def predict(archivo: UploadFile = File(...)):
    extension = validar_archivo(archivo)
    logger.info(f"POST /predict — archivo: {archivo.filename}")

    try:
        contents = await archivo.read()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir    = Path(tmp_dir)
            input_path = tmp_dir / f"input{extension}"
            input_path.write_bytes(contents)

            # TOFIX: Cuando pipeline.run() retorne dataclass, reemplazar
            # llamada directa al detector por result.detection.
            from src.core.detector import detect, load_image, load_model
            image            = load_image(input_path)
            model            = load_model()
            results, _       = detect(image, model)
            detections       = serialize_results(results)

            logger.info(f"Detecciones: {len(detections)} — {archivo.filename}")

            return JSONResponse(content={
                "filename":         archivo.filename,
                "content_type":     archivo.content_type,
                "total_detections": len(detections),
                "detections":       detections,
            })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en /predict: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error de inferencia: {e}"
        ) from e


@router.post("/predict/image")
async def predict_image(archivo: UploadFile = File(...)):
    extension = validar_archivo(archivo)
    logger.info(f"POST /predict/image — archivo: {archivo.filename}")

    try:
        contents = await archivo.read()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir     = Path(tmp_dir)
            input_path  = tmp_dir / f"input{extension}"
            output_path = tmp_dir / f"pred_{Path(archivo.filename).name}"

            input_path.write_bytes(contents)

            annotated = pipeline.run(
                image_path=input_path,
                out_path=output_path,
            )

            if annotated is None:
                raise HTTPException(
                    status_code=500,
                    detail="La inferencia retornó None."
                )

            if not output_path.exists():
                raise HTTPException(
                    status_code=500,
                    detail="La imagen anotada no fue creada."
                )

            image_bytes = output_path.read_bytes()
            logger.info(f"Imagen anotada lista: {output_path.name}")

            return StreamingResponse(
                BytesIO(image_bytes),
                media_type="image/jpeg",
                headers={
                    "Content-Disposition":
                        f'attachment; filename="{output_path.name}"'
                },
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en /predict/image: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error de inferencia: {e}"
        ) from e