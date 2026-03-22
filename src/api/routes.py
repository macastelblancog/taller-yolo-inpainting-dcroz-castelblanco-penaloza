# src/api/routes.py

import logging
import tempfile
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from src.core.pipeline import run_detection, run_inpainting

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


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def stream_image(image_path: Path, filename: str) -> StreamingResponse:
    """
    Helper compartido — lee una imagen de disco y la retorna como
    StreamingResponse. Evita duplicar este bloque en cada endpoint.
    """
    image_bytes = image_path.read_bytes()
    return StreamingResponse(
        BytesIO(image_bytes),
        media_type="image/jpeg",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )


# ── Root ──────────────────────────────────────────────────────────────────────

# Endpoint raíz — renombrado de raiz() a root() para consistencia con inglés.
@router.get("/")
def root():
    return {
        "api": "Facade & Pole Detection API",
        "endpoints": {
            "POST /detect":           "JSON con bounding boxes de fachadas y postes",
            "POST /detect/visualize": "Imagen anotada con bounding boxes",
            "POST /inpaint":          "JSON con resultado tras eliminar postes",
            "POST /inpaint/visualize":"Imagen con postes eliminados",
        },
    }


# ── Detection endpoints ───────────────────────────────────────────────────────

@router.post("/detect")
async def detect(archivo: UploadFile = File(...)):
    """Retorna detecciones de fachadas y postes en JSON."""
    extension = validar_archivo(archivo)
    logger.info(f"POST /detect — archivo: {archivo.filename}")

    try:
        contents = await archivo.read()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir    = Path(tmp_dir)
            input_path = tmp_dir / f"input{extension}"
            input_path.write_bytes(contents)

            detection_result = run_detection(image_path=input_path)
            if detection_result is None:
                raise HTTPException(
                    status_code=400,
                    detail="No se pudo procesar la imagen."
                )

            annotated, results, inference_ms = detection_result
            detections = serialize_results(results)

            logger.info(
                f"Detecciones: {len(detections)} — {archivo.filename} "
                f"({inference_ms:.1f}ms)"
            )

            return JSONResponse(content={
                "filename":         archivo.filename,
                "content_type":     archivo.content_type,
                "total_detections": len(detections),
                "inference_ms":     round(inference_ms, 2),
                "detections":       detections,
            })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en /detect: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}") from e


@router.post("/detect/visualize")
async def detect_visualize(archivo: UploadFile = File(...)):
    """Retorna imagen anotada con bounding boxes de fachadas y postes."""
    extension = validar_archivo(archivo)
    logger.info(f"POST /detect/visualize — archivo: {archivo.filename}")

    try:
        contents = await archivo.read()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir     = Path(tmp_dir)
            input_path  = tmp_dir / f"input{extension}"
            output_path = tmp_dir / f"det_{Path(archivo.filename).name}"

            input_path.write_bytes(contents)

            detection_result = run_detection(
                image_path=input_path,
                out_path=output_path,
            )

            if detection_result is None:
                raise HTTPException(
                    status_code=500,
                    detail="La inferencia retornó None."
                )

            annotated, results, inference_ms = detection_result

            if not output_path.exists():
                annotated.save(str(output_path))

            return stream_image(output_path, output_path.name)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en /detect/visualize: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}") from e


# ── Inpainting endpoints ──────────────────────────────────────────────────────

@router.post("/inpaint/visualize")
async def inpaint_visualize(archivo: UploadFile = File(...)):
    """
    Retorna imagen con postes eliminados usando LaMa inpainting.
    """
    extension = validar_archivo(archivo)
    logger.info(f"POST /inpaint/visualize — archivo: {archivo.filename}")

    try:
        contents = await archivo.read()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir    = Path(tmp_dir)
            input_path = tmp_dir / f"input{extension}"
            output_path = tmp_dir / f"inp_{Path(archivo.filename).name}"

            input_path.write_bytes(contents)

            result = run_inpainting(image_path=input_path)

            if result is None:
                raise HTTPException(
                    status_code=500,
                    detail="El pipeline retornó None."
                )

            inpainted, mask_path = result

            # Guardar la imagen inpainted en el directorio temporal
            # para poder streamearla de vuelta al cliente.
            # save_inpainted() ya la guardó en results/inpainted/
            # pero necesitamos una copia en tmp para StreamingResponse.
            inpainted.save(str(output_path), format="JPEG")

            return stream_image(output_path, output_path.name)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en /inpaint/visualize: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}") from e


@router.post("/inpaint")
async def inpaint(archivo: UploadFile = File(...)):
    """
    Retorna JSON con resultado tras detectar y eliminar postes.

    # TOFIX: Cuando inpainter esté implementado, agregar campos:
    #        poles_removed, mask_path, inpainting_ms al response.
    """
    extension = validar_archivo(archivo)
    logger.info(f"POST /inpaint — archivo: {archivo.filename}")

    try:
        contents = await archivo.read()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir    = Path(tmp_dir)
            input_path = tmp_dir / f"input{extension}"
            input_path.write_bytes(contents)

            result = run_inpainting(image_path=input_path)
            if result is None:
                raise HTTPException(
                    status_code=500,
                    detail="El pipeline retornó None."
                )

            annotated, mask_path = result

            return JSONResponse(content={
                "filename":    archivo.filename,
                "mask_path":   str(mask_path),
                "inpainted":   False,  # TOFIX: True cuando inpainter conectado
                "status":      "mask_generated",
            })

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error en /inpaint: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {e}") from e

