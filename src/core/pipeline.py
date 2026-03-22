# src/core/pipeline.py

import logging
from pathlib import Path

import supervision as sv

from src.core.detector import load_image, detect, plot_detections, filter_by_class
from src.core.masker import generate_mask, save_mask
from src.core.inpainter import run_inpainting as lama_inpaint, save_inpainted
from src.config import settings

logger = logging.getLogger(__name__)


def run_detection(image_path: str | Path, model=None, out_path: str | Path = None):
    """
    Pipeline de solo detección — encuentra fachadas y postes en la imagen.

    Returns:
        Tuple (annotated_image, results, inference_ms) o None si falla.

    # TOFIX: Reemplazar retorno por dataclass DetectionResult cuando
    #        se implemente schemas.py.
    # TOFIX: Mover plot_detections() a src/utils.py como visualize_detections().
    """
    image = load_image(image_path)
    if image is None:
        logger.error(f"No se pudo cargar la imagen: {image_path}")
        return None

    image_path = Path(image_path)

    if out_path is None:
        save_path = settings.paths.results_dir / "detections" / image_path.name
    else:
        out_path = Path(out_path)
        save_path = out_path / image_path.name if out_path.is_dir() else out_path

    save_path.parent.mkdir(parents=True, exist_ok=True)

    results, inference_ms = detect(image, model)
    logger.info(f"Detección completada en {inference_ms:.1f}ms.")

    annotated = plot_detections(image, results, save_path)

    return annotated, results, inference_ms


def _filter_poles_for_mask(results) -> list:
    """
    Filtra detecciones de postes aplicando un umbral de confianza
    más alto que el de detección general — reduce falsos positivos
    en la máscara de inpainting.

    Returns:
        Lista de bounding boxes xyxy de postes con confianza suficiente.

    # TOFIX: Mover MASK_CONF_THRESHOLD a settings.mask.conf_threshold
    # cuando se confirme el valor óptimo experimentalmente.
    """
    pole_class_id = settings.classes.index("poste")
    boxes         = results[0].boxes
    threshold     = settings.mask.conf_threshold

    filtered = []
    for i, (cls, conf) in enumerate(
        zip(boxes.cls.tolist(), boxes.conf.tolist())
    ):
        if int(cls) == pole_class_id and float(conf) >= threshold:
            filtered.append(boxes.xyxy[i].tolist())

    logger.info(
        f"Postes para máscara: {len(filtered)} "
        f"(umbral confianza: {threshold})"
    )
    return filtered


def run_inpainting(image_path: str | Path, model=None, out_path: str | Path = None):
    """
    Pipeline completo — detección + máscara + inpainting.
    Muestra dos ventanas emergentes: detecciones e imagen inpainted.

    Returns:
        Tuple (inpainted_image, mask_path) o None si falla.

    # TOFIX: Reemplazar retorno por dataclass PipelineResult cuando
    #        se implemente schemas.py.
    # TOFIX: Eliminar sv.plot_image() antes de desplegar en servidor
    #        o Docker — falla en entornos sin cabeza.
    """

    # ── Paso 1: Detección ─────────────────────────────────────────────────────
    detection_result = run_detection(image_path, model)
    if detection_result is None:
        return None

    annotated, results, inference_ms = detection_result

    # ── Paso 2: Cargar imagen original limpia ─────────────────────────────────
    image = load_image(image_path)

    # ── Paso 3: Filtrar postes con umbral de confianza + generar máscara ──────
    pole_boxes = _filter_poles_for_mask(results)

    if len(pole_boxes) == 0:
        logger.warning(
            f"Ningún poste superó el umbral de confianza "
            f"({settings.mask.conf_threshold}). "
            "Retornando imagen original sin inpainting."
        )
        return image, None

    mask = generate_mask(
        pole_boxes  = pole_boxes
        ,width      = image.width
        ,height     = image.height,
    )

    mask_path = save_mask(mask, image_path)
    logger.info(f"Máscara guardada en: {mask_path}")

    # ── Paso 4: Inpainting ────────────────────────────────────────────────────
    inpainted = lama_inpaint(image, mask)
    inpainted_path = save_inpainted(inpainted, image_path)
    logger.info(f"Imagen inpainted guardada en: {inpainted_path}")

    # ── Ventana 2: Resultado inpainted ────────────────────────────────────────
    # TOFIX: Eliminar sv.plot_image() antes de desplegar en servidor o Docker.
    sv.plot_image(inpainted)

    return inpainted, mask_path