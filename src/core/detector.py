# src/core/detectoy.py

"""
    Tomado de taller 2 inference.py
"""

# src/core/detector.py

import logging
from pathlib import Path

from PIL import Image
from ultralytics import YOLO
import supervision as sv

from src.config import settings

# TOFIX: Migrar a estructura de clase HousePolesDetector cuando se haga
# el refactor de arquitectura. Ver audit en el chat.
logger = logging.getLogger(__name__)


def load_model():
    # TOFIX: Implementar caché del modelo para evitar recarga en cada llamada.
    # Crítico antes de conectar con la API.
    logger.info(f"Cargando modelo desde {settings.paths.detector_model}")
    model = YOLO(str(settings.paths.detector_model))
    return model


def load_image(image_path: str | Path) -> Image.Image | None:
    # TOFIX: Reemplazar retorno None por excepción explícita (FileNotFoundError
    # o ValueError) para evitar propagación silenciosa del error.
    # TOFIX: Agregar validación de formato de archivo (jpeg, png, webp).
    try:
        image = Image.open(image_path)
        return image
    except Exception as e:
        logger.error(f"Error al abrir la imagen: {e}")
        return None


def detect(image: Image.Image, model=None):
    # TOFIX: Renombrar a detect() cuando se elimine la referencia a detect_house()
    # en los scripts que aún la usen — ya renombrado aquí para Taller 3
    # dado que ahora detectamos dos clases: fachada y poste.
    # TOFIX: Agregar medición de tiempo (time.perf_counter()) y retornar
    # inference_ms junto con results cuando se integre pipeline.py.
    if model is None:
        model = load_model()
    results = model.predict(
        source=image,
        **settings.inference.to_predict_args()
    )
    return results


def filter_by_class(results, class_name: str) -> list:
    """
    Filtra detecciones por nombre de clase.
    Retorna lista de bounding boxes xyxy para la clase solicitada.

    # TOFIX: Eliminar este método cuando se implemente schemas.py —
    # será reemplazado por detection.poles o detection.by_class().
    # TOFIX: Agregar manejo explícito si class_name no existe en
    # settings.classes — actualmente lanza ValueError sin contexto útil.
    """
    class_id = settings.classes.index(class_name)
    boxes = results[0].boxes

    filtered = [
        boxes.xyxy[i].tolist()
        for i, cls in enumerate(boxes.cls.tolist())
        if int(cls) == class_id
    ]

    logger.info(f"Clase '{class_name}': {len(filtered)} detecciones.")
    return filtered


def plot_detections(image: Image.Image, results, out_path: str | Path = None):
    """
    Anota las detecciones en la imagen y opcionalmente la guarda.

    # TOFIX: Mover esta función a src/utils.py como visualize_detections()
    # en el próximo refactor. El detector no debería saber cómo dibujar cajas.
    # TOFIX: Eliminar sv.plot_image() — abre ventana GUI, falla en entornos
    # sin cabeza (Docker, CI, servidores). Solo conservar el guardado a disco.
    """
    box_annotator   = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK)

    annotated_image = image.copy()
    for result in results:
        detections = sv.Detections.from_ultralytics(result)
        labels = [
            f"{result.names[int(class_id)]} {confidence:.2f}"
            for class_id, confidence
            in zip(detections.class_id, detections.confidence)
        ]
        annotated_image = box_annotator.annotate(
            annotated_image,
            detections=detections
        )
        annotated_image = label_annotator.annotate(
            annotated_image,
            detections=detections,
            labels=labels
        )

    # TOFIX: Eliminar esta línea antes de conectar con API o Docker.
    sv.plot_image(annotated_image)

    if out_path:
        out_path = Path(out_path)
        annotated_image.save(out_path)
        logger.info(f"Imagen anotada guardada en '{out_path}'")

    return annotated_image


def infer(image_path: str | Path, model=None, out_path: str | Path = None):
    """
    Orquesta carga, detección y anotación sobre una imagen.

    # TOFIX: Esta función desaparecerá cuando pipeline.py esté completo.
    # Su rol de orquestación pasa a src/core/pipeline.py.
    # TOFIX: La lógica de output path pertenece a pipeline.py o utils.py,
    # no al detector.
    """
    image = load_image(image_path)
    if image is None:
        return None

    image_path = Path(image_path)

    if out_path is None:
        save_path = Path.cwd() / image_path.name
    else:
        out_path = Path(out_path)
        save_path = out_path / image_path.name if out_path.is_dir() else out_path

    results = detect(image, model)
    annotated_image = plot_detections(image, results, save_path)
    return annotated_image


# TOFIX: Mover el bloque CLI a scripts/run_pipeline.py cuando pipeline.py
# esté implementado. El módulo core no debería tener punto de entrada CLI.
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Detección de fachadas y postes con YOLO"
    )
    parser.add_argument(
        "image",
        type=str,
        help="Ruta a la imagen de entrada"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Directorio o archivo de salida"
    )
    args = parser.parse_args()

    model = load_model()
    infer(
        image_path=args.image,
        model=model,
        out_path=args.output
    )