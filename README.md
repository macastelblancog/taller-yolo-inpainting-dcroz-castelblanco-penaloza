# YOLO Inpainting — Eliminación de postes en fachadas (Proyecto de Maestría)

## Resumen

Repositorio combina detección de objetos (YOLOv8) con técnicas de inpainting (LaMa) para detectar postes en imágenes de fachadas y retirarlos mediante inpainting. Incluye utilidades de entrenamiento, un pipeline de inferencia, una API ligera con FastAPI y helpers para manejar el dataset.

## Motivación

Los postes y elementos del mobiliario urbano suelen ocluir imágenes de fachadas y afectan tareas de análisis urbano y visualización. Este trabajo persigue (1) detectar fachadas y postes en imágenes a nivel de calle y (2) eliminar los postes mediante inpainting para minimizar su impacto en análisis posteriores. 

## Funcionalidades clave (implementadas / esbozadas)

- Wrappers de detector basados en YOLOv8 para inferencia y entrenamiento (`ultralytics`).
- Arnes de entrenamiento en `src/training/train.py` que ejecuta Ultralytics y copia los mejores pesos para la inferencia.
- Pipeline de inferencia ligero en `src/core/pipeline.py` que actualmente realiza detección y guarda imágenes anotadas.
- Endpoints FastAPI en `src/api`:
	- `POST /predict` — devuelve detecciones en JSON
	- `POST /predict/image` — devuelve la imagen anotada como stream
- Integración de inpainting esbozada usando `simple_lama_inpainting` (`src/core/inpainter.py`).
- Generación de máscaras para inpainting en `src/core/mask.py` (convierte bbox de postes en máscaras binarias).
- Dataset en formato YOLO bajo `images/` con `data.yaml` describiendo clases.


## Estructura del repositorio

- `src/` — paquete principal
	- `api/` — app FastAPI (`main.py`, `routes.py`)
	- `core/` — detector, pipeline, masker, inpainter
	- `training/` — entrenamiento (`train.py`)
	- `config.py` — configuración central con `pydantic`/`pydantic-settings`
- `images/` — dataset (train/valid/test) y `data.yaml`
- `models/` — ubicación esperada para pesos base y pesos de detector
- `results/` — ubicación esperada para salidas (anotaciones, inpainting)
- `pyproject.toml`, `requirements.txt`, `requirements_full.txt` — dependencias

Archivos relevantes:
- `src/api/main.py` — entrypoint FastAPI (exporta `app`)
- `src/api/routes.py` — rutas y manejo de peticiones
- `src/core/detector.py` — carga de modelo, lectura de imagen, detección y visualización
- `src/core/pipeline.py` — orquestación de pipeline (detección → visualización → TODO máscaras/inpainting)
- `src/core/inpainter.py` — wrapper de LaMa (SimpleLama)
- `src/core/mask.py` — generación y guardado de máscaras
- `src/training/train.py` — script de entrenamiento y copia de `best.pt`

## Visión arquitectural

Componentes principales:

- Detector (Ultralytics YOLOv8): carga pesos desde `models/detector.pt` y ejecuta inferencia por imagen.
- Pipeline: orquesta carga de imagen → detección → visualización; preparado para integrar generación de máscaras e inpainting.
- Masker: convierte bounding boxes de postes en una máscara binaria para inpainting.
- Inpainter (LaMa): recibe imagen + máscara y devuelve la imagen restaurada.
- Capa API (FastAPI): expone endpoints para inferencia; el servidor carga el modelo en el arranque (nota: hay TODOs relativos a cache de modelo).

Flujo de datos (actual):

imagen → `pipeline.run()` → `detector.detect()` → `detector.plot_detections()` → imagen anotada guardada en `results` y devuelta al cliente

Flujo deseado (planificado):

imagen → detección → filtrar bbox de postes → `mask.generate_mask()` → `inpainter.run_inpainting(imagen, máscara)` → guardar imagen inpainted

## Stack tecnológico

- Python 3.10–3.12 (según `pyproject.toml`)
- `ultralytics` (YOLOv8)
- `simple_lama_inpainting` (wrapper para LaMa)
- FastAPI + Uvicorn
- Pillow, numpy, albumentations
- pydantic / pydantic-settings para configuración

Dependencias listadas en `requirements.txt` (núcleo) y `requirements_full.txt` (lista fijada / desarrollo).

## Instalación y configuración del entorno

Prerequisitos



## Complemento: integración con el enunciado del taller

He integrado la información del enunciado del taller proporcionado como guía. A continuación se listan los puntos prácticos sugeridos por el enunciado y cómo aplicarlos a este repositorio. Compruebe y adapte según las instrucciones oficiales del curso si hay discrepancias.

- Dataset y anotaciones
	- El dataset usado fue exportado desde Roboflow y contiene 109 imágenes anotadas en formato YOLOv8 (ver `images/` y `images/data.yaml`).
	- Clases: `fachada`, `poste`.
	- Estructura esperada de carpetas: `images/train`, `images/valid`, `images/test` con subcarpetas `images` y `labels` (ya presentes en el repositorio).

- Experimentos y evaluación
	- Entrenamiento: use `python -m src.training.train` para lanzar un experimento que utiliza los hiperparámetros en `src/config.py` (`settings.training`).
	- Métricas: el enunciado sugiere evaluar calidad de detección (mAP, precision/recall) y calidad visual del inpainting. Implementar métricas cuantitativas adicionales (mAP @ IoU 0.5:0.95) es recomendable.
	- Recomendación práctica: guardar los resultados de cada experimento en `runs/train/<name>` (Ultralytics lo hace por defecto) y consolidar métricas en un CSV o notebook para el informe.

- Flujo de reproducibilidad (sugerido)
	1. Crear entorno virtual e instalar dependencias (ver sección Instalación).
	2. Verificar que `images/` está descomprimido y estructurado. Si dispone de `images/data.zip`, usar `src/core/utils.unzip_dataset()` o descomprimir manualmente.
	3. Ejecutar `python -m src.training.train` para entrenar un modelo base.
	4. Ejecutar inferencias de validación con `uvicorn` + llamadas a `/predict` o usando `src/core/detector.infer()` o `pipeline.run()` sobre imágenes de validación.
	5. Evaluar y añadir resultados al informe (notebook o PDF).

- Entregables recomendados para el taller
	- Código fuente (este repositorio con README actualizado).
	- Pesos resultantes (`runs/train/<exp>/weights/best.pt` y copia en `models/detector.pt` si procede).
	- Notebook o script de evaluación que reproduzca métricas y visualizaciones.
	- Informe técnico (PDF) con metodología, arquitectura, métricas y conclusiones.

- Criterios de revisión sugeridos (alineados con el enunciado)
	- Funcionamiento reproducible: pasos claros para instalar y ejecutar los experimentos.
	- Calidad de la detección: mAP y ejemplos visuales representativos.
	- Calidad del inpainting: ejemplos antes/después y discusión de artefactos.
	- Claridad del código y documentación: funciones/archivos comentados y `README.md` completo.

Si lo desea, puedo añadir un `notebook/` de ejemplo que reproduzca un experimento mínimo (entrenamiento rápido + inferencia sobre 5 imágenes) y/o preparar una plantilla de informe en LaTeX o Markdown siguiendo el enunciado del taller.

```bash
# crear y activar virtualenv
python -m venv .venv
.\.venv\Scripts\activate

# instalar dependencias principales
pip install -r requirements.txt

# (opcional) instalar dependencias fijadas/de desarrollo
pip install -r requirements_full.txt
```

Notas

- `pyproject.toml` define la distribución y la disposición del paquete `src`.
- `src/config.py` usa `.env` si existe (a través de `pydantic-settings`); no hay variables obligatorias documentadas en el repo.

## Uso

Arrancar el servidor FastAPI (modo desarrollo):

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Peticiones de ejemplo

Detecciones JSON:

```bash
curl -F "archivo=@/ruta/a/imagen.jpg" http://127.0.0.1:8000/predict
```

Imagen anotada (descarga):

```bash
curl -F "archivo=@/ruta/a/imagen.jpg" http://127.0.0.1:8000/predict/image -o anotada.jpg
```

Entrenamiento (experimento local):

```bash
python -m src.training.train
```

CLI rápido de detector (inferencia por imagen):

```bash
python -m src.core.detector ruta/a/imagen.jpg -o salida.jpg
```

Rutas y pesos

- El proyecto espera pesos base en `models/yolov8n.pt`; tras el entrenamiento se copia el mejor peso a `models/detector.pt` si `best.pt` es generado.
- Dataset en `images/` con `train/`, `valid/`, `test/` y `images/data.yaml`.

## Entradas y salidas

Entradas
- Dataset en formato YOLO bajo `images/` (labels en `train/labels`, `valid/labels`, `test/labels`).
- Imágenes sueltas aceptadas por la API (`jpeg`, `png`, `bmp`, `webp`).

Salidas / artefactos
- Imágenes anotadas en `results/` (salida del pipeline)
- Runs y checkpoints en `runs/train/` (por Ultralytics)
- Pesos finales de detector en `models/detector.pt` si se dispone de `best.pt`

