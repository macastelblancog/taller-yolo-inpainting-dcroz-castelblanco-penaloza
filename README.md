
# Detección de Fachadas y Eliminación de Postes
### YOLOv8 + LaMa Inpainting — Proyecto de Maestría

## Autores: D. Croz · Castelblanco · Peñaloza

## Resumen

Este proyecto combina detección de objetos con inpainting para resolver un problema
de análisis urbano: los postes de servicios públicos (energía, telecomunicaciones)
suelen obstruir la visualización de fachadas en imágenes a nivel de calle.
 
El sistema realiza dos tareas de forma automática:
1. **Detecta** fachadas y postes usando YOLOv8 con fine-tuning
2. **Elimina** los postes reconstruyendo el fondo con LaMa inpainting
 
---
 
## Pipeline
 
```
Imagen urbana
      │
      ▼
[detector.py]    →  YOLOv8 detecta fachadas y postes (bounding boxes)
      │
      ▼
[masker.py]      →  filtra postes y genera máscara binaria
      │                blanco (255) = zona a reconstruir
      ▼                negro  (0)   = zona a conservar
[inpainter.py]   →  LaMa reconstruye la imagen sin postes
      │
      ▼
Imagen limpia guardada en results/inpainting/
```
 
---
 
## Estructura del repositorio
 
taller-yolo-casas-dcroz-castelblanco-penaloza/

```
├── examples/ #Ejemplos de eliminación (poste)
│   ├── figure_1.png
│   ├──
│   ├──
│
├── images/ #Imagenes para entrenamiento
│   ├── config
│   ├── test
│   ├── train 
│   ├── valid
│   ├── README.roboflow.txt
│   ├── data.yaml
│   ├── tem_data.yaml
│
├── models/ #Modelos de entrenamiento
│   ├── detector.pt
│   ├── yolo8n.pt
│
├── results #Resultados del modelo
│   ├── detections
│   ├── inpainted
│   ├── mask
│
├── runs/train/house_poles_v1/ #Pesos-metricas
│   ├── weights
│   ├── BoxF1_curve.png
│   ├── BoxPR_curve.png
│   ├── BoxP_curve.png
│   ├── BoxR_curve.png
│   ├── args.yaml
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── labels.jpg
│   ├── results.csv
│   ├── results.png
│   ├── train_batch0.jpg
│   ├── train_batch1.jpg
│   ├── train_batch2.jpg
│   ├── train_batch280.jpg
│   ├── train_batch281.jpg
│   ├── train_batch282.jpg
│   ├── val_batch0_labels.jpg
│   ├── val_batch0_pred.jpg
│
├── src/
│   ├── api  # API
│   ├── core
│   ├── training
│
├── .gitignore
│
├── README.md   #Documentación del proyecto
│
├── pyproject.toml
│
├── requirements.txt #Dependencias
│
├── requirements_full.txt #Dependencias fijas

 ```
 
 
---
 
## Stack tecnológico
 
| Componente           | Librería                          |
|----------------------|-----------------------------------|
| Detección            | `ultralytics` (YOLOv8)            |
| Inpainting           | `simple_lama_inpainting` (LaMa)   |
| Visión por computador| `Pillow`, `numpy`, `albumentations`|
| Visualización        | `supervision`                     |
| API REST             | `FastAPI` + `Uvicorn`             |
| Configuración        | `pydantic` + `pydantic-settings`  |
| Python               | `>=3.10, <3.13`                   |
 
---
## Construcción de la herramienta

### Datos de entrenamiento

Se obtuvieron las imágenes del siguiente repositorio, con 162 imagenes de fachadas con postes.

En las imágenes se hizo el etiquetado de las catergorías fachada y poste usando Roboflow, las cuales se pueden consultar en este proyecto. Los datos se dividieron en 109 entradas para entrenamiento, 29 para validación y 24 para pruebas.

## Instalación
 
### 1. Clonar el repositorio
 
```bash
git clone https://github.com/macastelblancog/taller-yolo-inpainting-dcroz-castelblanco-penaloza
cd taller-yolo-inpainting-dcroz-castelblanco-penaloza
```
 
### 2. Crear y activar entorno virtual
 
```bash
python -m venv .venv
 
# Windows
.\\.venv\\Scripts\\activate
 
# Linux / Mac
source .venv/bin/activate
```
 
### 3. Instalar dependencias
 
```bash
# Dependencias principales
pip install -r requirements.txt
 
# Instalar el paquete en modo editable
# (obligatorio para que funcionen los imports from src.core import ...)
pip install -e .
 
# (opcional) dependencias fijadas para mayor reproducibilidad
pip install -r requirements_full.txt
```
 
> **Nota:** `pyproject.toml` define el paquete `yolo-inpainting` con layout `src/`.
> Sin el `pip install -e .` los imports `from src.core import ...` fallarán con
> `ModuleNotFoundError: No module named 'src'`.

### 4. Entrenar el modelo

Para entrenar el modelo ejecutar:

python src/training/train.py

Este script entrena el modelo YOLO utilizando el dataset definido en data.yaml y el modelo LAMA para la generación de mascaras y la eliminación de postes.

Los pesos generados durante el entrenamiento se almacenan en la carpeta:

runs/train/house_poles_v1


### 5. Evaluar el modelo
 
Para evaluar el desempeño del modelo y calcular métricas como matrices de confusión, curvas, etc, ejecutar:

python src/core/pipeline.py

Esto analiza las predicciones del modelo sobre el conjunto de validación y genera métricas de desempeño.

Si bien en la carpeta de las corridas del modelo tenemos ejemplos de las clasificaciones, y todas las métricas de evaluación y seguimiento de la entrenamiento epoca por epoca, se hace la clasificación de las imágenes según la matriz de confusión para tener la totalidad de ejemplo de FP y FN.

 
### Uso del modelo
 
### Pipeline completo sobre una imagen
 
```bash
python -m src.core.pipeline "images/test/images/mi_foto.jpg"
```
 
### Arrancar la API REST
 
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```
 
**Ejemplos con curl:**
 
```bash
# Detecciones JSON
curl -F "archivo=@ruta/imagen.jpg" http://127.0.0.1:8000/predict
 
# Imagen anotada
curl -F "archivo=@ruta/imagen.jpg" http://127.0.0.1:8000/predict/image -o anotada.jpg
```
 
### Entrenar el modelo
 
```bash
python -m src.training.train
```
 
Los pesos se guardan en `runs/train/<experimento>/weights/best.pt` y se copian
automáticamente a `models/detector.pt`.
 
### Usar como librería Python
 
```python
from src.core import pipeline
 
result = pipeline.run("images/test/images/mi_foto.jpg")
result.show()  # abre la imagen sin postes en el visor
```
 
---
 
## Dataset
 
- **Fuente:** [Roboflow](https://roboflow.com/) — formato YOLOv8
- **Total:** 109 imágenes de entornos urbanos colombianos
- **Clases:** `fachada`, `poste`
- **Etiquetado:** bounding boxes
 
| Split         | Carpeta          |
|---------------|------------------|
| Entrenamiento | `images/train/`  |
| Validación    | `images/valid/`  |
| Prueba        | `images/test/`   |
 
Cada carpeta contiene subcarpetas `images/` y `labels/`.
La configuración completa está en `images/data.yaml`.
 
---
 
## Descripción de módulos
 
### `src/core/detector.py`
Encapsula toda la interacción con YOLOv8. Funciones principales:
- `load_image()` — carga imagen desde disco como PIL
- `detect()` — ejecuta inferencia y retorna resultados + tiempo
- `filter_by_class(results, "poste")` — filtra bounding boxes por clase
- `plot_detections()` — dibuja bounding boxes y guarda imagen anotada
 
### `src/core/masker.py`
Convierte los bounding boxes de postes en una máscara binaria PIL:
- `generate_mask(pole_boxes, width, height)` — crea la máscara con dilatación morfológica
- `save_mask(mask, image_path)` — guarda con convención `<nombre>_mask001.png` que LaMa requiere
 
### `src/core/inpainter.py`
Wrapper de `SimpleLama`:
- `load_inpainter()` — carga LaMa una sola vez (patrón singleton)
- `run_inpainting(image, mask)` — recibe PIL image + PIL mask y devuelve imagen restaurada
 
### `src/core/pipeline.py`
Orquesta los tres módulos en secuencia:
```
detector.detect() → masker.generate_mask() → inpainter.run_inpainting()
```
 
### `src/api/`
Expone el pipeline como API REST. El modelo se carga al arrancar el servidor
y se reutiliza en cada petición.
 
---
 
## 📈 Métricas del experimento
 
Resultados registrados en `runs/train/house_poles_v1/`:
 
| Métrica      | Descripción                                         |
|--------------|-----------------------------------------------------|
| Precision    | % de detecciones correctas sobre el total           |
| Recall       | % de objetos reales detectados                      |
| mAP@0.5      | Precisión promedio con IoU ≥ 0.5                    |
 
---
 
## Reproducir el entenamiento
 
```bash
# 1. Clonar e instalar
git clone https://github.com/macastelblancog/taller-yolo-inpainting-dcroz-castelblanco-penaloza
cd taller-yolo-inpainting-dcroz-castelblanco-penaloza
pip install -r requirements.txt && pip install -e .
 
# 2. Verificar estructura del dataset
#    images/train/, images/valid/, images/test/
#    con subcarpetas images/ y labels/
 
# 3. Entrenar
python -m src.training.train
 
# 4. Ejecutar pipeline sobre imagen de prueba
python -m src.core.pipeline "images/test/images/mi_foto.jpg"
 
# 5. Ver resultados
#    results/detections/  → imágenes con bounding boxes
#    results/inpainting/  → imágenes sin postes
```
 
---

## Resultados (métricas) y ejemplos de detección

## Limitaciones y pasos futuros recomendados

### Limitaciones del modelo

A pesar de los resultados obtenidos, el modelo presenta algunas limitaciones:

#### 1. Tamaño reducido del dataset

El modelo fue entrenado con un conjunto de aproximadamente 162 imágenes, lo cual es un tamaño limitado para entrenar modelos de detección y eliminación de objetos robustos.

Esto puede provocar:

sobreajuste (overfitting)
baja capacidad de generalización a nuevas imágenes.

#### 2. Variabilidad limitada en las escenas
Las imágenes del dataset no cubren completamente todas las variaciones posibles de:

##### a. arquitectura
##### b. iluminación
##### c. ángulos de cámara
##### d. contextos urbanos y rurales.

#### 3. Confusión con estructuras similares

El modelo puede confundir postes con:

Columnas
Monumentos verticales
Estructuras alargadas verticales
Reflejos en agua del mismo poste

#### 4. Problemas para eliminar postes

En imágenes donde el poste aparece muy grande, el modelo presenta dificultades para eliminar correctamente el objeto.

### Pasos futuros recomendados

Para mejorar el desempeño del modelo se recomiendan las siguientes acciones:

#### 1. Ampliar y seleccionar el dataset correcto

Recolectar y etiquetar más imágenes de fachadas y postes en diferentes contextos:

##### a. urbano
##### b. rural
###### c. diferentes regiones 
###### d. diferentes condiciones de iluminación.
##### e. Idealmente aumentar el dataset a 200–500 imágenes.

Además, se recomienda escoger imagenes con postes pequeños ya que el modelo tiene un mejor desempeño con este tipo de imagenes.

#### 2. Aplicar técnicas de aumento de datos

Utilizar técnicas de data augmentation como:

##### a. rotaciones
##### b. cambios de brillo y contraste
##### c. escalamiento
##### d. transformaciones geométricas.

#### 3. Ajuste de hiperparámetros

Realizar experimentación con diferentes valores de:

##### a. número de épocas
##### b. tamaño de imagen
##### c. learning rate
##### d. batch size.


 