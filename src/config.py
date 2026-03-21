# src/core/config.py
"""
    Esto mejora el manejo de variables globales y paths que estabamos 
    dando en el taller 2 mediante el utils y las declaraciones al inicio
    de los diferntes .py
"""

from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ── Sub-models ────────────────────────────────────────────────────────────────

class ClassSettings(BaseModel):
    """Class names must match data.yaml order exactly."""
    names: list[str] = Field(default_factory=lambda: ["fachada", "poste"])

    def index(self, name: str) -> int:
        """Return class id by name. Raises ValueError if not found."""
        return self.names.index(name)


class PathSettings(BaseModel):
    """All directory and file paths anchored to repo root."""
    detector_model: Path = REPO_ROOT / "models/detector.pt"
    base_model:     Path = REPO_ROOT / "models/yolov8n.pt"
    lama_model:     Path = REPO_ROOT / "models/lama/big-lama"
    data_yaml:      Path = REPO_ROOT / "images/config/data.yaml"
    images_dir:     Path = REPO_ROOT / "images"
    results_dir:    Path = REPO_ROOT / "results"
    runs_dir:       Path = REPO_ROOT / "runs/train"


class InferenceSettings(BaseModel):
    """Parameters used at inference time.
    to_predict_args() maps to model.predict() keyword arguments."""
    conf_threshold: float = 0.25
    iou_threshold:  float = 0.45
    image_size:     int   = 640
    # device:         str   = "cpu"    # "cpu" | "cuda" | "mps"

    def to_predict_args(self) -> dict:
        """Returns kwargs ready to unpack into model.predict()."""
        return {
            "conf":   self.conf_threshold,
            "iou":    self.iou_threshold,
            "device": self.device,
            "imgsz":  self.image_size,
        }


class MaskSettings(BaseModel):
    """Controls how pole bounding boxes are converted to inpainting masks."""
    dilation_px: int = 12    # pixels added around each pole bbox
    fill_value:  int = 255   # white mask on black canvas


class InpaintingSettings(BaseModel):
    """Inpainting backend selection and its parameters."""
    backend: str = "lama"    # "lama" | "inpaint_anything"


class TrainingSettings(BaseModel):
    """Hyperparameters for the training run.
    to_train_args() maps to model.train() keyword arguments."""
    epochs:     int = 50
    image_size: int = 640
    batch_size: int = 16
    seed:       int = 42
    run_name:   str = "house_poles_v1"

    def to_train_args(self) -> dict:
        """Returns kwargs ready to unpack into model.train()."""
        return {
            "epochs":  self.epochs,
            "imgsz":   self.image_size,
            "batch":   self.batch_size,
            "seed":    self.seed,
            "name":    self.run_name,
        }


# ── Top-level Settings ────────────────────────────────────────────────────────

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    classes:    ClassSettings      = Field(default_factory=ClassSettings)
    paths:      PathSettings       = Field(default_factory=PathSettings)
    inference:  InferenceSettings  = Field(default_factory=InferenceSettings)
    mask:       MaskSettings       = Field(default_factory=MaskSettings)
    inpainting: InpaintingSettings = Field(default_factory=InpaintingSettings)
    training:   TrainingSettings   = Field(default_factory=TrainingSettings)


# ── Singleton ─────────────────────────────────────────────────────────────────

settings = Settings()