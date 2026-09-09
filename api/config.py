"""Application settings."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "AI Visual Quality Inspector"
    api_prefix: str = "/api"
    database_url: str = f"sqlite:///{ROOT / 'storage' / 'qc.db'}"
    storage_dir: Path = ROOT / "storage"
    images_dir: Path = ROOT / "storage" / "images"
    defective_dir: Path = ROOT / "storage" / "defective-images"
    heatmaps_dir: Path = ROOT / "storage" / "heatmaps"
    weights_dir: Path = ROOT / "weights"

    yolo_weights: Path = ROOT / "weights" / "yolo_gc10.pt"
    # Prefer MSDD weights for casting-part demos when present
    yolo_msdd_weights: Path = ROOT / "weights" / "yolo_msdd.pt"
    # Prefer KolektorSDD2 anomaly weights when present (faster public download)
    anomaly_weights: Path = ROOT / "weights" / "patchcore_kolektor2"
    anomaly_meta: Path = ROOT / "weights" / "patchcore_kolektor2" / "meta.json"

    yolo_conf_threshold: float = 0.35
    # Calibrated for PatchCore mean_top_1pct on KolektorSDD2
    anomaly_fail_threshold: float = 0.54
    anomaly_review_threshold: float = 0.49

    allow_demo_fallback: bool = True
    cors_origins: str = "http://localhost:5173,http://localhost:3000,http://localhost:80"

    camera_device_index: int = 0


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    for path in (
        settings.storage_dir,
        settings.images_dir,
        settings.defective_dir,
        settings.heatmaps_dir,
        settings.weights_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return settings
