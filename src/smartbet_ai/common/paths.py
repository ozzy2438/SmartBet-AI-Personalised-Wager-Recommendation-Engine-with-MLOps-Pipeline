"""Project path helpers that resolve from the repository root."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_CONFIG_PATH = CONFIGS_DIR / "model_config.yaml"
DATA_SCHEMA_PATH = CONFIGS_DIR / "data_schema.json"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"


def ensure_project_dirs() -> None:
    """Create all artifact directories used by the project."""
    for directory in (
        CONFIGS_DIR,
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FEATURES_DIR,
        MODELS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
