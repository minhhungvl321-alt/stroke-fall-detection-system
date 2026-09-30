from pathlib import Path

ROOT = Path(__file__).resolve().parent

APP_DIR = ROOT / "app"
UTILS_DIR = ROOT / "utils"
MODELS_DIR = ROOT / "models"
DATASET_DIR = ROOT / "dataset"
OUTPUT_DIR = ROOT / "output"
LOGS_DIR = ROOT / "logs"

for directory in [APP_DIR, UTILS_DIR, MODELS_DIR, DATASET_DIR, OUTPUT_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)
