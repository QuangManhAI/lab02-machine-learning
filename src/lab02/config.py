"""Project paths and constants."""

from pathlib import Path


def _find_project_root() -> Path:
    """Find the repository root from the current working directory."""
    candidates = [Path.cwd(), *Path.cwd().parents]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "lab02").is_dir():
            return candidate
    return Path.cwd()


PROJECT_ROOT = _find_project_root()
DATASETS_DIR = PROJECT_ROOT / "datasets"
HOUSING_DIR = DATASETS_DIR / "housing"
IMAGES_DIR = PROJECT_ROOT / "images" / "end_to_end_project"
MODELS_DIR = PROJECT_ROOT / "models"

HOUSING_URL = "https://github.com/ageron/data/raw/main/housing.tgz"
RANDOM_STATE = 42
TEST_SIZE = 0.2

for path in (DATASETS_DIR, IMAGES_DIR, MODELS_DIR):
    path.mkdir(parents=True, exist_ok=True)
