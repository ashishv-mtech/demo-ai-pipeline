import sys
import subprocess
from pathlib import Path
import pandas as pd

# Project Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MICRO_RAW_PATH = BASE_DIR / "dataset" / "pytest-micro-data" / "raw" / "heart_disease_risk_2026_300.csv"
MICRO_PROCESSED_PATH = BASE_DIR / "dataset" / "pytest-micro-data" / "processed" / "ml_processed_data_300.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest_classifier.joblib"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.joblib"
PREPROCESS_SCRIPT = BASE_DIR / "scripts" / "data-preprocess.py"
TRAINING_SCRIPT = BASE_DIR / "scripts" / "training.py"


def test_1_load_micro_dataset():
    """Verify raw dataset in dataset/pytest-micro-data/raw/ loads properly."""
    assert MICRO_RAW_PATH.exists(), f"Micro raw dataset not found at {MICRO_RAW_PATH}"
    df = pd.read_csv(MICRO_RAW_PATH)
    assert not df.empty, "Micro raw dataset is empty"
    assert "has_heart_disease" in df.columns, "Target column 'has_heart_disease' missing"


def test_2_process_micro_dataset():
    """Process micro dataset using --env ci-branch -> save processed CSV -> verify output."""
    MICRO_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)

    result = subprocess.run(
        [sys.executable, str(PREPROCESS_SCRIPT), "--env", "ci-branch"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"Preprocessing script failed: {result.stderr}"
    assert MICRO_PROCESSED_PATH.exists(), f"Processed CSV not created at {MICRO_PROCESSED_PATH}"
    assert PREPROCESSOR_PATH.exists(), f"Preprocessor artifact not created at {PREPROCESSOR_PATH}"

    df_proc = pd.read_csv(MICRO_PROCESSED_PATH)
    assert not df_proc.empty, "Processed CSV is empty"


def test_3_train_micro_model():
    """Train model on processed micro dataset using --env ci-branch -> save model -> cleanup."""
    assert MICRO_PROCESSED_PATH.exists(), "Processed micro dataset missing before training"

    result = subprocess.run(
        [sys.executable, str(TRAINING_SCRIPT), "--env", "ci-branch", "-w", "false"],
        capture_output=True,
        text=True
    )

    assert result.returncode == 0, f"Training script failed: {result.stderr}"
    assert MODEL_PATH.exists(), f"Trained model artifact missing at {MODEL_PATH}"

    # Cleanup processed micro file after quick test run
    if MICRO_PROCESSED_PATH.exists():
        MICRO_PROCESSED_PATH.unlink()
