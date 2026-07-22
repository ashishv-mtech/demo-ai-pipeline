import sys
import subprocess
from pathlib import Path
import pandas as pd

# Project Directory Paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from config import MODEL_NAME, MODEL_PROCESSOR

MICRO_PROCESSED_PATH = BASE_DIR / "dataset" / "pytest-micro-data" / "processed" / "ml_processed_data_300.csv"
MODEL_PATH = BASE_DIR / "models" / MODEL_NAME
PREPROCESSOR_PATH = BASE_DIR / "models" / MODEL_PROCESSOR
PREPROCESS_SCRIPT = BASE_DIR / "scripts" / "data-preprocess.py"
TRAINING_SCRIPT = BASE_DIR / "scripts" / "training.py"

def test_1_process_micro_dataset():
    """Process micro dataset using --env ci-branch -> save processed CSV -> verify output."""
    MICRO_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, str(PREPROCESS_SCRIPT), "--env", "ci", "--data", "micro"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Preprocessing script failed: {result.stderr}\n{result.stdout}"
    assert MICRO_PROCESSED_PATH.exists(), f"Processed CSV not created at {MICRO_PROCESSED_PATH}"
    assert PREPROCESSOR_PATH.exists(), f"Preprocessor artifact not created at {PREPROCESSOR_PATH}"

def test_2_train_micro_model():
    """Train model on micro dataset using --env ci-branch -> save model -> cleanup."""
    result = subprocess.run(
        [sys.executable, str(TRAINING_SCRIPT), "--env", "ci", "--data", "micro", "--wandb", "no"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, f"Training script failed: {result.stderr}\n{result.stdout}"
    assert MODEL_PATH.exists(), f"Trained model artifact missing at {MODEL_PATH}"

    # Cleanup processed micro file after quick test run
    if MICRO_PROCESSED_PATH.exists():
        MICRO_PROCESSED_PATH.unlink()
