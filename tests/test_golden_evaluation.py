import sys
import subprocess
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Golden Dataset & Production Pipeline Paths
BASE_DIR = Path(__file__).resolve().parent.parent
GOLDEN_RAW_PATH = BASE_DIR / "dataset" / "golden-data" / "raw" / "heart_disease_risk_2026.csv"
GOLDEN_PROCESSED_PATH = BASE_DIR / "dataset" / "golden-data" / "processed" / "ml_processed_data.csv"
MODEL_PATH = BASE_DIR / "models" / "random_forest_classifier.joblib"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.joblib"
PREPROCESS_SCRIPT = BASE_DIR / "scripts" / "data-preprocess.py"
TRAINING_SCRIPT = BASE_DIR / "scripts" / "training.py"

# Production Quality Thresholds
MIN_ACCURACY_THRESHOLD = 0.70
MIN_F1_THRESHOLD = 0.70


def test_1_load_golden_dataset():
    """1. Verify raw golden dataset exists and loads properly."""
    assert GOLDEN_RAW_PATH.exists(), f"Golden raw dataset missing at {GOLDEN_RAW_PATH}"
    df_raw = pd.read_csv(GOLDEN_RAW_PATH)
    assert not df_raw.empty, "Golden raw dataset is empty"
    assert "has_heart_disease" in df_raw.columns, "Target feature 'has_heart_disease' missing from golden raw data"


def test_2_process_golden_dataset():
    """2. Apply preprocessing script on golden raw data -> save processed dataset."""
    GOLDEN_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    prep_res = subprocess.run(
        [sys.executable, str(PREPROCESS_SCRIPT), "--env", "ci-prod"],
        capture_output=True,
        text=True
    )
    assert prep_res.returncode == 0, f"Preprocessing golden data failed: {prep_res.stderr}"
    assert GOLDEN_PROCESSED_PATH.exists(), f"Processed golden CSV missing at {GOLDEN_PROCESSED_PATH}"
    assert PREPROCESSOR_PATH.exists(), f"Preprocessor artifact missing at {PREPROCESSOR_PATH}"

    df_proc = pd.read_csv(GOLDEN_PROCESSED_PATH)
    assert not df_proc.empty, "Processed golden dataset is empty"


def test_3_train_and_evaluate_golden_model():
    """3. Train model on golden dataset -> evaluate production thresholds -> log to W&B."""
    assert GOLDEN_PROCESSED_PATH.exists(), "Processed golden dataset missing before training"

    train_res = subprocess.run(
        [sys.executable, str(TRAINING_SCRIPT), "--env", "ci-prod", "-w", "true"],
        capture_output=True,
        text=True
    )
    assert train_res.returncode == 0, f"Training model on golden data failed: {train_res.stderr}"
    assert MODEL_PATH.exists(), f"Trained model artifact missing at {MODEL_PATH}"

    # Evaluate Model Metrics on Golden Test Split
    df_golden = pd.read_csv(GOLDEN_PROCESSED_PATH)
    X = df_golden.drop(columns=["has_heart_disease"])
    y = df_golden["has_heart_disease"]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)
    y_probas = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test, y_probas[:, 1]) if y_probas.shape[1] == 2 else roc_auc_score(y_test, y_probas, multi_class="ovr")

    print(f"\n[Golden Evaluation Metrics] Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC AUC: {auc:.4f}")

    passed = (acc >= MIN_ACCURACY_THRESHOLD) and (f1 >= MIN_F1_THRESHOLD)

    # Log metrics & pass/fail status to W&B
    try:
        import wandb
        run = wandb.init(
            project="mtech-demo-experiments",
            name="golden-evaluation-master-merge",
            config={
                "dataset": "golden-dataset",
                "accuracy_threshold": MIN_ACCURACY_THRESHOLD,
                "f1_threshold": MIN_F1_THRESHOLD,
            },
            reinit=True
        )
        wandb.log({
            "golden_accuracy": acc,
            "golden_f1_score": f1,
            "golden_roc_auc": auc,
            "production_merge_status": "PASSED" if passed else "FAILED",
        })
        wandb.finish()
        print("Logged golden dataset evaluation metrics to W&B.")
    except Exception as e:
        print(f"Notice: W&B logging skipped/unconfigured ({e})")

    # Assert model metrics pass production threshold
    assert acc >= MIN_ACCURACY_THRESHOLD, (
        f"Production Merge Failed: Model accuracy ({acc:.4f}) below threshold ({MIN_ACCURACY_THRESHOLD})"
    )
    assert f1 >= MIN_F1_THRESHOLD, (
        f"Production Merge Failed: Model F1 score ({f1:.4f}) below threshold ({MIN_F1_THRESHOLD})"
    )
