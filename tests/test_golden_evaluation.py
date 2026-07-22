import sys
import subprocess
from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Golden Dataset & Production Pipeline Paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))
from config import WANDB_ENTITY, WANDB_PROJECT, WANDB_REGISTRY_DATASET, WANDB_COLLECTION_GOLDEN_DATASET, MODEL_NAME, MODEL_PROCESSOR

GOLDEN_PROCESSED_PATH = BASE_DIR / "dataset" / "golden-data" / "processed" / "ml_processed_data.csv"
MODEL_PATH = BASE_DIR / "models" / MODEL_NAME
PREPROCESSOR_PATH = BASE_DIR / "models" / MODEL_PROCESSOR
PREPROCESS_SCRIPT = BASE_DIR / "scripts" / "data-preprocess.py"
TRAINING_SCRIPT = BASE_DIR / "scripts" / "training.py"

# Production Quality Thresholds
MIN_ACCURACY_THRESHOLD = 0.70
MIN_F1_THRESHOLD = 0.70


def align_corrupted_dataset(df):
    """
    Check if the dataset was corrupted by vertical concatenation (pd.concat with axis=0)
    instead of horizontal (axis=1), and reconstruct the aligned dataset.
    """
    target_col = 'has_heart_disease'
    if df.shape[0] % 2 == 0 and target_col in df.columns:
        half = df.shape[0] // 2
        first_half_target_nan = df[target_col].iloc[:half].isna().sum() > 0.9 * half
        second_half_target_valid = df[target_col].iloc[half:].notna().sum() > 0.9 * half
        
        if first_half_target_nan and second_half_target_valid:
            print("Warning: Corrupted dataset format detected (vertical concatenation instead of horizontal). Aligning dataset...")
            # Drop target_col and any index-like / unnamed columns for features
            cols_to_drop = [target_col]
            for col in df.columns:
                if col.startswith("Unnamed:") or col == "patient_id":
                    cols_to_drop.append(col)
            
            df_features = df.iloc[:half].drop(columns=cols_to_drop, errors='ignore').reset_index(drop=True)
            df_target = df.iloc[half:][[target_col]].reset_index(drop=True)
            aligned_df = pd.concat([df_features, df_target], axis=1)
            # Add patient_id starting from 1
            aligned_df.insert(0, 'patient_id', range(1, len(aligned_df) + 1))
            return aligned_df
    return df


def test_1_process_golden_dataset():
    """1. Apply preprocessing script on golden raw data -> save processed dataset."""
    GOLDEN_PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    prep_res = subprocess.run(
        [sys.executable, str(PREPROCESS_SCRIPT), "--env", "ci", "--data", "golden"],
        capture_output=True,
        text=True
    )
    assert prep_res.returncode == 0, f"Preprocessing golden data failed: {prep_res.stderr}\n{prep_res.stdout}"
    assert GOLDEN_PROCESSED_PATH.exists(), f"Processed golden CSV missing at {GOLDEN_PROCESSED_PATH}"
    assert PREPROCESSOR_PATH.exists(), f"Preprocessor artifact missing at {PREPROCESSOR_PATH}"


def test_2_train_and_evaluate_golden_model():
    """2. Train model on golden dataset -> evaluate production thresholds -> log to W&B."""
    train_res = subprocess.run(
        [sys.executable, str(TRAINING_SCRIPT), "--env", "ci", "--data", "golden", "--wandb", "yes"],
        capture_output=True,
        text=True
    )
    assert train_res.returncode == 0, f"Training model on golden data failed: {train_res.stderr}\n{train_res.stdout}"
    assert MODEL_PATH.exists(), f"Trained model artifact missing at {MODEL_PATH}"

    # Evaluate Model Metrics on Golden Test Split
    import wandb
    api = wandb.Api(overrides={"entity": WANDB_ENTITY})
    artifact = api.artifact(f"{WANDB_REGISTRY_DATASET}/{WANDB_COLLECTION_GOLDEN_DATASET}:production")
    download_dir = artifact.download()
    csv_files = list(Path(download_dir).glob("*.csv"))
    assert len(csv_files) > 0, "No CSV found in downloaded golden dataset artifact"
    
    df_golden = pd.read_csv(csv_files[0])
    df_golden = align_corrupted_dataset(df_golden)
    
    # Drop columns as done in training pipeline
    cols_to_drop = ["cholesterol_total", "fasting_blood_sugar"]
    if "patient_id" in df_golden.columns:
        cols_to_drop.append("patient_id")
    df_golden.drop(columns=[c for c in cols_to_drop if c in df_golden.columns], inplace=True)

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
        run = wandb.init(
            entity=WANDB_ENTITY,
            project=WANDB_PROJECT,
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
