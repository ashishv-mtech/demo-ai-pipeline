import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import wandb

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    WANDB_ENTITY, WANDB_PROJECT, WANDB_REGISTRY_DATASET, 
    WANDB_REGISTRY_MODEL, WANDB_COLLECTION_GOLDEN_DATASET, 
    WANDB_COLLECTION_MICRO_DATASET, WANDB_COLLECTION_MODEL, 
    WANDB_COLLECTION_PROCESSOR, MODEL_ABR, MODEL_NAME, MODEL_PROCESSOR
)

BASE_DIR = Path(__file__).resolve().parent.parent
TARGET_FEATURE = 'has_heart_disease'
# define features
NUMERICAL_FEATURE = [
    'age', 'resting_bp_systolic', 'resting_bp_diastolic', 'hdl', 'ldl',
    'triglycerides', 'resting_heart_rate', 'max_heart_rate_achieved',
    'exercise_minutes_per_week', 'daily_steps', 'hba1c', 'bmi',
    'st_depression', 'alcohol_units_per_week', 'sleep_hours',
    'stress_score', 'diet_quality_score'
]
CATEGORICAL_FEATURE = [
    'sex', 'chest_pain_type', 'smoker_status',
    'exercise_induced_angina', 'family_history', 'wearable_owner'
]

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (true/false).")

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


def main():
    parser = argparse.ArgumentParser(
        description="Train ML Classification Model for Heart Disease Risk Prediction.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--env",
        type=str,
        default="local",
        choices=["local", "ci"],
        help="Target environment (local or ci)"
    )
    parser.add_argument(
        "--wandb",
        type=str,
        default="no",
        choices=["yes", "no"],
        help="Enable/disable Weights & Biases logging (yes or no)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="micro",
        choices=["micro", "golden"],
        help="Dataset type to use (micro or golden)"
    )
    parser.add_argument(
        "--output-path", "-o",
        type=str,
        default="models",
        help="Path or directory where trained model artifact will be saved"
    )

    args = parser.parse_args()
    env_name = args.env.lower()
    data_type = args.data.lower()
    wandb_enabled = (args.wandb.lower() == "yes")

    # 1. Initialize W&B Run
    run_rf = None
    if wandb_enabled:
        try:
            RUN_NAME = f"{MODEL_ABR}-run-{env_name}-{data_type}"
            run_rf = wandb.init(
                entity=WANDB_ENTITY,
                project=WANDB_PROJECT,
                name=RUN_NAME,
                config={
                    "algorithm": MODEL_ABR,
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42,
                    "environment": env_name,
                    "data": data_type,
                },
            )
            print(f"W&B Experiment Tracking enabled for environment: '{env_name}' ({data_type})")
        except Exception as e:
            run_rf = None
            print(f"Notice: W&B initialization skipped ({e}). Continuing training...")

    # Dataset Loading
    if data_type == "micro":
        input_path = BASE_DIR / "dataset" / "pytest-micro-data" / "raw" / "heart_disease_risk_2026_300.csv"
    else:  # golden
        input_path = BASE_DIR / "dataset" / "golden-data" / "raw" / "heart_disease_risk_2026.csv"

    # Decide if we need to download from registry
    if env_name == "ci":
        should_download = True
    else:  # local
        should_download = not input_path.exists()

    if should_download:
        print(f"Downloading {data_type} dataset from WandB Registry...")
        collection_name = WANDB_COLLECTION_MICRO_DATASET if data_type == "micro" else WANDB_COLLECTION_GOLDEN_DATASET
        artifact_path = f"{WANDB_REGISTRY_DATASET}/{collection_name}:production"
        if run_rf:
            artifact = run_rf.use_artifact(artifact_path)
        else:
            api = wandb.Api(overrides={"entity": WANDB_ENTITY})
            artifact = api.artifact(artifact_path)
        download_dir = artifact.download()
        csv_files = list(Path(download_dir).glob("*.csv"))
        if csv_files:
            input_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(csv_files[0], input_path)
    else:
        if not input_path.exists():
            raise FileNotFoundError(f"Local dataset file not found at: {input_path}")

    print(f"Loading dataset from: {input_path}")
    df = pd.read_csv(input_path)
    df = align_corrupted_dataset(df)
    print(f"Dataset loaded. Shape: {df.shape}")

    # Drop redundant columns
    cols_to_drop = ["cholesterol_total", "fasting_blood_sugar"]
    if "patient_id" in df.columns:
        cols_to_drop.append("patient_id")
    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    print(f"Dataset shape after dropping redundant columns: {df.shape}")

    if TARGET_FEATURE not in df.columns:
        raise KeyError(f"Target column '{TARGET_FEATURE}' not found in dataset.")

    X = df[NUMERICAL_FEATURE + CATEGORICAL_FEATURE]
    y = df[TARGET_FEATURE]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(
        [
            ('num-scaler', StandardScaler(), NUMERICAL_FEATURE),
            ('cat-encoder', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), CATEGORICAL_FEATURE)
        ]
    )

    rforest_classifier = Pipeline(
        steps=[
            ("data-preprocessor", preprocessor),
            (
                "rforest-classifier",
                RandomForestClassifier(
                    n_estimators=100, max_depth=10, random_state=42
                ),
            ),
        ]
    )

    print("Training RandomForestClassifier Pipeline...")
    rforestM = rforest_classifier.fit(X_train, y_train)

    y_pred = rforestM.predict(X_test)
    y_probas = rforestM.predict_proba(X_test)
    class_names = [str(cls) for cls in rforestM.classes_]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_probas[:, 1])

    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | ROC AUC: {auc:.4f}")

    # Save Models locally
    output_path_dir = BASE_DIR / args.output_path if not Path(args.output_path).is_absolute() else Path(args.output_path)
    output_path_dir.mkdir(parents=True, exist_ok=True)
    
    preprocessor_path = output_path_dir / MODEL_PROCESSOR
    model_path = output_path_dir / MODEL_NAME

    # Extract the preprocessor from pipeline to save separately
    fitted_preprocessor = rforest_classifier.named_steps["data-preprocessor"]
    joblib.dump(fitted_preprocessor, preprocessor_path)
    joblib.dump(rforest_classifier, model_path)
    
    if run_rf is not None and wandb.run is not None:
        wandb.log({
            "accuracy": acc,
            "precision": prec,
            "recall": recall,
            "f1_score": f1,
            "roc_auc_score": auc,
        })

        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=np.asarray(y_test),
                preds=y_pred,
                class_names=class_names
            ),
            "roc_curve": wandb.plot.roc_curve(
                y_true=np.asarray(y_test),
                y_probas=y_probas,
                labels=class_names
            ),
            "precision_recall_curve": wandb.plot.pr_curve(
                y_true=np.asarray(y_test),
                y_probas=y_probas,
                labels=class_names
            ),
        })

        preprocessor_artifact = wandb.Artifact(
            name="heart-risk-preprocessor",
            type="model",
            description="ColumnTransformer for encoding and scaling tabular features",
        )
        preprocessor_artifact.add_file(str(preprocessor_path))
        logged_prep = run_rf.log_artifact(preprocessor_artifact)

        artifact = wandb.Artifact(
            name="heart-risk-ml",
            type="model",
            description="Random Forest pipeline model with preprocessor",
        )
        artifact.add_file(str(model_path))
        logged_artifact = run_rf.log_artifact(artifact)

        if env_name == "ci" and data_type == "golden":
            run_rf.link_artifact(
                artifact=logged_prep,
                target_path=f"{WANDB_REGISTRY_MODEL}/{WANDB_COLLECTION_PROCESSOR}",
                aliases=["production"],
            )
            run_rf.link_artifact(
                artifact=logged_artifact,
                target_path=f"{WANDB_REGISTRY_MODEL}/{WANDB_COLLECTION_MODEL}",
                aliases=["production"],
            )

        run_rf.finish()

if __name__ == "__main__":
    main()
