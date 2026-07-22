import argparse
import os
from pathlib import Path
import pandas as pd
import joblib
import wandb
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    WANDB_ENTITY, WANDB_PROJECT, WANDB_REGISTRY_DATASET, 
    WANDB_COLLECTION_GOLDEN_DATASET, WANDB_COLLECTION_MICRO_DATASET,
    MODEL_PROCESSOR
)


# Paths configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
PROCESSED_DIR = DATASET_DIR / "ml-processed"
MODELS_DIR = BASE_DIR / "models"

DEFAULT_INPUT_PATH = BASE_DIR / "dataset" / "dev" / "raw" / "heart_disease_risk_2026.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "dataset" / "dev" / "processed" / "ml_processed_data.csv"
PREPROCESSOR_PATH = MODELS_DIR / MODEL_PROCESSOR


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
        description="Preprocess heart disease raw dataset into a cleaned dataset ready for ML model training.",
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
        help="Enable/disable Weights & Biases (yes or no)"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="micro",
        choices=["micro", "golden"],
        help="Dataset type to use (micro or golden)"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Custom path to local raw CSV dataset file (overrides --data)"
    )

    args = parser.parse_args()
    env_name = args.env.lower()
    data_type = args.data.lower()

    if args.data_path:
        input_path = Path(args.data_path.strip("'\""))
        output_path = input_path.parent / "ml_processed_data.csv"
        should_download = False
    else:
        # Get standard paths
        if data_type == "micro":
            input_path = BASE_DIR / "dataset" / "pytest-micro-data" / "raw" / "heart_disease_risk_2026_300.csv"
            output_path = BASE_DIR / "dataset" / "pytest-micro-data" / "processed" / "ml_processed_data_300.csv"
        else:  # golden
            input_path = BASE_DIR / "dataset" / "golden-data" / "raw" / "heart_disease_risk_2026.csv"
            output_path = BASE_DIR / "dataset" / "golden-data" / "processed" / "ml_processed_data.csv"

        # Decide if we need to download from registry
        if env_name == "ci":
            should_download = True
        else:  # local
            should_download = not input_path.exists()

    if should_download:
        print(f"Downloading {data_type} dataset from WandB Registry...")
        run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, job_type="preprocess")
        collection_name = WANDB_COLLECTION_MICRO_DATASET if data_type == "micro" else WANDB_COLLECTION_GOLDEN_DATASET
        artifact_path = f"{WANDB_REGISTRY_DATASET}/{collection_name}:production"
        artifact = run.use_artifact(artifact_path)
        download_dir = artifact.download()
        csv_files = list(Path(download_dir).glob("*.csv"))
        if csv_files:
            input_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy(csv_files[0], input_path)
        run.finish()
    else:
        if not input_path.exists():
            raise FileNotFoundError(f"Local dataset file not found at: {input_path}")

    # Ensure output directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw dataset from: {input_path}")
    df = pd.read_csv(input_path)
    df = align_corrupted_dataset(df)
    print(f"Initial dataset shape: {df.shape}")

    # 1. Drop redundant clinical measurements and identifiers
    cols_to_drop = ["cholesterol_total", "fasting_blood_sugar"]
    if "patient_id" in df.columns:
        cols_to_drop.append("patient_id")

    df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
    print(f"Dataset shape after dropping redundant columns {cols_to_drop}: {df.shape}")

    # 2. Define Features
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
    TARGET_FEATURE = 'has_heart_disease'

    X = df[NUMERICAL_FEATURE + CATEGORICAL_FEATURE]
    y = df[TARGET_FEATURE]

    # 3. ColumnTransformer & Pipeline
    preprocessor = ColumnTransformer(
        [
            ('num-scaler', StandardScaler(), NUMERICAL_FEATURE),
            ('cat-encoder', OneHotEncoder(sparse_output=False), CATEGORICAL_FEATURE)
        ]
    )

    print("Fitting preprocessor pipeline...")
    X_processed = preprocessor.fit_transform(X)

    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = preprocessor.get_feature_names_out()
    else:
        feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]

    df_processed = pd.DataFrame(X_processed, columns=feature_names)
    df_processed[TARGET_FEATURE] = y.values

    # 4. Save artifacts
    df_processed.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to: {output_path} (shape: {df_processed.shape})")

    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Fitted preprocessor saved to: {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    main()

