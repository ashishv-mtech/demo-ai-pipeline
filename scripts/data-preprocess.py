import argparse
import os
from pathlib import Path
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# Paths configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
PROCESSED_DIR = DATASET_DIR / "ml-processed"
MODELS_DIR = BASE_DIR / "models"

DEFAULT_INPUT_PATH = BASE_DIR / "dataset" / "dev" / "raw" / "heart_disease_risk_2026.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "dataset" / "dev" / "processed" / "ml_processed_data.csv"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"


def get_env_paths(env_name, custom_input=None, custom_output=None):
    env_name = (env_name or "dev").lower()
    if env_name in ("ci-branch", "micro"):
        input_path = BASE_DIR / "dataset" / "pytest-micro-data" / "raw" / "heart_disease_risk_2026_300.csv"
        output_path = BASE_DIR / "dataset" / "pytest-micro-data" / "processed" / "ml_processed_data_300.csv"
    elif env_name in ("ci-prod", "golden"):
        input_path = BASE_DIR / "dataset" / "golden-data" / "raw" / "heart_disease_risk_2026.csv"
        output_path = BASE_DIR / "dataset" / "golden-data" / "processed" / "ml_processed_data.csv"
    else:  # dev / local
        input_path = BASE_DIR / "dataset" / "dev" / "raw" / "heart_disease_risk_2026.csv"
        output_path = BASE_DIR / "dataset" / "dev" / "processed" / "ml_processed_data.csv"

    if custom_input:
        input_path = Path(custom_input)
    if custom_output:
        output_path = Path(custom_output)

    return input_path, output_path


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess heart disease raw dataset into a cleaned dataset ready for ML model training.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Commands / Arguments List:
  -e, --env          Target environment: dev (default), ci-branch, ci-prod
  -i, --input-file   Custom path to input raw CSV dataset file (overrides --env default)
  -o, --output-file  Custom path to output cleaned processed CSV file (overrides --env default)

Examples:
  python scripts/data-preprocess.py                          # Defaults to dev environment
  python scripts/data-preprocess.py --env ci-branch          # Uses pytest-micro-data folder
  python scripts/data-preprocess.py --env ci-prod            # Uses golden-data folder
"""
    )
    parser.add_argument(
        "--env", "-e",
        type=str,
        default="dev",
        choices=["dev", "local", "ci-branch", "micro", "ci-prod", "golden"],
        help="Target environment (dev, ci-branch, ci-prod)"
    )
    parser.add_argument(
        "--input-file", "-i",
        type=str,
        default=None,
        help="Custom path to input raw CSV dataset file"
    )
    parser.add_argument(
        "--output-file", "-o",
        type=str,
        default=None,
        help="Custom path to output cleaned processed CSV file"
    )

    args = parser.parse_args()

    input_path, output_path = get_env_paths(args.env, args.input_file, args.output_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    # Ensure output directories exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw dataset from: {input_path}")
    df = pd.read_csv(input_path)
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

