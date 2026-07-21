import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
import wandb

BASE_DIR = Path(__file__).resolve().parent.parent
TARGET_FEATURE = 'has_heart_disease'


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected (true/false).")


def get_env_input_path(env_name, custom_input=None):
    if custom_input:
        return Path(custom_input)
    env_name = (env_name or "dev").lower()
    if env_name in ("ci-branch", "micro"):
        return BASE_DIR / "dataset" / "pytest-micro-data" / "processed" / "ml_processed_data_300.csv"
    elif env_name in ("ci-prod", "golden"):
        return BASE_DIR / "dataset" / "golden-data" / "processed" / "ml_processed_data.csv"
    else:  # dev / local
        return BASE_DIR / "dataset" / "dev" / "processed" / "ml_processed_data.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Train ML Classification Model for Heart Disease Risk Prediction.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Commands / Arguments List:
  -e, --env            Target environment: dev (default), ci-branch, ci-prod
  -i, --input-dataset  Custom path to input preprocessed CSV dataset (overrides --env default)
  -o, --output-path    Path or directory where trained model artifact will be saved (default: models)
  -m, --model          Model algorithm to train (choices: random_forest_classifier, random-forest-classifier, rfc)
  -w, --wandb          Enable W&B experiment tracking (choices: true, false. default: false)

Examples:
  python scripts/training.py                          # Defaults to dev environment
  python scripts/training.py --env ci-branch          # Uses pytest-micro-data processed dataset
  python scripts/training.py --env ci-prod --wandb=true # Uses golden-data processed dataset
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
        "--input-dataset", "-i",
        type=str,
        default=None,
        help="Custom path to input preprocessed CSV dataset"
    )
    parser.add_argument(
        "--output-path", "-o",
        type=str,
        default="models",
        help="Path or directory where trained model artifact will be saved"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="random_forest_classifier",
        choices=["random_forest_classifier", "random-forest-classifier", "rfc"],
        help="Model algorithm to train"
    )
    parser.add_argument(
        "--wandb", "-w",
        type=str2bool,
        default=False,
        help="Enable Weights & Biases (W&B) experiment logging (default: false)"
    )

    args = parser.parse_args()

    input_dataset_path = get_env_input_path(args.env, args.input_dataset)
    if not input_dataset_path.exists():
        raise FileNotFoundError(f"Input dataset file not found at: {input_dataset_path}")

    # 1. Initialize W&B Run (if enabled)
    run_rf = None
    if args.wandb:
        try:
            run_rf = wandb.init(
                project="mtech-demo-experiments",
                name=f"random-forest-{args.env}",
                tags=[args.env],
                config={
                    "environment": args.env,
                    "algorithm": "RandomForest",
                    "n_estimators": 100,
                    "max_depth": 10,
                    "random_state": 42,
                    "input_dataset": str(input_dataset_path),
                },
            )
            print(f"W&B Experiment Tracking enabled for environment: '{args.env}'")
        except Exception as e:
            run_rf = None
            print(f"Notice: W&B initialization skipped ({e}). Continuing training...")
    else:
        print("W&B experiment logging is disabled (default). Pass --wandb=true to enable W&B tracking.")


    print(f"Loading dataset from: {input_dataset_path}")
    df = pd.read_csv(input_dataset_path)
    print(f"Dataset loaded. Shape: {df.shape}")

    if TARGET_FEATURE not in df.columns:
        raise KeyError(f"Target column '{TARGET_FEATURE}' not found in dataset.")

    X = df.drop(columns=[TARGET_FEATURE])
    y = df[TARGET_FEATURE]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Split data: Train={X_train.shape[0]} samples, Test={X_test.shape[0]} samples")

    # Instantiate Random Forest Classifier
    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    y_probas = model.predict_proba(X_test)
    class_names = [str(cls) for cls in model.classes_]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test, y_probas[:, 1]) if y_probas.shape[1] == 2 else roc_auc_score(y_test, y_probas, multi_class="ovr")

    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | ROC AUC: {auc:.4f}")

    # Save Model Artifact
    output_path = Path(args.output_path)
    if output_path.suffix == ".joblib":
        save_file = output_path
        save_file.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        save_file = output_path / "random_forest_classifier.joblib"

    joblib.dump(model, save_file)
    print(f"Trained model saved successfully to: {save_file}")

    # 2. Log Metrics, Plots & Model Artifact to Weights & Biases
    if run_rf is not None and wandb.run is not None:
        try:
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

            artifact = wandb.Artifact(
                name="RandomForestClassifier",
                type="model",
                description="Random Forest pipeline model with preprocessor",
            )
            artifact.add_file(str(save_file))

            run_rf.log_artifact(artifact)
            wandb.finish()
            print("W&B logging completed successfully.")
        except Exception as e:
            print(f"Notice: W&B logging failed/skipped: {e}")


if __name__ == "__main__":
    main()



