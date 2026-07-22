# Heart Disease Classification & ML Pipeline

End-to-end Machine Learning pipeline and FastAPI web service for heart disease risk prediction.

---

## File Structure

```text
.
├── app/                              # FastAPI application package
│   ├── __init__.py                   # Package initializer
│   ├── main.py                       # FastAPI application setup & endpoint routes
│   ├── schema.py                     # Pydantic data schemas (API & ML payloads)
│   └── service.py                    # Model loading & inference service handler
├── dataset/                          # Multitenant dataset directory
│   ├── dev/                          # Local development datasets
│   │   ├── raw/                      # Raw dataset for local dev experimentation
│   │   │   └── heart_disease_risk_2026.csv
│   │   └── processed/                # Preprocessed dataset output for local dev
│   │       └── ml_processed_data.csv
│   ├── golden-data/                  # Golden production datasets for master merge validation
│   │   ├── raw/                      # Full golden raw dataset
│   │   │   └── heart_disease_risk_2026.csv
│   │   └── processed/                # Preprocessed golden dataset output
│   │       └── ml_processed_data.csv
│   └── pytest-micro-data/            # Fast micro datasets for feature branch CI/CD
│       ├── raw/                      # Micro raw dataset slice
│       │   └── heart_disease_risk_2026.csv
│       └── processed/                # Preprocessed micro dataset output
│           └── ml_processed_data.csv
├── models/                           # Saved machine learning pipeline artifacts
│   ├── preprocessor.joblib           # Fitted ColumnTransformer (OneHotEncoder + StandardScaler)
│   └── random_forest_classifier.joblib # Serialized Random Forest model binary
├── notebooks/                        # Jupyter notebooks for data analysis & experimentation
│   ├── eda.ipynb                     # Exploratory Data Analysis notebook
│   ├── rough-work.ipynb              # Scratch work & quick tests
│   └── training.ipynb                # Model prototyping notebook
├── scripts/                          # Pipeline CLI scripts
│   ├── __init__.py                   # Package initializer
│   ├── data-preprocess.py            # Data cleaning, encoding, and scaling script
│   └── training.py                   # Model training, evaluation, and W&B logging script
├── tests/                            # Pytest test suite
│   ├── test_micro_pipeline.py        # Fast pipeline verification test (Feature branch CI)
│   └── test_golden_evaluation.py     # Production evaluation test (Master merge CI & W&B)
├── config.py                         # Application configuration & environment paths
├── Dockerfile                        # Docker container definition
├── index.html                        # Single-screen web frontend UI
├── pyproject.toml                    # UV project metadata and dependencies
├── requirements.txt                  # Full Python dependencies (local/dev)
├── requirements-docker.txt           # Minimal runtime dependencies for Docker container
└── README.md                         # Project documentation
```

---

## High-Level System Architecture

```text
+-----------------------------------------------------------------------------------+
|                        1. CI/CD DEPLOYMENT LIFECYCLE                              |
|                                                                                   |
|  [ Dev Workspace ]                [ Remote Repository ]       [ Production Cloud ]|
|  - Notebooks/Dataset  ──push/PR──>  Merge to Master  ──deploy──>  FastAPI Backend |
|  - Local Scripts                      Branch                         Container    |
+-----------------------------------------------------------------------------------+

+-----------------------------------------------------------------------------------+
|                        2. RUNTIME SYSTEM ARCHITECTURE                             |
|                                                                                   |
|   +-----------------------+                    +------------------------------+   |
|   |   Frontend (Client)   |                    |    FastAPI Backend Service   |   |
|   |                       |  HTTP POST /predict|                              |   |
|   |      index.html       | ─────────────────> |  - main.py (Endpoints)       |   |
|   |  (Interactive Web UI) | <───────────────── |  - schema.py (Validation)   |   |
|   |                       |    JSON Response   |  - service.py (Inference)    |   |
|   +-----------------------+                    +--------------┬---------------+   |
|                                                               │                   |
|                                                               v Loads Artifacts   |
|                                                +------------------------------+   |
|                                                |    models/ Artifacts         |   |
|                                                |  - preprocessor.joblib       |   |
|                                                |  - rf_classifier.joblib      |   |
|                                                +------------------------------+   |
+-----------------------------------------------------------------------------------+
```

---

## Pipeline Flow Diagram

```text
+-------------------------------------------------------------------+
|                     1. LOCAL DEVELOPMENT                          |
|  - Developer works in feature branch with notebooks & scripts    |
|  - Local dataset: dataset/dev/ (raw & processed)                  |
|  - Local testing: pytest tests/test_micro_pipeline.py            |
+--------------------------------─┬─────────────────────────────────+
                                  |
                                  v Push to Remote Branch
+-------------------------------------------------------------------+
|               2. FEATURE BRANCH CI/CD AUTOMATION                  |
|  - Automated trigger on push to remote feature branch             |
|  - Runs: pytest tests/test_micro_pipeline.py                      |
|  - Dataset: dataset/pytest-micro-data/                            |
|  - Fast pipeline verification & sanity checks                     |
+--------------------------------─┬─────────────────────────────────+
                                  |
                                  v Optional Pre-Merge Evaluation
+-------------------------------------------------------------------+
|               3. OPTIONAL BRANCH GOLDEN EVALUATION                |
|  - Run golden dataset evaluation on developer feature branch     |
|  - Runs: python scripts/training.py --env ci-prod --wandb=true    |
|  - Sends metrics & experiment traces to Weights & Biases (W&B)   |
+--------------------------------─┬─────────────────────────────────+
                                  |
                                  v PR Approved -> Merge to Master
+-------------------------------------------------------------------+
|              4. PRODUCTION MASTER MERGE AUTOMATION                |
|  - Automated trigger on merge to master / production branch       |
|  - Runs: pytest tests/test_golden_evaluation.py                   |
|  - Dataset: dataset/golden-data/                                  |
|  - Validates production quality thresholds (Accuracy & F1 >= 0.70)|
|  - Logs metrics, status & model artifacts (.joblib) to W&B        |
+-------------------------------------------------------------------+
```

---

## Future Expansion

- **W&B Model Registry**: Add W&B Registry for environment versioning (Production, Staging, Dev) to track dataset and model versions.
- **Automated Cloud Deployment**: Deploy containerized FastAPI backend to Cloud hosting upon successful golden-evaluation master merge.

---

## Setup

### Using `uv` (Recommended)
```bash
uv sync
```

### Using `pip`
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Scripts

All scripts accept the following standard CLI flags:
- `--env`: Target environment, either `local` (default) or `ci`.
- `--data`: Dataset type, either `micro` (default) or `golden`.
- `--wandb`: Weights & Biases experiment tracking, either `yes` or `no` (default).
- `--data-path`: Optional custom path to local raw CSV file (overrides `--data` and skips registry downloads).

### 1. Data Preprocessing
```bash
# Default (local environment, micro dataset)
python scripts/data-preprocess.py

# CI environment with micro dataset
python scripts/data-preprocess.py --env ci --data micro

# CI environment with golden dataset
python scripts/data-preprocess.py --env ci --data golden
```

### 2. Model Training
```bash
# Default (local environment, micro dataset, W&B disabled)
python scripts/training.py

# CI environment with micro dataset (W&B disabled)
python scripts/training.py --env ci --data micro --wandb no

# CI environment with golden dataset (W&B enabled)
python scripts/training.py --env ci --data golden --wandb yes
```

---

## Running the Web App & API

To run the FastAPI server and UI, follow these steps:

1. **Create the environment using `uv`**:
   ```bash
   uv sync
   ```

2. **Activate the virtual environment**:
   - On **Windows**:
     ```bash
     .venv\Scripts\activate
     ```
   - On **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```

3. **Run the FastAPI backend**:
   ```bash
   python -m app.main
   ```

4. **Verify the server**:
   - Open [http://localhost:8000/](http://localhost:8000/) in your browser.

5. **API Swagger Docs**:
   - View interactive documentation at [http://localhost:8000/docs](http://localhost:8000/docs).

6. **Web UI Interface**:
   - Open `index.html` at the root directory using any live-server extension or directly in a web browser.

7. **Demo Prefilled Data**:
   - Click the "Low Risk" or "High Risk" button in the right corner of the UI to auto-fill the form with sample data.

8. **Predict**:
   - Click the **Predict** button to view risk analysis results.

---

## Tests

```bash
# 1. Quick Micro Pipeline Test (Feature Branches / CI)
pytest tests/test_micro_pipeline.py

# 2. Production Master Merge Evaluation Test (Golden Dataset & W&B)
pytest tests/test_golden_evaluation.py

# 3. Api Test 
pytest tests/api_test.py

# Run all tests
pytest
```

---

## FastAPI Server as Docker Container

```bash
docker build -t heart-disease-api .
docker run -p 8000:8000 heart-disease-api
```
