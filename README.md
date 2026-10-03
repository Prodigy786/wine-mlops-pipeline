# Wine Cultivar Classification MLOps Pipeline

[![MLOps CI Quality Gate Pipeline](https://github.com/OWNER/wine-mlops-pipeline/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/wine-mlops-pipeline/actions)

An end-to-end, reproducible Machine Learning Operations (MLOps) pipeline for chemical cultivar classification using `sklearn.datasets.load_wine` (178 samples, 13 features, 3 target classes).

---

## 🏛️ Repository Architecture

```text
wine-mlops-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI/CD workflow definition
├── data/
│   └── .gitkeep               # Directory for dataset assets
├── src/
│   ├── __init__.py            # Package initializer
│   ├── data.py                # Data loading, validation, and stratified 80/20 split
│   ├── train.py               # Dual-classifier CV training, MLflow tracking & champion promotion
│   └── evaluate.py            # Registered champion model evaluation & latency verification
├── tests/
│   ├── __init__.py            # Test package initializer
│   ├── test_data.py           # Unit tests for data loading, splitting, and schema integrity
│   └── test_model_gate.py     # Automated MLOps Quality Gate tests (F1 >= 0.88, Latency <= 30ms)
├── .gitignore                 # Excludes transient artifacts, mlruns, caches, and virtualenvs
├── Makefile                   # GNU Makefile automating install, lint, test, train, and clean
├── requirements.txt           # Pinned dependencies for reproducible execution
└── README.md                  # Comprehensive project documentation
```

---

## ⚙️ Quick Start & Local Automation

This repository uses a standard GNU `Makefile` to achieve parity between local development and CI/CD environments.

### 1. Install Dependencies
```bash
make install
```

### 2. Run Lint Checks
Enforces PEP 8 compliance via `flake8` with a maximum line length of 100:
```bash
make lint
```

### 3. Train Models and Track with MLflow
Trains candidate models for both **RandomForestClassifier** (Model Family A) and **GradientBoostingClassifier** (Model Family B) across multiple hyperparameter configurations with 5-fold stratified cross-validation, logs artifacts/metrics to MLflow, and promotes the champion model to the MLflow Model Registry:
```bash
make train
```

### 4. Run Test Suite & Automated MLOps Quality Gates
Executes all unit tests and quality gate assertions with verbose output:
```bash
make test
```

### 5. Launch MLflow Tracking UI
To explore logged experiment runs, metrics comparison, parameters, and the registered model:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

### 6. Clean Temporary Artifacts
```bash
make clean
```

---

## 🔬 Automated MLOps Quality Gates

Every CI build enforces three rigorous quality gates inside `tests/test_model_gate.py`:

| Quality Gate | Condition / Requirement | Failure Impact |
| :--- | :--- | :--- |
| **Metric Threshold Gate** | Validation Macro F1-Score $\ge 0.88$ | Blocks degraded models from deployment |
| **Inference Latency Gate** | Batch inference execution time $\le 30\text{ ms}$ | Prevents high-latency regressions |
| **Output Schema Integrity** | Predictions strictly confined to cultivar class indices $\{0, 1, 2\}$ | Protects downstream consumers from invalid schemas |

---

## 🔄 CI/CD Automation

Continuous Integration is orchestrated via GitHub Actions (`.github/workflows/ci.yml`). On every pull request and push to `main`:
1. Codespace is provisioned on Ubuntu Latest with Python 3.10.
2. Dependencies are restored and installed via `make install`.
3. Code quality is verified via `make lint`.
4. Models are trained and logged via `make train`.
5. Unit and quality gate test suites run via `make test`.
