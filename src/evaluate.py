"""Evaluation and inference verification for the registered champion model."""

import os
import time
from typing import Dict
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss

try:
    from src.data import load_data
except ModuleNotFoundError:
    from data import load_data

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MODEL_REGISTRY_NAME = "WineClassifier"
CHAMPION_ALIAS = "champion"


def load_champion_model():
    """Load the champion model from MLflow Model Registry."""
    mlflow.set_tracking_uri(TRACKING_URI)
    model_uri = f"models:/{MODEL_REGISTRY_NAME}@{CHAMPION_ALIAS}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
    except Exception:
        # Fallback to latest version if alias lookup needs direct version resolution
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_model_version_by_alias(MODEL_REGISTRY_NAME, CHAMPION_ALIAS)
        model = mlflow.sklearn.load_model(f"models:/{MODEL_REGISTRY_NAME}/{model_version.version}")
    return model


def evaluate_champion_model() -> Dict[str, float]:
    """Compute test split metrics and inference latency on champion model.

    Returns:
        Dictionary of test metrics and latency.
    """
    _, X_test, _, y_test = load_data()
    model = load_champion_model()

    start_time = time.perf_counter()
    y_pred = model.predict(X_test)
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000.0

    # Probabilities for log loss
    y_prob = model.predict_proba(X_test)
    classes = np.unique(y_test)

    test_f1 = float(f1_score(y_test, y_pred, average="macro"))
    test_acc = float(accuracy_score(y_test, y_pred))
    test_loss = float(log_loss(y_test, y_prob, labels=classes))

    results = {
        "test_macro_f1": test_f1,
        "test_accuracy": test_acc,
        "test_log_loss": test_loss,
        "batch_latency_ms": latency_ms,
    }

    print("=" * 50)
    print("CHAMPION MODEL INFERENCE VERIFICATION")
    print("=" * 50)
    print(f"Test Macro F1-Score : {test_f1:.4f}")
    print(f"Test Accuracy       : {test_acc:.4f}")
    print(f"Test Log Loss       : {test_loss:.4f}")
    print(f"Batch Latency       : {latency_ms:.2f} ms")
    print("=" * 50)

    return results


if __name__ == "__main__":
    evaluate_champion_model()
