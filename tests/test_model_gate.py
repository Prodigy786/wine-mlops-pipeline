"""Automated MLOps Quality Gate tests.

Validates:
1. Metric Threshold Gate: Validation / Evaluation Macro F1-score >= 0.88.
2. Inference Latency Gate: Batch inference time <= 30 ms.
3. Output Schema Integrity: Predictions contain class indices only (0, 1, or 2).
"""

import time
import numpy as np
import pytest
from sklearn.metrics import f1_score
from src.data import load_data
from src.evaluate import load_champion_model
from src.train import main as run_training


@pytest.fixture(scope="session", autouse=True)
def ensure_model_available():
    """Ensure a trained champion model exists before running gates."""
    try:
        load_champion_model()
    except Exception:
        print("\nChampion model not found. Triggering training and registry promotion...")
        run_training()


def test_metric_threshold_gate():
    """Gate 1: Ensure Model Macro F1-score meets or exceeds threshold of 0.88."""
    _, X_test, _, y_test = load_data()
    model = load_champion_model()

    y_pred = model.predict(X_test)
    test_f1 = f1_score(y_test, y_pred, average="macro")

    min_f1_threshold = 0.88
    assert test_f1 >= min_f1_threshold, (
        f"Model quality gate failed: Macro F1-score {test_f1:.4f} is below "
        f"the required threshold of {min_f1_threshold}."
    )


def test_inference_latency_gate():
    """Gate 2: Ensure batch inference latency on test split is <= 30 ms."""
    _, X_test, _, _ = load_data()
    model = load_champion_model()

    # Warm-up run
    _ = model.predict(X_test)

    # Timed inference run
    start_time = time.perf_counter()
    _ = model.predict(X_test)
    elapsed_time_ms = (time.perf_counter() - start_time) * 1000.0

    max_latency_threshold_ms = 30.0
    assert elapsed_time_ms <= max_latency_threshold_ms, (
        f"Inference latency gate failed: Batch latency {elapsed_time_ms:.2f} ms "
        f"exceeds max allowable latency of {max_latency_threshold_ms} ms."
    )


def test_output_schema_integrity():
    """Gate 3: Ensure model predictions conform to schema (valid class indices 0, 1, or 2)."""
    _, X_test, _, _ = load_data()
    model = load_champion_model()

    predictions = model.predict(X_test)

    valid_classes = {0, 1, 2}
    unique_preds = set(np.unique(predictions))

    # All unique predicted values must be subset of valid classes
    assert unique_preds.issubset(valid_classes), (
        f"Output schema gate failed: Found unexpected class labels {unique_preds - valid_classes}."
    )

    # Output shape must equal input row count
    assert len(predictions) == len(X_test), (
        f"Output schema gate failed: Expected {len(X_test)} predictions, got {len(predictions)}."
    )
