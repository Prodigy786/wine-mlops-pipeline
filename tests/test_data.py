"""Unit tests for the data loading and validation pipeline."""

import numpy as np
import pandas as pd
import pytest
from src.data import load_data, validate_data


def test_load_data_shapes():
    """Verify that dataset is loaded and split into 80% train and 20% test."""
    X_train, X_test, y_train, y_test = load_data(test_size=0.2, random_state=42)

    total_samples = len(X_train) + len(X_test)
    assert total_samples == 178, f"Expected 178 total samples, got {total_samples}"

    # 80/20 split on 178 samples -> 142 train, 36 test
    assert len(X_train) == 142
    assert len(X_test) == 36
    assert len(y_train) == 142
    assert len(y_test) == 36


def test_feature_count_and_columns():
    """Verify that exactly 13 features are present and feature names match."""
    X_train, X_test, _, _ = load_data()
    assert X_train.shape[1] == 13
    assert X_test.shape[1] == 13


def test_target_classes():
    """Verify that target contains 3 distinct cultivar classes (0, 1, 2)."""
    _, _, y_train, y_test = load_data()
    unique_train = sorted(y_train.unique().tolist())
    unique_test = sorted(y_test.unique().tolist())

    assert unique_train == [0, 1, 2]
    assert unique_test == [0, 1, 2]


def test_no_null_values():
    """Verify that no NaN or null values exist in features or targets."""
    X_train, X_test, y_train, y_test = load_data()
    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0
    assert y_train.isnull().sum() == 0
    assert y_test.isnull().sum() == 0


def test_validate_data_error_on_missing_features():
    """Verify that validate_data raises ValueError when feature count != 13."""
    invalid_df = pd.DataFrame(np.random.rand(10, 10))
    with pytest.raises(ValueError, match="Expected 13 features"):
        validate_data(invalid_df)


def test_validate_data_error_on_nulls():
    """Verify that validate_data raises ValueError when null values are injected."""
    X_train, _, _, _ = load_data()
    corrupted_df = X_train.copy()
    corrupted_df.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="contain missing/null values"):
        validate_data(corrupted_df)
