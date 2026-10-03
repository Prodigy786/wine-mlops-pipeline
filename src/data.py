"""Data loading, validation, and splitting pipeline for the Wine cultivar dataset."""

from typing import Tuple
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def validate_data(df_features: pd.DataFrame, df_target: pd.Series = None) -> None:
    """Validate data integrity: feature count must be 13 and no null values present.

    Args:
        df_features: Features dataframe.
        df_target: Target series (optional).

    Raises:
        ValueError: If feature count != 13 or if any null/NaN values are detected.
    """
    expected_feature_count = 13
    actual_feature_count = df_features.shape[1]

    if actual_feature_count != expected_feature_count:
        raise ValueError(
            f"Expected {expected_feature_count} features, but got {actual_feature_count}."
        )

    if df_features.isnull().sum().sum() > 0:
        raise ValueError("Features contain missing/null values.")

    if df_target is not None and df_target.isnull().sum() > 0:
        raise ValueError("Target contains missing/null values.")


def load_data(
    test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load Wine cultivar dataset, validate, and perform stratified train-test split.

    Args:
        test_size: Fraction of dataset to include in test split (default: 0.2 for 80/20 split).
        random_state: Random state seed for reproducibility (default: 42).

    Returns:
        Tuple containing X_train, X_test, y_train, y_test.
    """
    raw_data = load_wine(as_frame=True)
    X: pd.DataFrame = raw_data.data
    y: pd.Series = raw_data.target

    # Validate dataset before split
    validate_data(X, y)

    # Stratified train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Validate resulting splits
    validate_data(X_train, y_train)
    validate_data(X_test, y_test)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    print("Dataset successfully loaded and validated.")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    print(f"Target distribution in train: {y_train.value_counts().to_dict()}")
