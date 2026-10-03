"""Model training, hyperparameter experimentation, MLflow tracking, and registry promotion."""

import os
import time
from typing import Any, Dict, List, Tuple
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

try:
    from src.data import load_data
except ModuleNotFoundError:
    from data import load_data

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

EXPERIMENT_NAME = "Wine-Cultivar-Classification"
MODEL_REGISTRY_NAME = "WineClassifier"
CHAMPION_ALIAS = "champion"
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def setup_mlflow() -> str:
    """Configure MLflow tracking URI and experiment.

    Returns:
        The experiment ID string.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
    else:
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(EXPERIMENT_NAME)
    return experiment_id


def get_hyperparameter_grids() -> List[Dict[str, Any]]:
    """Define candidate hyperparameter configurations for Model Family A and B.

    Returns:
        List of configuration dictionaries.
    """
    configs: List[Dict[str, Any]] = [
        # Model Family A: RandomForestClassifier (3 configurations)
        {
            "family": "RandomForestClassifier",
            "model_class": RandomForestClassifier,
            "params": {
                "n_estimators": 50,
                "max_depth": 3,
                "min_samples_split": 2,
                "random_state": 42,
            },
            "tags": {"model_family": "RandomForest", "config_id": "RF_cfg1"},
        },
        {
            "family": "RandomForestClassifier",
            "model_class": RandomForestClassifier,
            "params": {
                "n_estimators": 100,
                "max_depth": 5,
                "min_samples_split": 4,
                "random_state": 42,
            },
            "tags": {"model_family": "RandomForest", "config_id": "RF_cfg2"},
        },
        {
            "family": "RandomForestClassifier",
            "model_class": RandomForestClassifier,
            "params": {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_split": 2,
                "random_state": 42,
            },
            "tags": {"model_family": "RandomForest", "config_id": "RF_cfg3"},
        },
        # Model Family B: GradientBoostingClassifier (3 configurations)
        {
            "family": "GradientBoostingClassifier",
            "model_class": GradientBoostingClassifier,
            "params": {
                "n_estimators": 50,
                "learning_rate": 0.05,
                "max_depth": 3,
                "random_state": 42,
            },
            "tags": {"model_family": "GradientBoosting", "config_id": "GBM_cfg1"},
        },
        {
            "family": "GradientBoostingClassifier",
            "model_class": GradientBoostingClassifier,
            "params": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42,
            },
            "tags": {"model_family": "GradientBoosting", "config_id": "GBM_cfg2"},
        },
        {
            "family": "GradientBoostingClassifier",
            "model_class": GradientBoostingClassifier,
            "params": {
                "n_estimators": 120,
                "learning_rate": 0.2,
                "max_depth": 4,
                "random_state": 42,
            },
            "tags": {"model_family": "GradientBoosting", "config_id": "GBM_cfg3"},
        },
    ]
    return configs


def evaluate_cv(
    model_class: Any,
    params: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
) -> Dict[str, float]:
    """Perform 5-fold stratified cross-validation and compute train & val metrics.

    Args:
        model_class: Classifier class to instantiate.
        params: Hyperparameters for the classifier.
        X: Feature matrix (training split).
        y: Target array (training split).
        n_splits: Number of cross-validation folds.

    Returns:
        Dictionary containing mean cross-validation train and validation metrics.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    val_f1_list, val_acc_list, val_loss_list = [], [], []
    train_f1_list, train_acc_list, train_loss_list = [], [], []

    classes = np.unique(y)

    for train_idx, val_idx in skf.split(X, y):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]

        fold_model = model_class(**params)
        fold_model.fit(X_fold_train, y_fold_train)

        # Predictions on validation fold
        y_val_pred = fold_model.predict(X_fold_val)
        y_val_prob = fold_model.predict_proba(X_fold_val)

        val_f1_list.append(f1_score(y_fold_val, y_val_pred, average="macro"))
        val_acc_list.append(accuracy_score(y_fold_val, y_val_pred))
        val_loss_list.append(log_loss(y_fold_val, y_val_prob, labels=classes))

        # Predictions on train fold
        y_train_pred = fold_model.predict(X_fold_train)
        y_train_prob = fold_model.predict_proba(X_fold_train)

        train_f1_list.append(f1_score(y_fold_train, y_train_pred, average="macro"))
        train_acc_list.append(accuracy_score(y_fold_train, y_train_pred))
        train_loss_list.append(log_loss(y_fold_train, y_train_prob, labels=classes))

    return {
        "train_macro_f1": float(np.mean(train_f1_list)),
        "train_accuracy": float(np.mean(train_acc_list)),
        "train_log_loss": float(np.mean(train_loss_list)),
        "val_macro_f1": float(np.mean(val_f1_list)),
        "val_accuracy": float(np.mean(val_acc_list)),
        "val_log_loss": float(np.mean(val_loss_list)),
    }


def train_and_track_experiments() -> Tuple[str, str, float]:
    """Execute hyperparameter experimentation, log to MLflow, and return best run info.

    Returns:
        Tuple of (best_run_id, best_model_name, best_val_f1).
    """
    setup_mlflow()
    X_train, _, y_train, _ = load_data()
    configs = get_hyperparameter_grids()

    best_run_id = ""
    best_val_f1 = -1.0
    best_model_name = ""

    print(f"Starting MLflow experiment tracking for {len(configs)} configurations...")

    for i, cfg in enumerate(configs, 1):
        family = cfg["family"]
        params = cfg["params"]
        tags = cfg["tags"]
        model_class = cfg["model_class"]
        run_name = f"{family}_{tags.get('config_id', i)}"

        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            # 1. Evaluate via 5-Fold Stratified Cross-Validation
            metrics = evaluate_cv(model_class, params, X_train, y_train, n_splits=5)

            # 2. Fit final model on full training split
            final_model = model_class(**params)
            final_model.fit(X_train, y_train)

            # 3. Model signature and input example
            signature = infer_signature(X_train, final_model.predict(X_train))
            input_example = X_train.iloc[:5]

            # 4. Log parameters, metrics, tags, and model artifact
            # Handle None in params for MLflow logging
            sanitized_params = {
                k: ("None" if v is None else v) for k, v in params.items()
            }
            mlflow.log_params(sanitized_params)
            mlflow.log_metrics(metrics)
            mlflow.set_tags(tags)
            mlflow.set_tag("algorithm", family)

            mlflow.sklearn.log_model(
                sk_model=final_model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
                serialization_format="cloudpickle",
            )

            val_f1 = metrics["val_macro_f1"]
            print(
                f"[{i}/{len(configs)}] {run_name} | "
                f"Val Macro F1: {val_f1:.4f} | "
                f"Val Acc: {metrics['val_accuracy']:.4f} | "
                f"Val Loss: {metrics['val_log_loss']:.4f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_run_id = run_id
                best_model_name = run_name

    print(f"\nBest Run ID: {best_run_id} ({best_model_name}) with Val Macro F1: {best_val_f1:.4f}")
    return best_run_id, best_model_name, best_val_f1


def promote_champion_model(best_run_id: str) -> Any:
    """Register the winning model and assign the champion alias.

    Args:
        best_run_id: MLflow Run ID of the best performing configuration.

    Returns:
        The registered model version object.
    """
    client = MlflowClient()
    model_uri = f"runs:/{best_run_id}/model"

    print(f"Registering winning model from {model_uri} into registry '{MODEL_REGISTRY_NAME}'...")
    model_version = mlflow.register_model(model_uri=model_uri, name=MODEL_REGISTRY_NAME)

    # Allow brief moment for registry to register
    time.sleep(1)

    # Assign alias 'champion' to this version
    client.set_registered_model_alias(
        name=MODEL_REGISTRY_NAME,
        alias=CHAMPION_ALIAS,
        version=model_version.version,
    )
    print(
        f"Assigned alias '{CHAMPION_ALIAS}' to {MODEL_REGISTRY_NAME} "
        f"Version {model_version.version}."
    )
    return model_version


def main() -> None:
    """Main training and registry execution entry point."""
    best_run_id, _, _ = train_and_track_experiments()
    promote_champion_model(best_run_id)


if __name__ == "__main__":
    main()
