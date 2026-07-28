"""Compare saved classifiers at a common minimum validation recall."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
TARGET_RECALL = 0.60
THRESHOLDS = np.arange(0.05, 0.501, 0.01)
MODEL_NAMES = ("logistic_regression", "random_forest", "decision_tree", "xgboost")


def load_target(path: Path) -> pd.Series:
    """Read a one-column parquet target robustly as a Series."""
    target = pd.read_parquet(path)
    return target.squeeze("columns")


def best_threshold_at_recall(y_true: pd.Series, probabilities: np.ndarray) -> dict:
    """Maximize precision among thresholds meeting the target recall."""
    candidates = []
    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
        recall = recall_score(y_true, predictions, zero_division=0)
        precision = precision_score(y_true, predictions, zero_division=0)
        if recall >= TARGET_RECALL:
            candidates.append((precision, threshold, recall))

    if not candidates:
        return {"threshold": np.nan, "precision": np.nan, "recall": np.nan}

    precision, threshold, recall = max(candidates, key=lambda item: (item[0], item[1]))
    return {"threshold": threshold, "precision": precision, "recall": recall}


def main() -> None:
    X_val = pd.read_parquet(DATA_DIR / "X_val.parquet")
    y_val = load_target(DATA_DIR / "y_val.parquet")

    rows = []
    for model_name in MODEL_NAMES:
        model = joblib.load(RESULTS_DIR / f"{model_name}.joblib")
        probabilities = model.predict_proba(X_val)[:, 1]
        result = best_threshold_at_recall(y_val, probabilities)
        rows.append(
            {
                "model": model_name,
                "threshold": result["threshold"],
                "precision": result["precision"],
                "recall": result["recall"],
                "roc_auc": roc_auc_score(y_val, probabilities),
            }
        )

    comparison = pd.DataFrame(rows).sort_values(
        ["precision", "roc_auc"], ascending=False, na_position="last"
    )
    comparison.to_csv(RESULTS_DIR / "threshold_comparison_val.csv", index=False)

    print(f"Validation threshold comparison (recall >= {TARGET_RECALL:.2f})")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved: {RESULTS_DIR / 'threshold_comparison_val.csv'}")


if __name__ == "__main__":
    main()
