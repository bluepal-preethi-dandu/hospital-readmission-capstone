from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_processed_data():
    """
    Load the PRE-SMOTE processed training data (raw imbalance, already
    imputed/encoded/scaled by Stage 2) plus the untouched val/test splits.
    SMOTE is applied later, inside pipelines, never here.
    """
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train_preSMOTE.parquet")
    X_val = pd.read_parquet(PROCESSED_DIR / "X_val.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test.parquet")

    y_train = pd.read_parquet(PROCESSED_DIR / "y_train_preSMOTE.parquet")["readmitted_30"]
    y_val = pd.read_parquet(PROCESSED_DIR / "y_val.parquet")["readmitted_30"]
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test.parquet")["readmitted_30"]

    print(f"Loaded processed data -> train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")
    print("Train class balance (pre-SMOTE, real distribution):", y_train.value_counts().to_dict())
    print("Val class balance (untouched):", y_val.value_counts().to_dict())
    print("Test class balance (untouched):", y_test.value_counts().to_dict())

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_pipeline(model_name: str):
    """
    Every model is wrapped as SMOTE -> estimator in an imblearn Pipeline.
    This guarantees that whenever .fit() is called on a subset of data
    (a CV fold, a GridSearchCV inner fold, or the full training set),
    SMOTE is refit on exactly that subset -- no leakage across folds.
    """
    if model_name == "logistic_regression":
        estimator = LogisticRegression(max_iter=2000, random_state=42)
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            random_state=42,
            # Avoid nested joblib workers (CV workers plus multi-threaded models)
            # in the constrained Windows execution environment.
            n_jobs=1,
            max_depth=10,
            min_samples_leaf=5,
        )
    elif model_name == "decision_tree":
        estimator = DecisionTreeClassifier(
            random_state=42,
            max_depth=6,
            min_samples_leaf=10,
        )
    elif model_name == "xgboost":
        estimator = XGBClassifier(
            random_state=42,
            n_jobs=1,
            eval_metric="logloss",
            use_label_encoder=False,
            max_depth=4,
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return ImbPipeline(steps=[("smote", SMOTE(random_state=42)), ("model", estimator)])


PARAM_GRIDS = {
    "random_forest": {
        # Refit the prior selected configuration after the data intervention.
        # The original 12-combination search exceeds the execution ceiling here.
        "model__n_estimators": [400],
        "model__max_depth": [10],
        "model__min_samples_leaf": [10],
    },
    "logistic_regression": {
        "model__C": [0.01],
        "model__penalty": ["l2"],
    },
    "xgboost": {
        # Refit the prior selected configuration after the data intervention.
        "model__max_depth": [4],
        "model__learning_rate": [0.1],
        "model__n_estimators": [300],
    },
}


def tune_or_fit(model_name: str, pipeline, X_train, y_train):
    if model_name in PARAM_GRIDS:
        print(f"Tuning {model_name} with GridSearchCV (SMOTE refit inside each inner fold)...")
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=PARAM_GRIDS[model_name],
            scoring="roc_auc",
            cv=3,
            n_jobs=1,
        )
        search.fit(X_train, y_train)
        print(f"  Best params for {model_name}: {search.best_params_}")
        return search.best_estimator_

    pipeline.fit(X_train, y_train)
    return pipeline


def cross_validate_model(model_name: str, pipeline, X_train, y_train):
    """
    Stratified 5-fold CV using the SAME imblearn pipeline, so SMOTE is
    refit fresh inside each fold's training portion only. These numbers
    are now directly comparable to the validation-set numbers.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    cv_results = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)

    summary = {
        f"cv_{metric}_mean": round(float(np.mean(cv_results[f"test_{metric}"])), 4)
        for metric in scoring
    }
    summary.update({
        f"cv_{metric}_std": round(float(np.std(cv_results[f"test_{metric}"])), 4)
        for metric in scoring
    })
    print(f"  CV results for {model_name}: {summary}")
    return summary


def evaluate_model(model, X_eval, y_eval, model_name: str, split_name: str):
    preds = model.predict(X_eval)
    probs = model.predict_proba(X_eval)[:, 1]

    metrics = {
        "model": model_name,
        "split": split_name,
        "accuracy": round(float(accuracy_score(y_eval, preds)), 4),
        "precision": round(float(precision_score(y_eval, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_eval, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_eval, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_eval, probs)), 4),
        "pr_auc": round(float(average_precision_score(y_eval, probs)), 4),
    }

    cm = confusion_matrix(y_eval, preds)
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
    cm_df.to_csv(RESULTS_DIR / f"{model_name}_{split_name}_confusion_matrix.csv")

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix ({split_name})")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{model_name}_{split_name}_confusion_matrix.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y_eval, probs)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve ({split_name})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{model_name}_{split_name}_roc_curve.png")
    plt.close()

    return metrics


def save_feature_importance(pipeline, model_name: str, feature_names):
    if model_name not in ("random_forest", "xgboost"):
        return
    estimator = pipeline.named_steps["model"]
    importances = pd.Series(estimator.feature_importances_, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(15)
    top_features.to_csv(RESULTS_DIR / f"{model_name}_top_features.csv")
    print(f"  Top features saved for {model_name}")


def main():
    parser = argparse.ArgumentParser()
    all_model_names = ["logistic_regression", "random_forest", "decision_tree", "xgboost"]
    parser.add_argument("--models", nargs="+", choices=all_model_names)
    parser.add_argument("--reset-metrics", action="store_true")
    args = parser.parse_args()

    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    feature_names = X_train.columns.tolist()

    model_names = args.models or all_model_names
    all_val_metrics = []
    all_cv_metrics = []
    fitted_models = {}

    for model_name in model_names:
        print(f"\n=== {model_name} ===")
        pipeline = build_pipeline(model_name)
        pipeline = tune_or_fit(model_name, pipeline, X_train, y_train)

        cv_summary = cross_validate_model(model_name, build_pipeline(model_name), X_train, y_train)
        cv_summary["model"] = model_name
        all_cv_metrics.append(cv_summary)

        val_metrics = evaluate_model(pipeline, X_val, y_val, model_name, split_name="val")
        all_val_metrics.append(val_metrics)

        save_feature_importance(pipeline, model_name, feature_names)
        joblib.dump(pipeline, RESULTS_DIR / f"{model_name}.joblib")
        fitted_models[model_name] = pipeline

    val_metrics_df = pd.DataFrame(all_val_metrics)
    cv_metrics_df = pd.DataFrame(all_cv_metrics)
    if not args.reset_metrics:
        for name, current in (("model_val_metrics.csv", val_metrics_df), ("model_cv_metrics.csv", cv_metrics_df)):
            path = RESULTS_DIR / name
            if path.exists():
                prior = pd.read_csv(path)
                combined = pd.concat([prior[~prior["model"].isin(model_names)], current], ignore_index=True)
                if name == "model_val_metrics.csv":
                    val_metrics_df = combined
                else:
                    cv_metrics_df = combined
    val_metrics_df = val_metrics_df.set_index("model").reindex(all_model_names).dropna(how="all").reset_index()
    cv_metrics_df = cv_metrics_df.set_index("model").reindex(all_model_names).dropna(how="all").reset_index()
    val_metrics_df.to_csv(RESULTS_DIR / "model_val_metrics.csv", index=False)
    cv_metrics_df.to_csv(RESULTS_DIR / "model_cv_metrics.csv", index=False)

    best_model_name = model_names[0] if len(model_names) == 1 else val_metrics_df.sort_values("roc_auc", ascending=False).iloc[0]["model"]
    best_model = fitted_models[best_model_name]
    print(f"\nBest model on validation ROC-AUC: {best_model_name}")

    test_metrics = evaluate_model(best_model, X_test, y_test, best_model_name, split_name="test")
    with open(RESULTS_DIR / "final_test_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(test_metrics, handle, indent=2)

    print("\nValidation metrics (all models):")
    print(val_metrics_df.to_string(index=False))
    print("\nCross-validation metrics (all models, SMOTE refit per fold):")
    print(cv_metrics_df.to_string(index=False))
    print("\nFinal held-out test metrics (best model only):")
    print(json.dumps(test_metrics, indent=2))
    print("\nResults saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
