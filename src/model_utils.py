"""
model_utils.py
--------------
Reusable model training, evaluation, and artifact management utilities
for the Airbnb Price Intelligence project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import time
from typing import Dict, List, Optional, Tuple, Any

import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, KFold


# ── Metrics ────────────────────────────────────────────────────────────────

def mape_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Percentage Error computed in ORIGINAL price space
    (after undoing the log1p transform).

    A MAPE of 20% means predictions are on average 20% off from actual price.
    """
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    return 100.0 * float(np.mean(np.abs((y_true_orig - y_pred_orig) / (y_true_orig + 1e-9))))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute all regression metrics.

    Returns dict with: RMSE (log), MAE (log), R², MAPE (%)
    """
    return {
        "RMSE (log)": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "MAE (log)":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        "R²":         round(float(r2_score(y_true, y_pred)), 4),
        "MAPE (%)":   round(mape_score(y_true, y_pred), 2),
    }


# ── Training & Evaluation ──────────────────────────────────────────────────

def evaluate_model(
    model,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> Tuple[Dict[str, Any], Any]:
    """
    Train a model and evaluate on train + test sets.

    Returns:
        (results_dict, fitted_model)
    """
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - t0, 2)

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    train_metrics = compute_metrics(y_train, y_pred_train)
    test_metrics  = compute_metrics(y_test, y_pred_test)

    result = {
        "Model":         model_name,
        "R² Train":      train_metrics["R²"],
        "R² Test":       test_metrics["R²"],
        "RMSE (log)":    test_metrics["RMSE (log)"],
        "MAE (log)":     test_metrics["MAE (log)"],
        "MAPE (%)":      test_metrics["MAPE (%)"],
        "Train Time (s)": train_time,
    }

    overfit = result["R² Train"] - result["R² Test"]
    flag = "⚠️ " if overfit > 0.1 else "✅"
    print(
        f"{flag} [{model_name}] R²={result['R² Test']:.4f} | "
        f"RMSE={result['RMSE (log)']:.4f} | MAPE={result['MAPE (%)']:.2f}% | "
        f"{train_time}s"
    )
    if overfit > 0.1:
        print(f"   Overfitting gap: {overfit:.3f}")

    return result, model


def cross_validate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    scoring: str = "r2",
) -> Dict[str, float]:
    """
    Run k-fold cross-validation and return mean ± std.

    Args:
        model:    Unfitted estimator
        X, y:     Features and target
        n_splits: Number of CV folds
        scoring:  Sklearn scoring metric

    Returns:
        {"mean": ..., "std": ..., "scores": [...]}
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring=scoring, n_jobs=-1)
    return {
        "mean":   round(float(scores.mean()), 4),
        "std":    round(float(scores.std()), 4),
        "scores": [round(float(s), 4) for s in scores],
    }


# ── Visualizations ─────────────────────────────────────────────────────────

def plot_feature_importance(
    model,
    feature_names: List[str],
    top_n: int = 20,
    output_path: Optional[str] = None,
) -> pd.Series:
    """
    Plot horizontal bar chart of feature importances.

    Works with any model that has .feature_importances_ (RF, XGBoost, LGBM, GBM).

    Returns:
        Series of feature importances, sorted descending.
    """
    if not hasattr(model, "feature_importances_"):
        print("Model has no feature_importances_ attribute.")
        return pd.Series(dtype=float)

    imp = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    top = imp.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    ax.barh(top.index[::-1], top.values[::-1], color="#667eea", alpha=0.9)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()

    return imp


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted (log price)",
    output_path: Optional[str] = None,
) -> None:
    """Scatter plot of actual vs predicted values."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(y_true, y_pred, alpha=0.2, s=10, color="#667eea")
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="Perfect prediction")
    ax.set_xlabel("Actual", fontsize=11)
    ax.set_ylabel("Predicted", fontsize=11)
    r2 = r2_score(y_true, y_pred)
    ax.set_title(f"{title}\nR² = {r2:.4f}", fontsize=12, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_model_comparison(
    results_df: pd.DataFrame,
    output_path: Optional[str] = None,
) -> None:
    """Bar chart comparison of all models on key metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    models = results_df.index.tolist()
    palette = ["#667eea", "#764ba2", "#f093fb", "#4facfe", "#43e97b", "#fa709a", "#ffecd2"]

    for ax, metric, title in zip(
        axes,
        ["R² Test", "RMSE (log)", "MAPE (%)"],
        ["R² Test (↑)", "RMSE log (↓)", "MAPE % (↓)"],
    ):
        vals = results_df[metric].values
        colors = [palette[i % len(palette)] for i in range(len(models))]
        ax.bar(range(len(models)), vals, color=colors, alpha=0.9)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontweight="bold")

    plt.suptitle("Model Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


# ── Artifact I/O ───────────────────────────────────────────────────────────

def save_artifacts(
    model,
    scaler,
    feature_names: List[str],
    metadata: Dict[str, Any],
    model_dir: str = "models",
) -> None:
    """
    Save model, scaler, and metadata to disk.

    Files saved:
        - models/best_model.pkl   (compress=3 to reduce file size)
        - models/scaler.pkl
        - models/feature_names.json
        - models/model_metadata.json
    """
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "best_model.pkl")
    joblib.dump(model, model_path, compress=3)
    size_mb = os.path.getsize(model_path) / 1e6
    print(f"✅ Model saved: {model_path} ({size_mb:.1f} MB)")

    scaler_path = os.path.join(model_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved: {scaler_path}")

    feat_path = os.path.join(model_dir, "feature_names.json")
    feat_data = {"feature_names": feature_names, "n_features": len(feature_names)}
    with open(feat_path, "w") as f:
        json.dump(feat_data, f, indent=2)
    print(f"✅ Feature names saved: {feat_path}")

    meta_path = os.path.join(model_dir, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"✅ Metadata saved: {meta_path}")


def load_artifacts(
    model_dir: str = "models",
) -> Tuple[Any, Any, List[str], Dict[str, Any]]:
    """
    Load all model artifacts from disk.

    Returns:
        (model, scaler, feature_names, metadata)
    """
    model   = joblib.load(os.path.join(model_dir, "best_model.pkl"))
    scaler  = joblib.load(os.path.join(model_dir, "scaler.pkl"))

    with open(os.path.join(model_dir, "feature_names.json")) as f:
        feat_data = json.load(f)
    feature_names = feat_data["feature_names"]

    metadata = {}
    meta_path = os.path.join(model_dir, "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    print(f"✅ Artifacts loaded from {model_dir}")
    return model, scaler, feature_names, metadata


# ── Inference ──────────────────────────────────────────────────────────────

def predict_price(
    listing_features: Dict[str, Any],
    model,
    scaler,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Generate a price prediction from raw listing feature values.

    Args:
        listing_features: Dict mapping feature_name → value
        model:            Fitted XGBoost or similar model
        scaler:           Fitted RobustScaler
        feature_names:    Ordered list of feature names the model expects

    Returns:
        Dict with keys: predicted_price, price_low, price_high
    """
    # Build feature vector in correct order
    x = np.array(
        [listing_features.get(f, 0.0) for f in feature_names], dtype=np.float32
    ).reshape(1, -1)

    # Scale and predict
    x_scaled = scaler.transform(x)
    log_pred  = float(model.predict(x_scaled)[0])

    # Convert from log-space
    predicted_price = float(np.expm1(log_pred))

    # Approximate ±15% confidence interval
    price_low  = predicted_price * 0.85
    price_high = predicted_price * 1.15

    return {
        "predicted_price": round(predicted_price, 2),
        "price_low":       round(price_low, 2),
        "price_high":      round(price_high, 2),
        "log_prediction":  round(log_pred, 4),
    }
