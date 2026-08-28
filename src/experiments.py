"""Phase 2 experiment runner: baselines, ML models, ablation, sensitivity."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.baselines import (
    ModelSpec,
    build_catboost_model,
    build_model,
    get_ablation_feature_sets,
    get_business_baselines,
    get_ml_models,
)
from src.config import (
    ARTIFACTS_DIR,
    FEATURE_SETS,
    SPLIT_PRIMARY,
    SPLIT_SENSITIVITY,
)
from src.data import get_split, load_train_data, split_xy, summarize_split
from src.features import get_feature_columns
from src.metrics import evaluate_predictions


def _evaluate_run(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[dict[str, float], dict[str, float], float]:
    start = time.perf_counter()
    model.fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    train_metrics = evaluate_predictions(y_train, model.predict(X_train))
    val_metrics = evaluate_predictions(y_val, model.predict(X_val))
    return train_metrics, val_metrics, elapsed


def _result_row(
    model: str,
    feature_set: str,
    validation_split: str,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    train_rows: int,
    validation_rows: int,
    training_time_seconds: float,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "model": model,
        "feature_set": feature_set,
        "validation_split": validation_split,
        "train_mae": train_metrics["mae"],
        "MAE": val_metrics["mae"],
        "RMSE": val_metrics["rmse"],
        "R2": val_metrics["r2"],
        "MAPE": val_metrics["mape"],
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "training_time_seconds": round(training_time_seconds, 3),
        "notes": notes,
    }


def run_model_spec(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    split_name: str,
) -> dict[str, Any]:
    model = build_model(spec)
    train_m, val_m, elapsed = _evaluate_run(model, X_train, y_train, X_val, y_val)
    return _result_row(
        spec.name,
        spec.feature_set,
        split_name,
        train_m,
        val_m,
        len(X_train),
        len(X_val),
        elapsed,
        spec.notes,
    )


def run_all_models(split_name: str) -> list[dict[str, Any]]:
    df = load_train_data()
    train_df, val_df = get_split(df, split_name)
    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)

    rows: list[dict[str, Any]] = []
    for spec in get_business_baselines() + get_ml_models("full"):
        print(f"[{split_name}] {spec.name} ...", flush=True)
        rows.append(run_model_spec(spec, X_train, y_train, X_val, y_val, split_name))

    catboost = build_catboost_model("full")
    if catboost is not None:
        print(f"[{split_name}] catboost ...", flush=True)
        train_m, val_m, elapsed = _evaluate_run(catboost, X_train, y_train, X_val, y_val)
        rows.append(
            _result_row(
                "catboost",
                "full",
                split_name,
                train_m,
                val_m,
                len(X_train),
                len(X_val),
                elapsed,
                "Native categorical handling",
            )
        )
    return rows


def run_ablation(split_name: str = SPLIT_PRIMARY) -> list[dict[str, Any]]:
    df = load_train_data()
    train_df, val_df = get_split(df, split_name)
    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)

    estimator = HistGradientBoostingRegressor(
        max_depth=8,
        learning_rate=0.08,
        max_iter=300,
        random_state=42,
    )
    rows: list[dict[str, Any]] = []
    for fs in get_ablation_feature_sets():
        spec = ModelSpec(
            f"ablation_{fs}",
            estimator,
            fs,
            f"HistGB feature set {fs}",
        )
        print(f"[ablation/{split_name}] set {fs} ...", flush=True)
        rows.append(run_model_spec(spec, X_train, y_train, X_val, y_val, split_name))
    return rows


def run_log_target_experiment(split_name: str = SPLIT_PRIMARY) -> dict[str, Any]:
    df = load_train_data()
    train_df, val_df = get_split(df, split_name)
    X_train, y_train = split_xy(train_df)
    X_val, y_val = split_xy(val_df)

    normal_spec = ModelSpec(
        "hist_gradient_boosting",
        HistGradientBoostingRegressor(
            max_depth=8,
            learning_rate=0.08,
            max_iter=300,
            random_state=42,
        ),
        "full",
        "Normal target",
    )
    log_spec = ModelSpec(
        "hist_gradient_boosting_log1p",
        HistGradientBoostingRegressor(
            max_depth=8,
            learning_rate=0.08,
            max_iter=300,
            random_state=42,
        ),
        "full",
        "log1p target, expm1 predictions",
        use_log_target=True,
    )

    normal = run_model_spec(normal_spec, X_train, y_train, X_val, y_val, split_name)
    log_row = run_model_spec(log_spec, X_train, y_train, X_val, y_val, split_name)
    return {"normal": normal, "log1p": log_row}


def get_split_report() -> dict[str, Any]:
    df = load_train_data()
    primary_train, primary_val = get_split(df, SPLIT_PRIMARY)
    sens_train, sens_val = get_split(df, SPLIT_SENSITIVITY)
    return {
        "primary": summarize_split(primary_train, primary_val),
        "sensitivity": summarize_split(sens_train, sens_val),
    }


def run_phase2() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    split_report = get_split_report()
    print("Primary split:", split_report["primary"])
    print("Sensitivity split:", split_report["sensitivity"])

    primary_rows = run_all_models(SPLIT_PRIMARY)
    sensitivity_rows = run_all_models(SPLIT_SENSITIVITY)
    all_rows = primary_rows + sensitivity_rows

    log_exp = run_log_target_experiment(SPLIT_PRIMARY)
    all_rows.append(log_exp["log1p"])

    comparison = pd.DataFrame(all_rows).sort_values(["validation_split", "MAE"])
    comparison_path = ARTIFACTS_DIR / "phase2_model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    print(f"Saved {comparison_path}")

    ablation_rows = run_ablation(SPLIT_PRIMARY)
    ablation = pd.DataFrame(ablation_rows).sort_values("MAE")
    ablation_path = ARTIFACTS_DIR / "phase2_feature_ablation.csv"
    ablation.to_csv(ablation_path, index=False)
    print(f"Saved {ablation_path}")

    catboost_available = build_catboost_model("full") is not None
    meta = {
        "split_report": split_report,
        "log_target_experiment": log_exp,
        "catboost_available": catboost_available,
        "comparison": comparison,
        "ablation": ablation,
    }
    return comparison, ablation, meta
