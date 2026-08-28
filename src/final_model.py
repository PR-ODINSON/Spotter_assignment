"""Final model training and prediction generation (Phase 4)."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.baselines import Log1pTargetWrapper
from src.config import (
    ARTIFACTS_DIR,
    DATE_COLUMN,
    DECEMBER_CHART_PATH,
    FINAL_MODEL_FEATURE_SET,
    FINAL_PREDICTION_DIAGNOSTICS_PATH,
    FINAL_TRAIN_END_DATE,
    ID_COLUMN,
    LOCKED_HISTGB_PARAMS,
    RAW_FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRAIN_PATH,
    VALIDATION_HOLDOUT_PATH,
    VALIDATION_PREDICTIONS_PATH,
    VALIDATION_TEMPLATE_PATH,
)
from src.data import get_feature_matrix, load_train_data
from src.features import build_preprocessing_pipeline, get_feature_columns


def _histgb(**overrides) -> HistGradientBoostingRegressor:
    params = {**LOCKED_HISTGB_PARAMS, **overrides}
    for key in ("max_iter", "max_depth", "max_leaf_nodes", "min_samples_leaf", "random_state"):
        if key in params and params[key] is not None:
            params[key] = int(params[key])
    if "learning_rate" in params:
        params["learning_rate"] = float(params["learning_rate"])
    if "l2_regularization" in params:
        params["l2_regularization"] = float(params["l2_regularization"])
    return HistGradientBoostingRegressor(**params)


def build_final_model() -> Pipeline:
    """Locked HistGB + log1p + feature set Q."""
    est = _histgb()
    return Pipeline(
        steps=[
            ("features", build_preprocessing_pipeline(FINAL_MODEL_FEATURE_SET)),
            ("model", Log1pTargetWrapper(est)),
        ]
    )


def verify_locked_model_definition() -> dict[str, Any]:
    """Assert implementation matches the Phase 3.5 locked model."""
    numeric, categorical = get_feature_columns(FINAL_MODEL_FEATURE_SET)
    forbidden = {
        "market_index",
        "route",
        "pickup",
        "delivery",
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "haversine_miles",
        "distance_vs_haversine",
        "weight_per_mile",
        *["month", "day_of_week", "day_of_month", "week_of_year", "quarter"],
    }
    used = set(numeric) | set(categorical)
    leaked = sorted(used & forbidden)
    if leaked:
        raise ValueError(f"Locked feature set Q includes forbidden columns: {leaked}")

    expected_numeric = {
        "distance",
        "weight",
        "weight_is_missing",
        "log_distance",
        "distance_bin",
        "quote_signal",
    }
    expected_categorical = {"equipment"}
    if set(numeric) != expected_numeric:
        raise ValueError(f"Q numeric mismatch: got {numeric}, expected {sorted(expected_numeric)}")
    if set(categorical) != expected_categorical:
        raise ValueError(f"Q categorical mismatch: got {categorical}")

    return {
        "feature_set": FINAL_MODEL_FEATURE_SET,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "hyperparameters": LOCKED_HISTGB_PARAMS,
        "target_transform": "log1p -> expm1",
    }


def _city_coordinate_lookup(train_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Most frequent lat/lon per city from training data."""
    lookup: dict[str, tuple[float, float]] = {}
    for city_col, lat_col, lon_col in (
        ("pickup", "pickup_lat", "pickup_lon"),
        ("delivery", "delivery_lat", "delivery_lon"),
    ):
        mode = (
            train_df.groupby(city_col)[[lat_col, lon_col]]
            .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.median())
            .to_dict("index")
        )
        for city, coords in mode.items():
            lookup[str(city)] = (float(coords[lat_col]), float(coords[lon_col]))
    return lookup


def prepare_inference_matrix(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure raw feature columns exist for the engineering pipeline."""
    out = df.copy()
    if DATE_COLUMN in out.columns:
        out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN])

    coords = _city_coordinate_lookup(train_df)
    for col in RAW_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    for city_col, lat_col, lon_col in (
        ("pickup", "pickup_lat", "pickup_lon"),
        ("delivery", "delivery_lat", "delivery_lon"),
    ):
        missing = out[lat_col].isna() | out[lon_col].isna()
        if missing.any():
            mapped = out.loc[missing, city_col].astype(str).map(coords)
            out.loc[missing, lat_col] = mapped.map(lambda x: x[0] if isinstance(x, tuple) else np.nan)
            out.loc[missing, lon_col] = mapped.map(lambda x: x[1] if isinstance(x, tuple) else np.nan)

    return out[RAW_FEATURE_COLUMNS].copy()


def load_validation_holdout() -> pd.DataFrame:
    df = pd.read_csv(VALIDATION_HOLDOUT_PATH)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return df


def load_full_training_data() -> pd.DataFrame:
    df = load_train_data(TRAIN_PATH)
    end = pd.Timestamp(FINAL_TRAIN_END_DATE)
    df = df[df[DATE_COLUMN] <= end].copy()
    if df.empty:
        raise ValueError("Final training dataset is empty")
    return df


def train_final_model(train_df: pd.DataFrame) -> tuple[Pipeline, dict[str, Any]]:
    X = get_feature_matrix(train_df)
    y = train_df[TARGET_COLUMN]
    model = build_final_model()
    t0 = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - t0
    meta = {
        "training_rows": len(train_df),
        "train_date_min": str(train_df[DATE_COLUMN].min().date()),
        "train_date_max": str(train_df[DATE_COLUMN].max().date()),
        "target_mean": float(y.mean()),
        "target_median": float(y.median()),
        "training_time_seconds": round(elapsed, 3),
    }
    return model, meta


def predict_rates(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    pred = model.predict(X)
    if not np.all(np.isfinite(pred)):
        bad = int((~np.isfinite(pred)).sum())
        raise ValueError(f"Model produced {bad} non-finite predictions")
    if np.any(pred <= 0):
        bad = int((pred <= 0).sum())
        raise ValueError(f"Model produced {bad} non-positive predictions")
    return pred


def write_validation_predictions(
    model: Pipeline,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> pd.DataFrame:
    template = pd.read_csv(VALIDATION_TEMPLATE_PATH)
    X_val = prepare_inference_matrix(val_df, train_df)
    preds = predict_rates(model, X_val)
    pred_map = dict(zip(val_df[ID_COLUMN].astype(str), preds.astype(float)))
    output = template[[ID_COLUMN]].copy()
    output["predicted_rate"] = output[ID_COLUMN].astype(str).map(pred_map)
    if output["predicted_rate"].isna().any():
        missing = int(output["predicted_rate"].isna().sum())
        raise ValueError(f"Missing predictions for {missing} template load_ids")
    output.to_csv(VALIDATION_PREDICTIONS_PATH, index=False)
    return output


def write_december_predictions(model: Pipeline, train_df: pd.DataFrame) -> pd.DataFrame:
    december = pd.read_csv(DECEMBER_CHART_PATH)
    december[DATE_COLUMN] = pd.to_datetime(december[DATE_COLUMN])
    X_dec = prepare_inference_matrix(december, train_df)
    preds = predict_rates(model, X_dec)
    out = december.copy()
    out["predicted_rate"] = preds
    out.to_csv(DECEMBER_CHART_PATH, index=False)
    return out


def _quantile(series: pd.Series, q: float) -> float:
    return float(series.quantile(q))


def write_prediction_diagnostics(
    predictions: pd.Series,
    train_target: pd.Series,
) -> pd.DataFrame:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, series in (("predicted_rate", predictions), ("train_posted_rate", train_target)):
        rows.append(
            {
                "distribution": label,
                "count": int(series.count()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "q25": _quantile(series, 0.25),
                "q75": _quantile(series, 0.75),
                "q95": _quantile(series, 0.95),
                "q99": _quantile(series, 0.99),
            }
        )
    diag = pd.DataFrame(rows)
    diag.to_csv(FINAL_PREDICTION_DIAGNOSTICS_PATH, index=False)
    return diag


def validate_holdout_sanity(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict[str, Any]:
    train_cities = set(train_df["pickup"]).union(set(train_df["delivery"]))
    val_cities = set(val_df["pickup"]).union(set(val_df["delivery"]))
    unseen = sorted(val_cities - train_cities)
    return {
        "validation_rows": len(val_df),
        "validation_date_min": str(val_df[DATE_COLUMN].min().date()),
        "validation_date_max": str(val_df[DATE_COLUMN].max().date()),
        "unseen_cities_count": len(unseen),
        "unseen_cities": unseen[:20],
        "validation_missing_weight": int(val_df["weight"].isna().sum()),
        "validation_missing_market_index": int(val_df["market_index"].isna().sum()),
    }


def validate_submission_file() -> list[dict[str, Any]]:
    """Run strict checks on validation_predictions.csv."""
    path = VALIDATION_PREDICTIONS_PATH
    template = pd.read_csv(VALIDATION_TEMPLATE_PATH)
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "result": "PASS" if passed else "FAIL", "detail": detail})

    exists = path.is_file()
    add("1. file_exists", exists, str(path))
    if not exists:
        return checks

    df = pd.read_csv(path)
    add("2. row_count_12000", len(df) == 12_000, f"rows={len(df)}")
    add("3. column_count_2", df.shape[1] == 2, f"columns={list(df.columns)}")
    add(
        "4. columns_exact",
        list(df.columns) == ["load_id", "predicted_rate"],
        f"columns={list(df.columns)}",
    )
    add("5. load_id_unique", df["load_id"].is_unique, "")
    add("6. no_missing_load_id", not df["load_id"].isna().any(), "")
    add("7. no_missing_predicted_rate", not df["predicted_rate"].isna().any(), "")

    rates = pd.to_numeric(df["predicted_rate"], errors="coerce")
    add("8. predictions_numeric", rates.notna().all(), "")
    add("9. predictions_finite", np.isfinite(rates).all(), "")
    add("10. predictions_positive", (rates > 0).all(), f"min={rates.min() if rates.notna().any() else 'nan'}")

    template_ids = template["load_id"].astype(str).tolist()
    submitted_ids = df["load_id"].astype(str).tolist()
    add(
        "11. load_ids_match_template",
        set(template_ids) == set(submitted_ids),
        f"template={len(template_ids)}, submitted={len(submitted_ids)}",
    )
    add(
        "12. ordering_matches_template",
        submitted_ids == template_ids,
        "order mismatch" if submitted_ids != template_ids else "",
    )

    raw = path.read_text(encoding="utf-8").splitlines()
    header_ok = raw[0].strip() == "load_id,predicted_rate" if raw else False
    add("13. no_index_column", header_ok and "Unnamed" not in raw[0], raw[0] if raw else "")
    add(
        "14. no_target_column",
        TARGET_COLUMN not in df.columns,
        f"columns={list(df.columns)}",
    )
    return checks


def validate_december_file() -> list[dict[str, Any]]:
    df = pd.read_csv(DECEMBER_CHART_PATH)
    checks = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "result": "PASS" if passed else "FAIL", "detail": detail})

    add("december_rows_31", len(df) == 31, f"rows={len(df)}")
    dates = pd.to_datetime(df["date"], errors="coerce")
    add(
        "december_dates",
        dates.notna().all()
        and dates.min().date().isoformat() == "2025-12-01"
        and dates.max().date().isoformat() == "2025-12-31",
        "",
    )
    add("december_pickup", df["pickup"].eq("Lexington").all(), "")
    add("december_delivery", df["delivery"].eq("Fort Wayne").all(), "")
    add("december_distance", np.isclose(df["distance"], 360.0).all(), "")
    add("december_equipment", df["equipment"].eq("Dry Van").all(), "")
    add("december_weight", np.isclose(df["weight"], 32000.0).all(), "")
    rates = pd.to_numeric(df["predicted_rate"], errors="coerce")
    add(
        "december_predictions_valid",
        rates.notna().all() and np.isfinite(rates).all() and (rates > 0).all(),
        f"min={rates.min()}, max={rates.max()}",
    )
    return checks


def run_final_training_pipeline() -> dict[str, Any]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_spec = verify_locked_model_definition()
    train_df = load_full_training_data()
    val_df = load_validation_holdout()
    holdout_sanity = validate_holdout_sanity(train_df, val_df)

    model, train_meta = train_final_model(train_df)
    predictions = write_validation_predictions(model, train_df, val_df)
    december = write_december_predictions(model, train_df)
    diagnostics = write_prediction_diagnostics(
        predictions["predicted_rate"],
        train_df[TARGET_COLUMN],
    )
    submission_checks = validate_submission_file()
    december_checks = validate_december_file()

    rates = predictions["predicted_rate"].astype(float)
    invalid = int((~np.isfinite(rates) | (rates <= 0)).sum())

    return {
        "model_spec": model_spec,
        "train_meta": train_meta,
        "holdout_sanity": holdout_sanity,
        "prediction_summary": {
            "count": len(rates),
            "mean": float(rates.mean()),
            "median": float(rates.median()),
            "min": float(rates.min()),
            "max": float(rates.max()),
            "invalid_count": invalid,
        },
        "diagnostics_path": str(FINAL_PREDICTION_DIAGNOSTICS_PATH),
        "validation_predictions_path": str(VALIDATION_PREDICTIONS_PATH),
        "december_predictions_path": str(DECEMBER_CHART_PATH),
        "submission_checks": submission_checks,
        "december_checks": december_checks,
        "december_prediction_range": {
            "min": float(december["predicted_rate"].min()),
            "max": float(december["predicted_rate"].max()),
        },
    }
