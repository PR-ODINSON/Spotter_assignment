"""Phase 3 experiments: log-target, market_index, feature sets, tuning, diagnostics."""

from __future__ import annotations

import json
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from src.baselines import Log1pTargetWrapper, build_catboost_model
from src.config import (
    ARTIFACTS_DIR,
    DEFAULT_HISTGB_PARAMS,
    MARKET_INVESTIGATION_SETS,
    PHASE3_FEATURE_SETS,
    RANDOM_SEED,
    SPLIT_PRIMARY,
    SPLIT_SENSITIVITY,
)
from src.data import get_split, load_train_data, split_xy
from src.features import build_preprocessing_pipeline
from src.metrics import evaluate_predictions

DISTANCE_BUCKETS = [0, 300, 600, 1200, 2000, np.inf]
DISTANCE_LABELS = ["0-300", "300-600", "600-1200", "1200-2000", "2000+"]


def _histgb(**overrides) -> HistGradientBoostingRegressor:
    params = {**DEFAULT_HISTGB_PARAMS, **overrides}
    for key in ("max_iter", "max_depth", "max_leaf_nodes", "min_samples_leaf", "random_state"):
        if key in params and params[key] is not None:
            params[key] = int(params[key])
    if "learning_rate" in params:
        params["learning_rate"] = float(params["learning_rate"])
    if "l2_regularization" in params:
        params["l2_regularization"] = float(params["l2_regularization"])
    return HistGradientBoostingRegressor(**params)


def _build_model(
    feature_set: str,
    use_log_target: bool = False,
    **histgb_params,
) -> Pipeline:
    est = _histgb(**histgb_params)
    model_step = Log1pTargetWrapper(est) if use_log_target else clone(est)
    return Pipeline(
        steps=[
            ("features", build_preprocessing_pipeline(feature_set)),
            ("model", model_step),
        ]
    )


def _load_splits() -> dict[str, tuple]:
    df = load_train_data()
    out = {}
    for name in (SPLIT_PRIMARY, SPLIT_SENSITIVITY):
        tr, va = get_split(df, name)
        out[name] = (*split_xy(tr), *split_xy(va), tr, va)
    return out


def _fit_eval(model, X_tr, y_tr, X_va, y_va) -> tuple[dict, float, np.ndarray]:
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    elapsed = time.perf_counter() - t0
    pred = model.predict(X_va)
    return evaluate_predictions(y_va, pred), elapsed, pred


def _row(
    experiment: str,
    feature_set: str,
    split_name: str,
    metrics: dict,
    elapsed: float,
    use_log: bool = False,
    model: str = "histgb",
    **extra,
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "model": "histgb_log1p" if use_log else model,
        "feature_set": feature_set,
        "target_transform": "log1p" if use_log else "normal",
        "validation_split": split_name,
        "MAE": metrics["mae"],
        "RMSE": metrics["rmse"],
        "R2": metrics["r2"],
        "MAPE": metrics["mape"],
        "training_time_seconds": round(elapsed, 3),
        **extra,
    }


def _eval_both(
    feature_set: str,
    experiment: str,
    use_log: bool = False,
    **params,
) -> list[dict]:
    splits = _load_splits()
    rows = []
    for split_name in (SPLIT_PRIMARY, SPLIT_SENSITIVITY):
        X_tr, y_tr, X_va, y_va, _, _ = splits[split_name]
        m = _build_model(feature_set, use_log, **params)
        metrics, elapsed, _ = _fit_eval(m, X_tr, y_tr, X_va, y_va)
        rows.append(_row(experiment, feature_set, split_name, metrics, elapsed, use_log, **{
            f"param_{k}": v for k, v in {**DEFAULT_HISTGB_PARAMS, **params}.items()
        }))
    return rows


def run_log_target_study() -> pd.DataFrame:
    rows = []
    for use_log in (False, True):
        print(f"[log-target] {'log1p' if use_log else 'normal'} ...", flush=True)
        rows.extend(_eval_both("FULL", "log_target", use_log))
    return pd.DataFrame(rows)


def run_market_index_study() -> tuple[pd.DataFrame, pd.DataFrame]:
    splits = _load_splits()
    rows, seg_rows = [], []
    for name in MARKET_INVESTIGATION_SETS:
        print(f"[market] {name} ...", flush=True)
        for split_name in (SPLIT_PRIMARY, SPLIT_SENSITIVITY):
            X_tr, y_tr, X_va, y_va, tr_df, va_df = splits[split_name]
            m = _build_model(name)
            metrics, elapsed, pred = _fit_eval(m, X_tr, y_tr, X_va, y_va)
            rows.append(_row("market_index", name, split_name, metrics, elapsed))
            if split_name == SPLIT_PRIMARY:
                seg_rows.extend(_segment_breakdown(va_df, y_va, pred, f"market_{name}"))
    return pd.DataFrame(rows), pd.DataFrame(seg_rows)


def run_feature_set_study() -> pd.DataFrame:
    rows = []
    for fs in PHASE3_FEATURE_SETS:
        print(f"[feature_set] {fs} ...", flush=True)
        rows.extend(_eval_both(fs, "feature_set"))
    return pd.DataFrame(rows)


def run_histgb_tuning() -> pd.DataFrame:
    overrides_list = [
        {},
        {"learning_rate": 0.06},
        {"learning_rate": 0.10},
        {"max_iter": 250},
        {"max_iter": 400},
        {"max_depth": 6},
        {"max_depth": 10},
        {"max_leaf_nodes": 63},
        {"min_samples_leaf": 20},
        {"min_samples_leaf": 50},
        {"l2_regularization": 0.1},
        {"l2_regularization": 1.0},
        {"learning_rate": 0.06, "max_depth": 6},
        {"learning_rate": 0.10, "max_iter": 400},
        {"max_depth": 6, "l2_regularization": 0.1},
    ]
    rows = []
    splits = _load_splits()
    X_tr, y_tr, X_va, y_va, _, _ = splits[SPLIT_PRIMARY]

    for overrides in overrides_list:
        label = json.dumps(overrides) if overrides else "baseline"
        print(f"[tuning] {label} ...", flush=True)
        params = {**DEFAULT_HISTGB_PARAMS, **overrides}
        m = _build_model("C4", False, **overrides)
        metrics, elapsed, _ = _fit_eval(m, X_tr, y_tr, X_va, y_va)
        rows.append(_row("histgb_tuning", "C4", SPLIT_PRIMARY, metrics, elapsed, **{
            f"param_{k}": v for k, v in params.items()
        }))

    primary_df = pd.DataFrame(rows).sort_values("MAE")
    top3 = primary_df.head(3)
    X_tr, y_tr, X_va, y_va, _, _ = splits[SPLIT_SENSITIVITY]
    valid_param_keys = set(DEFAULT_HISTGB_PARAMS) | {"max_leaf_nodes", "min_samples_leaf", "l2_regularization"}
    for _, cfg in top3.iterrows():
        overrides = {
            k.replace("param_", ""): cfg[k]
            for k in cfg.index
            if str(k).startswith("param_") and pd.notna(cfg[k]) and k.replace("param_", "") in valid_param_keys
        }
        m = _build_model("C4", False, **overrides)
        metrics, elapsed, _ = _fit_eval(m, X_tr, y_tr, X_va, y_va)
        extra = {k: cfg[k] for k in cfg.index if str(k).startswith("param_") and pd.notna(cfg[k])}
        rows.append(_row("histgb_tuning", "C4", SPLIT_SENSITIVITY, metrics, elapsed, **extra))
    return pd.DataFrame(rows)


def run_optional_models() -> tuple[pd.DataFrame, bool]:
    splits = _load_splits()
    rows = []
    catboost_ok = build_catboost_model("R") is not None

    if catboost_ok:
        for fs in ("R", "FULL"):
            for use_log in (False, True):
                cb = build_catboost_model(fs)
                if use_log:
                    cb = Pipeline([
                        *cb.steps[:-1],
                        ("model", Log1pTargetWrapper(clone(cb.steps[-1][1]))),
                    ])
                for split_name in (SPLIT_PRIMARY, SPLIT_SENSITIVITY):
                    X_tr, y_tr, X_va, y_va, _, _ = splits[split_name]
                    metrics, elapsed, _ = _fit_eval(cb, X_tr, y_tr, X_va, y_va)
                    rows.append(_row("catboost", fs, split_name, metrics, elapsed, use_log, model="catboost"))

    for lib, mod, cls in [("xgboost", "xgboost", "XGBRegressor"), ("lightgbm", "lightgbm", "LGBMRegressor")]:
        try:
            Reg = getattr(__import__(mod, fromlist=[cls]), cls)
        except ImportError:
            continue
        est = Reg(n_estimators=300, max_depth=8, learning_rate=0.08, random_state=RANDOM_SEED, verbosity=0)
        for split_name in (SPLIT_PRIMARY, SPLIT_SENSITIVITY):
            X_tr, y_tr, X_va, y_va, _, _ = splits[split_name]
            m = Pipeline([("features", build_preprocessing_pipeline("C4")), ("model", est)])
            metrics, elapsed, _ = _fit_eval(m, X_tr, y_tr, X_va, y_va)
            rows.append(_row(lib, "C4", split_name, metrics, elapsed, model=lib))

    return pd.DataFrame(rows), catboost_ok


def _segment_breakdown(val_df, y_true, y_pred, experiment) -> list[dict]:
    f = val_df.copy()
    f["actual"] = y_true.values
    f["predicted"] = y_pred
    f["abs_error"] = np.abs(f["actual"] - f["predicted"])
    f["residual"] = f["actual"] - f["predicted"]
    f["dist_bucket"] = pd.cut(f["distance"], bins=DISTANCE_BUCKETS, labels=DISTANCE_LABELS, include_lowest=True)
    f["month"] = pd.to_datetime(f["date"]).dt.month
    f["weight_status"] = np.where(f["weight"].isna() | (f["weight"] < 0), "missing_or_negative", "present")
    f["mi_regime"] = pd.qcut(f["market_index"].rank(method="first"), 3, labels=["low", "mid", "high"])

    rows = []
    for dim, col in [
        ("distance_bucket", "dist_bucket"),
        ("equipment", "equipment"),
        ("month", "month"),
        ("weight_status", "weight_status"),
        ("market_index_regime", "mi_regime"),
    ]:
        for key, grp in f.groupby(col, observed=True):
            rows.append({
                "experiment": experiment,
                "segment_dimension": dim,
                "segment": str(key),
                "count": len(grp),
                "MAE": float(grp["abs_error"].mean()),
                "mean_actual": float(grp["actual"].mean()),
                "mean_predicted": float(grp["predicted"].mean()),
                "mean_residual": float(grp["residual"].mean()),
                "median_abs_error": float(grp["abs_error"].median()),
            })
    return rows


def run_diagnostics(configs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    splits = _load_splits()
    seg_all, high_all = [], []

    for cfg in configs:
        name = cfg["name"]
        print(f"[diagnostics] {name} ...", flush=True)
        for split_name in (SPLIT_PRIMARY, SPLIT_SENSITIVITY):
            X_tr, y_tr, X_va, y_va, tr_df, va_df = splits[split_name]
            m = _build_model(cfg["feature_set"], cfg.get("use_log_target", False), **cfg.get("params", {}))
            metrics, _, pred = _fit_eval(m, X_tr, y_tr, X_va, y_va)
            for row in _segment_breakdown(va_df, y_va, pred, name):
                row["model"] = name
                row["validation_split"] = split_name
                seg_all.append(row)
            actual = y_va.values
            res = actual - pred
            hr = {"model": name, "validation_split": split_name, "overall_mae": metrics["mae"],
                  "overall_mean_residual": float(res.mean())}
            for label, mask in {
                "top_1pct": actual >= np.quantile(actual, 0.99),
                "top_5pct": actual >= np.quantile(actual, 0.95),
                "distance_gt_2000": va_df["distance"].values > 2000,
                "rate_gt_5000": actual > 5000,
            }.items():
                if not mask.any():
                    continue
                hr[f"{label}_count"] = int(mask.sum())
                hr[f"{label}_mae"] = float(np.abs(res[mask]).mean())
                hr[f"{label}_mean_residual"] = float(res[mask].mean())
                hr[f"{label}_median_residual"] = float(np.median(res[mask]))
                hr[f"{label}_mean_actual"] = float(actual[mask].mean())
                hr[f"{label}_mean_predicted"] = float(pred[mask].mean())
            high_all.append(hr)

    resid_cfg = configs[0]
    X_tr, y_tr, X_va, y_va, _, va_df = splits[SPLIT_PRIMARY]
    m = _build_model(resid_cfg["feature_set"], resid_cfg.get("use_log_target", False), **resid_cfg.get("params", {}))
    m.fit(X_tr, y_tr)
    pred = m.predict(X_va)
    frame = va_df.copy()
    frame["residual"] = y_va.values - pred
    resid_rows = []
    for col in ["distance", "weight", "market_index", "quote_signal"]:
        resid_rows.append({
            "feature": col,
            "correlation_with_residual": float(frame[col].corr(frame["residual"])),
            "mean_residual": float(frame["residual"].mean()),
            "count": len(frame),
        })
    frame["month"] = pd.to_datetime(frame["date"]).dt.month
    for eq, grp in frame.groupby("equipment"):
        resid_rows.append({
            "feature": "equipment",
            "segment": eq,
            "mean_residual": float(grp["residual"].mean()),
            "mae": float(grp["residual"].abs().mean()),
            "count": len(grp),
        })
    return pd.DataFrame(seg_all), pd.DataFrame(high_all), pd.DataFrame(resid_rows)


def rank_candidates(results: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["experiment", "feature_set", "target_transform", "model"]
    for c in key_cols:
        if c not in results.columns:
            results[c] = ""
    p = results[results["validation_split"] == SPLIT_PRIMARY].copy()
    s = results[results["validation_split"] == SPLIT_SENSITIVITY].copy()
    merge_on = [c for c in key_cols if c in p.columns and c in s.columns]
    m = p.merge(s, on=merge_on, suffixes=("_primary", "_sensitivity"), how="inner")
    m["stability_gap"] = m["MAE_sensitivity"] - m["MAE_primary"]
    m["combined_score"] = m["MAE_primary"] + 0.5 * m["stability_gap"].clip(lower=0)
    return m.sort_values("combined_score")


def run_phase3() -> dict[str, Any]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    log_df = run_log_target_study()
    market_df, _ = run_market_index_study()
    feature_df = run_feature_set_study()
    feature_df.to_csv(ARTIFACTS_DIR / "phase3_feature_sets.csv", index=False)

    tuning_df = run_histgb_tuning()
    tuning_df.to_csv(ARTIFACTS_DIR / "phase3_histgb_tuning.csv", index=False)

    optional_df, catboost_ok = run_optional_models()

    best_tune = tuning_df[tuning_df["validation_split"] == SPLIT_PRIMARY].sort_values("MAE").iloc[0]
    tune_params = {k.replace("param_", ""): best_tune[k] for k in best_tune.index if str(k).startswith("param_")}

    diag_configs = [
        {"name": "histgb_C4", "feature_set": "C4", "use_log_target": False, "params": {}},
        {"name": "histgb_FULL", "feature_set": "FULL", "use_log_target": False, "params": {}},
        {"name": "histgb_C4_log1p", "feature_set": "C4", "use_log_target": True, "params": {}},
    ]
    seg_df, high_df, resid_df = run_diagnostics(diag_configs)
    seg_df.to_csv(ARTIFACTS_DIR / "phase3_segment_errors.csv", index=False)

    all_results = pd.concat([log_df, market_df, feature_df, tuning_df, optional_df], ignore_index=True)
    candidates = rank_candidates(all_results[all_results["model"].str.contains("histgb", na=False)])
    candidates.to_csv(ARTIFACTS_DIR / "phase3_candidates.csv", index=False)

    return {
        "log_target": log_df,
        "market_index": market_df,
        "feature_sets": feature_df,
        "tuning": tuning_df,
        "optional": optional_df,
        "catboost_available": catboost_ok,
        "segments": seg_df,
        "high_rate": high_df,
        "residuals": resid_df,
        "candidates": candidates,
        "tune_params": tune_params,
    }
