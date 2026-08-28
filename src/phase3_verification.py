"""Phase 3 final candidate verification before Phase 4."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import ARTIFACTS_DIR, DEFAULT_HISTGB_PARAMS, SPLIT_PRIMARY, SPLIT_SENSITIVITY
from src.phase3_experiments import (
    DISTANCE_BUCKETS,
    DISTANCE_LABELS,
    _build_model,
    _fit_eval,
    _load_splits,
    _segment_breakdown,
)

REPORTS_DIR = ARTIFACTS_DIR / "phase3_verification"

BASELINE_CONFIGS = [
    {
        "name": "log1p_FULL",
        "feature_set": "FULL",
        "use_log_target": True,
        "params": dict(DEFAULT_HISTGB_PARAMS),
    },
    {
        "name": "normal_Q",
        "feature_set": "Q",
        "use_log_target": False,
        "params": dict(DEFAULT_HISTGB_PARAMS),
    },
    {
        "name": "normal_C4",
        "feature_set": "C4",
        "use_log_target": False,
        "params": dict(DEFAULT_HISTGB_PARAMS),
    },
]

LOG1P_Q_BASE = {
    "name": "log1p_Q",
    "feature_set": "Q",
    "use_log_target": True,
    "params": dict(DEFAULT_HISTGB_PARAMS),
}

DEPTH_TUNING_GRID = [
    {"max_depth": 5, "l2_regularization": 0.0},
    {"max_depth": 5, "l2_regularization": 0.1},
    {"max_depth": 6, "l2_regularization": 0.0},
    {"max_depth": 6, "l2_regularization": 0.1},
    {"max_depth": 8, "l2_regularization": 0.0},
    {"max_depth": 8, "l2_regularization": 0.1},
]


def _config_label(cfg: dict) -> str:
    return cfg["name"]


def _hyperparams_str(params: dict) -> str:
    keys = ("max_depth", "learning_rate", "max_iter", "l2_regularization", "random_state")
    return json.dumps({k: params[k] for k in keys if k in params}, sort_keys=True)


def _high_rate_stats(y_true: np.ndarray, y_pred: np.ndarray, va_df: pd.DataFrame) -> dict[str, float]:
    actual = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    res = actual - pred
    out: dict[str, float] = {
        "mean_residual": float(res.mean()),
        "high_rate_mean_residual": float(res[actual >= np.quantile(actual, 0.99)].mean()),
    }
    for label, mask in {
        "top_1_percent": actual >= np.quantile(actual, 0.99),
        "rate_over_5000": actual > 5000,
        "distance_over_2000": va_df["distance"].values > 2000,
    }.items():
        if mask.any():
            out[f"{label}_mae"] = float(np.abs(res[mask]).mean())
            out[f"{label}_mean_residual"] = float(res[mask].mean())
    long_mask = va_df["distance"].values >= 2000
    if long_mask.any():
        out["long_haul_mae"] = float(np.abs(res[long_mask]).mean())
        out["long_haul_mean_residual"] = float(res[long_mask].mean())
    return out


def _distance_bucket_mae(y_true, y_pred, va_df) -> dict[str, float]:
    frame = va_df.copy()
    frame["actual"] = np.asarray(y_true, dtype=float)
    frame["predicted"] = np.asarray(y_pred, dtype=float)
    frame["abs_error"] = np.abs(frame["actual"] - frame["predicted"])
    frame["residual"] = frame["actual"] - frame["predicted"]
    frame["dist_bucket"] = pd.cut(
        frame["distance"], bins=DISTANCE_BUCKETS, labels=DISTANCE_LABELS, include_lowest=True
    )
    out = {}
    for label in DISTANCE_LABELS:
        grp = frame[frame["dist_bucket"] == label]
        if len(grp):
            out[f"mae_{label}"] = float(grp["abs_error"].mean())
            out[f"residual_{label}"] = float(grp["residual"].mean())
            out[f"mean_actual_{label}"] = float(grp["actual"].mean())
            out[f"mean_predicted_{label}"] = float(grp["predicted"].mean())
    return out


def _evaluate_config(cfg: dict, splits: dict | None = None) -> dict[str, Any]:
    if splits is None:
        splits = _load_splits()
    name = _config_label(cfg)
    params = cfg.get("params", dict(DEFAULT_HISTGB_PARAMS))
    row: dict[str, Any] = {
        "model": "HistGradientBoostingRegressor",
        "config_name": name,
        "feature_set": cfg["feature_set"],
        "target_transform": "log1p" if cfg.get("use_log_target") else "normal",
        "hyperparameters": _hyperparams_str(params),
    }
    total_time = 0.0
    for split_name in (SPLIT_PRIMARY, SPLIT_SENSITIVITY):
        X_tr, y_tr, X_va, y_va, _, va_df = splits[split_name]
        model = _build_model(cfg["feature_set"], cfg.get("use_log_target", False), **params)
        metrics, elapsed, pred = _fit_eval(model, X_tr, y_tr, X_va, y_va)
        total_time += elapsed
        prefix = "primary" if split_name == SPLIT_PRIMARY else "sensitivity"
        row[f"{prefix}_mae"] = metrics["mae"]
        row[f"{prefix}_rmse"] = metrics["rmse"]
        row[f"{prefix}_r2"] = metrics["r2"]
        hr = _high_rate_stats(y_va, pred, va_df)
        row[f"{prefix}_mean_residual"] = hr["mean_residual"]
        row[f"{prefix}_high_rate_mean_residual"] = hr["high_rate_mean_residual"]
        if split_name == SPLIT_PRIMARY:
            row["long_haul_mae"] = hr.get("long_haul_mae")
            row["top_1_percent_mae"] = hr.get("top_1_percent_mae")
            row["rate_over_5000_mae"] = hr.get("rate_over_5000_mae")
            row["distance_over_2000_mae"] = hr.get("distance_over_2000_mae")
            row["long_haul_mean_residual"] = hr.get("long_haul_mean_residual")
            row["top_1_percent_mean_residual"] = hr.get("top_1_percent_mean_residual")
            row["rate_over_5000_mean_residual"] = hr.get("rate_over_5000_mean_residual")
            for k, v in _distance_bucket_mae(y_va, pred, va_df).items():
                row[f"primary_{k}"] = v
    row["training_time"] = round(total_time, 3)
    row["stability_gap"] = row["sensitivity_mae"] - row["primary_mae"]
    row["combined_score"] = row["primary_mae"] + 0.5 * max(row["stability_gap"], 0)
    return row


def run_experiment1() -> pd.DataFrame:
    """Verify log1p + Q against log1p FULL, Q normal, C4 normal."""
    print("[exp1] log1p + Q verification ...", flush=True)
    configs = [LOG1P_Q_BASE, *BASELINE_CONFIGS]
    rows = [_evaluate_config(c) for c in configs]
    return pd.DataFrame(rows)


def run_experiment2(exp1: pd.DataFrame) -> pd.DataFrame:
    """Small depth/l2 grid on log1p + Q if exp1 is promising."""
    log1p_q = exp1[exp1["config_name"] == "log1p_Q"].iloc[0]
    promising = (
        log1p_q["primary_mae"] <= exp1["primary_mae"].min() + 2.0
        and log1p_q["sensitivity_mae"] <= exp1["sensitivity_mae"].min() + 5.0
    )
    if not promising:
        print("[exp2] log1p+Q not promising enough — skipping depth tuning", flush=True)
        return pd.DataFrame()

    print("[exp2] depth/l2 tuning on log1p + Q ...", flush=True)
    splits = _load_splits()
    primary_rows = []
    for overrides in DEPTH_TUNING_GRID:
        params = {**DEFAULT_HISTGB_PARAMS, **overrides}
        cfg = {
            "name": f"log1p_Q_d{overrides['max_depth']}_l2{overrides['l2_regularization']}",
            "feature_set": "Q",
            "use_log_target": True,
            "params": params,
        }
        primary_rows.append(_evaluate_config(cfg, splits))

    primary_df = pd.DataFrame(primary_rows).sort_values("primary_mae")
    top = primary_df.head(3)
    print(f"[exp2] top primary configs:\n{top[['config_name','primary_mae']].to_string(index=False)}", flush=True)

    # Re-evaluate top 3 on sensitivity (already included in _evaluate_config)
    return primary_df


def run_experiment3(configs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Long-haul and high-rate diagnostics for top candidates."""
    print("[exp3] long-haul diagnostics ...", flush=True)
    splits = _load_splits()
    bucket_rows, high_rows = [], []

    for cfg in configs:
        name = _config_label(cfg)
        for split_name in (SPLIT_PRIMARY, SPLIT_SENSITIVITY):
            X_tr, y_tr, X_va, y_va, _, va_df = splits[split_name]
            model = _build_model(cfg["feature_set"], cfg.get("use_log_target", False), **cfg.get("params", {}))
            metrics, _, pred = _fit_eval(model, X_tr, y_tr, X_va, y_va)
            actual = y_va.values
            res = actual - pred

            for label in DISTANCE_LABELS:
                mask = pd.cut(
                    va_df["distance"], bins=DISTANCE_BUCKETS, labels=DISTANCE_LABELS, include_lowest=True
                ) == label
                if not mask.any():
                    continue
                bucket_rows.append({
                    "config_name": name,
                    "feature_set": cfg["feature_set"],
                    "target_transform": "log1p" if cfg.get("use_log_target") else "normal",
                    "validation_split": split_name,
                    "distance_bucket": label,
                    "count": int(mask.sum()),
                    "mae": float(np.abs(res[mask]).mean()),
                    "mean_actual": float(actual[mask].mean()),
                    "mean_predicted": float(pred[mask].mean()),
                    "mean_residual": float(res[mask].mean()),
                })

            hr = {"config_name": name, "validation_split": split_name, "overall_mae": metrics["mae"]}
            for seg, mask in {
                "top_1_percent": actual >= np.quantile(actual, 0.99),
                "rate_over_5000": actual > 5000,
                "distance_over_2000": va_df["distance"].values > 2000,
                "long_haul_2000plus": va_df["distance"].values >= 2000,
            }.items():
                if mask.any():
                    hr[f"{seg}_count"] = int(mask.sum())
                    hr[f"{seg}_mae"] = float(np.abs(res[mask]).mean())
                    hr[f"{seg}_mean_residual"] = float(res[mask].mean())
                    hr[f"{seg}_median_residual"] = float(np.median(res[mask]))
                    hr[f"{seg}_mean_actual"] = float(actual[mask].mean())
                    hr[f"{seg}_mean_predicted"] = float(pred[mask].mean())
            high_rows.append(hr)

    return pd.DataFrame(bucket_rows), pd.DataFrame(high_rows)


def run_experiment4(best_cfg: dict) -> tuple[pd.DataFrame, Path]:
    """Residual vs distance diagnostic — no prediction correction applied."""
    print(f"[exp4] residual vs distance for {_config_label(best_cfg)} ...", flush=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    splits = _load_splits()
    X_tr, y_tr, X_va, y_va, _, va_df = splits[SPLIT_PRIMARY]
    model = _build_model(
        best_cfg["feature_set"], best_cfg.get("use_log_target", False), **best_cfg.get("params", {})
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_va)
    frame = va_df.copy()
    frame["actual"] = y_va.values
    frame["predicted"] = pred
    frame["residual"] = frame["actual"] - frame["predicted"]
    frame["dist_bucket"] = pd.cut(
        frame["distance"], bins=DISTANCE_BUCKETS, labels=DISTANCE_LABELS, include_lowest=True
    )

    bucket_rows = []
    for label in DISTANCE_LABELS:
        grp = frame[frame["dist_bucket"] == label]
        bucket_rows.append({
            "distance_bucket": label,
            "count": len(grp),
            "mean_actual": float(grp["actual"].mean()),
            "mean_predicted": float(grp["predicted"].mean()),
            "mean_residual": float(grp["residual"].mean()),
            "median_residual": float(grp["residual"].median()),
            "mae": float(grp["residual"].abs().mean()),
        })

    # Linear residual correction feasibility (fit on train fold only — diagnostic)
    tr_frame = splits[SPLIT_PRIMARY][4].copy()
    tr_pred = model.predict(splits[SPLIT_PRIMARY][0])
    tr_resid = splits[SPLIT_PRIMARY][1].values - tr_pred
    coef = np.polyfit(tr_frame["distance"].values, tr_resid, deg=1)
    bucket_df = pd.DataFrame(bucket_rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(frame["distance"], frame["residual"], alpha=0.15, s=8, c="steelblue")
    x_line = np.linspace(frame["distance"].min(), frame["distance"].max(), 200)
    axes[0].plot(x_line, coef[0] * x_line + coef[1], color="crimson", lw=2, label="train-fit linear")
    axes[0].axhline(0, color="gray", ls="--", lw=1)
    axes[0].set_xlabel("Distance (miles)")
    axes[0].set_ylabel("Residual (actual - predicted)")
    axes[0].set_title(f"Residual vs Distance — {_config_label(best_cfg)} (primary val)")
    axes[0].legend()

    axes[1].bar(bucket_df["distance_bucket"].astype(str), bucket_df["mean_residual"], color="steelblue")
    axes[1].axhline(0, color="gray", ls="--", lw=1)
    axes[1].set_xlabel("Distance bucket")
    axes[1].set_ylabel("Mean residual")
    axes[1].set_title("Mean residual by distance bucket")
    plt.tight_layout()
    plot_path = REPORTS_DIR / f"residual_vs_distance_{_config_label(best_cfg)}.png"
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    bucket_df["train_linear_slope"] = coef[0]
    bucket_df["train_linear_intercept"] = coef[1]
    bucket_df["note"] = (
        "Positive residual = underprediction. Train-fold linear fit shown for diagnostic only; "
        "not applied to predictions."
    )
    return bucket_df, plot_path


def _select_top_configs(exp1: pd.DataFrame, exp2: pd.DataFrame) -> list[dict]:
    """Build config dicts for diagnostics from verification results."""
    configs = [LOG1P_Q_BASE, *BASELINE_CONFIGS]
    if not exp2.empty:
        for _, row in exp2.sort_values("combined_score").head(2).iterrows():
            params = json.loads(row["hyperparameters"])
            configs.append({
                "name": row["config_name"],
                "feature_set": "Q",
                "use_log_target": True,
                "params": params,
            })
    return configs


def build_final_candidates(exp1: pd.DataFrame, exp2: pd.DataFrame) -> pd.DataFrame:
    frames = [exp1]
    if not exp2.empty:
        frames.append(exp2)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("combined_score").reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def run_verification() -> dict[str, Any]:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    exp1 = run_experiment1()
    exp1.to_csv(REPORTS_DIR / "experiment1_log1p_Q.csv", index=False)

    exp2 = run_experiment2(exp1)
    if not exp2.empty:
        exp2.to_csv(REPORTS_DIR / "experiment2_depth_tuning.csv", index=False)

    diag_configs = _select_top_configs(exp1, exp2)
    bucket_df, high_df = run_experiment3(diag_configs)
    bucket_df.to_csv(REPORTS_DIR / "experiment3_distance_buckets.csv", index=False)
    high_df.to_csv(REPORTS_DIR / "experiment3_high_rate.csv", index=False)

    best = build_final_candidates(exp1, exp2).iloc[0]
    best_cfg = {
        "name": best["config_name"],
        "feature_set": best["feature_set"],
        "use_log_target": best["target_transform"] == "log1p",
        "params": json.loads(best["hyperparameters"]),
    }
    resid_df, plot_path = run_experiment4(best_cfg)
    resid_df.to_csv(REPORTS_DIR / "experiment4_residual_distance.csv", index=False)

    final = build_final_candidates(exp1, exp2)
    final.to_csv(ARTIFACTS_DIR / "phase3_final_candidates.csv", index=False)

    return {
        "experiment1": exp1,
        "experiment2": exp2,
        "distance_buckets": bucket_df,
        "high_rate": high_df,
        "residual_distance": resid_df,
        "residual_plot": plot_path,
        "final_candidates": final,
        "best_config": best_cfg,
    }
