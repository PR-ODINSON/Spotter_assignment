"""Phase 0 data audit — read-only profiling."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
train = pd.read_csv(BASE / "train-test.csv")
val = pd.read_csv(BASE / "validation.csv")
template = pd.read_csv(BASE / "validation-predictions-template.csv")
december = pd.read_csv(BASE / "december-chart-inputs.csv")

results: dict = {}

results["train_shape"] = list(train.shape)
results["val_shape"] = list(val.shape)
results["template_shape"] = list(template.shape)
results["december_shape"] = list(december.shape)

results["train_columns"] = list(train.columns)
results["val_columns"] = list(val.columns)
results["template_columns"] = list(template.columns)
results["december_columns"] = list(december.columns)

results["target_column"] = "posted_rate" if "posted_rate" in train.columns else None
results["target_in_val"] = "posted_rate" in val.columns

results["train_load_id_unique"] = int(train["load_id"].nunique())
results["val_load_id_unique"] = int(val["load_id"].nunique())
results["train_load_id_duplicates"] = int(train["load_id"].duplicated().sum())
results["val_load_id_duplicates"] = int(val["load_id"].duplicated().sum())
results["id_overlap"] = int(len(set(train["load_id"]) & set(val["load_id"])))

results["train_dtypes"] = {c: str(t) for c, t in train.dtypes.items()}
results["val_dtypes"] = {c: str(t) for c, t in val.dtypes.items()}

results["train_missing"] = {k: int(v) for k, v in train.isna().sum().items()}
results["val_missing"] = {k: int(v) for k, v in val.isna().sum().items()}
results["template_missing"] = {k: int(v) for k, v in template.isna().sum().items()}

results["train_full_row_duplicates"] = int(train.duplicated().sum())
results["val_full_row_duplicates"] = int(val.duplicated().sum())
feat_cols = [c for c in train.columns if c not in ["load_id", "posted_rate"]]
results["train_feature_row_duplicates"] = int(train[feat_cols].duplicated().sum())
val_feat_cols = [c for c in val.columns if c != "load_id"]
results["val_feature_row_duplicates"] = int(val[val_feat_cols].duplicated().sum())

train["date"] = pd.to_datetime(train["date"])
val["date"] = pd.to_datetime(val["date"])
results["train_date_min"] = str(train["date"].min())
results["train_date_max"] = str(train["date"].max())
results["val_date_min"] = str(val["date"].min())
results["val_date_max"] = str(val["date"].max())
results["train_date_unique"] = int(train["date"].nunique())
results["val_date_unique"] = int(val["date"].nunique())
results["train_month_counts"] = {
    str(k): int(v) for k, v in train["date"].dt.to_period("M").value_counts().sort_index().items()
}
results["val_month_counts"] = {
    str(k): int(v) for k, v in val["date"].dt.to_period("M").value_counts().sort_index().items()
}

cat_cols = ["pickup", "delivery", "equipment"]
results["categorical"] = {}
for col in cat_cols:
    results["categorical"][col] = {
        "train_nunique": int(train[col].nunique()),
        "val_nunique": int(val[col].nunique()),
        "train_top5": {str(k): int(v) for k, v in train[col].value_counts().head(5).items()},
        "val_top5": {str(k): int(v) for k, v in val[col].value_counts().head(5).items()},
    }

num_cols = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "posted_rate",
]
results["numeric_summary"] = {}
for col in num_cols:
    entry: dict = {}
    if col in train.columns:
        entry["train_min"] = float(train[col].min())
        entry["train_max"] = float(train[col].max())
        entry["train_mean"] = float(train[col].mean())
        entry["train_std"] = float(train[col].std())
    if col in val.columns:
        entry["val_min"] = float(val[col].min())
        entry["val_max"] = float(val[col].max())
        entry["val_mean"] = float(val[col].mean())
        entry["val_std"] = float(val[col].std())
    if entry:
        results["numeric_summary"][col] = entry

results["target_stats"] = {k: float(v) for k, v in train["posted_rate"].describe().items()}

results["target_correlations"] = {}
for col in train.columns:
    if col not in ["posted_rate", "load_id"] and pd.api.types.is_numeric_dtype(train[col]):
        results["target_correlations"][col] = float(train[col].corr(train["posted_rate"]))

results["ks_tests"] = {}
shared_num = [c for c in num_cols if c in train.columns and c in val.columns and c != "posted_rate"]
for col in shared_num:
    ks, p = stats.ks_2samp(train[col], val[col])
    results["ks_tests"][col] = {"ks_stat": float(ks), "p_value": float(p)}

results["train_id_prefix"] = {str(k): int(v) for k, v in train["load_id"].str[:3].value_counts().items()}
results["val_id_prefix"] = {str(k): int(v) for k, v in val["load_id"].str[:3].value_counts().items()}

results["template_load_ids_match_val"] = bool((template["load_id"] == val["load_id"]).all())
results["template_empty_predictions"] = int(template["predicted_rate"].isna().sum())

train["route"] = train["pickup"] + "|" + train["delivery"]
val["route"] = val["pickup"] + "|" + val["delivery"]
results["train_routes_nunique"] = int(train["route"].nunique())
results["val_routes_nunique"] = int(val["route"].nunique())

pickup_coord = train.groupby("pickup")[["pickup_lat", "pickup_lon"]].nunique()
results["pickup_multiple_coords"] = int((pickup_coord.max(axis=1) > 1).sum())
delivery_coord = train.groupby("delivery")[["delivery_lat", "delivery_lon"]].nunique()
results["delivery_multiple_coords"] = int((delivery_coord.max(axis=1) > 1).sum())

results["max_same_route_date_train"] = int(train.groupby(["route", "date"]).size().max())
results["max_same_route_date_val"] = int(val.groupby(["route", "date"]).size().max())

monthly = train.groupby(train["date"].dt.to_period("M"))["posted_rate"].agg(["mean", "std", "count"])
results["monthly_target"] = {
    str(k): {"mean": float(v["mean"]), "std": float(v["std"]), "count": int(v["count"])}
    for k, v in monthly.iterrows()
}

results["equipment_train"] = {str(k): int(v) for k, v in train["equipment"].value_counts().items()}
results["equipment_val"] = {str(k): int(v) for k, v in val["equipment"].value_counts().items()}

results["train_nonpositive"] = {}
results["val_nonpositive"] = {}
for col in shared_num + ["posted_rate"]:
    if col in train.columns:
        results["train_nonpositive"][col] = int((train[col] <= 0).sum())
    if col in val.columns:
        results["val_nonpositive"][col] = int((val[col] <= 0).sum())

# Distance vs route consistency
route_dist = train.groupby("route")["distance"].nunique()
results["routes_with_multiple_distances_train"] = int((route_dist > 1).sum())
route_dist_val = val.groupby("route")["distance"].nunique()
results["routes_with_multiple_distances_val"] = int((route_dist_val > 1).sum())

# Cities in val not in train
train_pickups = set(train["pickup"])
train_deliveries = set(train["delivery"])
results["val_pickups_not_in_train"] = int((~val["pickup"].isin(train_pickups)).sum())
results["val_deliveries_not_in_train"] = int((~val["delivery"].isin(train_deliveries)).sum())
results["val_pickup_cities_not_in_train"] = sorted(set(val["pickup"]) - train_pickups)
results["val_delivery_cities_not_in_train"] = sorted(set(val["delivery"]) - train_deliveries)

print(json.dumps(results, indent=2))
