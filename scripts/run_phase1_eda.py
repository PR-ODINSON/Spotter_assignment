"""Phase 1 EDA: verify Phase 0, analyze data, save plots and stats JSON."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / "reports" / "eda"
REPORTS.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(BASE / "train-test.csv")
val = pd.read_csv(BASE / "validation.csv")
train["date"] = pd.to_datetime(train["date"])
val["date"] = pd.to_datetime(val["date"])

out: dict = {"phase0_verification": {}, "target": {}, "numerical": {}, "categorical": {},
             "geographic": {}, "temporal": {}, "drift": {}, "data_quality": {},
             "leakage": {}, "preprocessing": {}}


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


# ── Part 1: Phase 0 verification ──────────────────────────────────────────
p0 = out["phase0_verification"]
p0["train_shape"] = list(train.shape)
p0["val_shape"] = list(val.shape)
p0["train_columns"] = list(train.columns)
p0["val_columns"] = list(val.columns)
p0["target"] = "posted_rate"
p0["target_in_val"] = "posted_rate" in val.columns
p0["train_id_unique"] = int(train["load_id"].nunique())
p0["val_id_unique"] = int(val["load_id"].nunique())
p0["id_overlap"] = int(len(set(train["load_id"]) & set(val["load_id"])))
p0["missing_train"] = {k: int(v) for k, v in train.isna().sum().items()}
p0["missing_val"] = {k: int(v) for k, v in val.isna().sum().items()}
p0["negative_weight_train"] = int((train["weight"] < 0).sum())
p0["negative_weight_val"] = int((val["weight"] < 0).sum())
p0["train_date_range"] = [str(train["date"].min()), str(train["date"].max())]
p0["val_date_range"] = [str(val["date"].min()), str(val["date"].max())]
p0["duplicate_rows_train"] = int(train.duplicated().sum())
p0["duplicate_rows_val"] = int(val.duplicated().sum())
train_cities = set(train["pickup"]) | set(train["delivery"])
val_cities = set(val["pickup"]) | set(val["delivery"])
unseen = sorted(val_cities - train_cities)
p0["unseen_cities"] = unseen
p0["unseen_pickup_rows"] = int((~val["pickup"].isin(train_cities)).sum())
p0["unseen_delivery_rows"] = int((~val["delivery"].isin(train_cities)).sum())
p0["pearson_distance_target"] = float(train["distance"].corr(train["posted_rate"]))

# ── Part 2: Target EDA ────────────────────────────────────────────────────
y = train["posted_rate"]
t = out["target"]
t["count"] = int(y.count())
t["mean"] = float(y.mean())
t["median"] = float(y.median())
t["std"] = float(y.std())
t["min"] = float(y.min())
t["max"] = float(y.max())
t["q25"] = float(y.quantile(0.25))
t["q75"] = float(y.quantile(0.75))
t["q95"] = float(y.quantile(0.95))
t["q99"] = float(y.quantile(0.99))
t["iqr"] = float(y.quantile(0.75) - y.quantile(0.25))
t["skewness"] = float(y.skew())
t["kurtosis"] = float(y.kurtosis())
t["above_q99"] = int((y > y.quantile(0.99)).sum())
t["below_q01"] = int((y < y.quantile(0.01)).sum())
# Shapiro on sample (5000 max)
sample = y.sample(min(5000, len(y)), random_state=42)
t["shapiro_p_sample5000"] = float(stats.shapiro(sample)[1])
t["distribution_assessment"] = "right_skewed_heavy_tailed"

monthly = train.groupby(train["date"].dt.to_period("M"))["posted_rate"]
t["monthly"] = {}
for period, grp in monthly:
    t["monthly"][str(period)] = {
        "mean": float(grp.mean()), "median": float(grp.median()),
        "std": float(grp.std()), "q25": float(grp.quantile(0.25)),
        "q75": float(grp.quantile(0.75)), "count": int(len(grp)),
    }

# Target plots
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(y, bins=80, color="#064A56", edgecolor="white", alpha=0.85)
ax.set_xlabel("posted_rate ($)")
ax.set_ylabel("Count")
ax.set_title("Distribution of posted_rate (train)")
fig.tight_layout()
fig.savefig(REPORTS / "01_posted_rate_distribution.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ax.boxplot(y, vert=True)
ax.set_ylabel("posted_rate ($)")
ax.set_title("posted_rate Boxplot (train)")
fig.tight_layout()
fig.savefig(REPORTS / "02_posted_rate_boxplot.png", dpi=150)
plt.close(fig)

months = sorted(t["monthly"].keys())
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(months, [t["monthly"][m]["mean"] for m in months], marker="o", label="mean")
ax.plot(months, [t["monthly"][m]["median"] for m in months], marker="s", label="median")
ax.set_xticklabels(months, rotation=45, ha="right")
ax.set_ylabel("posted_rate ($)")
ax.set_title("posted_rate by Month (train)")
ax.legend()
fig.tight_layout()
fig.savefig(REPORTS / "03_posted_rate_by_month.png", dpi=150)
plt.close(fig)

# ── Part 3: Numerical features ────────────────────────────────────────────
num_cols = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
            "distance", "weight", "market_index", "quote_signal"]
out["numerical"] = {}
for col in num_cols:
    s = train[col]
    entry = {
        "missing_pct_train": float(train[col].isna().mean() * 100),
        "missing_pct_val": float(val[col].isna().mean() * 100),
        "nunique_train": int(s.nunique()),
        "min": float(s.min()) if s.notna().any() else None,
        "max": float(s.max()) if s.notna().any() else None,
        "mean": float(s.mean()),
        "median": float(s.median()),
        "q25": float(s.quantile(0.25)),
        "q75": float(s.quantile(0.75)),
        "pearson_with_target": float(s.corr(train["posted_rate"])) if s.notna().sum() > 2 else None,
        "spearman_with_target": float(s.corr(train["posted_rate"], method="spearman")) if s.notna().sum() > 2 else None,
    }
    if col == "weight":
        entry["negative_count"] = int((s < 0).sum())
        entry["zero_count"] = int((s == 0).sum())
        entry["positive_count"] = int((s > 0).sum())
        entry["missing_count"] = int(s.isna().sum())
        neg = train[train["weight"] < 0]
        entry["negative_unique_values"] = sorted(neg["weight"].unique().tolist())[:20]
        entry["negative_equipment"] = {str(k): int(v) for k, v in neg["equipment"].value_counts().items()}
        entry["negative_abs_corr_distance"] = float(neg["weight"].abs().corr(neg["distance"]))
    if col == "market_index":
        entry["train_mean"] = float(train["market_index"].mean())
        entry["val_mean"] = float(val["market_index"].mean())
    out["numerical"][col] = entry

# Distance analysis
train["rpm"] = train["posted_rate"] / train["distance"]
train["hav"] = haversine(train["pickup_lat"], train["pickup_lon"],
                         train["delivery_lat"], train["delivery_lon"])
train["dist_ratio"] = train["distance"] / train["hav"]
dist_bins = [0, 300, 600, 1200, 2000, 5000]
train["dist_bin"] = pd.cut(train["distance"], bins=dist_bins)
rpm_by_bin = train.groupby("dist_bin", observed=True)["rpm"].agg(["mean", "median", "std", "count"])
out["numerical"]["distance"]["rpm_by_bin"] = {
    str(k): {kk: float(vv) if isinstance(vv, (float, np.floating)) else int(vv)
             for kk, vv in v.items()} for k, v in rpm_by_bin.to_dict("index").items()
}
# Linear fit R2
slope, intercept, r, p, se = stats.linregress(train["distance"], train["posted_rate"])
out["numerical"]["distance"]["linear_r2"] = float(r ** 2)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(train["distance"], train["posted_rate"], alpha=0.05, s=3, color="#064A56")
ax.set_xlabel("distance (miles)")
ax.set_ylabel("posted_rate ($)")
ax.set_title("distance vs posted_rate")
fig.tight_layout()
fig.savefig(REPORTS / "04_distance_vs_posted_rate.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(train["distance"], train["rpm"], alpha=0.05, s=3, color="#064A56")
ax.set_xlabel("distance (miles)")
ax.set_ylabel("rate per mile ($/mi)")
ax.set_title("Rate per Mile vs Distance")
fig.tight_layout()
fig.savefig(REPORTS / "05_rpm_vs_distance.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
w = train.dropna(subset=["weight"])
ax.scatter(w["weight"], w["posted_rate"], alpha=0.05, s=3, color="#064A56")
ax.set_xlabel("weight (lb)")
ax.set_ylabel("posted_rate ($)")
ax.set_title("weight vs posted_rate (non-missing)")
fig.tight_layout()
fig.savefig(REPORTS / "06_weight_vs_posted_rate.png", dpi=150)
plt.close(fig)

# market_index over time
mi_monthly = train.groupby(train["date"].dt.to_period("M"))["market_index"].mean()
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot([str(p) for p in mi_monthly.index], mi_monthly.values, marker="o", color="#064A56")
ax.set_xticklabels([str(p) for p in mi_monthly.index], rotation=45, ha="right")
ax.set_ylabel("market_index")
ax.set_title("market_index Over Time (train monthly mean)")
fig.tight_layout()
fig.savefig(REPORTS / "07_market_index_over_time.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(train["quote_signal"], train["posted_rate"], alpha=0.05, s=3, color="#064A56")
ax.set_xlabel("quote_signal")
ax.set_ylabel("posted_rate ($)")
ax.set_title("quote_signal vs posted_rate")
fig.tight_layout()
fig.savefig(REPORTS / "08_quote_signal_vs_posted_rate.png", dpi=150)
plt.close(fig)

# ── Part 4: Categorical ───────────────────────────────────────────────────
out["categorical"] = {}
for col in ["pickup", "delivery", "equipment"]:
    vc = train[col].value_counts()
    entry = {
        "nunique_train": int(train[col].nunique()),
        "nunique_val": int(val[col].nunique()),
        "unseen_in_train": sorted(set(val[col]) - set(train[col])),
        "rare_train_lt10": int((vc < 10).sum()),
        "target_mean_by_cat": {str(k): float(v) for k, v in
                               train.groupby(col)["posted_rate"].mean().items()},
        "target_median_by_cat": {str(k): float(v) for k, v in
                                 train.groupby(col)["posted_rate"].median().items()},
    }
    out["categorical"][col] = entry

train["route"] = train["pickup"] + "|" + train["delivery"]
route_counts = train["route"].value_counts()
out["categorical"]["route"] = {
    "nunique_routes_train": int(train["route"].nunique()),
    "nunique_routes_val": int((val["pickup"] + "|" + val["delivery"]).nunique()),
    "routes_with_1_obs": int((route_counts == 1).sum()),
    "routes_with_2_5_obs": int(((route_counts >= 2) & (route_counts <= 5)).sum()),
    "routes_with_6plus_obs": int((route_counts >= 6).sum()),
    "median_route_count": float(route_counts.median()),
}

fig, ax = plt.subplots(figsize=(6, 4))
eq_order = ["Dry Van", "Reefer", "Flatbed"]
data = [train.loc[train["equipment"] == e, "posted_rate"] for e in eq_order]
ax.boxplot(data, tick_labels=eq_order)
ax.set_ylabel("posted_rate ($)")
ax.set_title("posted_rate by Equipment")
fig.tight_layout()
fig.savefig(REPORTS / "09_equipment_vs_posted_rate.png", dpi=150)
plt.close(fig)

# ── Part 5: Geographic ────────────────────────────────────────────────────
g = out["geographic"]
g["pickup_coord_inconsistent"] = int((train.groupby("pickup")[["pickup_lat", "pickup_lon"]].nunique().max(axis=1) > 1).sum())
g["delivery_coord_inconsistent"] = int((train.groupby("delivery")[["delivery_lat", "delivery_lon"]].nunique().max(axis=1) > 1).sum())
g["haversine_corr_distance"] = float(train["hav"].corr(train["distance"]))
g["dist_ratio_mean"] = float(train["dist_ratio"].mean())
g["dist_ratio_std"] = float(train["dist_ratio"].std())
g["dist_ratio_median"] = float(train["dist_ratio"].median())
g["coord_beyond_city"] = "Coordinates are 1:1 with city; no extra info beyond city identity for known cities. For unseen cities, coordinates provide geographic position when city OHE fails."

# ── Part 6: Temporal ──────────────────────────────────────────────────────
temp = out["temporal"]
temp["daily_counts_mean"] = float(train.groupby("date").size().mean())
temp["weekly_counts"] = {str(k): int(v) for k, v in
                         train.groupby(train["date"].dt.to_period("W")).size().items()}
temp["monthly_load_counts"] = {str(k): int(v) for k, v in
                               train.groupby(train["date"].dt.to_period("M")).size().items()}

quarters = {
    "Jan-Mar": train[train["date"].dt.month.isin([1, 2, 3])],
    "Apr-Jun": train[train["date"].dt.month.isin([4, 5, 6])],
    "Jul-Aug": train[train["date"].dt.month.isin([7, 8])],
    "Sep-Oct": train[train["date"].dt.month.isin([9, 10])],
}
temp["quarter_summary"] = {}
for name, df in quarters.items():
    temp["quarter_summary"][name] = {
        "count": int(len(df)),
        "target_mean": float(df["posted_rate"].mean()),
        "target_median": float(df["posted_rate"].median()),
        "distance_mean": float(df["distance"].mean()),
        "market_index_mean": float(df["market_index"].mean()),
        "quote_signal_mean": float(df["quote_signal"].mean()),
        "distance_target_corr": float(df["distance"].corr(df["posted_rate"])),
    }

# Monthly feature stats
for feat in ["market_index", "quote_signal", "distance", "weight"]:
    temp[f"monthly_{feat}"] = {
        str(k): float(v) for k, v in
        train.groupby(train["date"].dt.to_period("M"))[feat].mean().items()
    }

# ── Part 7: Drift ─────────────────────────────────────────────────────────
drift = out["drift"]
drift["numerical"] = {}
for col in num_cols:
    t_vals = train[col].dropna()
    v_vals = val[col].dropna()
    ks, p = stats.ks_2samp(t_vals, v_vals)
    q_train = t_vals.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
    q_val = v_vals.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).tolist()
    severity = "negligible"
    if ks > 0.3:
        severity = "severe"
    elif ks > 0.1:
        severity = "moderate"
    elif ks > 0.05:
        severity = "mild"
    drift["numerical"][col] = {
        "ks": float(ks), "p": float(p), "severity": severity,
        "q_train": q_train, "q_val": q_val,
        "mean_train": float(t_vals.mean()), "mean_val": float(v_vals.mean()),
    }

drift["categorical"] = {}
for col in ["pickup", "delivery", "equipment"]:
    unseen = set(val[col]) - set(train[col])
    drift["categorical"][col] = {
        "unseen_categories": sorted(unseen),
        "unseen_row_pct": float((~val[col].isin(train[col])).mean() * 100),
    }

# Drift plots
fig, ax = plt.subplots(figsize=(8, 4))
t_mi = train["market_index"].dropna()
v_mi = val["market_index"].dropna()
ax.hist(t_mi, bins=50, alpha=0.6, label="train", density=True, color="#064A56")
ax.hist(v_mi, bins=50, alpha=0.6, label="validation", density=True, color="#C0392B")
ax.set_xlabel("market_index")
ax.legend()
ax.set_title("Train vs Validation market_index")
fig.tight_layout()
fig.savefig(REPORTS / "10_train_vs_val_market_index.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(train["distance"], bins=50, alpha=0.6, label="train", density=True, color="#064A56")
ax.hist(val["distance"], bins=50, alpha=0.6, label="validation", density=True, color="#C0392B")
ax.set_xlabel("distance")
ax.legend()
ax.set_title("Train vs Validation distance")
fig.tight_layout()
fig.savefig(REPORTS / "11_train_vs_val_distance.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(train["quote_signal"], bins=50, alpha=0.6, label="train", density=True, color="#064A56")
ax.hist(val["quote_signal"], bins=50, alpha=0.6, label="validation", density=True, color="#C0392B")
ax.set_xlabel("quote_signal")
ax.legend()
ax.set_title("Train vs Validation quote_signal")
fig.tight_layout()
fig.savefig(REPORTS / "12_train_vs_val_quote_signal.png", dpi=150)
plt.close(fig)

# ── Part 8: Data quality ──────────────────────────────────────────────────
dq = out["data_quality"]

# Missing weight patterns
wt_miss = train["weight"].isna()
dq["missing_weight"] = {
    "train_count": int(wt_miss.sum()),
    "val_count": int(val["weight"].isna().sum()),
    "by_equipment_train": {str(k): int(v) for k, v in train.loc[wt_miss, "equipment"].value_counts().items()},
    "target_mean_missing": float(train.loc[wt_miss, "posted_rate"].mean()),
    "target_mean_present": float(train.loc[~wt_miss, "posted_rate"].mean()),
    "monthly_missing_pct": {str(k): float(v) for k, v in
                            (train.groupby(train["date"].dt.to_period("M"))["weight"]
                             .apply(lambda s: s.isna().mean() * 100)).items()},
}

# Negative weight deep dive
neg = train[train["weight"] < 0]
dq["negative_weight"] = {
    "train_count": int(len(neg)),
    "val_count": int((val["weight"] < 0).sum()),
    "unique_values_count": int(neg["weight"].nunique()),
    "always_negative_when_present": "All negative values are multiples/symmetric around magnitude; range -47500 to -70",
    "corr_weight_target_positive_only": float(train.loc[train["weight"] > 0, "weight"].corr(
        train.loc[train["weight"] > 0, "posted_rate"])),
    "mean_target_negative": float(neg["posted_rate"].mean()),
    "mean_target_positive": float(train.loc[train["weight"] > 0, "posted_rate"].mean()),
    "pct_same_abs_as_positive_max": float((neg["weight"].abs() == 47500).mean() * 100),
}

# Missing market_index
mi_miss = train["market_index"].isna()
dq["missing_market_index"] = {
    "train_count": int(mi_miss.sum()),
    "val_count": int(val["market_index"].isna().sum()),
    "by_equipment_train": {str(k): int(v) for k, v in train.loc[mi_miss, "equipment"].value_counts().items()},
    "monthly_missing_pct": {str(k): float(v) for k, v in
                            (train.groupby(train["date"].dt.to_period("M"))["market_index"]
                             .apply(lambda s: s.isna().mean() * 100)).items()},
}

dq["unseen_cities"] = {
    "cities": unseen,
    "pickup_rows": int((~val["pickup"].isin(train_cities)).sum()),
    "delivery_rows": int((~val["delivery"].isin(train_cities)).sum()),
    "total_val_rows_touched": int(((~val["pickup"].isin(train_cities)) |
                                   (~val["delivery"].isin(train_cities))).sum()),
}

# ── Part 9: Leakage (documented) ──────────────────────────────────────────
out["leakage"] = {
    "load_id": {"available": True, "leakage": "Encodes TR vs TE split; exclude"},
    "posted_rate": {"available": False, "leakage": "Target; exclude"},
    "distance": {"available": True, "leakage": "None; available at inference"},
    "market_index": {"available": True, "leakage": "None if treated as exogenous input at load time"},
    "quote_signal": {"available": True, "leakage": "None if treated as exogenous input at load time"},
    "route_target_encoding": {"leakage": "Unsafe if computed on full train or includes val fold labels"},
    "date_features": {"leakage": "Safe; calendar features only"},
    "historical_aggregates": {"leakage": "Must be computed on training fold only with temporal cutoff"},
}

# ── Part 10: Preprocessing recommendations ────────────────────────────────
out["preprocessing"] = {
    "negative_weight": "Set to NaN then impute with median by equipment (fit on train fold only). Negative weights are non-physical; symmetric with ±47500 suggests corrupted/sentinel values.",
    "missing_weight": "Median imputation by equipment with global fallback; add weight_is_missing indicator (missingness weakly structured).",
    "missing_market_index": "Median imputation by month (or global median on train fold); add market_index_is_missing indicator.",
    "categorical": "OneHotEncoder handle_unknown='ignore' for pickup, delivery, equipment, route.",
    "unseen_cities": "Rely on coordinates + distance + equipment; OHE ignores unknown cities.",
    "date": "Calendar decomposition (month, dow, etc.); no random CV.",
    "outliers": "Do not remove target outliers; tree models tolerate heavy tails.",
}

# Save JSON
stats_path = REPORTS / "phase1_stats.json"
with open(stats_path, "w") as f:
    json.dump(out, f, indent=2, default=str)

print(f"Saved stats to {stats_path}")
print(f"Saved {len(list(REPORTS.glob('*.png')))} plots to {REPORTS}")
