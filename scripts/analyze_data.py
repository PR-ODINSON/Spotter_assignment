"""Quick data analysis for validation strategy decisions."""
from pathlib import Path

import pandas as pd
from scipy import stats

BASE = Path(__file__).resolve().parents[1]
train = pd.read_csv(BASE / "train-test.csv")
val = pd.read_csv(BASE / "validation.csv")

print("SHAPE train", train.shape, "val", val.shape)
print("COLS train", list(train.columns))
print("COLS val", list(val.columns))

train["date"] = pd.to_datetime(train["date"])
val["date"] = pd.to_datetime(val["date"])

print("\nDATE train", train["date"].min(), train["date"].max(), "unique", train["date"].nunique())
print("DATE val", val["date"].min(), val["date"].max(), "unique", val["date"].nunique())

print("\nMONTH train:")
print(train["date"].dt.to_period("M").value_counts().sort_index())
print("\nMONTH val:")
print(val["date"].dt.to_period("M").value_counts().sort_index())

print("\nMISSING train:", train.isna().sum().to_dict())
print("MISSING val:", val.isna().sum().to_dict())

print("\nDUPLICATE load_id train", train["load_id"].duplicated().sum())
print("DUPLICATE load_id val", val["load_id"].duplicated().sum())
print("ID overlap", len(set(train["load_id"]) & set(val["load_id"])))

feat = [c for c in train.columns if c not in ["load_id", "posted_rate"]]
print("\nDUPLICATE feature rows train", train[feat].duplicated().sum())
print("DUPLICATE feature rows val", val[feat].duplicated().sum())

print("\nTARGET stats:", train["posted_rate"].describe())
print("corr with target:")
for c in train.columns:
    if c != "posted_rate" and pd.api.types.is_numeric_dtype(train[c]):
        print(f"  {c}: {train[c].corr(train['posted_rate']):.4f}")

print("\nEQUIPMENT train:", train["equipment"].value_counts().to_dict())
print("EQUIPMENT val:", val["equipment"].value_counts().to_dict())

print("\nPICKUP nunique train", train["pickup"].nunique(), "val", val["pickup"].nunique())
print("DELIVERY nunique train", train["delivery"].nunique(), "val", val["delivery"].nunique())

for c in ["distance", "weight", "market_index", "quote_signal"]:
    ks, p = stats.ks_2samp(train[c], val[c])
    print(f"KS {c}: {ks:.4f} p={p:.2e}")

train["route"] = train["pickup"] + "|" + train["delivery"]
val["route"] = val["pickup"] + "|" + val["delivery"]
print("\nSame route same date max count train", train.groupby(["route", "date"]).size().max())
print("Same route same date max count val", val.groupby(["route", "date"]).size().max())

monthly = train.groupby(train["date"].dt.to_period("M"))["posted_rate"].agg(["mean", "std", "count"])
print("\nMonthly target:")
print(monthly)

train["rpm"] = train["posted_rate"] / train["distance"]
monthly_rpm = train.groupby(train["date"].dt.to_period("M"))["rpm"].mean()
print("\nMonthly rate/mile:")
print(monthly_rpm)
