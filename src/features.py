"""Feature engineering and preprocessing pipeline (Phase 2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import DATE_COLUMN, DISTANCE_DERIVED_COLUMNS, FEATURE_SETS

DISTANCE_BIN_EDGES = [0, 300, 600, 1200, 2000, 5000]


class WeightCleaner(BaseEstimator, TransformerMixin):
    """Mask negative weights, flag missingness, impute by equipment median."""

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        valid = X["weight"].where(X["weight"] >= 0)
        self.global_median_ = float(valid.median())
        self.equipment_medians_ = valid.groupby(X["equipment"]).median().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        raw = X["weight"]
        X["weight_is_missing"] = (raw.isna() | (raw < 0)).astype(int)
        cleaned = raw.where(raw >= 0)
        eq_fill = X["equipment"].map(self.equipment_medians_)
        X["weight"] = cleaned.fillna(eq_fill).fillna(self.global_median_)
        return X


class MarketIndexImputer(BaseEstimator, TransformerMixin):
    """Flag missing market_index and impute with training-fold month medians."""

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        months = pd.to_datetime(X[DATE_COLUMN]).dt.month
        valid = X["market_index"]
        self.global_median_ = float(valid.median())
        self.month_medians_ = valid.groupby(months).median().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["market_index_is_missing"] = X["market_index"].isna().astype(int)
        months = pd.to_datetime(X[DATE_COLUMN]).dt.month
        month_fill = months.map(self.month_medians_)
        X["market_index"] = X["market_index"].fillna(month_fill).fillna(self.global_median_)
        return X


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Calendar features from date; drops raw date column."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        dates = pd.to_datetime(X[DATE_COLUMN])
        X["month"] = dates.dt.month
        X["day_of_week"] = dates.dt.dayofweek
        X["day_of_month"] = dates.dt.day
        X["week_of_year"] = dates.dt.isocalendar().week.astype(int)
        X["quarter"] = dates.dt.quarter
        return X.drop(columns=[DATE_COLUMN])


class HaversineFeatureExtractor(BaseEstimator, TransformerMixin):
    """Haversine miles and road-minus-geodesic residual."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        lat1 = np.radians(X["pickup_lat"])
        lon1 = np.radians(X["pickup_lon"])
        lat2 = np.radians(X["delivery_lat"])
        lon2 = np.radians(X["delivery_lon"])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        X["haversine_miles"] = 3958.8 * c
        X["distance_vs_haversine"] = X["distance"] - X["haversine_miles"]
        return X


class RouteFeatureExtractor(BaseEstimator, TransformerMixin):
    """Route string and weight-per-mile (uses cleaned weight)."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["route"] = X["pickup"].astype(str) + "->" + X["delivery"].astype(str)
        X["weight_per_mile"] = X["weight"] / X["distance"].replace(0, np.nan)
        return X


class DistanceDerivedExtractor(BaseEstimator, TransformerMixin):
    """log(distance) and binned distance."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["log_distance"] = np.log(X["distance"].clip(lower=1.0))
        X["distance_bin"] = pd.cut(
            X["distance"],
            bins=DISTANCE_BIN_EDGES,
            labels=False,
            include_lowest=True,
        ).astype(float)
        return X


class InteractionFeatureExtractor(BaseEstimator, TransformerMixin):
    """Non-target interaction features (Phase 3)."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if "market_index" in X.columns and "month" in X.columns:
            X["market_index_x_month"] = X["market_index"] * X["month"]
        return X


class FeatureEngineeringPipeline(BaseEstimator, TransformerMixin):
    """Fit/transform chain; learned stats come only from training fold."""

    def __init__(self):
        self.steps_: list[tuple[str, TransformerMixin]] = [
            ("weight", WeightCleaner()),
            ("market", MarketIndexImputer()),
            ("dates", DateFeatureExtractor()),
            ("interactions", InteractionFeatureExtractor()),
            ("haversine", HaversineFeatureExtractor()),
            ("route", RouteFeatureExtractor()),
            ("distance_derived", DistanceDerivedExtractor()),
        ]

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X).copy()
        self.fitted_steps_: list[tuple[str, TransformerMixin]] = []
        for name, step in self.steps_:
            step.fit(X, y)
            X = step.transform(X)
            self.fitted_steps_.append((name, step))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        for _, step in self.fitted_steps_:
            X = step.transform(X)
        return X


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select columns and preserve DataFrame (for CatBoost)."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(X)[self.columns].copy()


def get_feature_columns(feature_set: str) -> tuple[list[str], list[str]]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    spec = FEATURE_SETS[feature_set]
    return list(spec["numeric"]), list(spec["categorical"])


def build_preprocessing_pipeline(feature_set: str = "full") -> Pipeline:
    """Sklearn pipeline: engineer features, select columns, OHE categoricals."""
    numeric_cols, categorical_cols = get_feature_columns(feature_set)

    transformers = []
    if numeric_cols:
        transformers.append(("num", "passthrough", numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    return Pipeline(
        steps=[
            ("engineer", FeatureEngineeringPipeline()),
            ("select", preprocessor),
        ]
    )


def build_matrix_pipeline(feature_set: str = "full") -> Pipeline:
    """Engineered features as DataFrame (for CatBoost native categoricals)."""
    numeric_cols, categorical_cols = get_feature_columns(feature_set)
    all_cols = numeric_cols + categorical_cols

    return Pipeline(
        steps=[
            ("engineer", FeatureEngineeringPipeline()),
            ("select", ColumnSelector(all_cols)),
        ]
    )
