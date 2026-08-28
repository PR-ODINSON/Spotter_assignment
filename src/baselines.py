"""Baseline and ML model definitions for Phase 2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import RANDOM_SEED
from src.features import build_matrix_pipeline, build_preprocessing_pipeline


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: Any
    feature_set: str
    notes: str
    use_log_target: bool = False
    is_business_baseline: bool = False


class GlobalMedianPredictor(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        self.median_ = float(np.median(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.median_, dtype=float)


class EquipmentMedianPredictor(BaseEstimator, RegressorMixin):
    def fit(self, X: pd.DataFrame, y):
        X = pd.DataFrame(X)
        self.global_median_ = float(np.median(y))
        self.equipment_medians_ = (
            pd.Series(y, index=X.index).groupby(X["equipment"]).median().to_dict()
        )
        return self

    def predict(self, X: pd.DataFrame):
        X = pd.DataFrame(X)
        return X["equipment"].map(self.equipment_medians_).fillna(self.global_median_).to_numpy()


class DistanceLinearRegressor(BaseEstimator, RegressorMixin):
    """OLS on distance only."""

    def fit(self, X: pd.DataFrame, y):
        self.model_ = LinearRegression()
        self.model_.fit(pd.DataFrame(X)[["distance"]], y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.model_.predict(pd.DataFrame(X)[["distance"]])


class DistanceEquipmentLinearRegressor(BaseEstimator, RegressorMixin):
    """OLS on distance + one-hot equipment."""

    def fit(self, X: pd.DataFrame, y):
        X = pd.DataFrame(X)
        self.pipeline_ = Pipeline(
            steps=[
                (
                    "prep",
                    ColumnTransformer(
                        transformers=[
                            ("dist", "passthrough", ["distance"]),
                            (
                                "eq",
                                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                ["equipment"],
                            ),
                        ]
                    ),
                ),
                ("model", LinearRegression()),
            ]
        )
        self.pipeline_.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        return self.pipeline_.predict(pd.DataFrame(X))


class EquipmentDistanceRatePredictor(BaseEstimator, RegressorMixin):
    """distance * equipment-specific median RPM (training fold only)."""

    def fit(self, X: pd.DataFrame, y):
        X = pd.DataFrame(X)
        rpm = pd.Series(y, index=X.index) / X["distance"].replace(0, np.nan)
        self.global_rpm_ = float(rpm.median())
        self.equipment_rpm_ = rpm.groupby(X["equipment"]).median().to_dict()
        return self

    def predict(self, X: pd.DataFrame):
        X = pd.DataFrame(X)
        rpm = X["equipment"].map(self.equipment_rpm_).fillna(self.global_rpm_)
        return X["distance"].to_numpy(dtype=float) * rpm.to_numpy(dtype=float)


class Log1pTargetWrapper(BaseEstimator, RegressorMixin):
    """Fit base regressor on log1p(y); predict in dollar space via expm1."""

    def __init__(self, estimator: BaseEstimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, np.log1p(np.asarray(y, dtype=float)))
        return self

    def predict(self, X):
        return np.expm1(self.estimator_.predict(X))


def _ml_pipeline(estimator: BaseEstimator, feature_set: str = "full") -> Pipeline:
    return Pipeline(
        steps=[
            ("features", build_preprocessing_pipeline(feature_set)),
            ("model", clone(estimator)),
        ]
    )


def build_model(spec: ModelSpec) -> BaseEstimator:
    if spec.is_business_baseline:
        model = clone(spec.estimator)
    elif spec.use_log_target:
        model = Pipeline(
            steps=[
                ("features", build_preprocessing_pipeline(spec.feature_set)),
                ("model", Log1pTargetWrapper(clone(spec.estimator))),
            ]
        )
    else:
        model = _ml_pipeline(spec.estimator, spec.feature_set)
    return model


def get_business_baselines() -> list[ModelSpec]:
    return [
        ModelSpec("global_median", GlobalMedianPredictor(), "none", "Global median rate", is_business_baseline=True),
        ModelSpec(
            "equipment_median",
            EquipmentMedianPredictor(),
            "none",
            "Per-equipment median",
            is_business_baseline=True,
        ),
        ModelSpec(
            "distance_linear",
            DistanceLinearRegressor(),
            "none",
            "OLS on distance",
            is_business_baseline=True,
        ),
        ModelSpec(
            "distance_equipment_linear",
            DistanceEquipmentLinearRegressor(),
            "none",
            "OLS on distance + equipment OHE",
            is_business_baseline=True,
        ),
        ModelSpec(
            "equipment_distance_rate",
            EquipmentDistanceRatePredictor(),
            "none",
            "distance * equipment median RPM",
            is_business_baseline=True,
        ),
    ]


def get_ml_models(feature_set: str = "full") -> list[ModelSpec]:
    return [
        ModelSpec(
            "ridge",
            Ridge(alpha=10.0, random_state=RANDOM_SEED),
            feature_set,
            "Ridge on full engineered features",
        ),
        ModelSpec(
            "hist_gradient_boosting",
            HistGradientBoostingRegressor(
                max_depth=8,
                learning_rate=0.08,
                max_iter=300,
                random_state=RANDOM_SEED,
            ),
            feature_set,
            "HistGradientBoosting baseline",
        ),
        ModelSpec(
            "extra_trees",
            ExtraTreesRegressor(
                n_estimators=200,
                max_depth=16,
                min_samples_leaf=5,
                n_jobs=-1,
                random_state=RANDOM_SEED,
            ),
            feature_set,
            "ExtraTrees bagged ensemble",
        ),
    ]


def get_ablation_feature_sets() -> list[str]:
    return ["A", "B", "C", "D", "E", "F", "G"]


def build_catboost_model(feature_set: str = "full"):
    """CatBoost with native categorical handling. Returns None if unavailable."""
    try:
        from catboost import CatBoostRegressor
    except ImportError:
        return None

    from src.config import CATBOOST_CATEGORICAL_COLUMNS
    from src.features import get_feature_columns

    numeric_cols, categorical_cols = get_feature_columns(feature_set)
    cat_names = [c for c in CATBOOST_CATEGORICAL_COLUMNS if c in numeric_cols + categorical_cols]

    matrix_pipe = build_matrix_pipeline(feature_set)
    model = CatBoostRegressor(
        iterations=300,
        depth=8,
        learning_rate=0.08,
        loss_function="MAE",
        random_seed=RANDOM_SEED,
        verbose=0,
        cat_features=cat_names,
    )
    return Pipeline(steps=[("features", matrix_pipe), ("model", model)])
