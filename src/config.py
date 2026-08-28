"""Project configuration and constants."""

from pathlib import Path

RANDOM_SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "train-test.csv"
VALIDATION_HOLDOUT_PATH = PROJECT_ROOT / "validation.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TARGET_COLUMN = "posted_rate"
ID_COLUMN = "load_id"
DATE_COLUMN = "date"

# Primary chronological split (Phase 2 model selection).
TRAIN_END_DATE = "2025-08-31"
VAL_START_DATE = "2025-09-01"
VAL_END_DATE = "2025-10-31"

# Sensitivity split (stability check only).
SENSITIVITY_TRAIN_END_DATE = "2025-09-30"
SENSITIVITY_VAL_START_DATE = "2025-10-01"
SENSITIVITY_VAL_END_DATE = "2025-10-31"

SPLIT_PRIMARY = "primary"
SPLIT_SENSITIVITY = "sensitivity"

# Columns never used as model features.
EXCLUDE_COLUMNS = {ID_COLUMN, TARGET_COLUMN}

RAW_FEATURE_COLUMNS = [
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
]

CATEGORICAL_COLUMNS = ["pickup", "delivery", "equipment"]
NUMERIC_COLUMNS = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
]

CALENDAR_COLUMNS = ["month", "day_of_week", "day_of_month", "week_of_year", "quarter"]
GEO_NUMERIC_COLUMNS = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "haversine_miles",
    "distance_vs_haversine",
]
DISTANCE_DERIVED_COLUMNS = ["log_distance", "distance_bin"]

# Feature ablation sets (Phase 2 Part 10).
FEATURE_SETS: dict[str, dict[str, list[str]]] = {
    "A": {"numeric": ["distance"], "categorical": []},
    "B": {"numeric": ["distance"], "categorical": ["equipment"]},
    "C": {
        "numeric": ["distance", "weight", "weight_is_missing"],
        "categorical": ["equipment"],
    },
    "D": {
        "numeric": [
            "distance",
            "weight",
            "weight_is_missing",
            "market_index",
            "market_index_is_missing",
            "quote_signal",
        ],
        "categorical": ["equipment"],
    },
    "E": {
        "numeric": [
            "distance",
            "weight",
            "weight_is_missing",
            "market_index",
            "market_index_is_missing",
            "quote_signal",
            *CALENDAR_COLUMNS,
        ],
        "categorical": ["equipment"],
    },
    "F": {
        "numeric": [
            "distance",
            "weight",
            "weight_is_missing",
            "market_index",
            "market_index_is_missing",
            "quote_signal",
            *CALENDAR_COLUMNS,
        ],
        "categorical": ["equipment", "pickup", "delivery", "route"],
    },
    "G": {
        "numeric": [
            "distance",
            *DISTANCE_DERIVED_COLUMNS,
            "weight",
            "weight_is_missing",
            "market_index",
            "market_index_is_missing",
            "quote_signal",
            *CALENDAR_COLUMNS,
            *GEO_NUMERIC_COLUMNS,
            "weight_per_mile",
        ],
        "categorical": ["equipment", "pickup", "delivery", "route"],
    },
    "full": {
        "numeric": [
            "distance",
            *DISTANCE_DERIVED_COLUMNS,
            "weight",
            "weight_is_missing",
            "market_index",
            "market_index_is_missing",
            "quote_signal",
            *CALENDAR_COLUMNS,
            *GEO_NUMERIC_COLUMNS,
            "weight_per_mile",
        ],
        "categorical": ["equipment", "pickup", "delivery", "route"],
    },
}

CATBOOST_CATEGORICAL_COLUMNS = ["equipment", "pickup", "delivery", "route"]

# Phase 3 controlled feature sets (built on C base).
_C_NUM = ["distance", "weight", "weight_is_missing"]
_C_CAT = ["equipment"]
_C4_NUM = [*_C_NUM, *DISTANCE_DERIVED_COLUMNS]
_GEO = [*GEO_NUMERIC_COLUMNS, "weight_per_mile"]
_MARKET = ["market_index", "market_index_is_missing"]
_QUOTE = ["quote_signal"]

def _fs(numeric: list[str], categorical: list[str] | None = None) -> dict[str, list[str]]:
    return {"numeric": numeric, "categorical": categorical or list(_C_CAT)}


PHASE3_FEATURE_SETS: dict[str, dict[str, list[str]]] = {
    "C": _fs(list(_C_NUM)),
    "C2": _fs([*_C_NUM, "log_distance"]),
    "C3": _fs([*_C_NUM, "distance_bin"]),
    "C4": _fs(list(_C4_NUM)),
    "E": _fs([*_C4_NUM, *CALENDAR_COLUMNS]),
    "R": _fs(list(_C4_NUM), [*_C_CAT, "pickup", "delivery", "route"]),
    "G": _fs([*_C4_NUM, *_GEO], [*_C_CAT, "pickup", "delivery", "route"]),
    "M": _fs([*_C4_NUM, *_MARKET]),
    "Q": _fs([*_C4_NUM, *_QUOTE]),
    "MC": _fs([*_C4_NUM, *_MARKET, *CALENDAR_COLUMNS]),
    "FULL": _fs(
        [*_C4_NUM, *_MARKET, *_QUOTE, *CALENDAR_COLUMNS, *_GEO],
        [*_C_CAT, "pickup", "delivery", "route"],
    ),
}

# Market-index investigation sets (Part 2).
MARKET_INVESTIGATION_SETS: dict[str, dict[str, list[str]]] = {
    "C": _fs(list(_C_NUM)),
    "C_MI": _fs([*_C_NUM, *_MARKET]),
    "C_Q": _fs([*_C_NUM, *_QUOTE]),
    "C_MI_Q": _fs([*_C_NUM, *_MARKET, *_QUOTE]),
    "C_MI_CAL": _fs([*_C_NUM, *_MARKET, *CALENDAR_COLUMNS]),
    "C_MI_MONTH": _fs([*_C_NUM, *_MARKET, "market_index_x_month"]),
    "C_Q_CAL": _fs([*_C_NUM, *_QUOTE, *CALENDAR_COLUMNS]),
}

# Merge into FEATURE_SETS for pipeline lookup.
FEATURE_SETS.update(PHASE3_FEATURE_SETS)
FEATURE_SETS.update({k: v for k, v in MARKET_INVESTIGATION_SETS.items() if k not in FEATURE_SETS})

PRIMARY_METRIC = "mae"

# Default HistGB params (Phase 2 baseline).
DEFAULT_HISTGB_PARAMS = {
    "max_depth": 8,
    "learning_rate": 0.08,
    "max_iter": 300,
    "random_state": RANDOM_SEED,
}

# Locked final model (Phase 4).
FINAL_MODEL_FEATURE_SET = "Q"
LOCKED_HISTGB_PARAMS = {
    "max_depth": 6,
    "l2_regularization": 0.1,
    "learning_rate": 0.08,
    "max_iter": 300,
    "random_state": RANDOM_SEED,
}
FINAL_TRAIN_END_DATE = "2025-10-31"
VALIDATION_TEMPLATE_PATH = PROJECT_ROOT / "validation-predictions-template.csv"
VALIDATION_PREDICTIONS_PATH = PROJECT_ROOT / "validation_predictions.csv"
DECEMBER_CHART_PATH = PROJECT_ROOT / "december-chart-inputs.csv"
FINAL_PREDICTION_DIAGNOSTICS_PATH = ARTIFACTS_DIR / "final_prediction_diagnostics.csv"
