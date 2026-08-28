"""Data loading and chronological splitting."""

from __future__ import annotations

import pandas as pd

from src.config import (
    DATE_COLUMN,
    RAW_FEATURE_COLUMNS,
    SENSITIVITY_TRAIN_END_DATE,
    SENSITIVITY_VAL_END_DATE,
    SENSITIVITY_VAL_START_DATE,
    SPLIT_PRIMARY,
    SPLIT_SENSITIVITY,
    TARGET_COLUMN,
    TRAIN_END_DATE,
    TRAIN_PATH,
    VAL_END_DATE,
    VAL_START_DATE,
)


def load_train_data(path: str | None = None) -> pd.DataFrame:
    """Load labeled development data. Does not read validation.csv."""
    df = pd.read_csv(path or TRAIN_PATH)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    return df


def split_by_dates(
    df: pd.DataFrame,
    train_end: str,
    val_start: str,
    val_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split by inclusive date bounds. No shuffling."""
    train_end_ts = pd.Timestamp(train_end)
    val_start_ts = pd.Timestamp(val_start)
    val_end_ts = pd.Timestamp(val_end)

    train_mask = df[DATE_COLUMN] <= train_end_ts
    val_mask = (df[DATE_COLUMN] >= val_start_ts) & (df[DATE_COLUMN] <= val_end_ts)

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()

    if train_df.empty or val_df.empty:
        raise ValueError(
            "Chronological split produced an empty partition. "
            f"train={len(train_df)}, val={len(val_df)}"
        )
    return train_df, val_df


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Primary Phase 2 split: train Jan–Aug, validate Sep–Oct."""
    return split_by_dates(df, TRAIN_END_DATE, VAL_START_DATE, VAL_END_DATE)


def sensitivity_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sensitivity split: train Jan–Sep, validate Oct only."""
    return split_by_dates(
        df,
        SENSITIVITY_TRAIN_END_DATE,
        SENSITIVITY_VAL_START_DATE,
        SENSITIVITY_VAL_END_DATE,
    )


def get_split(df: pd.DataFrame, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if split_name == SPLIT_PRIMARY:
        return chronological_split(df)
    if split_name == SPLIT_SENSITIVITY:
        return sensitivity_split(df)
    raise ValueError(f"Unknown split: {split_name}")


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df[RAW_FEATURE_COLUMNS].copy()


def get_target(df: pd.DataFrame) -> pd.Series:
    return df[TARGET_COLUMN].copy()


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return get_feature_matrix(df), get_target(df)


def summarize_split(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Return row counts, date ranges, and target stats for reporting."""
    return {
        "train_rows": len(train_df),
        "validation_rows": len(val_df),
        "train_date_min": str(train_df[DATE_COLUMN].min().date()),
        "train_date_max": str(train_df[DATE_COLUMN].max().date()),
        "val_date_min": str(val_df[DATE_COLUMN].min().date()),
        "val_date_max": str(val_df[DATE_COLUMN].max().date()),
        "train_target_mean": float(train_df[TARGET_COLUMN].mean()),
        "train_target_median": float(train_df[TARGET_COLUMN].median()),
        "train_target_std": float(train_df[TARGET_COLUMN].std()),
        "val_target_mean": float(val_df[TARGET_COLUMN].mean()),
        "val_target_median": float(val_df[TARGET_COLUMN].median()),
        "val_target_std": float(val_df[TARGET_COLUMN].std()),
    }
