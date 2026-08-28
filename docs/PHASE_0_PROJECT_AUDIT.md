# Phase 0 Project Audit — Freight Rate Prediction

**Date:** 2025-08-27  
**Scope:** Read-only inspection of repository, data, and `score.py`. No modeling performed.

---

## Project Structure

```
spotter/
├── freight-rate-ml-assessment.pdf   # Official assessment instructions
├── readme.md                        # Setup and submission instructions
├── requirements.txt                 # Python dependencies
├── score.py                         # Submission validator + December chart generator
├── train-test.csv                   # Labeled development data (48,000 rows)
├── validation.csv                   # Unlabeled holdout features (12,000 rows)
├── validation-predictions-template.csv
├── validation_predictions.csv       # Pre-existing placeholder predictions (constant 2030.76)
├── december-chart-inputs.csv        # 31 fixed-scenario rows for December chart
├── docs/
│   └── PHASE_0_PROJECT_AUDIT.md     # This document
├── scripts/
│   ├── analyze_data.py              # Exploratory analysis helper
│   ├── phase0_audit.py              # Profiling script used for this audit
│   └── run_phase2_baselines.py      # Prior baseline runner (not executed in Phase 0)
├── src/
│   ├── config.py                    # Paths, split dates, column definitions
│   ├── data.py                      # Loading and chronological split helpers
│   ├── features.py                  # Feature engineering pipeline
│   ├── baselines.py                 # Baseline model definitions
│   └── metrics.py                   # Local MAE/RMSE/MAPE/R² helpers
├── artifacts/
│   └── model_comparison.csv         # Pre-existing baseline results (prior work)
├── scorer_results/                  # Empty; populated when score.py runs successfully
└── venv/                            # Local virtual environment
```

### Path discrepancy (important)

The assessment PDF and `readme.md` reference files under a `data/` directory with underscore names:

| Documented path | Actual path in repo |
|---|---|
| `data/train_test.csv` | `train-test.csv` (repo root) |
| `data/validation.csv` | `validation.csv` (repo root) |
| `data/validation_predictions_template.csv` | `validation-predictions-template.csv` (repo root) |
| `data/december_chart_inputs.csv` | `december-chart-inputs.csv` (repo root) |

No `data/` subdirectory exists. All CSVs live at the repository root with hyphenated filenames.

---

## Dataset Dimensions

| File | Rows | Columns |
|---|---:|---:|
| `train-test.csv` | 48,000 | 14 |
| `validation.csv` | 12,000 | 13 |
| `validation-predictions-template.csv` | 12,000 | 2 |
| `december-chart-inputs.csv` | 31 | 7 |

---

## Column Inventory

### `train-test.csv` (14 columns)

`load_id`, `pickup`, `delivery`, `pickup_lat`, `pickup_lon`, `delivery_lat`, `delivery_lon`, `distance`, `equipment`, `weight`, `date`, `market_index`, `quote_signal`, `posted_rate`

### `validation.csv` (13 columns)

Same as training **except** `posted_rate` is absent (unlabeled holdout).

### `validation-predictions-template.csv` (2 columns)

`load_id`, `predicted_rate` — all 12,000 `predicted_rate` values are empty.

### `december-chart-inputs.csv` (7 columns)

`pickup`, `delivery`, `distance`, `equipment`, `weight`, `date`, `predicted_rate`

---

## Target Definition

| Property | Value |
|---|---|
| **Target column** | `posted_rate` |
| **Present in** | `train-test.csv` only |
| **Absent from** | `validation.csv` (confirmed) |
| **Type** | Continuous (USD freight rate) |
| **Train stats** | min 57.22, max 25,533.0, mean 2,373.98, median 2,030.76, std 1,486.49 |

Submission output column: `predicted_rate` (one value per `load_id` in `validation_predictions.csv`).

---

## ID Definition

| Property | Value |
|---|---|
| **ID column** | `load_id` |
| **Train format** | `TR-000001` … `TR-048000` (prefix `TR-`) |
| **Validation format** | `TE-000001` … `TE-012000` (prefix `TE-`) |
| **Uniqueness** | 48,000 / 48,000 unique in train; 12,000 / 12,000 unique in validation |
| **Overlap** | 0 IDs shared between train and validation |
| **Template alignment** | `validation-predictions-template.csv` `load_id` values match `validation.csv` exactly (same order) |

`score.py` expects validation IDs exactly `{TE-000001, …, TE-012000}`.

---

## Data Types

| Column | Train dtype | Val dtype | Role |
|---|---|---|---|
| `load_id` | object | object | Identifier — **do not use as feature** |
| `pickup` | object | object | Categorical |
| `delivery` | object | object | Categorical |
| `pickup_lat` | float64 | float64 | Numeric |
| `pickup_lon` | float64 | float64 | Numeric |
| `delivery_lat` | float64 | float64 | Numeric |
| `delivery_lon` | float64 | float64 | Numeric |
| `distance` | float64 | float64 | Numeric |
| `equipment` | object | object | Categorical (3 levels) |
| `weight` | float64 | float64 | Numeric |
| `date` | object (parseable) | object (parseable) | Date/time |
| `market_index` | float64 | float64 | Numeric |
| `quote_signal` | float64 | float64 | Numeric |
| `posted_rate` | float64 | — | Target — **do not use as feature** |

**Categorical columns:** `pickup`, `delivery`, `equipment`  
**Numeric columns:** `pickup_lat`, `pickup_lon`, `delivery_lat`, `delivery_lon`, `distance`, `weight`, `market_index`, `quote_signal`  
**Date/time column:** `date` (daily granularity, no time component)

**Redundancy note:** Each city maps to exactly one coordinate pair (0 cities with inconsistent lat/lon in training). Coordinates are deterministic given `pickup` / `delivery` city names.

---

## Missing-Value Summary

| Column | Missing (train) | Missing (val) | % train | % val |
|---|---:|---:|---:|---:|
| `weight` | 300 | 165 | 0.63% | 1.38% |
| `market_index` | 374 | 249 | 0.78% | 2.08% |
| All other columns | 0 | 0 | 0% | 0% |

No missing values in `load_id`, `pickup`, `delivery`, coordinates, `distance`, `equipment`, `date`, `quote_signal`, or `posted_rate`.

---

## Duplicate Summary

| Check | Train | Validation |
|---|---:|---:|
| Duplicate `load_id` | 0 | 0 |
| Duplicate full rows | 0 | 0 |
| Duplicate feature rows (excluding `load_id`, `posted_rate`) | 0 | 0 |
| Max loads sharing same route + date | 3 | 3 |

---

## Data Quality Issues

### Negative `weight` values

| Split | Negative weights | Missing weights |
|---|---:|---:|
| Train | 292 | 300 |
| Validation | 145 | 165 |

Range is −47,500 to 47,500 (symmetric). Negative values are likely data errors or a sentinel encoding — requires cleaning/imputation in Phase 1.

### Variable `distance` for identical routes

- 3,944 / 4,014 unique routes in training have more than one `distance` value.
- 1,181 route+date groups have multiple distances on the same day (different equipment/weight/rate).
- `distance` correlates 0.9995 with haversine distance from coordinates (ratio mean ≈ 1.19, road-distance multiplier).
- Distance is a strong legitimate predictor (corr with target = 0.909) but not strictly determined by city pair alone.

### Unseen cities in validation

8 cities appear in validation but not in training (725 pickup rows, 722 delivery rows affected):

`Allentown`, `Charlotte`, `Chicago`, `Jackson`, `Knoxville`, `Laredo`, `Norfolk`, `San Diego`

Models using city one-hot encoding must handle unknown categories (`handle_unknown="ignore"` already planned in `src/features.py`).

---

## Train vs Validation Observations

### Temporal structure (clear holdout)

| Dataset | Date range | Unique dates | Row count |
|---|---|---|---|
| `train-test.csv` | 2025-01-01 → 2025-10-31 | 304 | 48,000 |
| `validation.csv` | 2025-11-01 → 2025-12-31 | 61 | 12,000 |

**Monthly row counts — train:**

| Month | Rows |
|---|---:|
| 2025-01 | 4,918 |
| 2025-02 | 4,337 |
| 2025-03 | 5,036 |
| 2025-04 | 4,819 |
| 2025-05 | 4,913 |
| 2025-06 | 4,783 |
| 2025-07 | 4,912 |
| 2025-08 | 4,759 |
| 2025-09 | 4,670 |
| 2025-10 | 4,853 |

**Monthly row counts — validation:**

| Month | Rows |
|---|---:|
| 2025-11 | 5,836 |
| 2025-12 | 6,164 |

Validation is a **future temporal holdout** (Nov–Dec 2025 vs Jan–Oct 2025 training). A chronological split within `train-test.csv` is the natural validation strategy (e.g., train Jan–Aug, validate Sep–Oct, as already configured in `src/config.py`).

### Monthly target trend (train only)

Mean `posted_rate` rises from ~2,256 (Jan) to ~2,497 (Jun), then drifts ~2,340–2,410 through Aug–Oct. Suggests seasonality / market dynamics worth modeling.

### Distribution comparisons (KS two-sample tests)

| Feature | KS statistic | p-value | Assessment |
|---|---:|---:|---|
| `distance` | 0.007 | 0.70 | Similar |
| Lat/lon columns | 0.010–0.012 | 0.10–0.28 | Similar |
| `quote_signal` | 0.046 | 4.8e-18 | Statistically different (small effect) |
| `weight` (complete cases) | 0.015 | 0.033 | Slightly different |
| `market_index` (complete cases) | 0.483 | ~0 | **Large shift** |

**`market_index` shift detail:**

| Period | Mean |
|---|---:|
| Train overall | 1.083 |
| Validation overall | 0.927 |
| Train Jan 2025 | 0.927 |
| Train Oct 2025 | 0.958 |
| Val Nov 2025 | 0.919 |
| Val Dec 2025 | 0.935 |

Validation `market_index` resembles early-2025 training levels, not mid-year peaks (~1.3 in May–Jun). Models must generalize across this regime change.

### Equipment mix (similar proportions)

| Equipment | Train | Validation |
|---|---:|---:|
| Dry Van | 27,202 (56.7%) | 6,780 (56.5%) |
| Reefer | 12,045 (25.1%) | 3,051 (25.4%) |
| Flatbed | 8,753 (18.2%) | 2,169 (18.1%) |

### Target correlations (train)

| Feature | Correlation with `posted_rate` |
|---|---:|
| `distance` | **+0.909** |
| `weight` | +0.035 |
| `market_index` | +0.034 |
| `quote_signal` | −0.040 |
| Coordinates | −0.09 to −0.26 |

---

## score.py Behavior

`score.py` is a **format validator and chart generator**, not a metric calculator.

### CLI

```powershell
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

Optional: `--output-dir scorer_results` (default).

### Validation predictions checks

1. Exactly two columns in order: `load_id`, `predicted_rate`
2. Exactly 12,000 rows
3. No missing or duplicate `load_id`
4. IDs must match `{TE-000001, …, TE-012000}` exactly
5. `predicted_rate` must be numeric, finite, and **strictly positive**

### December predictions checks

1. Columns in order: `pickup`, `delivery`, `distance`, `equipment`, `weight`, `date`, `predicted_rate`
2. Exactly 31 rows, one per day 2025-12-01 … 2025-12-31
3. Fixed inputs for all rows:
   - Pickup: **Lexington**
   - Delivery: **Fort Wayne**
   - Distance: **360**
   - Equipment: **Dry Van**
   - Weight: **32,000**
4. Only `date` and `predicted_rate` should vary
5. `predicted_rate` must be positive

On success, prints confirmation and writes `scorer_results/candidate_december.png`.

---

## Evaluation Metric

**`score.py` does not compute prediction accuracy.** Final line of output:

> *"Final validation metrics are calculated by Spotter after submission."*

Local development should use held-out labels from `train-test.csv` (e.g., Sep–Oct 2025 chronological split). Existing project code (`src/metrics.py`, `src/config.py`) uses **MAE** as the primary local comparison metric, with RMSE, MAPE, and R² as secondary metrics.

The official Spotter submission metric is **not specified** in the provided files.

---

## December Chart Behavior

1. Candidate fills `predicted_rate` in `december-chart-inputs.csv` for 31 fixed-scenario rows.
2. `score.py` validates the file structure and fixed inputs.
3. `save_december_chart()` plots `predicted_rate` vs `date` (Dec 1–31, 2025) as a line chart with filled area.
4. Output: `scorer_results/candidate_december.png`
5. Title: *"Candidate: December 2025 Predicted Load Rate"*
6. Subtitle documents fixed route: Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb

Current repo state: `december-chart-inputs.csv` contains placeholder constant **2030.76** for all 31 days (equals training median `posted_rate`).

---

## Potential Leakage Risks

| Column | Risk | Recommendation |
|---|---|---|
| `load_id` | Identifier only; prefix encodes dataset split | **Exclude from features** |
| `posted_rate` | Target variable | **Exclude from features** (not in validation anyway) |
| `pickup`/`delivery` + coordinates | Redundant (1:1 city→coords) | Use one representation; not leakage |
| `distance` | High target correlation (0.91) but available at inference | Legitimate feature; primary driver |
| `market_index` | Available in validation; temporal/market signal | Legitimate if treated as exogenous input; watch distribution shift |
| `quote_signal` | Available in validation | Legitimate if treated as exogenous input |
| `date` | Strong temporal structure | Use calendar features; avoid random shuffled CV |

**No direct target leakage columns** were found in the validation feature set. The main generalization risks are **temporal drift** (especially `market_index`) and **unseen cities**.

---

## Columns That Must NOT Be Used as Model Features

| Column | Reason |
|---|---|
| `load_id` | Unique identifier; encodes train vs test split (`TR-` vs `TE-`) |
| `posted_rate` | Target variable (training only) |

All other columns in `validation.csv` are candidate features, subject to cleaning decisions for `weight` and `market_index` missing/invalid values.

---

## Pre-existing Work (informational)

Prior sessions added `src/`, baseline scripts, and `artifacts/model_comparison.csv` (chronological split Jan–Aug / Sep–Oct with MAE results). Placeholder submission files exist:

- `validation_predictions.csv` — all rows predict 2030.76
- `december-chart-inputs.csv` — all 31 days predict 2030.76

These are **not** final submissions. Phase 0 did not re-run or modify them.

---

## Recommended Next Steps (Phase 1)

1. **Resolve path naming** — align code/docs with actual root-level hyphenated CSV paths (or create `data/` symlinks/copies if submission requires documented layout).
2. **Chronological CV** — train on Jan–Aug 2025, validate on Sep–Oct 2025 within `train-test.csv`; hold Nov–Dec in `validation.csv` for final inference only.
3. **Clean `weight`** — investigate negative values; impute missing (300 train / 165 val rows).
4. **Impute `market_index`** — 374 train / 249 val missing; account for distribution shift to Nov–Dec.
5. **Handle unseen cities** — ensure encoders ignore unknown categories; consider coordinate-based features as fallback.
6. **Feature engineering** — date decomposition, haversine/road distance, route encoding (existing pipeline in `src/features.py`).
7. **Baseline → model iteration** — start with distance-based baselines, then tree/linear models; track MAE on Sep–Oct holdout.
8. **Generate submissions** — fill `validation_predictions.csv` and `december-chart-inputs.csv`; run `score.py` to produce chart.
9. **Report + Loom** — document split strategy, EDA, model choice, and include `candidate_december.png`.

---

## Phase 0 Readiness Verdict

**Ready for Phase 1:** Yes.

All required inputs are present and understood. Key blockers to address early: path discrepancies, `weight`/`market_index` data quality, temporal holdout strategy, and `market_index` distribution shift between development and final validation periods.
