# Phase 4 — Final Training and Prediction

**Date:** 2025-08-27  
**Scope:** Final model training on full Jan–Oct labeled data, Nov–Dec holdout inference, December chart inputs. No `score.py` run in this phase.

**Reproduce:** `python scripts/train_final.py`

---

## 1. Locked Model

| Component | Value |
|---|---|
| Model | `HistGradientBoostingRegressor` |
| Feature set | **Q** (C4 + `quote_signal`) |
| Target | `log1p(posted_rate)` → `expm1(prediction)` |
| Hyperparameters | `max_depth=6`, `l2_regularization=0.1`, `learning_rate=0.08`, `max_iter=300`, `random_state=42` |

Phase 3.5 validation (development only — not Nov–Dec holdout):

| Split | MAE |
|---|---:|
| Primary (Jan–Aug → Sep–Oct) | 106.83 |
| Sensitivity (Jan–Sep → Oct) | 113.29 |
| Long-haul 2000+ (primary) | 188.5 |

**Note on feature definition:** Validated feature set Q uses numeric columns `distance`, `weight`, `weight_is_missing`, `log_distance`, `distance_bin`, `quote_signal` and categorical `equipment`. Raw `market_index` is **not** a model feature. The engineering pipeline still computes `market_index_is_missing` internally for other feature sets, but Q does not select it — matching Phase 3.5 experiments exactly.

Excluded from Q: raw `market_index`, route, pickup, delivery, geographic extras, calendar features.

---

## 2. Final Training Dataset

| Setting | Value |
|---|---|
| Source | `train-test.csv` |
| Date range | 2025-01-01 through 2025-10-31 |
| Rows | **48,000** |
| Target mean | $2,373.98 |
| Target median | $2,030.76 |

`validation.csv` was **not** used for fitting, feature selection, or tuning.

---

## 3. Preprocessing

Leakage-safe pipeline (fit on full Jan–Oct training only):

**Weight**
1. Negative values → NaN
2. `weight_is_missing` indicator
3. Impute by equipment median (from full Jan–Oct training)
4. Global training median fallback

**Market index**
- Raw `market_index` is imputed internally (month median) for pipeline consistency but **not passed to the model** under feature set Q.

**Quote signal**
- Used directly when present; missing values handled natively by HistGB.

**Categorical**
- Equipment only (3 levels + OHE with `handle_unknown="ignore"`).

---

## 4. Feature Set Q

| Type | Columns |
|---|---|
| Numeric | `distance`, `weight`, `weight_is_missing`, `log_distance`, `distance_bin`, `quote_signal` |
| Categorical | `equipment` |

Derived during engineering: `log_distance`, `distance_bin`, `weight_is_missing`.

---

## 5. Target Transformation

Training target: `y = log1p(posted_rate)`

Inference: `predicted_rate = expm1(model.predict(X))`

No outlier removal, clipping, or manual adjustment.

---

## 6. Hyperparameters

```python
LOCKED_HISTGB_PARAMS = {
    "max_depth": 6,
    "l2_regularization": 0.1,
    "learning_rate": 0.08,
    "max_iter": 300,
    "random_state": 42,
}
```

Training time on 48,000 rows: ~1.9 seconds.

---

## 7. Validation-to-Final-Training Transition

| Phase | Training data | Purpose |
|---|---|---|
| Phase 2–3.5 | Jan–Aug or Jan–Sep subsets | Model selection, ablation, tuning |
| Phase 4 | **Full Jan–Oct (48,000 rows)** | Final production model |

Development MAE results (106.83 / 113.29) are from chronological validation on labeled subsets. Nov–Dec predictions have **no labels** — accuracy is unknown until external Spotter evaluation.

---

## 8. Validation Prediction Generation

| Setting | Value |
|---|---|
| Input | `validation.csv` (12,000 rows) |
| Date range | 2025-11-01 → 2025-12-31 |
| Output | `validation_predictions.csv` |
| Ordering | Matches `validation-predictions-template.csv` |

Inference uses all raw columns from `validation.csv`. Unseen cities (8) and missing values (165 weight, 249 market_index) did not cause errors.

---

## 9. Prediction Diagnostics

`artifacts/final_prediction_diagnostics.csv`

| Statistic | Predicted (12,000) | Training target (48,000) |
|---|---:|---:|
| Mean | 2,345.93 | 2,373.98 |
| Median | 2,026.45 | 2,030.76 |
| Std | 1,364.05 | 1,486.49 |
| Min | 210.94 | 57.22 |
| Max | 6,548.41 | 25,533.00 |
| Q25 | 1,252.46 | 1,251.55 |
| Q75 | 3,351.80 | 3,330.75 |
| Q95 | 4,920.06 | 4,953.77 |
| Q99 | 5,746.76 | 5,972.83 |

Prediction distribution is aligned with training target distribution. No predictions were modified based on this comparison.

Invalid predictions: **0**

---

## 10. December Prediction Generation

| Setting | Value |
|---|---|
| Input | `december-chart-inputs.csv` |
| Rows | 31 (2025-12-01 → 2025-12-31) |
| Fixed scenario | Lexington → Fort Wayne, 360 mi, Dry Van, 32,000 lb |

Missing columns in December template (`pickup_lat`, `delivery_lat`, `market_index`, `quote_signal`) are filled for pipeline compatibility:
- City coordinates looked up from training data modes
- `quote_signal` left as NaN (HistGB-native missing handling)

Because feature set Q **excludes calendar features**, the fixed scenario produces identical model inputs for every December date except the unused internal date imputation path. All 31 rows predict **$841.48** — a flat line is expected for this locked model, not placeholder 2030.76 values.

---

## 11. Output Verification

### validation_predictions.csv

| # | Check | Result |
|---|---|---|
| 1 | File exists | PASS |
| 2 | Exactly 12,000 rows | PASS |
| 3 | Exactly 2 columns | PASS |
| 4 | Columns = `load_id,predicted_rate` | PASS |
| 5 | `load_id` unique | PASS |
| 6 | No missing `load_id` | PASS |
| 7 | No missing `predicted_rate` | PASS |
| 8 | All predictions numeric | PASS |
| 9 | All predictions finite | PASS |
| 10 | All predictions > 0 | PASS |
| 11 | Load IDs match template set | PASS |
| 12 | Ordering matches template | PASS |
| 13 | No pandas index column | PASS |
| 14 | No target column in file | PASS |

### december-chart-inputs.csv

| Check | Result |
|---|---|
| 31 rows | PASS |
| Dates 2025-12-01 → 2025-12-31 | PASS |
| Fixed pickup/delivery/distance/equipment/weight | PASS |
| Positive finite predictions | PASS |

---

## 12. Reproducibility

Single command:

```powershell
.\venv\Scripts\python.exe scripts\train_final.py
```

Pipeline steps:
1. Verify locked model definition (`src/final_model.py`)
2. Load full Jan–Oct training data
3. Fit log1p HistGB with feature set Q
4. Predict `validation.csv` → `validation_predictions.csv`
5. Predict `december-chart-inputs.csv`
6. Write `artifacts/final_prediction_diagnostics.csv`
7. Run file-format validation checks

Implementation files:
- `src/final_model.py` — training and prediction logic
- `src/config.py` — `LOCKED_HISTGB_PARAMS`, `FINAL_MODEL_FEATURE_SET`
- `scripts/train_final.py` — entry point

**Not run in Phase 4:** `score.py`, final report, GitHub submission.

---

## Remaining Risks

1. Nov–Dec holdout accuracy is **unknown** — no labels available locally.
2. December chart is flat because Q excludes calendar/date features by design.
3. Long-haul / high-rate underprediction observed in development validation may persist.
4. Official Spotter metric may differ from local MAE.
