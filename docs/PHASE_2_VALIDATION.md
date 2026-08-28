# Phase 2 — Validation and Baseline Modeling

**Date:** 2025-08-27  
**Scope:** Chronological validation, preprocessing pipeline, baseline models, feature ablation. No final submission files generated.

**Reproduce:** `python scripts/run_phase2_validation.py`  
**Artifacts:** `artifacts/phase2_model_comparison.csv`, `artifacts/phase2_feature_ablation.csv`

---

## 1. Validation Design

| Rule | Implementation |
|---|---|
| Primary split | Train Jan–Aug 2025, validate Sep–Oct 2025 |
| Sensitivity split | Train Jan–Sep 2025, validate Oct 2025 only |
| No random shuffle | Date-based masks in `src/data.py` |
| No validation.csv | Never loaded for fitting or tuning |
| Preprocessing | Fit on training fold only (`FeatureEngineeringPipeline`) |
| Primary metric | MAE (dollars) |

---

## 2. Primary Chronological Split

| | Train | Validation |
|---|---:|---:|
| **Rows** | 38,477 | 9,523 |
| **Date range** | 2025-01-01 → 2025-08-31 | 2025-09-01 → 2025-10-31 |
| **Target mean** | $2,369.41 | $2,392.45 |
| **Target median** | $2,026.04 | $2,044.32 |
| **Target std** | $1,476.50 | $1,526.13 |

Validation target mean is ~$23 higher than training (+1.0%) — modest shift, consistent with Phase 1 monthly trends.

**Feature distribution notes (train vs Sep–Oct val within labeled data):**

- `distance`, coordinates: stable (Phase 1 KS ≈ 0.01)
- `market_index` val mean (~0.926) aligns with Sep–Oct training mean — regime match
- `weight` / `quote_signal`: minor shifts

---

## 3. Sensitivity Split

| | Train | Validation |
|---|---:|---:|
| **Rows** | 43,147 | 4,853 |
| **Date range** | 2025-01-01 → 2025-09-30 | 2025-10-01 → 2025-10-31 |
| **Target mean** | $2,373.41 | $2,379.05 |
| **Target median** | $2,029.70 | $2,035.90 |

Single-month validation (October only) is noisier but conclusions are directionally stable (see Section 11).

---

## 4. Preprocessing

Implemented in `src/features.py` — all learned statistics fit on training fold only.

### Weight

1. Negative values → NaN  
2. `weight_is_missing` indicator (1 if originally missing or negative)  
3. Impute with equipment median → global median fallback  

### market_index

1. `market_index_is_missing` indicator  
2. Impute with calendar-month median → global median fallback  

### Categorical

- `OneHotEncoder(handle_unknown="ignore")` for pickup, delivery, equipment, route  
- Unseen validation cities will not error at inference  

### Excluded

- `load_id`, `posted_rate`, raw `date` (replaced by calendar features)

---

## 5. Feature Engineering

Full feature set (`full` / `G`) includes:

| Group | Features |
|---|---|
| Distance | `distance`, `log_distance`, `distance_bin` |
| Equipment | OHE `equipment` |
| Weight | cleaned `weight`, `weight_is_missing` |
| Market | `market_index`, `market_index_is_missing`, `quote_signal` |
| Calendar | month, day_of_week, day_of_month, week_of_year, quarter |
| Cities / route | OHE `pickup`, `delivery`, `route` |
| Geographic | lat/lon, `haversine_miles`, `distance_vs_haversine`, `weight_per_mile` |

No target-derived encodings used in Phase 2.

---

## 6. Business Baselines

Primary split validation MAE:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Global median | 1,148.92 | 1,569.42 | −0.058 |
| Equipment median | 1,144.88 | 1,561.59 | −0.047 |
| Distance linear (OLS) | 196.95 | 654.42 | 0.816 |
| Distance + equipment linear | 155.68 | 642.68 | 0.823 |
| Equipment distance × median RPM | 229.10 | 670.66 | 0.807 |

**Takeaway:** Ignoring distance fails completely (R² ≈ 0). Distance alone explains most variance (MAE ~197). Equipment adds ~$41 over distance-only linear model.

---

## 7. Ridge

| Split | MAE | RMSE | R² | Train time |
|---|---:|---:|---:|---:|
| Primary | 148.06 | 642.20 | 0.823 | 7.8s |
| Sensitivity | 148.92 | 657.63 | 0.815 | 8.2s |

Stable across splits. Beats business baselines by ~$8–48 but trails tree models.

---

## 8. Tree Baselines

### HistGradientBoosting (full features)

| Split | MAE | RMSE | R² | Train time |
|---|---:|---:|---:|---:|
| Primary | **129.69** | 633.68 | 0.828 | 31.6s |
| Sensitivity | 151.99 | 659.03 | 0.814 | 31.2s |

### ExtraTrees (full features)

| Split | MAE | RMSE | R² | Train time |
|---|---:|---:|---:|---:|
| Primary | 142.86 | 654.32 | 0.816 | 527.9s |
| Sensitivity | **146.42** | 672.48 | 0.806 | 627.2s |

ExtraTrees is ~8× slower with dense OHE (~4k route columns). Best on sensitivity split; second on primary.

---

## 9. Gradient Boosting

### CatBoost

**Not evaluated.** `catboost` is not in `requirements.txt` and installation failed (network interrupt during download). Documented for Phase 3 — native categorical handling would reduce OHE memory pressure for pickup/delivery/route.

### Log-target diagnostic (HistGB, primary only)

| Target transform | Val MAE ($) | Val RMSE | Val R² |
|---|---:|---:|---:|
| Normal | 129.69 | 633.68 | 0.828 |
| log1p → expm1 | **118.83** | 636.63 | 0.826 |

Log-target improves **dollar-scale MAE by ~$11** on this split. RMSE slightly worse (heavy-tail sensitivity). Worth Phase 3 comparison but not selected as primary pipeline yet.

---

## 10. Feature Ablation

HistGradientBoosting on primary split, incremental feature groups:

| Set | Features added | Val MAE |
|---|---|---:|
| A | distance | 192.24 |
| B | + equipment | 139.39 |
| C | + weight | **128.13** |
| D | + market_index, quote_signal | 138.19 |
| E | + calendar | 134.64 |
| F | + pickup, delivery, route | 133.76 |
| G | + log_distance, bins, geo | 129.69 |

**Incremental interpretation:**

| Question | Answer |
|---|---|
| Distance alone | MAE 192.24 — explains bulk of signal |
| Equipment adds | −52.9 MAE (A→B) |
| Weight adds | −11.3 MAE (B→C) |
| market_index + quote_signal | **+10.1 MAE (C→D)** — hurt on this split |
| Calendar adds | −3.5 MAE (D→E) — partial recovery |
| Route/city adds | −0.9 MAE (E→F) |
| Geographic adds | −4.1 MAE (F→G) |

**Note:** Set C (without market/route/geo) slightly beats full set G (128.13 vs 129.69). Simpler models may generalize better; market_index addition unexpectedly degraded ablation step D — investigate in Phase 3 (regime interaction vs noise).

---

## 11. Results

### Primary split — all models (sorted by MAE)

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| HistGB log1p (diagnostic) | 118.83 | 636.63 | 0.826 |
| HistGradientBoosting | 129.69 | 633.68 | 0.828 |
| ExtraTrees | 142.86 | 654.32 | 0.816 |
| Ridge | 148.06 | 642.20 | 0.823 |
| Distance + equipment linear | 155.68 | 642.68 | 0.823 |
| Distance linear | 196.95 | 654.42 | 0.816 |
| Equipment distance RPM | 229.10 | 670.66 | 0.807 |
| Equipment median | 1,144.88 | 1,561.59 | −0.047 |
| Global median | 1,148.92 | 1,569.42 | −0.058 |

### Sensitivity split — all models (sorted by MAE)

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| ExtraTrees | 146.42 | 672.48 | 0.806 |
| Ridge | 148.92 | 657.63 | 0.815 |
| HistGradientBoosting | 151.99 | 659.03 | 0.814 |
| Distance + equipment linear | 157.47 | 657.21 | 0.815 |
| Distance linear | 192.77 | 668.21 | 0.809 |
| Equipment distance RPM | 231.30 | 683.87 | 0.800 |
| Equipment median | 1,140.21 | 1,558.82 | −0.040 |
| Global median | 1,146.79 | 1,567.97 | −0.052 |

**Sensitivity supports primary conclusions:**

- ML ≫ median baseline (~$1,000 MAE improvement)
- Distance is essential
- Tree/linear ML beats business baselines
- Model ranking shifts slightly (ExtraTrees best on Oct-only val) — expect some instability with 4,853-row validation

---

## 12. Leakage Controls

| Control | Status |
|---|---|
| `load_id` excluded | ✓ |
| `posted_rate` excluded from features | ✓ |
| `validation.csv` not used | ✓ |
| Preprocessing fit on train fold only | ✓ (`WeightCleaner`, `MarketIndexImputer` in fitted pipeline) |
| No target encoding | ✓ |
| Chronological split (no shuffle) | ✓ |
| Business baselines use train-fold statistics only | ✓ |

No leakage concerns identified in Phase 2 pipeline.

---

## 13. Initial Model Recommendation

**Primary recommendation for Phase 3:** `HistGradientBoostingRegressor` with Phase 2 preprocessing.

**Feature set:** Start from **set C** (distance + equipment + weight) or **set G** (full) — ablation shows C marginally better; full set adds geographic/route signal with modest MAE cost.

**Also pursue:**

1. Log1p target variant — validated $11 MAE gain on primary split  
2. CatBoost with native categoricals — if installable; may reduce OHE cost  
3. Light tuning of HistGB depth/iterations/learning_rate only (not exhaustive search)

**Deprioritize:** Global/equipment median baselines, ExtraTrees for production (slow, mixed split performance).

---

## 14. Phase 3 Plan

1. Update `src/features.py` with optional feature-set parameter in training script  
2. Resolve CatBoost dependency (or justify sklearn-only path)  
3. Hyperparameter tuning for HistGB (and optionally CatBoost) on primary split only  
4. Investigate market_index ablation regression (step D) — interaction terms or monotonic constraint  
5. Retest log1p target on sensitivity split  
6. Train final model on full Jan–Oct labeled data for Nov–Dec inference  
7. Generate `validation_predictions.csv` and December chart predictions  
8. Run `score.py` for format validation

---

## Code Changes (Phase 2)

| File | Change |
|---|---|
| `src/config.py` | Sensitivity dates, feature set definitions |
| `src/data.py` | Flexible splits, `summarize_split()` |
| `src/features.py` | Weight/market cleaning, ablation-ready pipeline |
| `src/baselines.py` | Business + ML model specs, log-target wrapper |
| `src/experiments.py` | Experiment runner |
| `scripts/run_phase2_validation.py` | CLI entry point |
