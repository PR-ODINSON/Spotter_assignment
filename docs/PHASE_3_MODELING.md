# Phase 3 — Model Optimization

**Date:** 2025-08-27  
**Scope:** Log-target verification, market_index investigation, feature-set comparison, HistGB tuning, diagnostics. No final training or submission files.

**Reproduce:** `python scripts/run_phase3.py`  
**Artifacts:** `artifacts/phase3_*.csv`

---

## 1. Objective

Find the strongest model likely to generalize from Jan–Oct 2025 development data to the Nov–Dec 2025 holdout (`validation.csv`), using:

- Primary validation: train Jan–Aug → validate Sep–Oct  
- Sensitivity validation: train Jan–Sep → validate Oct  
- Dollar-scale MAE as the primary metric  
- Stability between splits weighted equally with raw MAE  

`validation.csv` was **not** used for any tuning or selection.

---

## 2. Current Baseline

Phase 2 reference (HistGB, FULL features, normal target):

| Split | MAE |
|---|---:|
| Primary | 129.69 |
| Sensitivity | 151.99 |
| Stability gap | +22.3 |

Global median baseline: MAE 1,148.92 (primary).

---

## 3. Log-Target Investigation

HistGradientBoosting, feature set **FULL**, Phase 2 hyperparameters:

| Target | Split | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Normal | Primary | 129.69 | 633.68 | 0.828 |
| Normal | Sensitivity | 151.99 | 659.03 | 0.814 |
| **log1p → expm1** | **Primary** | **118.83** | 636.63 | 0.826 |
| **log1p → expm1** | **Sensitivity** | **115.45** | 648.30 | 0.820 |

### Conclusions

- log1p **genuinely improves dollar-scale MAE** on both splits (−10.9 primary, −36.5 sensitivity).
- Improvement is **stable** — sensitivity MAE is actually *lower* than primary (gap −3.4 vs +22.3 normal).
- log1p **does not fix high-rate underprediction** — top-1% mean residual ~3,831 (primary) vs ~3,665 for normal Q; rate > $5,000 residual ~893 vs ~710 for Q normal.
- log1p compresses heavy-tail errors in loss space but still underpredicts extreme loads in dollars.

**Decision:** log1p target is justified for overall MAE and split stability. High-haul bias remains a known risk.

---

## 4. Market Index Investigation

Controlled HistGB experiments on base set C (distance + equipment + weight):

| Feature set | Primary MAE | Sensitivity MAE |
|---|---:|---:|
| **C** (baseline) | **128.13** | **129.49** |
| C + market_index | 151.46 | 138.17 |
| C + quote_signal | **118.90** | **124.54** |
| C + market_index + quote_signal | 138.19 | 134.00 |
| C + market_index + calendar | 136.92 | 158.51 |
| C + market_index × month | 142.94 | 145.68 |
| C + quote_signal + calendar | 131.99 | 152.65 |

### Findings

- **Raw `market_index` hurts consistently on primary** (+23 MAE vs C alone). Confirms Phase 2 ablation regression (128 → 138 when adding market_index + quote_signal vs C at 128 with weight).
- **`market_index` is not redundant with date** — calendar and interaction features do not recover performance.
- **`quote_signal` alone helps substantially** (−9.2 MAE primary, −4.9 sensitivity vs C).
- Adding both market_index and quote_signal is worse than quote alone — market_index **poisons** the combination.
- Likely cause: weak marginal signal (r ≈ 0.03), regime-sensitive imputation, and tree splits on noisy index values that don't generalize to Sep–Oct despite similar means.

### Segment evidence (market_index regime, HistGB C4 primary)

| Regime | MAE | Mean residual |
|---|---:|---:|
| Low market_index | 120.3 | −5.4 |
| Mid | 129.6 | +20.0 |
| High | 136.5 | +55.6 |

High market_index rows are systematically underpredicted when market_index is included in richer models.

**Decision:** **Do not retain raw `market_index`.** Retain **`quote_signal`**.

---

## 5. Feature Set Experiments

HistGB, normal target, both splits (`artifacts/phase3_feature_sets.csv`):

| Set | Description | Primary MAE | Sensitivity MAE | Gap |
|---|---|---:|---:|---:|
| **Q** | C4 + quote_signal | **118.95** | **123.91** | +5.0 |
| G | C4 + geographic + route | 127.44 | 127.62 | +0.2 |
| C / C2 | distance + equip + weight | 128.13 | 129.49 | +1.4 |
| C4 | + log_distance + bins | 128.71 | 129.25 | +0.5 |
| R | C4 + route/cities | 129.03 | 130.98 | +2.0 |
| FULL | all features | 129.69 | 151.99 | +22.3 |
| M | C4 + market_index | 152.58 | 138.11 | −14.5 |
| MC | C4 + market_index + calendar | 136.71 | 146.59 | +9.9 |

### Conclusions

| Question | Answer |
|---|---|
| log_distance / bins | No improvement over C alone (C2 = C) |
| quote_signal | **Strongest single addition** (Q best primary normal) |
| market_index | **Harmful** (M worst primary) |
| calendar | Modest; E = 135 primary |
| route/city (R) | +0.9 MAE vs E; unstable on sensitivity |
| geographic (G) | Stable gap (+0.2) but +8.5 MAE vs Q |
| FULL | Overfits route/geo — sensitivity gap +22.3 |

---

## 6. HistGradientBoosting Tuning

Feature set **C4**, normal target, primary split (`artifacts/phase3_histgb_tuning.csv`):

| Config | Primary MAE |
|---|---:|
| **max_depth=6, l2=0.1** | **126.58** |
| min_samples_leaf=50 | 127.42 |
| max_depth=6 | 127.46 |
| Baseline (depth=8) | 128.71 |

Best tuned config on sensitivity: **127.44 MAE** (max_depth=6, l2=0.1) — modest improvement, still worse than Q or log1p.

Tuning does not close the gap to quote_signal or log1p transforms. Prefer feature/transform choices over heavy hyperparameter search.

---

## 7. CatBoost Experiment

**Not run.** `catboost` is not installed; import failed. Installation was attempted in Phase 2 and blocked by network interruption. Per instructions, no repeated install attempts.

**XGBoost / LightGBM:** Not available in the environment.

---

## 8. Additional Model Experiments

No additional gradient boosting libraries evaluated. Sklearn HistGB remains the sole ML candidate.

---

## 9. Segment Error Analysis

`artifacts/phase3_segment_errors.csv` — HistGB C4 primary Sep–Oct:

| Segment | Count | MAE | Mean residual |
|---|---:|---:|---:|
| Distance 0–300 mi | 780 | 48.6 | −30.6 (overpredict short) |
| Distance 300–600 | 1,944 | 71.9 | +8.1 |
| Distance 600–1,200 | 3,202 | 104.7 | +1.2 |
| Distance 1,200–2,000 | 2,063 | 175.5 | +57.1 |
| Distance 2,000+ | 1,534 | 228.6 | +66.8 |
| Reefer | 2,393 | 136.1 | +34.9 |
| Flatbed | 1,770 | 126.3 | +16.0 |
| Dry Van | 5,360 | 126.2 | +19.4 |
| September | 4,670 | 126.7 | +19.7 |
| October | 4,853 | 130.7 | +25.6 |

**Failure modes:** Long-haul underprediction; slight October drift; Reefer hardest equipment type.

HistGB FULL normal has larger long-haul residuals (2000+ MAE 261, residual +81) than C4.

---

## 10. High-Rate Error Analysis

Mean residual (actual − predicted); positive = underprediction:

| Model | Split | Top 1% | Distance > 2,000 | Rate > $5,000 |
|---|---|---:|---:|---:|
| C4 normal | Primary | +3,704 | +67 | +764 |
| Q normal | Primary | +3,665 | +35 | +710 |
| log1p FULL | Primary | +3,831 | +161 | +893 |
| log1p FULL | Sensitivity | +3,960 | +44 | +814 |

All models underpredict extreme rates. log1p FULL improves overall MAE but **worsens** high-rate bias on primary. Q normal has the best high-rate behavior among tested configs.

---

## 11. Residual Analysis

HistGB C4 primary — residual correlations:

| Feature | Correlation with residual |
|---|---:|
| distance | +0.31 (underpredict long) |
| market_index | +0.08 |
| quote_signal | −0.05 |
| weight | +0.04 |

Equipment: Reefer mean residual +35, Flatbed +16, Dry Van +19.

Systematic positive residual vs distance confirms need for nonlinear distance handling (already partially addressed by tree model + log1p).

---

## 12. Candidate Models

Ranked by generalization (primary MAE + stability), not primary MAE alone:

| Rank | Model | Feature set | Target | Primary MAE | Sensitivity MAE | Gap | Risk |
|---|---|---|---|---:|---:|---:|---|
| **1** | HistGB | FULL | **log1p** | **118.83** | **115.45** | **−3.4** | High-rate underprediction; FULL includes unused harmful market_index in pipeline |
| **2** | HistGB | **Q** | normal | 118.95 | 123.91 | +5.0 | Best validated normal config; best high-rate behavior |
| **3** | HistGB | C4 | normal | 128.71 | 129.25 | +0.5 | Most stable normal; higher MAE |

---

## 13. Final Model Recommendation

### Recommended for Phase 4 final training

**HistGradientBoostingRegressor** with:

| Setting | Value |
|---|---|
| Target transform | **log1p → expm1** |
| Feature set | **Q** (C4 + quote_signal) |
| Hyperparameters | **max_depth=6, l2_regularization=0.1**, learning_rate=0.08, max_iter=300, random_state=42 |
| Exclude | market_index, route OHE, geographic extras |

See **Final Candidate Verification** below for full evidence. The log1p + Q combination is now confirmed on both splits.

### Alternative (conservative)

HistGB + Q + normal target (118.95 / 123.91) if high-rate tail accuracy is prioritized over overall MAE.

---

## 14. Why This Model Should Generalize to Nov–Dec

1. **Sep–Oct validation proxy** — best configs align with Nov–Dec `market_index` regime (~0.93); we exclude the harmful market_index feature entirely.  
2. **Split stability** — log1p FULL gap −3.4; Q normal gap +5.0. Both far better than FULL normal (+22.3).  
3. **Simplicity** — Q uses 7 numeric + 1 categorical column after engineering; no high-cardinality route OHE.  
4. **Unseen cities** — equipment + distance + quote_signal + weight do not depend on city identity; OHE only on equipment (3 levels).  
5. **Leakage-safe preprocessing** — weight/market cleaning fit on train fold only (market_index excluded from features).  

---

## 15. Remaining Risks

1. **Systematic underprediction of top-1% / long-haul loads** — log1p + Q improves long-haul MAE vs alternatives but mean residual on 2000+ miles remains +94 (primary).  
2. **CatBoost not evaluated** — native categoricals might help unseen cities; environment limitation.  
3. **Official Spotter metric unknown** — local MAE may not match submission scoring.  
4. **October-only sensitivity is noisy** (4,853 rows) — conclusions directionally supported but not definitive.  
5. **Distance-correction not validated** — train-fold linear residual slope ≈ 0.028 $/mile is too weak to justify post-hoc correction.

---

## Final Candidate Verification

**Reproduce:** `python scripts/run_phase3_verification.py`  
**Artifacts:** `artifacts/phase3_final_candidates.csv`, `artifacts/phase3_verification/`

This section confirms the previously untested **log1p + Q** combination before Phase 4 final training.

### Experiment 1 — log1p + Q vs baselines

HistGB with max_depth=8, learning_rate=0.08, max_iter=300, random_state=42:

| Config | Target | Features | Primary MAE | Sensitivity MAE | Gap | Primary R² | Sensitivity R² |
|---|---|---|---:|---:|---:|---:|---:|
| **log1p + Q** | log1p | Q | **106.93** | **113.87** | +6.9 | 0.829 | 0.819 |
| log1p + FULL | log1p | FULL | 118.83 | 115.45 | −3.4 | 0.826 | 0.820 |
| Q + normal | normal | Q | 118.95 | 123.91 | +5.0 | 0.828 | 0.819 |
| C4 + normal | normal | C4 | 128.71 | 129.25 | +0.5 | 0.828 | 0.820 |

**Conclusion:** log1p + Q is the strongest configuration on **both** splits. It beats log1p FULL by **−11.9 primary / −1.6 sensitivity MAE** and beats Q normal by **−12.0 / −10.0 MAE**. The combination is validated — not a Phase 3 hypothesis anymore.

Mean residual (primary): log1p+Q +46.5 vs normal Q +16.2 — log transform shifts the global bias upward but improves dollar MAE across segments.

### Experiment 2 — log1p + Q depth tuning

Small grid: max_depth ∈ {5, 6, 8}, l2 ∈ {0, 0.1}, max_iter=300. Primary selection, top configs evaluated on sensitivity:

| Config | Primary MAE | Sensitivity MAE | Gap |
|---|---:|---:|---:|
| **d=6, l2=0.1** | **106.83** | **113.29** | **+6.5** |
| d=8, l2=0.1 | 106.73 | 113.41 | +6.7 |
| d=8, l2=0.0 (baseline) | 106.93 | 113.87 | +6.9 |
| d=5, l2=0.1 | 107.87 | 115.58 | +7.7 |

**Conclusion:** Mild regularization (depth 6, l2=0.1) yields the best generalization score (combined = 110.06). Differences among top-3 configs are < 0.3 MAE — prefer the simpler d=6 model.

### Experiment 3 — Long-haul diagnostics

Primary split distance-bucket MAE (2000+ miles) and high-rate segments:

| Config | 0–300 | 300–600 | 600–1200 | 1200–2000 | **2000+** | Top 1% MAE | Top 1% residual | Rate > $5k residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| log1p Q (d6, l2=0.1) | 26.7 | 55.2 | 87.6 | 154.9 | **188.5 (+94)** | 3750 | +3735 | +805 |
| log1p Q (d8) | 26.8 | 55.1 | 87.3 | 154.7 | 190.0 (+90) | 3748 | +3735 | +801 |
| log1p FULL | 25.5 | 58.1 | 95.7 | 167.9 | 225.4 (+161) | 3832 | +3831 | +893 |
| Q normal | 38.7 | 62.8 | 94.5 | 171.7 | 211.1 (+35) | 3732 | +3665 | **+710** |

**Conclusion:** log1p + Q **does not sacrifice long-haul MAE** — it improves 2000+ MAE by ~37 vs log1p FULL and ~23 vs Q normal. High-rate mean residual is slightly worse than Q normal (+805 vs +710) but much better than log1p FULL (+893). The log1p + Q trade-off is favorable: large overall MAE gain with acceptable tail behavior.

Sensitivity 2000+ MAE: log1p Q d6 = 213 vs normal Q = 234 vs log1p FULL = 221.

### Experiment 4 — Residual vs distance (no correction applied)

For log1p Q d6 l2=0.1 on primary validation:

| Distance bucket | Mean actual | Mean predicted | Mean residual | MAE |
|---|---:|---:|---:|---:|
| 0–300 | 552 | 552 | −0.2 | 26.7 |
| 300–600 | 1,093 | 1,066 | +26 | 55.2 |
| 600–1,200 | 1,930 | 1,902 | +28 | 87.6 |
| 1,200–2,000 | 3,276 | 3,194 | +82 | 154.9 |
| 2,000+ | 4,753 | 4,659 | +94 | 188.5 |

Train-fold linear fit: residual ≈ 0.028 × distance + 0.49 ($/mile). Slope is too small and validation-bucket pattern is nonlinear (residual jumps at 1200+ miles). A simple `predicted + f(distance)` correction is **not justified** without a leakage-safe cross-validated calibration — documented only, not implemented.

Plot: `artifacts/phase3_verification/residual_vs_distance_log1p_Q_d6_l20.1.png`

### Final candidate ranking

Ranked by combined score (primary MAE + 0.5 × max(gap, 0)), prioritizing generalization (`artifacts/phase3_final_candidates.csv`):

| Rank | Config | Primary | Sensitivity | Gap | Long-haul MAE | Top 1% MAE |
|---|---|---:|---:|---:|---:|---:|
| **1** | **log1p Q d6 l2=0.1** | **106.83** | **113.29** | +6.5 | 188.5 | 3750 |
| 2 | log1p Q d8 l2=0.1 | 106.73 | 113.41 | +6.7 | 189.2 | 3746 |
| 3 | log1p Q d6 l2=0.0 | 107.17 | 113.52 | +6.3 | 188.0 | 3750 |
| 4 | log1p Q (baseline d8) | 106.93 | 113.87 | +6.9 | 190.0 | 3748 |
| 8 | log1p FULL | 118.83 | 115.45 | −3.4 | 225.4 | 3832 |
| 9 | Q normal | 118.95 | 123.91 | +5.0 | 211.1 | 3732 |

### Final recommended model

**FINAL MODEL A: HistGB + log1p + Q**

| Setting | Value |
|---|---|
| Model | HistGradientBoostingRegressor |
| Feature set | Q (distance, log_distance, distance_bin, weight, weight_is_missing, quote_signal, equipment) |
| Target | log1p → expm1 |
| Hyperparameters | max_depth=6, l2_regularization=0.1, learning_rate=0.08, max_iter=300, random_state=42 |

**Why not FINAL MODEL B (log1p FULL)?** FULL is 12 MAE worse on primary and includes harmful market_index/route/geo features that drove +22 sensitivity gap in Phase 3.

**Why not FINAL MODEL C (Q normal)?** Normal target Q is 12 MAE worse on primary and 10 MAE worse on sensitivity. Only advantage: slightly better high-rate mean residual (+710 vs +805).

**Why depth 6 over depth 8?** Nearly identical MAE; depth 6 + l2=0.1 has the smallest stability gap among top configs and is simpler.

---

## Artifacts

| File | Contents |
|---|---|
| `phase3_feature_sets.csv` | Feature set comparison, both splits |
| `phase3_histgb_tuning.csv` | Hyperparameter grid results |
| `phase3_segment_errors.csv` | Segment-level MAE for top models |
| `phase3_candidates.csv` | Ranked candidate configurations (Phase 3 main run) |
| `phase3_final_candidates.csv` | Final verification ranking with long-haul metrics |
| `phase3_verification/` | Experiment CSVs and residual plot |
