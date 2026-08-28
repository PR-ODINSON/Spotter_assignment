# Freight Rate Prediction

Machine learning assessment: predict freight load rates (`posted_rate`) from shipment attributes.

## Overview

This repository implements a leakage-safe regression pipeline for the Spotter freight rate challenge. The final model predicts USD freight rates for 12,000 Nov–Dec 2025 holdout loads using 48,000 labeled Jan–Oct 2025 development rows.

**Final model:** `HistGradientBoostingRegressor` with feature set **Q** and **log1p** target transform.

**Development chronological validation MAE** (not Nov–Dec holdout accuracy):

| Split | MAE |
|---|---:|
| Primary (Jan–Aug → Sep–Oct) | 106.83 |
| Sensitivity (Jan–Sep → Oct) | 113.29 |

Nov–Dec holdout labels are not available locally. `score.py` validates submission format only — it does not compute prediction accuracy.

## Dataset

| File | Rows | Description |
|---|---:|---|
| `train-test.csv` | 48,000 | Labeled development data (Jan–Oct 2025) |
| `validation.csv` | 12,000 | Unlabeled holdout features (Nov–Dec 2025) |
| `validation-predictions-template.csv` | 12,000 | Submission template |
| `december-chart-inputs.csv` | 31 | Fixed December chart scenario |

> CSVs live at the repository root (hyphenated names), not under `data/`.

## Approach

1. Exploratory analysis and data-quality fixes (negative/missing weight)
2. Leakage-safe chronological validation on labeled data
3. Feature ablation and model comparison (baselines → Ridge → HistGB)
4. log1p target + feature set Q selection
5. Final training on full Jan–Oct labeled data
6. Nov–Dec inference and `score.py` validation

## Validation Strategy

No random shuffling. Two chronological splits on `train-test.csv`:

- **Primary:** train Jan–Aug → validate Sep–Oct
- **Sensitivity:** train Jan–Sep → validate Oct

Preprocessing statistics are fit on the training fold only. `validation.csv` is never used for tuning.

## Model

| Setting | Value |
|---|---|
| Algorithm | HistGradientBoostingRegressor |
| Features (Q) | distance, log_distance, distance_bin, weight, weight_is_missing, quote_signal, equipment |
| Target | log1p(posted_rate) → expm1(prediction) |
| Hyperparameters | max_depth=6, l2=0.1, lr=0.08, max_iter=300, random_state=42 |

Excluded from final model: raw `market_index`, route/city OHE, geographic extras.

## Project Structure

```
spotter/
├── train-test.csv
├── validation.csv
├── validation_predictions.csv
├── december-chart-inputs.csv
├── score.py
├── requirements.txt
├── README.md
├── src/                    # Core library
├── scripts/                # Runnable pipelines
├── artifacts/              # Experiment CSV outputs
├── reports/                # EDA plots + final report
├── docs/                   # Phase documentation
└── scorer_results/         # candidate_december.png
```

## Installation

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Running the Pipeline

```powershell
# Phase 1 — EDA
python scripts/run_phase1_eda.py

# Phase 2 — Baselines and validation
python scripts/run_phase2_validation.py

# Phase 3 — Model optimization
python scripts/run_phase3.py
python scripts/run_phase3_verification.py

# Phase 4 — Final training and predictions
python scripts/train_final.py

# Phase 5 — Official scoring
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

## Generating Predictions

Regenerate final submission files:

```powershell
python scripts/train_final.py
```

Outputs:
- `validation_predictions.csv` (12,000 rows)
- `december-chart-inputs.csv` (31 rows with model predictions)

## Running score.py

```powershell
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

Validates:
- 12,000 prediction rows, correct IDs, positive numeric rates
- 31 December scenario rows (Lexington → Fort Wayne, 360 mi, Dry Van, 32k lb)

Generates: `scorer_results/candidate_december.png`

## Results

### Development validation (labeled data only)

- Primary MAE: **106.83**
- Sensitivity MAE: **113.29**

These are **development chronological validation** results — not final Nov–Dec holdout scores.

### Final predictions (Nov–Dec holdout)

| Statistic | Value |
|---|---:|
| Mean | $2,345.93 |
| Median | $2,026.45 |
| Min | $210.94 |
| Max | $6,548.41 |

### score.py status

Exit code **0** — all format checks passed, December chart generated.

## Limitations

- No local ground truth for Nov–Dec holdout
- Official Spotter metric unavailable locally
- Long-tail / high-rate underprediction in development validation
- CatBoost not evaluated (not installed)
- December chart is flat ($841.48) because feature set Q excludes calendar features

## Reproducibility

Full phase documentation: `docs/PHASE_0_PROJECT_AUDIT.md` through `docs/PHASE_5_SCORING.md`

Final report: `reports/freight_rate_ml_assessment.docx` (generate with `python scripts/generate_report.py`)

Loom script: `docs/LOOM_SCRIPT.md`

Submission checklist: `docs/GITHUB_CHECKLIST.md`
