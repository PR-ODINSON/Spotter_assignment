# Freight Rate Prediction

Machine learning assessment: predict freight load rates (`posted_rate`) from shipment attributes.

## Overview

This repository implements a leakage-safe regression pipeline for the Spotter freight rate challenge. The final model predicts USD freight rates for 12,000 Nov–Dec 2025 holdout loads using 48,000 labeled Jan–Oct 2025 development rows.

**Final model:** `HistGradientBoostingRegressor` with feature set **Q**, **raw** `posted_rate` target, and **`absolute_error`** loss.

**Development chronological validation MAE** (not Nov–Dec holdout accuracy):

| Split | MAE |
|---|---:|
| Primary (Jan–Aug → Sep–Oct) | 93.56 |
| Sensitivity (Jan–Sep → Oct) | 102.57 |

Additional development metrics for the final model: Primary RMSE 630.27 · High-rate MAE 816.54 · Long-haul MAE 164.93.

Earlier candidate (HistGB Q + log1p + squared_error): Primary MAE 106.83 / Sensitivity MAE 113.29 — retained only as historical comparison; **not** the final model.

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
4. Feature set Q selection; log1p evaluated as an earlier candidate
5. Controlled loss/target comparison → adopt raw target + absolute_error
6. Final training on full Jan–Oct labeled data
7. Nov–Dec inference and `score.py` validation

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
| Target | raw posted_rate (direct dollar prediction) |
| Loss | absolute_error |
| Hyperparameters | max_depth=6, l2=0.1, lr=0.08, max_iter=300, random_state=42 |

Excluded from final model: raw `market_index`, route/city OHE, geographic extras. No log1p/expm1 transform in the final pipeline.

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
├── scripts/                # train_final.py, generate_report.py
├── reports/                # EDA plots + final report PDF
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
# Final training and predictions
python scripts/train_final.py

# Official scoring validation
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

Final model (HistGB Q + raw + absolute_error):

- Primary MAE: **93.56**
- Sensitivity MAE: **102.57**
- Primary RMSE: **630.27**
- High-rate MAE: **816.54**
- Long-haul MAE: **164.93**

These are **development chronological validation** results — not final Nov–Dec holdout scores.

Earlier candidate (HistGB Q + log1p + squared_error): Primary MAE 106.83 / Sensitivity MAE 113.29.

### Final predictions (Nov–Dec holdout)

| Statistic | Value |
|---|---:|
| Mean | $2,350.58 |
| Median | $2,015.78 |
| Min | $201.58 |
| Max | $6,652.40 |

December scenario (Lexington → Fort Wayne, fixed inputs): flat at **$815.51** because feature set Q has no calendar features.

### score.py status

Exit code **0** — all format checks passed, December chart generated.

## Limitations

- No local ground truth for Nov–Dec holdout
- Official Spotter metric unavailable locally
- Long-tail / high-rate underprediction in development validation
- December chart is flat ($815.51) because feature set Q excludes calendar features (all other scenario inputs are fixed)

## Report

Written assessment: `reports/freight_rate_ml_assessment.pdf`

To regenerate the report from existing EDA plots and submission outputs:

```powershell
python scripts/generate_report.py
```
