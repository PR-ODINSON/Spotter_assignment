# GitHub Submission Checklist

Use this before pushing the repository and submitting to Spotter.

## Repository

- [ ] Suggested name: `spotter-freight-rate-ml` or `freight-rate-prediction`
- [ ] Initialize git repo (if not already)
- [ ] Verify `.gitignore` excludes `venv/`, `__pycache__/`, `.env`
- [ ] Do **not** commit `venv/`

## Files to commit

### Required deliverables

- [ ] `validation_predictions.csv` (12,000 rows)
- [ ] `reports/freight_rate_ml_assessment.docx` (or `.pdf`)
- [ ] `scorer_results/candidate_december.png`
- [ ] `december-chart-inputs.csv` (31 rows with predictions)

### Source code

- [ ] `src/` (config, data, features, baselines, metrics, experiments, final_model)
- [ ] `scripts/` (train_final.py, phase runners, generate_report.py)
- [ ] `score.py` (unmodified)
- [ ] `requirements.txt`
- [ ] `README.md`

### Documentation

- [ ] `docs/PHASE_0_PROJECT_AUDIT.md` through `docs/PHASE_5_SCORING.md`
- [ ] `docs/LOOM_SCRIPT.md`
- [ ] `docs/GITHUB_CHECKLIST.md` (this file)

### Supporting artifacts (recommended)

- [ ] `artifacts/` (experiment CSVs)
- [ ] `reports/eda/` (EDA plots)

## Files NOT to commit

- [ ] `venv/` or `.venv/`
- [ ] `__pycache__/`, `*.pyc`
- [ ] `.env`, credentials, API keys
- [ ] IDE-specific cache (`.idea/`, `.vscode/` unless team standard)
- [ ] `.cursor/` (optional — local rules)

## Pre-push verification

- [ ] `README.md` commands tested on clean venv
- [ ] `requirements.txt` matches final pipeline (no unused CatBoost/XGB/LGBM)
- [ ] `validation_predictions.csv`: 12,000 rows, `load_id,predicted_rate`, all positive
- [ ] Report present in `reports/`
- [ ] `score.py` exit code 0
- [ ] No secrets in source, README, or docs
- [ ] Development MAE (106.83 / 113.29) labeled as **development validation**, not holdout score

## Manual steps after push

- [ ] Record Loom video using `docs/LOOM_SCRIPT.md`
- [ ] Submit to Spotter: GitHub link, `validation_predictions.csv`, report PDF/DOCX, Loom link
- [ ] Do not invent an official Spotter accuracy score

## Quick verification commands

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/train_final.py
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

Expected: all checks PASS, chart at `scorer_results/candidate_december.png`.
