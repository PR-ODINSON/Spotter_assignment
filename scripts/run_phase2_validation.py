"""Run Phase 2 validation and baseline experiments."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import run_phase2


if __name__ == "__main__":
    comparison, ablation, meta = run_phase2()
    print("\n=== Primary split (sorted by MAE) ===")
    primary = comparison[comparison["validation_split"] == "primary"].sort_values("MAE")
    print(primary[["model", "feature_set", "MAE", "RMSE", "R2"]].to_string(index=False))
    print("\n=== Feature ablation ===")
    print(ablation[["model", "feature_set", "MAE"]].to_string(index=False))
