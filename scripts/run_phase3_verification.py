"""Run Phase 3 final candidate verification (pre-Phase 4)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase3_verification import run_verification


if __name__ == "__main__":
    meta = run_verification()
    print("\n=== Experiment 1: log1p + Q vs baselines ===")
    cols = [
        "config_name", "feature_set", "target_transform",
        "primary_mae", "sensitivity_mae", "stability_gap",
        "primary_r2", "sensitivity_r2",
        "long_haul_mae", "top_1_percent_mae", "rate_over_5000_mae",
    ]
    print(meta["experiment1"][cols].to_string(index=False))

    if not meta["experiment2"].empty:
        print("\n=== Experiment 2: log1p + Q depth tuning ===")
        print(meta["experiment2"][["config_name", "primary_mae", "sensitivity_mae", "stability_gap"]].to_string(index=False))

    print("\n=== Final candidate ranking ===")
    print(meta["final_candidates"][["rank", "config_name", "primary_mae", "sensitivity_mae", "stability_gap", "combined_score"]].head(5).to_string(index=False))
    print(f"\nResidual plot: {meta['residual_plot']}")
    print(f"Final candidates CSV: artifacts/phase3_final_candidates.csv")
