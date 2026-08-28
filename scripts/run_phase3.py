"""Run Phase 3 model optimization experiments."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase3_experiments import run_phase3


if __name__ == "__main__":
    meta = run_phase3()
    print("\n=== Log-target (both splits) ===")
    print(meta["log_target"].to_string(index=False))
    print("\n=== Top feature sets (primary) ===")
    fs = meta["feature_sets"]
    print(fs[fs["validation_split"] == "primary"].sort_values("MAE").to_string(index=False))
    print("\n=== Top candidates ===")
    print(meta["candidates"].head(5).to_string(index=False))
