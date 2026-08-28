"""Train locked final model and generate validation + December predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.final_model import run_final_training_pipeline


if __name__ == "__main__":
    result = run_final_training_pipeline()

    print("=== Locked Model ===")
    print(json.dumps(result["model_spec"], indent=2, default=str))

    print("\n=== Training ===")
    print(json.dumps(result["train_meta"], indent=2))

    print("\n=== Holdout Sanity ===")
    print(json.dumps(result["holdout_sanity"], indent=2))

    print("\n=== Prediction Summary ===")
    print(json.dumps(result["prediction_summary"], indent=2))

    print("\n=== validation_predictions.csv checks ===")
    for check in result["submission_checks"]:
        status = check["result"]
        detail = f" ({check['detail']})" if check.get("detail") else ""
        print(f"{status}: {check['check']}{detail}")

    print("\n=== december-chart-inputs.csv checks ===")
    for check in result["december_checks"]:
        status = check["result"]
        detail = f" ({check['detail']})" if check.get("detail") else ""
        print(f"{status}: {check['check']}{detail}")

    print(f"\nvalidation_predictions.csv -> {result['validation_predictions_path']}")
    print(f"december-chart-inputs.csv -> {result['december_predictions_path']}")
    print(f"diagnostics -> {result['diagnostics_path']}")
    print(f"December prediction range: {result['december_prediction_range']}")

    failed = [
        *[
            c
            for c in result["submission_checks"] + result["december_checks"]
            if c["result"] == "FAIL"
        ]
    ]
    if failed:
        raise SystemExit(f"{len(failed)} validation check(s) failed")
