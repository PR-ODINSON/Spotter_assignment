"""Run Phase 2 baseline comparison (superseded by run_phase2_validation.py)."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments import run_phase2

if __name__ == "__main__":
    run_phase2()
