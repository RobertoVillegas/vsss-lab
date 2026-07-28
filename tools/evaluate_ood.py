"""Run and persist the M11 paired OOD robustness suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vsss_env.randomization import evaluate_ood

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reports/m11/ood.json")
    args = parser.parse_args()
    report = evaluate_ood(
        ROOT / "tests/golden/m1_match_config.json",
        ROOT / "tests/golden/m1_match_state.json",
        ROOT / "experiments/configs/m11-ood.json",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = {
        key: report[key] for key in ("nominal_progress", "robust_progress", "margin", "passed")
    }
    print(json.dumps(summary, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
