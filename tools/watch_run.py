"""Block until a training run reaches a checkpoint, then print everything needed to judge it.

Watching a run by polling it costs a round trip every time. This waits in one process and
reports once, with the quantities a decision actually turns on: whether the policy is still
going for the ball, whether it scores, and which gate is holding it back. It also wakes early
when the run dies or when a stop condition fires, so a run that has gone wrong is not left to
burn hours.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

#: How long a policy may sit below the strike floor before the watcher calls it a trend.
TREND_EVALUATIONS = 3


def rows(run: Path) -> list[dict[str, Any]]:
    path = run / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluations(run: Path) -> list[dict[str, Any]]:
    path = run / "semantic-evaluations.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def alive() -> bool:
    found = subprocess.run(
        ["pgrep", "-f", "vsss_league.cli run"],
        capture_output=True,
        check=False,
    )
    return found.returncode == 0


def stats(row: dict[str, Any]) -> dict[str, float]:
    policy = row.get("policy_stats") or {}
    return {
        name: float(policy.get(name, 0.0))
        for name in ("strike_fraction", "navigate_fraction", "stop_fraction", "mean_intensity")
    }


def unresolved_share(row: dict[str, Any]) -> float:
    outcomes = ((row.get("curriculum") or {}).get("outcomes")) or {}
    total = sum(outcomes.values())
    return outcomes.get("unresolved", 0) / total if total else 0.0


def collapsing(run: Path, strike_floor: float) -> str | None:
    """A stop condition worth waking early for: not shooting and not scoring."""
    history = rows(run)
    if len(history) < 200:
        return None
    recent = history[-100:]
    strikes = sum(stats(row)["strike_fraction"] for row in recent) / len(recent)
    if strikes > strike_floor:
        return None
    reports = evaluations(run)[-TREND_EVALUATIONS:]
    if len(reports) < TREND_EVALUATIONS:
        return None
    goals = [
        float((report.get("behavior_gate") or {}).get("goals_for_per_minute", 0.0))
        for report in reports
    ]
    if max(goals) > 0.0:
        return None
    return (
        f"strikes averaged {strikes:.3f} over the last hundred iterations, below the "
        f"{strike_floor:.2f} floor, and the last {TREND_EVALUATIONS} evaluations scored nothing"
    )


def report(run: Path, reason: str) -> str:
    history = rows(run)
    reports = evaluations(run)
    lines = [f"== {run.name}: {reason} ==", ""]
    if not history:
        return "\n".join([*lines, "no metrics written"])

    pace = [
        1.0 / row["performance"]["iterations_per_second"]
        for row in history
        if (row.get("performance") or {}).get("iterations_per_second")
    ]
    seconds = sorted(pace)[len(pace) // 2] if pace else 0.0
    curriculum = history[-1].get("curriculum") or {}
    lines.append(
        f"iteration {len(history)} of 3052 ({100 * len(history) / 3052:.0f}%), "
        f"{seconds:.2f} s/iter, {(3052 - len(history)) * seconds / 3600:.1f} h remaining, "
        f"phase {curriculum.get('phase')}"
    )
    lines.append("")
    lines.append(
        f"{'iter':>6}{'strike':>9}{'navigate':>10}{'stop':>8}{'intensity':>11}{'unres':>8}"
    )
    step = max(1, len(history) // 10)
    for row in [*history[::step], history[-1]]:
        values = stats(row)
        lines.append(
            f"{row['iteration']:>6}{values['strike_fraction']:>9.3f}"
            f"{values['navigate_fraction']:>10.3f}{values['stop_fraction']:>8.3f}"
            f"{values['mean_intensity']:>11.3f}{unresolved_share(row):>8.2f}"
        )

    if reports:
        lines.append("")
        lines.append(
            f"{'iter':>6}{'approach':>10}{'shot':>7}{'intercept':>11}"
            f"{'rot_rec':>9}{'goals/min':>11}  gate"
        )
        for report_row in reports[-10:]:
            families = report_row.get("families") or {}
            rate = {
                name: float(families.get(name, {}).get("success_rate", 0.0))
                for name in ("approach", "shot", "interception", "rotation_recovery")
            }
            gate = report_row.get("behavior_gate") or {}
            lines.append(
                f"{report_row['iteration']:>6}{rate['approach']:>10.2f}{rate['shot']:>7.2f}"
                f"{rate['interception']:>11.2f}{rate['rotation_recovery']:>9.2f}"
                f"{float(gate.get('goals_for_per_minute', 0.0)):>11.1f}  {gate.get('failures')}"
            )
    lines.append("")
    lines.append(f"process alive: {alive()}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True, help="wake at this iteration")
    parser.add_argument(
        "--strike-floor",
        type=float,
        default=0.15,
        help="wake early if strikes fall below this while nothing is scored",
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    arguments = parser.parse_args()

    while True:
        history = rows(arguments.run_dir)
        if len(history) >= arguments.iteration:
            print(report(arguments.run_dir, f"reached iteration {arguments.iteration}"))
            return
        stopped = collapsing(arguments.run_dir, arguments.strike_floor)
        if stopped is not None:
            print(report(arguments.run_dir, f"STOP CONDITION — {stopped}"))
            return
        if history and not alive():
            print(report(arguments.run_dir, "the run exited before the checkpoint"))
            return
        time.sleep(arguments.poll_seconds)


if __name__ == "__main__":
    main()
