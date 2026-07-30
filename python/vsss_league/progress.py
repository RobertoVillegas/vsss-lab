"""Rich terminal dashboard with a deterministic plain-text fallback."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from vsss_league.training import IterationResult


@dataclass
class TrainingDashboard:
    """Keep current training metrics above a fixed progress bar."""

    start_iteration: int
    total_iterations: int
    device: str
    num_envs: int
    target_matches: int | None = None
    target_steps: int | None = None
    console: Console = field(default_factory=Console)

    def __post_init__(self) -> None:
        self._returns: deque[float] = deque(maxlen=20)
        self._progress_values: deque[float] = deque(maxlen=20)
        self._status = "starting"
        self._latest: IterationResult | None = None
        self._frame_rate = 0.0
        self._iteration_rate = 0.0
        self._checkpoint = "—"
        self._environment_steps = 0
        self._matches = 0
        self._match_rate = 0.0
        self._progress = Progress(
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TextColumn(
                "[progress.percentage]{task.completed:,.0f}/{task.total:,.0f}"
                " · {task.percentage:>6.2f}%"
            ),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
        )
        description = (
            "steps"
            if self.target_steps is not None
            else "matches"
            if self.target_matches is not None
            else "training"
        )
        total = self.target_steps or self.target_matches or self.total_iterations
        self._task = self._progress.add_task(description, total=total)
        self._live = Live(
            Group(self._table(), self._progress),
            console=self.console,
            auto_refresh=False,
            transient=False,
        )

    @property
    def interactive(self) -> bool:
        return self.console.is_terminal

    def start(self) -> None:
        if self.interactive:
            self._live.start(refresh=True)
        else:
            self.console.print(
                f"training device: {self.device} · vector environments: {self.num_envs}"
            )

    def request_stop(self) -> None:
        self._status = "stopping after current iteration"
        self.log("Stop requested; finishing the current iteration and saving a checkpoint…")
        self.refresh()

    def update(
        self,
        result: IterationResult,
        *,
        completed: int,
        iteration_rate: float,
        frame_rate: float,
        environment_steps: int,
        matches: int,
        match_rate: float,
        checkpoint: bool,
    ) -> None:
        self._latest = result
        self._returns.append(result.return_total)
        self._progress_values.append(result.progress)
        self._iteration_rate = iteration_rate
        self._frame_rate = frame_rate
        self._environment_steps = environment_steps
        self._matches = matches
        self._match_rate = match_rate
        self._status = "checkpointed" if checkpoint else "running"
        if checkpoint and result.checkpoint is not None:
            self._checkpoint = result.checkpoint.rsplit("/", maxsplit=1)[-1]
        progress_completed = (
            min(environment_steps, self.target_steps)
            if self.target_steps is not None
            else min(matches, self.target_matches)
            if self.target_matches is not None
            else completed
        )
        self._progress.update(self._task, completed=progress_completed)
        if self.interactive:
            self.refresh()
        else:
            self.console.print(
                f"{result.iteration:>6}/{self.start_iteration + self.total_iterations - 1:<6} "
                f"return {result.return_total:+8.3f}  progress {result.progress:+8.3f}  "
                f"{iteration_rate:5.2f} iter/s  {frame_rate:9,.0f} frames/s"
                f"  {match_rate:6.2f} matches/s"
                f"{'  checkpoint' if checkpoint else ''}"
            )

    def log(self, message: str) -> None:
        if self.interactive:
            self._live.console.print(message)
        else:
            self.console.print(message)

    def refresh(self) -> None:
        if self.interactive:
            self._live.update(Group(self._table(), self._progress), refresh=True)

    def stop(self) -> None:
        if self.interactive:
            self._live.stop()

    def _table(self) -> Table:
        table = Table(title="VSSS MAPPO training", expand=True)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Current", justify="right")
        table.add_column("Rolling 20", justify="right")
        latest = self._latest
        table.add_row("status", self._status, "")
        table.add_row("device / worlds", f"{self.device} / {self.num_envs}", "")
        table.add_row(
            "iteration",
            str(latest.iteration if latest is not None else self.start_iteration - 1),
            f"{self._iteration_rate:.2f} iter/s",
        )
        table.add_row("experience", f"{self._frame_rate:,.0f} frames/s", "")
        table.add_row(
            "environment steps",
            (
                f"{self._environment_steps:,} / {self.target_steps:,}"
                if self.target_steps is not None
                else f"{self._environment_steps:,}"
            ),
            (
                f"{self._environment_steps / self.target_steps:.3%}"
                if self.target_steps is not None
                else ""
            ),
        )
        table.add_row(
            "matches",
            f"{self._matches:,}",
            f"{self._match_rate:.2f} matches/s",
        )
        table.add_row(
            "return",
            f"{latest.return_total:+.4f}" if latest is not None else "—",
            f"{_mean(self._returns):+.4f}" if self._returns else "—",
        )
        if latest is not None:
            table.add_row(
                "completed episode return",
                (
                    f"{latest.completed_episode_return:+.4f}"
                    if latest.completed_episode_return is not None
                    else "—"
                ),
                "",
            )
            match_counts = latest.match_outcomes
            table.add_row(
                "full matches W/D/L",
                " / ".join(str(match_counts.get(name, 0)) for name in ("win", "draw", "loss")),
                "",
            )
        table.add_row(
            "progress",
            f"{latest.progress:+.4f}" if latest is not None else "—",
            f"{_mean(self._progress_values):+.4f}" if self._progress_values else "—",
        )
        for name in ("policy_loss", "value_loss", "entropy"):
            table.add_row(
                name,
                f"{latest.losses[name]:+.5f}" if latest is not None else "—",
                "",
            )
        for name in ("approx_kl", "clip_fraction", "mean_abs_action", "action_saturation"):
            table.add_row(
                name,
                (
                    f"{latest.losses[name]:+.5f}"
                    if latest is not None and name in latest.losses
                    else "—"
                ),
                "",
            )
        if latest is not None and latest.curriculum is not None:
            outcomes = latest.curriculum.get("outcomes")
            if isinstance(outcomes, dict):
                table.add_row(
                    "skill outcomes S/F/U",
                    " / ".join(
                        str(outcomes.get(status, 0))
                        for status in ("success", "failure", "unresolved")
                    ),
                    "",
                )
            rates = latest.curriculum.get("success_rate")
            if isinstance(rates, dict) and rates:
                table.add_row(
                    "skill success",
                    "  ".join(
                        f"{str(family)[:4]}:{float(value):.0%}"
                        for family, value in sorted(rates.items())
                        if isinstance(value, (int, float))
                    ),
                    "",
                )
            rotation = latest.curriculum.get("rotation")
            if isinstance(rotation, dict):
                table.add_row(
                    "rotations done/tries · uncovered",
                    (
                        f"{int(rotation.get('completed', 0))}/"
                        f"{int(rotation.get('attempts', 0))} · "
                        f"{float(rotation.get('uncovered_ratio', 0.0)):.1%}"
                    ),
                    "",
                )
            contact = latest.curriculum.get("contact")
            if isinstance(contact, dict):
                table.add_row(
                    "contact ally/enemy · deadlocks · escapes",
                    (
                        f"{float(contact.get('ally_seconds', 0.0)):.1f}s/"
                        f"{float(contact.get('opponent_seconds', 0.0)):.1f}s · "
                        f"{int(contact.get('ally_deadlocks', 0))}/"
                        f"{int(contact.get('opponent_deadlocks', 0))} · "
                        f"{int(contact.get('escapes', 0))}"
                    ),
                    "",
                )
            motion = latest.curriculum.get("motion")
            if isinstance(motion, dict):
                table.add_row(
                    "idle spin",
                    (
                        f"{float(motion.get('idle_spin_agent_seconds', 0.0)):.1f} agent-s · "
                        f"{float(motion.get('idle_spin_ratio', 0.0)):.1%}"
                    ),
                    "",
                )
            rosters = latest.curriculum.get("allocation_by_roster")
            if isinstance(rosters, dict) and rosters:
                table.add_row(
                    "roster curriculum",
                    "  ".join(
                        f"{roster}:{int(count)}" for roster, count in sorted(rosters.items())
                    ),
                    "",
                )
        table.add_row("latest checkpoint", self._checkpoint, "")
        return table


def _mean(values: deque[float]) -> float:
    return sum(values) / len(values)
