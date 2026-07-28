"""Rich terminal dashboard with a deterministic plain-text fallback."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
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
    console: Console = field(default_factory=Console)

    def __post_init__(self) -> None:
        self._returns: deque[float] = deque(maxlen=20)
        self._progress_values: deque[float] = deque(maxlen=20)
        self._status = "starting"
        self._latest: IterationResult | None = None
        self._frame_rate = 0.0
        self._iteration_rate = 0.0
        self._checkpoint = "—"
        self._progress = Progress(
            TextColumn("[bold green]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            expand=True,
        )
        self._task = self._progress.add_task("training", total=self.total_iterations)
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
        checkpoint: bool,
    ) -> None:
        self._latest = result
        self._returns.append(result.return_total)
        self._progress_values.append(result.progress)
        self._iteration_rate = iteration_rate
        self._frame_rate = frame_rate
        self._status = "checkpointed" if checkpoint else "running"
        if checkpoint and result.checkpoint is not None:
            self._checkpoint = result.checkpoint.rsplit("/", maxsplit=1)[-1]
        self._progress.update(self._task, completed=completed)
        if self.interactive:
            self.refresh()
        else:
            self.console.print(
                f"{result.iteration:>6}/{self.start_iteration + self.total_iterations - 1:<6} "
                f"return {result.return_total:+8.3f}  progress {result.progress:+8.3f}  "
                f"{iteration_rate:5.2f} iter/s  {frame_rate:9,.0f} frames/s"
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
            "return",
            f"{latest.return_total:+.4f}" if latest is not None else "—",
            f"{_mean(self._returns):+.4f}" if self._returns else "—",
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
        table.add_row("latest checkpoint", self._checkpoint, "")
        return table


def _mean(values: deque[float]) -> float:
    return sum(values) / len(values)
