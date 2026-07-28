"""TensorBoard telemetry derived from the canonical league iteration result."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from vsss_train.config import MarlConfig

from vsss_league.training import IterationResult


@dataclass
class TrainingTelemetry:
    """Write optional visual telemetry without replacing canonical JSONL metrics."""

    writer: SummaryWriter

    @classmethod
    def create(
        cls,
        run_dir: Path,
        config: MarlConfig,
        *,
        start_iteration: int,
    ) -> TrainingTelemetry:
        writer = SummaryWriter(
            log_dir=str(run_dir / "tensorboard"),
            purge_step=start_iteration,
            max_queue=20,
            flush_secs=10,
        )
        writer.add_text(
            "run/config",
            f"```json\n{json.dumps(asdict(config), indent=2, sort_keys=True)}\n```",
            global_step=start_iteration,
        )
        return cls(writer)

    def log_iteration(
        self,
        result: IterationResult,
        *,
        environment_steps: int,
        matches: int,
        frames_per_second: float,
        matches_per_second: float,
        iterations_per_second: float,
        actor_log_std: tuple[float, ...],
    ) -> None:
        step = result.iteration
        scalars = {
            "training/return": result.return_total,
            "training/progress": result.progress,
            "training/environment_steps": environment_steps,
            "training/matches": matches,
            "performance/frames_per_second": frames_per_second,
            "performance/matches_per_second": matches_per_second,
            "performance/iterations_per_second": iterations_per_second,
            **{f"loss/{name}": value for name, value in result.losses.items()},
            **{f"termination/{name}": value for name, value in result.terminations.items()},
            **{f"exploration/log_std_{index}": value for index, value in enumerate(actor_log_std)},
        }
        for name, value in scalars.items():
            self.writer.add_scalar(name, value, step)
        if result.checkpoint is not None:
            self.writer.flush()

    def close(self) -> None:
        self.writer.close()
