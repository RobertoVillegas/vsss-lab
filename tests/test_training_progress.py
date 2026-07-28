from rich.console import Console
from vsss_league.progress import TrainingDashboard
from vsss_league.training import IterationResult


def test_step_target_shows_exact_count_and_precise_percentage() -> None:
    console = Console(record=True, width=120)
    dashboard = TrainingDashboard(
        start_iteration=1,
        total_iterations=1_221,
        device="cuda",
        num_envs=64,
        target_steps=20_000_000,
        console=console,
    )
    result = IterationResult(
        iteration=3,
        policy_version=3,
        opponent="blue-shared@2",
        seed=10,
        frames=16_384,
        matches=4,
        return_total=0.0,
        progress=0.0,
        checkpoint=None,
        losses={"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0},
    )

    dashboard.update(
        result,
        completed=3,
        iteration_rate=0.32,
        frame_rate=5_278.0,
        environment_steps=49_152,
        matches=11,
        match_rate=1.18,
        checkpoint=False,
    )
    console.print(dashboard._table())
    rendered = console.export_text()

    assert "49,152 / 20,000,000" in rendered
    assert "0.246%" in rendered
