from __future__ import annotations

import json
from pathlib import Path

import optuna
import pytest
from vsss_train.search import (
    FidelityResult,
    SearchParameters,
    TrialLineage,
    create_study,
    record_lineage,
    run_multifidelity_trial,
    seed_set,
    suggest_parameters,
)


def test_study_resumes_without_repeating_completed_trials(tmp_path: Path) -> None:
    database = tmp_path / "m14.db"
    first = create_study(name="m14-test", storage_path=database)

    def objective(trial: optuna.trial.Trial) -> tuple[float, float, float]:
        parameters = suggest_parameters(trial)
        trial.set_user_attr("config_sha256", parameters.digest)
        return (0.6, 0.1, 0.01)

    first.optimize(objective, n_trials=1)
    second = create_study(name="m14-test", storage_path=database)
    assert len(second.trials) == 1
    second.optimize(objective, n_trials=1)
    assert [trial.number for trial in second.trials] == [0, 1]
    assert second.trials[0].user_attrs["config_sha256"]


def test_search_parameters_are_bounded_and_seed_sets_scale_by_fidelity() -> None:
    with pytest.raises(ValueError, match="learning_rate"):
        SearchParameters(1.0, 0.01, 0.2, 10.0, 1.0, 0.1, 0.1).validate()
    assert len(seed_set(trial_number=2, fidelity="smoke")) == 1
    assert len(seed_set(trial_number=2, fidelity="screen")) == 3
    assert len(seed_set(trial_number=2, fidelity="confirm")) == 5
    assert seed_set(trial_number=2, fidelity="confirm") == seed_set(
        trial_number=2, fidelity="confirm"
    )


def test_lineage_is_idempotent_and_rejects_conflicting_resume(tmp_path: Path) -> None:
    path = tmp_path / "lineage.jsonl"
    lineage = TrialLineage(
        study="m14",
        trial=3,
        fidelity="screen",
        seeds=(140_300, 140_301, 140_302),
        code_commit="abc",
        config_sha256="def",
        parent_trial=1,
        compute_seconds=12.5,
        status="complete",
        pruning_reason=None,
        objectives=(0.6, 0.1, 0.01),
    )
    record_lineage(path, lineage)
    record_lineage(path, lineage)
    assert len(path.read_text().splitlines()) == 1
    assert json.loads(path.read_text())["seeds"] == [140_300, 140_301, 140_302]
    conflicting = TrialLineage(**{**lineage.__dict__, "compute_seconds": 13.0})
    with pytest.raises(ValueError, match="conflicting"):
        record_lineage(path, conflicting)


def test_multifidelity_trial_records_exact_seeds_and_prunes_before_confirm(
    tmp_path: Path,
) -> None:
    study = create_study(name="m14-fidelity", storage_path=tmp_path / "study.db")
    calls: list[str] = []

    def evaluator(
        parameters: SearchParameters,
        fidelity: str,
        seeds: tuple[int, ...],
    ) -> FidelityResult:
        calls.append(fidelity)
        score = 0.5 if fidelity == "smoke" else 0.2
        return FidelityResult(score, 0.1, 1.0)

    trial = run_multifidelity_trial(
        study,
        evaluator,
        lineage_path=tmp_path / "lineage.jsonl",
    )
    assert trial.state == optuna.trial.TrialState.PRUNED
    assert calls == ["smoke", "screen"]
    records = [json.loads(line) for line in (tmp_path / "lineage.jsonl").read_text().splitlines()]
    assert [record["fidelity"] for record in records] == ["smoke", "screen"]
    assert len(records[1]["seeds"]) == 3
