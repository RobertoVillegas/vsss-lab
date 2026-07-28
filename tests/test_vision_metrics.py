import json
from pathlib import Path

import pytest
from vsss_vision.metrics import analyze_replay


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def test_reports_estimation_prediction_and_interception_error(tmp_path: Path) -> None:
    replay = tmp_path / "capture.jsonl"
    analysis = tmp_path / "capture.analysis.jsonl"
    header: dict[str, object] = {
        "type": "header",
        "config": {"control_period": 0.1},
    }
    perception: dict[str, object] = {
        "ball_estimate": {
            "state": [0.1, 0, 0, 0.0, 0, 0],
            "effective_time": 0.0,
            "update_time": 0.02,
            "measurement_accepted": True,
        },
        "goalkeeper_interception": {"elapsed": 0.1, "x": 0.2, "y": 0.0},
    }
    _write_jsonl(
        replay,
        [
            header,
            {
                "type": "tick",
                "index": 1,
                "snapshot": {"ball": {"x": 0.0, "y": 0.0}},
                "perception": perception,
            },
            {
                "type": "tick",
                "index": 2,
                "snapshot": {"ball": {"x": 0.3, "y": 0.0}},
                "perception": {"ball_estimate": None, "goalkeeper_interception": None},
            },
        ],
    )
    _write_jsonl(
        analysis,
        [
            {"type": "analysis_header"},
            {"type": "prediction_error", "error": 0.2},
        ],
    )

    report = analyze_replay(replay)

    assert report.ticks == 2
    assert report.estimate_coverage == 0.5
    assert report.accepted_measurement_rate == 1.0
    assert report.estimate_age_p95_s == pytest.approx(0.02)
    assert report.ball_estimation.rmse_m == pytest.approx(0.1)
    assert report.trajectory_prediction.rmse_m == pytest.approx(0.2)
    assert report.goalkeeper_interception.rmse_m == pytest.approx(0.1)


def test_rejects_incomplete_replay(tmp_path: Path) -> None:
    replay = tmp_path / "empty.jsonl"
    _write_jsonl(replay, [{"type": "header", "config": {"control_period": 0.1}}])

    with pytest.raises(ValueError, match="no ticks"):
        analyze_replay(replay)
