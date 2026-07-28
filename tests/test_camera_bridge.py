import json
from pathlib import Path

import pytest
from vsss_vision import CameraEstimatorBridge, camera_frame_from_json, camera_frame_from_mapping

FIXTURE = Path("tests/fixtures/m12_overhead_detections.jsonl")


def test_recorded_detection_fixture_replays_through_cpu_estimator() -> None:
    bridge = CameraEstimatorBridge()
    outputs = [
        bridge.update(camera_frame_from_json(line)) for line in FIXTURE.read_text().splitlines()
    ]

    assert len(outputs) == 3
    assert outputs[0].ball is not None
    assert outputs[0].ball.measurement_accepted
    assert len(outputs[0].robots) == 6
    assert outputs[-1].ball is not None
    assert outputs[-1].ball.rejection_reason == "measurement_missing"
    assert outputs[-1].ball.source_sequence == 1


def test_detection_contract_rejects_duplicate_markers() -> None:
    record = json.loads(FIXTURE.read_text().splitlines()[0])
    record["robots"].append(record["robots"][0])

    with pytest.raises(ValueError, match="unique"):
        camera_frame_from_mapping(record)


def test_detection_contract_rejects_time_travel_and_reordering() -> None:
    records = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
    records[0]["arrival_time"] = -1.0
    with pytest.raises(ValueError, match="must not precede"):
        camera_frame_from_mapping(records[0])

    bridge = CameraEstimatorBridge()
    frame = camera_frame_from_mapping(records[1])
    bridge.update(frame)
    with pytest.raises(ValueError, match="increasing"):
        bridge.update(frame)
