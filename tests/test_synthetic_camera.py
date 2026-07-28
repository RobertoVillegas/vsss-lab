import copy
import json
from pathlib import Path

import pytest
from vsss_vision.camera import CameraPerturbationProfile, SyntheticCamera

FIXTURE = Path("tests/golden/m1_match_state.json")


def test_seeded_camera_is_reproducible_and_preserves_truth() -> None:
    truth = json.loads(FIXTURE.read_text())
    original = copy.deepcopy(truth)
    profile = CameraPerturbationProfile(
        position_noise_std=0.01,
        heading_noise_std=0.02,
        occlusion_probability=0.1,
        false_detection_probability=0.1,
        misassociation_probability=0.2,
    )

    first = SyntheticCamera(profile, seed=42).observe(truth)
    second = SyntheticCamera(profile, seed=42).observe(truth)

    assert first == second
    assert truth == original
    assert first.arrival_time - first.capture_time == pytest.approx(profile.latency_seconds)


def test_camera_occlusion_does_not_fabricate_visibility() -> None:
    truth = json.loads(FIXTURE.read_text())
    frame = SyntheticCamera(CameraPerturbationProfile(occlusion_probability=1.0), seed=7).observe(
        truth
    )

    assert frame.ball is None
    assert frame.robots == ()


def test_misassociation_is_explicit_and_low_confidence() -> None:
    truth = json.loads(FIXTURE.read_text())
    frame = SyntheticCamera(
        CameraPerturbationProfile(misassociation_probability=1.0), seed=7
    ).observe(truth)

    assert frame.robots
    assert all(robot.association.ambiguous for robot in frame.robots)
    assert all(robot.association.confidence == 0.35 for robot in frame.robots)
