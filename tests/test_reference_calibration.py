from vsss_env.calibration import run_calibration


def test_reference_calibration_is_within_committed_tolerances() -> None:
    report = run_calibration()
    assert report["passed"], report
    assert len(report["results"]) == 3
