import signal
from unittest.mock import patch

import pytest
from vsss_league.cli import TrainingInterrupt


def test_first_sigint_requests_graceful_stop() -> None:
    interrupt = TrainingInterrupt()

    with patch("vsss_league.cli.time.monotonic", return_value=10.0):
        assert interrupt.handle(signal.SIGINT, None)

    assert interrupt.stop_requested


def test_second_quick_sigint_forces_stop() -> None:
    interrupt = TrainingInterrupt()

    with patch("vsss_league.cli.time.monotonic", side_effect=(10.0, 11.0)):
        interrupt.handle(signal.SIGINT, None)
        with pytest.raises(KeyboardInterrupt):
            interrupt.handle(signal.SIGINT, None)


def test_sigterm_remains_graceful_and_repeated_sigint_restarts_window() -> None:
    interrupt = TrainingInterrupt()

    with patch("vsss_league.cli.time.monotonic", side_effect=(10.0, 13.0)):
        assert interrupt.handle(signal.SIGTERM, None)
        assert not interrupt.handle(signal.SIGINT, None)

    assert interrupt.stop_requested
