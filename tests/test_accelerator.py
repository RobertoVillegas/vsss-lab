from __future__ import annotations

import numpy as np
import torch
from vsss_league.training import _host_actions
from vsss_train.accelerator import compare_traces, decide_accelerator


def test_fast_candidate_is_rejected_when_contacts_or_goals_diverge() -> None:
    authoritative = np.zeros((4, 7, 2), dtype=np.float32)
    candidate = authoritative.copy()
    candidate[2, 1, 0] = 0.02
    parity = compare_traces(
        authoritative,
        candidate,
        np.array([0, 0, 1, 0]),
        np.array([0, 0, 0, 0]),
    )
    decision = decide_accelerator(
        candidate_backend="torch-cuda-prototype",
        authoritative_fps=10_000,
        candidate_fps=100_000,
        parity=parity,
    )
    assert not decision.adopted
    assert decision.reason == "trace_parity_failed"


def test_parity_candidate_still_requires_material_end_to_end_speedup() -> None:
    trace = np.zeros((4, 7, 2), dtype=np.float32)
    events = np.zeros(4, dtype=np.int64)
    parity = compare_traces(trace, trace.copy(), events, events.copy())
    assert parity.passed
    assert not decide_accelerator(
        candidate_backend="prototype",
        authoritative_fps=10_000,
        candidate_fps=12_000,
        parity=parity,
    ).adopted
    assert decide_accelerator(
        candidate_backend="prototype",
        authoritative_fps=10_000,
        candidate_fps=20_000,
        parity=parity,
    ).adopted


def test_host_action_bridge_reuses_storage() -> None:
    actions = torch.ones((2, 3, 2), dtype=torch.float32)
    first, buffer = _host_actions(actions, None)
    second, reused = _host_actions(actions * 0.5, buffer)
    assert reused.data_ptr() == buffer.data_ptr()
    assert first.__array_interface__["data"][0] == second.__array_interface__["data"][0]
    assert np.all(second == 0.5)
