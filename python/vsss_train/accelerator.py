"""Evidence gate for alternate device-resident simulator prototypes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class TraceParity:
    samples: int
    maximum_position_error: float
    mean_position_error: float
    event_mismatches: int
    passed: bool


@dataclass(frozen=True)
class AcceleratorDecision:
    authoritative_backend: str
    candidate_backend: str
    authoritative_fps: float
    candidate_fps: float
    speedup: float
    parity: TraceParity
    adopted: bool
    reason: str


def compare_traces(
    authoritative_positions: FloatArray,
    candidate_positions: FloatArray,
    authoritative_events: IntArray,
    candidate_events: IntArray,
    *,
    position_tolerance: float = 1e-4,
) -> TraceParity:
    if authoritative_positions.shape != candidate_positions.shape:
        raise ValueError("position trace shapes differ")
    if authoritative_events.shape != candidate_events.shape:
        raise ValueError("event trace shapes differ")
    if authoritative_positions.ndim < 2 or authoritative_positions.shape[-1] != 2:
        raise ValueError("positions must end in planar x/y coordinates")
    errors = np.linalg.norm(
        authoritative_positions.astype(np.float64) - candidate_positions.astype(np.float64),
        axis=-1,
    )
    mismatches = int(np.count_nonzero(authoritative_events != candidate_events))
    maximum = float(errors.max(initial=0.0))
    return TraceParity(
        samples=int(errors.size),
        maximum_position_error=maximum,
        mean_position_error=float(errors.mean()) if errors.size else 0.0,
        event_mismatches=mismatches,
        passed=maximum <= position_tolerance and mismatches == 0,
    )


def decide_accelerator(
    *,
    candidate_backend: str,
    authoritative_fps: float,
    candidate_fps: float,
    parity: TraceParity,
    required_speedup: float = 1.5,
) -> AcceleratorDecision:
    if authoritative_fps <= 0.0 or candidate_fps <= 0.0:
        raise ValueError("throughput measurements must be positive")
    speedup = candidate_fps / authoritative_fps
    adopted = parity.passed and speedup >= required_speedup
    reason = (
        "adopted"
        if adopted
        else "trace_parity_failed"
        if not parity.passed
        else "end_to_end_speedup_below_gate"
    )
    return AcceleratorDecision(
        authoritative_backend="rust-rapier",
        candidate_backend=candidate_backend,
        authoritative_fps=authoritative_fps,
        candidate_fps=candidate_fps,
        speedup=speedup,
        parity=parity,
        adopted=adopted,
        reason=reason,
    )
