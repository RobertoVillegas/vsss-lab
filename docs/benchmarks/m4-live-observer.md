# M4.1 live observer benchmark

Date: 2026-07-27  
Host: linux/amd64 devbox  
Command: `just benchmark-observer 10000 7 4`

## Result

```json
{
  "observer_overhead_percent": 0.09,
  "repeats": 7,
  "sample_every": 4,
  "ticks": 10000,
  "unwatched_median_seconds": 2.635736,
  "watched_median_seconds": 2.638105
}
```

The benchmark alternates watched and unwatched runs to reduce ordering and
thermal bias. Both paths record the same lossless JSONL replay. The watched path
adapts every fourth completed simulation tick into a `VisualFrame` and publishes
it to a bounded latest-frame sink.

At this workload the measured 0.09% difference is operationally
indistinguishable from local timing noise. This is not a physics throughput
benchmark: Python controller and JSONL costs dominate. Re-run after the Rust
match runner and live transport exist.

## Compatibility and rollback

Sampling affects only lossy live observation. Canonical state, actions, replay
records, and checksums remain unchanged. Setting no observers avoids frame
construction; removing the observer hook restores the prior runner without a
data migration.
