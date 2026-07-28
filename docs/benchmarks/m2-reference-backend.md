# M2 reference backend baseline

- Date: 2026-07-27
- Commit: working tree on `feat/m2-physics-backend`
- Host: linux/amd64, Rust 1.97.1, Rapier2D 0.34.0
- Mode: release, one world, six equal wheel commands, 20,000 ticks
- Command: `cargo run --release --locked -p vsss-physics-rapier --example benchmark`

```json
{"ticks":20000,"seconds":0.039853998,"ticks_per_second":501831.7108361374}
```

This is a local baseline, not a cross-machine performance promise. Later batch
and optimized modes must record configuration and compare against this workload.
