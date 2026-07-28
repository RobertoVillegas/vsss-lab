//! Reproducible single-world M2 throughput benchmark.

use std::time::Instant;

use vsss_physics_api::PhysicsBackend;
use vsss_physics_rapier::RapierBackend;
use vsss_spec::{AngularVelocity, MatchConfig, MatchState, RobotAction, serialization};

fn main() {
    let config: MatchConfig =
        serialization::from_json(include_str!("../../../tests/golden/m1_match_config.json"))
            .unwrap();
    let state: MatchState =
        serialization::from_json(include_str!("../../../tests/golden/m1_match_state.json"))
            .unwrap();
    let mut backend = RapierBackend::new(config, state).unwrap();
    let actions = [RobotAction::wheel_velocity(AngularVelocity(10.0), AngularVelocity(10.0)); 6];
    let ticks = 20_000_u32;
    let start = Instant::now();
    for _ in 0..ticks {
        backend.step(&actions).unwrap();
    }
    let elapsed = start.elapsed();
    println!(
        "{{\"ticks\":{ticks},\"seconds\":{},\"ticks_per_second\":{}}}",
        elapsed.as_secs_f64(),
        f64::from(ticks) / elapsed.as_secs_f64()
    );
}
