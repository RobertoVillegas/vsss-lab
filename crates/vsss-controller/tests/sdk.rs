//! Rust controller SDK contract tests.

use vsss_controller::{Controller, StopController};
use vsss_spec::{MatchConfig, MatchState, serialization};

#[test]
fn sample_controller_returns_three_finite_commands() {
    let config: MatchConfig =
        serialization::from_json(include_str!("../../../tests/golden/m1_match_config.json"))
            .expect("config");
    let state: MatchState =
        serialization::from_json(include_str!("../../../tests/golden/m1_match_state.json"))
            .expect("state");
    let mut controller = StopController;

    controller.on_reset(&config, &state).expect("reset");
    let actions = controller.act(&state).expect("act");
    assert_eq!(actions.len(), 3);
    assert!(
        actions
            .iter()
            .all(|action| action.first.is_finite() && action.second.is_finite())
    );
}
