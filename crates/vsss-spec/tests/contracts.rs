//! Contract, compatibility, and symmetry tests for the canonical M1 model.

use vsss_spec::{
    Angle, AngularVelocity, BallState, Distance, EventFlags, LinearVelocity, MatchConfig,
    MatchState, Pose2, RobotId, RobotState, SCHEMA_VERSION, Seconds, Team, Twist2, Validate,
    canonical_types, serialization,
};

fn robot(id: RobotId, team: Team, x: f32, y: f32) -> RobotState {
    RobotState {
        id,
        team,
        pose: Pose2 {
            x: Distance(x),
            y: Distance(y),
            theta: Angle(0.0),
        },
        twist: Twist2::default(),
        wheel_speed_left: AngularVelocity(0.0),
        wheel_speed_right: AngularVelocity(0.0),
        enabled: true,
    }
}

fn state() -> MatchState {
    MatchState {
        schema_version: SCHEMA_VERSION,
        tick: 42,
        simulation_time: Seconds(0.84),
        score_blue: 2,
        score_yellow: 1,
        ball: BallState {
            x: Distance(0.25),
            y: Distance(-0.5),
            vx: LinearVelocity(1.0),
            vy: LinearVelocity(0.0),
            omega: AngularVelocity(0.2),
        },
        robots: [
            robot(RobotId::R0, Team::Blue, -0.5, 0.0),
            robot(RobotId::R1, Team::Blue, -0.25, 0.25),
            robot(RobotId::R2, Team::Blue, -0.25, -0.25),
            robot(RobotId::R3, Team::Yellow, 0.5, 0.0),
            robot(RobotId::R4, Team::Yellow, 0.25, 0.25),
            robot(RobotId::R5, Team::Yellow, 0.25, -0.25),
        ],
        events: EventFlags::GOAL_BLUE,
    }
}

#[test]
fn valid_state_round_trips_strict_json() {
    let original = state();
    original.validate().unwrap();
    let json = serialization::to_json(&original).unwrap();
    let decoded: MatchState = serialization::from_json(&json).unwrap();
    assert_eq!(decoded, original);
    assert!(
        serialization::from_json::<MatchState>(
            &json.replace("\"tick\": 42,", "\"tick\": 42,\n  \"unknown\": 1,")
        )
        .is_err()
    );
}

#[test]
fn golden_match_state_is_compatible() {
    let fixture = include_str!("../../../tests/golden/m1_match_state.json");
    let decoded: MatchState = serialization::from_json(fixture).unwrap();
    decoded.validate().unwrap();
    assert_eq!(decoded, state());
}

#[test]
fn golden_match_config_is_compatible_and_validated() {
    let fixture = include_str!("../../../tests/golden/m1_match_config.json");
    let mut config: MatchConfig = serialization::from_json(fixture).unwrap();
    config.validate().unwrap();
    config.control_period = Seconds(0.001);
    assert_eq!(config.validate().unwrap_err().path(), "control_period");
}

#[test]
fn reflection_is_an_involution_and_swaps_team_results() {
    let original = state();
    let reflected = original.reflected();
    assert_eq!(reflected.score_blue, original.score_yellow);
    assert!(reflected.events.contains(EventFlags::GOAL_YELLOW));
    assert_eq!(reflected.robots[0].team, Team::Yellow);
    assert_eq!(reflected.reflected(), original);
}

#[test]
fn duplicate_robot_identity_is_rejected() {
    let mut invalid = state();
    invalid.robots[1].id = RobotId::R0;
    assert_eq!(invalid.validate().unwrap_err().path(), "robots.id");
}

#[test]
fn reflection_catalog_exposes_match_roots() {
    let state_type = canonical_types()
        .iter()
        .find(|item| item.name == "MatchState")
        .unwrap();
    assert!(state_type.fields.iter().any(|field| field.name == "robots"));
    assert!(
        canonical_types()
            .iter()
            .any(|item| item.name == "MatchConfig")
    );
}
