//! M2 correctness, batch, and replay determinism suite.

use vsss_batch::PhysicsBatch;
use vsss_physics_api::PhysicsBackend;
use vsss_physics_rapier::RapierBackend;
use vsss_spec::{
    Angle, AngularVelocity, Distance, EventFlags, MatchConfig, MatchState, RobotAction,
    serialization,
};

fn world() -> RapierBackend {
    let config: MatchConfig =
        serialization::from_json(include_str!("../../../tests/golden/m1_match_config.json"))
            .unwrap();
    let state: MatchState =
        serialization::from_json(include_str!("../../../tests/golden/m1_match_state.json"))
            .unwrap();
    RapierBackend::new(config, state).unwrap()
}

fn stopped() -> [RobotAction; 6] {
    [RobotAction::wheel_velocity(AngularVelocity(0.0), AngularVelocity(0.0)); 6]
}

#[test]
fn equal_wheels_move_forward_without_commanded_rotation() {
    let mut backend = world();
    let mut actions = stopped();
    actions[0] = RobotAction::wheel_velocity(AngularVelocity(20.0), AngularVelocity(20.0));
    let before = backend.snapshot().robots[0];
    let after = backend.step(&actions).unwrap().robots[0];
    assert!(after.pose.x.get() > before.pose.x.get());
    assert!(after.twist.omega.get().abs() < 1.0e-6);
}

#[test]
fn wheel_velocity_respects_actuator_acceleration_limit() {
    let mut backend = world();
    let mut actions = stopped();
    actions[0] = RobotAction::wheel_velocity(AngularVelocity(30.0), AngularVelocity(-30.0));

    let after = backend.step(&actions).unwrap().robots[0];

    assert!((after.wheel_speed_left.get() - 0.8).abs() < 1.0e-6);
    assert!((after.wheel_speed_right.get() + 0.8).abs() < 1.0e-6);
}

#[test]
fn goal_box_keeps_robot_inside_back_and_side_walls() {
    let mut backend = world();
    let mut snapshot = backend.snapshot();
    snapshot.robots[0].pose.x = Distance(0.78);
    snapshot.robots[0].pose.y = Distance(0.0);
    snapshot.robots[0].pose.theta = Angle(0.0);
    backend.restore(&snapshot).unwrap();
    let mut actions = stopped();
    actions[0] = RobotAction::wheel_velocity(AngularVelocity(30.0), AngularVelocity(30.0));
    for _ in 0..500 {
        backend.step(&actions).unwrap();
    }
    assert!(backend.snapshot().robots[0].pose.x.get() < 0.86);

    snapshot.robots[0].pose.x = Distance(0.80);
    snapshot.robots[0].pose.y = Distance(0.0);
    snapshot.robots[0].pose.theta = Angle(core::f32::consts::FRAC_PI_2);
    snapshot.robots[0].wheel_speed_left = AngularVelocity(0.0);
    snapshot.robots[0].wheel_speed_right = AngularVelocity(0.0);
    backend.restore(&snapshot).unwrap();
    for _ in 0..500 {
        backend.step(&actions).unwrap();
    }
    assert!(backend.snapshot().robots[0].pose.y.get() < 0.21);
}

#[test]
fn snapshot_restore_replays_identically() {
    let mut backend = world();
    let snapshot = backend.snapshot();
    let actions = [RobotAction::wheel_velocity(AngularVelocity(8.0), AngularVelocity(12.0)); 6];
    for _ in 0..100 {
        backend.step(&actions).unwrap();
    }
    let expected = backend.snapshot();
    let checksum = backend.checksum();
    backend.restore(&snapshot).unwrap();
    for _ in 0..100 {
        backend.step(&actions).unwrap();
    }
    assert_eq!(backend.snapshot(), expected);
    assert_eq!(backend.checksum(), checksum);
}

#[test]
fn positive_goal_line_scores_for_blue() {
    let mut backend = world();
    let mut snapshot = backend.snapshot();
    snapshot.ball.x.0 = 0.751;
    snapshot.ball.y.0 = 0.0;
    backend.restore(&snapshot).unwrap();
    let state = backend.step(&stopped()).unwrap();
    assert_eq!(state.score_blue, snapshot.score_blue + 1);
    assert!(state.events.contains(EventFlags::GOAL_BLUE));
}

#[test]
fn resetting_one_batch_world_preserves_neighbor() {
    let mut batch = PhysicsBatch::new(vec![world(), world()]);
    batch.step(&[stopped(), stopped()]).unwrap();
    let neighbor = batch.world(0).snapshot();
    batch.reset_world(1).unwrap();
    assert_eq!(batch.world(0).snapshot(), neighbor);
    assert_eq!(batch.world(1).snapshot().tick, 42);
}
