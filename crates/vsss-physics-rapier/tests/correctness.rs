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
    snapshot.ball.x.0 = 0.770;
    snapshot.ball.y.0 = 0.0;
    snapshot.ball.vx.0 = 1.0;
    backend.restore(&snapshot).unwrap();
    let state = backend.step(&stopped()).unwrap();
    assert_eq!(state.score_blue, snapshot.score_blue + 1);
    assert!(state.events.contains(EventFlags::GOAL_BLUE));
    let next = backend.step(&stopped()).unwrap();
    assert_eq!(next.score_blue, state.score_blue);
    assert!(!next.events.contains(EventFlags::GOAL_BLUE));
}

#[test]
fn ball_center_crossing_alone_is_not_a_goal() {
    let mut backend = world();
    let mut snapshot = backend.snapshot();
    snapshot.ball.x.0 = 0.751;
    snapshot.ball.y.0 = 0.0;
    backend.restore(&snapshot).unwrap();
    let state = backend.step(&stopped()).unwrap();
    assert_eq!(state.score_blue, snapshot.score_blue);
    assert!(!state.events.contains(EventFlags::GOAL_BLUE));
}

#[test]
fn low_speed_ball_decelerates_continuously_past_rapier_sleep_window() {
    let mut backend = world();
    let mut snapshot = backend.snapshot();
    snapshot.ball.x = Distance(0.0);
    snapshot.ball.y = Distance(0.0);
    snapshot.ball.vx.0 = 0.0;
    snapshot.ball.vy.0 = 0.2;
    snapshot.ball.omega.0 = 0.0;
    backend.restore(&snapshot).unwrap();

    let mut previous_speed = snapshot.ball.vy.get();
    for _ in 0..600 {
        let state = backend.step(&stopped()).unwrap();
        let speed = state.ball.vx.get().hypot(state.ball.vy.get());
        assert!(speed > 0.0, "ball froze at tick {}", state.tick);
        assert!(
            speed <= previous_speed + 1.0e-6,
            "free motion accelerated from {previous_speed} to {speed}"
        );
        previous_speed = speed;
    }

    let state = backend.snapshot();
    assert!((0.12..0.14).contains(&previous_speed), "{previous_speed}");
    assert!(
        (0.45..0.55).contains(&state.ball.y.get()),
        "{}",
        state.ball.y.get()
    );
}

#[test]
fn active_stationary_ball_does_not_drift() {
    let mut backend = world();
    let mut snapshot = backend.snapshot();
    snapshot.ball.x = Distance(0.0);
    snapshot.ball.y = Distance(0.0);
    snapshot.ball.vx.0 = 0.0;
    snapshot.ball.vy.0 = 0.0;
    snapshot.ball.omega.0 = 0.0;
    backend.restore(&snapshot).unwrap();
    let initial = snapshot.ball;
    for _ in 0..1_200 {
        backend.step(&stopped()).unwrap();
    }
    let ball = backend.snapshot().ball;
    assert_eq!(ball.x, initial.x);
    assert_eq!(ball.y, initial.y);
    assert_eq!(ball.vx, initial.vx);
    assert_eq!(ball.vy, initial.vy);
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

#[test]
fn parallel_repeated_batch_matches_independent_worlds() {
    let actions = [RobotAction::wheel_velocity(AngularVelocity(8.0), AngularVelocity(12.0)); 6];
    let mut expected_worlds = (0..64).map(|_| world()).collect::<Vec<_>>();
    for backend in &mut expected_worlds {
        for _ in 0..4 {
            backend.step(&actions).unwrap();
        }
    }
    let mut batch = PhysicsBatch::new((0..64).map(|_| world()).collect());
    let actual = batch.step_repeated(&[actions; 64], 4).unwrap();
    assert_eq!(
        actual,
        expected_worlds
            .iter()
            .map(PhysicsBackend::snapshot)
            .collect::<Vec<_>>()
    );
}

#[test]
fn sustained_head_on_commands_do_not_overlap_robots() {
    let mut backend = world();
    let mut snapshot = backend.snapshot();
    snapshot.robots[0].pose.x = Distance(-0.10);
    snapshot.robots[0].pose.y = Distance(0.0);
    snapshot.robots[0].pose.theta = Angle(0.0);
    snapshot.robots[1].pose.x = Distance(0.10);
    snapshot.robots[1].pose.y = Distance(0.0);
    snapshot.robots[1].pose.theta = Angle(core::f32::consts::PI);
    backend.restore(&snapshot).unwrap();
    let mut actions = stopped();
    actions[0] = RobotAction::wheel_velocity(AngularVelocity(30.0), AngularVelocity(30.0));
    actions[1] = RobotAction::wheel_velocity(AngularVelocity(30.0), AngularVelocity(30.0));

    let mut minimum_separation = f32::MAX;
    for _ in 0..1_000 {
        let state = backend.step(&actions).unwrap();
        minimum_separation = minimum_separation
            .min((state.robots[1].pose.x.get() - state.robots[0].pose.x.get()).abs());
    }

    assert!(minimum_separation >= 0.0739, "{minimum_separation}");
}

#[test]
fn sustained_robot_command_does_not_engulf_ball() {
    let mut backend = world();
    let mut snapshot = backend.snapshot();
    snapshot.robots[0].pose.x = Distance(-0.10);
    snapshot.robots[0].pose.y = Distance(0.0);
    snapshot.robots[0].pose.theta = Angle(0.0);
    snapshot.ball.x = Distance(0.0);
    snapshot.ball.y = Distance(0.0);
    snapshot.ball.vx.0 = 0.0;
    snapshot.ball.vy.0 = 0.0;
    backend.restore(&snapshot).unwrap();
    let mut actions = stopped();
    actions[0] = RobotAction::wheel_velocity(AngularVelocity(30.0), AngularVelocity(30.0));

    let mut minimum_separation = f32::MAX;
    for _ in 0..1_000 {
        let state = backend.step(&actions).unwrap();
        minimum_separation = minimum_separation.min(
            (state.ball.x.get() - state.robots[0].pose.x.get())
                .hypot(state.ball.y.get() - state.robots[0].pose.y.get()),
        );
    }

    let required = 0.075 / 2.0 + 0.0215 - 0.0011;
    assert!(minimum_separation >= required, "{minimum_separation}");
}

#[test]
fn ball_is_deflected_by_clipped_field_corner() {
    let mut backend = world();
    let mut snapshot = backend.snapshot();
    snapshot.ball.x = Distance(0.60);
    snapshot.ball.y = Distance(0.50);
    snapshot.ball.vx.0 = 1.0;
    snapshot.ball.vy.0 = 1.0;
    backend.restore(&snapshot).unwrap();

    let mut maximum_corner_reach = f32::MIN;
    for _ in 0..300 {
        let state = backend.step(&stopped()).unwrap();
        maximum_corner_reach = maximum_corner_reach.max(state.ball.x.get() + state.ball.y.get());
    }

    // The chamfer face is x + y = 1.33 m before accounting for ball radius.
    assert!(maximum_corner_reach < 1.31, "{maximum_corner_reach}");
}
