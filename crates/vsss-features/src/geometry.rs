//! The attacking-line potential the reward shapes against.
//!
//! A port of `_goal_geometry_metrics`. The assignment it performs is deliberately the stateless
//! one: a potential must be a function of the state alone, so it cannot use the hysteretic
//! assignment the observation path carries, even though both are computed for the same state.

use crate::FeatureError;
use crate::roles::{Role, assign_roles};

/// Weight on how well the attacker is lined up between the ball and the goal.
const ALIGNMENT_WEIGHT: f64 = 0.45;
/// Weight on how much of the goal mouth the shot line still sees.
const APERTURE_WEIGHT: f64 = 0.25;
/// Weight on the attacker being close enough to act on the ball.
const PROXIMITY_WEIGHT: f64 = 0.15;
/// Weight on how far up the field the ball has been carried.
const PROGRESS_WEIGHT: f64 = 0.15;
/// Length scale over which proximity decays, in metres.
const PROXIMITY_SCALE: f64 = 0.25;

/// The field and ball dimensions the potential is measured against.
#[derive(Clone, Copy, Debug)]
pub struct Geometry {
    /// Field length in metres.
    pub field_length: f64,
    /// Goal mouth width in metres.
    pub goal_width: f64,
    /// Ball radius in metres.
    pub ball_radius: f64,
}

/// A controllable attacking line, described without declaring any field zone good or bad.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Metrics {
    /// The bounded potential the shaping term uses.
    pub potential: f64,
    /// How well the attacker is lined up between ball and goal, in `[0, 1]`.
    pub attacker_alignment: f64,
    /// How much of the goal mouth the shot line sees, in `[0, 1]`.
    pub goal_aperture: f64,
    /// How close the attacker is to the ball, in `[0, 1]`.
    pub controllable_proximity: f64,
    /// How far up the field the ball has been carried, in `[0, 1]`.
    pub attacking_progress: f64,
}

/// Cosine similarity, treating a degenerate vector as carrying no direction.
fn cosine_similarity(first: (f64, f64), second: (f64, f64)) -> f64 {
    let first_norm = first.0.hypot(first.1);
    let second_norm = second.0.hypot(second.1);
    if first_norm <= 1e-9 || second_norm <= 1e-9 {
        return 0.0;
    }
    (first.0.mul_add(second.0, first.1 * second.1) / (first_norm * second_norm)).clamp(-1.0, 1.0)
}

/// How much of the goal mouth the line from attacker through ball still reaches.
fn aperture(
    to_ball: (f64, f64),
    ball: (f64, f64),
    robot_y: f64,
    goal_x: f64,
    attack_sign: f64,
    geometry: Geometry,
) -> f64 {
    let forward_separation = attack_sign * to_ball.0;
    if forward_separation <= 1e-6 {
        return 0.0;
    }
    let usable_half_goal = (geometry.goal_width / 2.0 - geometry.ball_radius).max(1e-6);
    let intersection_y =
        (ball.1 - robot_y).mul_add((goal_x - ball.0).abs() / forward_separation, ball.1);
    (1.0 - intersection_y.abs() / usable_half_goal).clamp(0.0, 1.0)
}

/// Describe a controllable attacking line for one team.
///
/// # Errors
///
/// Returns an error when the role assignment cannot be made for this state and team.
pub fn goal_geometry_metrics(
    state: &[f32],
    team: u8,
    geometry: Geometry,
) -> Result<Metrics, FeatureError> {
    let assignment = assign_roles(state, team, None)?;
    let local_slot = assignment
        .roles
        .iter()
        .position(|role| *role == Role::Attacker)
        .ok_or(FeatureError::RosterNotCanonical)?;
    let slot = local_slot + if team == 0 { 0 } else { crate::TEAM_SIZE };
    let base = crate::ROBOT_BASE + slot * crate::ROBOT_WIDTH;
    let robot = (f64::from(state[base + 2]), f64::from(state[base + 3]));
    let ball = (f64::from(state[5]), f64::from(state[6]));
    let attack_sign = if team == 0 { 1.0 } else { -1.0 };
    let goal_x = attack_sign * geometry.field_length / 2.0;

    let to_ball = (ball.0 - robot.0, ball.1 - robot.1);
    let ball_to_goal = (goal_x - ball.0, -ball.1);
    let attacker_alignment = 0.5 * (cosine_similarity(to_ball, ball_to_goal) + 1.0);
    let goal_aperture = aperture(to_ball, ball, robot.1, goal_x, attack_sign, geometry);
    let controllable_proximity = (-to_ball.0.hypot(to_ball.1) / PROXIMITY_SCALE).exp();
    let attacking_progress = (attack_sign.mul_add(ball.0, geometry.field_length / 2.0)
        / geometry.field_length)
        .clamp(0.0, 1.0);

    let potential = PROGRESS_WEIGHT
        .mul_add(
            attacking_progress,
            PROXIMITY_WEIGHT.mul_add(
                controllable_proximity,
                ALIGNMENT_WEIGHT.mul_add(attacker_alignment, APERTURE_WEIGHT * goal_aperture),
            ),
        )
        .clamp(0.0, 1.0);

    Ok(Metrics {
        potential,
        attacker_alignment,
        goal_aperture,
        controllable_proximity,
        attacking_progress,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> Vec<f32> {
        let mut state = vec![0.0f32; crate::ROBOT_BASE + crate::ROBOT_COUNT * crate::ROBOT_WIDTH];
        state[5] = 0.30;
        for slot in 0..crate::ROBOT_COUNT {
            let base = crate::ROBOT_BASE + slot * crate::ROBOT_WIDTH;
            state[base + 1] = if slot < 3 { 0.0 } else { 1.0 };
            state[base + 10] = 1.0;
        }
        state
    }

    const GEOMETRY: Geometry = Geometry {
        field_length: 1.5,
        goal_width: 0.4,
        ball_radius: 0.0215,
    };

    #[test]
    fn the_potential_stays_bounded() {
        let metrics = goal_geometry_metrics(&state(), 0, GEOMETRY).expect("metrics");
        assert!((0.0..=1.0).contains(&metrics.potential));
        assert!((0.0..=1.0).contains(&metrics.goal_aperture));
        assert!((0.0..=1.0).contains(&metrics.attacking_progress));
    }

    #[test]
    fn a_ball_behind_the_attacker_sees_no_aperture() {
        let mut behind = state();
        // Put the ball behind every blue robot, so the line to the goal points backwards.
        behind[5] = -0.60;
        for slot in 0..crate::TEAM_SIZE {
            behind[crate::ROBOT_BASE + slot * crate::ROBOT_WIDTH + 2] = 0.20;
        }
        let metrics = goal_geometry_metrics(&behind, 0, GEOMETRY).expect("metrics");
        assert!(metrics.goal_aperture.abs() < f64::EPSILON);
    }
}
