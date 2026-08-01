//! The scripted opponent's team controller.
//!
//! A port of `DynamicTeamController`. It is a controller, not an oracle: it plans once per
//! decision like everything else on the field, and it assigns its three duties from geometry
//! alone so no robot identity owns one.

use crate::FeatureError;
use crate::actions::{go_to_target, robot_pose};

/// Duties the scripted team distributes, in the reference's declaration order.
const DUTIES: usize = 3;

/// The six permutations in the order `itertools.permutations` yields them.
///
/// Python's `min` keeps the first of equal candidates, so enumerating in this order is what
/// makes a tie resolve to the same assignment.
const PERMUTATIONS: [[usize; DUTIES]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

/// How far in front of its own goal line the keeper sits, in metres.
const GOALIE_DEPTH: f64 = 0.68;
/// How far the keeper may stray from the centre of its goal, in metres.
const GOALIE_REACH: f64 = 0.18;
/// How far behind the ball the supporting robot waits, in metres.
const SUPPORT_TRAIL: f64 = 0.28;
/// How much of the ball's lateral offset the supporting robot mirrors.
const SUPPORT_MIRROR: f64 = -0.5;

/// Clamp a value the way the reference's `np.clip` does.
fn clamp(value: f64, low: f64, high: f64) -> f64 {
    value.max(low).min(high)
}

/// Where each duty wants a robot to be, in duty order: keeper, presser, support.
fn duty_targets(state: &[f32], attack_sign: f64) -> [(f64, f64); DUTIES] {
    let ball = (f64::from(state[5]), f64::from(state[6]));
    [
        (
            -GOALIE_DEPTH * attack_sign,
            clamp(ball.1, -GOALIE_REACH, GOALIE_REACH),
        ),
        ball,
        (
            attack_sign.mul_add(-SUPPORT_TRAIL, ball.0),
            SUPPORT_MIRROR * ball.1,
        ),
    ]
}

/// Assign duties to slots by minimizing the total distance each robot must travel.
fn assign(poses: &[(f64, f64, f64); DUTIES], targets: &[(f64, f64); DUTIES]) -> [usize; DUTIES] {
    let mut best = PERMUTATIONS[0];
    let mut best_cost = f64::INFINITY;
    for order in PERMUTATIONS {
        let cost: f64 = order
            .iter()
            .enumerate()
            .map(|(slot, duty)| {
                (poses[slot].0 - targets[*duty].0).hypot(poses[slot].1 - targets[*duty].1)
            })
            .sum();
        // Strictly less, so equal candidates keep the earlier permutation as `min` does.
        if cost < best_cost {
            best_cost = cost;
            best = order;
        }
    }
    best
}

/// Return one scripted team's wheel commands, ordered by its current slots.
///
/// The slot offset and attacking direction are derived from the team rather than passed in, so
/// the two cannot be combined into a geometry that does not exist.
///
/// # Errors
///
/// Returns an error when the team index is not 0 or 1, or the state is too short.
pub fn scripted_team_actions(state: &[f32], team: u8) -> Result<[[f32; 2]; DUTIES], FeatureError> {
    if team > 1 {
        return Err(FeatureError::UnknownTeam);
    }
    if state.len() < crate::ROBOT_BASE + crate::ROBOT_COUNT * crate::ROBOT_WIDTH {
        return Err(FeatureError::RosterNotCanonical);
    }
    let team_offset = if team == 0 { 0 } else { DUTIES };
    let attack_sign = if team == 0 { 1.0 } else { -1.0 };
    let targets = duty_targets(state, attack_sign);
    let poses: [(f64, f64, f64); DUTIES] =
        core::array::from_fn(|slot| robot_pose(state, team_offset + slot));
    let order = assign(&poses, &targets);
    Ok(core::array::from_fn(|slot| {
        go_to_target(poses[slot], targets[order[slot]])
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> Vec<f32> {
        let mut state = vec![0.0f32; crate::ROBOT_BASE + crate::ROBOT_COUNT * crate::ROBOT_WIDTH];
        state[5] = 0.40;
        state[6] = 0.10;
        for slot in 0..crate::ROBOT_COUNT {
            let base = crate::ROBOT_BASE + slot * crate::ROBOT_WIDTH;
            #[allow(clippy::cast_precision_loss)] // six slots
            {
                state[base + 2] = 0.15f32.mul_add(slot as f32, -0.40);
            }
            state[base + 10] = 1.0;
        }
        state
    }

    #[test]
    fn every_duty_is_taken_exactly_once() {
        let targets = duty_targets(&state(), 1.0);
        let poses = core::array::from_fn(|slot| robot_pose(&state(), slot));
        let mut order = assign(&poses, &targets);
        order.sort_unstable();
        assert_eq!(order, [0, 1, 2]);
    }

    #[test]
    fn both_teams_produce_commands_and_nothing_else_does() {
        assert!(scripted_team_actions(&state(), 0).is_ok());
        assert!(scripted_team_actions(&state(), 1).is_ok());
        assert!(matches!(
            scripted_team_actions(&state(), 2),
            Err(FeatureError::UnknownTeam)
        ));
    }

    #[test]
    fn commands_stay_within_the_wheel_limit() {
        let wheels = scripted_team_actions(&state(), 0).expect("actions");
        assert!(wheels.iter().flatten().all(|value| value.abs() <= 1.0));
    }
}
