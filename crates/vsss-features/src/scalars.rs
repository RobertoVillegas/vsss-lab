//! Four per-world scalars the reward and the terminal conditions read.
//!
//! Each is a few lines of geometry over the same state, and each was its own Python call per
//! world per decision. They are computed together because the crossing costs more than the
//! arithmetic: separately they would be four traversals and four boundary crossings for what is
//! one pass over six robot rows.

use crate::FeatureError;

/// Clearance a robot needs from the goal line when holding the defensive post, in metres.
const POST_INSET: f64 = 0.12;
/// Slack added to the contact radius so a resting touch still registers, in metres.
const CONTACT_SLACK: f64 = 0.002;

/// Dimensions the scalars are measured against.
#[derive(Clone, Copy, Debug)]
pub struct Field {
    /// Field length in metres.
    pub length: f64,
    /// Goal mouth width in metres.
    pub goal_width: f64,
    /// Robot body length in metres.
    pub robot_length: f64,
    /// Robot body width in metres.
    pub robot_width: f64,
    /// Ball radius in metres.
    pub ball_radius: f64,
    /// Distance teammates are expected to keep from each other, in metres.
    pub teammate_spacing: f64,
}

/// What one pass over a team's robots yields.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Scalars {
    /// Whether any active robot on the team is touching the ball.
    pub touches_ball: bool,
    /// Distance from the ball to the nearest active robot, in metres.
    pub closest_distance: f64,
    /// Mean squared crowding among active teammates, in `[0, 1]`.
    pub congestion: f64,
    /// Distance from the defensive post to the nearest active robot, in metres.
    pub defensive_distance: f64,
}

/// One active robot's position.
struct Active {
    x: f64,
    y: f64,
}

/// Collect the team's active robots, which every scalar here is measured over.
fn active_robots(state: &[f32], team: u8) -> Vec<Active> {
    let offset = if team == 0 { 0 } else { crate::TEAM_SIZE };
    (offset..offset + crate::TEAM_SIZE)
        .filter_map(|slot| {
            let base = crate::ROBOT_BASE + slot * crate::ROBOT_WIDTH;
            (f64::from(state[base + 10]).abs() > 0.0).then(|| Active {
                x: f64::from(state[base + 2]),
                y: f64::from(state[base + 3]),
            })
        })
        .collect()
}

/// Mean squared crowding over every active pair, zero when there is no pair to crowd.
fn congestion(robots: &[Active], spacing: f64) -> f64 {
    let mut total = 0.0;
    let mut pairs = 0usize;
    for (index, first) in robots.iter().enumerate() {
        for second in &robots[index + 1..] {
            let gap = (first.x - second.x).hypot(first.y - second.y);
            let crowding = ((spacing - gap) / spacing).max(0.0);
            total += crowding * crowding;
            pairs += 1;
        }
    }
    if pairs == 0 {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss)] // at most three pairs
    let divisor = pairs as f64;
    total / divisor
}

/// Measure the four scalars for one world.
///
/// # Errors
///
/// Returns an error when the team index is not 0 or 1, the state is too short, or the team has
/// no active robot — a distance to the nearest of nothing has no value to report.
pub fn team_scalars(state: &[f32], team: u8, field: Field) -> Result<Scalars, FeatureError> {
    if team > 1 {
        return Err(FeatureError::UnknownTeam);
    }
    if state.len() < crate::ROBOT_BASE + crate::ROBOT_COUNT * crate::ROBOT_WIDTH {
        return Err(FeatureError::RosterNotCanonical);
    }
    let robots = active_robots(state, team);
    if robots.is_empty() {
        return Err(FeatureError::RosterNotCanonical);
    }

    let ball_x = f64::from(state[5]);
    let ball_y = f64::from(state[6]);
    let contact_radius =
        field.robot_length.hypot(field.robot_width) / 2.0 + field.ball_radius + CONTACT_SLACK;

    let attack_sign = if team == 0 { 1.0 } else { -1.0 };
    let post_x = -attack_sign * (field.length / 2.0 - POST_INSET);
    let half_goal = field.goal_width / 2.0;
    let post_y = ball_y.clamp(-half_goal, half_goal);

    let mut closest = f64::INFINITY;
    let mut defensive = f64::INFINITY;
    for robot in &robots {
        closest = closest.min((ball_x - robot.x).hypot(ball_y - robot.y));
        defensive = defensive.min((post_x - robot.x).hypot(post_y - robot.y));
    }

    Ok(Scalars {
        touches_ball: closest <= contact_radius,
        closest_distance: closest,
        congestion: congestion(&robots, field.teammate_spacing),
        defensive_distance: defensive,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIELD: Field = Field {
        length: 1.5,
        goal_width: 0.4,
        robot_length: 0.075,
        robot_width: 0.075,
        ball_radius: 0.0215,
        teammate_spacing: 0.30,
    };

    fn state() -> Vec<f32> {
        let mut state = vec![0.0f32; crate::ROBOT_BASE + crate::ROBOT_COUNT * crate::ROBOT_WIDTH];
        state[5] = 0.50;
        for slot in 0..crate::ROBOT_COUNT {
            let base = crate::ROBOT_BASE + slot * crate::ROBOT_WIDTH;
            state[base + 10] = 1.0;
            #[allow(clippy::cast_precision_loss)] // six slots
            {
                state[base + 2] = 0.40f32.mul_add(slot as f32, -0.60);
            }
        }
        state
    }

    #[test]
    fn a_robot_on_the_ball_registers_a_touch() {
        let mut touching = state();
        touching[crate::ROBOT_BASE + 2] = touching[5];
        let scalars = team_scalars(&touching, 0, FIELD).expect("scalars");
        assert!(scalars.touches_ball);
        assert!(scalars.closest_distance < 1e-6);
    }

    #[test]
    fn robots_on_top_of_each_other_are_maximally_congested() {
        let mut stacked = state();
        for slot in 0..crate::TEAM_SIZE {
            stacked[crate::ROBOT_BASE + slot * crate::ROBOT_WIDTH + 2] = 0.0;
        }
        let scalars = team_scalars(&stacked, 0, FIELD).expect("scalars");
        assert!((scalars.congestion - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_lone_robot_has_no_pair_to_crowd() {
        let mut alone = state();
        alone[crate::ROBOT_BASE + crate::ROBOT_WIDTH + 10] = 0.0;
        alone[crate::ROBOT_BASE + 2 * crate::ROBOT_WIDTH + 10] = 0.0;
        let scalars = team_scalars(&alone, 0, FIELD).expect("scalars");
        assert!(scalars.congestion.abs() < f64::EPSILON);
    }

    #[test]
    fn a_team_with_nobody_in_play_has_no_nearest_robot() {
        let mut empty = state();
        for slot in 0..crate::TEAM_SIZE {
            empty[crate::ROBOT_BASE + slot * crate::ROBOT_WIDTH + 10] = 0.0;
        }
        assert!(matches!(
            team_scalars(&empty, 0, FIELD),
            Err(FeatureError::RosterNotCanonical)
        ));
    }
}
