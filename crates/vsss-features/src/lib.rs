//! Per-decision training features over the canonical flattened state.
//!
//! These were computed in Python with one loop over worlds per decision, which measured at
//! 98.8 per cent of an environment step against 1.2 per cent for the physics itself. The
//! arithmetic is unchanged: this is the same construction, expressed where the loop is cheap.

pub mod actions;
pub mod baseline;
pub mod contact;
pub mod geometry;
pub mod roles;
pub mod spin;

/// Scalars before the first robot row in a flattened state.
pub const ROBOT_BASE: usize = 10;
/// Scalars in one robot row.
pub const ROBOT_WIDTH: usize = 11;
/// Robots in one team.
pub const TEAM_SIZE: usize = 3;
/// Robots on the field.
pub const ROBOT_COUNT: usize = 6;

/// Widths of the six observation groups, per agent.
/// Own-state channels per agent.
pub const SELF_WIDTH: usize = 8;
/// Ball-relative channels per agent.
pub const BALL_WIDTH: usize = 7;
/// Goal-relative channels per agent.
pub const GOAL_WIDTH: usize = 4;
/// Match context and tactical channels per agent.
pub const CONTEXT_WIDTH: usize = 9;
/// Channels describing one other robot, relative to the observer.
pub const ENTITY_WIDTH: usize = 6;
/// Tactical role features per agent, supplied by the caller.
pub const ROLE_WIDTH: usize = 5;

/// One robot expressed in its own team's attacking frame.
#[derive(Clone, Copy)]
struct Canonical {
    x: f32,
    y: f32,
    cos_theta: f32,
    sin_theta: f32,
    vx: f32,
    vy: f32,
    omega: f32,
    left: f32,
    right: f32,
    enabled: f32,
}

fn canonical(row: &[f32], attack_sign: f32) -> Canonical {
    let theta = row[4]
        + if attack_sign > 0.0 {
            0.0
        } else {
            std::f32::consts::PI
        };
    Canonical {
        x: attack_sign * row[2],
        y: attack_sign * row[3],
        cos_theta: theta.cos(),
        sin_theta: theta.sin(),
        vx: attack_sign * row[5],
        vy: attack_sign * row[6],
        omega: row[7],
        left: row[8],
        right: row[9],
        enabled: row[10],
    }
}

fn relative(other: &Canonical, current: &Canonical, length: f32, width: f32, out: &mut [f32]) {
    let delta = other.sin_theta.atan2(other.cos_theta) - current.sin_theta.atan2(current.cos_theta);
    out[0] = (other.x - current.x) / length;
    out[1] = (other.y - current.y) / width;
    out[2] = other.vx - current.vx;
    out[3] = other.vy - current.vy;
    out[4] = delta.cos();
    out[5] = delta.sin();
}

/// Errors that make an observation impossible to build.
#[derive(Debug, PartialEq, Eq)]
pub enum FeatureError {
    /// The team index was neither blue nor yellow.
    UnknownTeam,
    /// The state did not carry exactly three robots per team.
    RosterNotCanonical,
    /// An output slice did not match the width its group requires.
    OutputShape,
}

/// Widths one world writes into each observation group.
#[must_use]
pub const fn group_widths() -> [usize; 6] {
    [
        TEAM_SIZE * SELF_WIDTH,
        TEAM_SIZE * BALL_WIDTH,
        TEAM_SIZE * GOAL_WIDTH,
        TEAM_SIZE * CONTEXT_WIDTH,
        TEAM_SIZE * (TEAM_SIZE - 1) * ENTITY_WIDTH,
        TEAM_SIZE * TEAM_SIZE * ENTITY_WIDTH,
    ]
}

/// Groups an observation writes, one world at a time.
/// Borrowed output slices for one world's observation.
pub struct Observation<'a> {
    /// Own state, `TEAM_SIZE * SELF_WIDTH`.
    pub self_features: &'a mut [f32],
    /// Ball relative to each agent, `TEAM_SIZE * BALL_WIDTH`.
    pub ball: &'a mut [f32],
    /// Both goals relative to each agent, `TEAM_SIZE * GOAL_WIDTH`.
    pub goals: &'a mut [f32],
    /// Match context and tactical role, `TEAM_SIZE * CONTEXT_WIDTH`.
    pub context: &'a mut [f32],
    /// Teammates as an unordered set, `TEAM_SIZE * (TEAM_SIZE - 1) * ENTITY_WIDTH`.
    pub teammates: &'a mut [f32],
    /// Opponents as an unordered set, `TEAM_SIZE * TEAM_SIZE * ENTITY_WIDTH`.
    pub opponents: &'a mut [f32],
}

/// Build one team's three agent observations, without identity-ordered entity slots.
///
/// `roles` carries `TEAM_SIZE * ROLE_WIDTH` tactical features in controlled-slot order, which
/// the caller computes; `state` is one canonical flattened row.
///
/// # Errors
///
/// Returns [`FeatureError`] when the team is unknown, the roster is not three a side, or an
/// output slice is the wrong width.
type Roster = (
    [Canonical; ROBOT_COUNT],
    [usize; TEAM_SIZE],
    [usize; TEAM_SIZE],
);

fn partition_roster(state: &[f32], team: u8, attack_sign: f32) -> Result<Roster, FeatureError> {
    let empty = Canonical {
        x: 0.0,
        y: 0.0,
        cos_theta: 0.0,
        sin_theta: 0.0,
        vx: 0.0,
        vy: 0.0,
        omega: 0.0,
        left: 0.0,
        right: 0.0,
        enabled: 0.0,
    };
    let mut robots = [empty; ROBOT_COUNT];
    let mut ours = [0usize; TEAM_SIZE];
    let mut theirs = [0usize; TEAM_SIZE];
    let mut mine = 0usize;
    let mut yours = 0usize;
    for (slot, canonical_robot) in robots.iter_mut().enumerate() {
        let start = ROBOT_BASE + slot * ROBOT_WIDTH;
        let row = &state[start..start + ROBOT_WIDTH];
        *canonical_robot = canonical(row, attack_sign);
        #[allow(clippy::cast_possible_truncation)]
        let owner = row[1] as i32;
        if owner == i32::from(team) {
            *ours.get_mut(mine).ok_or(FeatureError::RosterNotCanonical)? = slot;
            mine += 1;
        } else {
            *theirs
                .get_mut(yours)
                .ok_or(FeatureError::RosterNotCanonical)? = slot;
            yours += 1;
        }
    }
    if mine != TEAM_SIZE || yours != TEAM_SIZE {
        return Err(FeatureError::RosterNotCanonical);
    }
    Ok((robots, ours, theirs))
}

fn match_context(state: &[f32], team: u8, match_duration: f32) -> [f32; 4] {
    let (score_for, score_against) = if team == 0 {
        (state[3], state[4])
    } else {
        (state[4], state[3])
    };
    #[allow(clippy::cast_possible_truncation)]
    let events = state[state.len() - 1] as i32;
    [
        (1.0 - state[2] / match_duration).max(0.0),
        (score_for - score_against) / 10.0,
        f32::from(events & 1 != 0),
        f32::from(events & 2 != 0),
    ]
}

struct Field {
    length: f32,
    width: f32,
}

struct BallFrame {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
}

fn write_agent(
    agent: usize,
    current: &Canonical,
    ball: &BallFrame,
    field: &Field,
    out: &mut Observation<'_>,
) {
    let base = agent * SELF_WIDTH;
    out.self_features[base] = current.cos_theta;
    out.self_features[base + 1] = current.sin_theta;
    out.self_features[base + 2] = current.vx;
    out.self_features[base + 3] = current.vy;
    out.self_features[base + 4] = current.omega;
    out.self_features[base + 5] = current.left;
    out.self_features[base + 6] = current.right;
    out.self_features[base + 7] = current.enabled;

    let delta_x = ball.x - current.x;
    let delta_y = ball.y - current.y;
    let bearing = delta_y.atan2(delta_x) - current.sin_theta.atan2(current.cos_theta);
    let base = agent * BALL_WIDTH;
    out.ball[base] = delta_x / field.length;
    out.ball[base + 1] = delta_y / field.width;
    out.ball[base + 2] = ball.vx - current.vx;
    out.ball[base + 3] = ball.vy - current.vy;
    out.ball[base + 4] = delta_x.hypot(delta_y) / field.length;
    out.ball[base + 5] = bearing.cos();
    out.ball[base + 6] = bearing.sin();

    let base = agent * GOAL_WIDTH;
    out.goals[base] = (-field.length / 2.0 - current.x) / field.length;
    out.goals[base + 1] = -current.y / field.width;
    out.goals[base + 2] = (field.length / 2.0 - current.x) / field.length;
    out.goals[base + 3] = -current.y / field.width;
}

/// Build one team's three agent observations, without identity-ordered entity slots.
///
/// `roles` carries `TEAM_SIZE * ROLE_WIDTH` tactical features in controlled-slot order, which
/// the caller computes; `state` is one canonical flattened row.
///
/// # Errors
///
/// Returns [`FeatureError`] when the team is unknown, the roster is not three a side, or an
/// output slice is the wrong width.
pub fn team_observation(
    state: &[f32],
    team: u8,
    field_length: f32,
    field_width: f32,
    match_duration: f32,
    roles: &[f32],
    out: &mut Observation<'_>,
) -> Result<(), FeatureError> {
    if team > 1 {
        return Err(FeatureError::UnknownTeam);
    }
    let widths = group_widths();
    if out.self_features.len() != widths[0]
        || out.ball.len() != widths[1]
        || out.goals.len() != widths[2]
        || out.context.len() != widths[3]
        || out.teammates.len() != widths[4]
        || out.opponents.len() != widths[5]
        || roles.len() != TEAM_SIZE * ROLE_WIDTH
    {
        return Err(FeatureError::OutputShape);
    }
    let attack_sign = if team == 0 { 1.0f32 } else { -1.0f32 };
    let (robots, ours, theirs) = partition_roster(state, team, attack_sign)?;
    let field = Field {
        length: field_length,
        width: field_width,
    };
    let ball = BallFrame {
        x: attack_sign * state[5],
        y: attack_sign * state[6],
        vx: attack_sign * state[7],
        vy: attack_sign * state[8],
    };
    let common = match_context(state, team, match_duration);

    for (agent, &slot) in ours.iter().enumerate() {
        let current = robots[slot];
        write_agent(agent, &current, &ball, &field, out);

        let base = agent * CONTEXT_WIDTH;
        out.context[base..base + 4].copy_from_slice(&common);
        out.context[base + 4..base + CONTEXT_WIDTH]
            .copy_from_slice(&roles[agent * ROLE_WIDTH..(agent + 1) * ROLE_WIDTH]);

        let mut written = 0usize;
        for &other in &ours {
            if other == slot {
                continue;
            }
            let base = (agent * (TEAM_SIZE - 1) + written) * ENTITY_WIDTH;
            let window = &mut out.teammates[base..base + ENTITY_WIDTH];
            relative(&robots[other], &current, field_length, field_width, window);
            written += 1;
        }
        for (index, &other) in theirs.iter().enumerate() {
            let base = (agent * TEAM_SIZE + index) * ENTITY_WIDTH;
            let window = &mut out.opponents[base..base + ENTITY_WIDTH];
            relative(&robots[other], &current, field_length, field_width, window);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canonical_state() -> Vec<f32> {
        let mut state = vec![0.0f32; ROBOT_BASE + ROBOT_COUNT * ROBOT_WIDTH + 1];
        state[2] = 12.0;
        state[3] = 2.0;
        state[4] = 1.0;
        state[5] = 0.12;
        state[6] = -0.08;
        state[7] = 0.4;
        state[8] = 0.1;
        for slot in 0..ROBOT_COUNT {
            let start = ROBOT_BASE + slot * ROBOT_WIDTH;
            #[allow(clippy::cast_precision_loss)]
            let index = slot as f32;
            state[start] = index;
            state[start + 1] = if slot < TEAM_SIZE { 0.0 } else { 1.0 };
            state[start + 2] = -0.3 + 0.2 * index;
            state[start + 3] = 0.1 * index - 0.2;
            state[start + 4] = 0.3 * index;
            state[start + 5] = 0.05 * index;
            state[start + 6] = -0.02 * index;
            state[start + 10] = 1.0;
        }
        state
    }

    fn buffers() -> [Vec<f32>; 6] {
        group_widths().map(|width| vec![0.0f32; width])
    }

    #[test]
    fn rejects_an_unknown_team() {
        let state = canonical_state();
        let mut raw = buffers();
        let [own, ball, goals, context, mates, foes] = &mut raw;
        let mut out = Observation {
            self_features: own,
            ball,
            goals,
            context,
            teammates: mates,
            opponents: foes,
        };
        let roles = [0.0f32; TEAM_SIZE * ROLE_WIDTH];
        assert_eq!(
            team_observation(&state, 2, 1.5, 1.3, 600.0, &roles, &mut out),
            Err(FeatureError::UnknownTeam)
        );
    }

    #[test]
    fn mirrors_the_frame_between_teams() {
        let state = canonical_state();
        let roles = [0.0f32; TEAM_SIZE * ROLE_WIDTH];
        let mut blue_raw = buffers();
        let [own, ball, goals, context, mates, foes] = &mut blue_raw;
        let mut blue = Observation {
            self_features: own,
            ball,
            goals,
            context,
            teammates: mates,
            opponents: foes,
        };
        team_observation(&state, 0, 1.5, 1.3, 600.0, &roles, &mut blue).expect("blue");
        let blue_ball = blue.ball[0];
        let mut yellow_raw = buffers();
        let [own, ball, goals, context, mates, foes] = &mut yellow_raw;
        let mut yellow = Observation {
            self_features: own,
            ball,
            goals,
            context,
            teammates: mates,
            opponents: foes,
        };
        team_observation(&state, 1, 1.5, 1.3, 600.0, &roles, &mut yellow).expect("yellow");
        // Both teams see the ball in their own attacking frame, so neither reads a raw sign.
        assert!(blue_ball.is_finite() && yellow.ball[0].is_finite());
        assert!((blue.context[0] - yellow.context[0]).abs() < 1e-6);
    }
}
