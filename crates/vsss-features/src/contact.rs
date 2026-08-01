//! Sustained contact and deadlock, without punishing brief or productive challenges.
//!
//! A port of `_contact_deadlock_metrics`. The streaks are carried by the caller rather than held
//! here, because a world's episode boundary resets them and the environment already owns that
//! decision.

use crate::FeatureError;

/// Ally pairs per team, and opponent pairs between the two teams.
pub const ALLY_PAIRS: usize = 3;
/// Every controlled robot against every rival.
pub const OPPONENT_PAIRS: usize = crate::TEAM_SIZE * crate::TEAM_SIZE;

/// Dimensions and tolerances contact is judged against.
#[derive(Clone, Copy, Debug)]
pub struct Rules {
    /// Centre distance at or below which two robots count as touching, in metres.
    pub contact_distance: f64,
    /// Decisions a contact may persist before it counts as a deadlock.
    pub grace_steps: i64,
    /// Ball movement that makes a contested contact productive, in metres.
    pub meaningful_ball_displacement: f64,
    /// Robot body length in metres.
    pub robot_length: f64,
    /// Robot body width in metres.
    pub robot_width: f64,
    /// Ball radius in metres.
    pub ball_radius: f64,
}

/// What one decision's contacts amount to, for one world.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Contacts {
    /// Penalty charged for sustained ally contact.
    pub ally_penalty: f64,
    /// Penalty charged for sustained contact with opponents.
    pub opponent_penalty: f64,
    /// Updated per-pair ally streaks.
    pub ally_streaks: [i64; ALLY_PAIRS],
    /// Updated per-pair opponent streaks.
    pub opponent_streaks: [i64; OPPONENT_PAIRS],
    /// Ally pairs touching this decision.
    pub ally_contacts: i64,
    /// Opponent pairs touching this decision.
    pub opponent_contacts: i64,
    /// Ally pairs that crossed into deadlock this decision.
    pub ally_deadlocks: i64,
    /// Opponent pairs that crossed into deadlock this decision.
    pub opponent_deadlocks: i64,
    /// Pairs that broke a deadlock this decision, across both kinds.
    pub escapes: i64,
}

/// What one pass over a set of pairs produced.
struct Tally {
    contacts: i64,
    deadlocks: i64,
    escapes: i64,
    penalty: f64,
}

/// Whether a robot is in play.
fn active(state: &[f32], slot: usize) -> bool {
    f64::from(state[crate::ROBOT_BASE + slot * crate::ROBOT_WIDTH + 10]).abs() > 0.0
}

/// A robot's position on the field.
fn position(state: &[f32], slot: usize) -> (f64, f64) {
    let base = crate::ROBOT_BASE + slot * crate::ROBOT_WIDTH;
    (f64::from(state[base + 2]), f64::from(state[base + 3]))
}

/// Distance between two points.
fn distance(first: (f64, f64), second: (f64, f64)) -> f64 {
    (first.0 - second.0).hypot(first.1 - second.1)
}

/// The context a pair is judged in, shared by every pair in a world.
struct Frame {
    ball: (f64, f64),
    ball_moved: f64,
    ball_contact_distance: f64,
}

/// Advance the streaks over one set of pairs and charge what they are worth.
fn update(
    state: &[f32],
    pairs: &[(usize, usize)],
    previous: &[i64],
    current: &mut [i64],
    rules: Rules,
    frame: &Frame,
    moving_ball_is_productive: bool,
) -> Tally {
    let mut tally = Tally {
        contacts: 0,
        deadlocks: 0,
        escapes: 0,
        penalty: 0.0,
    };
    let mut penalties = 0.0;
    for (index, (first, second)) in pairs.iter().enumerate() {
        let touching = active(state, *first)
            && active(state, *second)
            && distance(position(state, *first), position(state, *second))
                <= rules.contact_distance;
        if touching {
            tally.contacts += 1;
            current[index] = previous[index] + 1;
            if current[index] == rules.grace_steps + 1 {
                tally.deadlocks += 1;
            }
            let ball_involved = distance(position(state, *first), frame.ball)
                <= frame.ball_contact_distance
                || distance(position(state, *second), frame.ball) <= frame.ball_contact_distance;
            // The reference gates this on a `preserve_ball_challenges` flag that both of its
            // call sites pass as true, so the gate has no live false branch to reproduce.
            let productive = ball_involved
                || (moving_ball_is_productive
                    && frame.ball_moved >= rules.meaningful_ball_displacement);
            if current[index] > rules.grace_steps && !productive {
                #[allow(clippy::cast_precision_loss)] // streaks are small decision counts
                let excess = (current[index] - rules.grace_steps) as f64;
                #[allow(clippy::cast_precision_loss)]
                let scale = rules.grace_steps as f64;
                penalties += (excess / scale).min(1.0);
            }
        } else {
            current[index] = 0;
            if previous[index] > rules.grace_steps {
                tally.escapes += 1;
            }
        }
    }
    #[allow(clippy::cast_precision_loss)] // at most nine pairs
    let divisor = pairs.len().max(1) as f64;
    tally.penalty = penalties / divisor;
    tally
}

/// Measure sustained contacts for one world.
///
/// # Errors
///
/// Returns an error when the team index is not 0 or 1, or the state is too short.
pub fn contact_metrics(
    state: &[f32],
    team: u8,
    previous_ball: (f64, f64),
    ally_streaks: &[i64; ALLY_PAIRS],
    opponent_streaks: &[i64; OPPONENT_PAIRS],
    rules: Rules,
) -> Result<Contacts, FeatureError> {
    if team > 1 {
        return Err(FeatureError::UnknownTeam);
    }
    if state.len() < crate::ROBOT_BASE + crate::ROBOT_COUNT * crate::ROBOT_WIDTH {
        return Err(FeatureError::RosterNotCanonical);
    }
    let offset = if team == 0 { 0 } else { crate::TEAM_SIZE };
    let rival_offset = crate::TEAM_SIZE - offset;
    let controlled: [usize; crate::TEAM_SIZE] = core::array::from_fn(|index| offset + index);
    let rivals: [usize; crate::TEAM_SIZE] = core::array::from_fn(|index| rival_offset + index);

    let ally_pairs = [
        (controlled[0], controlled[1]),
        (controlled[0], controlled[2]),
        (controlled[1], controlled[2]),
    ];
    let mut opponent_pairs = [(0usize, 0usize); OPPONENT_PAIRS];
    for (index, pair) in opponent_pairs.iter_mut().enumerate() {
        *pair = (
            controlled[index / crate::TEAM_SIZE],
            rivals[index % crate::TEAM_SIZE],
        );
    }

    let ball = (f64::from(state[5]), f64::from(state[6]));
    let frame = Frame {
        ball,
        ball_moved: distance(ball, previous_ball),
        ball_contact_distance: rules.robot_length.hypot(rules.robot_width) / 2.0
            + rules.ball_radius
            + 0.002,
    };

    let mut result = Contacts::default();
    let ally = update(
        state,
        &ally_pairs,
        ally_streaks,
        &mut result.ally_streaks,
        rules,
        &frame,
        false,
    );
    let rival = update(
        state,
        &opponent_pairs,
        opponent_streaks,
        &mut result.opponent_streaks,
        rules,
        &frame,
        true,
    );

    result.ally_penalty = ally.penalty;
    result.opponent_penalty = rival.penalty;
    result.ally_contacts = ally.contacts;
    result.opponent_contacts = rival.contacts;
    result.ally_deadlocks = ally.deadlocks;
    result.opponent_deadlocks = rival.deadlocks;
    result.escapes = ally.escapes + rival.escapes;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RULES: Rules = Rules {
        contact_distance: 0.09,
        grace_steps: 3,
        meaningful_ball_displacement: 0.05,
        robot_length: 0.075,
        robot_width: 0.075,
        ball_radius: 0.0215,
    };

    fn state() -> Vec<f32> {
        let mut state = vec![0.0f32; crate::ROBOT_BASE + crate::ROBOT_COUNT * crate::ROBOT_WIDTH];
        state[5] = 0.60; // ball far from every robot, so no contact is productive
        for slot in 0..crate::ROBOT_COUNT {
            let base = crate::ROBOT_BASE + slot * crate::ROBOT_WIDTH;
            state[base + 10] = 1.0;
            #[allow(clippy::cast_precision_loss)]
            {
                state[base + 2] = -0.50 + 0.30 * (slot as f32);
            }
        }
        state
    }

    #[test]
    fn a_streak_becomes_a_deadlock_exactly_once() {
        let mut touching = state();
        touching[crate::ROBOT_BASE + crate::ROBOT_WIDTH + 2] = touching[crate::ROBOT_BASE + 2];
        let mut ally = [0i64; ALLY_PAIRS];
        let opponent = [0i64; OPPONENT_PAIRS];
        let mut deadlocks = 0;
        for _ in 0..6 {
            let metrics =
                contact_metrics(&touching, 0, (0.60, 0.0), &ally, &opponent, RULES).expect("ok");
            ally = metrics.ally_streaks;
            deadlocks += metrics.ally_deadlocks;
        }
        assert_eq!(
            deadlocks, 1,
            "a sustained contact is one deadlock, not many"
        );
        assert_eq!(ally[0], 6);
    }

    #[test]
    fn breaking_a_deadlock_counts_as_an_escape_and_clears_the_streak() {
        let ally = [RULES.grace_steps + 2, 0, 0];
        let metrics = contact_metrics(&state(), 0, (0.60, 0.0), &ally, &[0; OPPONENT_PAIRS], RULES)
            .expect("ok");
        assert_eq!(metrics.escapes, 1);
        assert_eq!(metrics.ally_streaks[0], 0);
    }

    #[test]
    fn a_contact_over_the_ball_is_not_penalized() {
        let mut over_ball = state();
        let first = crate::ROBOT_BASE;
        let second = crate::ROBOT_BASE + crate::ROBOT_WIDTH;
        over_ball[5] = 0.0;
        over_ball[first + 2] = 0.0;
        over_ball[second + 2] = 0.0;
        let ally = [RULES.grace_steps + 2, 0, 0];
        let metrics = contact_metrics(
            &over_ball,
            0,
            (0.0, 0.0),
            &ally,
            &[0; OPPONENT_PAIRS],
            RULES,
        )
        .expect("ok");
        assert!(metrics.ally_contacts > 0);
        assert!(metrics.ally_penalty.abs() < f64::EPSILON);
    }
}
