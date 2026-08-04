//! Identity-free tactical responsibility assignment.
//!
//! A port of `vsss_train.roles`. The output is discrete, so equivalence with the reference is
//! exact agreement on the chosen permutation rather than a numeric tolerance, and the arithmetic
//! is kept in `f64` in the reference's own order to avoid a rounding difference flipping a tie.

use core::cmp::Ordering;

use crate::FeatureError;

/// The three responsibilities a team distributes among its robots on every decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Goes for the ball.
    Attacker,
    /// Offers an option behind the ball.
    Support,
    /// Holds the defensive line.
    Coverage,
}

impl Role {
    /// Rank matching the lexicographic order of the reference's role names.
    ///
    /// Python breaks a cost tie by comparing the tuple of role strings, so `attacker` precedes
    /// `coverage` precedes `support`. Reproducing that order is what makes ties agree.
    const fn rank(self) -> u8 {
        match self {
            Self::Attacker => 0,
            Self::Coverage => 1,
            Self::Support => 2,
        }
    }

    /// Index into a per-robot cost row.
    const fn cost_index(self) -> usize {
        match self {
            Self::Attacker => 0,
            Self::Support => 1,
            Self::Coverage => 2,
        }
    }

    /// One-hot position in the role feature block.
    const fn feature_index(self) -> usize {
        self.cost_index()
    }
}

/// Roles in the reference's declaration order, which fixes how permutations are enumerated.
const ROLES: [Role; 3] = [Role::Attacker, Role::Support, Role::Coverage];

/// The six permutations in the order `itertools.permutations` yields them.
const PERMUTATIONS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

/// Hysteresis strength: how much a role change must save before it is taken.
pub const SWITCH_PENALTY: f64 = 0.18;
/// How far the hysteretic choice may fall behind the unpenalized one before it is abandoned.
pub const EMERGENCY_MARGIN: f64 = 0.20;

/// The outcome of one assignment, including what it says about the previous one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Assignment {
    /// Chosen role per team slot, in slot order.
    pub roles: [Role; 3],
    /// Whether each slot's role differs from the previous assignment.
    pub changed: [bool; 3],
    /// Joint cost of the chosen permutation, including any switch penalty applied.
    pub cost: f64,
    /// Whether a defensive threat exists with nobody covering it.
    pub uncovered: bool,
}

/// Carries the previous assignment and the hysteresis strength so successive decisions do not
/// thrash between roles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HystereticAssigner {
    switch_penalty: f64,
    emergency_margin: f64,
    previous: Option<[Role; 3]>,
}

impl Default for HystereticAssigner {
    fn default() -> Self {
        Self::new()
    }
}

impl HystereticAssigner {
    /// Start with no history and the reference hysteresis strength, as at the beginning of an
    /// episode.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            switch_penalty: SWITCH_PENALTY,
            emergency_margin: EMERGENCY_MARGIN,
            previous: None,
        }
    }

    /// Start with no history and an explicit hysteresis strength.
    #[must_use]
    pub const fn with_hysteresis(switch_penalty: f64, emergency_margin: f64) -> Self {
        Self {
            switch_penalty,
            emergency_margin,
            previous: None,
        }
    }

    /// Forget the previous assignment; the next one is decided on cost alone.
    pub const fn reset(&mut self) {
        self.previous = None;
    }

    /// Assign roles for one state, remembering the result.
    ///
    /// # Errors
    ///
    /// Returns an error when the team index is not 0 or 1, or the state does not hold exactly
    /// three robots for that team.
    pub fn assign(&mut self, state: &[f32], team: u8) -> Result<Assignment, FeatureError> {
        let result = assign_roles_parameterized(
            state,
            team,
            self.previous,
            self.switch_penalty,
            self.emergency_margin,
        )?;
        self.previous = Some(result.roles);
        Ok(result)
    }
}

/// Per-robot cost of taking each role, indexed by [`Role::cost_index`].
struct RobotCosts {
    costs: [f64; 3],
    active: bool,
    x: f64,
    y: f64,
}

/// Where the ball is, where it is going and how much it threatens the defended goal.
struct BallFrame {
    x: f64,
    y: f64,
    projected_x: f64,
    projected_y: f64,
    defensive_threat: f64,
}

/// Clamp a value the way the reference's nested `max`/`min` calls do.
fn clamp(value: f64, low: f64, high: f64) -> f64 {
    value.max(low).min(high)
}

/// Read one robot's row from the flattened state.
fn robot_row(state: &[f32], slot: usize) -> &[f32] {
    let start = crate::ROBOT_BASE + slot * crate::ROBOT_WIDTH;
    &state[start..start + crate::ROBOT_WIDTH]
}

/// Summarize the ball once, since every robot's cost is measured against the same frame.
fn ball_frame(state: &[f32], attack_sign: f64) -> BallFrame {
    let x = f64::from(state[5]);
    let y = f64::from(state[6]);
    let velocity_x = f64::from(state[7]);
    let velocity_y = f64::from(state[8]);
    BallFrame {
        x,
        y,
        projected_x: clamp(x + velocity_x * 0.35, -0.75, 0.75),
        projected_y: clamp(y + velocity_y * 0.35, -0.55, 0.55),
        defensive_threat: clamp((-attack_sign * x - 0.15) / 0.55, 0.0, 1.0),
    }
}

/// Cost for one robot of taking each of the three roles.
fn robot_costs(row: &[f32], ball: &BallFrame, attack_sign: f64) -> RobotCosts {
    let x = f64::from(row[2]);
    let y = f64::from(row[3]);
    let speed = (f64::from(row[5]).hypot(f64::from(row[6])) + 0.20).max(0.20);
    let time_to_ball = (ball.projected_x - x).hypot(ball.projected_y - y) / speed;
    let goal_side = attack_sign * (ball.x - x);
    let attack_angle = (ball.projected_y - y).atan2(ball.projected_x - x).abs();

    let support_x = ball.x - attack_sign * 0.22;
    let support_y = clamp(ball.y * 0.55, -0.42, 0.42);
    let coverage_target_x = clamp(
        (1.0 - ball.defensive_threat).mul_add(
            -attack_sign * 0.75,
            ball.defensive_threat * (ball.x + attack_sign * 0.06),
        ),
        -0.70,
        0.70,
    );
    let coverage_y = clamp(
        ball.y * ball.defensive_threat.mul_add(0.35, 0.65),
        -0.34,
        0.34,
    );
    let behind = (attack_sign * (x - ball.x)).max(0.0);

    RobotCosts {
        costs: [
            0.20f64.mul_add(attack_angle, time_to_ball) + 0.45 * (-goal_side).max(0.0),
            0.35f64.mul_add(behind, (support_x - x).hypot(support_y - y)),
            1.1f64.mul_add(behind, (coverage_target_x - x).hypot(coverage_y - y)),
        ],
        active: f64::from(row[10]).abs() > 0.0,
        x,
        y,
    }
}

/// Joint cost of one permutation, before any hysteresis.
///
/// A robot that is out of play must take one of the trailing roles and an active one must not,
/// which the reference enforces with a prohibitive constant rather than by pruning candidates.
fn raw_cost(robots: &[RobotCosts; 3], roles: [Role; 3], active_count: usize) -> f64 {
    let mut total = 0.0;
    for (robot, role) in robots.iter().zip(roles) {
        total += robot.costs[role.cost_index()];
        let within_active = role.cost_index() < active_count;
        if robot.active != within_active {
            total += 1_000.0;
        }
    }
    total
}

/// Order two candidates the way Python's `min` over `(cost, roles)` tuples does.
fn precedes(cost: f64, roles: [Role; 3], best_cost: f64, best_roles: [Role; 3]) -> bool {
    match cost.partial_cmp(&best_cost) {
        Some(Ordering::Less) => true,
        Some(Ordering::Equal) => roles.map(Role::rank) < best_roles.map(Role::rank),
        _ => false,
    }
}

/// Best permutation under a cost function, with the reference's tie-break.
fn best_permutation(
    robots: &[RobotCosts; 3],
    active_count: usize,
    previous: Option<[Role; 3]>,
    switch_penalty: f64,
) -> ([Role; 3], f64) {
    let mut best_roles = [Role::Attacker; 3];
    let mut best_cost = f64::INFINITY;
    for order in PERMUTATIONS {
        let roles = order.map(|index| ROLES[index]);
        let switches = previous.map_or(0, |before| {
            roles
                .iter()
                .zip(before)
                .filter(|(role, was)| **role != *was)
                .count()
        });
        #[allow(clippy::cast_precision_loss)] // switches is at most three
        let cost = switch_penalty.mul_add(switches as f64, raw_cost(robots, roles, active_count));
        if best_cost.is_infinite() || precedes(cost, roles, best_cost, best_roles) {
            best_cost = cost;
            best_roles = roles;
        }
    }
    (best_roles, best_cost)
}

/// Whether any active robot is between the ball and the defended goal, or close enough to it.
fn defended(robots: &[RobotCosts; 3], ball: &BallFrame, attack_sign: f64) -> bool {
    robots.iter().any(|robot| {
        robot.active
            && (attack_sign * (robot.x - ball.x) <= -0.02
                || (robot.x - ball.x).hypot(robot.y - ball.y) <= 0.14)
    })
}

/// Minimize joint tactical cost over all six role permutations, with the reference hysteresis.
///
/// This delegates to [`assign_roles_parameterized`] with the [`SWITCH_PENALTY`] and
/// [`EMERGENCY_MARGIN`] constants, so existing callers and the native/Python equivalence tests
/// are unchanged.
///
/// # Errors
///
/// Returns an error when the team index is not 0 or 1, the state is too short, or the state does
/// not hold exactly three robots for that team.
pub fn assign_roles(
    state: &[f32],
    team: u8,
    previous: Option<[Role; 3]>,
) -> Result<Assignment, FeatureError> {
    assign_roles_parameterized(state, team, previous, SWITCH_PENALTY, EMERGENCY_MARGIN)
}

/// Minimize joint tactical cost over all six role permutations, with explicit hysteresis.
///
/// Passing `previous` applies hysteresis: a permutation that changes roles pays `switch_penalty`
/// per change, unless staying put costs `emergency_margin` more than the unpenalized best, in
/// which case the unpenalized choice wins.
///
/// # Errors
///
/// Returns an error when the team index is not 0 or 1, the state is too short, or the state does
/// not hold exactly three robots for that team.
pub fn assign_roles_parameterized(
    state: &[f32],
    team: u8,
    previous: Option<[Role; 3]>,
    switch_penalty: f64,
    emergency_margin: f64,
) -> Result<Assignment, FeatureError> {
    if team > 1 {
        return Err(FeatureError::UnknownTeam);
    }
    if state.len() < crate::ROBOT_BASE + crate::ROBOT_COUNT * crate::ROBOT_WIDTH {
        return Err(FeatureError::RosterNotCanonical);
    }
    let attack_sign = if team == 0 { 1.0 } else { -1.0 };
    let ball = ball_frame(state, attack_sign);

    let mut rows = Vec::with_capacity(crate::TEAM_SIZE);
    for slot in 0..crate::ROBOT_COUNT {
        let row = robot_row(state, slot);
        #[allow(clippy::cast_possible_truncation)]
        let owner = row[1] as i32;
        if owner == i32::from(team) {
            rows.push(robot_costs(row, &ball, attack_sign));
        }
    }
    let robots: [RobotCosts; 3] = rows
        .try_into()
        .map_err(|_: Vec<RobotCosts>| FeatureError::RosterNotCanonical)?;
    let active_count = robots.iter().filter(|robot| robot.active).count();

    let (mut selected, mut selected_cost) =
        best_permutation(&robots, active_count, previous, switch_penalty);
    if let Some(before) = previous {
        let (raw_roles, raw_best) = best_permutation(&robots, active_count, None, switch_penalty);
        if raw_cost(&robots, before, active_count) - raw_best >= emergency_margin {
            selected = raw_roles;
            selected_cost = raw_best;
        }
    }

    let changed =
        [0, 1, 2].map(|index| previous.is_some_and(|before| selected[index] != before[index]));
    Ok(Assignment {
        roles: selected,
        changed,
        cost: selected_cost,
        uncovered: ball.defensive_threat > 0.0 && !defended(&robots, &ball, attack_sign),
    })
}

/// Write the five role features per team slot: three one-hot, a change flag, a coverage flag.
///
/// # Errors
///
/// Returns an error when `out` is not exactly [`crate::TEAM_SIZE`] rows of
/// [`crate::ROLE_WIDTH`].
pub fn role_features(assignment: &Assignment, out: &mut [f32]) -> Result<(), FeatureError> {
    if out.len() != crate::TEAM_SIZE * crate::ROLE_WIDTH {
        return Err(FeatureError::OutputShape);
    }
    out.fill(0.0);
    for (slot, row) in out.chunks_exact_mut(crate::ROLE_WIDTH).enumerate() {
        row[assignment.roles[slot].feature_index()] = 1.0;
        row[3] = f32::from(u8::from(assignment.changed[slot]));
        row[4] = f32::from(u8::from(assignment.uncovered));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> Vec<f32> {
        let mut state = vec![0.0f32; crate::ROBOT_BASE + crate::ROBOT_COUNT * crate::ROBOT_WIDTH];
        state[5] = 0.30;
        state[6] = 0.05;
        for slot in 0..crate::ROBOT_COUNT {
            let base = crate::ROBOT_BASE + slot * crate::ROBOT_WIDTH;
            #[allow(clippy::cast_precision_loss)] // six slots
            {
                state[base + 1] = if slot < 3 { 0.0 } else { 1.0 };
                state[base + 2] = 0.1 * (slot as f32) - 0.3;
                state[base + 3] = 0.08 * (slot as f32) - 0.2;
            }
            state[base + 10] = 1.0;
        }
        state
    }

    #[test]
    fn every_role_is_taken_exactly_once() {
        let assignment = assign_roles(&state(), 0, None).expect("assignment");
        let mut ranks = assignment.roles.map(Role::rank);
        ranks.sort_unstable();
        assert_eq!(ranks, [0, 1, 2]);
    }

    #[test]
    fn hysteresis_reports_no_change_on_a_repeated_state() {
        let mut assigner = HystereticAssigner::new();
        let first = assigner.assign(&state(), 0).expect("first");
        let second = assigner.assign(&state(), 0).expect("second");
        assert_eq!(first.roles, second.roles);
        assert_eq!(second.changed, [false; 3]);
    }

    #[test]
    fn a_missing_team_is_rejected() {
        let mut broken = state();
        broken[crate::ROBOT_BASE + 1] = 1.0;
        assert!(matches!(
            assign_roles(&broken, 0, None),
            Err(FeatureError::RosterNotCanonical)
        ));
        assert!(matches!(
            assign_roles(&state(), 3, None),
            Err(FeatureError::UnknownTeam)
        ));
    }
}
