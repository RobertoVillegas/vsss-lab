//! Turning circular primitive tokens into wheel commands.
//!
//! A port of `circular_primitive_wheel_actions` and the controller geometry it calls. Two
//! details of the reference are load-bearing and easy to lose in translation: it decodes the
//! token through `numpy.float32` arithmetic, and it rounds the skill index with Python's
//! round-half-to-even rather than the half-away-from-zero every other language means by "round".
//! Both are reproduced here rather than approximated.

use core::f64::consts::PI;

use crate::FeatureError;

/// Fraction of the wheel-speed limit the controller will spend on turning.
///
/// A small differential is already a fast yaw command on a 60 mm axle.
pub const TURN_AUTHORITY: f64 = 0.08;

/// Distance behind the ball the striker aims for before driving through it.
const CONTACT_OFFSET: f64 = 0.10;
/// How far past the ball the drive-through target sits once the robot is in the envelope.
const DRIVE_THROUGH: f64 = 0.28;
/// Radius within which the robot is considered to have acquired its contact point.
const ACQUISITION_ENVELOPE: f64 = 0.11;
/// Half-angle of the exit half-plane the robot must face before driving through.
const EXIT_HALF_ANGLE: f64 = 0.60;
/// Distance a navigate token projects its target ahead of the robot.
const NAVIGATE_REACH: f64 = 0.4;
/// Speed scale a strike falls back to when its target is not yet past the ball.
const ACQUIRE_SCALE: f64 = 0.72;
/// Authority factor for the ADR 0027 clearing-waypoint approach. The `go_to_target` arc has a
/// yaw-limited curvature proportional to speed, and at full authority the robot cuts inside
/// the arc and brushes the ball's contact radius while turning (measured: 0.079 m at max
/// speed, inside 0.082). Scaling both wheel requests keeps the commanded path identical but
/// lets the yaw build against the acceleration limit, and the tracked arc pulls clear of the
/// ball.
const CLEARING_APPROACH_AUTHORITY: f64 = 0.35;

/// What one decoded token asks a robot to do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Command {
    /// Which of stop, navigate or strike the token selected.
    pub skill: Skill,
    /// Requested heading in radians, in the canonical frame.
    pub direction: f64,
    /// Requested authority in `[0, 1]`.
    pub intensity: f64,
}

/// The three things a circular primitive token can ask for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Skill {
    /// Hold position.
    Stop,
    /// Drive along the requested heading.
    Navigate,
    /// Take the ball out along the requested heading.
    Strike,
}

/// Round half to even, which is what Python's one-argument `round` does.
///
/// The usual `f64::round` rounds halves away from zero, so a token landing exactly on `.5`
/// would select a different skill than the reference does.
fn round_half_to_even(value: f64) -> f64 {
    let nearest = value.round();
    let exactly_half = ((value - value.trunc()).abs() - 0.5).abs() < f64::EPSILON;
    let landed_odd = (nearest % 2.0).abs() > f64::EPSILON;
    if exactly_half && landed_odd {
        nearest - value.signum()
    } else {
        nearest
    }
}

/// Decode a bounded transport token into physical controller parameters.
///
/// The intensity and heading are clamped in `f32`, matching the reference's numpy arithmetic,
/// before being widened for the geometry that follows.
#[must_use]
pub fn decode(token: [f32; 3]) -> Command {
    let index = round_half_to_even(f64::from(token[0]) + 1.0).clamp(0.0, 2.0);
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // clamped to 0..=2
    let skill = match index as u8 {
        0 => Skill::Stop,
        1 => Skill::Navigate,
        _ => Skill::Strike,
    };
    Command {
        skill,
        direction: f64::from(token[1].clamp(-1.0, 1.0)) * PI,
        intensity: f64::from(((token[2] + 1.0) * 0.5).clamp(0.0, 1.0)),
    }
}

/// Read one robot's pose from the flattened state.
#[must_use]
pub fn robot_pose(state: &[f32], slot: usize) -> (f64, f64, f64) {
    let base = crate::ROBOT_BASE + slot * crate::ROBOT_WIDTH;
    (
        f64::from(state[base + 2]),
        f64::from(state[base + 3]),
        f64::from(state[base + 4]),
    )
}

/// Bounded wheel commands driving a robot toward a target.
///
/// `settle` tapers the speed inside half a metre so the robot comes to rest on the point. A
/// striker closing on a contact point that moves with the ball must not: measured, it crawls
/// after it at twelve per cent of its speed and the ball rolls away. See ADR 0022.
#[must_use]
pub fn go_to_target(pose: (f64, f64, f64), target: (f64, f64), settle: bool) -> [f32; 2] {
    let (x, y, theta) = pose;
    let delta_x = target.0 - x;
    let delta_y = target.1 - y;
    let distance = delta_x.hypot(delta_y);
    let error = (delta_y.atan2(delta_x) - theta + PI).rem_euclid(2.0 * PI) - PI;
    let taper = if settle {
        (2.0 * distance).min(1.0)
    } else {
        1.0
    };
    let forward = taper * error.cos().max(0.0);
    let turn = TURN_AUTHORITY * (error / (PI / 2.0)).clamp(-1.0, 1.0);
    #[allow(clippy::cast_possible_truncation)] // the controller's output is f32 by definition
    [
        (forward - turn).clamp(-1.0, 1.0) as f32,
        (forward + turn).clamp(-1.0, 1.0) as f32,
    ]
}

/// A candidate contact point and the ball position it was chosen for.
struct Contact {
    ball_x: f64,
    ball_y: f64,
    x: f64,
    y: f64,
}

/// Select a reachable behind-ball point, then drive through contact.
///
/// Reachability is judged at the authority the caller will actually execute, so a
/// reduced-intensity request cannot select an intercept it can never arrive at. The second
/// return says whether the target is the drive-through one, which is what decides whether the
/// approach may taper: an acquisition point moves with the ball and is not one to settle on.
#[must_use]
pub fn strike_target(
    state: &[f32],
    pose: (f64, f64, f64),
    direction: (f64, f64),
    ball_deceleration: f64,
    authority: f64,
    strike_clearing_enabled: bool,
    strike_clearing_distance: f64,
) -> ((f64, f64), bool) {
    let ball_x = f64::from(state[5]);
    let ball_y = f64::from(state[6]);
    let velocity_x = f64::from(state[7]);
    let velocity_y = f64::from(state[8]);
    let (robot_x, robot_y, theta) = pose;
    let (exit_x, exit_y) = direction;

    let scale = authority.max(1e-3);
    let maximum_robot_speed = 0.62 * scale;
    let maximum_turn_rate = 5.0 * scale;
    let heading_x = theta.cos();
    let heading_y = theta.sin();
    let speed = velocity_x.hypot(velocity_y);

    let mut selected = Contact {
        ball_x,
        ball_y,
        x: exit_x.mul_add(-CONTACT_OFFSET, ball_x),
        y: exit_y.mul_add(-CONTACT_OFFSET, ball_y),
    };
    for index in 0..7 {
        let elapsed = f64::from(index) * 0.1;
        let (candidate_x, candidate_y) = if index == 0 || speed <= 1e-8 {
            (ball_x, ball_y)
        } else {
            let travel = (speed * elapsed).min(speed * speed / (2.0 * ball_deceleration));
            (
                travel.mul_add(velocity_x / speed, ball_x),
                travel.mul_add(velocity_y / speed, ball_y),
            )
        };
        let acquisition_x = exit_x.mul_add(-CONTACT_OFFSET, candidate_x);
        let acquisition_y = exit_y.mul_add(-CONTACT_OFFSET, candidate_y);
        let displacement_x = acquisition_x - robot_x;
        let displacement_y = acquisition_y - robot_y;
        let distance = displacement_x.hypot(displacement_y);
        let heading_error = if distance <= 1e-8 {
            0.0
        } else {
            let cosine = heading_x.mul_add(displacement_x, heading_y * displacement_y) / distance;
            cosine.clamp(-1.0, 1.0).acos()
        };
        let arrival = distance / maximum_robot_speed + heading_error / maximum_turn_rate;
        selected = Contact {
            ball_x: candidate_x,
            ball_y: candidate_y,
            x: acquisition_x,
            y: acquisition_y,
        };
        if arrival <= elapsed + 0.08 {
            break;
        }
    }

    let acquisition_error = (selected.x - robot_x).hypot(selected.y - robot_y);
    let ball_vector_x = selected.ball_x - robot_x;
    let ball_vector_y = selected.ball_y - robot_y;
    let ball_distance = ball_vector_x.hypot(ball_vector_y);
    let aligned = ball_distance > 1e-8
        && ball_vector_x.mul_add(exit_x, ball_vector_y * exit_y) / ball_distance
            >= EXIT_HALF_ANGLE.cos();
    // ADR 0027: "arrive aligned, not merely near". A robot behind the ball is near; it must
    // also face the exit direction. The drive-through starts by pushing through the ball, and
    // a push launched from a sideways heading deflects the ball along that heading, up the
    // wing. Measured: the positional gate alone fires while the robot is still ~90° off the
    // exit, and the strike then grinds the ball off the line instead of converting.
    let heading_aligned = heading_x.mul_add(exit_x, heading_y * exit_y) >= EXIT_HALF_ANGLE.cos();
    // Differential drive cannot settle on a point with millimetric precision without
    // oscillation. Enter the drive-through phase once the robot is inside a body-scale
    // acquisition envelope, behind the ball, and faces the exit half-plane.
    if acquisition_error <= ACQUISITION_ENVELOPE && aligned && heading_aligned {
        return (
            (
                exit_x.mul_add(DRIVE_THROUGH, selected.ball_x),
                exit_y.mul_add(DRIVE_THROUGH, selected.ball_y),
            ),
            true,
        );
    }
    if strike_clearing_enabled && !(aligned && heading_aligned) {
        // ADR 0027: the defect is the approach, not the gate. From an angled start the robot
        // reaches the acquisition point still facing well off the exit direction; with the
        // acquisition phase unable to settle, it overshoots the point and crosses the ball's
        // contact radius, deflecting the ball along its heading. Route through a point on the
        // exit line behind the ball: the robot settles there, turns onto the exit direction
        // safely clear of the ball, and only then is sent to the acquisition point.
        // Returning `settle=true` is deliberate — the waypoint is a standing point, and a
        // full-speed passage through it drifts into an orbit around the ball while turning
        // (measured).
        let clearing_x = ball_x - (CONTACT_OFFSET + strike_clearing_distance) * exit_x;
        let clearing_y = ball_y - (CONTACT_OFFSET + strike_clearing_distance) * exit_y;
        if acquisition_error <= 0.20 && aligned {
            // The robot is inside the clearing region and behind the ball. Re-aim at the
            // acquisition point: the go_to_target holds the heading on the clearing approach
            // bearing, which faces away from the ball, and the release gate would never fire.
            // From here the acquisition is ahead, so the turn carries the heading onto the
            // exit direction while still clear of contact. Settling is deliberate: a full
            // authority turn carries the closing momentum through the acquisition point and
            // into the ball before the drive-through gate fires.
            return ((selected.x, selected.y), true);
        }
        return ((clearing_x, clearing_y), true);
    }
    ((selected.x, selected.y), false)
}

/// Execute one team's circular headings at the authority each token requested.
///
/// # Errors
///
/// Returns an error when the team index is not 0 or 1, the state is too short, or the
/// deceleration is not positive.
pub fn circular_primitive_wheel_actions(
    state: &[f32],
    team: u8,
    tokens: &[[f32; 3]; crate::TEAM_SIZE],
    ball_deceleration: f64,
    strike_clearing_enabled: bool,
    strike_clearing_distance: f64,
) -> Result<[[f32; 2]; crate::TEAM_SIZE], FeatureError> {
    if team > 1 {
        return Err(FeatureError::UnknownTeam);
    }
    if state.len() < crate::ROBOT_BASE + crate::ROBOT_COUNT * crate::ROBOT_WIDTH {
        return Err(FeatureError::RosterNotCanonical);
    }
    if ball_deceleration <= 0.0 {
        return Err(FeatureError::OutputShape);
    }
    let offset = if team == 0 { 0 } else { crate::TEAM_SIZE };
    let sign = if team == 0 { 1.0 } else { -1.0 };
    let mut result = [[0.0f32; 2]; crate::TEAM_SIZE];

    for (local_slot, token) in tokens.iter().enumerate() {
        let slot = offset + local_slot;
        let enabled = state[crate::ROBOT_BASE + slot * crate::ROBOT_WIDTH + 10];
        if enabled.abs() == 0.0 {
            continue;
        }
        let command = decode(*token);
        if command.skill == Skill::Stop || command.intensity <= 1e-4 {
            continue;
        }
        let direction = (
            sign * command.direction.cos(),
            sign * command.direction.sin(),
        );
        let pose = robot_pose(state, slot);
        // ADR 0027 identities: the clearing waypoint and the static acquisition point are
        // recognized by exact equality with the values strike_target returns, so the caller
        // can give the clearing phases their own authority without threading more returns.
        let static_acquisition = (
            direction.0.mul_add(-CONTACT_OFFSET, f64::from(state[5])),
            direction.1.mul_add(-CONTACT_OFFSET, f64::from(state[6])),
        );
        let clearing_waypoint = (
            f64::from(state[5]) - (CONTACT_OFFSET + strike_clearing_distance) * direction.0,
            f64::from(state[6]) - (CONTACT_OFFSET + strike_clearing_distance) * direction.1,
        );
        let (target, arrival_scale, settle) = if command.skill == Skill::Navigate {
            (
                (
                    direction.0.mul_add(NAVIGATE_REACH, pose.0),
                    direction.1.mul_add(NAVIGATE_REACH, pose.1),
                ),
                1.0,
                true,
            )
        } else {
            let (target, driving_through) = strike_target(
                state,
                pose,
                direction,
                ball_deceleration,
                command.intensity,
                strike_clearing_enabled,
                strike_clearing_distance,
            );
            // The clearing phases run at full turning authority: the waypoint approach and
            // the re-aimed turn are already slowed by the approach authority and the settle
            // taper, and the acquisition scale (0.72) is for the direct behind-ball
            // approach the clearing replaces.
            let arrival_scale = if target == static_acquisition || target == clearing_waypoint {
                1.0
            } else {
                let to_target_x = target.0 - f64::from(state[5]);
                let to_target_y = target.1 - f64::from(state[6]);
                if to_target_x.mul_add(direction.0, to_target_y * direction.1) > 0.0 {
                    1.0
                } else {
                    ACQUIRE_SCALE
                }
            };
            (target, arrival_scale, driving_through)
        };
        let wheels = go_to_target(pose, target, settle);
        // ADR 0027: the clearing waypoint is approached at reduced authority. Scaling both
        // wheel requests keeps the commanded path identical but lets the yaw build against
        // the acceleration limit, and the tracked arc pulls clear of the ball (measured: a
        // forward-only cut on the arc passes 0.061 from the ball, inside 0.082). The
        // re-aimed turn at the acquisition is exempt: it is a turn in place (the settle
        // taper already zeroes the forward), and scaling the yaw turns the release into a
        // crawl.
        let wheels = if target == clearing_waypoint {
            #[allow(clippy::cast_possible_truncation)]
            [
                wheels[0] * CLEARING_APPROACH_AUTHORITY as f32,
                wheels[1] * CLEARING_APPROACH_AUTHORITY as f32,
            ]
        } else {
            wheels
        };
        // The reference narrows the authority to f32 before scaling, so the product is an f32
        // multiplication rather than an f64 one rounded afterwards.
        #[allow(clippy::cast_possible_truncation)]
        let authority = (command.intensity * arrival_scale) as f32;
        result[local_slot] = [wheels[0] * authority, wheels[1] * authority];
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halves_round_to_even_as_python_does() {
        for (input, expected) in [
            (0.5, 0.0),
            (1.5, 2.0),
            (2.5, 2.0),
            (-0.5, 0.0),
            (-1.5, -2.0),
        ] {
            assert!(
                (round_half_to_even(input) - expected).abs() < f64::EPSILON,
                "{input}"
            );
        }
    }

    #[test]
    fn a_stop_token_leaves_the_wheels_alone() {
        let mut state = vec![0.0f32; crate::ROBOT_BASE + crate::ROBOT_COUNT * crate::ROBOT_WIDTH];
        for slot in 0..crate::ROBOT_COUNT {
            state[crate::ROBOT_BASE + slot * crate::ROBOT_WIDTH + 10] = 1.0;
        }
        let tokens = [[-1.0, 0.0, 1.0]; crate::TEAM_SIZE];
        let wheels =
            circular_primitive_wheel_actions(&state, 0, &tokens, 0.8, true, 0.16).expect("actions");
        assert!(wheels.iter().flatten().all(|value| value.abs() == 0.0));
    }

    #[test]
    fn a_disabled_robot_is_never_commanded() {
        let mut state = vec![0.0f32; crate::ROBOT_BASE + crate::ROBOT_COUNT * crate::ROBOT_WIDTH];
        state[5] = 0.4;
        state[crate::ROBOT_BASE + 10] = 0.0; // slot 0 out of play
        state[crate::ROBOT_BASE + crate::ROBOT_WIDTH + 10] = 1.0;
        state[crate::ROBOT_BASE + 2 * crate::ROBOT_WIDTH + 10] = 1.0;
        let tokens = [[1.0, 0.0, 1.0]; crate::TEAM_SIZE];
        let wheels =
            circular_primitive_wheel_actions(&state, 0, &tokens, 0.8, true, 0.16).expect("actions");
        assert!(wheels[0].iter().all(|value| value.abs() == 0.0));
        assert!(wheels[1].iter().any(|value| value.abs() > 0.0));
    }
}
