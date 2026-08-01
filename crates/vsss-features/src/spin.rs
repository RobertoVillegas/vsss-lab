//! Finding robots that rotate in place instead of playing.
//!
//! A port of `_idle_spin_flags`. Rotation is judged on measured angular speed rather than on the
//! wheel differential, because the differential a policy can request depends on the action
//! parser: a geometric controller spends at most a small fraction of the wheel limit on turning,
//! so a command-space threshold either cannot fire at all or degenerates into "is the robot
//! aiming a few degrees off". Angular speed carries the same meaning for every parser.

use crate::FeatureError;

/// The four conditions that together mean a robot is spinning rather than playing.
#[derive(Clone, Copy, Debug)]
pub struct Thresholds {
    /// Angular speed above which a robot counts as rotating, in radians per second.
    pub angular_speed: f64,
    /// Drive intensity below which the robot is not asking to move.
    pub drive: f64,
    /// Linear speed below which the robot is not moving.
    pub speed: f64,
    /// Distance from the ball beyond which spinning cannot be play, in metres.
    pub ball_distance: f64,
}

/// Per-slot spin flags and the intensity the penalty is scaled by.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Spin {
    /// Whether each team slot is spinning in place.
    pub flags: [bool; crate::TEAM_SIZE],
    /// Angular speed relative to twice the threshold, saturating at one.
    pub intensity: [f32; crate::TEAM_SIZE],
}

/// Find robots rotating in place, slow, remote from the ball, and not asking to drive.
///
/// `actions` holds the normalized wheel commands the team is executing, in slot order.
///
/// # Errors
///
/// Returns an error when the team index is not 0 or 1, or the state is too short.
pub fn idle_spin(
    state: &[f32],
    team: u8,
    actions: &[[f32; 2]; crate::TEAM_SIZE],
    thresholds: Thresholds,
) -> Result<Spin, FeatureError> {
    if team > 1 {
        return Err(FeatureError::UnknownTeam);
    }
    if state.len() < crate::ROBOT_BASE + crate::ROBOT_COUNT * crate::ROBOT_WIDTH {
        return Err(FeatureError::RosterNotCanonical);
    }
    let offset = if team == 0 { 0 } else { crate::TEAM_SIZE };
    let ball_x = f64::from(state[5]);
    let ball_y = f64::from(state[6]);
    // Proportional above the threshold and saturating at twice it, so the configured
    // coefficient keeps a bounded per-decision meaning.
    let reference = (2.0 * thresholds.angular_speed).max(1e-6);

    let mut spin = Spin::default();
    for (local_slot, action) in actions.iter().enumerate() {
        let base = crate::ROBOT_BASE + (offset + local_slot) * crate::ROBOT_WIDTH;
        if f64::from(state[base + 10]).abs() == 0.0 {
            continue;
        }
        let angular_speed = f64::from(state[base + 7]).abs();
        let speed = f64::from(state[base + 5]).hypot(f64::from(state[base + 6]));
        let distance =
            (ball_x - f64::from(state[base + 2])).hypot(ball_y - f64::from(state[base + 3]));
        let drive = f64::from(action[1] + action[0]).abs() / 2.0;
        spin.flags[local_slot] = angular_speed > thresholds.angular_speed
            && drive < thresholds.drive
            && speed < thresholds.speed
            && distance > thresholds.ball_distance;
        #[allow(clippy::cast_possible_truncation)] // the penalty scale is f32 by definition
        {
            spin.intensity[local_slot] = (angular_speed / reference).clamp(0.0, 1.0) as f32;
        }
    }
    Ok(spin)
}

#[cfg(test)]
mod tests {
    use super::*;

    const THRESHOLDS: Thresholds = Thresholds {
        angular_speed: 2.0,
        drive: 0.2,
        speed: 0.05,
        ball_distance: 0.25,
    };

    fn state() -> Vec<f32> {
        let mut state = vec![0.0f32; crate::ROBOT_BASE + crate::ROBOT_COUNT * crate::ROBOT_WIDTH];
        state[5] = 0.60;
        for slot in 0..crate::ROBOT_COUNT {
            state[crate::ROBOT_BASE + slot * crate::ROBOT_WIDTH + 10] = 1.0;
        }
        state
    }

    #[test]
    fn a_spinning_robot_far_from_the_ball_is_flagged() {
        let mut spinning = state();
        spinning[crate::ROBOT_BASE + 7] = 6.0; // angular speed well above the threshold
        let spin =
            idle_spin(&spinning, 0, &[[0.0; 2]; crate::TEAM_SIZE], THRESHOLDS).expect("spin");
        assert!(spin.flags[0]);
        assert!(!spin.flags[1]);
        assert!((spin.intensity[0] - 1.0).abs() < f32::EPSILON); // saturates at twice
    }

    #[test]
    fn a_robot_asking_to_drive_is_not_idle() {
        let mut spinning = state();
        spinning[crate::ROBOT_BASE + 7] = 6.0;
        let driving = [[1.0f32, 1.0]; crate::TEAM_SIZE];
        let spin = idle_spin(&spinning, 0, &driving, THRESHOLDS).expect("spin");
        assert!(!spin.flags[0]);
        // The intensity still reports the rotation; only the flag judges intent.
        assert!(spin.intensity[0] > 0.0);
    }

    #[test]
    fn a_robot_next_to_the_ball_is_never_idle_spinning() {
        let mut close = state();
        close[5] = 0.0;
        close[crate::ROBOT_BASE + 7] = 6.0;
        let spin = idle_spin(&close, 0, &[[0.0; 2]; crate::TEAM_SIZE], THRESHOLDS).expect("spin");
        assert!(!spin.flags[0]);
    }
}
