//! Backend-neutral physics lifecycle.

use vsss_spec::{MatchState, RobotAction, ValidationError};

/// Errors produced by a physics backend.
#[derive(Debug)]
pub enum PhysicsError {
    /// Canonical input was invalid.
    InvalidState(ValidationError),
    /// Snapshot schema/config is incompatible.
    IncompatibleSnapshot,
}

impl core::fmt::Display for PhysicsError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidState(error) => write!(formatter, "invalid state: {error}"),
            Self::IncompatibleSnapshot => formatter.write_str("incompatible snapshot"),
        }
    }
}

impl std::error::Error for PhysicsError {}

/// Fixed-step backend contract.
pub trait PhysicsBackend {
    /// Restores the configured kickoff state and returns it.
    ///
    /// # Errors
    ///
    /// Returns an error if the configured initial state cannot be restored.
    fn reset(&mut self) -> Result<MatchState, PhysicsError>;
    /// Advances exactly one configured fixed timestep.
    ///
    /// # Errors
    ///
    /// Returns an error if actions or resulting state violate the contract.
    fn step(&mut self, actions: &[RobotAction; 6]) -> Result<MatchState, PhysicsError>;
    /// Returns the complete canonical snapshot.
    fn snapshot(&self) -> MatchState;
    /// Rebuilds this world from a canonical snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error if the snapshot is invalid or incompatible.
    fn restore(&mut self, snapshot: &MatchState) -> Result<(), PhysicsError>;
    /// Returns a deterministic FNV-1a checksum of canonical state bits.
    fn checksum(&self) -> u64 {
        checksum_state(&self.snapshot())
    }
}

/// Computes a deterministic checksum without depending on a serialization format.
#[must_use]
pub fn checksum_state(state: &MatchState) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    let mut add = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    add(&state.schema_version.to_le_bytes());
    add(&state.tick.to_le_bytes());
    add(&state.simulation_time.get().to_bits().to_le_bytes());
    add(&state.score_blue.to_le_bytes());
    add(&state.score_yellow.to_le_bytes());
    for value in [
        state.ball.x.get(),
        state.ball.y.get(),
        state.ball.vx.get(),
        state.ball.vy.get(),
        state.ball.omega.get(),
    ] {
        add(&value.to_bits().to_le_bytes());
    }
    for robot in state.robots {
        add(&[robot.id as u8, robot.team as u8, u8::from(robot.enabled)]);
        for value in [
            robot.pose.x.get(),
            robot.pose.y.get(),
            robot.pose.theta.get(),
            robot.twist.vx.get(),
            robot.twist.vy.get(),
            robot.twist.omega.get(),
            robot.wheel_speed_left.get(),
            robot.wheel_speed_right.get(),
        ] {
            add(&value.to_bits().to_le_bytes());
        }
    }
    add(&state.events.0.to_le_bytes());
    hash
}
