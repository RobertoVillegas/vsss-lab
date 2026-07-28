//! Typed callbacks for independently implemented Rust controllers.

use thiserror::Error;
use vsss_protocol::{ROBOTS_PER_TEAM, RobotCommand, wire};
use vsss_spec::{MatchConfig, MatchState};

/// Error returned by controller policy code.
#[derive(Debug, Error)]
#[error("{message}")]
pub struct ControllerError {
    message: String,
}

impl ControllerError {
    /// Construct a controller error without exposing an SDK-specific dependency.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Server event delivered outside the action callback.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatchEvent {
    /// Stable event category.
    pub kind: wire::MatchEventKind,
    /// Optional human-readable diagnostic.
    pub detail: Option<String>,
}

/// Terminal result delivered exactly once.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatchResult {
    /// Blue goals.
    pub score_blue: u16,
    /// Yellow goals.
    pub score_yellow: u16,
    /// Replay digest produced by the authoritative server.
    pub replay_sha256: [u8; 32],
    /// Adjudication reason.
    pub reason: Option<String>,
}

/// Minimal policy interface; transport, heartbeat, and framing stay in the SDK.
pub trait Controller {
    /// Reset policy-local state for a new match.
    ///
    /// # Errors
    ///
    /// May reject an unsupported configuration.
    fn on_reset(
        &mut self,
        config: &MatchConfig,
        initial: &MatchState,
    ) -> Result<(), ControllerError>;

    /// Produce exactly three robot commands for one observation.
    ///
    /// # Errors
    ///
    /// May report a policy inference failure; the server will apply fallback.
    fn act(
        &mut self,
        observation: &MatchState,
    ) -> Result<[RobotCommand; ROBOTS_PER_TEAM], ControllerError>;

    /// Observe a non-terminal match event.
    fn on_event(&mut self, _event: &MatchEvent) {}

    /// Observe the final immutable result.
    fn on_result(&mut self, _result: &MatchResult) {}
}

/// Deterministic sample controller that safely stops every robot.
#[derive(Debug, Default)]
pub struct StopController;

impl Controller for StopController {
    fn on_reset(
        &mut self,
        _config: &MatchConfig,
        _initial: &MatchState,
    ) -> Result<(), ControllerError> {
        Ok(())
    }

    fn act(
        &mut self,
        _observation: &MatchState,
    ) -> Result<[RobotCommand; ROBOTS_PER_TEAM], ControllerError> {
        Ok([RobotCommand {
            mode: wire::ControlMode::WheelVelocity,
            first: 0.0,
            second: 0.0,
        }; ROBOTS_PER_TEAM])
    }
}
