//! Pure fixed-tick authoritative match state machine.

use std::time::Duration;

use thiserror::Error;
use vsss_physics_api::{PhysicsBackend, PhysicsError};
use vsss_protocol::wire::ControllerSlot;
use vsss_spec::{AngularVelocity, MatchConfig, MatchState, RobotAction, Validate, ValidationError};

use crate::Clock;

/// Exactly one team's three commands.
pub type SlotActions = [RobotAction; 3];

/// Deterministic action used when a controller misses its deadline.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FallbackPolicy {
    /// Reuse the last accepted safe action.
    RepeatLast,
    /// Stop all robots controlled by that slot.
    Zero,
}

/// Match lifecycle independent of transport.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MatchPhase {
    /// Constructed but not reset.
    Ready,
    /// Accepting actions and advancing fixed ticks.
    Running,
    /// Simulated match duration reached.
    Finished,
}

/// Per-slot adjudication at a control boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TickDecision {
    /// Submitted on-time action was applied.
    Accepted,
    /// No on-time action arrived; configured fallback was applied.
    DeadlineFallback,
}

/// State emitted after one control boundary.
#[derive(Clone, Debug, PartialEq)]
pub struct Advance {
    /// Resulting canonical physics state.
    pub state: MatchState,
    /// Blue controller adjudication.
    pub blue: TickDecision,
    /// Yellow controller adjudication.
    pub yellow: TickDecision,
    /// Deadline for the next action boundary.
    pub next_deadline_ns: u64,
    /// Whether this boundary completed the match.
    pub finished: bool,
}

/// Match construction, submission, or advancement failure.
#[derive(Debug, Error)]
pub enum MachineError {
    /// Canonical match configuration failed validation.
    #[error("invalid match configuration: {0}")]
    InvalidConfig(ValidationError),
    /// Control period is not an integer multiple of the physics timestep.
    #[error("control period must be an integer multiple of timestep")]
    NonIntegralControlPeriod,
    /// Timing cannot be represented safely in nanoseconds or step counts.
    #[error("match timing is outside supported range")]
    TimingOutOfRange,
    /// Backend failed to reset or advance.
    #[error("physics backend failed: {0}")]
    Physics(#[from] PhysicsError),
    /// Operation requires a running match.
    #[error("match is not running")]
    NotRunning,
    /// Only Blue and Yellow can submit actions.
    #[error("invalid controller slot")]
    InvalidSlot,
    /// Action targets a different authoritative tick.
    #[error("action tick {received} does not match expected tick {expected}")]
    WrongTick {
        /// Received target tick.
        received: u64,
        /// Expected target tick.
        expected: u64,
    },
    /// Action arrived after the server-issued deadline.
    #[error("action arrived after its deadline")]
    DeadlineExceeded,
    /// Slot already submitted an action for this boundary.
    #[error("slot already submitted an action for this tick")]
    DuplicateAction,
    /// Action violates canonical finite/range limits.
    #[error("invalid action: {0}")]
    InvalidAction(ValidationError),
}

/// Authoritative match logic parameterized by backend and monotonic clock.
pub struct MatchMachine<B, C> {
    backend: B,
    clock: C,
    config: MatchConfig,
    fallback: FallbackPolicy,
    phase: MatchPhase,
    state: Option<MatchState>,
    physics_steps_per_control: u32,
    control_period_ns: u64,
    deadline_ns: u64,
    pending: [Option<SlotActions>; 2],
    last_safe: [SlotActions; 2],
}

impl<B: PhysicsBackend, C: Clock> MatchMachine<B, C> {
    /// Construct a match without touching the backend.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid timing or canonical configuration.
    pub fn new(
        backend: B,
        clock: C,
        config: MatchConfig,
        fallback: FallbackPolicy,
    ) -> Result<Self, MachineError> {
        config.validate().map_err(MachineError::InvalidConfig)?;
        let control_period_ns = u64::try_from(
            Duration::try_from_secs_f32(config.control_period.get())
                .map_err(|_| MachineError::TimingOutOfRange)?
                .as_nanos(),
        )
        .map_err(|_| MachineError::TimingOutOfRange)?;
        let timestep_ns = u64::try_from(
            Duration::try_from_secs_f32(config.timestep.get())
                .map_err(|_| MachineError::TimingOutOfRange)?
                .as_nanos(),
        )
        .map_err(|_| MachineError::TimingOutOfRange)?;
        if timestep_ns == 0 || !control_period_ns.is_multiple_of(timestep_ns) {
            return Err(MachineError::NonIntegralControlPeriod);
        }
        let physics_steps_per_control = u32::try_from(control_period_ns / timestep_ns)
            .map_err(|_| MachineError::TimingOutOfRange)?;
        let stopped = [RobotAction::wheel_velocity(AngularVelocity(0.0), AngularVelocity(0.0)); 3];
        Ok(Self {
            backend,
            clock,
            config,
            fallback,
            phase: MatchPhase::Ready,
            state: None,
            physics_steps_per_control,
            control_period_ns,
            deadline_ns: 0,
            pending: [None, None],
            last_safe: [stopped; 2],
        })
    }

    /// Reset the backend and begin accepting tick-zero actions.
    ///
    /// # Errors
    ///
    /// Propagates a backend reset failure.
    pub fn start(&mut self) -> Result<&MatchState, MachineError> {
        let state = self.backend.reset()?;
        self.deadline_ns = self.clock.now_ns().saturating_add(self.control_period_ns);
        self.pending = [None, None];
        self.last_safe = [Self::stopped(); 2];
        self.phase = MatchPhase::Running;
        self.state = Some(state);
        self.state.as_ref().ok_or(MachineError::NotRunning)
    }

    /// Submit one slot's complete action for the next control boundary.
    ///
    /// # Errors
    ///
    /// Rejects invalid slots, timing, duplicates, and unsafe values.
    pub fn submit(
        &mut self,
        slot: ControllerSlot,
        target_tick: u64,
        actions: SlotActions,
    ) -> Result<(), MachineError> {
        if self.phase != MatchPhase::Running {
            return Err(MachineError::NotRunning);
        }
        let index = Self::slot_index(slot)?;
        let expected = self.state.as_ref().ok_or(MachineError::NotRunning)?.tick;
        if target_tick != expected {
            return Err(MachineError::WrongTick {
                received: target_tick,
                expected,
            });
        }
        if self.clock.now_ns() > self.deadline_ns {
            return Err(MachineError::DeadlineExceeded);
        }
        if self.pending[index].is_some() {
            return Err(MachineError::DuplicateAction);
        }
        for action in actions {
            action.validate().map_err(MachineError::InvalidAction)?;
            if action.mode == vsss_spec::ControlMode::WheelVelocity
                && (action.left.abs() > self.config.max_wheel_speed.get()
                    || action.right.abs() > self.config.max_wheel_speed.get())
            {
                return Err(MachineError::InvalidAction(ValidationError::new(
                    "action",
                    "wheel velocity exceeds configured limit",
                )));
            }
        }
        self.pending[index] = Some(actions);
        Ok(())
    }

    /// Advance one control boundary once its deadline is reached.
    ///
    /// Returns `Ok(None)` before the deadline.
    ///
    /// # Errors
    ///
    /// Returns an error when not running or when the backend fails.
    pub fn advance_if_due(&mut self) -> Result<Option<Advance>, MachineError> {
        if self.phase != MatchPhase::Running {
            return Err(MachineError::NotRunning);
        }
        if self.clock.now_ns() < self.deadline_ns {
            return Ok(None);
        }
        let (blue, blue_decision) = self.resolve_slot(0);
        let (yellow, yellow_decision) = self.resolve_slot(1);
        let actions = [blue[0], blue[1], blue[2], yellow[0], yellow[1], yellow[2]];
        let mut state = self.state.take().ok_or(MachineError::NotRunning)?;
        for _ in 0..self.physics_steps_per_control {
            state = self.backend.step(&actions)?;
        }
        let finished = state.simulation_time.get() >= self.config.match_duration.get();
        self.phase = if finished {
            MatchPhase::Finished
        } else {
            MatchPhase::Running
        };
        self.deadline_ns = self.deadline_ns.saturating_add(self.control_period_ns);
        self.pending = [None, None];
        self.state = Some(state.clone());
        Ok(Some(Advance {
            state,
            blue: blue_decision,
            yellow: yellow_decision,
            next_deadline_ns: self.deadline_ns,
            finished,
        }))
    }

    /// Current lifecycle phase.
    #[must_use]
    pub const fn phase(&self) -> MatchPhase {
        self.phase
    }

    /// Current canonical state, once started.
    #[must_use]
    pub fn state(&self) -> Option<&MatchState> {
        self.state.as_ref()
    }

    /// Current action deadline.
    #[must_use]
    pub const fn deadline_ns(&self) -> u64 {
        self.deadline_ns
    }

    fn resolve_slot(&mut self, index: usize) -> (SlotActions, TickDecision) {
        if let Some(actions) = self.pending[index] {
            self.last_safe[index] = actions;
            (actions, TickDecision::Accepted)
        } else {
            let actions = match self.fallback {
                FallbackPolicy::RepeatLast => self.last_safe[index],
                FallbackPolicy::Zero => Self::stopped(),
            };
            (actions, TickDecision::DeadlineFallback)
        }
    }

    fn stopped() -> SlotActions {
        [RobotAction::wheel_velocity(AngularVelocity(0.0), AngularVelocity(0.0)); 3]
    }

    fn slot_index(slot: ControllerSlot) -> Result<usize, MachineError> {
        match slot {
            ControllerSlot::Blue => Ok(0),
            ControllerSlot::Yellow => Ok(1),
            _ => Err(MachineError::InvalidSlot),
        }
    }
}
