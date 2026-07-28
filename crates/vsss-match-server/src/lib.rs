//! Authoritative external-controller match orchestration.

mod clock;
mod machine;
mod session;

pub use clock::{Clock, SystemClock};
pub use machine::{
    Advance, FallbackPolicy, MachineError, MatchMachine, MatchPhase, SlotActions, TickDecision,
};
pub use session::{
    ControllerIdentity, ControllerSession, SessionError, SessionRegistry, SessionState,
};
