//! Authoritative external-controller match orchestration.

mod clock;
mod machine;
mod session;
mod transport;

pub use clock::{Clock, SystemClock};
pub use machine::{
    Advance, FallbackPolicy, MachineError, MatchMachine, MatchPhase, SlotActions, TickDecision,
};
pub use session::{
    ControllerIdentity, ControllerSession, NegotiatedCapabilities, SessionError, SessionRegistry,
    SessionState,
};
pub use transport::{IncomingMessage, RouterTransport, TransportError};
