//! Authoritative external-controller match orchestration.

mod artifact;
mod clock;
mod machine;
mod session;
mod transport;

pub use artifact::{ArtifactError, MatchArtifact, MatchMetadata, MatchOutcome};
pub use clock::{Clock, SystemClock};
pub use machine::{
    Advance, FallbackPolicy, MachineError, MatchMachine, MatchPhase, SlotActions, TickDecision,
};
pub use session::{
    ControllerIdentity, ControllerSession, LeaseAdjudication, NegotiatedCapabilities, SessionError,
    SessionRegistry, SessionState,
};
pub use transport::{IncomingMessage, RouterTransport, TransportError};
