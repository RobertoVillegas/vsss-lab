//! Controller slot negotiation and sender ordering.

use std::collections::BTreeMap;

use thiserror::Error;
use vsss_protocol::wire::ControllerSlot;

/// Stable transport identity and human-readable controller name.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub struct ControllerIdentity {
    /// Opaque ROUTER identity bytes.
    pub routing_id: Vec<u8>,
    /// Name supplied during the protocol handshake.
    pub name: String,
}

/// Lifecycle of one external controller.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionState {
    /// Handshake accepted and slot assigned.
    Active,
    /// Heartbeat lease expired.
    Disconnected,
    /// Match completed or controller forfeited.
    Closed,
}

/// Server-owned state for one controller connection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ControllerSession {
    /// Controller identity.
    pub identity: ControllerIdentity,
    /// Ephemeral match slot.
    pub slot: ControllerSlot,
    /// Last accepted sender sequence.
    pub last_sequence: u64,
    /// Last received message time.
    pub last_seen_ns: u64,
    /// Current lifecycle state.
    pub state: SessionState,
}

/// Session negotiation or ordering failure.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum SessionError {
    /// Only Blue and Yellow are assignable.
    #[error("controller requested an invalid slot")]
    InvalidSlot,
    /// A controller already owns the slot.
    #[error("slot is already assigned")]
    SlotOccupied,
    /// Routing identity is already registered.
    #[error("controller identity is already registered")]
    DuplicateIdentity,
    /// Routing identity is unknown.
    #[error("unknown controller identity")]
    UnknownIdentity,
    /// Message sequence did not strictly increase.
    #[error("sequence {received} is not newer than {last}")]
    StaleSequence {
        /// Received sequence.
        received: u64,
        /// Last accepted sequence.
        last: u64,
    },
    /// Claimed slot differs from the negotiated slot.
    #[error("message claims the wrong controller slot")]
    WrongSlot,
    /// Session no longer accepts messages.
    #[error("controller session is not active")]
    Inactive,
}

/// Two-slot controller registry independent of transport implementation.
#[derive(Debug, Default)]
pub struct SessionRegistry {
    sessions: BTreeMap<Vec<u8>, ControllerSession>,
}

impl SessionRegistry {
    /// Register a controller in one unoccupied team slot.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid/occupied slots or duplicate identities.
    pub fn register(
        &mut self,
        identity: ControllerIdentity,
        slot: ControllerSlot,
        now_ns: u64,
    ) -> Result<&ControllerSession, SessionError> {
        if !matches!(slot, ControllerSlot::Blue | ControllerSlot::Yellow) {
            return Err(SessionError::InvalidSlot);
        }
        if self.sessions.contains_key(&identity.routing_id) {
            return Err(SessionError::DuplicateIdentity);
        }
        if self
            .sessions
            .values()
            .any(|session| session.slot == slot && session.state == SessionState::Active)
        {
            return Err(SessionError::SlotOccupied);
        }
        let key = identity.routing_id.clone();
        self.sessions.insert(
            key.clone(),
            ControllerSession {
                identity,
                slot,
                last_sequence: 0,
                last_seen_ns: now_ns,
                state: SessionState::Active,
            },
        );
        Ok(&self.sessions[&key])
    }

    /// Validate routing identity, slot, state, and monotonic sequence.
    ///
    /// # Errors
    ///
    /// Returns a typed rejection without modifying the session on failure.
    pub fn accept(
        &mut self,
        routing_id: &[u8],
        slot: ControllerSlot,
        sequence: u64,
        now_ns: u64,
    ) -> Result<&ControllerSession, SessionError> {
        let session = self
            .sessions
            .get_mut(routing_id)
            .ok_or(SessionError::UnknownIdentity)?;
        if session.state != SessionState::Active {
            return Err(SessionError::Inactive);
        }
        if session.slot != slot {
            return Err(SessionError::WrongSlot);
        }
        if sequence <= session.last_sequence {
            return Err(SessionError::StaleSequence {
                received: sequence,
                last: session.last_sequence,
            });
        }
        session.last_sequence = sequence;
        session.last_seen_ns = now_ns;
        Ok(session)
    }

    /// Mark leases older than `lease_ns` as disconnected.
    pub fn expire(&mut self, now_ns: u64, lease_ns: u64) -> Vec<ControllerSlot> {
        let mut expired = Vec::new();
        for session in self.sessions.values_mut() {
            if session.state == SessionState::Active
                && now_ns.saturating_sub(session.last_seen_ns) > lease_ns
            {
                session.state = SessionState::Disconnected;
                expired.push(session.slot);
            }
        }
        expired.sort_by_key(|slot| slot.0);
        expired
    }

    /// Return a session by opaque transport identity.
    #[must_use]
    pub fn get(&self, routing_id: &[u8]) -> Option<&ControllerSession> {
        self.sessions.get(routing_id)
    }
}
