//! Controller slot negotiation and sender ordering.

use std::collections::BTreeMap;

use thiserror::Error;
use vsss_protocol::{MAX_MESSAGE_BYTES, PROTOCOL_VERSION, VerifiedEnvelope, wire::ControllerSlot};

/// Capabilities fixed by a successful handshake.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NegotiatedCapabilities {
    /// Selected wire protocol version.
    pub protocol_version: u32,
    /// Server control interval.
    pub control_period_ns: u64,
    /// Largest accepted wire message.
    pub max_message_bytes: usize,
}

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
    /// Agreed protocol and timing limits.
    pub capabilities: NegotiatedCapabilities,
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
    /// First message was not an unassigned Hello envelope.
    #[error("controller handshake must be an unassigned Hello")]
    InvalidHandshake,
    /// Controller and server have no common protocol version.
    #[error("controller has no compatible protocol version")]
    CapabilityMismatch,
}

/// Two-slot controller registry independent of transport implementation.
#[derive(Debug, Default)]
pub struct SessionRegistry {
    sessions: BTreeMap<Vec<u8>, ControllerSession>,
}

impl SessionRegistry {
    /// Negotiate the first available slot from a verified Hello envelope.
    ///
    /// # Errors
    ///
    /// Rejects non-Hello input, incompatible versions, or a full registry.
    pub fn negotiate(
        &mut self,
        identity: ControllerIdentity,
        envelope: VerifiedEnvelope<'_>,
        now_ns: u64,
        control_period_ns: u64,
        max_message_bytes: usize,
    ) -> Result<&ControllerSession, SessionError> {
        let wire = envelope.wire();
        if wire.controller_slot() != ControllerSlot::Unassigned {
            return Err(SessionError::InvalidHandshake);
        }
        let hello = wire
            .payload_as_hello()
            .ok_or(SessionError::InvalidHandshake)?;
        if hello.min_protocol_version() > PROTOCOL_VERSION
            || hello.max_protocol_version() < PROTOCOL_VERSION
        {
            return Err(SessionError::CapabilityMismatch);
        }
        let slot = [ControllerSlot::Blue, ControllerSlot::Yellow]
            .into_iter()
            .find(|candidate| {
                !self.sessions.values().any(|session| {
                    session.slot == *candidate && session.state == SessionState::Active
                })
            })
            .ok_or(SessionError::SlotOccupied)?;
        self.register_with_capabilities(
            identity,
            slot,
            now_ns,
            NegotiatedCapabilities {
                protocol_version: PROTOCOL_VERSION,
                control_period_ns,
                max_message_bytes: max_message_bytes.min(MAX_MESSAGE_BYTES),
            },
        )
    }

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
        self.register_with_capabilities(
            identity,
            slot,
            now_ns,
            NegotiatedCapabilities {
                protocol_version: PROTOCOL_VERSION,
                control_period_ns: 0,
                max_message_bytes: MAX_MESSAGE_BYTES,
            },
        )
    }

    fn register_with_capabilities(
        &mut self,
        identity: ControllerIdentity,
        slot: ControllerSlot,
        now_ns: u64,
        capabilities: NegotiatedCapabilities,
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
                capabilities,
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
