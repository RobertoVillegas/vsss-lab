//! Versioned wire protocol shared by the match server and controller SDKs.

#[allow(
    clippy::all,
    clippy::nursery,
    clippy::pedantic,
    clippy::restriction,
    missing_docs,
    unsafe_code
)]
mod generated {
    include!("generated/vsss_match_v1_generated.rs");
}

/// Generated `FlatBuffers` v1 types.
pub use generated::vsss::protocol::v_1 as wire;

use flatbuffers::FlatBufferBuilder;
use thiserror::Error;

/// Current protocol version.
pub const PROTOCOL_VERSION: u32 = 1;
/// Maximum accepted wire message size.
pub const MAX_MESSAGE_BYTES: usize = 1_048_576;
/// Number of robots controlled by one VSSS team.
pub const ROBOTS_PER_TEAM: usize = 3;

/// Metadata common to every protocol message.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EnvelopeMeta {
    /// Stable 16-byte identifier for the match.
    pub match_id: [u8; 16],
    /// Controller slot assigned by the server.
    pub controller_slot: wire::ControllerSlot,
    /// Monotonically increasing sender sequence.
    pub sequence: u64,
    /// Authoritative server tick.
    pub server_tick: u64,
    /// Sender monotonic timestamp, for diagnostics.
    pub sent_monotonic_ns: u64,
    /// Server-issued action deadline.
    pub deadline_monotonic_ns: u64,
}

/// A controller command for one robot.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RobotCommand {
    /// Interpretation of the two command values.
    pub mode: wire::ControlMode,
    /// Left wheel or forward body velocity.
    pub first: f32,
    /// Right wheel or angular body velocity.
    pub second: f32,
}

/// Error returned before an untrusted message reaches match logic.
#[derive(Debug, Error, Eq, PartialEq)]
pub enum DecodeError {
    /// Message exceeds the negotiated safety limit.
    #[error("message is {actual} bytes; limit is {limit}")]
    PayloadTooLarge {
        /// Received bytes.
        actual: usize,
        /// Configured limit.
        limit: usize,
    },
    /// The `FlatBuffers` file identifier is absent or wrong.
    #[error("expected FlatBuffers identifier VSS1")]
    WrongIdentifier,
    /// `FlatBuffers` verification failed.
    #[error("invalid FlatBuffer: {0}")]
    InvalidBuffer(String),
    /// Sender uses an unsupported protocol version.
    #[error("unsupported protocol version {0}")]
    UnsupportedVersion(u32),
    /// Match IDs are always encoded as exactly 16 bytes.
    #[error("match ID must contain exactly 16 bytes, got {0}")]
    InvalidMatchId(usize),
    /// Envelope does not identify a payload variant.
    #[error("envelope payload is missing")]
    MissingPayload,
    /// Team actions must contain exactly three robot commands.
    #[error("action must contain exactly {ROBOTS_PER_TEAM} robots, got {0}")]
    InvalidActionCount(usize),
    /// NaN and infinity cannot enter the authoritative simulation.
    #[error("robot {robot} contains a non-finite command")]
    NonFiniteAction {
        /// Zero-based robot index.
        robot: usize,
    },
}

/// Verified view of an envelope backed by the caller's input bytes.
#[derive(Clone, Copy, Debug)]
pub struct VerifiedEnvelope<'a>(wire::Envelope<'a>);

impl<'a> VerifiedEnvelope<'a> {
    /// Access the verified generated representation.
    #[must_use]
    pub const fn wire(self) -> wire::Envelope<'a> {
        self.0
    }

    /// Copy common metadata into a representation independent of `FlatBuffers`.
    #[must_use]
    pub fn meta(self) -> EnvelopeMeta {
        let envelope = self.0;
        let mut match_id = [0; 16];
        match_id.copy_from_slice(envelope.match_id().bytes());
        EnvelopeMeta {
            match_id,
            controller_slot: envelope.controller_slot(),
            sequence: envelope.sequence(),
            server_tick: envelope.server_tick(),
            sent_monotonic_ns: envelope.sent_monotonic_ns(),
            deadline_monotonic_ns: envelope.deadline_monotonic_ns(),
        }
    }
}

/// Verify structural and protocol invariants on an untrusted message.
///
/// # Errors
///
/// Returns a typed error when the buffer is oversized, malformed, belongs to
/// another protocol, or violates a semantic invariant enforced at the boundary.
pub fn decode_envelope(bytes: &[u8]) -> Result<VerifiedEnvelope<'_>, DecodeError> {
    if bytes.len() > MAX_MESSAGE_BYTES {
        return Err(DecodeError::PayloadTooLarge {
            actual: bytes.len(),
            limit: MAX_MESSAGE_BYTES,
        });
    }
    if !wire::envelope_buffer_has_identifier(bytes) {
        return Err(DecodeError::WrongIdentifier);
    }
    let envelope = wire::root_as_envelope(bytes)
        .map_err(|error| DecodeError::InvalidBuffer(error.to_string()))?;
    if envelope.protocol_version() != PROTOCOL_VERSION {
        return Err(DecodeError::UnsupportedVersion(envelope.protocol_version()));
    }
    if envelope.match_id().len() != 16 {
        return Err(DecodeError::InvalidMatchId(envelope.match_id().len()));
    }
    if envelope.payload_type() == wire::Payload::NONE {
        return Err(DecodeError::MissingPayload);
    }
    if let Some(action) = envelope.payload_as_action() {
        let robots = action.robots();
        if robots.len() != ROBOTS_PER_TEAM {
            return Err(DecodeError::InvalidActionCount(robots.len()));
        }
        for index in 0..robots.len() {
            let robot = robots.get(index);
            if !robot.first().is_finite() || !robot.second().is_finite() {
                return Err(DecodeError::NonFiniteAction { robot: index });
            }
        }
    }
    Ok(VerifiedEnvelope(envelope))
}

/// Encode the initial controller handshake.
#[must_use]
pub fn encode_hello(
    meta: EnvelopeMeta,
    controller_name: &str,
    sdk_name: &str,
    sdk_version: &str,
) -> Vec<u8> {
    let mut builder = FlatBufferBuilder::new();
    let controller_name = builder.create_string(controller_name);
    let sdk_name = builder.create_string(sdk_name);
    let sdk_version = builder.create_string(sdk_version);
    let hello = wire::Hello::create(
        &mut builder,
        &wire::HelloArgs {
            controller_name: Some(controller_name),
            sdk_name: Some(sdk_name),
            sdk_version: Some(sdk_version),
            min_protocol_version: PROTOCOL_VERSION,
            max_protocol_version: PROTOCOL_VERSION,
        },
    );
    finish_envelope(builder, meta, wire::Payload::Hello, hello.as_union_value())
}

/// Encode one complete three-robot team action.
#[must_use]
pub fn encode_action(meta: EnvelopeMeta, commands: [RobotCommand; ROBOTS_PER_TEAM]) -> Vec<u8> {
    let mut builder = FlatBufferBuilder::new();
    let robots =
        commands.map(|command| wire::RobotAction::new(command.mode, command.first, command.second));
    let robots = builder.create_vector(&robots);
    let action = wire::Action::create(
        &mut builder,
        &wire::ActionArgs {
            robots: Some(robots),
        },
    );
    finish_envelope(
        builder,
        meta,
        wire::Payload::Action,
        action.as_union_value(),
    )
}

/// Encode negotiated server capabilities.
#[must_use]
pub fn encode_capabilities(
    meta: EnvelopeMeta,
    assigned_slot: wire::ControllerSlot,
    control_period_ns: u64,
    max_message_bytes: u32,
) -> Vec<u8> {
    let mut builder = FlatBufferBuilder::new();
    let capabilities = wire::Capabilities::create(
        &mut builder,
        &wire::CapabilitiesArgs {
            accepted_protocol_version: PROTOCOL_VERSION,
            assigned_slot,
            control_period_ns,
            max_message_bytes,
        },
    );
    finish_envelope(
        builder,
        meta,
        wire::Payload::Capabilities,
        capabilities.as_union_value(),
    )
}

/// Encode canonical reset configuration and initial state JSON.
#[must_use]
pub fn encode_reset(
    meta: EnvelopeMeta,
    config_json: &str,
    config_sha256: &[u8; 32],
    initial_state_json: &str,
    seed: u64,
) -> Vec<u8> {
    let mut builder = FlatBufferBuilder::new();
    let canonical_json = builder.create_string(config_json);
    let sha256 = builder.create_vector(config_sha256);
    let config = wire::MatchConfigJson::create(
        &mut builder,
        &wire::MatchConfigJsonArgs {
            canonical_json: Some(canonical_json),
            sha256: Some(sha256),
        },
    );
    let initial_state_json = builder.create_string(initial_state_json);
    let reset = wire::Reset::create(
        &mut builder,
        &wire::ResetArgs {
            config: Some(config),
            initial_state_json: Some(initial_state_json),
            seed,
        },
    );
    finish_envelope(builder, meta, wire::Payload::Reset, reset.as_union_value())
}

/// Encode one canonical state observation.
#[must_use]
pub fn encode_observation(
    meta: EnvelopeMeta,
    canonical_state_json: &str,
    state_sha256: &[u8; 32],
) -> Vec<u8> {
    let mut builder = FlatBufferBuilder::new();
    let canonical_state_json = builder.create_string(canonical_state_json);
    let state_sha256 = builder.create_vector(state_sha256);
    let observation = wire::Observation::create(
        &mut builder,
        &wire::ObservationArgs {
            canonical_state_json: Some(canonical_state_json),
            state_sha256: Some(state_sha256),
        },
    );
    finish_envelope(
        builder,
        meta,
        wire::Payload::Observation,
        observation.as_union_value(),
    )
}

/// Encode an immutable terminal match result.
#[must_use]
pub fn encode_match_result(
    meta: EnvelopeMeta,
    score_blue: u16,
    score_yellow: u16,
    replay_sha256: &[u8; 32],
    reason: &str,
) -> Vec<u8> {
    let mut builder = FlatBufferBuilder::new();
    let replay_sha256 = builder.create_vector(replay_sha256);
    let reason = builder.create_string(reason);
    let result = wire::MatchResult::create(
        &mut builder,
        &wire::MatchResultArgs {
            score_blue,
            score_yellow,
            replay_sha256: Some(replay_sha256),
            reason: Some(reason),
        },
    );
    finish_envelope(
        builder,
        meta,
        wire::Payload::MatchResult,
        result.as_union_value(),
    )
}

fn finish_envelope(
    mut builder: FlatBufferBuilder<'static>,
    meta: EnvelopeMeta,
    payload_type: wire::Payload,
    payload: flatbuffers::WIPOffset<flatbuffers::UnionWIPOffset>,
) -> Vec<u8> {
    let match_id = builder.create_vector(&meta.match_id);
    let envelope = wire::Envelope::create(
        &mut builder,
        &wire::EnvelopeArgs {
            protocol_version: PROTOCOL_VERSION,
            match_id: Some(match_id),
            controller_slot: meta.controller_slot,
            sequence: meta.sequence,
            server_tick: meta.server_tick,
            sent_monotonic_ns: meta.sent_monotonic_ns,
            deadline_monotonic_ns: meta.deadline_monotonic_ns,
            payload_type,
            payload: Some(payload),
        },
    );
    wire::finish_envelope_buffer(&mut builder, envelope);
    builder.finished_data().to_vec()
}
