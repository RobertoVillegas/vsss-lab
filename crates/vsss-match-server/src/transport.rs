//! Bounded loopback ROUTER transport.

use bytes::Bytes;
use thiserror::Error;
use vsss_protocol::{MAX_MESSAGE_BYTES, decode_envelope};
use zeromq::{Endpoint, RouterSocket, Socket, SocketRecv, SocketSend, ZmqError, ZmqMessage};

/// Owned identity and verified protocol bytes received from a DEALER.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IncomingMessage {
    /// Opaque `ZeroMQ` routing identity.
    pub routing_id: Vec<u8>,
    /// Structurally and semantically verified `FlatBuffer` bytes.
    pub payload: Vec<u8>,
}

/// Network framing or protocol-boundary failure.
#[derive(Debug, Error)]
pub enum TransportError {
    /// External endpoints are intentionally forbidden in M8.
    #[error("M8 transport must bind to tcp://127.0.0.1")]
    NonLoopbackEndpoint,
    /// `ZeroMQ` socket operation failed.
    #[error("ZeroMQ transport failed: {0}")]
    ZeroMq(#[from] ZmqError),
    /// ROUTER input was not exactly identity plus one payload frame.
    #[error("ROUTER message must contain exactly identity and payload frames")]
    InvalidFraming,
    /// Payload exceeded the configured or protocol hard limit.
    #[error("payload exceeds the transport limit")]
    PayloadTooLarge,
    /// Payload failed protocol verification.
    #[error("invalid protocol payload: {0}")]
    InvalidProtocol(String),
    /// Empty routing identities cannot be addressed safely.
    #[error("routing identity must not be empty")]
    EmptyIdentity,
    /// Message construction unexpectedly failed.
    #[error("failed to construct ZeroMQ message")]
    MessageConstruction,
}

/// ROUTER socket constrained to loopback and one bounded payload frame.
pub struct RouterTransport {
    socket: RouterSocket,
    endpoint: Endpoint,
    max_message_bytes: usize,
}

impl RouterTransport {
    /// Bind a ROUTER socket to a loopback TCP endpoint.
    ///
    /// # Errors
    ///
    /// Rejects non-loopback endpoints and propagates socket binding failures.
    pub async fn bind(endpoint: &str, max_message_bytes: usize) -> Result<Self, TransportError> {
        if !endpoint.starts_with("tcp://127.0.0.1:") {
            return Err(TransportError::NonLoopbackEndpoint);
        }
        Self::bind_allowed(endpoint, max_message_bytes).await
    }

    /// Bind inside a container-only private network.
    ///
    /// # Errors
    ///
    /// Only an all-interfaces TCP bind is accepted; publication is controlled
    /// by the container network, which must not expose a host port.
    pub async fn bind_private(
        endpoint: &str,
        max_message_bytes: usize,
    ) -> Result<Self, TransportError> {
        if !endpoint.starts_with("tcp://0.0.0.0:") {
            return Err(TransportError::NonLoopbackEndpoint);
        }
        Self::bind_allowed(endpoint, max_message_bytes).await
    }

    async fn bind_allowed(
        endpoint: &str,
        max_message_bytes: usize,
    ) -> Result<Self, TransportError> {
        let mut socket = RouterSocket::new();
        let endpoint = socket.bind(endpoint).await?;
        Ok(Self {
            socket,
            endpoint,
            max_message_bytes: max_message_bytes.min(MAX_MESSAGE_BYTES),
        })
    }

    /// Actual endpoint, including an operating-system-assigned port.
    #[must_use]
    pub fn endpoint(&self) -> String {
        self.endpoint.to_string()
    }

    /// Receive and verify one identity-framed payload.
    ///
    /// # Errors
    ///
    /// Rejects malformed framing, oversized payloads, and invalid protocol data.
    pub async fn receive(&mut self) -> Result<IncomingMessage, TransportError> {
        let frames = self.socket.recv().await?.into_vec();
        if frames.len() != 2 || frames[0].is_empty() {
            return Err(if frames.first().is_some_and(Bytes::is_empty) {
                TransportError::EmptyIdentity
            } else {
                TransportError::InvalidFraming
            });
        }
        if frames[1].len() > self.max_message_bytes {
            return Err(TransportError::PayloadTooLarge);
        }
        decode_envelope(&frames[1])
            .map_err(|error| TransportError::InvalidProtocol(error.to_string()))?;
        Ok(IncomingMessage {
            routing_id: frames[0].to_vec(),
            payload: frames[1].to_vec(),
        })
    }

    /// Send one protocol payload to a previously observed identity.
    ///
    /// # Errors
    ///
    /// Rejects empty identities, oversized/invalid payloads, and socket errors.
    pub async fn send(
        &mut self,
        routing_id: &[u8],
        payload: Vec<u8>,
    ) -> Result<(), TransportError> {
        if routing_id.is_empty() {
            return Err(TransportError::EmptyIdentity);
        }
        if payload.len() > self.max_message_bytes {
            return Err(TransportError::PayloadTooLarge);
        }
        decode_envelope(&payload)
            .map_err(|error| TransportError::InvalidProtocol(error.to_string()))?;
        let message = ZmqMessage::try_from(vec![
            Bytes::copy_from_slice(routing_id),
            Bytes::from(payload),
        ])
        .map_err(|_| TransportError::MessageConstruction)?;
        self.socket.send(message).await?;
        Ok(())
    }
}
