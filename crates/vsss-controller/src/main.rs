//! Executable safe Rust sample controller.

use std::{env, time::Instant};

use vsss_protocol::{
    EnvelopeMeta, RobotCommand, decode_envelope, encode_action, encode_hello, wire,
};
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

type AnyError = Box<dyn std::error::Error>;

#[tokio::main]
async fn main() -> Result<(), AnyError> {
    let endpoint = env::args().nth(1).ok_or("missing server endpoint")?;
    let mut socket = DealerSocket::new();
    socket.connect(&endpoint).await?;
    let origin = Instant::now();
    let match_id = *b"external-match01";
    let mut slot = wire::ControllerSlot::Unassigned;
    let mut sequence = 1;
    socket
        .send(ZmqMessage::from(encode_hello(
            meta(match_id, slot, sequence, 0, 0, &origin),
            "rust-stop-controller",
            "vsss-rust",
            env!("CARGO_PKG_VERSION"),
        )))
        .await?;
    loop {
        let bytes: Vec<u8> = socket.recv().await?.try_into()?;
        let envelope = decode_envelope(&bytes)?;
        match envelope.wire().payload_type() {
            wire::Payload::Capabilities => {
                slot = envelope
                    .wire()
                    .payload_as_capabilities()
                    .ok_or("invalid capabilities")?
                    .assigned_slot();
            }
            wire::Payload::Reset => {}
            wire::Payload::Observation => {
                sequence += 1;
                let stopped = [RobotCommand {
                    mode: wire::ControlMode::WheelVelocity,
                    first: 0.0,
                    second: 0.0,
                }; 3];
                socket
                    .send(ZmqMessage::from(encode_action(
                        meta(
                            match_id,
                            slot,
                            sequence,
                            envelope.meta().server_tick,
                            envelope.meta().deadline_monotonic_ns,
                            &origin,
                        ),
                        stopped,
                    )))
                    .await?;
            }
            wire::Payload::MatchResult => break,
            _ => return Err("unsupported server payload".into()),
        }
    }
    Ok(())
}

fn meta(
    match_id: [u8; 16],
    slot: wire::ControllerSlot,
    sequence: u64,
    server_tick: u64,
    deadline_monotonic_ns: u64,
    origin: &Instant,
) -> EnvelopeMeta {
    EnvelopeMeta {
        match_id,
        controller_slot: slot,
        sequence,
        server_tick,
        sent_monotonic_ns: u64::try_from(origin.elapsed().as_nanos()).unwrap_or(u64::MAX),
        deadline_monotonic_ns,
    }
}
