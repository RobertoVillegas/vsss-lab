//! Loopback ROUTER/DEALER transport contract tests.

use vsss_match_server::RouterTransport;
use vsss_protocol::{EnvelopeMeta, encode_hello, wire};
use zeromq::{DealerSocket, Socket, SocketRecv, SocketSend, ZmqMessage};

fn hello(sequence: u64) -> Vec<u8> {
    encode_hello(
        EnvelopeMeta {
            match_id: *b"transport-test01",
            controller_slot: wire::ControllerSlot::Unassigned,
            sequence,
            server_tick: 0,
            sent_monotonic_ns: 1,
            deadline_monotonic_ns: 0,
        },
        "test-controller",
        "rust-test",
        "0.0.0",
    )
}

#[tokio::test]
async fn loopback_transport_verifies_and_routes_payloads() {
    let mut router = RouterTransport::bind("tcp://127.0.0.1:0", 4096)
        .await
        .expect("bind router");
    let mut dealer = DealerSocket::new();
    dealer
        .connect(&router.endpoint())
        .await
        .expect("connect dealer");

    dealer
        .send(ZmqMessage::from(hello(1)))
        .await
        .expect("send hello");
    let incoming = router.receive().await.expect("receive verified hello");
    assert!(!incoming.routing_id.is_empty());

    router
        .send(&incoming.routing_id, hello(2))
        .await
        .expect("route reply");
    let reply: Vec<u8> = dealer
        .recv()
        .await
        .expect("receive reply")
        .try_into()
        .expect("single frame");
    assert_eq!(
        vsss_protocol::decode_envelope(&reply)
            .expect("valid reply")
            .meta()
            .sequence,
        2
    );
}

#[tokio::test]
async fn rejects_non_loopback_bind() {
    assert!(
        RouterTransport::bind("tcp://0.0.0.0:0", 4096)
            .await
            .is_err()
    );
}
