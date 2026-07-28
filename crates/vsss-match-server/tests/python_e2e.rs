//! Rust ROUTER and Python DEALER interoperability smoke.

use std::{path::PathBuf, process::Command, time::Duration};

use vsss_match_server::RouterTransport;

#[tokio::test]
async fn rust_server_exchanges_verified_message_with_python_sdk() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let python = root.join(".venv/bin/python");
    if !python.exists() {
        eprintln!("skipping: uv environment has not been bootstrapped");
        return;
    }
    let mut router = RouterTransport::bind("tcp://127.0.0.1:0", 4096)
        .await
        .expect("bind");
    let mut child = Command::new(python)
        .args([
            "-m",
            "vsss_controller.sample",
            "--endpoint",
            &router.endpoint(),
            "--exchange-only",
        ])
        .env("PYTHONPATH", root.join("python"))
        .spawn()
        .expect("spawn Python controller");

    let incoming = tokio::time::timeout(Duration::from_secs(5), router.receive())
        .await
        .expect("controller send timeout")
        .expect("receive Python hello");
    router
        .send(&incoming.routing_id, incoming.payload)
        .await
        .expect("send verified reply");

    let status = child.wait().expect("wait for Python controller");
    assert!(status.success());
}
