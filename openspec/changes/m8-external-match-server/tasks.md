## 1. Protocol foundation

- [x] 1.1 Research current VSSS rules, Julio thesis/marker assets, FlatBuffers, and ZeroMQ
- [x] 1.2 Record logical/team/visual identity boundary in ADR-0010
- [x] 1.3 Define M8 OpenSpec scope, contracts, non-goals, and validation
- [ ] 1.4 Add pinned FlatBuffers compiler/runtime and native ZeroMQ dependencies
- [ ] 1.5 Define v1 schema, generated Rust/Python bindings, and conformity check
- [ ] 1.6 Add golden valid/invalid/cross-version protocol fixtures

## 2. Authoritative server

- [ ] 2.1 Implement pure fixed-tick match state machine with injected clock
- [ ] 2.2 Implement slot negotiation, capabilities, sequence, and message validation
- [ ] 2.3 Implement deadline fallback, heartbeat lease, disconnect, and forfeit policy
- [ ] 2.4 Integrate canonical backend, replay, events, result, and checksums
- [ ] 2.5 Add loopback ROUTER transport with bounded queues and payload limits

## 3. Controller SDKs and competition

- [ ] 3.1 Implement typed Rust controller SDK and sample controller
- [ ] 3.2 Implement typed Python controller SDK and sample controller
- [ ] 3.3 Add private container execution and heterogeneous tournament command
- [ ] 3.4 Run Roberto-versus-Julio compatibility tournament when controller is available

## 4. Verification and handoff

- [ ] 4.1 Add fake-clock deadline, malformed input, side-switch, and isolation tests
- [ ] 4.2 Add Rust-server/Python-controller end-to-end smoke and replay inspection
- [ ] 4.3 Record performance, artifacts, evidence, limitations, and rollback
- [ ] 4.4 Run doctor, build, test, lint, container gates, and signed commits
