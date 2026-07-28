# Live visual frame protocol

Status: M4.1 local development protocol  
Authority: non-authoritative, lossy visualization only

## Transport

- UDP datagrams, default listener `127.0.0.1:42042`;
- producer socket is non-blocking;
- `VSS1` four-byte magic followed by zlib-compressed canonical JSON;
- complete datagram MUST be at most 1400 bytes;
- sampled on simulation tick count before constructing `VisualFrame`;
- sequence gaps and producer send failures are displayed as drops.

Loopback is the default and recommended binding. This protocol has no
authentication or encryption and MUST NOT be exposed publicly. Tailnet or
remote-browser support requires a later authenticated transport adapter.

## Envelope

```json
{
  "type": "visual_frame",
  "version": 1,
  "sequence": 0,
  "sample_every": 4,
  "send_errors": 0,
  "config": {},
  "frame": {
    "version": 1,
    "tick": 43,
    "simulation_time": 0.215,
    "snapshot": {},
    "actions": [],
    "events": 0,
    "checksum": "...",
    "rewards": null
  }
}
```

`config` makes each packet independently renderable after loss or late join.
The viewer keeps only a bounded 120-frame trail. It MUST NOT treat receipt,
ordering, or absence of live packets as simulation truth.

## Compatibility

Unknown JSON fields are ignored by the M4.1 viewer. A consumer rejects unknown
magic, envelope version, type, invalid compression, zero sampling interval, or
malformed canonical state. Any future incompatible wire change uses a new magic
or envelope version.

## Replay relationship

Both JSONL ticks and live packets decode into the same scene input. UDP loss
does not alter JSONL recording, simulation state, actions, checksums, or policy
execution. Exact analysis, seeking, and regression evidence always use replay.
