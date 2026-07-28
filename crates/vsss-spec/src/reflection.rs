//! Deterministic static contract reflection.

/// Broad serialized field shape.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FieldKind {
    /// Unsigned integer.
    Unsigned,
    /// Boolean.
    Boolean,
    /// Explicit SI scalar.
    Quantity,
    /// Enumeration.
    Enumeration,
    /// Nested record.
    Record,
    /// Fixed-size sequence.
    FixedArray,
    /// Event bit mask.
    BitFlags,
}

/// One public serialized field.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FieldDescriptor {
    /// Serialized field name.
    pub name: &'static str,
    /// Broad field shape.
    pub kind: FieldKind,
}

/// One canonical public type.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TypeDescriptor {
    /// Rust type name.
    pub name: &'static str,
    /// Serialized fields.
    pub fields: &'static [FieldDescriptor],
}

const MATCH_STATE_FIELDS: &[FieldDescriptor] = &[
    FieldDescriptor {
        name: "schema_version",
        kind: FieldKind::Unsigned,
    },
    FieldDescriptor {
        name: "tick",
        kind: FieldKind::Unsigned,
    },
    FieldDescriptor {
        name: "simulation_time",
        kind: FieldKind::Quantity,
    },
    FieldDescriptor {
        name: "score_blue",
        kind: FieldKind::Unsigned,
    },
    FieldDescriptor {
        name: "score_yellow",
        kind: FieldKind::Unsigned,
    },
    FieldDescriptor {
        name: "ball",
        kind: FieldKind::Record,
    },
    FieldDescriptor {
        name: "robots",
        kind: FieldKind::FixedArray,
    },
    FieldDescriptor {
        name: "events",
        kind: FieldKind::BitFlags,
    },
];
const MATCH_CONFIG_FIELDS: &[FieldDescriptor] = &[
    FieldDescriptor {
        name: "schema_version",
        kind: FieldKind::Unsigned,
    },
    FieldDescriptor {
        name: "field",
        kind: FieldKind::Record,
    },
    FieldDescriptor {
        name: "robot",
        kind: FieldKind::Record,
    },
    FieldDescriptor {
        name: "wheel",
        kind: FieldKind::Record,
    },
    FieldDescriptor {
        name: "ball",
        kind: FieldKind::Record,
    },
    FieldDescriptor {
        name: "timestep",
        kind: FieldKind::Quantity,
    },
    FieldDescriptor {
        name: "control_period",
        kind: FieldKind::Quantity,
    },
    FieldDescriptor {
        name: "seed",
        kind: FieldKind::Unsigned,
    },
    FieldDescriptor {
        name: "backend",
        kind: FieldKind::Enumeration,
    },
];
const TYPES: &[TypeDescriptor] = &[
    TypeDescriptor {
        name: "MatchState",
        fields: MATCH_STATE_FIELDS,
    },
    TypeDescriptor {
        name: "MatchConfig",
        fields: MATCH_CONFIG_FIELDS,
    },
];

/// Returns the stable root-type catalog.
#[must_use]
pub const fn canonical_types() -> &'static [TypeDescriptor] {
    TYPES
}
