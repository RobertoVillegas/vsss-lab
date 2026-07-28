//! Stable event bit assignments.

use serde::{Deserialize, Serialize};

/// Events emitted for a match tick.
#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct EventFlags(pub u32);

impl EventFlags {
    /// No events.
    pub const NONE: Self = Self(0);
    /// Blue scored.
    pub const GOAL_BLUE: Self = Self(1 << 0);
    /// Yellow scored.
    pub const GOAL_YELLOW: Self = Self(1 << 1);
    /// Ball left the playable area.
    pub const BALL_OUT: Self = Self(1 << 2);
    /// Match time expired.
    pub const MATCH_ENDED: Self = Self(1 << 3);
    /// Bits known by schema version 1.
    pub const KNOWN_BITS: u32 =
        Self::GOAL_BLUE.0 | Self::GOAL_YELLOW.0 | Self::BALL_OUT.0 | Self::MATCH_ENDED.0;

    /// Returns whether all bits in `other` are set.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    /// Swaps team-relative event bits.
    #[must_use]
    pub const fn reflected(self) -> Self {
        let invariant = self.0 & !(Self::GOAL_BLUE.0 | Self::GOAL_YELLOW.0);
        let blue = if self.contains(Self::GOAL_YELLOW) {
            Self::GOAL_BLUE.0
        } else {
            0
        };
        let yellow = if self.contains(Self::GOAL_BLUE) {
            Self::GOAL_YELLOW.0
        } else {
            0
        };
        Self(invariant | blue | yellow)
    }
}
