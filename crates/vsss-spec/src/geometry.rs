//! Canonical field and body geometry.

use serde::{Deserialize, Serialize};

use crate::{Distance, Mass};

/// Rectangular field dimensions and goal geometry.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FieldGeometry {
    /// Playable length along x.
    pub length: Distance,
    /// Playable width along y.
    pub width: Distance,
    /// Goal mouth width.
    pub goal_width: Distance,
    /// Goal depth outside the field.
    pub goal_depth: Distance,
    /// Wall thickness.
    pub wall_thickness: Distance,
}

/// Robot footprint and mass.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RobotGeometry {
    /// Body length.
    pub length: Distance,
    /// Body width.
    pub width: Distance,
    /// Body mass.
    pub mass: Mass,
}

/// Differential-drive wheel geometry.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WheelGeometry {
    /// Wheel radius.
    pub radius: Distance,
    /// Distance between wheel contact centers.
    pub axle_track: Distance,
}

/// Ball physical properties.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BallProperties {
    /// Ball radius.
    pub radius: Distance,
    /// Ball mass.
    pub mass: Mass,
}
