//! Strong SI unit wrappers used by every canonical contract.

use serde::{Deserialize, Serialize};

macro_rules! quantity {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, PartialOrd, Serialize)]
        #[repr(transparent)]
        #[serde(transparent)]
        pub struct $name(pub f32);

        impl $name {
            /// Creates a quantity from its canonical SI scalar.
            #[must_use]
            pub const fn new(value: f32) -> Self {
                Self(value)
            }

            /// Returns the canonical SI scalar.
            #[must_use]
            pub const fn get(self) -> f32 {
                self.0
            }

            /// Returns whether the scalar is finite.
            #[must_use]
            pub fn is_finite(self) -> bool {
                self.0.is_finite()
            }
        }
    };
}

quantity!(Distance, "Distance in metres.");
quantity!(Seconds, "Time in seconds.");
quantity!(Angle, "Angle in radians.");
quantity!(LinearVelocity, "Linear velocity in metres per second.");
quantity!(
    AngularVelocity,
    "Angular or wheel velocity in radians per second."
);
quantity!(Mass, "Mass in kilograms.");
quantity!(Force, "Force in newtons.");
quantity!(Torque, "Torque in newton-metres.");

impl Angle {
    /// Normalizes an angle to `[-π, π)`.
    #[must_use]
    pub fn normalized(self) -> Self {
        let two_pi = 2.0 * core::f32::consts::PI;
        Self((self.0 + core::f32::consts::PI).rem_euclid(two_pi) - core::f32::consts::PI)
    }
}
