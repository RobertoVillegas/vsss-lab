//! Shared validation contracts.

use core::fmt;

/// A deterministic validation failure at a canonical field path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidationError {
    path: &'static str,
    message: &'static str,
}

impl ValidationError {
    /// Creates a validation error.
    #[must_use]
    pub const fn new(path: &'static str, message: &'static str) -> Self {
        Self { path, message }
    }

    /// Returns the failing field path.
    #[must_use]
    pub const fn path(&self) -> &'static str {
        self.path
    }
}

impl fmt::Display for ValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.path, self.message)
    }
}

impl std::error::Error for ValidationError {}

/// Validates semantic invariants that serialization alone cannot express.
pub trait Validate {
    /// Returns the first deterministic validation failure.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError`] when a value is non-finite or outside its
    /// canonical domain.
    fn validate(&self) -> Result<(), ValidationError>;
}
