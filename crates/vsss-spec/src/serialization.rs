//! Strict JSON interchange for canonical roots.

use serde::{Serialize, de::DeserializeOwned};

/// Serializes a canonical value as stable pretty JSON.
///
/// # Errors
///
/// Returns a JSON error if a value cannot be represented.
pub fn to_json<T: Serialize>(value: &T) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(value)
}

/// Deserializes a canonical value from JSON.
///
/// # Errors
///
/// Returns a JSON error for malformed or contract-incompatible input.
pub fn from_json<T: DeserializeOwned>(json: &str) -> Result<T, serde_json::Error> {
    serde_json::from_str(json)
}
