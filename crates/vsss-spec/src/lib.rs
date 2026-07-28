//! Canonical VSSS contracts.
//!
//! M0 intentionally exposes no domain model. M1 will introduce units, geometry,
//! entities, actions, events, reflection, and serialization behind reviewed ADRs.

/// Identifies the current repository milestone without defining domain behavior.
pub const BOOTSTRAP_MILESTONE: &str = "M0";

#[cfg(test)]
mod tests {
    use super::BOOTSTRAP_MILESTONE;

    #[test]
    fn milestone_is_explicit() {
        assert_eq!(BOOTSTRAP_MILESTONE, "M0");
    }
}
