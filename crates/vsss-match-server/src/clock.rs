//! Injectable monotonic clocks.

use std::time::Instant;

/// Monotonic time source used for deadlines and leases.
pub trait Clock {
    /// Returns nanoseconds since an arbitrary stable epoch.
    fn now_ns(&self) -> u64;
}

/// Process-local production monotonic clock.
#[derive(Debug)]
pub struct SystemClock {
    origin: Instant,
}

impl Default for SystemClock {
    fn default() -> Self {
        Self {
            origin: Instant::now(),
        }
    }
}

impl Clock for SystemClock {
    fn now_ns(&self) -> u64 {
        u64::try_from(self.origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }
}
