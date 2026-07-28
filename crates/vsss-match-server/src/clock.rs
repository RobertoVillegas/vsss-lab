//! Injectable monotonic clocks.

use std::{sync::Arc, time::Instant};

/// Monotonic time source used for deadlines and leases.
pub trait Clock {
    /// Returns nanoseconds since an arbitrary stable epoch.
    fn now_ns(&self) -> u64;
}

/// Process-local production monotonic clock.
#[derive(Clone, Debug)]
pub struct SystemClock {
    origin: Arc<Instant>,
}

impl Default for SystemClock {
    fn default() -> Self {
        Self {
            origin: Arc::new(Instant::now()),
        }
    }
}

impl Clock for SystemClock {
    fn now_ns(&self) -> u64 {
        u64::try_from(self.origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }
}
