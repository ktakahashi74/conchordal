use std::time::{Duration, Instant};

pub(crate) const STATS_INTERVAL: Duration = Duration::from_secs(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum StreamState {
    Running,
    Recovering { since: Instant, attempts: u32 },
    Failed,
}

pub(super) fn stream_lost(kind: cpal::ErrorKind) -> bool {
    matches!(
        kind,
        cpal::ErrorKind::DeviceNotAvailable
            | cpal::ErrorKind::StreamInvalidated
            | cpal::ErrorKind::HostUnavailable
            | cpal::ErrorKind::DeviceBusy
    )
}

pub(super) struct Recovery {
    pub(super) state: StreamState,
    frames: u64,
    last_progress: Instant,
    next_attempt: Instant,
}

impl Recovery {
    pub(super) fn new(now: Instant) -> Self {
        Self {
            state: StreamState::Running,
            frames: 0,
            last_progress: now,
            next_attempt: now,
        }
    }

    pub(super) fn observe(
        &mut self,
        now: Instant,
        frames: u64,
        lost: bool,
        ring_duration: Duration,
        cadence: Duration,
    ) -> bool {
        if self.state != StreamState::Running {
            return false;
        }
        if frames != self.frames {
            self.frames = frames;
            self.last_progress = now;
        }
        if lost || now.duration_since(self.last_progress) > ring_duration {
            self.state = StreamState::Recovering {
                since: now,
                attempts: 0,
            };
            self.next_attempt = now + cadence.min(STATS_INTERVAL);
            return true;
        }
        false
    }

    pub(super) fn retry_due(&self, now: Instant) -> bool {
        matches!(self.state, StreamState::Recovering { .. }) && now >= self.next_attempt
    }

    pub(super) fn attempted(
        &mut self,
        now: Instant,
        result: Result<(), cpal::ErrorKind>,
        cadence: Duration,
        frames: u64,
    ) {
        let StreamState::Recovering { since, attempts } = self.state else {
            return;
        };
        match result {
            Ok(()) => {
                self.state = StreamState::Running;
                self.frames = frames;
                self.last_progress = now;
            }
            Err(cpal::ErrorKind::UnsupportedConfig) => self.state = StreamState::Failed,
            Err(_) => {
                let attempts = attempts.saturating_add(1);
                self.state = StreamState::Recovering { since, attempts };
                let delay = cadence
                    .saturating_mul(1u32 << attempts.min(31))
                    .min(STATS_INTERVAL);
                self.next_attempt = now + delay;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cpal::ErrorKind::*;

    #[test]
    fn classifies_all_known_errors() {
        for kind in [
            DeviceNotAvailable,
            StreamInvalidated,
            HostUnavailable,
            DeviceBusy,
        ] {
            assert!(stream_lost(kind), "{kind:?}");
        }
        for kind in [
            Xrun,
            DeviceChanged,
            RealtimeDenied,
            InvalidInput,
            PermissionDenied,
            ResourceExhausted,
            UnsupportedConfig,
            UnsupportedOperation,
            BackendError,
            Other,
        ] {
            assert!(!stream_lost(kind), "{kind:?}");
        }
    }

    #[test]
    fn liveness_requires_a_stall_longer_than_the_ring_duration() {
        let now = Instant::now();
        let ring = Duration::from_millis(100);
        let cadence = Duration::from_millis(50);
        let mut recovery = Recovery::new(now);
        assert!(!recovery.observe(now + ring, 0, false, ring, cadence));
        assert!(!recovery.observe(now + ring, 100, false, ring, cadence));
        assert!(!recovery.observe(now + ring * 2, 100, false, ring, cadence));
        assert!(recovery.observe(now + ring * 2 + cadence, 100, false, ring, cadence));
    }

    #[test]
    fn retries_back_off_and_success_resets_liveness() {
        let now = Instant::now();
        let cadence = Duration::from_millis(50);
        let ring = Duration::from_millis(100);
        let mut recovery = Recovery::new(now);
        assert!(recovery.observe(now, 10, true, ring, cadence));
        assert!(!recovery.retry_due(now));
        let mut time = now + cadence;
        for attempt in 1..=8 {
            assert!(recovery.retry_due(time));
            recovery.attempted(time, Err(HostUnavailable), cadence, 10);
            assert_eq!(
                recovery.state,
                StreamState::Recovering {
                    since: now,
                    attempts: attempt
                }
            );
            let delay = (cadence * (1 << attempt)).min(STATS_INTERVAL);
            assert!(!recovery.retry_due(time + delay - Duration::from_nanos(1)));
            time += delay;
        }
        recovery.attempted(time, Ok(()), cadence, 20);
        assert_eq!(recovery.state, StreamState::Running);
        assert!(!recovery.observe(time + ring, 20, false, ring, cadence));
        assert!(recovery.observe(time + ring + cadence, 20, false, ring, cadence));
        assert_eq!(
            recovery.state,
            StreamState::Recovering {
                since: time + ring + cadence,
                attempts: 0
            }
        );
    }

    #[test]
    fn only_unsupported_rebuild_configuration_is_terminal() {
        let now = Instant::now();
        for kind in [
            Xrun,
            DeviceChanged,
            RealtimeDenied,
            DeviceNotAvailable,
            StreamInvalidated,
            HostUnavailable,
            DeviceBusy,
            InvalidInput,
            PermissionDenied,
            ResourceExhausted,
            UnsupportedConfig,
            UnsupportedOperation,
            BackendError,
            Other,
        ] {
            let mut recovery = Recovery::new(now);
            recovery.observe(now, 0, true, STATS_INTERVAL, STATS_INTERVAL);
            recovery.attempted(now + STATS_INTERVAL, Err(kind), STATS_INTERVAL, 0);
            assert_eq!(
                recovery.state == StreamState::Failed,
                kind == UnsupportedConfig
            );
        }
    }
}
