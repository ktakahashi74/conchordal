//! Complete event populations; wall time is not thread CPU time.

use serde::Serialize;

pub(super) fn elapsed_ns(started: std::time::Instant) -> u64 {
    started.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64
}

pub(super) const UPPER_US: [u64; 16] = [
    100, 250, 500, 1_000, 2_000, 4_000, 8_000, 16_000, 25_000, 40_000, 50_000, 75_000, 100_000,
    200_000, 500_000, 1_000_000,
];

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct Distribution {
    pub count: u64,
    pub total_ns: u64,
    pub maximum_ns: u64,
    pub limit_ns: u64,
    pub over_limit: u64,
    pub histogram: [u64; 17],
}

impl Distribution {
    fn record(&mut self, ns: u64) {
        self.count += 1;
        self.total_ns += ns;
        self.maximum_ns = self.maximum_ns.max(ns);
        self.over_limit += u64::from(ns > self.limit_ns);
        let bin = UPPER_US.partition_point(|upper| upper * 1_000 < ns);
        self.histogram[bin] += 1;
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct Window {
    pub epoch: u64,
    pub start_sample: u64,
    pub end_sample: u64,
    pub frames: u64,
    pub wall_ns: u64,
    pub table_ns: u64,
    pub complete_delivery: bool,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct Snapshot {
    pub startup_ns: u64,
    pub window_samples: u64,
    pub bucket_upper_us: [u64; 16],
    pub frames: Distribution,
    pub table: Distribution,
    pub delivery: Distribution,
    pub rejected: Distribution,
    pub finish: Distribution,
    pub windows: Distribution,
    pub complete_windows: Distribution,
    pub open_window: Option<Window>,
    pub last_window: Option<Window>,
    pub maximum_window: Option<Window>,
}

pub(super) struct Meter {
    pub snapshot: Snapshot,
    hop: u64,
    epoch: u64,
    origin: u64,
}

impl Meter {
    pub fn new(sample_rate: u32, hop: u64, startup_ns: u64) -> Self {
        let processing = Distribution {
            limit_ns: 50_000_000,
            ..Distribution::default()
        };
        Self {
            snapshot: Snapshot {
                startup_ns,
                window_samples: u64::from(sample_rate).div_ceil(10),
                bucket_upper_us: UPPER_US,
                frames: processing,
                table: processing,
                delivery: Distribution {
                    limit_ns: 25_000_000,
                    ..processing
                },
                rejected: processing,
                finish: processing,
                windows: processing,
                complete_windows: processing,
                ..Snapshot::default()
            },
            hop,
            epoch: 0,
            origin: 0,
        }
    }

    pub fn frame(&mut self, epoch: u64, start: u64, wall_ns: u64, table_ns: u64, delivery_ns: u64) {
        assert!(table_ns <= wall_ns);
        if epoch != self.epoch {
            self.close(start);
            self.epoch = epoch;
            self.origin = start;
        }
        let window_start = start - (start - self.origin) % self.snapshot.window_samples;
        if self
            .snapshot
            .open_window
            .is_some_and(|w| w.start_sample != window_start)
        {
            self.close(start);
        }
        let window = self.snapshot.open_window.get_or_insert(Window {
            epoch,
            start_sample: window_start,
            end_sample: window_start.saturating_add(self.snapshot.window_samples),
            ..Window::default()
        });
        window.frames += 1;
        window.wall_ns += wall_ns;
        window.table_ns += table_ns;
        self.snapshot.frames.record(wall_ns);
        self.snapshot.table.record(table_ns);
        self.snapshot.delivery.record(delivery_ns);
    }

    pub fn rejected(&mut self, wall_ns: u64) {
        self.snapshot.rejected.record(wall_ns);
    }

    pub fn finish(&mut self, end: u64, wall_ns: u64) {
        self.close(end);
        self.snapshot.finish.record(wall_ns);
    }

    pub fn close(&mut self, reached_sample: u64) {
        let Some(mut window) = self.snapshot.open_window.take() else {
            return;
        };
        let expected =
            (window.end_sample - 1) / self.hop + 1 - window.start_sample.div_ceil(self.hop);
        window.complete_delivery = reached_sample >= window.end_sample && window.frames == expected;
        self.snapshot.windows.record(window.wall_ns);
        if window.complete_delivery {
            self.snapshot.complete_windows.record(window.wall_ns);
        }
        if self
            .snapshot
            .maximum_window
            .is_none_or(|w| window.wall_ns > w.wall_ns)
        {
            self.snapshot.maximum_window = Some(window);
        }
        self.snapshot.last_window = Some(window);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_windows_preserve_every_cost_and_distinguish_gaps_epochs_and_partial_eof() {
        let mut meter = Meter::new(48000, 512, 17);
        for i in 0..20 {
            if i == 12 {
                continue;
            }
            meter.frame(0, i * 512, 6_000_000, 2_000_000, 26_000_000);
        }
        meter.rejected(13);
        // A restart cuts the open old-epoch bucket; it does not fabricate missing work.
        meter.frame(1, 20 * 512, 9_000_000, 0, 9_000_000);
        meter.finish(21 * 512, 19);
        let s = meter.snapshot;
        assert_eq!((s.frames.count, s.frames.total_ns), (20, 123_000_000));
        assert_eq!(
            (s.windows.count, s.windows.total_ns),
            (4, s.frames.total_ns)
        );
        assert_eq!(
            (s.complete_windows.count, s.complete_windows.total_ns),
            (1, 60_000_000)
        );
        assert_eq!(s.windows.over_limit, 1);
        assert_eq!(s.delivery.over_limit, 19);
        assert_eq!(s.rejected.total_ns, 13);
        assert_eq!(s.finish.total_ns, 19);
        assert_eq!(s.startup_ns, 17);
        assert_eq!(s.maximum_window.unwrap().wall_ns, 60_000_000);
        assert!(!s.last_window.unwrap().complete_delivery);
        assert_eq!(s.last_window.unwrap().epoch, 1);
        assert!(s.open_window.is_none());
        for d in [
            s.frames,
            s.table,
            s.delivery,
            s.windows,
            s.complete_windows,
            s.rejected,
            s.finish,
        ] {
            assert_eq!(d.histogram.iter().sum::<u64>(), d.count);
        }
    }

    #[test]
    fn histogram_limits_are_inclusive_and_overflow_is_explicit() {
        let mut d = Distribution {
            limit_ns: 50_000_000,
            ..Distribution::default()
        };
        for ns in [0, 100_000, 100_001, 50_000_000, 50_000_001, 1_000_000_001] {
            d.record(ns);
        }
        assert_eq!(
            (
                d.histogram[0],
                d.histogram[1],
                d.histogram[10],
                d.histogram[11],
                d.histogram[16]
            ),
            (2, 1, 1, 1, 1)
        );
        assert_eq!(d.over_limit, 2);
        let mut headroom = Distribution {
            limit_ns: 40_000_000,
            ..Distribution::default()
        };
        headroom.record(40_000_000);
        headroom.record(40_000_001);
        assert_eq!((headroom.histogram[9], headroom.histogram[10]), (1, 1));
        assert_eq!(headroom.over_limit, 1);
    }
}
