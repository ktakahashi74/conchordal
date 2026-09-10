//! Ordered NSGT evidence before delivery coalescing; not source or stream identity.

use std::sync::Arc;

use crate::core::log2space::Log2Space;
use crate::core::temporal_history::{AuditoryHistory, HISTORY_AGES_SEC};

#[derive(Clone, Debug)]
pub(crate) struct SpectralHistorySnapshot {
    pub(crate) space: Arc<Log2Space>,
    pub(crate) sample_rate: f64,
    pub(crate) observed_through_sample: u64,
    /// FFT container support, not the latency of each smoothed NSGT band.
    pub(crate) window_samples: usize,
    pub(crate) post_order: usize,
    pub(crate) known_rms_by_age_scan: Vec<[f32; 8]>,
    pub(crate) known_coverage_by_age: [f32; 8],
}

pub(crate) struct SpectralHistory {
    space: Arc<Log2Space>,
    sample_rate: f64,
    window_samples: usize,
    next_sample: u64,
    contiguous_samples: usize,
    // Three independent bins reuse the existing numerical operator without pooling.
    lanes: Vec<AuditoryHistory>,
}

impl SpectralHistory {
    pub(crate) fn new(space: Log2Space, sample_rate: f64, window_samples: usize) -> Self {
        assert!(sample_rate.is_finite() && sample_rate > 0.0 && window_samples > 0);
        Self {
            lanes: vec![AuditoryHistory::new(HISTORY_AGES_SEC); space.n_bins().div_ceil(3)],
            space: Arc::new(space),
            sample_rate,
            window_samples,
            next_sample: 0,
            contiguous_samples: 0,
        }
    }

    pub(crate) fn observe(&mut self, start_sample: u64, end_sample: u64, power_scan: &[f32]) {
        self.space
            .assert_scan_len_named(power_scan, "spectral_history_power_scan");
        assert!(start_sample >= self.next_sample && end_sample > start_sample);
        if start_sample > self.next_sample {
            let dt = (start_sample - self.next_sample) as f64 / self.sample_rate;
            for lane in &mut self.lanes {
                lane.advance(dt, None);
            }
            self.contiguous_samples = 0;
        }
        let count = end_sample - start_sample;
        self.contiguous_samples = self
            .contiguous_samples
            .saturating_add(count as usize)
            .min(self.window_samples);
        let available = self.contiguous_samples == self.window_samples;
        let dt = count as f64 / self.sample_rate;
        for (index, lane) in self.lanes.iter_mut().enumerate() {
            let input = available.then(|| {
                std::array::from_fn(|channel| {
                    power_scan
                        .get(index * 3 + channel)
                        .copied()
                        .unwrap_or(0.0)
                        .sqrt()
                })
            });
            lane.advance(dt, input);
        }
        self.next_sample = end_sample;
    }

    /// Allocation belongs to the analysis handoff, not to the history update.
    pub(crate) fn snapshot(&self) -> Option<Arc<SpectralHistorySnapshot>> {
        if self.contiguous_samples < self.window_samples {
            return None;
        }
        let mut scan = Vec::with_capacity(self.space.n_bins());
        for lane in &self.lanes {
            let snapshot = lane.snapshot();
            for channel in 0..3 {
                if scan.len() < self.space.n_bins() {
                    scan.push(snapshot.known_band_rms_by_age.map(|age| age[channel]));
                }
            }
        }
        self.space
            .assert_scan_len_named(&scan, "known_rms_by_age_scan");
        Some(Arc::new(SpectralHistorySnapshot {
            space: Arc::clone(&self.space),
            sample_rate: self.sample_rate,
            observed_through_sample: self.next_sample,
            window_samples: self.window_samples,
            post_order: self.lanes[0].snapshot().post_order,
            known_rms_by_age_scan: scan,
            known_coverage_by_age: self.lanes[0].snapshot().known_coverage_by_age,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "profile-alloc")]
    #[test]
    fn history_updates_allocate_nothing_and_snapshots_only_allocate_the_handoff() {
        let space = Log2Space::new(55.0, 8000.0, 96);
        let power = vec![0.25; space.n_bins()];
        let mut history = SpectralHistory::new(space.clone(), 48000.0, 2048);
        for frame in 0..8 {
            history.observe(frame * 512, (frame + 1) * 512, &power);
        }
        crate::runtime_profile::begin_allocations();
        for frame in 8..24 {
            history.observe(frame * 512, (frame + 1) * 512, &power);
        }
        let counts = crate::runtime_profile::finish_allocations().unwrap();
        assert_eq!(counts.count, 0);
        crate::runtime_profile::begin_allocations();
        let snapshot = history.snapshot().unwrap();
        let counts = crate::runtime_profile::finish_allocations().unwrap();
        assert_eq!(counts.count, 2);
        assert_eq!(snapshot.known_rms_by_age_scan.len(), space.n_bins());
    }

    #[test]
    fn frequency_order_survives_a_common_present_and_incomplete_last_lane() {
        let space = Log2Space::new(100.0, 200.0, 3);
        assert_eq!(space.n_bins(), 4);
        let mut ab = SpectralHistory::new(space.clone(), 1000.0, 10);
        let mut ba = SpectralHistory::new(space, 1000.0, 10);
        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [0.0, 0.0, 0.0, 1.0];
        for i in 0..60 {
            let (x, y) = if i < 20 {
                (&a, &b)
            } else if i < 40 {
                (&b, &a)
            } else {
                (&a, &a)
            };
            ab.observe(i * 10, (i + 1) * 10, x);
            ba.observe(i * 10, (i + 1) * 10, y);
        }
        let a = ab.snapshot().unwrap();
        let b = ba.snapshot().unwrap();
        assert_eq!(a.observed_through_sample, 600);
        assert_eq!(a.known_coverage_by_age, b.known_coverage_by_age);
        assert!(a.known_rms_by_age_scan[3][1] > b.known_rms_by_age_scan[3][1]);
        assert!(a.known_rms_by_age_scan[3][2] < b.known_rms_by_age_scan[3][2]);
        assert!(a.known_rms_by_age_scan[1].iter().all(|v| *v == 0.0));
        assert!(a.known_rms_by_age_scan[2].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn gap_retains_prior_evidence_but_never_invents_silence_or_refill_observations() {
        let space = Log2Space::new(100.0, 200.0, 3);
        let mut history = SpectralHistory::new(space, 1000.0, 40);
        let mut reference = AuditoryHistory::new(HISTORY_AGES_SEC);
        for i in 0..20 {
            history.observe(i * 10, (i + 1) * 10, &[1.0; 4]);
            reference.advance(0.01, (i >= 3).then_some([1.0; 3]));
            assert_eq!(history.snapshot().is_some(), i >= 3);
        }
        reference.advance(0.2, None);
        for i in 40..44 {
            history.observe(i * 10, (i + 1) * 10, &[0.0; 4]);
            reference.advance(0.01, (i >= 43).then_some([0.0; 3]));
            assert_eq!(history.snapshot().is_some(), i >= 43);
        }
        let actual = history.snapshot().unwrap();
        let expected = reference.snapshot();
        assert_eq!(actual.known_coverage_by_age, expected.known_coverage_by_age);
        assert_eq!(
            actual.known_rms_by_age_scan[3],
            expected.known_band_rms_by_age.map(|a| a[0])
        );
        assert!(actual.known_rms_by_age_scan[3][2] > 0.0);
        assert!(actual.known_coverage_by_age[2] < 0.4);
    }

    #[test]
    #[should_panic(expected = "spectral_history_power_scan")]
    fn input_scan_must_match_frequency_coordinates() {
        SpectralHistory::new(Log2Space::new(100.0, 200.0, 3), 1000.0, 10).observe(0, 10, &[0.0; 3]);
    }

    #[test]
    #[should_panic]
    fn observation_clock_cannot_move_backwards() {
        let mut history = SpectralHistory::new(Log2Space::new(100.0, 200.0, 3), 1000.0, 10);
        history.observe(10, 20, &[0.0; 4]);
        history.observe(0, 10, &[0.0; 4]);
    }
}
