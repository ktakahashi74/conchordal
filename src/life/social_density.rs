use crate::core::timebase::Tick;

#[derive(Clone, Debug, Default)]
pub struct SocialDensityTrace {
    pub start_tick: Tick,
    pub bin_ticks: u32,
    pub bins: Vec<f32>,
}

impl SocialDensityTrace {
    pub fn density_at(&self, tick: Tick) -> f32 {
        if self.bin_ticks == 0 {
            return 0.0;
        }
        if tick < self.start_tick {
            return 0.0;
        }
        let idx = ((tick - self.start_tick) / self.bin_ticks as Tick) as usize;
        self.bins.get(idx).copied().unwrap_or(0.0)
    }

    pub fn from_onsets(
        start_tick: Tick,
        end_tick: Tick,
        bin_ticks: u32,
        smooth: f32,
        population_size: usize,
        onsets: &[(Tick, f32)],
    ) -> Self {
        if bin_ticks == 0 || end_tick <= start_tick {
            return Self::default();
        }
        let span = end_tick - start_tick;
        let n_bins = span.div_ceil(bin_ticks as Tick) as usize;
        let mut bins = vec![0.0; n_bins];
        for (tick, strength) in onsets {
            if *tick < start_tick || *tick >= end_tick {
                continue;
            }
            let idx = ((*tick - start_tick) / bin_ticks as Tick) as usize;
            if let Some(slot) = bins.get_mut(idx) {
                *slot += strength.max(0.0);
            }
        }
        if population_size > 0 {
            let norm = population_size as f32;
            for v in &mut bins {
                *v /= norm;
            }
        }
        let smooth = smooth.clamp(0.0, 1.0);
        if smooth > 0.0 {
            let mut prev = bins.first().copied().unwrap_or(0.0);
            for v in &mut bins {
                *v = smooth * prev + (1.0 - smooth) * *v;
                prev = *v;
            }
        }
        Self {
            start_tick,
            bin_ticks,
            bins,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn density_at_out_of_range_is_zero() {
        let trace = SocialDensityTrace {
            start_tick: 10,
            bin_ticks: 5,
            bins: vec![1.0],
        };
        assert_eq!(trace.density_at(9), 0.0);
        assert_eq!(trace.density_at(10), 1.0);
        assert_eq!(trace.density_at(20), 0.0);
    }

    #[test]
    fn from_onsets_bins_and_normalizes() {
        let trace =
            SocialDensityTrace::from_onsets(0, 10, 5, 0.0, 2, &[(0, 1.0), (1, 1.0), (6, 1.0)]);
        assert_eq!(trace.bins.len(), 2);
        assert!((trace.bins[0] - 1.0).abs() < 1e-6);
        assert!((trace.bins[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn from_onsets_bin_count_and_boundary_assignment() {
        let trace = SocialDensityTrace::from_onsets(
            100,
            200,
            10,
            0.0,
            2,
            &[(100, 1.0), (110, 1.0), (110, 1.0)],
        );
        assert_eq!(trace.bins.len(), 10);
        assert!((trace.bins[0] - 0.5).abs() < 1e-6);
        assert!((trace.bins[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn from_onsets_boundary_tick_goes_to_next_bin() {
        let trace = SocialDensityTrace::from_onsets(0, 12, 4, 0.0, 1, &[(4, 1.0)]);
        assert_eq!(trace.bins.len(), 3);
        assert_eq!(trace.bins[0], 0.0);
        assert_eq!(trace.bins[1], 1.0);
    }

    #[test]
    fn from_onsets_clamps_smoothing() {
        let trace = SocialDensityTrace::from_onsets(0, 10, 5, 2.0, 1, &[(0, 1.0)]);
        assert!(trace.bins.iter().all(|v| v.is_finite()));
    }
}
