//! Shared author-rule opportunities; no pitch or harmonic vocabulary.
use super::flora::TemporalObservation;
use super::phonation_engine::{CandidatePoint, CouplingClock};
use crate::core::meter::{MeterShaping, sane_basin};
use crate::core::modulation::NeuralRhythms;

pub(crate) struct HarmonicRhythm {
    clock: CouplingClock,
    interval: u64,
    remaining: u64,
    crossings: Vec<CandidatePoint>,
    pub(crate) ticks: Vec<u64>,
    pub(crate) observed: bool,
}

impl HarmonicRhythm {
    pub(crate) fn new(interval: u64, shaping: MeterShaping) -> Self {
        assert!(interval > 0);
        let base = shaping.basin_hz.map_or(2.0, |(lo, hi)| {
            let (lo, hi) = sane_basin(lo, hi);
            0.5 * (lo + hi)
        });
        Self {
            clock: CouplingClock::new(0.95, base, 0.0, 0.0, 0).scene_origin(),
            interval,
            remaining: 0,
            crossings: Vec::new(),
            ticks: Vec::new(),
            observed: false,
        }
    }

    pub(crate) fn advance(
        &mut self,
        fs: f32,
        now: u64,
        end: u64,
        observation: TemporalObservation,
    ) {
        self.observed = observation.known
            && observation.observed_end == Some(now)
            && observation.state.beat.confidence > 0.0
            && observation.state.beat.freq_hz.is_finite()
            && observation.state.beat.freq_hz > 0.0;
        let mut rhythms = NeuralRhythms::from_meter_state(&observation.state);
        if !self.observed {
            rhythms.delta.alpha = 0.0;
        }
        self.crossings.clear();
        self.ticks.clear();
        self.clock
            .gather_rhythm(fs, now, end, &rhythms, &mut self.crossings);
        self.crossings.dedup_by_key(|c| c.tick);
        for crossing in &self.crossings {
            if self.remaining == 0 {
                self.ticks.push(crossing.tick);
                self.remaining = self.interval - 1;
            } else {
                self.remaining -= 1;
            }
        }
    }
}

#[derive(Debug, serde::Serialize)]
pub(crate) struct VoiceEvent {
    pub(crate) tick: u64,
    pub(crate) action: &'static str,
    pub(crate) frequency_hz: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn harmonic_clock_counts_all_crossings_without_n_scaled_storage() {
        for (n, expected) in [
            (1, vec![500, 1000, 1500, 2000]),
            (3, vec![500, 2000]),
            (i64::MAX as u64, vec![500]),
        ] {
            let mut clock = HarmonicRhythm::new(n, Default::default());
            clock.advance(1000., 0, 2200, Default::default());
            assert_eq!(clock.ticks, expected);
            assert!(!clock.observed);
            assert!(clock.crossings.len() <= 5);
        }
    }

    #[test]
    fn harmonic_clock_unknown_and_reacquisition_preserve_phase_and_remainder() {
        let mut clock = HarmonicRhythm::new(3, Default::default());
        clock.advance(1000., 0, 750, Default::default());
        assert_eq!(clock.ticks, [500]);
        assert_eq!(clock.remaining, 2);
        let mut obs = TemporalObservation {
            known: true,
            observed_end: Some(750),
            ..Default::default()
        };
        obs.state.beat.freq_hz = 2.;
        obs.state.beat.confidence = 0.0;
        clock.advance(1000., 750, 1600, obs);
        assert!(clock.ticks.is_empty());
        assert_eq!(clock.remaining, 0);
        obs.observed_end = Some(1600);
        obs.state.beat.confidence = 1.0;
        obs.state.beat.phase = 0.2 * std::f32::consts::TAU;
        clock.advance(1000., 1600, 2200, obs);
        assert!(clock.observed);
        assert_eq!(clock.ticks, [2000]);
        assert_eq!(clock.remaining, 2);
    }
}
