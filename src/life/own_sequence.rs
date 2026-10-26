//! Author adapter: committed self onsets -> rounded intervals -> follow score.
use super::ppm_decay::{MAX_INTERVALS, Parameters, PpmDecay};
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ExecutionKind {
    On,
    Off,
    PitchUpdate,
    Missing,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct ExecutionRecord {
    pub(crate) voice_id: u64,
    pub(crate) generation: u32,
    pub(crate) tone_id: u64,
    pub(crate) kind: ExecutionKind,
    pub(crate) sample: u64,
    pub(crate) available_at: u64,
    pub(crate) execution_sequence: u64,
    pub(crate) frequency_hz: Option<f32>,
    pub(crate) strength: Option<f32>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct SequenceDecision {
    pub(crate) at: u64,
    pub(crate) epoch: u64,
    pub(crate) version: u64,
    pub(crate) active: bool,
    pub(crate) reason: Option<&'static str>,
    pub(crate) baseline_log2: f32,
    pub(crate) selected_log2: f32,
    pub(crate) without_sequence_log2: f32,
    pub(crate) changed: bool,
    pub(crate) previous_hz: Option<f64>,
    pub(crate) distribution: Vec<f64>,
    pub(crate) candidates: Vec<CandidateScore>,
    pub(crate) precision: Option<SelectionPrecision>,
}

#[derive(Clone, Debug, Default, serde::Serialize)]
pub(crate) struct SelectionPrecision {
    pub(crate) score_error_bound: f64,
    pub(crate) greedy_margin: f64,
    pub(crate) greedy_uncertain: bool,
    pub(crate) uncertain_rank_comparisons: usize,
    pub(crate) acceptance_probability: Option<f32>,
    pub(crate) acceptance_draw: Option<f32>,
    pub(crate) acceptance_relative_error: Option<f64>,
    pub(crate) acceptance_uncertain: bool,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct CandidateScore {
    pub(crate) pitch_log2: f32,
    pub(crate) existing: f32,
    pub(crate) sequence: f64,
    pub(crate) combined: f32,
    pub(crate) rounding_error: f64,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct SequenceLearning {
    pub(crate) sample: u64,
    pub(crate) tone_id: u64,
    pub(crate) epoch: u64,
    pub(crate) interval: Option<i32>,
    pub(crate) proposal_at: Option<u64>,
    pub(crate) predicted_interval: Option<i32>,
    pub(crate) mismatch: Option<&'static str>,
}

#[derive(Clone, Debug)]
pub(crate) struct SequenceSnapshot {
    pub(crate) probabilities: [f64; 241],
    pub(crate) alphabet: usize,
    pub(crate) offset: i32,
    pub(crate) previous_hz: f64,
    pub(crate) weight: f64,
    pub(crate) bounds_log2: (f32, f32),
    pub(crate) at: u64,
    pub(crate) version: u64,
    pub(crate) epoch: u64,
}

pub(crate) fn encode(frequency: f64, previous: f64) -> Option<i32> {
    (frequency.is_finite() && frequency > 0.0 && previous.is_finite() && previous > 0.0)
        .then(|| (12.0 * (frequency / previous).log2()).round() as i32)
}

impl SequenceSnapshot {
    pub(crate) fn symbol(&self, pitch_log2: f32) -> Option<usize> {
        let delta = encode(pitch_log2.exp2() as f64, self.previous_hz)?;
        let index = delta + self.offset;
        (index >= 0 && index < self.alphabet as i32).then_some(index as usize)
    }

    pub(crate) fn contribution(&self, pitch_log2: f32, baseline: f32) -> Option<f64> {
        let candidate = self.symbol(pitch_log2)?;
        let current = self.symbol(baseline)?;
        if candidate == current {
            return Some(0.0);
        }
        let result = self.weight
            * (self.probabilities[candidate].log2() - self.probabilities[current].log2());
        result.is_finite().then_some(result)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct OwnSequence {
    pub(crate) model: PpmDecay,
    pub(crate) epoch: u64,
    pub(crate) version: u64,
    pub(crate) reason: Option<&'static str>,
    pub(crate) resource_stopped: bool,
    pub(crate) resident: bool,
    pub(crate) last_snapshot: Option<SequenceSnapshot>,
    pub(crate) executions: Vec<ExecutionRecord>,
    pub(crate) learning: Vec<SequenceLearning>,
    pub(crate) trace_enabled: bool,
    pub(crate) supported: bool,
    previous: Option<(u64, f64)>,
    active: HashSet<u64>,
    seen: HashSet<(u64, u64)>,
    epoch_origin: u64,
    intervals: usize,
    decisions: usize,
    fs: f64,
    bounds: (f32, f32),
    offset: i32,
    weight: f64,
    seed: u64,
    rng: u64,
    pending: Option<(u64, i32, f64)>,
}

// Version 1: SplitMix64 domain separation and Box-Muller; independent of pitch RNG.
fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

impl OwnSequence {
    pub(crate) fn new(
        seed: u64,
        voice: u64,
        generation: u32,
        fs: f32,
        weight: f64,
        bounds: (f32, f32),
    ) -> Self {
        let offset = (12.0 * (bounds.1 as f64 - bounds.0 as f64)).ceil() as i32;
        let valid = fs.is_finite()
            && fs > 0.0
            && bounds.0.is_finite()
            && bounds.1.is_finite()
            && bounds.0 >= 20.0f32.log2()
            && bounds.1 <= 20000.0f32.log2()
            && bounds.0 <= bounds.1
            && (0..=120).contains(&offset);
        let offset = if valid { offset } else { 0 };
        let mut domain = seed ^ 0x53512d315f763031;
        domain = next(&mut domain) ^ voice;
        domain = next(&mut domain) ^ generation as u64;
        let seed = next(&mut domain);
        Self {
            model: PpmDecay::new((2 * offset + 1) as usize, 4, Parameters::default()),
            epoch: 0,
            version: 0,
            reason: (!valid).then_some("unsupported_range"),
            resource_stopped: !valid,
            resident: true,
            last_snapshot: None,
            executions: Vec::new(),
            learning: Vec::new(),
            trace_enabled: false,
            supported: true,
            previous: None,
            active: HashSet::with_capacity(16),
            seen: HashSet::with_capacity(MAX_INTERVALS + 1),
            epoch_origin: 0,
            intervals: 0,
            decisions: 0,
            fs: fs as f64,
            bounds,
            offset,
            weight,
            seed,
            rng: seed,
            pending: None,
        }
    }

    pub(crate) fn reject_residency(&mut self) {
        self.model.release_unstarted_storage();
        self.seen.shrink_to_fit();
        self.resource_stopped = true;
        self.resident = false;
        self.reason = Some("resident_state_resource_limit");
    }

    pub(crate) fn close_epoch(&mut self, reason: &'static str) {
        self.epoch += 1;
        self.version += 1;
        self.reason = Some(reason);
        self.model.clear();
        self.previous = None;
        self.last_snapshot = None;
        self.pending = None;
        self.rng = self.seed ^ self.epoch.wrapping_mul(0x9e3779b97f4a7c15);
        self.rng = next(&mut self.rng);
    }

    pub(crate) fn bind_target(&mut self, at: u64, pitch: f32) {
        self.pending = self
            .last_snapshot
            .as_ref()
            .filter(|s| s.at == at)
            .and_then(|snapshot| {
                snapshot
                    .symbol(pitch)
                    .map(|symbol| (at, symbol as i32 - self.offset, snapshot.previous_hz))
            });
    }

    pub(crate) fn reconfigure(&mut self, supported: bool, bounds: (f32, f32)) {
        if self.resource_stopped {
            self.supported = supported;
            return;
        }
        if self.supported != supported || self.bounds != bounds {
            self.close_epoch(if self.bounds != bounds {
                "range_changed"
            } else {
                "support_changed"
            });
            self.supported = supported;
            if self.bounds != bounds {
                let offset = (12.0 * (bounds.1 as f64 - bounds.0 as f64)).ceil() as i32;
                if bounds.0 < 20.0f32.log2()
                    || bounds.1 > 20000.0f32.log2()
                    || !(0..=120).contains(&offset)
                    || bounds.0 > bounds.1
                {
                    self.resource_stopped = true;
                    self.reason = Some("unsupported_range");
                } else {
                    self.bounds = bounds;
                    self.offset = offset;
                    self.model = PpmDecay::new((2 * offset + 1) as usize, 4, Parameters::default());
                }
            }
        }
    }

    pub(crate) fn observe(&mut self, record: ExecutionRecord) {
        if self.trace_enabled {
            self.executions.push(record.clone());
        }
        if record.sample as f64 > 600.0 * self.fs {
            self.resource_stopped = true;
            self.reason = Some("duration_resource_limit");
        }
        if self.resource_stopped {
            return;
        }
        if record.available_at != record.sample {
            self.close_epoch("late_or_missing_execution");
            return;
        }
        match record.kind {
            ExecutionKind::Off => {
                self.active.remove(&record.tone_id);
            }
            ExecutionKind::PitchUpdate | ExecutionKind::Missing => {
                self.close_epoch("pitch_update_or_missing")
            }
            ExecutionKind::On => {
                if !self.seen.insert((record.tone_id, record.sample)) {
                    return;
                }
                if self.seen.len() > MAX_INTERVALS + 1 {
                    self.resource_stopped = true;
                    self.reason = Some("event_resource_limit");
                    return;
                }
                self.active.insert(record.tone_id);
                if !self.supported {
                    return;
                }
                if self.active.len() > 1 {
                    self.close_epoch("unreleased_overlap");
                    return;
                }
                let Some(frequency) = record
                    .frequency_hz
                    .map(f64::from)
                    .filter(|f| f.is_finite() && *f > 0.0)
                else {
                    self.close_epoch("unknown_frequency");
                    return;
                };
                let log = (frequency as f32).log2();
                if log < self.bounds.0 || log > self.bounds.1 {
                    self.close_epoch("execution_outside_range");
                    return;
                }
                let actual_interval = self
                    .previous
                    .and_then(|(_, previous)| encode(frequency, previous));
                let predicted = self.pending.take();
                if self.trace_enabled {
                    self.learning.push(SequenceLearning {
                        sample: record.sample,
                        tone_id: record.tone_id,
                        epoch: self.epoch,
                        interval: actual_interval,
                        proposal_at: predicted.map(|p| p.0),
                        predicted_interval: predicted.map(|p| p.1),
                        mismatch: predicted.filter(|p| actual_interval != Some(p.1)).map(|p| {
                            if self
                                .previous
                                .is_some_and(|(_, previous)| previous.to_bits() != p.2.to_bits())
                            {
                                "intervening_onset"
                            } else {
                                "execution_constraint"
                            }
                        }),
                    });
                }
                if let Some((sample, previous)) = self.previous {
                    if record.sample <= sample {
                        self.close_epoch("nonincreasing_onset");
                        return;
                    }
                    let index = encode(frequency, previous).unwrap() + self.offset;
                    if index < 0 || index >= (2 * self.offset + 1) {
                        self.close_epoch("unknown_interval");
                        return;
                    }
                    if self.intervals == MAX_INTERVALS {
                        self.resource_stopped = true;
                        self.reason = Some("interval_resource_limit");
                        return;
                    }
                    let time = (record.sample - self.epoch_origin) as f64 / self.fs;
                    if let Err(reason) = self.model.learn(index as u16, time) {
                        self.close_epoch(reason);
                        return;
                    }
                    self.intervals += 1;
                } else {
                    self.epoch_origin = record.sample;
                }
                self.previous = Some((record.sample, frequency));
                self.version += 1;
            }
        }
    }

    pub(crate) fn predict(&mut self, at: u64) -> Option<SequenceSnapshot> {
        if !self.supported || self.resource_stopped {
            return None;
        }
        if self
            .last_snapshot
            .as_ref()
            .is_some_and(|s| s.at == at && s.version == self.version)
        {
            return self.last_snapshot.clone();
        }
        if self.decisions == 38400 || at as f64 > 600.0 * self.fs {
            self.resource_stopped = true;
            self.reason = Some("decision_or_duration_resource_limit");
            return None;
        }
        self.decisions += 1;
        if self.model.symbols.is_empty() {
            return None;
        }
        let time = at.checked_sub(self.epoch_origin)? as f64 / self.fs;
        let rng = &mut self.rng;
        let result = self.model.predict(time, || {
            let u = ((next(rng) >> 11) as f64 + 0.5) / 9007199254740992.0;
            let v = ((next(rng) >> 11) as f64 + 0.5) / 9007199254740992.0;
            0.8 * ((-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()).abs()
        });
        if let Err(reason) = result {
            self.reason = Some(reason);
            return None;
        }
        let mut probabilities = [0.0; 241];
        probabilities[..self.model.distribution.len()].copy_from_slice(&self.model.distribution);
        let snapshot = SequenceSnapshot {
            probabilities,
            alphabet: self.model.distribution.len(),
            offset: self.offset,
            previous_hz: self.previous?.1,
            weight: self.weight,
            bounds_log2: self.bounds,
            at,
            version: self.version,
            epoch: self.epoch,
        };
        self.last_snapshot = Some(snapshot.clone());
        Some(snapshot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(kind: ExecutionKind, tone: u64, sample: u64, hz: Option<f32>) -> ExecutionRecord {
        ExecutionRecord {
            voice_id: 2,
            generation: 0,
            tone_id: tone,
            kind,
            sample,
            available_at: sample,
            execution_sequence: sample,
            frequency_hz: hz,
            strength: Some(1.0),
        }
    }

    fn sequence() -> OwnSequence {
        OwnSequence::new(21, 2, 0, 48000.0, 1.0, (220.0f32.log2(), 880.0f32.log2()))
    }

    #[test]
    fn own_sequence_resource_stops_and_range_support_epochs() {
        let mut s = sequence();
        assert!(s.predict(600 * 48000 + 1).is_none());
        assert!(s.resource_stopped);
        s.reconfigure(false, s.bounds);
        s.reconfigure(true, s.bounds);
        assert!(s.resource_stopped);
        let mut s = sequence();
        s.decisions = 38400;
        assert!(s.predict(0).is_none() && s.resource_stopped);
        let mut s = sequence();
        for i in 0..=4800 {
            if i > 0 {
                s.observe(event(ExecutionKind::Off, i, 2 * i, None));
            }
            s.observe(event(ExecutionKind::On, i + 1, 2 * i, Some(440.0)));
        }
        assert_eq!(s.model.symbols.len(), 4800);
        assert!(!s.resource_stopped);
        s.observe(event(ExecutionKind::Off, 4801, 9602, None));
        s.observe(event(ExecutionKind::On, 4802, 9602, Some(440.0)));
        assert!(s.resource_stopped);
        assert_eq!(s.model.symbols.len(), 4800);
        let mut s = sequence();
        s.observe(event(ExecutionKind::On, 1, 0, Some(440.0)));
        s.reconfigure(false, s.bounds);
        s.observe(event(ExecutionKind::Off, 1, 1, None));
        s.observe(event(ExecutionKind::On, 2, 2, Some(660.0)));
        assert!(s.model.symbols.is_empty());
        s.reconfigure(true, (110.0f32.log2(), 880.0f32.log2()));
        s.observe(event(ExecutionKind::Off, 2, 3, None));
        s.observe(event(ExecutionKind::On, 3, 4, Some(440.0)));
        assert!(s.model.symbols.is_empty());
        assert_eq!(s.epoch, 2);
        assert_eq!(s.offset, 36);
    }

    #[test]
    fn own_sequence_learns_only_committed_onsets_and_freezes_prediction() {
        let mut s = sequence();
        s.observe(event(ExecutionKind::On, 1, 0, Some(440.0)));
        assert!(s.predict(100).is_none());
        s.observe(event(ExecutionKind::Off, 1, 100, None));
        s.observe(event(ExecutionKind::On, 2, 200, Some(440.0)));
        let snapshot = s.predict(300).unwrap();
        let before = s.model.symbols.clone();
        let rng = s.rng;
        for pitches in [[440.0f32, 660.0, 550.0], [550.0, 440.0, 660.0]] {
            for hz in pitches {
                let first = snapshot.contribution(hz.log2(), 440.0f32.log2());
                assert_eq!(
                    first,
                    s.predict(300)
                        .unwrap()
                        .contribution(hz.log2(), 440.0f32.log2())
                );
            }
        }
        assert_eq!(rng, s.rng);
        assert_eq!(before, s.model.symbols);
        s.observe(event(ExecutionKind::On, 2, 200, Some(440.0)));
        assert_eq!(before, s.model.symbols);
        assert_eq!(
            snapshot.contribution(440.0f32.log2(), 440.0f32.log2()),
            Some(0.0)
        );
        s.observe(event(ExecutionKind::PitchUpdate, 2, 400, Some(441.0)));
        assert!(s.model.symbols.is_empty());
        assert!(s.predict(500).is_none());
        s.observe(event(ExecutionKind::Off, 2, 600, None));
        s.observe(event(ExecutionKind::On, 3, 700, Some(440.0)));
        assert!(s.model.symbols.is_empty());
        s.observe(event(ExecutionKind::Off, 3, 800, None));
        s.observe(event(ExecutionKind::On, 4, 900, Some(660.0)));
        assert_eq!(s.model.symbols.len(), 1);
        s.observe(event(ExecutionKind::Missing, 4, 1000, None));
        assert!(s.model.symbols.is_empty());
    }

    #[test]
    fn own_sequence_overlap_and_late_records_close_epoch() {
        let mut s = sequence();
        s.observe(event(ExecutionKind::On, 1, 0, Some(440.0)));
        s.observe(event(ExecutionKind::On, 2, 10, Some(660.0)));
        assert_eq!(s.reason, Some("unreleased_overlap"));
        assert!(s.previous.is_none());
        s.observe(event(ExecutionKind::Off, 1, 20, None));
        s.observe(event(ExecutionKind::Off, 2, 20, None));
        let mut late = event(ExecutionKind::On, 3, 30, Some(440.0));
        late.available_at = 31;
        s.observe(late);
        assert_eq!(s.reason, Some("late_or_missing_execution"));
        assert!(s.model.symbols.is_empty());
    }

    #[test]
    fn own_sequence_encoder_cells_boundaries_and_transposition() {
        let hz = |cents: f64| 440.0 * 2.0f64.powf(cents / 1200.0);
        for (cents, delta) in [
            (50.0, 1),
            (-50.0, -1),
            (49.0, 0),
            (51.0, 1),
            (-49.0, 0),
            (-51.0, -1),
            (10.0, 0),
            (40.0, 0),
            (700.0, 7),
        ] {
            assert_eq!(encode(hz(cents), 440.0), Some(delta));
            assert_eq!(encode(1.25 * hz(cents), 550.0), Some(delta));
        }
        assert_eq!(encode(f64::NAN, 440.0), None);
        assert_eq!(encode(0.0, 440.0), None);
    }
}
