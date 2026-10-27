//! TK2003 Model 1, with a separately identified self-command/generation adapter.
use super::own_sequence::{ExecutionKind, ExecutionRecord};

pub(crate) const PROFILES: [[f64; 12]; 2] = [
    [
        6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88,
    ],
    [
        6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17,
    ],
];

// Author adapter: half ties away from zero; Euclidean pitch-class numbering.
pub(crate) fn pitch_class(hz: f64) -> Option<usize> {
    (hz.is_finite() && hz > 0.0)
        .then(|| ((69.0 + 12.0 * (hz / 440.0).log2()).round() as i32).rem_euclid(12) as usize)
}

#[derive(Clone, Debug)]
pub(crate) struct TonalityModel {
    pub(crate) memory: [f64; 12],
    log_scale: f64,
    active: Option<(usize, f64)>,
    at: u64,
    error: f64,
}

impl Default for TonalityModel {
    fn default() -> Self {
        Self {
            memory: [0.0; 12],
            log_scale: 0.0,
            active: None,
            at: 0,
            error: 0.0,
        }
    }
}

impl TonalityModel {
    pub(crate) fn advance(&mut self, at: u64, fs: f64) {
        let dt = (at - self.at) as f64 / fs;
        self.at = at;
        if dt == 0.0 {
            return;
        }
        let old_scale = self.log_scale - dt / 3.0;
        if let Some((class, log_p)) = self.active {
            // Integral of exp(-t/.5) through dm/dt = p - m/3, not (p-m)/3.
            let log_input =
                log_p - dt / 3.0 + (-(-dt * (5.0 / 3.0)).exp_m1()).ln() - (5.0f64 / 3.0).ln();
            let scale = if self.memory.iter().any(|&x| x > 0.0) {
                old_scale.max(log_input)
            } else {
                log_input
            };
            let decay = (old_scale - scale).exp();
            for m in &mut self.memory {
                *m *= decay;
            }
            self.memory[class] += (log_input - scale).exp();
            let max = self.memory.iter().copied().fold(0.0, f64::max);
            for m in &mut self.memory {
                *m /= max;
            }
            self.error = self.error * decay / max + 64.0 * f64::EPSILON;
            self.log_scale = scale + max.ln();
            self.active = Some((class, log_p - 2.0 * dt));
        } else {
            // Common decay leaves the shape, including its numerical error, unchanged.
            self.log_scale = old_scale;
        }
    }

    pub(crate) fn estimate(&self) -> Estimate {
        let mean = self.memory.iter().sum::<f64>() / 12.0;
        let centered = self.memory.map(|x| x - mean);
        let norm = centered.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 {
            return Estimate {
                correlations: None,
                key: None,
                error_bound: 0.0,
                reason: Some("undefined_correlation"),
            };
        }
        let bound =
            4.0 * 12.0f64.sqrt() * (self.error + 64.0 * f64::EPSILON) / norm + 128.0 * f64::EPSILON;
        let mut correlations = [0.0; 24];
        for (key, r) in correlations.iter_mut().enumerate() {
            let profile = PROFILES[key / 12];
            let mean = profile.iter().sum::<f64>() / 12.0;
            let mut dot = 0.0;
            let mut square = 0.0;
            for (c, m) in centered.iter().enumerate() {
                let k = profile[(c + 12 - key % 12) % 12] - mean;
                dot += m * k;
                square += k * k;
            }
            *r = (dot / (norm * square.sqrt())).clamp(-1.0, 1.0);
        }
        let best = (0..24)
            .max_by(|&a, &b| correlations[a].total_cmp(&correlations[b]))
            .unwrap();
        let runner_up = (0..24)
            .filter(|&i| i != best)
            .map(|i| correlations[i])
            .fold(f64::NEG_INFINITY, f64::max);
        let key = (correlations[best] - runner_up > 2.0 * bound).then_some(best);
        Estimate {
            correlations: Some(correlations),
            key,
            error_bound: bound,
            reason: key.is_none().then_some("numerical_selection_unresolved"),
        }
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Estimate {
    pub(crate) correlations: Option<[f64; 24]>,
    pub(crate) key: Option<usize>,
    pub(crate) error_bound: f64,
    pub(crate) reason: Option<&'static str>,
}

#[derive(Clone, Debug)]
pub(crate) struct TonalitySnapshot {
    pub(crate) estimate: Estimate,
    pub(crate) weight: f64,
    pub(crate) bounds_log2: (f32, f32),
}

impl TonalitySnapshot {
    pub(crate) fn contribution(&self, pitch: f32, baseline: f32) -> Option<f64> {
        let key = self.estimate.key?;
        let profile = PROFILES[key / 12];
        let candidate = pitch_class(pitch.exp2() as f64)?;
        let current = pitch_class(baseline.exp2() as f64)?;
        let min = profile.iter().copied().fold(f64::INFINITY, f64::min);
        let max = profile.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Some(
            self.weight
                * ((profile[(candidate + 12 - key % 12) % 12] - min) / (max - min)
                    - (profile[(current + 12 - key % 12) % 12] - min) / (max - min)),
        )
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct TonalityDecision {
    pub(crate) at: u64,
    pub(crate) epoch: u64,
    pub(crate) estimate: Option<Estimate>,
    pub(crate) reason: Option<&'static str>,
    pub(crate) baseline_log2: f32,
    pub(crate) selected_log2: f32,
    pub(crate) without_tonality_log2: f32,
    pub(crate) changed: bool,
    pub(crate) candidates: Vec<TonalityScore>,
    pub(crate) precision: super::own_sequence::SelectionPrecision,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct TonalityScore {
    pub(crate) pitch_log2: f32,
    pub(crate) existing: f32,
    pub(crate) sequence: f64,
    pub(crate) tonality: f64,
    pub(crate) combined: f32,
}

#[derive(Debug)]
pub(crate) struct OwnTonality {
    pub(crate) model: TonalityModel,
    pub(crate) weight: f64,
    pub(crate) supported: bool,
    pub(crate) stopped: bool,
    pub(crate) reason: Option<&'static str>,
    pub(crate) epoch: u64,
    pub(crate) trace_enabled: bool,
    pub(crate) executions: Vec<ExecutionRecord>,
    pub(crate) decisions: Vec<TonalityDecision>,
    voice: u64,
    generation: u32,
    fs: f64,
    active: [Option<u64>; 16],
    last_sequence: Option<u64>,
    last_sample: u64,
    last_on: Option<u64>,
    last_tone: Option<u64>,
    recovering: bool,
}

impl OwnTonality {
    pub(crate) fn new(voice: u64, generation: u32, fs: f32, weight: f64) -> Self {
        Self {
            model: TonalityModel::default(),
            weight,
            supported: true,
            stopped: false,
            reason: None,
            epoch: 0,
            trace_enabled: false,
            executions: Vec::new(),
            decisions: Vec::new(),
            voice,
            generation,
            fs: fs as f64,
            active: [None; 16],
            last_sequence: None,
            last_sample: 0,
            last_on: None,
            last_tone: None,
            recovering: false,
        }
    }

    pub(crate) fn stop(&mut self, reason: &'static str) {
        self.stopped = true;
        self.reason = Some(reason);
    }

    fn close(&mut self, reason: &'static str) {
        self.model = TonalityModel::default();
        self.epoch += 1;
        self.recovering = true;
        self.reason = Some(reason);
    }

    pub(crate) fn reconfigure(&mut self, supported: bool) {
        if self.supported != supported {
            self.close("support_changed");
            self.supported = supported;
        }
    }

    pub(crate) fn observe(&mut self, record: &ExecutionRecord) {
        if self.stopped {
            return;
        }
        if record.voice_id != self.voice || record.generation != self.generation {
            self.stop("identity_mismatch");
            return;
        }
        if self.last_sequence == Some(record.execution_sequence) {
            return;
        }
        if self
            .last_sequence
            .is_some_and(|s| record.execution_sequence < s)
        {
            self.stop("unordered_execution");
            return;
        }
        self.last_sequence = Some(record.execution_sequence);
        if self.trace_enabled {
            self.executions.push(record.clone());
        }
        if record.available_at != record.sample || record.sample < self.last_sample {
            self.stop("late_or_unordered_execution");
            return;
        }
        self.last_sample = record.sample;
        self.model.advance(record.sample, self.fs);
        match record.kind {
            ExecutionKind::Missing => self.stop("missing_execution"),
            ExecutionKind::PitchUpdate => self.close("pitch_update"),
            ExecutionKind::Off => {
                for tone in &mut self.active {
                    if *tone == Some(record.tone_id) {
                        *tone = None;
                    }
                }
                if self.active.iter().all(Option::is_none) {
                    self.model.active = None;
                }
            }
            ExecutionKind::On => {
                if self.active.contains(&Some(record.tone_id)) {
                    if self.last_on != Some(record.sample) || self.last_tone != Some(record.tone_id)
                    {
                        self.stop("reused_active_identity");
                    }
                    return;
                }
                let empty = self.active.iter().all(Option::is_none);
                let Some(slot) = self.active.iter_mut().find(|x| x.is_none()) else {
                    self.stop("active_identity_capacity");
                    return;
                };
                *slot = Some(record.tone_id);
                if self.last_on.is_some_and(|t| record.sample <= t)
                    || self.last_tone.is_some_and(|id| record.tone_id <= id)
                {
                    self.stop("nonincreasing_onset");
                    return;
                }
                self.last_on = Some(record.sample);
                self.last_tone = Some(record.tone_id);
                if !empty {
                    self.close("unreleased_overlap");
                    return;
                }
                if !self.supported {
                    return;
                }
                let Some(class) = record.frequency_hz.and_then(|hz| pitch_class(hz as f64)) else {
                    self.stop("unknown_frequency");
                    return;
                };
                if self.recovering {
                    self.model = TonalityModel {
                        at: record.sample,
                        ..Default::default()
                    };
                    self.recovering = false;
                }
                self.reason = None;
                self.model.active = Some((class, 0.0));
            }
        }
    }

    pub(crate) fn snapshot(&self, at: u64, bounds_log2: (f32, f32)) -> Option<TonalitySnapshot> {
        if self.stopped || !self.supported || self.recovering || at < self.model.at {
            return None;
        }
        let mut model = self.model.clone();
        model.advance(at, self.fs);
        Some(TonalitySnapshot {
            estimate: model.estimate(),
            weight: self.weight,
            bounds_log2,
        })
    }
}

#[cfg(test)]
#[path = "own_tonality_tests.rs"]
mod tests;
