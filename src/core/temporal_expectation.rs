//! Causal recurrence inference. Period candidates are not a common onset clock.

use crate::core::history_prediction::HistoryEnergyPrediction;
use crate::core::onset::OnsetDetector;
use crate::core::stream::dorsal::DorsalStream;
use crate::core::temporal_history::{AuditoryHistory, AuditoryHistorySnapshot, HISTORY_AGES_SEC};

const MAX_DELAY: usize = 400;
const HISTORY: usize = 3 * MAX_DELAY + 5;
const KERNEL: [f32; 5] = [0.08, 0.24, 0.36, 0.24, 0.08];
pub(super) const FORECAST_LEN: usize = 201;
pub(super) const FORECAST_STRIDE: usize = 2;
pub(super) const ENERGY_PENDING_LEN: usize = (FORECAST_LEN - 1) * FORECAST_STRIDE + 1;

#[derive(Clone)]
struct IssuedEnergyForecast {
    step: Option<u64>,
    recurrence: [[f32; 3]; FORECAST_LEN],
    persistence: [f32; 3],
    weight: [[f32; 3]; FORECAST_LEN],
}

/// Delayed regression, separately for each lead and acoustic band.
/// Its cumulative statistics are an engineering estimator, not cognitive forgetting.
struct EnergyForecastLearning {
    pending: Box<[IssuedEnergyForecast]>,
    feature_square: [[f64; 3]; FORECAST_LEN],
    residual_cross: [[f64; 3]; FORECAST_LEN],
}

impl EnergyForecastLearning {
    fn new() -> Self {
        Self {
            pending: vec![
                IssuedEnergyForecast {
                    step: None,
                    recurrence: [[0.0; 3]; FORECAST_LEN],
                    persistence: [0.0; 3],
                    weight: [[1.0; 3]; FORECAST_LEN],
                };
                ENERGY_PENDING_LEN
            ]
            .into_boxed_slice(),
            feature_square: [[0.0; 3]; FORECAST_LEN],
            residual_cross: [[0.0; 3]; FORECAST_LEN],
        }
    }

    fn reset(&mut self) {
        self.feature_square.fill([0.0; 3]);
        self.residual_cross.fill([0.0; 3]);
        for issued in &mut self.pending {
            issued.step = None;
        }
    }

    fn observe(&mut self, step: u64, energy: [f32; 3]) {
        for lead in 0..FORECAST_LEN {
            let Some(issued_step) = step.checked_sub((lead * FORECAST_STRIDE) as u64) else {
                continue;
            };
            let issued = &self.pending[(issued_step % ENERGY_PENDING_LEN as u64) as usize];
            if issued.step != Some(issued_step) {
                continue;
            }
            for (band, observed) in energy.into_iter().enumerate() {
                let delta = issued.persistence[band] as f64 - issued.recurrence[lead][band] as f64;
                self.residual_cross[lead][band] +=
                    delta * (observed as f64 - issued.recurrence[lead][band] as f64);
            }
        }
    }

    fn issue(&mut self, step: u64, recurrence: [[f32; 3]; FORECAST_LEN], persistence: [f32; 3]) {
        let issued = &mut self.pending[(step % ENERGY_PENDING_LEN as u64) as usize];
        debug_assert_ne!(
            issued.step,
            Some(step),
            "issue each observation boundary once"
        );
        issued.step = Some(step);
        issued.recurrence = recurrence;
        issued.persistence = persistence;
        for lead in 0..FORECAST_LEN {
            for band in 0..3 {
                // Candidate differences are already known; their outcomes are not.
                let delta = persistence[band] as f64 - recurrence[lead][band] as f64;
                self.feature_square[lead][band] += delta * delta;
                issued.weight[lead][band] = if self.feature_square[lead][band] > 0.0 {
                    (1.0 - (self.residual_cross[lead][band] / self.feature_square[lead][band])
                        .clamp(0.0, 1.0)) as f32
                } else {
                    1.0
                };
            }
        }
    }
}

#[cfg(test)]
#[path = "temporal_assay.rs"]
mod audio_assay;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Prediction {
    /// No onset, low, mid, high; these categories sum to one.
    pub(crate) probability: [f32; 4],
    pub(crate) null_probability: [f32; 4],
    /// Mixture mass, not perceptual beat confidence.
    pub(crate) recurrence_weight: f32,
    pub(crate) recent_onsets: f32,
    pub(crate) leading_period_sec: Option<f32>,
}

pub(crate) struct TemporalExpectation {
    step_sec: f32,
    min_delay: usize,
    max_delay: usize,
    step_index: u64,
    history: [[f32; 3]; HISTORY],
    recent_marks: [u8; 5],
    weights: [f32; MAX_DELAY + 1],
    prior: [f32; MAX_DELAY + 1],
    predictions: [[f32; 4]; MAX_DELAY + 1],
    counts: [f32; 4],
    recent_onsets: f32,
    switch_decay: f32,
    count_decay: f32,
}

/// An acoustic or self-produced energy trace; it does not infer a separate meter.
#[derive(Clone)]
pub(crate) struct BandEnergyHistory {
    history: [[f32; 3]; HISTORY],
    mean: [f32; 3],
    decay: f32,
    step_index: u64,
    step_sec: f32,
    temporal: AuditoryHistory,
    history_prediction: HistoryEnergyPrediction,
}

impl BandEnergyHistory {
    pub(crate) fn new(step_sec: f32, start_step: u64) -> Self {
        Self {
            history: [[0.0; 3]; HISTORY],
            mean: [0.0; 3],
            decay: (-step_sec / 4.0).exp(),
            step_index: start_step,
            step_sec,
            temporal: AuditoryHistory::new(HISTORY_AGES_SEC),
            history_prediction: HistoryEnergyPrediction::new(),
        }
    }

    pub(crate) fn observe(
        &mut self,
        energy: [f32; 3],
        emit: impl FnMut(&crate::core::history_prediction::PredictionMatch<'_>),
    ) {
        self.history_prediction
            .observe(self.step_index, energy, emit);
        self.temporal
            .advance(self.step_sec as f64, Some(energy.map(f32::sqrt)));
        self.history[(self.step_index % HISTORY as u64) as usize] = energy;
        for (mean, value) in self.mean.iter_mut().zip(energy) {
            *mean = self.decay * *mean + (1.0 - self.decay) * value;
        }
        self.step_index += 1;
        self.history_prediction
            .issue(self.step_index, energy, &self.temporal.snapshot());
    }
}

/// Known self PCM is removed before acoustic energy analysis, preserving interference.
pub(crate) struct OwnSoundHistory {
    pub(crate) external: BandEnergyHistory,
    pub(crate) profile: [f32; 3],
    body_dorsal: DorsalStream,
    external_dorsal: DorsalStream,
    body_window: Vec<f32>,
    external_window: Vec<f32>,
    filled: usize,
    next_frame: u64,
    profile_decay: f32,
    reported_prediction_frame: u64,
}

impl OwnSoundHistory {
    pub(crate) fn take_prediction_errors(
        &mut self,
    ) -> Option<(
        u64,
        u64,
        usize,
        crate::core::history_prediction::PredictionErrorTotals,
    )> {
        let from = self.reported_prediction_frame;
        let through = self.observed_through_frame();
        self.reported_prediction_frame = through;
        self.external
            .history_prediction
            .take_errors()
            .map(|errors| (from, through, self.body_window.len(), errors))
    }

    pub(crate) fn observed_through_frame(&self) -> u64 {
        self.next_frame - self.filled as u64
    }

    /// Exact completed window only; future, expired and off-grid evidence is unavailable.
    pub(crate) fn observed_external_energy(&self, start: u64, end: u64) -> Option<[f32; 3]> {
        let window = self.body_window.len() as u64;
        let lag = self.observed_through_frame().checked_sub(end)?;
        if end.checked_sub(start)? != window || !lag.is_multiple_of(window) {
            return None;
        }
        let back = lag / window;
        if back >= HISTORY as u64 || back >= self.external.step_index {
            return None;
        }
        Some(
            self.external.history
                [((self.external.step_index - 1 - back) % HISTORY as u64) as usize],
        )
    }

    /// Start tracking at birth, before this actor has contributed any sound.
    pub(crate) fn new(observer: &AcousticTemporalExpectation, start_frame: u64) -> Self {
        assert_eq!(observer.next_input_frame.unwrap_or(0), start_frame);
        let window = observer.window.len();
        let step_sec = observer.model.step_sec;
        Self {
            external: observer.energy.clone(),
            profile: [0.0; 3],
            body_dorsal: DorsalStream::new(observer.sample_rate as f32),
            external_dorsal: observer.dorsal.clone(),
            body_window: vec![0.0; window],
            // Pre-birth habitat, including the pending prefix, is all external.
            external_window: observer.window.clone(),
            filled: observer.filled,
            next_frame: start_frame,
            profile_decay: (-step_sec / 2.0).exp(),
            reported_prediction_frame: observer.observed_through_frame,
        }
    }

    pub(crate) fn process(
        &mut self,
        now: u64,
        mut body: &[f32],
        mut own_habitat: &[f32],
        mut habitat_mix: &[f32],
        mut emit: impl FnMut(u64, usize, &crate::core::history_prediction::PredictionMatch<'_>),
    ) {
        assert_eq!(
            now, self.next_frame,
            "self sound must advance with the audio clock"
        );
        assert_eq!(body.len(), own_habitat.len());
        assert_eq!(body.len(), habitat_mix.len());
        self.next_frame += body.len() as u64;
        while !body.is_empty() {
            let take = body.len().min(self.body_window.len() - self.filled);
            self.body_window[self.filled..self.filled + take].copy_from_slice(&body[..take]);
            for (i, value) in self.external_window[self.filled..self.filled + take]
                .iter_mut()
                .enumerate()
            {
                *value = habitat_mix[i] - own_habitat[i];
            }
            self.filled += take;
            body = &body[take..];
            own_habitat = &own_habitat[take..];
            habitat_mix = &habitat_mix[take..];
            if self.filled == self.body_window.len() {
                self.body_dorsal.process(&self.body_window);
                self.external_dorsal.process(&self.external_window);
                let h = self.external_dorsal.last_metrics();
                let window = self.external_window.len();
                let target_start = self.next_frame - body.len() as u64 - window as u64;
                self.external
                    .observe([h.e_low, h.e_mid, h.e_high], |matched| {
                        emit(target_start, window, matched)
                    });
                let b = self.body_dorsal.last_metrics();
                let energy = [b.e_low, b.e_mid, b.e_high];
                self.profile = self.profile.map(|e| e * self.profile_decay);
                if energy.iter().sum::<f32>() > self.profile.iter().sum::<f32>() {
                    self.profile = energy;
                }
                self.filled = 0;
            }
        }
    }
}

impl TemporalExpectation {
    /// Called at a 50–200 Hz observation rate, independent of output hop size.
    pub(crate) fn new(step_sec: f32) -> Self {
        assert!(step_sec.is_finite() && (0.005..=0.02).contains(&step_sec));
        let min_delay = (0.25 / step_sec).ceil() as usize;
        let max_delay = (2.0 / step_sec).floor() as usize;
        assert!(max_delay <= MAX_DELAY && min_delay > KERNEL.len());
        let mut prior = [0.0; MAX_DELAY + 1];
        prior[0] = 0.5;
        let norm: f32 = (min_delay..=max_delay).map(|lag| 1.0 / lag as f32).sum();
        for lag in min_delay..=max_delay {
            prior[lag] = 0.5 / lag as f32 / norm;
        }
        Self {
            step_sec,
            min_delay,
            max_delay,
            step_index: 0,
            history: [[0.0; 3]; HISTORY],
            recent_marks: [0; 5],
            weights: prior,
            prior,
            predictions: [[0.0; 4]; MAX_DELAY + 1],
            counts: [1.0, 0.001, 0.001, 0.001],
            recent_onsets: 0.0,
            switch_decay: (-step_sec / 6.0).exp(),
            count_decay: (-step_sec / 4.0).exp(),
        }
    }

    /// Repeat only fully observed cycles, fading to the rate baseline with distance.
    pub(crate) fn predict_at(&mut self, ahead_steps: usize) -> Prediction {
        let retention = (-(ahead_steps as f32) * self.step_sec / 6.0).exp();
        let count: f32 = self.counts.iter().sum();
        let null = self.counts.map(|v| (v + 0.01) / (count + 0.04));
        self.predictions[0] = null;
        let mut mixture = null.map(|p| self.weights[0] * p);
        let mut best = self.min_delay;
        for lag in self.min_delay..=self.max_delay {
            let prediction = if self.step_index < (3 * lag + 2) as u64 || self.recent_onsets < 1.0 {
                null
            } else {
                // Wrap into the latest fully observed cycle, never future ring slots.
                let at =
                    self.step_index + ahead_steps as u64 - ((ahead_steps + 2) / lag * lag) as u64;
                let a = self.history[((at - lag as u64) % HISTORY as u64) as usize];
                let b = self.history[((at - 2 * lag as u64) % HISTORY as u64) as usize];
                let c = self.history[((at - 3 * lag as u64) % HISTORY as u64) as usize];
                let mass = std::array::from_fn::<_, 3, _>(|k| 0.6 * a[k] + 0.3 * b[k] + 0.1 * c[k]);
                let sum: f32 = mass.iter().sum();
                let scale = sum.max(1.0);
                let recurrence = [
                    (1.0 - sum).max(0.0),
                    mass[0] / scale,
                    mass[1] / scale,
                    mass[2] / scale,
                ];
                std::array::from_fn(|k| {
                    let short = 0.9 * recurrence[k] + 0.1 * null[k];
                    retention * short + (1.0 - retention) * null[k]
                })
            };
            self.predictions[lag] = prediction;
            for k in 0..4 {
                mixture[k] += self.weights[lag] * prediction[k];
            }
            if self.weights[lag] > self.weights[best] {
                best = lag;
            }
        }
        Prediction {
            probability: mixture,
            null_probability: null,
            recurrence_weight: 1.0 - self.weights[0],
            recent_onsets: self.recent_onsets,
            leading_period_sec: (self.recent_onsets >= 1.0
                && self.step_index >= (3 * best + 2) as u64
                && self.weights[0] < 0.5)
                .then_some(best as f32 * self.step_sec),
        }
    }

    /// Returns the forecast made before this evidence and its relative log score.
    pub(crate) fn observe(&mut self, band: u8) -> (Prediction, f32) {
        assert!(band <= 3);
        // Recompute at zero: a preceding look-ahead query must not change learning.
        let predicted = self.predict_at(0);
        let observed = band as usize;
        let gain_bits =
            (predicted.probability[observed] / predicted.null_probability[observed]).log2();
        let mut norm = self.weights[0] * self.predictions[0][observed];
        for lag in self.min_delay..=self.max_delay {
            norm += self.weights[lag] * self.predictions[lag][observed];
        }
        self.weights[0] = self.switch_decay * self.weights[0] * self.predictions[0][observed]
            / norm
            + (1.0 - self.switch_decay) * self.prior[0];
        for lag in self.min_delay..=self.max_delay {
            self.weights[lag] =
                self.switch_decay * self.weights[lag] * self.predictions[lag][observed] / norm
                    + (1.0 - self.switch_decay) * self.prior[lag];
        }
        for k in 0..4 {
            self.counts[k] = self.counts[k] * self.count_decay + u8::from(k == observed) as f32;
        }
        self.recent_onsets = self.recent_onsets * self.count_decay + u8::from(band != 0) as f32;
        let now = self.step_index;
        self.recent_marks[(now % 5) as usize] = band;
        if let Some(center) = now.checked_sub(2) {
            let mut value = [0.0; 3];
            for (offset, weight) in KERNEL.iter().copied().enumerate() {
                if let Some(source) = (center + offset as u64).checked_sub(2) {
                    let mark = self.recent_marks[(source % 5) as usize];
                    if mark != 0 {
                        value[mark as usize - 1] += weight;
                    }
                }
            }
            self.history[(center % HISTORY as u64) as usize] = value;
        }
        self.step_index += 1;
        (predicted, gain_bits)
    }

    fn energy_at(&self, history: &BandEnergyHistory, ahead: usize) -> [f32; 3] {
        assert_eq!(
            history.step_index, self.step_index,
            "energy and event observations must align"
        );
        let retention = (-(ahead as f32) * self.step_sec / 6.0).exp();
        let mut energy = history.mean.map(|e| self.weights[0] * e);
        for lag in self.min_delay..=self.max_delay {
            let predicted = if self.step_index < (3 * lag + 2) as u64 || self.recent_onsets < 1.0 {
                history.mean
            } else {
                let at = self.step_index + ahead as u64 - ((ahead + 2) / lag * lag) as u64;
                let a = history.history[((at - lag as u64) % HISTORY as u64) as usize];
                let b = history.history[((at - 2 * lag as u64) % HISTORY as u64) as usize];
                let c = history.history[((at - 3 * lag as u64) % HISTORY as u64) as usize];
                std::array::from_fn(|k| {
                    let short =
                        0.9 * (0.6 * a[k] + 0.3 * b[k] + 0.1 * c[k]) + 0.1 * history.mean[k];
                    retention * short + (1.0 - retention) * history.mean[k]
                })
            };
            for k in 0..3 {
                energy[k] += self.weights[lag] * predicted[k];
            }
        }
        energy
    }

    fn energy_forecast(&self, history: &BandEnergyHistory) -> [[f32; 3]; FORECAST_LEN] {
        // No lag can consult history here, so every horizon has the same ordered sum.
        if self.recent_onsets < 1.0 || self.step_index < (3 * self.min_delay + 2) as u64 {
            [self.energy_at(history, 0); FORECAST_LEN]
        } else {
            std::array::from_fn(|lead| self.energy_at(history, lead * FORECAST_STRIDE))
        }
    }
}

/// Acoustic evidence is framed by its own window, not by runtime hop boundaries.
pub(crate) struct AcousticTemporalExpectation {
    model: TemporalExpectation,
    energy: BandEnergyHistory,
    energy_learning: EnergyForecastLearning,
    dorsal: DorsalStream,
    detector: OnsetDetector,
    sample_rate: u32,
    window: Vec<f32>,
    filled: usize,
    next_input_frame: Option<u64>,
    observed_through_frame: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct TemporalObservation {
    pub(crate) start_frame: u64,
    pub(crate) end_frame: u64,
    pub(crate) band: u8,
    pub(crate) band_energy: [f32; 3],
    pub(crate) prediction: Prediction,
    pub(crate) gain_bits: f32,
}

/// Bounded future evidence, shared without selecting a common period or onset phase.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TemporalForecast {
    pub(crate) observed_history: Option<AuditoryHistorySnapshot>,
    pub(crate) energy_prediction_model: &'static str,
    start_frame: u64,
    available_through_frame: u64,
    step_frames: usize,
    len: usize,
    contrast: [[f32; 3]; FORECAST_LEN],
    band_energy: [[f32; 3]; FORECAST_LEN],
    known_energy_windows: [u64; (FORECAST_LEN * FORECAST_STRIDE).div_ceil(64)],
    energy_recurrence_weight: [[f32; 3]; FORECAST_LEN],
    background_energy: [f32; 3],
    period_step_sec: f32,
    period_weights: [f32; MAX_DELAY + 1],
    recurrence_support: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) struct EnergyFootprintPoint {
    pub sample: u64,
    pub sample_fraction: f64,
    pub band_energy_sum: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) struct ExternalEnergyFootprint {
    pub version: u8,
    pub requested: [u64; 2],
    pub horizon_intersection: Option<[u64; 2]>,
    pub continuous_support: Option<bool>,
    pub omitted_prefix_samples: Option<u64>,
    pub omitted_tail_samples: Option<u64>,
    pub points: [Option<EnergyFootprintPoint>; 16],
}

impl TemporalForecast {
    fn refresh_energy_support(&mut self) {
        self.known_energy_windows.fill(0);
        let mut previous = false;
        for (i, bands) in self.band_energy[..self.len].iter().enumerate() {
            let known = bands.iter().all(|v| v.is_finite() && *v >= 0.);
            let index = i * FORECAST_STRIDE;
            if known {
                self.known_energy_windows[index / 64] |= 1 << (index % 64);
            }
            if i > 0 && known && previous {
                for between in index - FORECAST_STRIDE + 1..index {
                    self.known_energy_windows[between / 64] |= 1 << (between % 64);
                }
            }
            previous = known;
        }
    }

    fn scalar_window(&self, index: u64) -> Option<f64> {
        let window = self.step_frames / FORECAST_STRIDE;
        let at = self
            .start_frame
            .checked_add(index.checked_mul(window as u64)?)?;
        if at < self.available_through_frame {
            return None;
        }
        let left = usize::try_from(index / FORECAST_STRIDE as u64).ok()?;
        if left >= self.len {
            return None;
        }
        let blend = (index % FORECAST_STRIDE as u64) as f64 / FORECAST_STRIDE as f64;
        let mut value = 0.;
        for band in 0..3 {
            let a = f64::from(self.band_energy[left][band]);
            if !a.is_finite() || a < 0. {
                return None;
            }
            value += if blend == 0. {
                a
            } else {
                if left + 1 >= self.len {
                    return None;
                }
                let b = f64::from(self.band_energy[left + 1][band]);
                if !b.is_finite() || b < 0. {
                    return None;
                }
                a * (1. - blend) + b * blend
            };
        }
        Some(value)
    }

    /// Center interpolation stays within the connected support of the containing window.
    pub(crate) fn centered_energy(&self, sample: u64, fraction: f64) -> Option<f64> {
        if !fraction.is_finite() || !(0. ..1.).contains(&fraction) {
            return None;
        }
        let width = (self.step_frames / FORECAST_STRIDE) as u64;
        let offset = sample.checked_sub(self.start_frame)?;
        let index = offset / width;
        let value = self.scalar_window(index)?;
        let within = (offset % width) as f64 + fraction;
        let half = width as f64 / 2.;
        if within < half {
            if let Some(previous) = index.checked_sub(1).and_then(|i| self.scalar_window(i)) {
                let mix = 0.5 + within / width as f64;
                return Some(previous * (1. - mix) + value * mix);
            }
        } else if within > half
            && let Some(next) = self.scalar_window(index + 1)
        {
            let mix = within / width as f64 - 0.5;
            return Some(value * (1. - mix) + next * mix);
        }
        Some(value)
    }

    pub(crate) fn external_footprint(
        &self,
        requested: [u64; 2],
        limit: u64,
    ) -> ExternalEnergyFootprint {
        let mut result = ExternalEnergyFootprint {
            version: 1,
            requested,
            horizon_intersection: None,
            continuous_support: None,
            omitted_prefix_samples: None,
            omitted_tail_samples: None,
            points: [None; 16],
        };
        let width = (self.step_frames / FORECAST_STRIDE) as u64;
        let first = self
            .available_through_frame
            .saturating_sub(self.start_frame)
            .div_ceil(width);
        let Some(begin) = first
            .checked_mul(width)
            .and_then(|v| self.start_frame.checked_add(v))
        else {
            return result;
        };
        let Some(end) = (self.len.saturating_sub(1) as u64)
            .checked_mul(self.step_frames as u64)
            .and_then(|v| self.start_frame.checked_add(v))
            .and_then(|v| v.checked_add(width))
        else {
            return result;
        };
        let begin = begin.max(requested[0]);
        let end = end.min(limit).min(requested[1]);
        if end <= begin {
            return result;
        }
        result.horizon_intersection = Some([begin, end]);
        let first = ((begin - self.start_frame) / width) as usize;
        let last = ((end - self.start_frame - 1) / width) as usize;
        result.continuous_support = Some((first / 64..=last / 64).all(|word| {
            let low = first.saturating_sub(word * 64);
            let high = (last - word * 64).min(63);
            let mask = (u64::MAX << low) & (u64::MAX >> (63 - high));
            self.known_energy_windows[word] & mask == mask
        }));
        result.omitted_prefix_samples = Some(begin - requested[0]);
        result.omitted_tail_samples = Some(requested[1] - end);
        for (i, point) in result.points.iter_mut().enumerate() {
            let numerator = u128::from(end - begin) * (2 * i + 1) as u128;
            let sample = begin + (numerator / 32) as u64;
            let fraction = (numerator % 32) as f64 / 32.;
            *point = Some(EnergyFootprintPoint {
                sample,
                sample_fraction: fraction,
                band_energy_sum: self.centered_energy(sample, fraction),
            });
        }
        result
    }

    #[cfg(test)]
    pub(crate) fn energy_fixture(
        fs: u32,
        start: u64,
        mut energy: impl FnMut(f64) -> [f32; 3],
    ) -> Self {
        let step_frames = (fs as f64 / 50.0).round() as usize;
        let band_energy =
            std::array::from_fn(|i| energy(i as f64 * step_frames as f64 / fs as f64));
        let mut forecast = Self {
            observed_history: None,
            energy_prediction_model: "fixture",
            start_frame: start,
            available_through_frame: start,
            step_frames,
            len: FORECAST_LEN,
            contrast: [[0.0; 3]; FORECAST_LEN],
            background_energy: band_energy[FORECAST_LEN - 1],
            band_energy,
            known_energy_windows: [0; (FORECAST_LEN * FORECAST_STRIDE).div_ceil(64)],
            energy_recurrence_weight: [[1.0; 3]; FORECAST_LEN],
            period_step_sec: 0.01,
            period_weights: [0.0; MAX_DELAY + 1],
            recurrence_support: 0.0,
        };
        forecast.refresh_energy_support();
        forecast
    }

    pub(crate) fn observed_frame(&self) -> u64 {
        self.start_frame
    }

    /// No window before this frame may inform a prediction.
    pub(crate) fn available_through_frame(&self) -> u64 {
        self.available_through_frame
    }

    /// First complete observation window at or after an action, on this observer's grid.
    pub(crate) fn energy_window_after(&self, frame: u64) -> Option<(u64, u64, [f32; 3])> {
        let window = (self.step_frames / FORECAST_STRIDE) as u64;
        let offset = frame.checked_sub(self.start_frame)?.div_ceil(window);
        let start = self.start_frame.checked_add(offset.checked_mul(window)?)?;
        let end = start.checked_add(window)?;
        Some((start, end, self.band_energy_at(start as f64)?))
    }

    /// Select a supported recurrence near this body's scale, without a phase target.
    pub(crate) fn period_reference(&self, intrinsic_period_sec: f32) -> Option<(f32, f32)> {
        if self.recurrence_support <= 0.0 {
            return None;
        }
        let mut best = None;
        let mut best_weight = 0.0;
        for (lag, weight) in self.period_weights.iter().copied().enumerate().skip(1) {
            if weight <= 0.0 {
                continue;
            }
            let period = lag as f32 * self.period_step_sec;
            let distance = (period / intrinsic_period_sec).log2();
            let affinity = crate::core::float::unit_gaussian(distance, 0.5);
            if weight * affinity > best_weight {
                best_weight = weight * affinity;
                best = Some((period, self.recurrence_support));
            }
        }
        best
    }
    /// Signed predictive excess over the adaptive event-rate baseline, not confidence.
    pub(crate) fn contrast_at(&self, frame: f64) -> Option<[f32; 3]> {
        if !frame.is_finite() || frame < self.available_through_frame as f64 {
            return None;
        }
        let position = (frame - self.start_frame as f64) / self.step_frames as f64;
        if position < 0.0 || position > (self.len - 1) as f64 {
            return None;
        }
        let index = position.floor() as usize;
        let next = (index + 1).min(self.len - 1);
        let mix = (position - index as f64) as f32;
        Some(std::array::from_fn(|k| {
            self.contrast[index][k] * (1.0 - mix) + self.contrast[next][k] * mix
        }))
    }

    pub(crate) fn band_energy_at(&self, frame: f64) -> Option<[f32; 3]> {
        if !frame.is_finite() || frame < self.available_through_frame as f64 {
            return None;
        }
        let position = (frame - self.start_frame as f64) / self.step_frames as f64;
        if position < 0.0 || position > (self.len - 1) as f64 {
            return None;
        }
        let index = position.floor() as usize;
        let next = (index + 1).min(self.len - 1);
        let mix = (position - index as f64) as f32;
        Some(std::array::from_fn(|k| {
            self.band_energy[index][k] * (1.0 - mix) + self.band_energy[next][k] * mix
        }))
    }

    /// Beyond the recurrence horizon, sustained sound meets the observed mean, not silence.
    pub(crate) fn sustained_energy_at(&self, frame: f64) -> Option<[f32; 3]> {
        if frame > self.start_frame as f64 + (self.len - 1) as f64 * self.step_frames as f64
            && frame.is_finite()
        {
            Some(self.background_energy)
        } else {
            self.band_energy_at(frame)
        }
    }
}

impl AcousticTemporalExpectation {
    /// Rates below 50 Hz cannot resolve the minimum observation window.
    pub(crate) fn new(sample_rate: u32) -> Option<Self> {
        if sample_rate < 50 {
            return None;
        }
        let window_len = (sample_rate as f64 / 100.0).round() as usize;
        let step_sec = window_len as f32 / sample_rate as f32;
        Some(Self {
            model: TemporalExpectation::new(step_sec),
            energy: BandEnergyHistory::new(step_sec, 0),
            energy_learning: EnergyForecastLearning::new(),
            dorsal: DorsalStream::new(sample_rate as f32),
            detector: OnsetDetector::default(),
            sample_rate,
            window: vec![0.0; window_len],
            filled: 0,
            next_input_frame: None,
            observed_through_frame: 0,
        })
    }

    pub(crate) fn process(
        &mut self,
        start_frame: u64,
        mut audio: &[f32],
        mut report: impl FnMut(TemporalObservation),
    ) {
        if audio.is_empty() {
            return;
        }
        if self.next_input_frame != Some(start_frame) {
            // Missing audio is not observed silence, and partial windows cannot span it.
            let mut temporal = self.energy.temporal.clone();
            if start_frame < self.observed_through_frame {
                temporal = AuditoryHistory::new(HISTORY_AGES_SEC);
            } else if start_frame > self.observed_through_frame {
                temporal.advance(
                    (start_frame - self.observed_through_frame) as f64 / self.sample_rate as f64,
                    None,
                );
            }
            self.model = TemporalExpectation::new(self.model.step_sec);
            self.energy.history.fill([0.0; 3]);
            self.energy.mean = [0.0; 3];
            self.energy.step_index = 0;
            self.energy
                .history_prediction
                .clear_pending(start_frame < self.observed_through_frame);
            self.energy.temporal = temporal;
            self.energy_learning.reset();
            self.dorsal = DorsalStream::new(self.sample_rate as f32);
            self.detector = OnsetDetector::default();
            self.filled = 0;
            self.observed_through_frame = start_frame;
        }
        self.next_input_frame = Some(start_frame + audio.len() as u64);
        while !audio.is_empty() {
            let take = audio.len().min(self.window.len() - self.filled);
            self.window[self.filled..self.filled + take].copy_from_slice(&audio[..take]);
            self.filled += take;
            audio = &audio[take..];
            if self.filled == self.window.len() {
                self.dorsal.process(&self.window);
                let metrics = self.dorsal.last_metrics();
                let drive = (metrics.flux.max(0.0) * 500.0).tanh().clamp(0.0, 1.0);
                let onset = self.detector.process(self.model.step_sec, drive);
                let energies = [metrics.e_low, metrics.e_mid, metrics.e_high];
                self.energy_learning
                    .observe(self.model.step_index, energies);
                self.energy.observe(energies, |_| {});
                let mut band = 0;
                if onset.fired {
                    let mut largest = 0;
                    for k in 1..3 {
                        if energies[k] > energies[largest] {
                            largest = k;
                        }
                    }
                    band = 1 + largest as u8;
                }
                let (prediction, gain_bits) = self.model.observe(band);
                // Issue on the observation clock, independently of queries and reports.
                let recurrence = self.model.energy_forecast(&self.energy);
                self.energy_learning
                    .issue(self.model.step_index, recurrence, energies);
                let start_frame = self.observed_through_frame;
                self.observed_through_frame += self.window.len() as u64;
                report(TemporalObservation {
                    start_frame,
                    end_frame: self.observed_through_frame,
                    band,
                    band_energy: energies,
                    prediction,
                    gain_bits,
                });
                self.filled = 0;
            }
        }
    }

    /// The queried frame is a future window position, never a retrospective estimate.
    #[cfg(test)]
    pub(crate) fn predict_at_frame(&mut self, frame: u64) -> Option<Prediction> {
        if frame < self.next_input_frame? {
            return None;
        }
        let ahead_frames = frame.checked_sub(self.observed_through_frame)?;
        let ahead = ahead_frames / self.window.len() as u64;
        if ahead + 2 >= self.model.min_delay as u64 {
            return None;
        }
        Some(self.model.predict_at(ahead as usize))
    }

    pub(crate) fn forecast(&mut self) -> Option<TemporalForecast> {
        let mut forecast = TemporalForecast {
            observed_history: Some(self.energy.temporal.snapshot()),
            energy_prediction_model: "shared_recurrence_mix",
            start_frame: self.observed_through_frame,
            available_through_frame: self.next_input_frame?,
            step_frames: self.window.len() * FORECAST_STRIDE,
            len: FORECAST_LEN,
            contrast: [[0.0; 3]; FORECAST_LEN],
            band_energy: [[0.0; 3]; FORECAST_LEN],
            known_energy_windows: [0; (FORECAST_LEN * FORECAST_STRIDE).div_ceil(64)],
            energy_recurrence_weight: [[1.0; 3]; FORECAST_LEN],
            background_energy: self.energy.mean,
            period_step_sec: self.model.step_sec,
            period_weights: [0.0; MAX_DELAY + 1],
            recurrence_support: if self.model.recent_onsets >= 1.0 {
                (1.0 - 2.0 * self.model.weights[0]).clamp(0.0, 1.0)
            } else {
                0.0
            },
        };
        for lag in self.model.min_delay..=self.model.max_delay {
            if self.model.step_index >= (3 * lag + 2) as u64 {
                forecast.period_weights[lag] = self.model.weights[lag];
            }
        }
        let issued = &self.energy_learning.pending
            [(self.model.step_index % ENERGY_PENDING_LEN as u64) as usize];
        if issued.step != Some(self.model.step_index) {
            return None;
        }
        forecast.energy_recurrence_weight = issued.weight;
        for (ahead, row) in forecast.contrast[..forecast.len].iter_mut().enumerate() {
            let prediction = self.model.predict_at(ahead * FORECAST_STRIDE);
            forecast.band_energy[ahead] = std::array::from_fn(|band| {
                let weight = issued.weight[ahead][band];
                weight * issued.recurrence[ahead][band] + (1.0 - weight) * issued.persistence[band]
            });
            for (k, value) in row.iter_mut().enumerate() {
                let p = prediction.probability[k + 1];
                let base = prediction.null_probability[k + 1];
                *value = (p - base) / (p + base + 0.01);
            }
        }
        forecast.refresh_energy_support();
        Some(forecast)
    }

    /// Issue local candidates; their matching actual outcomes fit the next mixture.
    pub(crate) fn use_external_energy(
        &self,
        forecast: &mut TemporalForecast,
        external: &mut BandEnergyHistory,
    ) {
        assert_eq!(forecast.start_frame, self.observed_through_frame);
        assert_eq!(external.step_index, self.model.step_index);
        let history = external.temporal.snapshot();
        forecast.observed_history = Some(history);
        forecast.energy_prediction_model = "local_history_mix";
        let latest = if external.step_index > 0 {
            external.history[((external.step_index - 1) % HISTORY as u64) as usize]
        } else {
            [0.0; 3]
        };
        let raw_recurrence = self.model.energy_forecast(external);
        let recurrence = std::array::from_fn(|ahead| {
            std::array::from_fn(|band| {
                let weight = forecast.energy_recurrence_weight[ahead][band];
                weight * raw_recurrence[ahead][band] + (1.0 - weight) * latest[band]
            })
        });
        let candidate = external.history_prediction.forecast(latest, &history);
        (forecast.band_energy, _) = external.history_prediction.issue_comparison(
            external.step_index,
            recurrence,
            candidate,
            self.next_input_frame,
        );
        forecast.background_energy = external.mean;
        forecast.refresh_energy_support();
    }

    /// Energy-only diagnostic; neither recurrence scratch nor comparison learning is advanced.
    pub(crate) fn preview_external_energy(
        &self,
        external: &BandEnergyHistory,
    ) -> Option<TemporalForecast> {
        if external.step_index != self.model.step_index {
            return None;
        }
        let issued = &self.energy_learning.pending
            [(self.model.step_index % ENERGY_PENDING_LEN as u64) as usize];
        if issued.step != Some(self.model.step_index) {
            return None;
        }
        let history = external.temporal.snapshot();
        let latest = if external.step_index > 0 {
            external.history[((external.step_index - 1) % HISTORY as u64) as usize]
        } else {
            [0.; 3]
        };
        let raw_recurrence = self.model.energy_forecast(external);
        let recurrence = std::array::from_fn(|ahead| {
            std::array::from_fn(|band| {
                let weight = issued.weight[ahead][band];
                weight * raw_recurrence[ahead][band] + (1. - weight) * latest[band]
            })
        });
        let candidate = external.history_prediction.forecast(latest, &history);
        let (band_energy, _) = external
            .history_prediction
            .preview_comparison(recurrence, candidate);
        let mut forecast = TemporalForecast {
            observed_history: Some(history),
            energy_prediction_model: "local_history_mix_readonly",
            start_frame: self.observed_through_frame,
            available_through_frame: self.next_input_frame?,
            step_frames: self.window.len() * FORECAST_STRIDE,
            len: FORECAST_LEN,
            contrast: [[0.; 3]; FORECAST_LEN],
            band_energy,
            known_energy_windows: [0; (FORECAST_LEN * FORECAST_STRIDE).div_ceil(64)],
            energy_recurrence_weight: issued.weight,
            background_energy: external.mean,
            period_step_sec: self.model.step_sec,
            period_weights: [0.; MAX_DELAY + 1],
            recurrence_support: 0.,
        };
        forecast.refresh_energy_support();
        Some(forecast)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn energy_forecast_preserves_all_horizons_at_readiness_and_onset_boundaries() {
        for dt in [0.005, 0.01, 0.02] {
            let mut model = TemporalExpectation::new(dt);
            let mut history = BandEnergyHistory::new(dt, 0);
            history.mean = [0., 0.03, 1.];
            for (i, bands) in history.history.iter_mut().enumerate() {
                *bands = [
                    0.0001 * (i % 7) as f32,
                    0.03 * (i % 11) as f32,
                    (i % 13) as f32,
                ];
            }
            let ready = (3 * model.min_delay + 2) as u64;
            for step in [
                0,
                ready - 1,
                ready,
                ready + 1,
                (3 * model.max_delay + 2) as u64,
                100_003,
            ] {
                model.step_index = step;
                history.step_index = step;
                for recent in [0., f32::from_bits(1.0_f32.to_bits() - 1), 1., 1.25] {
                    model.recent_onsets = recent;
                    let actual = model.energy_forecast(&history);
                    for (lead, bands) in actual.iter().enumerate() {
                        let expected = model.energy_at(&history, lead * FORECAST_STRIDE);
                        assert_eq!(
                            bands.map(f32::to_bits),
                            expected.map(f32::to_bits),
                            "dt={dt} step={step} recent={recent} lead={lead}"
                        );
                    }
                    if step >= ready && recent >= 1. {
                        assert!(actual.windows(2).any(|w| w[0] != w[1]));
                    }
                }
            }
        }
    }

    #[test]
    fn centered_energy_uses_window_centers_and_stops_at_support_edges() {
        let mut forecast =
            TemporalForecast::energy_fixture(1000, 7, |t| [(100. * t) as f32, 2., 3.]);
        forecast.len = 3;
        for (sample, fraction, expected) in [
            (7, 0., 5.),
            (12, 0., 5.),
            (17, 0., 5.5),
            (22, 0., 6.),
            (22, 0.5, 6.05),
            (56, 0.999, 9.),
        ] {
            assert!((forecast.centered_energy(sample, fraction).unwrap() - expected).abs() < 1e-12);
        }
        assert_eq!(forecast.centered_energy(6, 0.999), None);
        assert_eq!(forecast.centered_energy(57, 0.), None);
        for fraction in [-0.1, 1., f64::NAN, f64::INFINITY] {
            assert_eq!(forecast.centered_energy(22, fraction), None);
        }
        assert_eq!(
            forecast.energy_window_after(17),
            Some((17, 27, [1., 2., 3.]))
        );
        forecast.available_through_frame = 18;
        assert_eq!(forecast.centered_energy(26, 0.999), None);
        assert_eq!(forecast.centered_energy(27, 0.), Some(7.));
        assert_eq!(forecast.centered_energy(32, 0.), Some(7.));
    }

    #[test]
    fn centered_energy_does_not_bridge_unknown_windows_or_erase_known_silence() {
        for invalid in [f32::NAN, f32::INFINITY, -1.] {
            let mut forecast =
                TemporalForecast::energy_fixture(1000, 7, |t| [(100. * t) as f32, 2., 3.]);
            forecast.len = 3;
            forecast.band_energy[1][0] = invalid;
            forecast.refresh_energy_support();
            assert_eq!(forecast.centered_energy(16, 0.999), Some(5.));
            for sample in 17..47 {
                assert_eq!(forecast.centered_energy(sample, 0.), None);
            }
            assert_eq!(forecast.centered_energy(47, 0.), Some(9.));
            let points = forecast.external_footprint([7, 57], 57);
            assert_eq!(points.horizon_intersection, Some([7, 57]));
            assert_eq!(points.continuous_support, Some(false));
            assert!(
                points
                    .points
                    .iter()
                    .flatten()
                    .any(|p| p.band_energy_sum.is_none())
            );
            assert!(
                points
                    .points
                    .iter()
                    .flatten()
                    .any(|p| p.band_energy_sum.is_some())
            );
        }
        let silent = TemporalForecast::energy_fixture(1000, 0, |_| [0.; 3]);
        assert_eq!(silent.centered_energy(100, 0.5), Some(0.));
    }

    #[test]
    fn external_footprint_clips_time_without_claiming_energy_mass_or_empty_silence() {
        let forecast = TemporalForecast::energy_fixture(1000, 7, |_| [1., 2., 3.]);
        let points = forecast.external_footprint([0, 8000], 4007);
        assert_eq!(points.horizon_intersection, Some([7, 4007]));
        assert_eq!(points.continuous_support, Some(true));
        assert_eq!(points.omitted_prefix_samples, Some(7));
        assert_eq!(points.omitted_tail_samples, Some(3993));
        for (i, point) in points.points.into_iter().enumerate() {
            let point = point.unwrap();
            assert_eq!(point.sample, 132 + i as u64 * 250);
            assert_eq!(point.sample_fraction, 0.);
            assert_eq!(point.band_energy_sum, Some(6.));
        }
        let partial = forecast.external_footprint([37, 103], 4007);
        assert_eq!(
            (
                partial.points[0].unwrap().sample,
                partial.points[0].unwrap().sample_fraction
            ),
            (39, 0.0625)
        );
        assert_eq!(
            (
                partial.points[15].unwrap().sample,
                partial.points[15].unwrap().sample_fraction
            ),
            (100, 0.9375)
        );
        let full = forecast.external_footprint([0, 8000], u64::MAX);
        assert_eq!(full.horizon_intersection, Some([7, 4017]));
        for requested in [[5000, 8000], [0, 7], [50, 50], [50, 49]] {
            let empty = forecast.external_footprint(requested, 4007);
            assert_eq!(empty.horizon_intersection, None);
            assert_eq!(empty.continuous_support, None);
            assert_eq!(empty.omitted_prefix_samples, None);
            assert_eq!(empty.omitted_tail_samples, None);
            assert!(empty.points.iter().all(Option::is_none));
        }
    }

    #[test]
    fn footprint_support_detects_unsampled_gaps_across_bitset_word_boundaries() {
        let mut forecast = TemporalForecast::energy_fixture(1000, 0, |_| [1.; 3]);
        forecast.band_energy[1][0] = f32::NAN;
        forecast.refresh_energy_support();
        let result = forecast.external_footprint([0, 4000], 4000);
        assert!(
            result
                .points
                .iter()
                .flatten()
                .all(|point| point.band_energy_sum.is_some())
        );
        assert_eq!(result.continuous_support, Some(false));
        for bad in [1, 31, 32, 63, 64, 127, 128, 199, 200] {
            forecast.band_energy.fill([1.; 3]);
            forecast.band_energy[bad][2] = -1.;
            forecast.refresh_energy_support();
            for requested in [
                [0, 4000],
                [640, 1280],
                [1260, 1290],
                [190, 200],
                [4000, 4010],
            ] {
                let expected = (requested[0] / 10..requested[1] / 10)
                    .all(|window| forecast.scalar_window(window).is_some());
                assert_eq!(
                    forecast
                        .external_footprint(requested, 4010)
                        .continuous_support,
                    Some(expected),
                    "invalid row {bad}, request {requested:?}"
                );
            }
        }
    }

    #[test]
    fn centered_energy_preserves_fractional_samples_at_large_clock_origins() {
        let origin = u64::MAX - 10000;
        let near = TemporalForecast::energy_fixture(1000, 0, |t| [t as f32, 0., 0.]);
        let far = TemporalForecast::energy_fixture(1000, origin, |t| [t as f32, 0., 0.]);
        for offset in [0, 9, 10, 17, 2000, 4009, 4010] {
            assert_eq!(
                near.centered_energy(offset, 0.125),
                far.centered_energy(origin + offset, 0.125)
            );
        }
        let a = near.external_footprint([37, 103], 4000);
        let b = far.external_footprint([origin + 37, origin + 103], origin + 4000);
        for (a, b) in a.points.into_iter().zip(b.points) {
            let (a, b) = (a.unwrap(), b.unwrap());
            assert_eq!(a.sample, b.sample - origin);
            assert_eq!(a.sample_fraction, b.sample_fraction);
            assert_eq!(a.band_energy_sum, b.band_energy_sum);
        }
    }

    #[test]
    fn centered_energy_midpoint_means_match_an_affine_reference_at_three_resolutions() {
        let forecast = TemporalForecast::energy_fixture(1000, 7, |t| [(100. * t) as f32, 2., 3.]);
        // E(t) = 5 + (t - 12) / 10 between supported window centers.
        let expected = 5. + ((107. + 807.) / 2. - 12.) / 10.;
        for count in [8, 16, 32] {
            let mean = (0..count)
                .map(|i| {
                    let at = 107. + (i as f64 + 0.5) * 700. / count as f64;
                    forecast
                        .centered_energy(at.floor() as u64, at.fract())
                        .unwrap()
                })
                .sum::<f64>()
                / count as f64;
            assert!(
                (mean - expected).abs() < 1e-12,
                "{count}: {mean} != {expected}"
            );
        }
    }

    #[test]
    fn energy_results_update_only_the_matching_lead_and_preserve_issued_weights() {
        let mut learning = EnergyForecastLearning::new();
        learning.issue(0, [[1.0; 3]; FORECAST_LEN], [0.0; 3]);
        let frozen = learning.pending[0].clone();
        learning.observe(0, [0.0, 0.25, 1.0]);
        learning.issue(1, [[1.0; 3]; FORECAST_LEN], [0.0; 3]);
        assert_eq!(learning.pending[1].weight[0], [0.5, 0.625, 1.0]);
        assert_eq!(learning.pending[1].weight[1], [1.0; 3]);
        learning.observe(1, [0.0, 0.25, 1.0]);
        learning.issue(2, [[1.0; 3]; FORECAST_LEN], [0.0; 3]);
        learning.observe(2, [1.0, 0.75, 0.0]);
        learning.issue(3, [[1.0; 3]; FORECAST_LEN], [0.0; 3]);
        assert_eq!(learning.pending[3].weight[1], [1.0, 0.9375, 0.75]);
        assert_eq!(learning.pending[3].weight[0], [0.5, 0.5625, 0.75]);
        assert_eq!(learning.pending[0].weight, frozen.weight);
        learning.reset();
        learning.observe(4, [100.0; 3]);
        learning.issue(5, [[1.0; 3]; FORECAST_LEN], [0.0; 3]);
        assert_eq!(learning.pending[5].weight, [[1.0; 3]; FORECAST_LEN]);
    }

    #[test]
    fn delayed_energy_fit_matches_a_batch_objective_across_ring_wraps() {
        let mut learning = EnergyForecastLearning::new();
        let observations: Vec<[f32; 3]> = (0..1000)
            .map(|i| [(i % 11) as f32 * 0.07, (i % 17) as f32 * 0.02, 0.0])
            .collect();
        let mut archive = Vec::new();
        for (step, actual) in observations.iter().copied().enumerate() {
            learning.observe(step as u64, actual);
            let recurrence = std::array::from_fn(|lead| {
                [(step % 13) as f32 * 0.04 + lead as f32 * 0.001, 0.03, 0.0]
            });
            learning.issue((step + 1) as u64, recurrence, actual);
            archive.push((step + 1, recurrence, actual));
        }
        let issued = &learning.pending[1000 % ENERGY_PENDING_LEN];
        for lead in [0, 1, 20, 100, 200] {
            for band in 0..3 {
                let pairs: Vec<_> = archive
                    .iter()
                    .map(|(step, recurrence, persistence)| {
                        let target = step + lead * FORECAST_STRIDE;
                        (
                            recurrence[lead][band] as f64,
                            persistence[band] as f64,
                            observations.get(target).map(|actual| actual[band] as f64),
                        )
                    })
                    .collect();
                let loss = |weight: f64| {
                    pairs
                        .iter()
                        .map(|(r, p, y)| match y {
                            Some(y) => (weight * r + (1.0 - weight) * p - y).powi(2),
                            None => ((1.0 - weight) * (p - r)).powi(2),
                        })
                        .sum::<f64>()
                };
                let selected = issued.weight[lead][band] as f64;
                let selected_loss = loss(selected);
                for grid in 0..=1000 {
                    assert!(selected_loss <= loss(grid as f64 / 1000.0) + 1e-9);
                }
            }
        }
        assert_eq!(
            issued.weight[0][2], 1.0,
            "equal forecasts identify no mixture weight"
        );
    }

    #[test]
    fn unobserved_candidates_regularize_changes_without_fabricating_labels() {
        let mut learning = EnergyForecastLearning::new();
        learning.issue(0, [[1.0; 3]; FORECAST_LEN], [0.0; 3]);
        learning.observe(0, [0.0; 3]);
        let cross = learning.residual_cross;
        learning.issue(1, [[100.0; 3]; FORECAST_LEN], [0.0; 3]);
        assert_eq!(learning.residual_cross, cross);
        assert!(learning.pending[1].weight[0][0] > 0.999);
        assert_eq!(learning.pending[1].weight[1], [1.0; 3]);
        // Changing physical units must not introduce a new fitting timescale.
        let mut rescaled = EnergyForecastLearning::new();
        rescaled.issue(0, [[0.01; 3]; FORECAST_LEN], [0.0; 3]);
        rescaled.observe(0, [0.0; 3]);
        rescaled.issue(1, [[1.0; 3]; FORECAST_LEN], [0.0; 3]);
        assert_eq!(rescaled.pending[1].weight, learning.pending[1].weight);
    }

    #[test]
    fn continuous_energy_can_correct_a_forecast_without_new_onsets() {
        let mut observer = AcousticTemporalExpectation::new(24_000).unwrap();
        let mut last = None;
        for window in 0..200 {
            let audio: [f32; 240] = std::array::from_fn(|i| {
                0.2 * (std::f32::consts::TAU * 180.0 * (window * 240 + i) as f32 / 24_000.0).sin()
            });
            observer.process((window * 240) as u64, &audio, |row| last = Some(row));
        }
        let forecast = observer.forecast().unwrap();
        let raw = observer.model.energy_at(&observer.energy, 0);
        let issued = forecast.band_energy[0];
        let mut actual = None;
        let audio: [f32; 240] = std::array::from_fn(|i| {
            0.2 * (std::f32::consts::TAU * 180.0 * (48_000 + i) as f32 / 24_000.0).sin()
        });
        observer.process(48_000, &audio, |row| actual = Some(row));
        assert_eq!(last.unwrap().band, 0);
        let actual = actual.unwrap();
        assert_eq!(actual.band, 0);
        let mse = |pred: [f32; 3]| {
            pred.into_iter()
                .zip(actual.band_energy)
                .map(|(p, y)| (p - y).powi(2))
                .sum::<f32>()
        };
        assert!(mse(issued) < mse(raw), "{} vs {}", mse(issued), mse(raw));
        assert_eq!(
            forecast.band_energy[0], issued,
            "issued forecasts stay frozen"
        );
    }

    #[test]
    fn outcome_window_uses_the_observation_grid_and_rejects_uncovered_time() {
        let forecast = TemporalForecast::energy_fixture(1000, 7, |_| [0.25; 3]);
        assert_eq!(forecast.energy_window_after(7), Some((7, 17, [0.25; 3])));
        assert_eq!(forecast.energy_window_after(8), Some((17, 27, [0.25; 3])));
        assert_eq!(forecast.energy_window_after(17), Some((17, 27, [0.25; 3])));
        assert!(forecast.energy_window_after(6).is_none());
        assert!(forecast.energy_window_after(5000).is_none());
    }

    #[test]
    fn acoustic_period_reference_follows_recurrence_and_releases_after_silence_or_gap() {
        let fs = 48_000;
        let mut observer = AcousticTemporalExpectation::new(fs).unwrap();
        assert!(observer.forecast().is_none());
        for hop in 0..7000 {
            let now = hop * 480;
            let audio: [f32; 480] = std::array::from_fn(|i| {
                if hop >= 4000 {
                    return 0.0;
                }
                let period = if hop < 2000 { 24_000 } else { 28_800 };
                let age = ((now + i) % period) as f32 / fs as f32;
                0.5 * (std::f32::consts::TAU * 180.0 * age).sin() * (-age / 0.012).exp()
            });
            observer.process(now as u64, &audio, |_| {});
            if hop == 1999 || hop == 3999 {
                let forecast = observer.forecast().unwrap();
                let (period, support) = forecast.period_reference(0.5).unwrap();
                let expected = if hop == 1999 { 0.5 } else { 0.6 };
                assert!((period - expected).abs() < 0.021, "{period} vs {expected}");
                assert!(support > 0.5, "{support}");
                assert_eq!(
                    forecast.period_reference(0.5),
                    observer.forecast().unwrap().period_reference(0.5)
                );
            }
        }
        assert!(observer.forecast().unwrap().period_reference(0.5).is_none());
        observer.process(80 * fs as u64, &[0.0; 480], |_| {});
        assert!(observer.forecast().unwrap().period_reference(0.5).is_none());
    }

    #[test]
    fn long_forecast_repeats_observed_cycles_and_fades_without_changing_learning() {
        let mut model = TemporalExpectation::new(0.01);
        let mut control = TemporalExpectation::new(0.01);
        let mut energy = BandEnergyHistory::new(0.01, 0);
        for step in 0..5000 {
            let band = u8::from(step % 50 == 0);
            assert_eq!(model.observe(band), control.observe(band));
            energy.observe([band as f32, 0.0, 0.0], |_| {});
        }
        for ahead in [0, 50, 100, 300] {
            let pulse = model.predict_at(ahead);
            let between = model.predict_at(ahead + 20);
            assert!(pulse.probability[1] > between.probability[1]);
            assert!((pulse.probability.iter().sum::<f32>() - 1.0).abs() < 1e-5);
            assert!(model.energy_at(&energy, ahead)[0] > model.energy_at(&energy, ahead + 20)[0]);
        }
        let distant = model.predict_at(100_000);
        for (p, base) in distant
            .probability
            .into_iter()
            .zip(distant.null_probability)
        {
            assert!((p - base).abs() < 1e-5);
        }
        assert!((model.energy_at(&energy, 100_000)[0] - energy.mean[0]).abs() < 1e-5);
        assert_eq!(model.observe(1), control.observe(1));
    }

    #[test]
    fn sustained_forecast_tail_uses_background_and_excludes_self() {
        let fs = 48_000;
        let mut observer = AcousticTemporalExpectation::new(fs).unwrap();
        for hop in 0..200 {
            let now = hop * 480;
            let audio: [f32; 480] = std::array::from_fn(|i| {
                (std::f32::consts::TAU * 180.0 * (now + i) as f32 / fs as f32).sin() * 0.2
            });
            observer.process(now as u64, &audio, |_| {});
        }
        let mut forecast = observer.forecast().unwrap();
        assert!(forecast.sustained_energy_at(10.0 * fs as f64).unwrap()[0] > 0.0);
        observer.use_external_energy(
            &mut forecast,
            &mut BandEnergyHistory::new(0.01, observer.model.step_index),
        );
        for seconds in [2.0, 3.0, 5.0, 10.0] {
            assert_eq!(
                forecast.sustained_energy_at(seconds * fs as f64),
                Some([0.0; 3])
            );
        }
        assert!(forecast.sustained_energy_at(f64::NAN).is_none());
        assert!(forecast.sustained_energy_at(0.0).is_none());
    }

    #[test]
    fn external_observation_lookup_rejects_future_off_grid_and_expired_windows() {
        let mut observer = AcousticTemporalExpectation::new(1000).unwrap();
        observer.process(0, &[0.0; 13], |_| {});
        let mut own = OwnSoundHistory::new(&observer, 13);
        assert_eq!(own.observed_external_energy(0, 10), Some([0.0; 3]));
        assert!(own.observed_external_energy(10, 20).is_none());
        assert!(own.observed_external_energy(1, 10).is_none());
        assert!(own.observed_external_energy(0, 0).is_none());
        let silent = [0.0; 10];
        for hop in 0..1300 {
            own.process(13 + hop * 10, &silent, &silent, &[0.125; 10], |_, _, _| {});
        }
        assert!(own.observed_external_energy(0, 10).is_none());
        let end = own.observed_through_frame();
        assert!(own.observed_external_energy(end - 11, end - 1).is_none());
        assert!(own.observed_external_energy(end - 10, end).unwrap()[0] > 0.01);
        assert!(own.observed_external_energy(end, end + 10).is_none());
    }

    #[test]
    fn matches_keep_actual_frames_across_partial_birth_and_large_chunks() {
        let origin = 1007;
        let mut observer = AcousticTemporalExpectation::new(1000).unwrap();
        observer.process(origin, &[0.125; 13], |_| {});
        let mut whole = OwnSoundHistory::new(&observer, origin + 13);
        let mut split = OwnSoundHistory::new(&observer, origin + 13);
        let shared = observer.forecast().unwrap();
        for own in [&mut whole, &mut split] {
            let mut forecast = shared;
            observer.use_external_energy(&mut forecast, &mut own.external);
        }
        observer.process(origin + 13, &[0.25; 2], |_| {});
        for own in [&mut whole, &mut split] {
            own.process(origin + 13, &[0.0; 2], &[0.0; 2], &[0.25; 2], |_, _, _| {});
            let mut forecast = observer.forecast().unwrap();
            observer.use_external_energy(&mut forecast, &mut own.external);
        }
        let mut a = Vec::new();
        whole.process(
            origin + 15,
            &[0.0; 105],
            &[0.0; 105],
            &[0.25; 105],
            |start, window, m| {
                a.push((start, window, serde_json::to_value(m).unwrap()));
            },
        );
        let mut b = Vec::new();
        for start in (0..105).step_by(3) {
            let len = (105 - start).min(3);
            split.process(
                origin + 15 + start as u64,
                &[0.0; 3][..len],
                &[0.0; 3][..len],
                &[0.25; 3][..len],
                |start, window, m| {
                    b.push((start, window, serde_json::to_value(m).unwrap()));
                },
            );
        }
        assert_eq!(a, b);
        assert_eq!(
            a.iter().map(|x| x.0).collect::<Vec<_>>(),
            [origin + 10, origin + 110]
        );
        assert!(a.iter().all(|x| x.1 == 10
            && x.2["issued_step"] == 1
            && x.2["requested_frame"] == origin + 13));
    }

    #[test]
    fn self_sound_born_mid_window_preserves_external_history_routing_and_chunking() {
        let birth = 1536;
        for to_habitat in [false, true] {
            let mut observer = AcousticTemporalExpectation::new(48_000).unwrap();
            let mut reference = AcousticTemporalExpectation::new(48_000).unwrap();
            let mut own = None;
            let mut fragmented = None;
            for hop in 0..200 {
                let now = hop * 512;
                if now == birth {
                    own = Some(OwnSoundHistory::new(&observer, now));
                    fragmented = Some(OwnSoundHistory::new(&observer, now));
                    assert_eq!(
                        own.as_ref().unwrap().external.history,
                        reference.energy.history
                    );
                }
                let external: [f32; 512] = std::array::from_fn(|i| {
                    0.2 * (std::f32::consts::TAU * 440.0 * (now + i as u64) as f32 / 48_000.0).sin()
                });
                let body = external.map(|x| if now >= birth { -x } else { 0.0 });
                let routed = body.map(|x| if to_habitat { x } else { 0.0 });
                let mix: [f32; 512] = std::array::from_fn(|i| external[i] + routed[i]);
                if let (Some(own), Some(fragmented)) = (own.as_mut(), fragmented.as_mut()) {
                    own.process(now, &body, &routed, &mix, |_, _, _| {});
                    for start in (0..512).step_by(17) {
                        let end = (start + 17).min(512);
                        fragmented.process(
                            now + start as u64,
                            &body[start..end],
                            &routed[start..end],
                            &mix[start..end],
                            |_, _, _| {},
                        );
                    }
                    assert_eq!(own.external.history, fragmented.external.history);
                    assert_eq!(
                        own.external.temporal.snapshot(),
                        fragmented.external.temporal.snapshot()
                    );
                    assert_eq!(own.external.mean, fragmented.external.mean);
                    assert_eq!(own.profile, fragmented.profile);
                }
                observer.process(now, &mix, |_| {});
                reference.process(now, &external, |_| {});
                if let Some(own) = &own {
                    assert_eq!(own.external.history, reference.energy.history);
                    assert_eq!(
                        own.external.temporal.snapshot(),
                        reference.energy.temporal.snapshot()
                    );
                    assert_eq!(own.external.mean, reference.energy.mean);
                }
            }
            assert!(own.unwrap().profile.iter().sum::<f32>() > 0.001);
        }
    }

    #[test]
    fn energy_projection_is_linear_in_observed_energy_histories() {
        let mut model = TemporalExpectation::new(0.01);
        let mut own = BandEnergyHistory::new(0.01, 0);
        let mut other = BandEnergyHistory::new(0.01, 0);
        let mut total = BandEnergyHistory::new(0.01, 0);
        for tick in 0..2500 {
            let a = if tick % 50 < 4 {
                [0.02, 0.001, 0.0]
            } else {
                [0.0; 3]
            };
            let b = if tick % 63 < 5 {
                [0.0, 0.03, 0.01]
            } else {
                [0.0; 3]
            };
            own.observe(a, |_| {});
            other.observe(b, |_| {});
            total.observe(std::array::from_fn(|k| a[k] + b[k]), |_| {});
            model.observe(if tick % 50 == 0 {
                1
            } else if tick % 63 == 0 {
                2
            } else {
                0
            });
        }
        for ahead in 0..20 {
            let a = model.energy_at(&own, ahead);
            let b = model.energy_at(&other, ahead);
            let all = model.energy_at(&total, ahead);
            for k in 0..3 {
                assert!((a[k] + b[k] - all[k]).abs() < 1e-7);
                assert!(all[k].is_finite() && all[k] >= 0.0);
            }
        }
    }

    #[test]
    fn local_energy_keeps_a_coherent_neighbor_when_self_cancels_or_reinforces_it() {
        for phase in [1.0, -1.0] {
            let mut observer = AcousticTemporalExpectation::new(48_000).unwrap();
            let mut own = OwnSoundHistory::new(&observer, 0);
            let mut neighbor_dorsal = DorsalStream::new(48_000.0);
            let mut neighbor = BandEnergyHistory::new(0.01, 0);
            for hop in 0..200 {
                let body: [f32; 480] = std::array::from_fn(|i| {
                    0.2 * (std::f32::consts::TAU * 440.0 * (hop * 480 + i) as f32 / 48_000.0).sin()
                });
                let external = body.map(|x| phase * x);
                let mix = std::array::from_fn::<_, 480, _>(|i| body[i] + external[i]);
                own.process((hop * 480) as u64, &body, &body, &mix, |_, _, _| {});
                neighbor_dorsal.process(&external);
                let m = neighbor_dorsal.last_metrics();
                neighbor.observe([m.e_low, m.e_mid, m.e_high], |_| {});
                observer.process((hop * 480) as u64, &mix, |_| {});
            }
            let mut forecast = observer.forecast().unwrap();
            let shared = forecast;
            let scratch = observer.model.predictions;
            let preview = observer.preview_external_energy(&own.external).unwrap();
            assert_eq!(observer.model.predictions, scratch);
            assert!(own.external.history_prediction.take_errors().is_none());
            for _ in 0..3 {
                assert_eq!(
                    observer
                        .preview_external_energy(&own.external)
                        .unwrap()
                        .band_energy,
                    preview.band_energy
                );
            }
            assert_eq!(observer.model.predictions, scratch);
            assert!(own.external.history_prediction.take_errors().is_none());
            observer.use_external_energy(&mut forecast, &mut own.external);
            assert_eq!(preview.band_energy, forecast.band_energy);
            assert_eq!(preview.known_energy_windows, forecast.known_energy_windows);
            assert_eq!(
                own.external
                    .history_prediction
                    .take_errors()
                    .unwrap()
                    .issued,
                1
            );
            assert_eq!(forecast.contrast, shared.contrast);
            assert_eq!(forecast.period_weights, shared.period_weights);
            assert_eq!(
                forecast.energy_recurrence_weight,
                shared.energy_recurrence_weight
            );
            let start = forecast.observed_frame();
            let footprint = forecast.external_footprint([start, start + 48000], start + 192000);
            assert!(
                footprint
                    .points
                    .iter()
                    .flatten()
                    .all(|point| point.band_energy_sum.unwrap() > 0.)
            );
            if phase == -1. {
                let mixed = shared.external_footprint([start, start + 48000], start + 192000);
                assert!(
                    mixed
                        .points
                        .iter()
                        .flatten()
                        .all(|point| point.band_energy_sum == Some(0.))
                );
            }
            assert!(
                forecast
                    .band_energy
                    .iter()
                    .all(|bands| bands.iter().all(|energy| *energy > 0.0))
            );
            for band in 0..3 {
                assert!(
                    (forecast.background_energy[band] - neighbor.mean[band]).abs() < 1e-7,
                    "phase {phase}, band {band}: local {} vs neighbor {}",
                    forecast.background_energy[band],
                    neighbor.mean[band]
                );
            }
        }
    }

    #[test]
    fn a_solitary_sound_has_no_predicted_competitor_after_self_exclusion() {
        let mut observer = AcousticTemporalExpectation::new(48_000).unwrap();
        let mut own = OwnSoundHistory::new(&observer, 0);
        for hop in 0..1500 {
            let audio: [f32; 480] = std::array::from_fn(|i| {
                let age = ((hop % 50) * 480 + i) as f32 / 48_000.0;
                0.5 * (std::f32::consts::TAU * 180.0 * age).sin() * (-age / 0.018).exp()
            });
            own.process(hop as u64 * 480, &audio, &audio, &audio, |_, _, _| {});
            observer.process(hop as u64 * 480, &audio, |_| {});
            let mut forecast = observer.forecast().unwrap();
            observer.use_external_energy(&mut forecast, &mut own.external);
            assert!(
                forecast.band_energy[..forecast.len]
                    .iter()
                    .all(|row| *row == [0.0; 3])
            );
        }
    }

    #[test]
    fn observations_cannot_change_their_own_forecast_or_retrofit_lookahead() {
        let mut a = TemporalExpectation::new(0.01);
        let mut b = TemporalExpectation::new(0.01);
        for tick in 0..2500 {
            let band = if tick % 100 == 0 {
                1
            } else if tick % 100 == 75 {
                2
            } else {
                0
            };
            let _ = a.predict_at(12);
            assert_eq!(a.observe(band), b.observe(band));
        }
        let (left, _) = a.observe(0);
        let (right, _) = b.observe(3);
        assert_eq!(left, right);
        assert_ne!(a.weights, b.weights);
    }

    #[test]
    fn fixed_memory_retains_a_multi_position_relation_across_ring_wraps() {
        let mut model = TemporalExpectation::new(0.01);
        let mut gain = 0.0;
        for tick in 0..12_000 {
            let phase = tick % 137;
            let band = match phase {
                0 => 1,
                18 | 91 => 2,
                53 => 3,
                _ => 0,
            };
            let (prediction, delta) = model.observe(band);
            if tick > 10_000 {
                gain += delta;
            }
            assert!((prediction.probability.iter().sum::<f32>() - 1.0).abs() < 2e-5);
            assert!(
                prediction
                    .probability
                    .iter()
                    .all(|p| p.is_finite() && *p >= 0.0 && *p <= 1.0)
            );
        }
        assert!(gain > 120.0, "predictive gain across ring wraps: {gain}");
        let prediction = model.predict_at(0);
        assert!((prediction.leading_period_sec.unwrap() - 1.37).abs() < 0.03);
        assert!(std::mem::size_of::<TemporalExpectation>() < 28_000);
    }

    #[test]
    fn silence_and_long_gaps_do_not_supply_period_evidence() {
        let mut model = TemporalExpectation::new(0.01);
        for _ in 0..1500 {
            let (predicted, gain) = model.observe(0);
            assert!(predicted.leading_period_sec.is_none());
            assert!(gain.abs() < 2e-5);
        }
        for i in 0..1500 {
            model.observe(u8::from(i % 50 == 0));
        }
        assert!(model.predict_at(0).leading_period_sec.is_some());
        for _ in 0..2400 {
            model.observe(0);
        }
        assert!(model.predict_at(0).leading_period_sec.is_none());
    }

    #[test]
    fn recurrence_can_be_observed_at_three_temporal_resolutions() {
        for step in [0.005, 0.01, 0.02] {
            let mut model = TemporalExpectation::new(step);
            let period = (0.7 / step).round() as usize;
            let mut gain = 0.0;
            for i in 0..(20.0 / step) as usize {
                let (_, delta) = model.observe(u8::from(i % period == 0));
                if i as f32 * step > 12.0 {
                    gain += delta;
                }
            }
            assert!(gain > 10.0);
            let period = model.predict_at(0).leading_period_sec.unwrap();
            assert!((period - 0.7).abs() < 0.04 || (period - 1.4).abs() < 0.04);
        }
    }

    #[test]
    fn acoustic_observation_is_independent_of_delivery_chunks_and_causal() {
        for fs in [44_100, 48_000] {
            let audio: Vec<f32> = (0..fs * 8 + 137)
                .map(|i| {
                    let t = i as f32 / fs as f32;
                    let age = t % 0.5;
                    let hz = if ((t * 2.0) as usize).is_multiple_of(2) {
                        180.0
                    } else {
                        1800.0
                    };
                    (std::f32::consts::TAU * hz * age).sin() * (-age / 0.012).exp() * 0.7
                })
                .collect();
            let mut expected = Vec::new();
            let mut reference = AcousticTemporalExpectation::new(fs).unwrap();
            reference.process(0, &audio, |r| expected.push(r));
            let expected_forecast = reference.forecast().unwrap();
            for chunk_size in [1, 257, 512, 4093] {
                let mut observer = AcousticTemporalExpectation::new(fs).unwrap();
                let mut actual = Vec::new();
                for (index, chunk) in audio.chunks(chunk_size).enumerate() {
                    observer.process((index * chunk_size) as u64, chunk, |r| actual.push(r));
                    if chunk_size == 4093 {
                        let _ = observer.forecast();
                    }
                }
                assert_eq!(actual, expected);
                assert!(observer.predict_at_frame(audio.len() as u64).is_some());
                assert!(
                    observer
                        .predict_at_frame(audio.len() as u64 + fs as u64)
                        .is_none()
                );
                let forecast = observer.forecast().unwrap();
                assert_eq!(forecast.band_energy, expected_forecast.band_energy);
                assert_eq!(
                    forecast.observed_history,
                    expected_forecast.observed_history
                );
                assert_eq!(
                    forecast.energy_recurrence_weight,
                    expected_forecast.energy_recurrence_weight
                );
                assert!(forecast.contrast_at(audio.len() as f64 - 1.0).is_none());
                assert!(forecast.contrast_at(audio.len() as f64).is_some());
                assert!(
                    forecast
                        .contrast_at(audio.len() as f64 + fs as f64)
                        .is_some()
                );
                assert!(
                    forecast
                        .contrast_at(audio.len() as f64 + 5.0 * fs as f64)
                        .is_none()
                );
            }
            let mut prefix = Vec::new();
            let end = fs as usize * 4 + 7;
            AcousticTemporalExpectation::new(fs)
                .unwrap()
                .process(0, &audio[..end], |r| prefix.push(r));
            assert_eq!(prefix, expected[..prefix.len()]);
            assert!(expected.iter().filter(|r| r.band == 1).count() >= 7);
            assert!(expected.iter().filter(|r| r.band == 2).count() >= 7);
        }
    }

    #[test]
    fn missing_audio_resets_evidence_without_manufacturing_silent_observations() {
        let mut observer = AcousticTemporalExpectation::new(48_000).unwrap();
        assert!(observer.predict_at_frame(0).is_none());
        let audio = vec![0.25; 1000];
        observer.process(0, &audio, |_| {});
        assert!(observer.predict_at_frame(970).is_none());
        let mut after_gap = Vec::new();
        observer.process(240_000, &audio, |r| after_gap.push(r));
        let mut fresh = Vec::new();
        let mut reference = AcousticTemporalExpectation::new(48_000).unwrap();
        reference.process(240_000, &audio, |r| fresh.push(r));
        assert_eq!(after_gap, fresh);
        let actual = observer.forecast().unwrap();
        let expected = reference.forecast().unwrap();
        assert_eq!(actual.band_energy, expected.band_energy);
        let before_gap = actual.observed_history.unwrap();
        let fresh_history = expected.observed_history.unwrap();
        assert!(before_gap.known_band_rms_by_age[5][0] > fresh_history.known_band_rms_by_age[5][0]);
        assert!(before_gap.known_coverage_by_age[5] > fresh_history.known_coverage_by_age[5]);
        assert_eq!(
            actual.energy_recurrence_weight,
            expected.energy_recurrence_weight
        );
        assert_eq!(after_gap[0].start_frame, 240_000);
        assert!(
            after_gap
                .iter()
                .all(|r| r.prediction.leading_period_sec.is_none())
        );
        assert!(observer.predict_at_frame(1000).is_none());
        assert!(AcousticTemporalExpectation::new(49).is_none());
        assert!(AcousticTemporalExpectation::new(50).is_some());
    }

    #[cfg(feature = "profile-alloc")]
    #[test]
    fn prediction_and_observation_allocate_nothing_after_construction() {
        let mut model = TemporalExpectation::new(0.01);
        let mut acoustic = AcousticTemporalExpectation::new(48_000).unwrap();
        let audio = [0.125; 512];
        let mut own = OwnSoundHistory::new(&acoustic, 0);
        let mut matches = 0;
        crate::runtime_profile::begin_allocations();
        for i in 0..3000 {
            std::hint::black_box(model.observe(u8::from(i % 50 == 0)));
            std::hint::black_box(model.predict_at(12));
            own.process(i * 512, &audio, &audio, &audio, |_, _, matched| {
                matches += 1;
                std::hint::black_box(matched.issued_features.unwrap());
            });
            acoustic.process(i * 512, &audio, |r| {
                std::hint::black_box(r);
            });
            std::hint::black_box(acoustic.predict_at_frame((i + 1) * 512));
            let mut forecast = acoustic.forecast().unwrap();
            acoustic.use_external_energy(&mut forecast, &mut own.external);
            std::hint::black_box(forecast);
        }
        let allocations = crate::runtime_profile::finish_allocations().unwrap();
        assert_eq!(allocations.count, 0);
        assert_eq!(allocations.bytes, 0);
        assert!(matches > 0);
    }
}
