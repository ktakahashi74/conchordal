//! Passive waveform-periodicity observations; no source identity or closure claim.

use std::collections::VecDeque;
use std::sync::Arc;

use rustfft::{Fft, FftPlanner, num_complex::Complex32};

use crate::core::float::unit_gaussian;

const MEMORY: usize = 128;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ContourScore {
    pub(crate) delta_log2: Option<f32>,
    pub(crate) gain_bits: Option<f32>,
    pub(crate) loss_bits: Option<f32>,
    pub(crate) context_support: f32,
    pub(crate) error_threshold_bits: Option<f32>,
    pub(crate) calibration_events: usize,
    pub(crate) error_candidate: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ContourEvent {
    pub(crate) kind: &'static str,
    pub(crate) available_sec: f64,
    pub(crate) onset_sec: Option<f64>,
    pub(crate) periodic_frequency_hz: Option<f32>,
    pub(crate) periodicity: Option<f32>,
    pub(crate) score: Option<ContourScore>,
    pub(crate) gap_threshold_sec: Option<f64>,
    pub(crate) missing_start_sec: Option<f64>,
}

#[derive(Clone, Copy)]
struct Transition {
    context: [f32; 2],
    context_len: usize,
    delta: f32,
}

struct ContourPredictor {
    history: VecDeque<Transition>,
    losses: VecDeque<f32>,
    context: [f32; 2],
    context_len: usize,
    previous_pitch: Option<f32>,
}

impl ContourPredictor {
    fn new() -> Self {
        Self {
            history: VecDeque::with_capacity(MEMORY),
            losses: VecDeque::with_capacity(MEMORY),
            context: [0.0; 2],
            context_len: 0,
            previous_pitch: None,
        }
    }

    fn step(&mut self, frequency: Option<f32>) -> ContourScore {
        // The context weights and threshold precede the current observation.
        let mut kernels = [0.0; MEMORY];
        let mut support = 0.0;
        for (kernel, row) in kernels.iter_mut().zip(&self.history) {
            if row.context_len == 2 && self.context_len == 2 {
                let distance = (row.context[0] - self.context[0]).powi(2)
                    + (row.context[1] - self.context[1]).powi(2);
                *kernel = unit_gaussian(distance.sqrt(), 0.04);
                support += *kernel;
            }
        }
        let eligible = support >= 2.0;
        let threshold = if eligible && self.losses.len() >= 16 {
            let mut sorted = [0.0; MEMORY];
            for (slot, value) in sorted.iter_mut().zip(&self.losses) {
                *slot = *value;
            }
            let sorted = &mut sorted[..self.losses.len()];
            sorted.sort_unstable_by(f32::total_cmp);
            let index = (sorted.len() - 1) as f32 * 0.95;
            let low = index.floor() as usize;
            let high = index.ceil() as usize;
            Some(sorted[low] + (sorted[high] - sorted[low]) * index.fract() + 2.0)
        } else {
            None
        };
        let mut score = ContourScore {
            context_support: support,
            error_threshold_bits: threshold,
            calibration_events: self.losses.len(),
            ..Default::default()
        };
        let Some(frequency) = frequency else {
            self.previous_pitch = None;
            self.context_len = 0;
            return score;
        };
        assert!(frequency.is_finite() && frequency > 0.0);
        let pitch = frequency.log2();
        if let Some(previous) = self.previous_pitch {
            let delta = pitch - previous;
            let prior_weight = if self.history.is_empty() { 1.0 } else { 0.05 };
            let normalizer = std::f32::consts::TAU.sqrt();
            let prior = prior_weight * unit_gaussian(delta, 0.8) / (0.8 * normalizer);
            let mut conditional = prior;
            let mut marginal = prior;
            let strength = support / (support + 4.0);
            let base = 0.95 / self.history.len().max(1) as f32;
            for (kernel, row) in kernels.iter().zip(&self.history) {
                let weight = if support > 0.0 {
                    (1.0 - strength) * base + strength * 0.95 * kernel / support
                } else {
                    base
                };
                let density = unit_gaussian(delta - row.delta, 0.025) / (0.025 * normalizer);
                conditional += weight * density;
                marginal += base * density;
            }
            let conditional = conditional.max(f32::MIN_POSITIVE);
            let marginal = marginal.max(f32::MIN_POSITIVE);
            let loss = -conditional.log2();
            score.delta_log2 = Some(delta);
            score.loss_bits = Some(loss);
            score.gain_bits = Some((conditional / marginal).log2());
            score.error_candidate = threshold.is_some_and(|threshold| loss > threshold);
            if self.history.len() == MEMORY {
                self.history.pop_front();
            }
            self.history.push_back(Transition {
                context: self.context,
                context_len: self.context_len,
                delta,
            });
            if self.context_len < 2 {
                self.context[self.context_len] = delta;
                self.context_len += 1;
            } else {
                self.context = [self.context[1], delta];
            }
            if eligible {
                if self.losses.len() == MEMORY {
                    self.losses.pop_front();
                }
                self.losses.push_back(loss);
            }
        }
        self.previous_pitch = Some(pitch);
        score
    }
}

pub(crate) struct ContourObserver {
    fs: f32,
    hop: usize,
    window: Vec<f32>,
    cursor: usize,
    fft: Arc<dyn Fft<f32>>,
    inverse: Arc<dyn Fft<f32>>,
    spectrum: Vec<Complex32>,
    scratch: Vec<Complex32>,
    energy: Vec<f32>,
    similarity: Vec<f32>,
    periods: Vec<f32>,
    block_count: usize,
    block_energy: f64,
    next_sample: Option<u64>,
    active: bool,
    quiet: usize,
    await_silence: bool,
    onset: Option<u64>,
    last_onset: Option<u64>,
    published: bool,
    stable: VecDeque<(f32, f32)>,
    intervals: VecDeque<f32>,
    gap_sent: bool,
    predictor: ContourPredictor,
}

impl ContourObserver {
    pub(crate) fn new(fs: f32) -> Option<Self> {
        if !fs.is_finite() || fs < 2400.0 {
            return None;
        }
        let window = (2048.0 * fs / 24_000.0).round() as usize;
        let nfft = (2 * window).next_power_of_two();
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(nfft);
        let inverse = planner.plan_fft_inverse(nfft);
        let scratch = fft
            .get_inplace_scratch_len()
            .max(inverse.get_inplace_scratch_len());
        Some(Self {
            fs,
            hop: (fs / 100.0).round() as usize,
            window: vec![0.0; window],
            cursor: 0,
            fft,
            inverse,
            spectrum: vec![Complex32::default(); nfft],
            scratch: vec![Complex32::default(); scratch],
            energy: vec![0.0; window + 1],
            similarity: vec![0.0; window / 2],
            periods: Vec::with_capacity(window / (fs / 1200.0) as usize),
            block_count: 0,
            block_energy: 0.0,
            next_sample: None,
            active: false,
            quiet: 0,
            await_silence: false,
            onset: None,
            last_onset: None,
            published: false,
            stable: VecDeque::with_capacity(3),
            intervals: VecDeque::with_capacity(8),
            gap_sent: false,
            predictor: ContourPredictor::new(),
        })
    }

    pub(crate) fn process(
        &mut self,
        start_sample: u64,
        audio: &[f32],
        mut emit: impl FnMut(ContourEvent),
    ) {
        assert!(
            audio.iter().all(|x| x.is_finite()),
            "nonfinite presentation audio"
        );
        assert!(
            self.next_sample.is_none_or(|next| start_sample >= next),
            "reordered presentation audio"
        );
        if audio.is_empty() {
            return;
        }
        if let Some(next) = self.next_sample
            && next != start_sample
        {
            self.window.fill(0.0);
            self.cursor = 0;
            self.block_count = 0;
            self.block_energy = 0.0;
            self.active = false;
            self.quiet = 0;
            self.await_silence = true;
            self.onset = None;
            self.last_onset = None;
            self.stable.clear();
            self.intervals.clear();
            self.gap_sent = false;
            self.predictor.step(None);
            emit(ContourEvent {
                kind: "input_gap",
                available_sec: start_sample as f64 / self.fs as f64,
                missing_start_sec: Some(next as f64 / self.fs as f64),
                ..Default::default()
            });
        }
        for (index, sample) in audio.iter().copied().enumerate() {
            self.window[self.cursor] = sample;
            self.cursor = (self.cursor + 1) % self.window.len();
            self.block_energy += (sample as f64).powi(2);
            self.block_count += 1;
            if self.block_count != self.hop {
                continue;
            }
            let end = start_sample + index as u64 + 1;
            let now = end as f64 / self.fs as f64;
            let rms = (self.block_energy / self.hop as f64).sqrt();
            self.block_energy = 0.0;
            self.block_count = 0;
            self.quiet = if rms < 0.003 {
                (self.quiet + 1).min(2)
            } else {
                0
            };
            if self.await_silence {
                if self.quiet == 2 {
                    self.await_silence = false;
                }
                continue;
            }
            if !self.active && rms >= 0.005 {
                let onset = end - self.hop as u64;
                if let Some(previous) = self.last_onset {
                    if self.intervals.len() == 8 {
                        self.intervals.pop_front();
                    }
                    self.intervals.push_back((onset - previous) as f32);
                }
                self.last_onset = Some(onset);
                self.onset = Some(onset);
                self.active = true;
                self.published = false;
                self.gap_sent = false;
                self.stable.clear();
            }
            if self.active {
                let onset = self.onset.expect("an active episode has an onset");
                let mut observation = None;
                if self.quiet == 2 {
                    if !self.published {
                        observation = Some(None);
                    }
                    self.active = false;
                } else if !self.published && rms >= 0.003 && end - onset >= self.window.len() as u64
                {
                    if let Some((frequency, periodicity)) = self.periodic_estimate() {
                        if self.stable.len() == 3 {
                            self.stable.pop_front();
                        }
                        self.stable.push_back((frequency.log2(), periodicity));
                        if self.stable.len() == 3 {
                            let mut pitches =
                                [self.stable[0].0, self.stable[1].0, self.stable[2].0];
                            let pitch = median(&mut pitches);
                            if pitches[2] - pitches[0] < 0.025 {
                                let periodicity =
                                    self.stable.iter().map(|p| p.1).fold(1.0, f32::min);
                                observation = Some(Some((pitch.exp2(), periodicity)));
                            }
                        }
                    } else {
                        self.stable.clear();
                    }
                }
                if let Some(estimate) = observation {
                    let frequency = estimate.map(|p| p.0);
                    let score = self.predictor.step(frequency);
                    self.published = true;
                    emit(ContourEvent {
                        kind: if estimate.is_some() {
                            "periodic_episode"
                        } else {
                            "unresolved_episode"
                        },
                        available_sec: now,
                        onset_sec: Some(onset as f64 / self.fs as f64),
                        periodic_frequency_hz: frequency,
                        periodicity: estimate.map(|p| p.1),
                        score: Some(score),
                        ..Default::default()
                    });
                }
            }
            if self.intervals.len() >= 4 && !self.active && !self.gap_sent {
                let mut intervals = [0.0; 8];
                for (slot, value) in intervals.iter_mut().zip(&self.intervals) {
                    *slot = *value;
                }
                let threshold = 2.5 * median(&mut intervals[..self.intervals.len()]);
                let onset = self.last_onset.expect("intervals have an onset");
                if (end - onset) as f64 >= threshold as f64 {
                    self.gap_sent = true;
                    emit(ContourEvent {
                        kind: "silence_gap",
                        available_sec: now,
                        onset_sec: Some(onset as f64 / self.fs as f64),
                        gap_threshold_sec: Some(threshold as f64 / self.fs as f64),
                        ..Default::default()
                    });
                }
            }
        }
        self.next_sample = Some(start_sample + audio.len() as u64);
    }

    fn periodic_estimate(&mut self) -> Option<(f32, f32)> {
        let n = self.window.len();
        let mean = self.window.iter().sum::<f32>() / n as f32;
        self.energy[0] = 0.0;
        self.spectrum.fill(Complex32::default());
        for i in 0..n {
            let x = self.window[(self.cursor + i) % n] - mean;
            self.energy[i + 1] = self.energy[i] + x * x;
            self.spectrum[i].re = x;
        }
        if self.energy[n] / (n as f32) < 1e-6 {
            return None;
        }
        self.fft
            .process_with_scratch(&mut self.spectrum, &mut self.scratch);
        for bin in &mut self.spectrum {
            *bin = Complex32::new(bin.norm_sqr(), 0.0);
        }
        self.inverse
            .process_with_scratch(&mut self.spectrum, &mut self.scratch);
        for lag in 1..self.similarity.len() {
            let denominator = self.energy[n - lag] + self.energy[n] - self.energy[lag];
            self.similarity[lag] =
                2.0 * self.spectrum[lag].re / (self.spectrum.len() as f32 * denominator.max(1e-15));
        }
        let lo = (self.fs / 1200.0) as usize;
        let hi = ((self.fs / 100.0) as usize).min(n / 2);
        let peaks = |lag: &usize| {
            self.similarity[*lag] >= self.similarity[*lag - 1]
                && self.similarity[*lag] > self.similarity[*lag + 1]
        };
        let best = (lo + 1..hi - 1)
            .filter(peaks)
            .map(|lag| self.similarity[lag])
            .fold(0.0, f32::max);
        if best < 0.9 {
            return None;
        }
        let lag = (lo + 1..hi - 1)
            .filter(peaks)
            .find(|lag| self.similarity[*lag] >= (0.95 * best).max(0.9))?;
        let center = self.similarity[lag];
        let left = self.similarity[lag - 1];
        let right = self.similarity[lag + 1];
        let period = lag as f32 + 0.5 * (left - right) / (left - 2.0 * center + right);
        self.periods.clear();
        let multiples = ((self.similarity.len() - 1) as f32 / period) as usize;
        for multiple in 1..=multiples {
            let target = period * multiple as f32;
            let low = target.floor() as usize;
            let high = (low + 1).min(self.similarity.len() - 1);
            let interpolated = self.similarity[low]
                + (self.similarity[high] - self.similarity[low]) * target.fract();
            if interpolated < 0.9 {
                return None;
            }
            let index = target.round() as usize;
            if index < 3 || index + 2 >= self.similarity.len() {
                continue;
            }
            let mut peak = index - 1;
            for candidate in index..=index + 1 {
                if self.similarity[candidate] > self.similarity[peak] {
                    peak = candidate;
                }
            }
            let left = self.similarity[peak - 1];
            let center = self.similarity[peak];
            let right = self.similarity[peak + 1];
            let curvature = left - 2.0 * center + right;
            if curvature >= 0.0 {
                return None;
            }
            let shift = 0.5 * (left - right) / curvature;
            if shift.abs() > 1.0 {
                return None;
            }
            self.periods.push((peak as f32 + shift) / multiple as f32);
        }
        Some((self.fs / median(&mut self.periods), center))
    }
}

fn median(values: &mut [f32]) -> f32 {
    assert!(!values.is_empty());
    values.sort_unstable_by(f32::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        0.5 * (values[middle - 1] + values[middle])
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tones(fs: f32, count: usize) -> Vec<f32> {
        let mut audio = vec![0.0; ((0.6 + count as f32 * 0.36 + 1.2) * fs).round() as usize];
        for index in 0..count {
            let start = ((0.6 + index as f32 * 0.36) * fs).round() as usize;
            let n = (0.22 * fs).round() as usize;
            let frequency =
                220.0 * [0.0_f32, 0.2, 0.5, 0.32, 0.71, 0.61, 0.12, 0.42][index % 8].exp2();
            for i in 0..n {
                let t = i as f32 / fs;
                let envelope = (t / 0.012).min(1.0) * ((n as f32 / fs - t) / 0.025).min(1.0);
                audio[start + i] = 0.2 * envelope * (std::f32::consts::TAU * frequency * t).sin();
            }
        }
        audio
    }

    #[test]
    fn periodicity_is_a_waveform_feature_across_sample_rates() {
        assert!(ContourObserver::new(2000.0).is_none());
        assert!(ContourObserver::new(f32::NAN).is_none());
        for fs in [24_000.0, 44_100.0, 48_000.0, 96_000.0] {
            let mut observer = ContourObserver::new(fs).unwrap();
            for amplitudes in [[1.0, 0.0, 0.0], [1.0, 0.55, 0.3], [0.0, 1.0, 0.8]] {
                for (i, x) in observer.window.iter_mut().enumerate() {
                    let phase = std::f32::consts::TAU * 261.3 * i as f32 / fs;
                    *x = amplitudes
                        .iter()
                        .enumerate()
                        .map(|(k, a)| a * ((k + 1) as f32 * phase).sin())
                        .sum();
                }
                let (frequency, _) = observer
                    .periodic_estimate()
                    .expect("coherent periodic signal");
                assert!(
                    (1200.0 * (frequency / 261.3).log2()).abs() < 0.1,
                    "fs={fs}, f={frequency}"
                );
            }
            for frequency in [233.0, 220.0 * 2.0_f32.sqrt()] {
                for (i, x) in observer.window.iter_mut().enumerate() {
                    let phase = std::f32::consts::TAU * i as f32 / fs;
                    *x = (220.0 * phase).sin() + (frequency * phase).sin();
                }
                assert!(
                    observer.periodic_estimate().is_none(),
                    "fs={fs}, interferer={frequency}"
                );
            }
        }
    }

    #[test]
    fn observation_is_chunk_invariant_and_does_not_flush_an_ending() {
        for fs in [24_000.0, 44_100.0, 48_000.0] {
            let audio = tones(fs, 12);
            let mut whole = Vec::new();
            ContourObserver::new(fs)
                .unwrap()
                .process(0, &audio, |event| whole.push(event));
            assert_eq!(
                whole
                    .iter()
                    .filter(|event| event.kind == "periodic_episode")
                    .count(),
                12
            );
            let mut observer = ContourObserver::new(fs).unwrap();
            let mut pieces = Vec::new();
            for (index, chunk) in audio.chunks(511).enumerate() {
                observer.process((index * 511) as u64, chunk, |event| pieces.push(event));
            }
            assert_eq!(pieces, whole);
            let cutoff = (2.867 * fs) as usize;
            let mut prefix = Vec::new();
            ContourObserver::new(fs)
                .unwrap()
                .process(0, &audio[..cutoff], |event| prefix.push(event));
            let expected: Vec<_> = whole
                .into_iter()
                .filter(|event| event.available_sec <= cutoff as f64 / fs as f64)
                .collect();
            assert_eq!(prefix, expected);
        }
    }

    #[test]
    fn missing_audio_breaks_context_without_inventing_silence_or_reentry() {
        let fs = 24_000.0;
        let mut audio = tones(fs, 20);
        let missing = (3.51 * fs) as usize;
        let resumed = (5.09 * fs) as usize;
        let mut observer = ContourObserver::new(fs).unwrap();
        let mut events = Vec::new();
        observer.process(0, &audio[..missing], |event| events.push(event));
        observer.process(resumed as u64, &audio[resumed..], |event| {
            events.push(event)
        });
        assert_eq!(
            events
                .iter()
                .filter(|event| event.kind == "input_gap")
                .count(),
            1
        );
        assert!(
            !events
                .iter()
                .any(|event| event.kind == "silence_gap" && event.available_sec <= 5.3)
        );
        let post = events
            .iter()
            .find(|event| event.available_sec > 5.09 && event.score.is_some())
            .unwrap();
        assert!((post.onset_sec.unwrap() - 5.28).abs() < 1e-8);
        assert!(post.score.unwrap().delta_log2.is_none());
        assert!(!post.score.unwrap().error_candidate);
        audio[missing..resumed].fill(0.0);
        let mut silence = Vec::new();
        ContourObserver::new(fs)
            .unwrap()
            .process(0, &audio, |event| silence.push(event));
        assert!(
            silence
                .iter()
                .any(|event| event.kind == "silence_gap" && event.available_sec < 5.09)
        );
    }

    #[test]
    fn prediction_uses_previous_context_and_bounded_supported_calibration() {
        let contour = [0.0_f32, 0.2, 0.5, 0.32, 0.71, 0.61, 0.12, 0.42];
        let mut a = ContourPredictor::new();
        let mut b = ContourPredictor::new();
        for index in 0..51 {
            let f = 220.0 * contour[index % 8].exp2();
            a.step(Some(f));
            b.step(Some(f));
        }
        let normal = a.step(Some(220.0 * contour[3].exp2()));
        let changed = b.step(Some(220.0 * contour[4].exp2()));
        assert_eq!(normal.context_support, changed.context_support);
        assert_eq!(normal.error_threshold_bits, changed.error_threshold_bits);
        assert!(!normal.error_candidate);
        assert!(changed.error_candidate);
        for index in 0..350 {
            a.step(Some(220.0 * contour[index % 8].exp2()));
        }
        assert_eq!(a.history.len(), MEMORY);
        assert_eq!(a.losses.len(), MEMORY);
        assert_eq!(a.history.capacity(), MEMORY);
        assert_eq!(a.losses.capacity(), MEMORY);
        a.step(None);
        let count = a.history.len();
        assert!(a.step(Some(440.0)).gain_bits.is_none());
        assert_eq!(a.history.len(), count);
    }

    #[test]
    #[cfg(feature = "profile-alloc")]
    fn streaming_observation_allocates_nothing_after_construction() {
        let audio = tones(48_000.0, 350);
        let mut observer = ContourObserver::new(48_000.0).unwrap();
        let mut count = 0;
        crate::runtime_profile::begin_allocations();
        observer.process(0, &audio, |_| count += 1);
        let allocations = crate::runtime_profile::finish_allocations().unwrap();
        assert!(count >= 350);
        assert_eq!(allocations.count, 0);
        assert_eq!(allocations.bytes, 0);
    }

    #[test]
    #[ignore = "Replays a frozen Python acoustic campaign selected by CONCHORDAL_CONTOUR_ASSAY_ROOT"]
    fn replay_frozen_audio_campaign() {
        use serde_json::{Value, json};
        use std::fs;
        let root = std::path::PathBuf::from(
            std::env::var("CONCHORDAL_CONTOUR_ASSAY_ROOT").expect("assay root"),
        );
        let manifest: Value =
            serde_json::from_str(&fs::read_to_string(root.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["status"], "complete");
        let mut rows = Vec::new();
        for case in manifest["cases"].as_array().unwrap() {
            let folder = root.join(case["directory"].as_str().unwrap());
            let expected_path = if folder.join("coherent.json").exists() {
                folder.join("coherent.json")
            } else {
                folder.join("observations.json")
            };
            let expected: Value =
                serde_json::from_str(&fs::read_to_string(expected_path).unwrap()).unwrap();
            let mut wav = hound::WavReader::open(folder.join("audio.wav")).unwrap();
            assert_eq!(wav.spec().channels, 1);
            let fs = wav.spec().sample_rate as f32;
            let audio: Vec<f32> = wav
                .samples::<i16>()
                .map(|s| s.unwrap() as f32 / 32768.0)
                .collect();
            let mut events = Vec::new();
            ContourObserver::new(fs)
                .unwrap()
                .process(0, &audio, |event| events.push(event));
            let notes: Vec<_> = events
                .iter()
                .filter(|event| event.score.is_some())
                .collect();
            let expected_notes = expected["notes"].as_array().unwrap();
            assert_eq!(notes.len(), expected_notes.len(), "{}", folder.display());
            let mut max_error = 0.0_f64;
            for (note, expected) in notes.iter().zip(expected_notes) {
                assert!(
                    (note.onset_sec.unwrap() - expected["onset_sec"].as_f64().unwrap()).abs()
                        < 1e-6
                );
                assert!(
                    (note.available_sec - expected["available_sec"].as_f64().unwrap()).abs()
                        < 0.01001
                );
                if let Some(frequency) = expected["frequency_hz"].as_f64() {
                    let error = (1200.0
                        * (note.periodic_frequency_hz.unwrap() as f64 / frequency).log2())
                    .abs();
                    max_error = max_error.max(error);
                    assert!(error < 0.1, "{}: {error} cents", folder.display());
                } else {
                    assert!(note.periodic_frequency_hz.is_none());
                }
            }
            if folder.join("predictions.json").exists() {
                let predictions: Value = serde_json::from_str(
                    &fs::read_to_string(folder.join("predictions.json")).unwrap(),
                )
                .unwrap();
                for (note, expected) in notes.iter().zip(predictions.as_array().unwrap()) {
                    let score = note.score.unwrap();
                    assert_eq!(
                        score.error_candidate,
                        expected["error_candidate"].as_bool().unwrap(),
                        "{}: candidate",
                        folder.display()
                    );
                    for (actual, field) in [
                        (score.gain_bits, "gain_bits"),
                        (score.loss_bits, "loss_bits"),
                    ] {
                        match (actual, expected[field].as_f64()) {
                            (Some(a), Some(b)) => assert!(
                                (a as f64 - b).abs() < 0.005,
                                "{}: {field} {a} vs {b}",
                                folder.display()
                            ),
                            (None, None) => (),
                            _ => panic!("score availability differs"),
                        }
                    }
                }
            }
            let gaps: Vec<_> = events.iter().filter(|e| e.kind == "silence_gap").collect();
            assert_eq!(gaps.len(), expected["gaps"].as_array().unwrap().len());
            for (gap, expected) in gaps.iter().zip(expected["gaps"].as_array().unwrap()) {
                assert!(
                    (gap.available_sec - expected["available_sec"].as_f64().unwrap()).abs()
                        < 0.01001
                );
            }
            rows.push(json!({"directory": case["directory"], "notes": notes.len(), "gaps": gaps.len(), "max_pitch_difference_cents": max_error}));
        }
        let path = std::env::var("CONCHORDAL_CONTOUR_ASSAY_OUTPUT").expect("assay output path");
        fs::write(
            path,
            serde_json::to_string_pretty(&json!({"cases": rows, "status": "pass"})).unwrap(),
        )
        .unwrap();
    }
}
