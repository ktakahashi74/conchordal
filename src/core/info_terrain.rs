//! Passive bin-local observation; never an input to perception or generation.

use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};

use crossbeam_channel::{Receiver, Sender, bounded};
use serde::Serialize;

use super::{log2space::Log2Space, spectral_history::SpectralHistory};

mod efe;

const DIM: usize = 10;
const LAMBDA: f64 = 0.999;
const DELTA_L_DB: f64 = 1.0; // Registration A-2 §2.1: authored perceptual floor.
const SURPRISE_CAP: f64 = 27.631021115928547;

fn amplitude_delta() -> f64 {
    10_f64.powf(DELTA_L_DB / 20.0) - 1.0
}

#[derive(Clone, Debug, PartialEq)]
struct Predictor {
    w: [f64; DIM],
    p: [[f64; DIM]; DIM],
    amplitude_ema: Option<f64>,
    residual_mean: f64,
    residual_variance: f64,
}

impl Default for Predictor {
    fn default() -> Self {
        Self {
            w: std::array::from_fn(|i| if i == 1 { 1.0 } else { 0.0 }),
            p: std::array::from_fn(|i| std::array::from_fn(|j| if i == j { 100.0 } else { 0.0 })),
            amplitude_ema: None,
            residual_mean: 0.0,
            residual_variance: amplitude_delta().powi(2),
        }
    }
}

impl Predictor {
    fn observe(
        &mut self,
        amplitude: f64,
        previous: f64,
        window_mean: f64,
        ages: [f32; 8],
        coverage: [f32; 8],
        alpha: f64,
    ) -> (f64, f64) {
        let missing = (f64::NAN, f64::NAN);
        if !amplitude.is_finite()
            || !self.residual_mean.is_finite()
            || !self.residual_variance.is_finite()
        {
            *self = Self::default();
            return missing;
        }
        let Some(scale) = self.amplitude_ema else {
            if window_mean.is_finite() && window_mean > 0.0 {
                self.amplitude_ema = Some(window_mean);
            }
            return missing;
        };
        let mut phi = [1.0; DIM];
        phi[1] = previous / scale;
        for i in 0..8 {
            if coverage[i] >= 0.5 {
                phi[i + 2] = f64::from(ages[i]) / scale;
            }
        }
        let target = amplitude / scale;
        let prediction: f64 = self.w.iter().zip(phi).map(|(w, x)| w * x).sum();
        let error = target - prediction;
        let variance_eff = self.residual_variance.max(amplitude_delta().powi(2));
        let surprise = (error * error / (2.0 * variance_eff)).min(SURPRISE_CAP);
        let precision = 1.0 / (1.0 + variance_eff);
        if !scale.is_finite()
            || scale <= 0.0
            || !phi.iter().all(|x| x.is_finite())
            || !target.is_finite()
            || !error.is_finite()
            || !surprise.is_finite()
            || !precision.is_finite()
        {
            *self = Self::default();
            return missing;
        }
        let p_phi: [f64; DIM] =
            std::array::from_fn(|i| self.p[i].iter().zip(phi).map(|(p, x)| p * x).sum());
        let denominator = LAMBDA + phi.iter().zip(p_phi).map(|(x, p)| x * p).sum::<f64>();
        let gain = p_phi.map(|p| p / denominator);
        // Use phi^T P explicitly; symmetrization follows the registered update.
        let phi_p: [f64; DIM] =
            std::array::from_fn(|j| (0..DIM).map(|i| phi[i] * self.p[i][j]).sum());
        for i in 0..DIM {
            self.w[i] += gain[i] * error;
            for (j, value) in phi_p.iter().enumerate() {
                self.p[i][j] = (self.p[i][j] - gain[i] * value) / LAMBDA;
            }
        }
        for i in 0..DIM {
            for j in 0..i {
                let value = (self.p[i][j] + self.p[j][i]) / 2.0;
                self.p[i][j] = value;
                self.p[j][i] = value;
            }
        }
        let bad_weights = !self.w.iter().all(|x| x.is_finite());
        if bad_weights
            || !self.p.iter().flatten().all(|x| x.is_finite())
            || (0..DIM).map(|i| self.p[i][i]).sum::<f64>() > 1e6
        {
            self.p = Self::default().p;
        }
        if bad_weights {
            self.w = Self::default().w;
        }
        let delta = error - self.residual_mean;
        self.residual_mean += alpha * delta;
        self.residual_variance = (1.0 - alpha) * (self.residual_variance + alpha * delta * delta);
        self.amplitude_ema = Some(scale + alpha * (amplitude - scale));
        if !self.residual_mean.is_finite()
            || !self.residual_variance.is_finite()
            || self.amplitude_ema.is_none_or(|x| !x.is_finite())
        {
            *self = Self::default();
            return missing;
        }
        if bad_weights {
            missing
        } else {
            (surprise, precision)
        }
    }
}

struct IndicatorScans {
    surprise_scan: Vec<f64>,
    precision_scan: Vec<f64>,
}

struct InfoTerrain {
    space: Arc<Log2Space>,
    alpha: f64,
    history: Vec<f64>,
    history_len: usize,
    cursor: usize,
    frames: usize,
    predictors: Vec<Predictor>,
    mean_scan: Vec<f64>,
    soundgen_weight_scan: Vec<f64>,
    osf_weight_scan: Vec<f64>,
    versions: [IndicatorScans; 2],
}

impl InfoTerrain {
    fn new(space: Log2Space, hop_sec: f64) -> Self {
        let window = (1.0 / hop_sec).floor() as usize;
        assert!(
            window >= 2,
            "info_terrain requires at least two frames per 1 s window"
        );
        let n = space.n_bins();
        Self {
            space: Arc::new(space),
            alpha: 1.0 - (-hop_sec).exp(),
            history: vec![0.0; (window - 1) * n],
            history_len: window - 1,
            cursor: 0,
            frames: 0,
            predictors: vec![Predictor::default(); n],
            mean_scan: vec![f64::NAN; n],
            soundgen_weight_scan: vec![0.0; n],
            osf_weight_scan: vec![0.0; n],
            versions: std::array::from_fn(|_| IndicatorScans {
                surprise_scan: vec![f64::NAN; n],
                precision_scan: vec![f64::NAN; n],
            }),
        }
    }

    fn reset_learning(&mut self) {
        self.history.fill(0.0);
        self.predictors.fill(Predictor::default());
        self.cursor = 0;
        self.frames = 0;
    }

    fn observe(
        &mut self,
        power_scan: &[f32],
        age_rms_scan: &[[f32; 8]],
        coverage: [f32; 8],
        warming_up: bool,
    ) {
        self.space
            .assert_scan_len_named(power_scan, "info_power_scan");
        self.space
            .assert_scan_len_named(age_rms_scan, "age_rms_scan");
        for version in &mut self.versions {
            version.surprise_scan.fill(f64::NAN);
            version.precision_scan.fill(f64::NAN);
        }
        self.mean_scan.fill(f64::NAN);
        self.soundgen_weight_scan.fill(0.0);
        self.osf_weight_scan.fill(0.0);
        if warming_up {
            return;
        }
        let n = self.space.n_bins();
        if self.frames >= self.history_len {
            for b in 0..n {
                self.mean_scan[b] = (0..self.history_len)
                    .map(|t| self.history[t * n + b])
                    .sum::<f64>()
                    / self.history_len as f64;
            }
            let maximum = self.mean_scan.iter().copied().fold(0.0, f64::max);
            for b in 0..n {
                let amplitude = f64::from(power_scan[b]).sqrt();
                let mean = self.mean_scan[b];
                let mut minimum = f64::INFINITY;
                let mut maximum_history = f64::NEG_INFINITY;
                let mut squared = 0.0;
                for t in 0..self.history_len {
                    let a = self.history[t * n + b];
                    minimum = minimum.min(a);
                    maximum_history = maximum_history.max(a);
                    squared += (a - mean).powi(2);
                }
                self.soundgen_weight_scan[b] = maximum_history.max(0.0);
                self.osf_weight_scan[b] = maximum_history.max(amplitude);
                if maximum == 0.0 || !mean.is_finite() || mean < 1e-4 * maximum {
                    continue;
                }
                let variance = squared / self.history_len as f64;
                self.versions[0].precision_scan[b] = 1.0 / (1.0 + variance / (mean * mean));
                if maximum_history - minimum > 0.0 {
                    let sd_eff = variance.sqrt().max(amplitude_delta() * mean);
                    let z = (amplitude - mean) / sd_eff;
                    if !z.is_nan() {
                        self.versions[0].surprise_scan[b] = (0.5 * z * z).min(SURPRISE_CAP);
                    }
                }
                let previous_index = (self.cursor + self.history_len - 1) % self.history_len;
                let (surprise, precision) = self.predictors[b].observe(
                    amplitude,
                    self.history[previous_index * n + b],
                    mean,
                    age_rms_scan[b],
                    coverage,
                    self.alpha,
                );
                self.versions[1].surprise_scan[b] = surprise;
                self.versions[1].precision_scan[b] = precision;
            }
        }
        for (b, power) in power_scan.iter().enumerate() {
            self.history[self.cursor * n + b] = f64::from(*power).sqrt();
        }
        self.cursor = (self.cursor + 1) % self.history_len;
        self.frames = (self.frames + 1).min(self.history_len);
    }
}

fn integrate(space: &Log2Space, surprise_scan: &[f64], weight_scan: &[f64]) -> f64 {
    space.assert_scan_len_named(surprise_scan, "surprise_scan");
    space.assert_scan_len_named(weight_scan, "weight_scan");
    if surprise_scan.iter().all(|s| !s.is_finite()) {
        return f64::NAN;
    }
    let total: f64 = weight_scan.iter().sum();
    surprise_scan
        .iter()
        .zip(weight_scan)
        .filter(|(s, _)| s.is_finite())
        .map(|(s, w)| {
            s * if total == 0.0 {
                1.0 / space.n_bins() as f64
            } else {
                w / total
            }
        })
        .sum()
}

fn weighted_quantile(values: &mut [(f64, f64)], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f64 = values.iter().map(|x| x.1).sum();
    if total <= 0.0 {
        return None;
    }
    let mut cumulative = 0.0;
    let mut previous = (values[0].0, values[0].1 / (2.0 * total));
    for &(value, weight) in values.iter() {
        let midpoint = (cumulative + weight / 2.0) / total;
        if q <= midpoint {
            if q <= previous.1 {
                return Some(previous.0);
            }
            return Some(
                previous.0 + (value - previous.0) * (q - previous.1) / (midpoint - previous.1),
            );
        }
        previous = (value, midpoint);
        cumulative += weight;
    }
    Some(previous.0)
}

#[derive(Clone, Copy, Default)]
struct Moments {
    sum: f64,
    maximum: f64,
    count: u64,
}

impl Moments {
    fn add(&mut self, value: f64) {
        if value.is_finite() {
            if self.count == 0 || value > self.maximum {
                self.maximum = value;
            }
            self.sum += value;
            self.count += 1;
        }
    }
    fn mean(self) -> Option<f64> {
        (self.count > 0).then(|| self.sum / self.count as f64)
    }
}

#[derive(Serialize)]
pub(crate) struct ScalarSummary {
    mean: Option<f64>,
    maximum: Option<f64>,
    defined_hops: u64,
    missing_hops: u64,
}

#[derive(Serialize)]
pub(crate) struct VersionSummary {
    soundgen: ScalarSummary,
    osf: ScalarSummary,
    precision_mean: Option<f64>,
    precision_defined_hops: u64,
    precision_missing_hops: u64,
    surprise_median: Option<f64>,
    surprise_p90: Option<f64>,
    missing_surprise_bin_hops: u64,
    missing_precision_bin_hops: u64,
    surprise_mean_scan: Vec<Option<f64>>,
    precision_mean_scan: Vec<Option<f64>>,
}

#[derive(Serialize)]
struct HopTrace {
    frame_id: u64,
    // Version order: original, extended. Rule order: soundgen, OSF.
    surprise: [[Option<f64>; 2]; 2],
}

#[derive(Serialize)]
pub(crate) struct Block {
    #[serde(skip)]
    space: Arc<Log2Space>,
    stream: &'static str,
    block_id: u64,
    hop_start: u64,
    hop_end_exclusive: u64,
    start_sample: u64,
    end_sample: u64,
    hops: u64,
    input_gap: bool,
    incomplete: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    efe: Option<efe::Summary>,
    original: VersionSummary,
    extended: VersionSummary,
    #[serde(skip_serializing_if = "Option::is_none")]
    trace: Option<Vec<HopTrace>>,
}

impl Block {
    pub(crate) fn assert_scan_lengths(&self) {
        for version in [&self.original, &self.extended] {
            self.space
                .assert_scan_len_named(&version.surprise_mean_scan, "surprise_mean_scan");
            self.space
                .assert_scan_len_named(&version.precision_mean_scan, "precision_mean_scan");
        }
    }
}

struct VersionAccumulator {
    scalar: [Moments; 2],
    precision: Moments,
    surprise_bins: Vec<Moments>,
    precision_bins: Vec<Moments>,
    amplitude_bins: Vec<f64>,
}

struct BlockAccumulator {
    block_id: Option<u64>,
    hop_start: u64,
    hop_end: u64,
    hops: u64,
    input_gap: bool,
    versions: [VersionAccumulator; 2],
    quantile_scratch: Vec<(f64, f64)>,
    trace: Option<Vec<HopTrace>>,
}

impl BlockAccumulator {
    fn new(n: usize, trace: bool, trace_capacity: usize) -> Self {
        Self {
            block_id: None,
            hop_start: 0,
            hop_end: 0,
            hops: 0,
            input_gap: false,
            versions: std::array::from_fn(|_| VersionAccumulator {
                scalar: [Moments::default(); 2],
                precision: Moments::default(),
                surprise_bins: vec![Moments::default(); n],
                precision_bins: vec![Moments::default(); n],
                amplitude_bins: vec![0.0; n],
            }),
            quantile_scratch: Vec::with_capacity(n),
            trace: trace.then(|| Vec::with_capacity(trace_capacity)),
        }
    }

    fn add(&mut self, terrain: &InfoTerrain, frame_id: u64, block_id: u64, gap: bool) {
        let space = &terrain.space;
        space.assert_scan_len_named(&terrain.mean_scan, "mean_scan");
        if self.block_id.is_none() {
            self.block_id = Some(block_id);
            self.hop_start = frame_id;
        }
        self.hop_end = frame_id + 1;
        self.hops += 1;
        self.input_gap |= gap;
        let mut trace = HopTrace {
            frame_id,
            surprise: [[None; 2]; 2],
        };
        for (v, (acc, scans)) in self.versions.iter_mut().zip(&terrain.versions).enumerate() {
            space.assert_scan_len_named(&scans.surprise_scan, "surprise_scan");
            space.assert_scan_len_named(&scans.precision_scan, "precision_scan");
            let scalars = [
                integrate(space, &scans.surprise_scan, &terrain.soundgen_weight_scan),
                integrate(space, &scans.surprise_scan, &terrain.osf_weight_scan),
            ];
            for (rule, scalar) in scalars.into_iter().enumerate() {
                acc.scalar[rule].add(scalar);
                trace.surprise[v][rule] = scalar.is_finite().then_some(scalar);
            }
            let mut precision_sum = 0.0;
            let mut amplitude_sum = 0.0;
            for b in 0..space.n_bins() {
                let surprise = scans.surprise_scan[b];
                let precision = scans.precision_scan[b];
                let mean = terrain.mean_scan[b];
                acc.surprise_bins[b].add(surprise);
                acc.precision_bins[b].add(precision);
                if surprise.is_finite() {
                    acc.amplitude_bins[b] += mean;
                }
                if precision.is_finite() {
                    precision_sum += mean * precision;
                    amplitude_sum += mean;
                }
            }
            if amplitude_sum > 0.0 {
                acc.precision.add(precision_sum / amplitude_sum);
            }
        }
        if let Some(rows) = &mut self.trace {
            rows.push(trace);
        }
    }

    fn finish(
        &mut self,
        space: &Arc<Log2Space>,
        stream: &'static str,
        hop: usize,
        incomplete: bool,
    ) -> Option<Block> {
        let block_id = self.block_id.take()?;
        let summaries = self.versions.each_mut().map(|acc| {
            space.assert_scan_len_named(&acc.surprise_bins, "surprise_bins_scan");
            space.assert_scan_len_named(&acc.precision_bins, "precision_bins_scan");
            space.assert_scan_len_named(&acc.amplitude_bins, "amplitude_bins_scan");
            self.quantile_scratch.clear();
            for (b, moment) in acc.surprise_bins.iter().enumerate() {
                if let Some(mean) = moment.mean() {
                    let weight = acc.amplitude_bins[b] / moment.count as f64;
                    if weight > 0.0 {
                        self.quantile_scratch.push((mean, weight));
                    }
                }
            }
            let summary = VersionSummary {
                soundgen: ScalarSummary {
                    mean: acc.scalar[0].mean(),
                    maximum: (acc.scalar[0].count > 0).then_some(acc.scalar[0].maximum),
                    defined_hops: acc.scalar[0].count,
                    missing_hops: self.hops - acc.scalar[0].count,
                },
                osf: ScalarSummary {
                    mean: acc.scalar[1].mean(),
                    maximum: (acc.scalar[1].count > 0).then_some(acc.scalar[1].maximum),
                    defined_hops: acc.scalar[1].count,
                    missing_hops: self.hops - acc.scalar[1].count,
                },
                precision_mean: acc.precision.mean(),
                precision_defined_hops: acc.precision.count,
                precision_missing_hops: self.hops - acc.precision.count,
                surprise_median: weighted_quantile(&mut self.quantile_scratch, 0.5),
                surprise_p90: weighted_quantile(&mut self.quantile_scratch, 0.9),
                missing_surprise_bin_hops: acc
                    .surprise_bins
                    .iter()
                    .map(|m| self.hops - m.count)
                    .sum(),
                missing_precision_bin_hops: acc
                    .precision_bins
                    .iter()
                    .map(|m| self.hops - m.count)
                    .sum(),
                surprise_mean_scan: acc.surprise_bins.iter().map(|m| m.mean()).collect(),
                precision_mean_scan: acc.precision_bins.iter().map(|m| m.mean()).collect(),
            };
            acc.scalar.fill(Moments::default());
            acc.precision = Moments::default();
            acc.surprise_bins.fill(Moments::default());
            acc.precision_bins.fill(Moments::default());
            acc.amplitude_bins.fill(0.0);
            summary
        });
        let [original, extended] = summaries;
        let trace = self.trace.as_mut().map(|rows| {
            let capacity = rows.capacity();
            std::mem::replace(rows, Vec::with_capacity(capacity))
        });
        let block = Block {
            space: Arc::clone(space),
            stream,
            block_id,
            hop_start: self.hop_start,
            hop_end_exclusive: self.hop_end,
            start_sample: self.hop_start * hop as u64,
            end_sample: self.hop_end * hop as u64,
            hops: self.hops,
            input_gap: self.input_gap,
            incomplete,
            efe: None,
            original,
            extended,
            trace,
        };
        self.hops = 0;
        self.input_gap = false;
        block.assert_scan_lengths();
        Some(block)
    }
}

pub(crate) struct Output {
    pub(crate) rx: Receiver<Block>,
    pub(crate) dropped: Arc<AtomicU64>,
}

pub(crate) struct Observer {
    efe: Option<efe::Diagnostic>,
    terrain: InfoTerrain,
    age_rms_scan: Vec<[f32; 8]>,
    blocks: BlockAccumulator,
    tx: Sender<Block>,
    dropped: Arc<AtomicU64>,
    deterministic: bool,
    stream: &'static str,
    sample_rate: u64,
    hop: usize,
    gap_pending: bool,
}

impl Observer {
    pub(crate) fn new(
        space: Log2Space,
        sample_rate: u32,
        hop: usize,
        stream: &'static str,
        trace: bool,
        deterministic: bool,
    ) -> (Self, Output) {
        let n = space.n_bins();
        let (tx, rx) = bounded(64);
        let dropped = Arc::new(AtomicU64::new(0));
        (
            Self {
                efe: None,
                terrain: InfoTerrain::new(space, hop as f64 / sample_rate as f64),
                age_rms_scan: vec![[0.0; 8]; n],
                blocks: BlockAccumulator::new(n, trace, (sample_rate as usize).div_ceil(hop)),
                tx,
                dropped: Arc::clone(&dropped),
                deterministic,
                stream,
                sample_rate: sample_rate as u64,
                hop,
                gap_pending: false,
            },
            Output { rx, dropped },
        )
    }

    pub(crate) fn enable_efe(&mut self, config: &crate::config::AppConfig, nfft: usize) {
        self.efe = Some(efe::Diagnostic::new(
            (*self.terrain.space).clone(),
            u64::from(self.stream == "presentation"),
            efe::effective_settings(
                config,
                &self.terrain.space,
                self.sample_rate as u32,
                self.hop,
                nfft,
            ),
        ));
    }

    pub(crate) fn observe_analysis(
        &mut self,
        frame_id: u64,
        frame: &crate::core::landscape::Landscape,
        history: &SpectralHistory,
        warming_up: bool,
        params: &crate::core::landscape::LandscapeParams,
    ) {
        self.observe(frame_id, &frame.nsgt_power, history, warming_up);
        if let Some(efe) = &mut self.efe {
            self.terrain
                .space
                .assert_scan_len_named(&self.terrain.mean_scan, "mean_scan");
            let maximum = self
                .terrain
                .mean_scan
                .iter()
                .copied()
                .filter(|x| x.is_finite())
                .max_by(f64::total_cmp);
            efe.observe(
                frame_id,
                frame,
                params,
                maximum,
                warming_up,
                self.hop as f32 / self.sample_rate as f32,
            );
        }
    }

    pub(crate) fn reset_gap(&mut self) {
        if let Some(efe) = &mut self.efe {
            efe.reset();
        }
        self.terrain.reset_learning();
        self.gap_pending = true;
    }

    pub(crate) fn observe(
        &mut self,
        frame_id: u64,
        power_scan: &[f32],
        history: &SpectralHistory,
        warming_up: bool,
    ) {
        self.terrain
            .space
            .assert_scan_len_named(power_scan, "info_power_scan");
        let block_id = frame_id * self.hop as u64 / self.sample_rate;
        if self.blocks.block_id.is_some_and(|id| id != block_id) {
            self.flush(false);
        }
        let coverage = if warming_up {
            [0.0; 8]
        } else {
            history.read_age_features(&mut self.age_rms_scan)
        };
        self.terrain
            .observe(power_scan, &self.age_rms_scan, coverage, warming_up);
        self.blocks
            .add(&self.terrain, frame_id, block_id, self.gap_pending);
        self.gap_pending = false;
    }

    fn flush(&mut self, incomplete: bool) {
        if let Some(mut block) =
            self.blocks
                .finish(&self.terrain.space, self.stream, self.hop, incomplete)
        {
            block.efe = self.efe.as_mut().map(efe::Diagnostic::finish);
            let lost = if self.deterministic {
                self.tx.send(block).is_err()
            } else {
                self.tx.try_send(block).is_err()
            };
            if lost {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

impl Drop for Observer {
    fn drop(&mut self) {
        let incomplete = self.blocks.block_id.is_some_and(|id| {
            self.blocks.hop_end * (self.hop as u64) < (id + 1) * self.sample_rate
        });
        self.flush(incomplete);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn space() -> Log2Space {
        Log2Space::new(100.0, 200.0, 1)
    }

    fn observe(terrain: &mut InfoTerrain, amplitudes: [f32; 2]) {
        terrain.observe(&amplitudes.map(|a| a * a), &[[0.0; 8]; 2], [0.0; 8], false);
    }

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-10, "{a} != {b}");
    }

    #[test]
    fn soundgen_port_population_sd_excludes_current_and_caps() {
        let mut terrain = InfoTerrain::new(space(), 0.25); // W = 4, n = 3.
        for a in [1.0, 2.0, 3.0] {
            observe(&mut terrain, [a, 2.0]);
            assert!(terrain.versions[0].surprise_scan[0].is_nan());
        }
        observe(&mut terrain, [4.0, 20.0]);
        // History [1,2,3]: m=2, population variance=2/3, S=3, CV precision=6/7.
        close(terrain.versions[0].surprise_scan[0], 3.0);
        close(terrain.versions[0].precision_scan[0], 6.0 / 7.0);
        assert!(terrain.versions[0].surprise_scan[1].is_nan());
        close(terrain.versions[0].precision_scan[1], 1.0);
        close(terrain.soundgen_weight_scan[1], 2.0);
        close(terrain.osf_weight_scan[1], 20.0);
        observe(&mut terrain, [1000.0, 2.0]);
        close(terrain.versions[0].surprise_scan[0], -(1e-12_f64).ln());
    }

    #[test]
    fn both_integration_rules_keep_missing_weight_and_zero_sum_fallback() {
        let s = space();
        // soundgen uses max(history)>0; OSF includes the current amplitude.
        close(integrate(&s, &[3.0, f64::NAN], &[3.0, 2.0]), 1.8);
        close(integrate(&s, &[3.0, f64::NAN], &[4.0, 20.0]), 0.5);
        close(integrate(&s, &[3.0, f64::NAN], &[0.0, 0.0]), 1.5);
        close(integrate(&s, &[2.0, 4.0], &[-1.0, 3.0]), 5.0);
        close(integrate(&s, &[2.0, 4.0], &[-1.0, 1.0]), 3.0);
        assert!(integrate(&s, &[f64::NAN; 2], &[0.0; 2]).is_nan());
    }

    #[test]
    fn original_floor_bounds_threshold_deviations_on_both_sides() {
        for sign in [-1.0, 1.0] {
            for factor in [0.25, 1.0, 2.0] {
                let mut terrain = InfoTerrain::new(space(), 0.25);
                let mean = 2.0 / (1.0 + sign * amplitude_delta());
                let spread = factor * amplitude_delta() * mean * 1.5_f64.sqrt();
                for (t, value) in [mean - spread, mean, mean + spread].into_iter().enumerate() {
                    terrain.history[t * 2..t * 2 + 2].fill(value);
                }
                terrain.frames = terrain.history_len;
                terrain.observe(&[4.0; 2], &[[0.0; 8]; 2], [0.0; 8], false);
                let surprise = terrain.versions[0].surprise_scan[0];
                assert!(surprise <= 0.5 + 1e-14);
                assert_eq!((surprise - 0.5).abs() < 1e-12, factor <= 1.0);
                close(surprise, 0.5 / factor.max(1.0).powi(2));
                close(
                    terrain.versions[0].precision_scan[0],
                    1.0 / (1.0 + (factor * amplitude_delta()).powi(2)),
                );
            }
        }
    }

    #[test]
    fn extended_floor_applies_only_to_evaluation_and_uses_old_variance() {
        let floor = amplitude_delta().powi(2);
        for variance in [0.0, floor / 4.0, floor * 4.0] {
            let mut p = Predictor {
                amplitude_ema: Some(1.0),
                residual_variance: variance,
                ..Default::default()
            };
            let error = amplitude_delta() / 2.0;
            let (surprise, precision) = p.observe(1.0 + error, 1.0, 1.0, [0.0; 8], [0.0; 8], 0.1);
            close(surprise, error.powi(2) / (2.0 * variance.max(floor)));
            close(precision, 1.0 / (1.0 + variance.max(floor)));
            close(p.residual_variance, 0.9 * (variance + 0.1 * error.powi(2)));
            assert_eq!(p.residual_variance < floor, variance < floor);
        }
    }

    #[test]
    fn persistence_startup_and_gap_reset_use_registered_initial_state() {
        let mut terrain = InfoTerrain::new(space(), 0.25);
        for _ in 0..4 {
            observe(&mut terrain, [2.0; 2]);
        }
        for _ in 0..8 {
            observe(&mut terrain, [2.0; 2]);
            close(terrain.versions[1].surprise_scan[0], 0.0);
            close(
                terrain.versions[1].precision_scan[0],
                1.0 / (1.0 + amplitude_delta().powi(2)),
            );
            assert_eq!(terrain.predictors[0].w, Predictor::default().w);
            close(terrain.predictors[0].residual_mean, 0.0);
        }
        assert!(terrain.predictors[0].residual_variance < amplitude_delta().powi(2));
        observe(&mut terrain, [4.0; 2]);
        assert_ne!(terrain.predictors[0].w, Predictor::default().w);
        terrain.reset_learning();
        assert_eq!(terrain.predictors, vec![Predictor::default(); 2]);
        assert_eq!(terrain.frames, 0);
    }

    #[test]
    fn initialization_retries_until_eligible_and_can_initialize_with_zero_current() {
        let mut p = Predictor::default();
        for mean in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let (s, precision) = p.observe(2.0, 1.0, mean, [0.0; 8], [0.0; 8], 0.1);
            assert!(s.is_nan() && precision.is_nan());
            assert_eq!(p, Predictor::default());
        }
        let mut terrain = InfoTerrain::new(space(), 0.25);
        for _ in 0..6 {
            observe(&mut terrain, [1.0, 0.0]);
            assert_eq!(terrain.predictors[1], Predictor::default());
        }
        observe(&mut terrain, [1.0, 3.0]);
        assert_eq!(terrain.predictors[1], Predictor::default());
        observe(&mut terrain, [1.0, 0.0]);
        let initialized = Predictor {
            amplitude_ema: Some(1.0),
            ..Default::default()
        };
        assert_eq!(terrain.predictors[1], initialized);
        assert!(terrain.versions[1].surprise_scan[1].is_nan());
        assert!(terrain.versions[1].precision_scan[1].is_nan());
        observe(&mut terrain, [1.0, 0.0]);
        close(terrain.versions[1].surprise_scan[1], 0.0);
    }

    #[test]
    fn silence_precedes_constant_history_and_warmup_does_not_learn() {
        let mut terrain = InfoTerrain::new(space(), 0.25);
        terrain.observe(&[4.0; 2], &[[0.0; 8]; 2], [0.0; 8], true);
        assert_eq!(terrain.frames, 0);
        for _ in 0..6 {
            observe(&mut terrain, [0.0, 0.0]);
            assert!(
                terrain
                    .versions
                    .iter()
                    .all(|v| v.precision_scan.iter().all(|p| p.is_nan()))
            );
            assert_eq!(terrain.predictors[0], Predictor::default());
        }
        for _ in 0..5 {
            observe(&mut terrain, [1.0, 0.00001]);
        }
        close(terrain.versions[0].precision_scan[0], 1.0);
        assert!(terrain.versions[0].surprise_scan[0].is_nan());
        assert!(terrain.versions[0].precision_scan[1].is_nan());
        assert_eq!(terrain.predictors[1], Predictor::default());
        // The registered threshold is strict: equality remains audible.
        for _ in 0..4 {
            observe(&mut terrain, [10000.0, 1.0]);
        }
        assert!(terrain.versions[0].precision_scan[1].is_finite());
    }

    #[test]
    fn extension_starts_only_after_full_window_and_init_frame_is_missing() {
        let mut terrain = InfoTerrain::new(space(), 0.25);
        for a in [1.0, 2.0, 3.0] {
            observe(&mut terrain, [a; 2]);
            assert_eq!(terrain.predictors[0], Predictor::default());
        }
        observe(&mut terrain, [4.0; 2]);
        let predictor = &terrain.predictors[0];
        close(predictor.amplitude_ema.unwrap(), 2.0);
        assert_eq!(predictor.w, Predictor::default().w);
        assert_eq!(predictor.p, Predictor::default().p);
        close(predictor.residual_mean, 0.0);
        close(predictor.residual_variance, amplitude_delta().powi(2));
        assert!(terrain.versions[1].surprise_scan.iter().all(|x| x.is_nan()));
        assert!(
            terrain.versions[1]
                .precision_scan
                .iter()
                .all(|x| x.is_nan())
        );
        observe(&mut terrain, [8.0; 2]);
        close(terrain.versions[1].surprise_scan[0], SURPRISE_CAP);
        close(
            terrain.versions[1].precision_scan[0],
            1.0 / (1.0 + amplitude_delta().powi(2)),
        );
        close(terrain.predictors[0].residual_mean, terrain.alpha * 2.0);
    }

    #[test]
    fn normalization_uses_old_ema_and_coverage_neutral_features() {
        let mut p = Predictor {
            amplitude_ema: Some(2.0),
            ..Default::default()
        };
        p.w[1] = 0.5;
        p.w[2] = 0.5;
        p.w[3] = 1.0;
        p.residual_variance = 2.0;
        // phi=[1,2,3,1,...]; e=8/2 - (1+1.5+1) = 0.5.
        let (surprise, precision) = p.observe(
            8.0,
            4.0,
            2.0,
            [6.0; 8],
            [0.5, 0.49, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            0.25,
        );
        close(surprise, 0.0625);
        close(precision, 1.0 / 3.0);
        close(p.amplitude_ema.unwrap(), 3.5);
        close(p.residual_mean, 0.125);
        close(p.residual_variance, 0.75 * (2.0 + 0.25 * 0.25));
    }

    #[test]
    fn rls_matches_independent_first_update_and_centred_ema() {
        let mut p = Predictor {
            amplitude_ema: Some(1.0),
            ..Default::default()
        };
        let (surprise, precision) = p.observe(2.0, 1.0, 1.0, [0.0; 8], [0.0; 8], 0.1);
        close(surprise, SURPRISE_CAP);
        close(precision, 1.0 / (1.0 + amplitude_delta().powi(2)));
        let denominator = LAMBDA + 1000.0;
        for i in 0..DIM {
            close(p.w[i], f64::from(i == 1) + 100.0 / denominator);
            for j in 0..DIM {
                close(
                    p.p[i][j],
                    ((if i == j { 100.0 } else { 0.0 }) - 10000.0 / denominator) / LAMBDA,
                );
                assert_eq!(p.p[i][j], p.p[j][i]);
            }
        }
        close(p.residual_mean, 0.1);
        close(p.residual_variance, 0.9 * (amplitude_delta().powi(2) + 0.1));
        let error = 0.0 - p.w.iter().sum::<f64>();
        let old_variance = p.residual_variance;
        let old_mean = p.residual_mean;
        let old_scale = p.amplitude_ema.unwrap();
        p.observe(0.0, old_scale, 1.0, [0.0; 8], [0.0; 8], 0.1);
        close(
            p.residual_variance,
            0.9 * (old_variance + 0.1 * (error - old_mean).powi(2)),
        );
    }

    #[test]
    fn covariance_resets_after_update_at_trace_limit_or_nonfinite() {
        for diagonal in [200_000.0, f64::INFINITY, f64::NAN] {
            let mut p = Predictor {
                amplitude_ema: Some(1.0),
                ..Default::default()
            };
            for i in 0..DIM {
                p.p[i][i] = diagonal;
            }
            let result = p.observe(1.0, 1.0, 1.0, [0.0; 8], [0.0; 8], 0.1);
            assert_eq!(p.p, Predictor::default().p);
            if diagonal.is_finite() {
                assert!(result.0.is_finite());
                assert!(p.w.iter().any(|w| *w != 0.0));
            } else {
                assert!(result.0.is_nan() && result.1.is_nan());
                assert_eq!(p.w, Predictor::default().w);
            }
        }
        // A non-finite off-diagonal must also reset, independently of trace.
        let mut p = Predictor {
            amplitude_ema: Some(1.0),
            ..Default::default()
        };
        p.p[0][1] = f64::NAN;
        assert!(p.observe(1.0, 1.0, 1.0, [0.0; 8], [0.0; 8], 0.1).0.is_nan());
        assert_eq!(p.p, Predictor::default().p);
    }

    #[test]
    fn nonfinite_resets_the_bin_and_outputs_both_missing() {
        for case in 0..6 {
            let mut p = Predictor {
                amplitude_ema: Some(1.0),
                ..Default::default()
            };
            let mut amplitude = 1.0;
            let mut ages = [1.0; 8];
            match case {
                0 => amplitude = f64::NAN,
                1 => p.residual_variance = f64::INFINITY,
                2 => p.residual_mean = f64::NAN,
                3 => p.amplitude_ema = Some(f64::INFINITY),
                4 => ages[0] = f32::NAN,
                _ => amplitude = 1e200,
            }
            let (s, precision) = p.observe(amplitude, 1.0, 1.0, ages, [1.0; 8], 0.1);
            assert!(s.is_nan() && precision.is_nan());
            assert_eq!(p, Predictor::default());
        }
    }

    #[test]
    fn silence_freezes_extension_but_ring_keeps_advancing_and_resume_keeps_model() {
        let mut terrain = InfoTerrain::new(space(), 0.25);
        for _ in 0..6 {
            observe(&mut terrain, [1.0; 2]);
        }
        for _ in 0..3 {
            observe(&mut terrain, [0.0; 2]);
        }
        let frozen = terrain.predictors.clone();
        for _ in 0..4 {
            observe(&mut terrain, [0.0; 2]);
        }
        assert_eq!(terrain.predictors, frozen);
        assert!(terrain.history.iter().all(|a| *a == 0.0));
        observe(&mut terrain, [2.0; 2]); // Global silence is decided from history.
        assert_eq!(terrain.predictors, frozen);
        observe(&mut terrain, [2.0; 2]);
        assert_ne!(terrain.predictors, frozen);
        assert!(terrain.versions[1].precision_scan[0].is_finite());
    }

    #[test]
    fn gain_invariance_including_age_features_and_missing_masks() {
        let gain = 10.0_f32.powf(-30.0 / 20.0);
        let mut a = InfoTerrain::new(space(), 0.01);
        let mut b = InfoTerrain::new(space(), 0.01);
        let mut ha = SpectralHistory::new(space(), 1000.0, 10);
        let mut hb = SpectralHistory::new(space(), 1000.0, 10);
        for frame in 0_u64..1800 {
            let hash = frame.wrapping_mul(0x9e3779b97f4a7c15).rotate_left(23);
            let jitter = (hash >> 40) as f32 / (1_u32 << 24) as f32 - 0.5;
            let amplitudes = if (900..1100).contains(&frame) {
                [0.0; 2]
            } else {
                [
                    0.6 + 0.3 * (frame as f32 * 0.13).sin() + 0.2 * jitter,
                    if frame < 700 {
                        0.000001
                    } else {
                        0.2 + 0.1 * (frame as f32 * 0.071).cos() - 0.15 * jitter
                    },
                ]
            };
            let power = amplitudes.map(|x| x * x);
            let scaled = amplitudes.map(|x| (gain * x).powi(2));
            let mut ages_a = [[0.0; 8]; 2];
            let mut ages_b = [[0.0; 8]; 2];
            let ca = ha.read_age_features(&mut ages_a);
            let cb = hb.read_age_features(&mut ages_b);
            a.observe(&power, &ages_a, ca, false);
            b.observe(&scaled, &ages_b, cb, false);
            for (a, b) in a.versions.iter().zip(&b.versions) {
                for (x, y) in a
                    .surprise_scan
                    .iter()
                    .chain(&a.precision_scan)
                    .zip(b.surprise_scan.iter().chain(&b.precision_scan))
                {
                    assert_eq!(x.is_nan(), y.is_nan(), "frame {frame}");
                    if x.is_finite() {
                        assert!(
                            (x - y).abs() <= 1e-6 + 1e-4 * x.abs().max(y.abs()),
                            "frame {frame}: {x} != {y}"
                        );
                    }
                }
            }
            ha.observe(frame * 10, (frame + 1) * 10, &power);
            hb.observe(frame * 10, (frame + 1) * 10, &scaled);
        }
    }

    #[test]
    fn gain_invariance_with_active_perceptual_floors() {
        let mut terrains =
            [1.0, 32.0, 1.0 / 32.0].map(|gain| (gain, InfoTerrain::new(space(), 0.1)));
        for frame in 0..100 {
            for (gain, terrain) in &mut terrains {
                let amplitude = if frame % 2 == 0 { 1.005 } else { 0.995 };
                observe(terrain, [amplitude * *gain; 2]);
            }
            for (_, terrain) in &terrains[1..] {
                for (a, b) in terrains[0].1.versions.iter().zip(&terrain.versions) {
                    for (x, y) in a
                        .surprise_scan
                        .iter()
                        .chain(&a.precision_scan)
                        .zip(b.surprise_scan.iter().chain(&b.precision_scan))
                    {
                        assert_eq!(x.is_nan(), y.is_nan());
                        if x.is_finite() {
                            close(*x, *y);
                        }
                    }
                }
            }
        }
        for (_, terrain) in &terrains {
            assert!(terrain.versions[0].surprise_scan[0] < 0.01);
            assert!(terrain.predictors[0].residual_variance < amplitude_delta().powi(2));
            close(
                terrain.versions[1].precision_scan[0],
                1.0 / (1.0 + amplitude_delta().powi(2)),
            );
        }
    }

    #[test]
    fn weighted_quantiles_use_midpoints_interpolation_and_endpoints() {
        let mut values = [(30.0, 3.0), (10.0, 1.0)]; // Midpoints .125 and .625.
        close(weighted_quantile(&mut values, 0.0).unwrap(), 10.0);
        close(weighted_quantile(&mut values, 0.125).unwrap(), 10.0);
        close(weighted_quantile(&mut values, 0.375).unwrap(), 20.0);
        close(weighted_quantile(&mut values, 0.5).unwrap(), 25.0);
        close(weighted_quantile(&mut values, 0.625).unwrap(), 30.0);
        close(weighted_quantile(&mut values, 1.0).unwrap(), 30.0);
        assert_eq!(weighted_quantile(&mut [], 0.5), None);
        assert_eq!(weighted_quantile(&mut [(1.0, 0.0)], 0.5), None);
    }

    #[test]
    fn block_indicator_denominators_quantile_weights_and_nulls_are_independent() {
        let mut terrain = InfoTerrain::new(space(), 0.25);
        let mut block = BlockAccumulator::new(2, true, 4);
        terrain.mean_scan.copy_from_slice(&[1.0, 3.0]);
        terrain.soundgen_weight_scan.copy_from_slice(&[1.0, 3.0]);
        terrain.osf_weight_scan.copy_from_slice(&[1.0, 3.0]);
        terrain.versions[0]
            .precision_scan
            .copy_from_slice(&[1.0, 0.5]);
        block.add(&terrain, 0, 0, false); // Constant history can define precision alone.
        terrain.versions[0]
            .surprise_scan
            .copy_from_slice(&[2.0, f64::NAN]);
        terrain.versions[0]
            .precision_scan
            .copy_from_slice(&[f64::NAN, 0.25]);
        block.add(&terrain, 1, 0, false);
        terrain.mean_scan.copy_from_slice(&[10.0, 2.0]);
        terrain.versions[0]
            .surprise_scan
            .copy_from_slice(&[f64::NAN, 4.0]);
        terrain.versions[0].precision_scan.fill(f64::NAN);
        block.add(&terrain, 2, 0, false);
        let result = block
            .finish(&Arc::new(space()), "habitat", 250, true)
            .unwrap();
        close(result.original.soundgen.mean.unwrap(), 1.75); // (.5+3)/2, not /3.
        close(result.original.soundgen.maximum.unwrap(), 3.0);
        close(
            result.original.precision_mean.unwrap(),
            (0.625 + 0.25) / 2.0,
        );
        assert_eq!(result.original.precision_defined_hops, 2);
        assert_eq!(result.original.surprise_mean_scan, [Some(2.0), Some(4.0)]);
        assert_eq!(
            result.original.precision_mean_scan,
            [Some(1.0), Some(0.375)]
        );
        assert_eq!(result.original.missing_surprise_bin_hops, 4);
        assert_eq!(result.original.missing_precision_bin_hops, 3);
        // Surprise-valid mean amplitudes are 1 and 2; other hops do not count.
        close(result.original.surprise_median.unwrap(), 10.0 / 3.0);
        assert_eq!(result.extended.soundgen.mean, None);
        assert_eq!(result.extended.precision_mean, None);
        assert_eq!(result.extended.surprise_median, None);
        let json = serde_json::to_value(&result).unwrap();
        assert!(json["extended"]["surprise_mean_scan"][0].is_null());
        assert!(json["trace"][0]["surprise"][0][0].is_null());
    }

    #[test]
    fn block_ids_follow_start_sample_and_final_partial_flushes_on_disconnect() {
        let (mut observer, output) = Observer::new(space(), 1000, 300, "habitat", true, true);
        let history = SpectralHistory::new(space(), 1000.0, 300);
        for frame in 0..8 {
            observer.observe(frame, &[1.0; 2], &history, false);
        }
        drop(observer);
        let blocks: Vec<_> = output.rx.iter().collect();
        assert_eq!(
            blocks.iter().map(|b| b.block_id).collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert_eq!(blocks.iter().map(|b| b.hops).collect::<Vec<_>>(), [4, 3, 1]);
        assert_eq!(blocks[0].hop_start, 0);
        assert_eq!(blocks[0].hop_end_exclusive, 4);
        assert!(!blocks[0].incomplete);
        assert!(blocks[2].incomplete);
        assert_eq!(blocks[2].trace.as_ref().unwrap().len(), 1);
        assert_eq!(output.dropped.load(Ordering::Acquire), 0);
    }

    #[test]
    fn gap_reset_retains_block_missing_counts_and_lifetime_losses() {
        let (mut observer, output) =
            Observer::new(space(), 1000, 100, "presentation", false, false);
        let history = SpectralHistory::new(space(), 1000.0, 100);
        for frame in 0..12 {
            observer.observe(frame, &[1.0; 2], &history, false);
        }
        output.dropped.store(7, Ordering::Relaxed);
        let hops = observer.blocks.hops;
        observer.reset_gap();
        assert_eq!(observer.blocks.hops, hops);
        assert_eq!(observer.terrain.frames, 0);
        assert!(observer.terrain.history.iter().all(|x| *x == 0.0));
        assert_eq!(observer.terrain.predictors[0], Predictor::default());
        observer.observe(14, &[1.0; 2], &history, true);
        assert_eq!(observer.terrain.frames, 0);
        assert!(observer.blocks.input_gap);
        drop(observer);
        let blocks: Vec<_> = output.rx.iter().collect();
        assert_eq!(blocks[1].hops, 3);
        assert!(blocks[1].input_gap);
        assert_eq!(blocks[1].original.missing_surprise_bin_hops, 6);
        assert_eq!(output.dropped.load(Ordering::Acquire), 7);
    }

    #[test]
    fn realtime_delivery_counts_every_full_queue_drop() {
        let (mut observer, output) = Observer::new(space(), 1000, 100, "habitat", false, false);
        let history = SpectralHistory::new(space(), 1000.0, 100);
        for frame in 0..700 {
            observer.observe(frame, &[1.0; 2], &history, false);
        }
        drop(observer);
        assert_eq!(output.rx.len(), 64);
        assert_eq!(output.dropped.load(Ordering::Acquire), 6);
    }

    #[test]
    fn deterministic_delivery_waits_for_capacity_and_loses_no_blocks() {
        let (mut observer, output) = Observer::new(space(), 1000, 100, "habitat", false, true);
        let (done_tx, done_rx) = bounded(1);
        let worker = std::thread::spawn(move || {
            let history = SpectralHistory::new(space(), 1000.0, 100);
            for frame in 0..700 {
                observer.observe(frame, &[1.0; 2], &history, false);
            }
            drop(observer);
            done_tx.send(()).unwrap();
        });
        assert!(
            done_rx
                .recv_timeout(std::time::Duration::from_millis(30))
                .is_err()
        );
        let blocks: Vec<_> = output.rx.iter().collect();
        worker.join().unwrap();
        assert_eq!(blocks.len(), 70);
        for (id, block) in blocks.iter().enumerate() {
            assert_eq!(block.block_id, id as u64);
        }
        assert_eq!(output.dropped.load(Ordering::Acquire), 0);
    }

    #[test]
    fn delivered_aggregate_scans_keep_frequency_boundaries() {
        for (version, surprise) in [(0, true), (0, false), (1, true), (1, false)] {
            let terrain = InfoTerrain::new(space(), 0.1);
            let mut accumulator = BlockAccumulator::new(2, false, 10);
            accumulator.add(&terrain, 0, 0, false);
            let mut block = accumulator
                .finish(&terrain.space, "habitat", 100, true)
                .unwrap();
            let target = if version == 0 {
                &mut block.original
            } else {
                &mut block.extended
            };
            let name = if surprise {
                target.surprise_mean_scan.pop();
                "surprise_mean_scan"
            } else {
                target.precision_mean_scan.pop();
                "precision_mean_scan"
            };
            let error = std::panic::catch_unwind(|| block.assert_scan_lengths()).unwrap_err();
            assert!(error.downcast_ref::<String>().unwrap().contains(name));
        }
    }

    #[test]
    fn every_scan_boundary_names_length_errors() {
        fn panics_named(name: &str, f: impl FnOnce() + std::panic::UnwindSafe) {
            let err = std::panic::catch_unwind(f).expect_err("must reject wrong length");
            let message = err
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| err.downcast_ref::<&str>().copied())
                .unwrap();
            assert!(message.contains(name), "{message}");
        }
        panics_named("info_power_scan", || {
            InfoTerrain::new(space(), 0.1).observe(&[0.0], &[[0.0; 8]; 2], [0.0; 8], false)
        });
        panics_named("age_rms_scan", || {
            InfoTerrain::new(space(), 0.1).observe(&[0.0; 2], &[[0.0; 8]; 1], [0.0; 8], false)
        });
        panics_named("surprise_scan", || {
            integrate(&space(), &[0.0], &[0.0; 2]);
        });
        panics_named("weight_scan", || {
            integrate(&space(), &[0.0; 2], &[0.0]);
        });
        panics_named("age_rms_scan", || {
            SpectralHistory::new(space(), 1000.0, 100).read_age_features(&mut [[0.0; 8]; 1]);
        });
        panics_named("info_power_scan", || {
            let (mut observer, _) = Observer::new(space(), 1000, 100, "habitat", false, false);
            observer.observe(
                0,
                &[0.0],
                &SpectralHistory::new(space(), 1000.0, 100),
                false,
            );
        });
        for name in [
            "mean_scan",
            "surprise_scan",
            "precision_scan",
            "weight_scan",
            "surprise_bins_scan",
            "precision_bins_scan",
            "amplitude_bins_scan",
        ] {
            panics_named(name, || {
                let mut terrain = InfoTerrain::new(space(), 0.1);
                let mut block = BlockAccumulator::new(2, false, 10);
                match name {
                    "mean_scan" => {
                        terrain.mean_scan.pop();
                    }
                    "surprise_scan" => {
                        terrain.versions[0].surprise_scan.pop();
                    }
                    "precision_scan" => {
                        terrain.versions[0].precision_scan.pop();
                    }
                    "weight_scan" => {
                        terrain.soundgen_weight_scan.pop();
                    }
                    "surprise_bins_scan" => {
                        block.versions[0].surprise_bins.pop();
                    }
                    "precision_bins_scan" => {
                        block.versions[0].precision_bins.pop();
                    }
                    _ => {
                        block.versions[0].amplitude_bins.pop();
                    }
                }
                if name.ends_with("bins_scan") {
                    block.block_id = Some(0);
                    block.finish(&Arc::new(space()), "habitat", 100, false);
                } else {
                    block.add(&terrain, 0, 0, false);
                }
            });
        }
    }
}

#[cfg(test)]
mod info_terrain_eval;
