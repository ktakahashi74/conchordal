//! Author-rule continuous periodicity memory; no pitch classes or authored scale.
use super::{FA, W};
use std::sync::Arc;

const INTERVALS: usize = 775;

#[derive(Clone, Debug)]
struct Lag {
    index: usize,
    weights: [f64; 4],
}

#[derive(Clone, Debug)]
pub(crate) struct Geometry {
    knots: Vec<f64>,
    lags: Vec<Lag>,
    areas: Vec<f64>,
    lo: f64,
    hi: f64,
    intervals: usize,
}

impl Geometry {
    pub(crate) fn new() -> Self {
        Self::with_intervals(INTERVALS)
    }

    fn with_intervals(intervals: usize) -> Self {
        let lo = (FA / 174.).log2();
        let hi = (FA / 2.).log2();
        let mut knots = vec![lo; 3];
        knots.extend((0..=intervals).map(|i| lo + (hi - lo) * i as f64 / intervals as f64));
        knots.extend([hi; 3]);
        let mut lags = Vec::with_capacity(intervals + 3);
        let mut areas = Vec::with_capacity(intervals + 3);
        for i in 0..intervals + 3 {
            let x = (knots[i + 1] + knots[i + 2] + knots[i + 3]) / 3.;
            let lag = (FA / x.exp2()).clamp(2., 174.);
            let index = lag.floor() as usize;
            let t = lag - index as f64;
            lags.push(Lag {
                index,
                weights: [
                    -t * (t - 1.) * (t - 2.) / 6.,
                    (t + 1.) * (t - 1.) * (t - 2.) / 2.,
                    -(t + 1.) * t * (t - 2.) / 2.,
                    (t + 1.) * t * (t - 1.) / 6.,
                ],
            });
            areas.push((knots[i + 4] - knots[i]) / 4.);
        }
        Self {
            knots,
            lags,
            areas,
            lo,
            hi,
            intervals,
        }
    }

    fn value(&self, coefficients: &[f64], x: f64) -> f64 {
        if !x.is_finite() || x < self.lo || x > self.hi {
            return 0.;
        }
        let i = self
            .knots
            .partition_point(|&t| t <= x)
            .saturating_sub(1)
            .clamp(3, coefficients.len() - 1);
        let mut d = [0.; 4];
        d.copy_from_slice(&coefficients[i - 3..=i]);
        for r in 1..=3 {
            for j in (r..=3).rev() {
                let left = self.knots[i - 3 + j];
                let right = self.knots[i + 1 + j - r];
                let a = (x - left) / (right - left);
                d[j] = (1. - a) * d[j - 1] + a * d[j];
            }
        }
        d[3]
    }

    /// Direct continuous-lag reference; rectification precedes correlation.
    fn coefficients(&self, ring: &[[f64; 40]], start: usize, out: &mut [f64]) {
        let mut channel = [0.; 2 * W];
        out.fill(0.);
        for c in 0..40 {
            for (n, v) in channel.iter_mut().enumerate() {
                *v = ring[(start + n) % (2 * W)][c];
            }
            for (lag, total) in self.lags.iter().zip(out.iter_mut()) {
                let [a, b, c, d] = lag.weights;
                let mut sum = 0.;
                for n in 0..W {
                    let j = n + lag.index;
                    let y = a * channel[j - 1]
                        + b * channel[j]
                        + c * channel[j + 1]
                        + d * channel[j + 2];
                    sum += channel[n] * y.max(0.);
                }
                *total += sum;
            }
        }
    }

    fn peaks(&self, coefficients: &[f64], out: &mut Vec<(f64, f64)>) -> f64 {
        out.clear();
        let h = (self.hi - self.lo) / self.intervals as f64;
        let mut maximum = self
            .value(coefficients, self.lo)
            .max(self.value(coefficients, self.hi));
        for i in 0..self.intervals {
            let x = self.lo + h * i as f64;
            // Recover the exact segment polynomial from four values, in u in [0,1].
            let y0 = self.value(coefficients, x);
            let y1 = self.value(coefficients, x + h / 3.);
            let y2 = self.value(coefficients, x + 2. * h / 3.);
            let y3 = self.value(coefficients, (x + h).min(self.hi));
            let a = 4.5 * (-y0 + 3. * y1 - 3. * y2 + y3);
            let b = 4.5 * (2. * y0 - 5. * y1 + 4. * y2 - y3);
            let c = 0.5 * (-11. * y0 + 18. * y1 - 9. * y2 + 2. * y3);
            let mut roots = [f64::NAN; 2];
            if a == 0. {
                if b != 0. {
                    roots[0] = -c / (2. * b);
                }
            } else {
                let disc = b * b - 3. * a * c;
                if disc >= 0. {
                    let q = -b - disc.sqrt().copysign(b);
                    roots = [q / (3. * a), c / q];
                }
            }
            for u in roots {
                let position = x + h * u;
                if (0. ..1.).contains(&u)
                    && position > self.lo
                    && position < self.hi
                    && 6. * a * u + 2. * b < 0.
                {
                    let value = self.value(coefficients, position);
                    if value > 0. {
                        out.push((position, value));
                        maximum = maximum.max(value);
                    }
                }
            }
        }
        out.sort_by(|a, b| a.0.total_cmp(&b.0));
        maximum
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Snapshot {
    geometry: Arc<Geometry>,
    coefficients: Arc<[f64]>,
    pub(crate) peaks: Arc<[(f64, f64)]>,
    maximum: f64,
    pub(crate) exposure: f64,
    pub(crate) epoch: u64,
    pub(crate) version: u64,
    pub(crate) support_end: u64,
    pub(crate) available_at: u64,
}

impl Snapshot {
    pub(crate) fn affinity(&self, pitch: f64) -> f64 {
        if self.maximum > 0. {
            (self.geometry.value(&self.coefficients, pitch) / self.maximum).clamp(0., 1.)
        } else {
            0.
        }
    }

    pub(crate) fn candidates(&self, target: f32, bounds: (f32, f32)) -> [Option<f32>; 2] {
        let mut strongest: Option<(f64, f64)> = None;
        let mut nearest: Option<(f64, f64)> = None;
        for &(x, y) in self.peaks.iter() {
            let p = (x.exp2() as f32).log2();
            if p < bounds.0 || p > bounds.1 {
                continue;
            }
            let distance = (x - f64::from(target)).abs();
            if strongest.is_none_or(|(a, b)| {
                y > b
                    || (y == b
                        && (distance < (a - f64::from(target)).abs()
                            || (distance == (a - f64::from(target)).abs() && x < a)))
            }) {
                strongest = Some((x, y));
            }
            if nearest.is_none_or(|(a, b)| {
                distance < (a - f64::from(target)).abs()
                    || (distance == (a - f64::from(target)).abs() && (y > b || (y == b && x < a)))
            }) {
                nearest = Some((x, y));
            }
        }
        [strongest, nearest].map(|p| p.map(|(x, _)| (x.exp2() as f32).log2()))
    }
}

#[derive(Clone)]
pub(crate) struct Memory {
    geometry: Arc<Geometry>,
    coefficients: Vec<f64>,
    scratch: Vec<f64>,
    peaks: Vec<(f64, f64)>,
    exposure: f64,
    epoch: u64,
    version: u64,
}
impl Memory {
    pub(crate) fn new() -> Self {
        let geometry = Arc::new(Geometry::new());
        let len = geometry.lags.len();
        Self {
            geometry,
            coefficients: vec![0.; len],
            scratch: vec![0.; len],
            peaks: Vec::with_capacity(2 * INTERVALS),
            exposure: 0.,
            epoch: 0,
            version: 0,
        }
    }
    pub(crate) fn restart(&mut self) {
        self.epoch += 1;
    }
    pub(crate) fn observe(&mut self, ring: &[[f64; 40]], start: usize, duration: f64) {
        if duration <= 0. || !duration.is_finite() {
            return;
        }
        self.geometry.coefficients(ring, start, &mut self.scratch);
        let mass: f64 = self
            .scratch
            .iter()
            .zip(&self.geometry.areas)
            .map(|(v, a)| v * a)
            .sum();
        if !mass.is_finite() || mass <= 0. {
            return;
        }
        let exposure = self.exposure + duration;
        let alpha = duration / exposure;
        for (m, q) in self.coefficients.iter_mut().zip(&self.scratch) {
            *m = (1. - alpha) * *m + alpha * (q / mass);
        }
        self.exposure = exposure;
        self.version += 1;
    }
    pub(crate) fn snapshot(
        &mut self,
        support_end: u64,
        available_at: u64,
    ) -> Option<Arc<Snapshot>> {
        if self.exposure == 0. {
            return None;
        }
        let maximum = self.geometry.peaks(&self.coefficients, &mut self.peaks);
        Some(Arc::new(Snapshot {
            geometry: self.geometry.clone(),
            coefficients: self.coefficients.clone().into(),
            peaks: self.peaks.clone().into(),
            maximum,
            exposure: self.exposure,
            epoch: self.epoch,
            version: self.version,
            support_end,
            available_at,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn spline_constant_mass_and_interior_maxima() {
        let g = Geometry::new();
        assert_eq!(g.lags.len(), 778);
        let c = vec![1.; 778];
        assert!((g.areas.iter().sum::<f64>() - (g.hi - g.lo)).abs() < 1e-12);
        for i in 0..=1000 {
            assert!((g.value(&c, g.lo + (g.hi - g.lo) * i as f64 / 1000.) - 1.).abs() < 1e-14);
        }
        let mut peaks = vec![];
        assert_eq!(g.peaks(&c, &mut peaks), 1.);
        assert!(peaks.is_empty());
    }
}

/// Mixed-only owner. TC retains its invalid state when PS starts a new epoch.
#[derive(Clone)]
pub(crate) struct Tracker {
    memory: Memory,
    standalone: Option<super::Model>,
    gates: std::collections::VecDeque<(u64, u64, bool)>,
    gate_capacity: usize,
    epoch_start: u64,
    window: usize,
    support_end: u64,
    published: Option<Arc<Snapshot>>,
}
impl Tracker {
    pub(crate) fn new(shared: bool, hop: usize, window: usize, at: u64) -> Self {
        // Each PP interval is STEP * 2560 / 147 original samples long.
        let capacity = (super::STEP * 2560).div_ceil(147 * hop) + 2;
        Self {
            memory: Memory::new(),
            standalone: (!shared).then(|| super::Model::new(48000)),
            gates: std::collections::VecDeque::with_capacity(capacity),
            gate_capacity: capacity,
            epoch_start: at,
            window,
            support_end: at,
            published: None,
        }
    }
    pub(crate) fn gap(&mut self, at: u64) {
        self.memory.restart();
        self.standalone = Some(super::Model::new(48000));
        self.gates.clear();
        self.epoch_start = at;
    }
    pub(crate) fn process_standalone(&mut self, audio: &[f32], at: u64, gate: bool) {
        self.process(audio, at, gate, None);
    }
    pub(super) fn process(
        &mut self,
        audio: &[f32],
        at: u64,
        gate: bool,
        shared: Option<&mut super::Model>,
    ) {
        let ready =
            (at + audio.len() as u64).saturating_sub(self.epoch_start) >= self.window as u64;
        self.gates
            .push_back((at, at + audio.len() as u64, gate && ready));
        while self.gates.len() > self.gate_capacity {
            self.gates.pop_front();
        }
        let model = match self.standalone.as_mut() {
            Some(m) => m,
            None => shared.expect("PS shared observation owner"),
        };
        for &x in audio {
            if let Some(frame) = model.push(f64::from(x)) {
                // Fixed IPEM wrapper support, including both resamplers and LP lookahead.
                let numerator = (1 + 2 * (4 * (frame.index * super::STEP + 357) + 40)) * 320 + 3200;
                let end = self.epoch_start as f64 + numerator as f64 / 147.;
                let begin = end - (super::STEP * 2560) as f64 / 147.;
                let mut samples = 0.;
                for &(lo, hi, valid) in &self.gates {
                    if valid {
                        samples += (end.min(hi as f64) - begin.max(lo as f64)).max(0.);
                    }
                }
                let available = self.epoch_start + model.n0 as u64;
                assert!(end <= available as f64, "PP cannot consume future evidence");
                self.memory.observe(
                    &model.pitch.fani,
                    model.pitch.fcount - 2 * W,
                    samples / 48000.,
                );
                self.support_end = available;
            }
        }
        if self
            .published
            .as_ref()
            .is_none_or(|s| s.version != self.memory.version || s.epoch != self.memory.epoch)
        {
            self.published = self
                .memory
                .snapshot(self.support_end, at + audio.len() as u64);
        }
    }
    pub(crate) fn snapshot(&self) -> Option<Arc<Snapshot>> {
        self.published.clone()
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Use {
    pub(crate) snapshot: Option<Arc<Snapshot>>,
    pub(crate) weight: f64,
    pub(crate) bounds: (f32, f32),
    pub(crate) trace: bool,
    pub(crate) decision: std::cell::RefCell<Option<Decision>>,
    pub(crate) added: std::cell::RefCell<Vec<f32>>,
    pub(crate) tc_failed_added: std::cell::Cell<bool>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct Decision {
    pub at: Option<u64>,
    pub epoch: Option<u64>,
    pub version: Option<u64>,
    pub support_end: Option<u64>,
    pub available_at: Option<u64>,
    pub exposure: f64,
    pub weight: f64,
    pub baseline: (f32, f32, f64),
    pub candidates: Vec<(f32, f32, f64)>,
    pub selected: f32,
    pub without_term: f32,
    pub outer_probes: [(f64, f32); 2],
    pub added_candidates: Vec<f32>,
    pub without_candidates: f32,
    pub saturated_base: bool,
    pub tc_failed_added: bool,
    pub changed: bool,
    pub salience: f32,
}

#[cfg(test)]
mod observation_tests {
    use super::*;
    fn pcm(hz: f64, seconds: f64) -> Vec<f32> {
        (0..(48000. * seconds) as usize)
            .map(|n| (0.1 * (std::f64::consts::TAU * hz * n as f64 / 48000.).sin()) as f32)
            .collect()
    }
    #[test]
    fn pitch_set_shared_tc_gap_and_history() {
        let decoder = super::super::Decoder::unqueried_for_test();
        let mut tc = super::super::Context::new(decoder.clone(), 0);
        let mut shared = super::super::Context::new(decoder, 0);
        let mut tracker = Tracker::new(true, 512, 0, 0);
        let mut standalone = Tracker::new(false, 512, 0, 0);
        let audio = pcm(220., 0.5);
        for (i, hop) in audio.chunks(512).enumerate() {
            tc.process(hop);
            shared.process_with_pitch_set(hop, (i * 512) as u64, true, &mut tracker);
            standalone.process(hop, (i * 512) as u64, true, None);
        }
        let a = tc.snapshot().unwrap();
        let b = shared.snapshot().unwrap();
        assert_eq!(a.observed, b.observed);
        let before = tracker.snapshot().unwrap();
        let independent = standalone.snapshot().unwrap();
        assert_eq!(before.coefficients, independent.coefficients);
        assert_eq!(before.exposure, independent.exposure);
        assert!(before.exposure > 0.3 && before.exposure < 0.5);
        assert!(
            before
                .peaks
                .iter()
                .any(|(x, _)| (x.exp2() - 220.).abs() < 5.)
        );
        tc.invalidate();
        shared.invalidate();
        tracker.gap(48000);
        let after_audio = pcm(307., 0.5);
        for (i, hop) in after_audio.chunks(512).enumerate() {
            tc.process(hop);
            shared.process_with_pitch_set(hop, 48000 + (i * 512) as u64, true, &mut tracker);
        }
        assert!(tc.snapshot().is_none() && shared.snapshot().is_none());
        let after = tracker.snapshot().unwrap();
        assert!((after.exposure - 2. * before.exposure).abs() < 1e-12);
        assert_eq!(after.epoch, 1);
        assert!(
            after
                .peaks
                .iter()
                .any(|(x, _)| (x.exp2() - 220.).abs() < 5.)
        );
        // Independent C/continuous-lag integration puts this averaged peak near 296.5 Hz.
        // It is not a faithful replay of the 307 Hz physical fundamental.
        assert!(
            after
                .peaks
                .iter()
                .any(|(x, _)| (x.exp2() - 296.5).abs() < 0.5)
        );
        assert!(after.affinity(220_f64.log2()) > 0.);
        let mut current_only = Tracker::new(false, 512, 0, 48000);
        for (i, hop) in after_audio.chunks(512).enumerate() {
            current_only.process(hop, 48000 + (i * 512) as u64, true, None);
        }
        assert!(
            after.affinity(220_f64.log2())
                > current_only.snapshot().unwrap().affinity(220_f64.log2())
        );
    }
    #[test]
    fn pitch_set_gate_partition_and_duplicate_reads() {
        let audio = pcm(220., 0.4);
        let mut a = Tracker::new(false, 512, 0, 0);
        let mut b = Tracker::new(false, 257, 0, 0);
        for (i, hop) in audio.chunks(512).enumerate() {
            a.process(hop, (i * 512) as u64, true, None);
        }
        for (i, hop) in audio.chunks(257).enumerate() {
            b.process(hop, (i * 257) as u64, true, None);
        }
        let sa = a.snapshot().unwrap();
        let sb = b.snapshot().unwrap();
        assert!((sa.exposure - sb.exposure).abs() < 1e-12);
        for (x, y) in sa.coefficients.iter().zip(sb.coefficients.iter()) {
            assert!((x - y).abs() < 1e-12);
        }
        for _ in 0..20 {
            assert_eq!(a.snapshot().unwrap().version, sa.version);
        }
        for (i, hop) in audio.chunks(512).enumerate() {
            a.process(hop, audio.len() as u64 + (i * 512) as u64, false, None);
        }
        let held = a.snapshot().unwrap();
        // Only the first PP interval can overlap the previously valid hop.
        assert!(held.exposure - sa.exposure <= super::super::STEP as f64 / FA + 1e-12);
    }
}

#[cfg(test)]
mod reference_tests {
    use super::*;
    #[test]
    #[ignore = "PS1_FIXTURES: independently acquired C/NumPy controls"]
    fn pitch_set_independent_c_continuous_history() {
        let root = std::path::PathBuf::from(std::env::var("PS1_FIXTURES").unwrap());
        let mut memories = Vec::new();
        for hz in [220, 307, 900] {
            let bytes = std::fs::read(root.join(format!("fixtures/tone{hz}/pcm.f64"))).unwrap();
            let pcm: Vec<f32> = bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| f64::from_le_bytes(*b) as f32)
                .collect();
            let mut tracker = Tracker::new(false, 512, 0, 0);
            for (i, hop) in pcm.chunks(512).enumerate() {
                tracker.process_standalone(hop, (i * 512) as u64, true);
            }
            let actual = tracker.snapshot().unwrap();
            let reference: serde_json::Value = serde_json::from_slice(
                &std::fs::read(root.join(format!("history-{hz}.json"))).unwrap(),
            )
            .unwrap();
            let expected = reference["coefficients"].as_array().unwrap();
            let error = actual
                .coefficients
                .iter()
                .zip(expected)
                .map(|(a, b)| (a - b.as_f64().unwrap()).abs())
                .fold(0_f64, f64::max);
            println!(
                "PS C/NumPy tone{hz}: coefficient error={error:e}, exposure={}, peaks={}",
                actual.exposure,
                actual.peaks.len()
            );
            // The retained C ASCII boundary and measured ANI propagation bound are 1e-7.
            assert!(error < 1e-7);
            memories.push(actual);
        }
        let mut mean = Memory::new();
        for i in 0..mean.coefficients.len() {
            mean.coefficients[i] = (memories[0].coefficients[i] + memories[1].coefficients[i]) / 2.;
        }
        mean.exposure = memories[0].exposure + memories[1].exposure;
        let actual = mean.snapshot(48000, 48000).unwrap();
        let reference: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("history-220+307.json")).unwrap())
                .unwrap();
        let step = reference["reference_grid_step"].as_f64().unwrap();
        let refs = reference["reference_peaks"].as_array().unwrap();
        // The two separate high-register peaks survive; averaging moves the second.
        for p in refs.iter().filter(|p| p[0].as_f64().unwrap().exp2() > 180.) {
            let x = p[0].as_f64().unwrap();
            let distance = actual
                .peaks
                .iter()
                .map(|(a, _)| (a - x).abs())
                .fold(f64::INFINITY, f64::min);
            println!(
                "PS averaged reference peak={} Hz, error={} cents",
                x.exp2(),
                distance * 1200.
            );
            assert!(distance <= 2. * step);
        }
    }
}

#[cfg(test)]
pub(crate) fn heard_for_test(hz: f64) -> Arc<Snapshot> {
    let mut tracker = Tracker::new(false, 512, 0, 0);
    let pcm: Vec<_> = (0..24000)
        .map(|n| (0.1 * (std::f64::consts::TAU * hz * n as f64 / 48000.).sin()) as f32)
        .collect();
    for (i, hop) in pcm.chunks(512).enumerate() {
        tracker.process_standalone(hop, (i * 512) as u64, true);
    }
    tracker.snapshot().unwrap()
}

#[cfg(test)]
mod cumulative_tests {
    use super::*;
    #[test]
    fn pitch_set_duration_weighting_and_retained_gap() {
        let first: Vec<[f64; 40]> = (0..352)
            .map(|n| [1. + (n as f64 * 0.07).sin(); 40])
            .collect();
        let second: Vec<[f64; 40]> = (0..352)
            .map(|n| [1. + (n as f64 * 0.11).sin(); 40])
            .collect();
        let mut a = Memory::new();
        a.observe(&first, 0, 3.);
        let mut b = Memory::new();
        b.observe(&second, 0, 1.);
        let aa = a.snapshot(3, 3).unwrap();
        let bb = b.snapshot(1, 1).unwrap();
        a.restart();
        a.observe(&second, 0, 1.);
        let mixed = a.snapshot(10, 10).unwrap();
        assert_eq!(mixed.exposure, 4.);
        for ((&x, &y), &z) in aa
            .coefficients
            .iter()
            .zip(bb.coefficients.iter())
            .zip(mixed.coefficients.iter())
        {
            assert!((z - (0.75 * x + 0.25 * y)).abs() < 1e-14);
        }
        let mass: f64 = mixed
            .coefficients
            .iter()
            .zip(&mixed.geometry.areas)
            .map(|(x, a)| x * a)
            .sum();
        assert!((mass - 1.).abs() < 1e-12);
        assert!(mixed.coefficients.iter().all(|x| x.is_finite() && *x >= 0.));
    }
}

#[cfg(test)]
pub(crate) fn coefficients_for_test(coefficients: Vec<f64>) -> Arc<Snapshot> {
    let mut memory = Memory::new();
    assert_eq!(coefficients.len(), memory.coefficients.len());
    memory.coefficients = coefficients;
    memory.exposure = 1.;
    memory.snapshot(48000, 48000).unwrap()
}
