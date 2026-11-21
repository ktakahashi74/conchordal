//! Causal auditory filterbank, hop statistics, reassignment and loudness density.

use crate::core::landscape::LandscapeParams;
use crate::core::log2space::Log2Space;
use crate::core::roughness_kernel::erb_grid;
use crate::core::utils::a_weighting_gain_pow;
use fearless_simd::{Level, Simd, prelude::*};
use rustfft::num_complex::Complex32;
use std::f32::consts::TAU;

#[cfg(any(test, feature = "research"))]
pub(crate) struct SpectralFrameView<'a> {
    pub subjective_intensity: &'a [f32],
}

#[cfg(test)]
#[path = "auditory_frontend_test_support.rs"]
mod test_support;

#[cfg(test)]
pub(crate) use test_support::{
    ShadowFrontEnd, save_v3_witness, selected_test_active, with_selected_test_frontend,
};

#[cfg(test)]
#[path = "auditory_spread.rs"]
mod spread;

#[path = "auditory_statistics.rs"]
mod statistics;

fn erbn(f: f64) -> f64 {
    9.2645 * (1.0 + 0.00437 * f).ln()
}

fn inv_erbn(e: f64) -> f64 {
    ((e / 9.2645).exp() - 1.0) / 0.00437
}

pub(crate) fn validate_sample_rate(space: &Log2Space, sample_rate: u32) -> Result<(), String> {
    let highest = Bank::centers(space, 0.1875, 0.25)
        .next_back()
        .expect("auditory channels");
    if highest >= sample_rate as f32 / 2.0 {
        return Err(format!(
            "audio.sample_rate {sample_rate} Hz must put the auditory front end's highest channel ({highest:.3} Hz, including margin) below Nyquist"
        ));
    }
    Ok(())
}

#[derive(Clone)]
struct Bank {
    pub centers: Vec<f32>,
    poles: Vec<Complex32>,
    gains: Vec<f32>,
    // Stage-major storage keeps channels contiguous for vectorization.
    state: [Vec<Complex32>; 4],
    previous: Vec<Complex32>,
    // One hop of input, reused so the lane loop can run every sample per channel block.
    input: Vec<Complex32>,
    pub energy: Vec<f32>,
    pub frequency: Vec<f32>,
    pub coherence: Vec<f32>,
    correlation: Vec<Complex32>,
    statistics: Option<statistics::StatisticsWindow>,
    fs: f32,
}

impl Bank {
    fn centers(
        space: &Log2Space,
        step_erb: f64,
        beta: f64,
    ) -> impl DoubleEndedIterator<Item = f32> {
        let anchor = erbn(1000.0);
        let first = ((erbn(f64::from(space.fmin)) - 1.5 * beta - anchor) / step_erb).ceil() as i32;
        let last = ((erbn(f64::from(space.fmax)) + 1.5 * beta - anchor) / step_erb).floor() as i32;
        (first..=last).map(move |k| inv_erbn(anchor + f64::from(k) * step_erb) as f32)
    }

    fn new(space: &Log2Space, fs: f32, step_erb: f64, beta: f64) -> Self {
        assert!(beta > 0.0 && beta <= 1.0);
        let centers: Vec<_> = Self::centers(space, step_erb, beta).collect();
        assert!(centers[0] > 0.0 && *centers.last().unwrap() < fs / 2.0);
        let (poles, gains): (Vec<_>, Vec<_>) = centers
            .iter()
            .map(|&f| {
                let radius =
                    (-std::f64::consts::TAU * beta * 1.0186 * (24.7 + f64::from(f) / 9.265)
                        / f64::from(fs))
                    .exp();
                let phase = std::f64::consts::TAU * f64::from(f) / f64::from(fs);
                (
                    Complex32::new((radius * phase.cos()) as f32, (radius * phase.sin()) as f32),
                    (1.0 - radius) as f32,
                )
            })
            .unzip();
        let n = centers.len();
        Self {
            centers,
            poles,
            gains,
            fs,
            state: std::array::from_fn(|_| vec![Complex32::default(); n]),
            previous: vec![Complex32::default(); n],
            energy: vec![0.0; n],
            frequency: vec![0.0; n],
            coherence: vec![0.0; n],
            correlation: vec![Complex32::default(); n],
            input: Vec::new(),
            statistics: None,
        }
    }

    fn one(&mut self, x: Complex32) {
        for c in 0..self.centers.len() {
            self.state[0][c] = self.poles[c] * self.state[0][c] + self.gains[c] * x;
        }
        for stage in 1..4 {
            for c in 0..self.centers.len() {
                self.state[stage][c] =
                    self.poles[c] * self.state[stage][c] + self.gains[c] * self.state[stage - 1][c];
            }
        }
    }

    fn read_complex_hop(&mut self, input: impl ExactSizeIterator<Item = Complex32>) {
        let samples = input.len();
        self.read_boxcar(input);
        self.finish_boxcar(samples);
    }

    fn read_boxcar(&mut self, input: impl ExactSizeIterator<Item = Complex32>) {
        let samples = self.buffer_hop(input);
        self.read_dispatched();
        self.push_statistics(samples);
    }

    fn buffer_hop(&mut self, input: impl ExactSizeIterator<Item = Complex32>) -> usize {
        let samples = input.len();
        assert!(samples > 0);
        self.input.clear();
        self.input.extend(input);
        samples
    }

    fn push_statistics(&mut self, samples: usize) {
        if let Some(statistics) = &mut self.statistics {
            statistics.push(&mut self.energy, &mut self.correlation, samples);
        }
    }

    fn finish_boxcar(&mut self, samples: usize) {
        for c in 0..self.centers.len() {
            self.frequency[c] = self.fs * self.correlation[c].arg() / TAU;
            self.coherence[c] = if self.energy[c] > 0.0 {
                self.correlation[c].norm() / self.energy[c]
            } else {
                0.0
            };
            let hops = self.statistics.as_ref().map_or(1, |s| s.counts[c]);
            self.energy[c] *= 2.0 / (samples * hops) as f32;
        }
    }

    // On x86, lanes pay off only with 256-bit or wider registers; on its narrower levels
    // the per-sample channel loops below vectorize as well, with identical bits. Wasm
    // SIMD128 and NEON have no wider level, and their lanes beat those loops.
    fn read_dispatched(&mut self) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let level = Level::new();
            if let Some(simd) = level.as_avx512() {
                return simd.vectorize(|| self.read_lanes(simd));
            }
            if let Some(simd) = level.as_avx2() {
                return simd.vectorize(|| self.read_lanes(simd));
            }
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(simd) = Level::new().as_neon() {
            return simd.vectorize(|| self.read_lanes(simd));
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        if let Some(simd) = Level::new().as_wasm_simd128() {
            return simd.vectorize(|| self.read_lanes(simd));
        }
        self.read_scalar();
    }

    // Per-sample loops; also the reference and the spread diagnostic path.
    fn read_scalar(&mut self) {
        self.read_scalar_observing(|_| {});
    }

    #[inline(always)]
    fn read_scalar_observing(&mut self, mut observe: impl FnMut(&[Complex32])) {
        self.energy.fill(0.0);
        self.correlation.fill(Complex32::default());
        for i in 0..self.input.len() {
            self.one(self.input[i]);
            observe(&self.state[3]);
            for c in 0..self.centers.len() {
                let z = self.state[3][c];
                self.energy[c] += z.norm_sqr();
                self.correlation[c] += z * self.previous[c].conj();
                self.previous[c] = z;
            }
        }
    }

    /// Runs the hop for sixteen channels at a time with all state in lanes. Each lane
    /// repeats `read_scalar`'s Complex32 arithmetic in the same order, so every SIMD
    /// level returns the same bits as the scalar reference.
    #[cfg(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "aarch64",
        all(target_arch = "wasm32", target_feature = "simd128")
    ))]
    #[inline(always)]
    fn read_lanes<S: Simd>(&mut self, simd: S) {
        use fearless_simd::f32x16;
        let Self {
            centers,
            poles,
            gains,
            state,
            previous,
            energy,
            correlation,
            input,
            ..
        } = self;
        for start in (0..centers.len()).step_by(16) {
            let end = (start + 16).min(centers.len());
            let real = |values: &[f32]| {
                let mut lanes = [0.; 16];
                lanes[..end - start].copy_from_slice(&values[start..end]);
                f32x16::load_array(simd, lanes)
            };
            let split = |values: &[Complex32]| {
                let (mut re, mut im) = ([0.; 16], [0.; 16]);
                for (k, z) in values[start..end].iter().enumerate() {
                    (re[k], im[k]) = (z.re, z.im);
                }
                (f32x16::load_array(simd, re), f32x16::load_array(simd, im))
            };
            let (pr, pi) = split(poles);
            let g = real(gains);
            let mut s = [0, 1, 2, 3].map(|stage| split(&state[stage]));
            let (mut qr, mut qi) = split(previous);
            let zero = f32x16::splat(simd, 0.);
            let (mut e, mut cr, mut ci) = (zero, zero, zero);
            for x in input.iter() {
                let (mut ur, mut ui) = (f32x16::splat(simd, x.re), f32x16::splat(simd, x.im));
                for (sr, si) in &mut s {
                    let re = (pr * *sr - pi * *si) + g * ur;
                    let im = (pr * *si + pi * *sr) + g * ui;
                    (*sr, *si, ur, ui) = (re, im, re, im);
                }
                e += ur * ur + ui * ui;
                let nqi = -qi;
                cr += ur * qr - ui * nqi;
                ci += ur * nqi + ui * qr;
                (qr, qi) = (ur, ui);
            }
            let store = |values: &mut [Complex32], (re, im): (f32x16<S>, f32x16<S>)| {
                let (re, im) = (re.to_array(), im.to_array());
                for (k, z) in values[start..end].iter_mut().enumerate() {
                    *z = Complex32::new(re[k], im[k]);
                }
            };
            for (stage, lanes) in state.iter_mut().zip(s) {
                store(stage, lanes);
            }
            store(previous, (qr, qi));
            store(correlation, (cr, ci));
            energy[start..end].copy_from_slice(&e.to_array()[..end - start]);
        }
    }

    fn gain_power(&self, channel: usize, unit: Complex32) -> f32 {
        let denominator = (unit - self.poles[channel]).norm_sqr();
        (self.gains[channel].powi(2) / denominator).powi(4)
    }

    fn clear(&mut self) {
        for stage in &mut self.state {
            stage.fill(Complex32::default());
        }
        self.previous.fill(Complex32::default());
        self.energy.fill(0.0);
        self.correlation.fill(Complex32::default());
        self.frequency.fill(0.0);
        self.coherence.fill(0.0);
        if let Some(statistics) = &mut self.statistics {
            statistics.clear();
        }
    }

    fn band_hop_counts(&self) -> &[usize] {
        &self
            .statistics
            .as_ref()
            .expect("selected front end has hop statistics")
            .counts
    }

    fn band_settling_samples(&self) -> Vec<usize> {
        use rustfft::num_complex::Complex64;
        self.poles
            .iter()
            .zip(&self.gains)
            .zip(&self.centers)
            .map(|((&pole, &gain), &hz)| {
                let pole = Complex64::new(f64::from(pole.re), f64::from(pole.im));
                let r = pole.norm();
                if r >= 1.0 {
                    return usize::MAX;
                }
                let g = f64::from(gain);
                let omega = std::f64::consts::TAU * f64::from(hz) / f64::from(self.fs);
                let positive = (g / (Complex64::from_polar(1.0, omega) - pole).norm()).powi(4);
                let negative = (g / (Complex64::from_polar(1.0, -omega) - pole).norm()).powi(4);
                // Bound the omitted impulse-response tail for every sine phase.
                // The lower steady amplitude also includes the negative-frequency image.
                let margin =
                    0.5 * (positive - negative).max(0.0) * (1.0 - 10.0_f64.powf(-1.0 / 20.0));
                if margin <= 0.0 {
                    return usize::MAX;
                }
                let tail = |samples: usize| {
                    let n = samples as f64;
                    let q = 1.0 - r;
                    (g / q).powi(4)
                        * r.powf(n)
                        * (1.0
                            + n * q
                            + n * (n + 1.0) * q * q / 2.0
                            + n * (n + 1.0) * (n + 2.0) * q * q * q / 6.0)
                };
                let mut hi = 1_usize;
                while tail(hi) > margin {
                    let Some(next) = hi.checked_mul(2) else {
                        return usize::MAX;
                    };
                    hi = next;
                }
                let mut lo = hi / 2;
                while lo + 1 < hi {
                    let mid = lo + (hi - lo) / 2;
                    if tail(mid) > margin {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                hi
            })
            .collect()
    }
}

#[derive(Clone)]
pub(crate) struct AuditoryFrontEnd {
    bank: Bank,
    reassignment: Reassignment,
    pub density_scan: Vec<f32>,
    pub warmup_samples: usize,
    missing_samples: usize,
}

impl AuditoryFrontEnd {
    pub(crate) fn band_hop_counts(&self) -> &[usize] {
        self.bank.band_hop_counts()
    }

    pub(crate) fn band_energy(&self) -> (&[f32], &[f32]) {
        (&self.bank.centers, &self.bank.energy)
    }

    pub(crate) fn band_settling_samples(&self) -> Vec<usize> {
        self.bank.band_settling_samples()
    }

    pub(crate) fn selected(space: Log2Space, fs: f32, hop: usize, window: usize) -> Self {
        let mut front = Self::configured(space, fs, 0.25, window);
        front
            .reassignment
            .configure_statistics(&mut front.bank, hop, true);
        front
    }

    fn configured(space: Log2Space, fs: f32, beta: f64, window: usize) -> Self {
        let step = bank_step(beta);
        let bank = Bank::new(&space, fs, step, beta);
        let density_scan = vec![0.0; space.n_bins()];
        let reassignment = Reassignment::new(space, &bank);
        Self {
            bank,
            reassignment,
            density_scan,
            warmup_samples: window,
            missing_samples: 0,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.bank.clear();
        self.density_scan.fill(0.0);
        self.reassignment.deposit_power.fill(0.0);
        self.missing_samples = self.warmup_samples;
    }

    // Unselected bodies still advance the filter and hop-statistic history.
    #[cfg(any(test, feature = "research"))]
    pub(crate) fn skip(&mut self, pcm: &[f32]) {
        self.bank
            .read_complex_hop(pcm.iter().map(|&x| Complex32::new(x, 0.0)));
        self.missing_samples = self.missing_samples.saturating_sub(pcm.len());
    }

    pub(crate) fn process(&mut self, pcm: &[f32], params: &LandscapeParams) -> bool {
        self.process_body_gap(pcm, 1, params)
    }

    pub(crate) fn process_body_gap(
        &mut self,
        pcm: &[f32],
        gap: usize,
        params: &LandscapeParams,
    ) -> bool {
        assert!(gap > 0);
        assert_eq!(self.bank.fs, params.fs);
        self.reassignment
            .space
            .assert_scan_len_named(&self.density_scan, "subjective_intensity_scan");
        self.bank
            .read_complex_hop(pcm.iter().map(|&x| Complex32::new(x, 0.0)));
        let leak = self.reassignment.prepare(
            &self.bank,
            &mut self.density_scan,
            params,
            pcm.len(),
            pcm.len() as f32 * gap as f32 / params.fs,
        );
        assert!(
            self.reassignment.erb_context.is_some(),
            "selected ERB context"
        );
        for c in 0..self.bank.centers.len() {
            let raw_frequency = self.bank.frequency[c];
            let f = raw_frequency.clamp(1.0, self.bank.fs / 2.0 - 1.0);
            let d = self.reassignment.deposit_power[c];
            let local = self.reassignment.local_power[c].max(d);
            let (deposit, _) =
                self.reassignment
                    .channel_deposit(raw_frequency, f, d, local, params);
            let edge = deposit.edge(self.reassignment.du.len());
            for (i, weight) in deposit.linear_weights() {
                self.density_scan[i] +=
                    (1.0 - leak) * edge * weight * deposit.mass / self.reassignment.du[i];
            }
        }
        self.missing_samples = self.missing_samples.saturating_sub(pcm.len());
        self.missing_samples == 0
    }
}

fn bank_step(beta: f64) -> f64 {
    0.25_f64.min(0.75 * beta)
}

#[derive(Clone)]
struct Reassignment {
    space: Log2Space,
    du: Vec<f32>,
    deposit_power: Vec<f32>,
    // Critical-band context power per channel, batched before the deposit loop.
    local_power: Vec<f32>,
    erb_context: Option<statistics::ErbContext>,
    cover: std::sync::Arc<[f32]>,
    cover_first: f64,
    cover_step: f64,
}

impl Reassignment {
    fn new(space: Log2Space, bank: &Bank) -> Self {
        let fs = bank.fs;
        let cover_first = erbn(1.0);
        // The table is shared by cloned histories; only construction sums over C.
        let cover_step = 1.0 / 128.0;
        let count =
            ((erbn(f64::from(fs / 2.0 - 1.0)) - cover_first) / cover_step).ceil() as usize + 1;
        let cover = (0..count)
            .map(|i| {
                let f = inv_erbn(cover_first + i as f64 * cover_step) as f32;
                let unit = Complex32::from_polar(1.0, TAU * f / fs);
                (0..bank.centers.len())
                    .map(|c| bank.gain_power(c, unit))
                    .sum()
            })
            .collect::<Vec<f32>>()
            .into();
        let channels = bank.centers.len();
        Self {
            du: erb_grid(&space).1,
            space,
            deposit_power: vec![0.0; channels],
            local_power: vec![0.0; channels],
            erb_context: None,
            cover,
            cover_first,
            cover_step,
        }
    }

    fn configure_statistics(&mut self, bank: &mut Bank, hop: usize, erb_context: bool) {
        bank.statistics = Some(statistics::StatisticsWindow::new(
            &bank.centers,
            bank.fs,
            hop,
        ));
        // Full length up front: a clone's capacity is its length, so clones never grow.
        bank.input = vec![Complex32::default(); hop];
        let channels = bank.centers.len();
        self.erb_context = erb_context.then(|| statistics::ErbContext::new(bank.fs, channels));
    }

    fn coverage(&self, f: f32) -> f32 {
        let pos = ((erbn(f64::from(f)) - self.cover_first) / self.cover_step)
            .clamp(0.0, (self.cover.len() - 1) as f64);
        let lo = (pos.floor() as usize).min(self.cover.len() - 2);
        let w = (pos - lo as f64) as f32;
        (1.0 - w) * self.cover[lo] + w * self.cover[lo + 1]
    }

    fn prepare(
        &mut self,
        bank: &Bank,
        density_scan: &mut [f32],
        params: &LandscapeParams,
        samples: usize,
        dt: f32,
    ) -> f32 {
        let leak = (-dt / (params.tau_ms.max(1.0) * 1e-3)).exp();
        for density in density_scan {
            *density *= leak;
        }
        for c in 0..bank.centers.len() {
            let f = bank.frequency[c].clamp(1.0, bank.fs / 2.0 - 1.0);
            let hops = bank.statistics.as_ref().map_or(1, |s| s.counts[c]);
            self.deposit_power[c] =
                2.0 * bank.correlation[c].norm() / (samples * hops) as f32 / self.coverage(f);
        }
        if let Some(context) = &mut self.erb_context {
            context.prepare(&bank.frequency, &self.deposit_power);
            context.powers_into(&bank.frequency, &mut self.local_power);
        }
        leak
    }

    #[inline]
    fn channel_deposit(
        &self,
        raw_frequency: f32,
        f: f32,
        d: f32,
        local: f32,
        params: &LandscapeParams,
    ) -> (ChannelDeposit, f32) {
        // Same power-per-ERB floor as the former NSGT peak extractor.
        let floor_density = 1e-6;
        let position = if raw_frequency > 0.0 {
            (raw_frequency.log2() - self.space.centers_log2[0]) / self.space.step()
        } else {
            f32::NEG_INFINITY
        };
        let bin_pos = position.clamp(0.0, (self.du.len() - 1) as f32);
        let lo = bin_pos.floor() as usize;
        let hi = (lo + 1).min(self.du.len() - 1);
        let fraction = (bin_pos - lo as f32).clamp(0.0, 1.0);
        let width = (1.0 - fraction) * self.du[lo] + fraction * self.du[hi];
        // The gate uses P_cb density, independently of deposit splitting.
        let rho = local / width;
        let gamma = if rho > 0.0 {
            (1.0 - floor_density / rho).max(0.0)
        } else {
            0.0
        };
        let kept = gamma * d;
        let mass = if local > 0.0 {
            gamma * d / local
                * (a_weighting_gain_pow(f) * local / params.ref_power.max(1e-12))
                    .powf(params.loudness_exp.max(0.01))
        } else {
            0.0
        };
        debug_assert!(kept.is_finite() && mass.is_finite());
        (
            ChannelDeposit {
                position,
                lo,
                hi,
                fraction,
                mass,
            },
            kept,
        )
    }
}

struct ChannelDeposit {
    position: f32,
    lo: usize,
    hi: usize,
    fraction: f32,
    mass: f32,
}

impl ChannelDeposit {
    #[inline]
    fn edge(&self, bins: usize) -> f32 {
        let position = self.position;
        if position < 0.0 {
            (1.0 + position).max(0.0)
        } else if position > (bins - 1) as f32 {
            (bins as f32 - position).max(0.0)
        } else {
            1.0
        }
    }

    #[inline]
    fn linear_weights(&self) -> [(usize, f32); 2] {
        [(self.lo, 1.0 - self.fraction), (self.hi, self.fraction)]
    }
}

#[cfg(test)]
#[path = "auditory_shadow_tests.rs"]
mod tests;
