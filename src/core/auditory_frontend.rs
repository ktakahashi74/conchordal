//! Causal auditory filterbank, hop statistics, reassignment and loudness density.

use crate::core::landscape::LandscapeParams;
use crate::core::log2space::Log2Space;
use crate::core::roughness_kernel::erb_grid;
use crate::core::utils::a_weighting_gain_pow;
use rustfft::num_complex::Complex32;
use std::f32::consts::TAU;

pub(crate) struct SpectralFrameView<'a> {
    pub subjective_intensity: &'a [f32],
}

#[cfg(test)]
thread_local! {
    static SELECTED_TEST_FRONTEND: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn selected_test_active() -> bool {
    SELECTED_TEST_FRONTEND.with(std::cell::Cell::get)
}

#[cfg(test)]
pub(crate) fn with_selected_test_frontend<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            SELECTED_TEST_FRONTEND.with(|s| s.set(self.0));
        }
    }
    let _restore = Restore(SELECTED_TEST_FRONTEND.with(|s| s.replace(true)));
    run()
}

#[cfg(test)]
pub(crate) fn save_v3_witness(mut payload: serde_json::Value, history: &[Vec<f32>; 4]) {
    use sha2::Digest;
    let dir = std::path::PathBuf::from(
        std::env::var("AUD_M1_OUTPUT").expect("explicit V3 witness directory"),
    );
    if dir.join("witness.json").exists() {
        return;
    }
    let mut files = Vec::new();
    for (name, pcm) in ["mix", "own", "removed", "oracle"].into_iter().zip(history) {
        let bytes: Vec<_> = pcm.iter().flat_map(|x| x.to_le_bytes()).collect();
        let filename = format!("{name}.f32");
        std::fs::write(dir.join(&filename), &bytes).unwrap();
        files.push(serde_json::json!({"role":name,"pcm":filename,"samples":pcm.len(),"sha256":sha2::Sha256::digest(&bytes).iter().map(|b|format!("{b:02x}")).collect::<String>()}));
    }
    payload["pcm"] = serde_json::json!(files);
    payload["configuration"] =
        serde_json::json!({"beta":0.25,"boxcar":true,"erb_context":true,"s1":false});
    std::fs::write(
        dir.join("witness.json"),
        serde_json::to_vec(&payload).unwrap(),
    )
    .unwrap();
}

#[cfg(test)]
#[path = "auditory_spread.rs"]
mod spread;

#[path = "auditory_statistics.rs"]
mod statistics;

// Niemitalo's published evolutionary design, H2 + i H1, scaled by 1/2.
// https://yehar.com/blog/?p=368 (Equations 1-4).
#[cfg(test)]
const ALLPASS_POLES: [[f64; 4]; 2] = [
    [
        0.4021921162426,
        0.8561710882420,
        0.9722909545651,
        0.9952884791278,
    ],
    [0.6923878, 0.9360654322959, 0.9882295226860, 0.9987488452737],
];

fn erbn(f: f64) -> f64 {
    9.2645 * (1.0 + 0.00437 * f).ln()
}

fn inv_erbn(e: f64) -> f64 {
    ((e / 9.2645).exp() - 1.0) / 0.00437
}

#[cfg(test)]
#[derive(Clone, Default)]
struct AnalyticPair {
    input: [[[f32; 2]; 4]; 2],
    output: [[[f32; 2]; 4]; 2],
    delayed: f32,
    parity: usize,
}

#[cfg(test)]
impl AnalyticPair {
    fn one(&mut self, x: f32) -> Complex32 {
        let mut pair = [x; 2];
        for (branch, value) in pair.iter_mut().enumerate() {
            for stage in 0..4 {
                let a = (ALLPASS_POLES[branch][stage].powi(2)) as f32;
                let y = a * (*value + self.output[branch][stage][self.parity])
                    - self.input[branch][stage][self.parity];
                self.input[branch][stage][self.parity] = *value;
                self.output[branch][stage][self.parity] = y;
                *value = y;
            }
        }
        let z = Complex32::new(0.5 * pair[0], 0.5 * self.delayed);
        self.delayed = pair[1];
        self.parity ^= 1;
        z
    }
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
    pub energy: Vec<f32>,
    pub frequency: Vec<f32>,
    pub coherence: Vec<f32>,
    correlation: Vec<Complex32>,
    #[cfg(test)]
    spread: Option<spread::SpreadHistory>,
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
            #[cfg(test)]
            spread: None,
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
        #[cfg(test)]
        if let Some(spread) = &mut self.spread {
            spread.observe(&self.state[3]);
        }
    }

    #[cfg(test)]
    fn read_hop(&mut self, pcm: &[f32], pair: &mut Option<AnalyticPair>) {
        self.read_complex_hop(
            pcm.iter()
                .map(|&x| pair.as_mut().map_or(Complex32::new(x, 0.0), |p| p.one(x))),
        );
    }

    fn read_complex_hop(&mut self, input: impl ExactSizeIterator<Item = Complex32>) {
        let samples = input.len();
        assert!(samples > 0);
        self.energy.fill(0.0);
        self.correlation.fill(Complex32::default());
        #[cfg(test)]
        if let Some(spread) = &mut self.spread {
            spread.start_hop();
        }
        for x in input {
            self.one(x);
            for c in 0..self.centers.len() {
                let z = self.state[3][c];
                self.energy[c] += z.norm_sqr();
                self.correlation[c] += z * self.previous[c].conj();
                self.previous[c] = z;
            }
        }
        if let Some(statistics) = &mut self.statistics {
            statistics.push(&mut self.energy, &mut self.correlation, samples);
        }
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
        #[cfg(test)]
        if let Some(spread) = &mut self.spread {
            spread.finish(self.fs);
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
        #[cfg(test)]
        if let Some(spread) = &mut self.spread {
            spread.clear();
        }
        if let Some(statistics) = &mut self.statistics {
            statistics.clear();
        }
    }
}

#[derive(Clone)]
pub(crate) struct AuditoryFrontEnd {
    bank: Bank,
    #[cfg(test)]
    pair: Option<AnalyticPair>,
    space: Log2Space,
    du: Vec<f32>,
    #[cfg(test)]
    pub power_scan: Vec<f32>,
    pub density_scan: Vec<f32>,
    #[cfg(test)]
    pub channel_power: Vec<f32>,
    #[cfg(test)]
    pub channel_mass: Vec<f32>,
    deposit_power: Vec<f32>,
    #[cfg(test)]
    context_power: Vec<f32>,
    erb_context: Option<statistics::ErbContext>,
    #[cfg(test)]
    reference_context: bool,
    #[cfg(test)]
    pub beta: f64,
    #[cfg(test)]
    pub step_erb: f64,
    cover: std::sync::Arc<[f32]>,
    cover_first: f64,
    cover_step: f64,
    pub warmup_samples: usize,
    missing_samples: usize,
    #[cfg(test)]
    erb_first: f64,
}

#[cfg(test)]
pub(crate) type ShadowFrontEnd = AuditoryFrontEnd;

impl AuditoryFrontEnd {
    pub(crate) fn selected(space: Log2Space, fs: f32, hop: usize, window: usize) -> Self {
        Self::configured(space, fs, 0.25, false, window).with_statistics(hop, true)
    }

    #[cfg(test)]
    pub(crate) fn diagnostic(&self) -> serde_json::Value {
        serde_json::json!({"centers_hz":self.bank.centers,"frequency_hz":self.bank.frequency,"energy":self.bank.energy,
            "correlation":self.bank.correlation.iter().map(|z|[z.re,z.im]).collect::<Vec<_>>(),
            "statistic_hops":self.bank.statistics.as_ref().map(|s|&s.counts),"deposit_power":self.deposit_power,
            "pcb":self.context_power,"mass":self.channel_mass,"density":self.density_scan,"power":self.power_scan})
    }

    fn with_statistics(mut self, hop: usize, erb_context: bool) -> Self {
        self.bank.statistics = Some(statistics::StatisticsWindow::new(
            &self.bank.centers,
            self.bank.fs,
            hop,
        ));
        self.erb_context = erb_context.then(|| statistics::ErbContext::new(self.bank.fs));
        self
    }

    #[cfg(test)]
    pub(crate) fn with_gaussian_spread(mut self) -> Self {
        self.bank.spread = Some(spread::SpreadHistory::new(
            &self.bank.centers,
            self.bank.fs,
            self.beta,
        ));
        self
    }

    #[cfg(test)]
    pub(crate) fn new(space: Log2Space, fs: f32, beta: f64, analytic: bool, window: usize) -> Self {
        Self::configured(space, fs, beta, analytic, window)
    }

    fn configured(space: Log2Space, fs: f32, beta: f64, analytic: bool, window: usize) -> Self {
        let step = 0.25_f64.min(0.75 * beta);
        let bank = Bank::new(&space, fs, step, beta);
        #[cfg(test)]
        let pair = analytic.then(AnalyticPair::default);
        #[cfg(not(test))]
        assert!(!analytic);
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
        let n = space.n_bins();
        let channels = bank.centers.len();
        #[cfg(test)]
        let erb_first = erbn(f64::from(bank.centers[0]));
        Self {
            bank,
            #[cfg(test)]
            pair,
            du: erb_grid(&space).1,
            space,
            #[cfg(test)]
            power_scan: vec![0.0; n],
            density_scan: vec![0.0; n],
            #[cfg(test)]
            channel_power: vec![0.0; channels],
            #[cfg(test)]
            channel_mass: vec![0.0; channels],
            deposit_power: vec![0.0; channels],
            #[cfg(test)]
            context_power: vec![0.0; channels],
            erb_context: None,
            #[cfg(test)]
            reference_context: false,
            #[cfg(test)]
            beta,
            #[cfg(test)]
            step_erb: step,
            cover,
            cover_first,
            cover_step,
            #[cfg(test)]
            erb_first,

            warmup_samples: window,
            missing_samples: 0,
        }
    }

    pub(crate) fn reset(&mut self) {
        self.bank.clear();
        #[cfg(test)]
        if let Some(pair) = &mut self.pair {
            *pair = AnalyticPair::default();
        }
        #[cfg(test)]
        self.power_scan.fill(0.0);
        self.density_scan.fill(0.0);
        #[cfg(test)]
        self.channel_power.fill(0.0);
        #[cfg(test)]
        self.channel_mass.fill(0.0);
        self.deposit_power.fill(0.0);
        #[cfg(test)]
        self.context_power.fill(0.0);
        self.missing_samples = self.warmup_samples;
    }

    // Unselected bodies still advance the filter and hop-statistic history.
    pub(crate) fn skip(&mut self, pcm: &[f32]) {
        #[cfg(test)]
        if self.bank.statistics.is_none() {
            for &x in pcm {
                self.bank.one(
                    self.pair
                        .as_mut()
                        .map_or(Complex32::new(x, 0.0), |p| p.one(x)),
                );
            }
            self.bank.previous.copy_from_slice(&self.bank.state[3]);
            self.missing_samples = self.missing_samples.saturating_sub(pcm.len());
            return;
        }
        #[cfg(test)]
        self.bank.read_hop(pcm, &mut self.pair);
        #[cfg(not(test))]
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
        #[cfg(test)]
        self.space
            .assert_scan_len_named(&self.power_scan, "reassigned_power_scan");
        self.space
            .assert_scan_len_named(&self.density_scan, "subjective_intensity_scan");
        #[cfg(test)]
        self.bank.read_hop(pcm, &mut self.pair);
        #[cfg(not(test))]
        self.bank
            .read_complex_hop(pcm.iter().map(|&x| Complex32::new(x, 0.0)));
        self.reassign(params, pcm.len(), pcm.len() as f32 * gap as f32 / params.fs);
        self.missing_samples = self.missing_samples.saturating_sub(pcm.len());
        self.missing_samples == 0
    }

    fn coverage(&self, f: f32) -> f32 {
        let pos = ((erbn(f64::from(f)) - self.cover_first) / self.cover_step)
            .clamp(0.0, (self.cover.len() - 1) as f64);
        let lo = (pos.floor() as usize).min(self.cover.len() - 2);
        let w = (pos - lo as f64) as f32;
        (1.0 - w) * self.cover[lo] + w * self.cover[lo + 1]
    }

    fn reassign(&mut self, params: &LandscapeParams, samples: usize, dt: f32) {
        #[cfg(test)]
        self.power_scan.fill(0.0);
        let leak = (-dt / (params.tau_ms.max(1.0) * 1e-3)).exp();
        for density in &mut self.density_scan {
            *density *= leak;
        }
        // Same power-per-ERB floor as the former NSGT peak extractor.
        let floor_density = 1e-6;
        for c in 0..self.bank.centers.len() {
            let f = self.bank.frequency[c].clamp(1.0, self.bank.fs / 2.0 - 1.0);
            let hops = self.bank.statistics.as_ref().map_or(1, |s| s.counts[c]);
            self.deposit_power[c] =
                2.0 * self.bank.correlation[c].norm() / (samples * hops) as f32 / self.coverage(f);
        }
        if let Some(context) = &mut self.erb_context {
            context.prepare(&self.bank.frequency, &self.deposit_power);
        }
        for c in 0..self.bank.centers.len() {
            let raw_frequency = self.bank.frequency[c];
            let f = raw_frequency.clamp(1.0, self.bank.fs / 2.0 - 1.0);
            let d = self.deposit_power[c];
            #[cfg(not(test))]
            let local = self
                .erb_context
                .as_ref()
                .expect("selected ERB context")
                .power_at(f)
                .max(d);
            #[cfg(test)]
            let local = {
                let unit = Complex32::from_polar(1.0, TAU * f / self.bank.fs);
                // Both S1 choices preserve the real PCM's positive amplitude A/2.
                // Multiplying complex power by two restores real mean-square power.
                let channel_pos = ((erbn(f64::from(f)) - self.erb_first) / self.step_erb)
                    .clamp(0.0, (self.bank.centers.len() - 1) as f64);
                let lower = channel_pos.floor() as usize;
                let upper = (lower + 1).min(self.bank.centers.len() - 1);
                let w = (channel_pos - lower as f64) as f32;
                let power_at = |j| self.bank.energy[j] / self.bank.gain_power(j, unit);
                self.erb_context
                    .as_ref()
                    .map_or_else(
                        || (1.0 - w) * power_at(lower) + w * power_at(upper),
                        |context| {
                            #[cfg(test)]
                            if self.reference_context {
                                let b = 1.0186 * (24.7 + f64::from(f) / 9.265);
                                return self
                                    .bank
                                    .frequency
                                    .iter()
                                    .zip(&self.deposit_power)
                                    .map(|(&y, &p)| {
                                        f64::from(p)
                                            * (1.0
                                                + ((f64::from(
                                                    y.clamp(1.0, self.bank.fs / 2.0 - 1.0),
                                                ) - f64::from(f))
                                                    / b)
                                                    .powi(2))
                                            .powi(-4)
                                    })
                                    .sum::<f64>() as f32;
                            }
                            context.power_at(f)
                        },
                    )
                    .max(d)
            };
            #[cfg(test)]
            {
                self.context_power[c] = local;
            }
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
            #[cfg(test)]
            {
                self.channel_power[c] = kept;
            }
            #[cfg(test)]
            {
                self.channel_mass[c] = mass;
            }
            debug_assert!(kept.is_finite() && mass.is_finite());
            #[cfg(test)]
            if let Some(spread) = &self.bank.spread
                && spread.sigma_hz[c] > 0.0
                && (kept > 0.0 || mass > 0.0)
            {
                spread::gaussian_weights(
                    &self.space,
                    f64::from(raw_frequency),
                    spread.sigma_hz[c],
                    |i, weight| {
                        self.power_scan[i] += weight * kept / self.du[i];
                        self.density_scan[i] += (1.0 - leak) * weight * mass / self.du[i];
                    },
                );
                continue;
            }
            let edge = if position < 0.0 {
                (1.0 + position).max(0.0)
            } else if position > (self.du.len() - 1) as f32 {
                (self.du.len() as f32 - position).max(0.0)
            } else {
                1.0
            };
            for (i, weight) in [(lo, 1.0 - fraction), (hi, fraction)] {
                #[cfg(test)]
                {
                    self.power_scan[i] += edge * weight * kept / self.du[i];
                }
                self.density_scan[i] += (1.0 - leak) * edge * weight * mass / self.du[i];
            }
        }
    }
}

#[cfg(test)]
#[path = "auditory_shadow_tests.rs"]
mod tests;
