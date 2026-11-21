//! Alternate auditory front ends and instrumentation for tests and research assays.

use super::{
    AuditoryFrontEnd as ProductionFrontEnd, Bank, Complex32, LandscapeParams, Log2Space,
    Reassignment, TAU, bank_step, erbn, spread,
};

thread_local! {
    static SELECTED_TEST_FRONTEND: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) fn selected_test_active() -> bool {
    SELECTED_TEST_FRONTEND.with(std::cell::Cell::get)
}

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

// Niemitalo's published evolutionary design, H2 + i H1, scaled by 1/2.
// https://yehar.com/blog/?p=368 (Equations 1-4).
const ALLPASS_POLES: [[f64; 4]; 2] = [
    [
        0.4021921162426,
        0.8561710882420,
        0.9722909545651,
        0.9952884791278,
    ],
    [0.6923878, 0.9360654322959, 0.9882295226860, 0.9987488452737],
];

#[derive(Clone, Default)]
struct AnalyticPair {
    input: [[[f32; 2]; 4]; 2],
    output: [[[f32; 2]; 4]; 2],
    delayed: f32,
    parity: usize,
}

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

#[derive(Clone)]
pub(crate) struct ShadowFrontEnd {
    pub(super) bank: Bank,
    pub(super) reassignment: Reassignment,
    #[cfg(all(feature = "native", feature = "loo-diagnostic"))]
    oracle: Option<Box<PowerOracle>>,
    pair: Option<AnalyticPair>,
    pub(super) spread: Option<spread::SpreadHistory>,
    pub(super) power_scan: Vec<f32>,
    pub density_scan: Vec<f32>,
    pub(super) channel_power: Vec<f32>,
    pub(super) channel_mass: Vec<f32>,
    pub(super) context_power: Vec<f32>,
    pub(super) reference_context: bool,
    pub(super) beta: f64,
    step_erb: f64,
    pub(super) warmup_samples: usize,
    missing_samples: usize,
    erb_first: f64,
}

#[cfg(all(feature = "native", feature = "loo-diagnostic"))]
type AuditoryFrontEnd = ShadowFrontEnd;

#[cfg(all(feature = "native", feature = "loo-diagnostic"))]
include!("../../tests/research/room_oracle_power.rs");

impl ShadowFrontEnd {
    pub(crate) fn selected(space: Log2Space, fs: f32, hop: usize, window: usize) -> Self {
        Self::from_frontend(
            ProductionFrontEnd::selected(space, fs, hop, window),
            0.25,
            false,
        )
    }

    pub(super) fn new(space: Log2Space, fs: f32, beta: f64, analytic: bool, window: usize) -> Self {
        Self::from_frontend(
            ProductionFrontEnd::configured(space, fs, beta, window),
            beta,
            analytic,
        )
    }

    fn from_frontend(front: ProductionFrontEnd, beta: f64, analytic: bool) -> Self {
        let ProductionFrontEnd {
            bank,
            reassignment,
            density_scan,
            warmup_samples,
            missing_samples,
        } = front;
        let channels = bank.centers.len();
        let erb_first = erbn(f64::from(bank.centers[0]));
        Self {
            bank,
            reassignment,
            #[cfg(all(feature = "native", feature = "loo-diagnostic"))]
            oracle: None,
            pair: analytic.then(AnalyticPair::default),
            spread: None,
            power_scan: vec![0.0; density_scan.len()],
            density_scan,
            channel_power: vec![0.0; channels],
            channel_mass: vec![0.0; channels],
            context_power: vec![0.0; channels],
            reference_context: false,
            beta,
            step_erb: bank_step(beta),
            warmup_samples,
            missing_samples,
            erb_first,
        }
    }

    pub(crate) fn band_hop_counts(&self) -> &[usize] {
        self.bank.band_hop_counts()
    }

    pub(crate) fn band_energy(&self) -> (&[f32], &[f32]) {
        (&self.bank.centers, &self.bank.energy)
    }

    pub(crate) fn band_settling_samples(&self) -> Vec<usize> {
        self.bank.band_settling_samples()
    }

    pub(super) fn with_statistics(mut self, hop: usize, erb_context: bool) -> Self {
        self.reassignment
            .configure_statistics(&mut self.bank, hop, erb_context);
        self
    }

    pub(crate) fn diagnostic(&self) -> serde_json::Value {
        serde_json::json!({"centers_hz":self.bank.centers,"frequency_hz":self.bank.frequency,"energy":self.bank.energy,
            "correlation":self.bank.correlation.iter().map(|z|[z.re,z.im]).collect::<Vec<_>>(),
            "statistic_hops":self.bank.statistics.as_ref().map(|s|&s.counts),"deposit_power":self.reassignment.deposit_power,
            "pcb":self.context_power,"mass":self.channel_mass,"density":self.density_scan,"power":self.power_scan})
    }

    pub(super) fn with_gaussian_spread(mut self) -> Self {
        self.spread = Some(spread::SpreadHistory::new(
            &self.bank.centers,
            self.bank.fs,
            self.beta,
        ));
        self
    }

    fn read_hop(&mut self, pcm: &[f32]) {
        let pair = &mut self.pair;
        let input = pcm
            .iter()
            .map(|&x| pair.as_mut().map_or(Complex32::new(x, 0.0), |p| p.one(x)));
        if let Some(spread) = &mut self.spread {
            let samples = self.bank.buffer_hop(input);
            spread.start_hop();
            self.bank
                .read_scalar_observing(|state| spread.observe(state));
            self.bank.push_statistics(samples);
            self.bank.finish_boxcar(samples);
            spread.finish(self.bank.fs);
        } else {
            self.bank.read_complex_hop(input);
        }
    }

    pub(crate) fn reset(&mut self) {
        self.bank.clear();
        if let Some(spread) = &mut self.spread {
            spread.clear();
        }
        #[cfg(all(feature = "native", feature = "loo-diagnostic"))]
        if let Some(oracle) = &mut self.oracle {
            oracle.own.clear();
        }
        if let Some(pair) = &mut self.pair {
            *pair = AnalyticPair::default();
        }
        self.power_scan.fill(0.0);
        self.density_scan.fill(0.0);
        self.channel_power.fill(0.0);
        self.channel_mass.fill(0.0);
        self.reassignment.deposit_power.fill(0.0);
        self.context_power.fill(0.0);
        self.missing_samples = self.warmup_samples;
    }

    pub(crate) fn skip(&mut self, pcm: &[f32]) {
        if self.bank.statistics.is_none() {
            for &x in pcm {
                self.bank.one(
                    self.pair
                        .as_mut()
                        .map_or(Complex32::new(x, 0.0), |p| p.one(x)),
                );
                if let Some(spread) = &mut self.spread {
                    spread.observe(&self.bank.state[3]);
                }
            }
            self.bank.previous.copy_from_slice(&self.bank.state[3]);
            self.missing_samples = self.missing_samples.saturating_sub(pcm.len());
            return;
        }
        self.read_hop(pcm);
        self.missing_samples = self.missing_samples.saturating_sub(pcm.len());
    }

    pub(crate) fn process(&mut self, pcm: &[f32], params: &LandscapeParams) -> bool {
        #[cfg(all(feature = "native", feature = "loo-diagnostic"))]
        if let Some(oracle) = &mut self.oracle {
            assert!(self.pair.is_none() && self.spread.is_none());
            self.bank
                .read_boxcar(pcm.iter().map(|&x| Complex32::new(x, 0.0)));
            oracle.subtract(&mut self.bank, pcm.len());
            self.bank.finish_boxcar(pcm.len());
            self.reassign(params, pcm.len(), pcm.len() as f32 / params.fs);
            self.missing_samples = self.missing_samples.saturating_sub(pcm.len());
            return self.missing_samples == 0;
        }
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
            .assert_scan_len_named(&self.power_scan, "reassigned_power_scan");
        self.reassignment
            .space
            .assert_scan_len_named(&self.density_scan, "subjective_intensity_scan");
        self.read_hop(pcm);
        self.reassign(params, pcm.len(), pcm.len() as f32 * gap as f32 / params.fs);
        self.missing_samples = self.missing_samples.saturating_sub(pcm.len());
        self.missing_samples == 0
    }

    fn context_power_at(&self, c: usize, f: f32) -> f32 {
        let unit = Complex32::from_polar(1.0, TAU * f / self.bank.fs);
        // Both S1 choices preserve the real PCM's positive amplitude A/2.
        // Multiplying complex power by two restores real mean-square power.
        let channel_pos = ((erbn(f64::from(f)) - self.erb_first) / self.step_erb)
            .clamp(0.0, (self.bank.centers.len() - 1) as f64);
        let lower = channel_pos.floor() as usize;
        let upper = (lower + 1).min(self.bank.centers.len() - 1);
        let w = (channel_pos - lower as f64) as f32;
        let power_at = |j| self.bank.energy[j] / self.bank.gain_power(j, unit);
        self.reassignment.erb_context.as_ref().map_or_else(
            || (1.0 - w) * power_at(lower) + w * power_at(upper),
            |context| {
                if self.reference_context {
                    let b = 1.0186 * (24.7 + f64::from(f) / 9.265);
                    return self
                        .bank
                        .frequency
                        .iter()
                        .zip(&self.reassignment.deposit_power)
                        .map(|(&y, &p)| {
                            f64::from(p)
                                * (1.0
                                    + ((f64::from(y.clamp(1.0, self.bank.fs / 2.0 - 1.0))
                                        - f64::from(f))
                                        / b)
                                        .powi(2))
                                .powi(-4)
                        })
                        .sum::<f64>() as f32;
                }
                debug_assert_eq!(
                    context.power_at(f).to_bits(),
                    self.reassignment.local_power[c].to_bits()
                );
                self.reassignment.local_power[c]
            },
        )
    }

    pub(super) fn reassign(&mut self, params: &LandscapeParams, samples: usize, dt: f32) {
        self.power_scan.fill(0.0);
        let leak =
            self.reassignment
                .prepare(&self.bank, &mut self.density_scan, params, samples, dt);
        let du = &self.reassignment.du;
        for c in 0..self.bank.centers.len() {
            let raw_frequency = self.bank.frequency[c];
            let f = raw_frequency.clamp(1.0, self.bank.fs / 2.0 - 1.0);
            let d = self.reassignment.deposit_power[c];
            let local = self.context_power_at(c, f).max(d);
            self.context_power[c] = local;
            let (deposit, kept) =
                self.reassignment
                    .channel_deposit(raw_frequency, f, d, local, params);
            let mass = deposit.mass;
            self.channel_power[c] = kept;
            self.channel_mass[c] = mass;
            if let Some(spread) = &self.spread
                && spread.sigma_hz[c] > 0.0
                && (kept > 0.0 || mass > 0.0)
            {
                spread::gaussian_weights(
                    &self.reassignment.space,
                    f64::from(raw_frequency),
                    spread.sigma_hz[c],
                    |i, weight| {
                        self.power_scan[i] += weight * kept / du[i];
                        self.density_scan[i] += (1.0 - leak) * weight * mass / du[i];
                    },
                );
                continue;
            }
            let edge = deposit.edge(du.len());
            for (i, weight) in deposit.linear_weights() {
                self.power_scan[i] += edge * weight * kept / du[i];
                self.density_scan[i] += (1.0 - leak) * edge * weight * mass / du[i];
            }
        }
    }
}
