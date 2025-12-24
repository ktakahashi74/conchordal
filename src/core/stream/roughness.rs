use crate::core::a_weighting;
use crate::core::density;
use crate::core::landscape::{RoughnessScalarMode, map_roughness01};
use crate::core::landscape::{Landscape, LandscapeParams, LandscapeUpdate};
use crate::core::nsgt_rt::RtNsgtKernelLog2;
use crate::core::peak_extraction::{PeakExtractConfig, extract_peaks_density};
use crate::core::roughness_kernel::erb_grid;

/// Roughness Stream (formerly Ventral).
/// Handles slow spectral analysis focused on roughness and habituation.
pub struct RoughnessStream {
    nsgt_rt: RtNsgtKernelLog2,
    params: LandscapeParams,

    // Internal States
    norm_state: Vec<f32>,           // Leaky integrator for loudness
    habituation_state: Vec<f32>,    // Boredom integrator
    loudness_weights_pow: Vec<f32>, // A-weighting power curve (gain^2)

    // Last computed state (roughness-side of the landscape)
    last_landscape: Landscape,
}

impl RoughnessStream {
    pub fn new(params: LandscapeParams, nsgt_rt: RtNsgtKernelLog2) -> Self {
        let n_bins = nsgt_rt.space().n_bins();
        let loudness_weights_pow = nsgt_rt
            .space()
            .centers_hz
            .iter()
            .map(|&f| a_weighting::a_weighting_gain_pow(f))
            .collect();

        Self {
            nsgt_rt: nsgt_rt.clone(),
            params,
            norm_state: vec![0.0; n_bins],
            habituation_state: vec![0.0; n_bins],
            loudness_weights_pow,
            last_landscape: Landscape::new(nsgt_rt.space().clone()),
        }
    }

    /// Access the most recent landscape snapshot without processing.
    pub fn last(&self) -> &Landscape {
        &self.last_landscape
    }

    /// Process audio chunk asynchronously.
    /// Returns the updated Landscape (roughness/habituation only).
    pub fn process(&mut self, audio: &[f32]) -> Landscape {
        if audio.is_empty() {
            return self.last_landscape.clone();
        }

        // 1. Update Spectrum (NSGT + Normalization)
        let envelope: Vec<f32> = {
            let env = self.nsgt_rt.process_hop(audio);
            env.to_vec()
        };
        self.last_landscape.nsgt_power = envelope.clone();
        let dt_sec = audio.len() as f32 / self.params.fs;
        let norm_env = self.normalize(&envelope, dt_sec);

        // 2. Compute Roughness and habituation
        self.compute_potentials(&norm_env);

        // 3. Return snapshot
        self.last_landscape.clone()
    }

    fn normalize(&mut self, envelope: &[f32], dt_sec: f32) -> Vec<f32> {
        let exp = self.params.loudness_exp.max(0.01);
        let tau_s = (self.params.tau_ms.max(1.0)) * 1e-3;
        let a = (-dt_sec / tau_s).exp();
        let mut out = vec![0.0f32; envelope.len()];
        let ref_power = self.params.ref_power.max(1e-12);
        let space = self.nsgt_rt.space();
        let (_erb, du) = erb_grid(space);

        // RT NSGT provides band power; convert to ERB density and extract peaks before compression.
        let mut pow_density_w = vec![0.0f32; envelope.len()];
        for (i, &pow) in envelope.iter().enumerate() {
            let dui = du[i].max(1e-12);
            let pow_density = pow / dui;
            pow_density_w[i] = pow_density * self.loudness_weights_pow[i];
        }

        let peaks = extract_peaks_density(&pow_density_w, space, &PeakExtractConfig::default());
        let mut subj_density_delta = vec![0.0f32; envelope.len()];
        for peak in peaks {
            if peak.bin_idx >= du.len() {
                continue;
            }
            let dui = du[peak.bin_idx];
            if dui <= 0.0 {
                continue;
            }
            let subj_mass = (peak.mass / ref_power).powf(exp);
            subj_density_delta[peak.bin_idx] += subj_mass / dui;
        }

        // Leaky integrator on compressed delta density.
        for i in 0..envelope.len() {
            let y = a * self.norm_state[i] + (1.0 - a) * subj_density_delta[i];
            self.norm_state[i] = y;
            out[i] = y;
        }
        out
    }

    fn compute_potentials(&mut self, density: &[f32]) {
        let space = self.nsgt_rt.space();
        let (_erb, du) = erb_grid(space);
        let loudness_mass = density::density_to_mass(density, &du);

        // Roughness
        let (r, r_total) = self
            .params
            .roughness_kernel
            .potential_r_from_log2_spectrum(density, space);
        let r_max = r.iter().cloned().fold(0.0f32, f32::max);
        let r_p95 = percentile_95(&r);
        let r_scalar_raw = match self.params.roughness_scalar_mode {
            RoughnessScalarMode::Total => r_total,
            RoughnessScalarMode::Max => r_max,
            RoughnessScalarMode::P95 => r_p95,
        };
        let r_norm = r_scalar_raw / (loudness_mass + 1e-12);
        let r01_scalar = map_roughness01(r_norm, self.params.roughness_half);
        let r01 = if loudness_mass > 0.0 {
            r.iter()
                .map(|&ri| {
                    let r_norm_i = ri / (loudness_mass + 1e-12);
                    map_roughness01(r_norm_i, self.params.roughness_half)
                })
                .collect()
        } else {
            vec![0.0; r.len()]
        };

        // Habituation
        let tau = self.params.habituation_tau.max(1e-3);
        let dt = self.nsgt_rt.dt();
        let a = (-dt / tau).exp();
        let max_depth = self.params.habituation_max_depth.max(0.0);

        for (state, &val) in self.habituation_state.iter_mut().zip(density) {
            let y = a * *state + (1.0 - a) * val;
            *state = y.min(max_depth);
        }

        self.last_landscape.roughness = r;
        self.last_landscape.roughness01 = r01;
        self.last_landscape.roughness_total = r_total;
        self.last_landscape.roughness_max = r_max;
        self.last_landscape.roughness_p95 = r_p95;
        self.last_landscape.roughness_scalar_raw = r_scalar_raw;
        self.last_landscape.roughness_norm = r_norm;
        self.last_landscape.roughness01_scalar = r01_scalar;
        self.last_landscape.loudness_mass = loudness_mass;
        self.last_landscape.habituation = self.habituation_state.clone();
        self.last_landscape.subjective_intensity = density.to_vec();
    }

    pub fn reset(&mut self) {
        self.nsgt_rt.reset();
        self.norm_state.fill(0.0);
        self.habituation_state.fill(0.0);
        self.last_landscape = Landscape::new(self.nsgt_rt.space().clone());
    }

    pub fn apply_update(&mut self, upd: LandscapeUpdate) {
        if let Some(k) = upd.roughness_k {
            self.params.roughness_k = k.max(1e-6);
        }
        if let Some(w) = upd.habituation_weight {
            self.params.habituation_weight = w;
        }
        if let Some(tau) = upd.habituation_tau {
            self.params.habituation_tau = tau;
        }
        if let Some(depth) = upd.habituation_max_depth {
            self.params.habituation_max_depth = depth;
        }
    }
}

fn percentile_95(vals: &[f32]) -> f32 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut buf = vals.to_vec();
    let idx = ((buf.len() - 1) as f32 * 0.95).round() as usize;
    let (_, v, _) = buf.select_nth_unstable_by(idx, |a, b| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });
    *v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
    use crate::core::log2space::Log2Space;
    use crate::core::nsgt_kernel::{NsgtKernelLog2, NsgtLog2Config};
    use crate::core::nsgt_rt::{RtConfig, RtNsgtKernelLog2};
    use crate::core::roughness_kernel::{KernelParams, RoughnessKernel};

    fn build_stream(fs: f32) -> RoughnessStream {
        let space = Log2Space::new(200.0, 4000.0, 12);
        let nsgt = NsgtKernelLog2::new_coherent(
            NsgtLog2Config {
                fs,
                overlap: 0.5,
                nfft_override: Some(256),
                ..Default::default()
            },
            space.clone(),
            None,
        );
        let rt = RtNsgtKernelLog2::with_config(
            nsgt,
            RtConfig {
                tau_min: 1e-6,
                tau_max: 1e-6,
                f_ref: 200.0,
            },
        );

        let params = LandscapeParams {
            fs,
            max_hist_cols: 1,
            alpha: 0.0,
            roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.005),
            harmonicity_kernel: HarmonicityKernel::new(&space, HarmonicityParams::default()),
            roughness_scalar_mode: RoughnessScalarMode::Total,
            roughness_half: 0.1,
            habituation_tau: 1.0,
            habituation_weight: 0.0,
            habituation_max_depth: 1.0,
            loudness_exp: 1.0,
            ref_power: 1.0,
            tau_ms: 1.0,
            roughness_k: 1.0,
        };

        RoughnessStream::new(params, rt)
    }

    #[test]
    fn normalize_scales_with_power_input() {
        let fs = 48_000.0;
        let mut stream = build_stream(fs);
        let n = stream.nsgt_rt.space().n_bins();
        let mut base = vec![0.0f32; n];
        let peak = n / 2;
        base[peak] = 1.0;
        let scaled: Vec<f32> = base.iter().map(|v| v * 4.0).collect();

        let out1 = stream.normalize(&base, 1.0);
        let mut stream2 = build_stream(fs);
        let out2 = stream2.normalize(&scaled, 1.0);

        let v1 = out1[peak];
        assert!(v1 > 1e-6, "unexpected near-zero normalized power");
        let ratio = out2[peak] / v1;
        assert!(
            (ratio - 4.0).abs() < 0.1,
            "expected ~4x scaling for power input, got {ratio:.3}"
        );
    }

    #[test]
    fn a_weighting_uses_power_gain() {
        let fs = 48_000.0;
        let stream = build_stream(fs);
        let i = stream.loudness_weights_pow.len() / 2;
        let f = stream.nsgt_rt.space().centers_hz[i];
        let expected = a_weighting::a_weighting_gain_pow(f);
        let got = stream.loudness_weights_pow[i];
        assert!(
            (got - expected).abs() < 1e-6,
            "expected A-weighting power gain, got {got:.6} vs {expected:.6}"
        );
    }
}
