use crate::core::landscape::{Landscape, LandscapeParams, LandscapeUpdate};
use crate::core::nsgt_rt::RtNsgtKernelLog2;
use crate::core::peak_extraction::{PeakExtractConfig, extract_peaks_density, peaks_to_delta_density};
use crate::core::roughness_kernel::erb_grid;
use crate::core::utils;

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
            .map(|&f| {
                let g = utils::a_weighting_gain(f);
                g * g
            })
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

        // RT NSGT provides band power, so apply power-domain weighting and compression.
        for (i, &pow) in envelope.iter().enumerate() {
            let weighted_pow = pow * self.loudness_weights_pow[i];
            let subj = (weighted_pow / self.params.ref_power).powf(exp);
            let y = a * self.norm_state[i] + (1.0 - a) * subj;
            self.norm_state[i] = y;
            out[i] = y;
        }
        out
    }

    fn compute_potentials(&mut self, amps: &[f32]) {
        let space = self.nsgt_rt.space();

        // Roughness: convert density spectrum to delta peaks first.
        let peaks_cfg = PeakExtractConfig::default();
        let peaks = extract_peaks_density(amps, space, &peaks_cfg);
        let (erb, du) = erb_grid(space);
        let (r, r_total, intensity) = if peaks.is_empty() {
            let (r_vec, total) = self
                .params
                .roughness_kernel
                .potential_r_from_log2_spectrum(amps, space);
            (r_vec, total, amps.to_vec())
        } else {
            let r_vec = self.params.roughness_kernel.potential_r_from_peaks(&peaks, space);
            let total: f32 = r_vec.iter().zip(du.iter()).map(|(ri, dui)| ri * dui).sum();
            let delta = peaks_to_delta_density(&peaks, &du, erb.len());
            (r_vec, total, delta)
        };

        // Habituation
        let tau = self.params.habituation_tau.max(1e-3);
        let dt = self.nsgt_rt.dt();
        let a = (-dt / tau).exp();
        let max_depth = self.params.habituation_max_depth.max(0.0);

        for (state, &amp) in self.habituation_state.iter_mut().zip(amps) {
            let y = a * *state + (1.0 - a) * amp;
            *state = y.min(max_depth);
        }

        self.last_landscape.roughness = r;
        self.last_landscape.roughness_total = r_total;
        self.last_landscape.habituation = self.habituation_state.clone();
        self.last_landscape.subjective_intensity = intensity;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
    use crate::core::log2space::Log2Space;
    use crate::core::nsgt_kernel::{NsgtKernelLog2, NsgtLog2Config};
    use crate::core::nsgt_rt::{InstMeasure, RtConfig, RtNsgtKernelLog2};
    use crate::core::roughness_kernel::{KernelParams, RoughnessKernel};

    fn build_stream(fs: f32) -> RoughnessStream {
        let space = Log2Space::new(200.0, 4000.0, 12);
        let nsgt = NsgtKernelLog2::new(
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
                measure: InstMeasure::RawPower,
            },
        );

        let params = LandscapeParams {
            fs,
            max_hist_cols: 1,
            alpha: 0.0,
            roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.005),
            harmonicity_kernel: HarmonicityKernel::new(&space, HarmonicityParams::default()),
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
        let base: Vec<f32> = (0..n).map(|i| 0.1 + (i as f32) * 0.001).collect();
        let scaled: Vec<f32> = base.iter().map(|v| v * 4.0).collect();

        let out1 = stream.normalize(&base, 1.0);
        let mut stream2 = build_stream(fs);
        let out2 = stream2.normalize(&scaled, 1.0);

        let (imax, v1) = out1
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        assert!(*v1 > 1e-6, "unexpected near-zero normalized power");
        let ratio = out2[imax] / *v1;
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
        let g = utils::a_weighting_gain(f);
        let expected = g * g;
        let got = stream.loudness_weights_pow[i];
        assert!(
            (got - expected).abs() < 1e-6,
            "expected A-weighting power gain, got {got:.6} vs {expected:.6}"
        );
    }
}
