use crate::core::landscape::{RoughnessScalarMode, map_roughness01};
use crate::core::landscape::{Landscape, LandscapeParams, LandscapeUpdate};
use crate::core::landscape_spectral::SpectralFrontEnd;
use crate::core::nsgt_rt::RtNsgtKernelLog2;
use crate::core::roughness_kernel::erb_grid;

/// Roughness Stream (formerly Ventral).
/// Handles slow spectral analysis focused on roughness and habituation.
pub struct RoughnessStream {
    nsgt_rt: RtNsgtKernelLog2,
    params: LandscapeParams,
    spectral_frontend: SpectralFrontEnd,

    // Internal States
    habituation_state: Vec<f32>,    // Boredom integrator

    // Last computed state (roughness-side of the landscape)
    last_landscape: Landscape,
}

impl RoughnessStream {
    pub fn new(params: LandscapeParams, nsgt_rt: RtNsgtKernelLog2) -> Self {
        let n_bins = nsgt_rt.space().n_bins();
        let spectral_frontend = SpectralFrontEnd::new(nsgt_rt.space().clone(), &params);

        Self {
            nsgt_rt: nsgt_rt.clone(),
            params,
            spectral_frontend,
            habituation_state: vec![0.0; n_bins],
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
        let spectral_frame =
            self.spectral_frontend
                .process_nsgt_power(&envelope, dt_sec, &self.params);

        // 2. Compute Roughness and habituation
        self.compute_potentials(
            &spectral_frame.subjective_intensity,
            spectral_frame.loudness_mass,
        );

        // 3. Return snapshot
        self.last_landscape.clone()
    }

    fn compute_potentials(&mut self, density: &[f32], loudness_mass: f32) {
        let space = self.nsgt_rt.space();
        let (_erb, du) = erb_grid(space);

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
        self.spectral_frontend.reset();
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
