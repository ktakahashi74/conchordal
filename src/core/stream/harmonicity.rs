use crate::core::harmonicity_kernel::HarmonicityKernel;
use crate::core::landscape::LandscapeUpdate;
use crate::core::log2space::Log2Space;

/// Harmonicity Stream: maps linear spectra to log2 space and applies the H-kernel.
pub struct HarmonicityStream {
    kernel: HarmonicityKernel,
    space: Log2Space,
    fs: f32,
}

impl HarmonicityStream {
    pub fn new(fs: f32, space: Log2Space, kernel: HarmonicityKernel) -> Self {
        Self { kernel, space, fs }
    }

    pub fn update_params(&mut self, upd: LandscapeUpdate) {
        if upd.mirror.is_some() || upd.limit.is_some() {
            let mut params = self.kernel.params;
            if let Some(m) = upd.mirror {
                params.mirror_weight = m;
            }
            if let Some(l) = upd.limit {
                params.param_limit = l;
            }
            self.kernel = HarmonicityKernel::new(&self.space, params);
        }
    }

    /// Process linear amplitude spectrum into log2 bins and compute harmonicity scan.
    /// Returns `(h_scan, log_amps)`.
    pub fn process(&mut self, spectrum_lin: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n_bins_lin = spectrum_lin.len();
        let nfft = (n_bins_lin.saturating_sub(1)) * 2;
        let df = if nfft > 0 { self.fs / nfft as f32 } else { 0.0 };

        let mut amps_log = vec![0.0f32; self.space.n_bins()];
        for (i, &mag) in spectrum_lin.iter().enumerate().skip(1) {
            let f = i as f32 * df;
            if f < self.space.fmin || f > self.space.fmax {
                continue;
            }

            let log_f = f.log2();
            let exact_idx = (log_f - self.space.centers_log2[0]) / self.space.step();
            let idx = exact_idx.floor() as usize;
            let frac = exact_idx - exact_idx.floor();

            if idx + 1 < self.space.n_bins() {
                amps_log[idx] += mag * (1.0 - frac);
                amps_log[idx + 1] += mag * frac;
            } else if idx < self.space.n_bins() {
                amps_log[idx] += mag;
            }
        }

        let (h_scan, _) = self
            .kernel
            .potential_h_from_log2_spectrum(&amps_log, &self.space);
        (h_scan, amps_log)
    }
}
