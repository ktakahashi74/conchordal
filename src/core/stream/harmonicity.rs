use crate::core::harmonicity_kernel::HarmonicityKernel;
use crate::core::landscape::LandscapeUpdate;
use crate::core::log2space::Log2Space;

/// Harmonicity Stream: accepts log2 spectra and applies the H-kernel.
pub struct HarmonicityStream {
    kernel: HarmonicityKernel,
    space: Log2Space,
}

impl HarmonicityStream {
    pub fn new(space: Log2Space, kernel: HarmonicityKernel) -> Self {
        Self { kernel, space }
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

    /// Process log2 amplitude spectrum and compute harmonicity scan.
    /// Returns `(h_scan, log_amps)`.
    pub fn process(&mut self, spectrum_log: &[f32]) -> (Vec<f32>, Vec<f32>) {
        assert_eq!(
            spectrum_log.len(),
            self.space.n_bins(),
            "harmonicity spectrum must match log2 space"
        );
        let amps_log = spectrum_log.to_vec();
        let (h_scan, _) =
            self.kernel
                .potential_h_from_log2_spectrum(&amps_log, &self.space);
        (h_scan, amps_log)
    }
}
