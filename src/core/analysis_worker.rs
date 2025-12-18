use crossbeam_channel::{Receiver, Sender};

use crate::core::harmonicity_kernel::HarmonicityKernel;
use crate::core::landscape::LandscapeUpdate;
use crate::core::log2space::Log2Space;

/// Result payload from the analysis worker:
/// `(frame_id, h_scan, body_log_spectrum)`.
pub type HarmonicityResult = (u64, Vec<f32>, Vec<f32>);

/// Analysis worker: receives mixed spectral bodies, computes harmonicity only, and publishes
/// the result for the main thread to merge.
pub fn run(
    fs: f32,
    space: Log2Space,
    mut harmonicity_kernel: HarmonicityKernel,
    spectrum_rx: Receiver<(u64, Vec<f32>)>,
    result_tx: Sender<HarmonicityResult>,
    update_rx: Receiver<LandscapeUpdate>,
) {
    while let Ok((mut frame_id, mut spectrum_body)) = spectrum_rx.recv() {
        // Drain backlog and keep only the most recent spectrum to avoid fixed lag.
        for (latest_id, latest_body) in spectrum_rx.try_iter() {
            frame_id = latest_id;
            spectrum_body = latest_body;
        }

        // Apply parameter updates (only H-kernel parameters matter here).
        for upd in update_rx.try_iter() {
            if upd.mirror.is_some() || upd.limit.is_some() {
                let mut p = harmonicity_kernel.params;
                if let Some(m) = upd.mirror {
                    p.mirror_weight = m;
                }
                if let Some(l) = upd.limit {
                    p.param_limit = l;
                }
                harmonicity_kernel = HarmonicityKernel::new(&space, p);
            }
        }

        // 1) Linear -> Log2 mapping.
        let n_bins_lin = spectrum_body.len();
        let nfft = (n_bins_lin.saturating_sub(1)) * 2;
        let df = if nfft > 0 { fs / nfft as f32 } else { 0.0 };

        let mut amps_log = vec![0.0f32; space.n_bins()];
        for (i, &mag) in spectrum_body.iter().enumerate() {
            let f = i as f32 * df;
            if let Some(idx) = space.index_of_freq(f) {
                amps_log[idx] += mag;
            }
        }

        // 2) Compute H.
        let (h_scan, _) = harmonicity_kernel.potential_h_from_log2_spectrum(&amps_log, &space);

        // 3) Publish (drop if receiver is full).
        let _ = result_tx.try_send((frame_id, h_scan, amps_log));
    }
}
