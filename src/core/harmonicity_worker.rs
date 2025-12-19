use crossbeam_channel::{Receiver, Sender};

use crate::core::landscape::LandscapeUpdate;
use crate::core::stream::harmonicity::HarmonicityStream;

/// Result payload from the harmonicity worker:
/// `(frame_id, h_scan, body_log_spectrum)`.
pub type HarmonicityResult = (u64, Vec<f32>, Vec<f32>);

/// Harmonicity worker: receives mixed spectral bodies, computes harmonicity only, and publishes
/// the result for the main thread to merge.
pub fn run(
    mut stream: HarmonicityStream,
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

        for upd in update_rx.try_iter() {
            stream.update_params(upd);
        }

        let (h_scan, amps_log) = stream.process(&spectrum_body);
        let _ = result_tx.try_send((frame_id, h_scan, amps_log));
    }
}
