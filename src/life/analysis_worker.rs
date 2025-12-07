use crossbeam_channel::{Receiver, Sender};

use crate::core::landscape::{Landscape, LandscapeFrame};

/// Heavy-weight analysis worker: receives mixed spectral bodies, runs landscape update, and publishes frames.
pub fn run(
    mut landscape: Landscape,
    spectrum_rx: Receiver<(u64, Vec<f32>)>,
    landscape_tx: Sender<(u64, LandscapeFrame)>,
) {
    while let Ok((frame_id, spectrum_body)) = spectrum_rx.recv() {
        // 1. Map linear spectrum to log2 space and compute potentials.
        let snapshot = landscape.process_precomputed_spectrum(&spectrum_body);

        // 2. Publish (drop if receiver is full)
        let _ = landscape_tx.try_send((frame_id, snapshot));
    }
}
