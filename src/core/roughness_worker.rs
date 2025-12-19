use crossbeam_channel::{Receiver, Sender};

use crate::core::landscape::{Landscape, LandscapeUpdate};
use crate::core::ventral::VentralStream;

/// Result payload from the roughness worker:
/// `(frame_id, landscape_snapshot)`.
pub type RoughnessResult = (u64, Landscape);

/// Roughness worker: receives time-domain hops, runs NSGT-based audio analysis (R + habituation),
/// and publishes the latest analysis for the main thread to merge.
pub fn run(
    mut ventral: VentralStream,
    hop_rx: Receiver<(u64, Vec<f32>)>,
    result_tx: Sender<RoughnessResult>,
    update_rx: Receiver<LandscapeUpdate>,
) {
    while let Ok((mut frame_id, audio_hop)) = hop_rx.recv() {
        // Drain backlog but *do not* skip audio hops.
        // NSGT-RT maintains an internal ring buffer and assumes time continuity; dropping hops
        // effectively deletes samples and can create broadband artifacts ("mystery peaks").
        let mut hops = Vec::with_capacity(8);
        hops.push(audio_hop);
        for (latest_id, latest_hop) in hop_rx.try_iter() {
            frame_id = latest_id;
            hops.push(latest_hop);
        }

        // Apply parameter updates (habituation params primarily; others are harmless here).
        for upd in update_rx.try_iter() {
            ventral.apply_update(upd);
        }

        // Process each hop in-order to preserve the per-hop dt used by the normalizers.
        let mut analysis = ventral.process(&hops[0]);
        for hop in &hops[1..] {
            analysis = ventral.process(hop);
        }
        let _ = result_tx.try_send((frame_id, analysis));
    }
}
