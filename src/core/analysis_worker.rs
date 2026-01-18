use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};

use crate::core::landscape::{Landscape, LandscapeUpdate};
use crate::core::stream::analysis::AnalysisStream;

/// Result payload from the analysis worker:
/// `(frame_id, landscape_snapshot)`.
pub type AnalysisResult = (u64, Landscape);

/// Analysis worker: receives time-domain hops, runs NSGT-based audio analysis,
/// and publishes the latest analysis for the main thread to merge.
pub fn run(
    mut stream: AnalysisStream,
    hop_rx: Receiver<(u64, Arc<[f32]>)>,
    result_tx: Sender<AnalysisResult>,
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

        // Apply parameter updates (landscape params primarily; others are harmless here).
        for upd in update_rx.try_iter() {
            stream.apply_update(upd);
        }

        // Process each hop in-order to preserve the per-hop dt used by the normalizers.
        let mut analysis = stream.process(hops[0].as_ref());
        for hop in &hops[1..] {
            analysis = stream.process(hop.as_ref());
        }
        let _ = result_tx.try_send((frame_id, analysis));
    }
}
