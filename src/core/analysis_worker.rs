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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
    use crate::core::landscape::{LandscapeParams, RoughnessScalarMode};
    use crate::core::log2space::Log2Space;
    use crate::core::nsgt_kernel::{NsgtKernelLog2, NsgtLog2Config, PowerMode};
    use crate::core::nsgt_rt::RtNsgtKernelLog2;
    use crate::core::roughness_kernel::{KernelParams, RoughnessKernel};
    use std::f32::consts::PI;
    use std::sync::Arc;
    use std::thread;
    use std::time::Duration;

    fn build_params(space: &Log2Space) -> LandscapeParams {
        LandscapeParams {
            fs: 48_000.0,
            max_hist_cols: 1,
            alpha: 0.0,
            roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.005),
            harmonicity_kernel: HarmonicityKernel::new(space, HarmonicityParams::default()),
            roughness_scalar_mode: RoughnessScalarMode::Total,
            roughness_half: 0.1,
            consonance_harmonicity_weight: 1.0,
            consonance_roughness_weight_floor: 0.35,
            consonance_roughness_weight: 0.5,
            c_state_beta: 2.0,
            c_state_theta: 0.0,
            loudness_exp: 1.0,
            ref_power: 1.0,
            tau_ms: 1.0,
            roughness_k: 1.0,
            roughness_ref_f0_hz: 1000.0,
            roughness_ref_sep_erb: 0.25,
            roughness_ref_mass_split: 0.5,
            roughness_ref_eps: 1e-12,
        }
    }

    #[test]
    fn analysis_landscape_age_at_most_one_after_warmup() {
        let fs = 48_000.0;
        let hop = 256usize;
        let space = Log2Space::new(200.0, 4000.0, 12);
        let params = build_params(&space);
        let nsgt = NsgtKernelLog2::new(
            NsgtLog2Config {
                fs,
                overlap: 0.5,
                nfft_override: Some(256),
                ..Default::default()
            },
            space,
            None,
            PowerMode::Coherent,
        );
        let nsgt_rt = RtNsgtKernelLog2::new(nsgt);
        let stream = AnalysisStream::new(params, nsgt_rt);

        // Use unbounded channels so try_send never drops in tests.
        let (hop_tx, hop_rx) = crossbeam_channel::unbounded::<(u64, Arc<[f32]>)>();
        let (result_tx, result_rx) = crossbeam_channel::unbounded::<AnalysisResult>();
        let (_update_tx, update_rx) = crossbeam_channel::unbounded::<LandscapeUpdate>();

        let handle = thread::spawn(move || run(stream, hop_rx, result_tx, update_rx));

        let warmup = 2u64;
        let steps = 8u64;
        let freq_hz = 440.0f32;
        let audio: Arc<[f32]> = Arc::from(
            (0..hop)
                .map(|i| {
                    let t = i as f32 / fs;
                    (2.0 * PI * freq_hz * t).sin() * 0.1
                })
                .collect::<Vec<f32>>(),
        );
        let mut last_analysis: Option<u64> = None;

        for frame_idx in 0..steps {
            if frame_idx >= warmup {
                if let Some(analysis_id) = last_analysis {
                    let age = frame_idx.saturating_sub(analysis_id);
                    assert!(
                        age <= 1,
                        "expected landscape_age<=1 after warmup (frame={frame_idx}, id={analysis_id}, age={age})"
                    );
                }
            }

            hop_tx
                .send((frame_idx, Arc::clone(&audio)))
                .expect("hop send failed");

            // Wait for the analysis result for this frame.
            let (analysis_id, _) = result_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("analysis result timeout");
            assert_eq!(analysis_id, frame_idx);
            last_analysis = Some(analysis_id);
        }

        drop(hop_tx);
        let _ = handle.join();
    }
}
