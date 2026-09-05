use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};

use crate::core::landscape::{Landscape, LandscapeUpdate};
use crate::core::stream::analysis::AnalysisStream;

/// Result payload from the analysis worker:
/// `(frame_id, landscape_snapshot)`, with `None` invalidating pre-gap observations.
pub type AnalysisResult = (u64, Option<Landscape>);

/// Analysis worker: receives time-domain hops, runs NSGT-based audio analysis,
/// and publishes the latest analysis for the main thread to merge.
pub fn run(
    mut stream: AnalysisStream,
    hop_rx: Receiver<(u64, Arc<[f32]>)>,
    result_tx: Sender<AnalysisResult>,
    update_rx: Receiver<LandscapeUpdate>,
) {
    let mut next_frame_id = 0;
    let mut warmup_samples = 0;
    while let Ok(first_hop) = hop_rx.recv() {
        // Preserve frame IDs through backlog draining so gaps cannot splice audio together.
        let mut hops = Vec::with_capacity(8);
        hops.push(first_hop);
        hops.extend(hop_rx.try_iter());

        // Apply parameter updates (landscape params primarily; others are harmless here).
        for upd in update_rx.try_iter() {
            stream.apply_update(upd);
        }

        // Process each hop in-order to preserve the per-hop dt used by the normalizers.
        let mut analysis = None;
        for (frame_id, hop) in hops {
            if frame_id != next_frame_id {
                stream.reset();
                warmup_samples = stream.window_samples();
                // Invalidation must arrive even when old snapshots fill the result queue.
                if result_tx.send((frame_id, None)).is_err() {
                    return;
                }
            }
            next_frame_id = frame_id.wrapping_add(1);
            let frame = stream.process(hop.as_ref());
            warmup_samples = warmup_samples.saturating_sub(hop.len());
            analysis = (warmup_samples == 0).then_some((frame_id, Some(frame)));
        }
        // Result snapshots are latest-observed state. The main runtime merges only the newest
        // available snapshot, so dropping a stale result under backpressure is acceptable here.
        if let Some(analysis) = analysis {
            let _ = result_tx.try_send(analysis);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::consonance_kernel::{ConsonanceKernel, ConsonanceRepresentationParams};
    use crate::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
    use crate::core::landscape::LandscapeParams;
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
            roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.005),
            harmonicity_kernel: HarmonicityKernel::new(space, HarmonicityParams::default()),
            consonance_kernel: ConsonanceKernel::default(),
            consonance_representation: ConsonanceRepresentationParams::default(),
            consonance_density_roughness_gain: 1.0,
            habituation: crate::core::habituation::HabituationParams::default(),
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
    fn analysis_gap_invalidates_results_until_a_full_window_is_refilled() {
        let fs = 48_000.0;
        let hop = 128;
        for recovery_hops in [1, 3, 4] {
            let space = Log2Space::new(200.0, 4_000.0, 12);
            let params = build_params(&space);
            let nsgt = NsgtKernelLog2::new(
                NsgtLog2Config {
                    fs,
                    overlap: 0.75,
                    nfft_override: Some(512),
                    ..Default::default()
                },
                space,
                None,
                PowerMode::Coherent,
            );
            let mut stream = AnalysisStream::new(params, RtNsgtKernelLog2::new(nsgt));
            let tone: Arc<[f32]> = (0..hop)
                .map(|i| (2.0 * PI * 440.0 * i as f32 / fs).sin() * 0.1)
                .collect();
            for _ in 0..4 {
                stream.process(&tone);
            }
            assert!(stream.last().nsgt_power.iter().any(|power| *power > 0.0));

            let (hop_tx, hop_rx) = crossbeam_channel::unbounded();
            let (result_tx, result_rx) = crossbeam_channel::unbounded();
            let (_update_tx, update_rx) = crossbeam_channel::unbounded();
            hop_tx.send((0, tone)).unwrap();
            // A gap inside the already-queued backlog must reset the complete history.
            let silence: Arc<[f32]> = vec![0.0; hop].into();
            for frame_id in 2..2 + recovery_hops {
                hop_tx.send((frame_id, Arc::clone(&silence))).unwrap();
            }
            drop(hop_tx);
            run(stream, hop_rx, result_tx, update_rx);

            let (gap_id, invalidated) = result_rx.recv().unwrap();
            assert_eq!(gap_id, 2);
            assert!(invalidated.is_none());
            if recovery_hops == 4 {
                let (frame_id, frame) = result_rx.recv().unwrap();
                assert_eq!(frame_id, 5);
                assert!(frame.unwrap().nsgt_power.iter().all(|power| *power == 0.0));
            }
            assert!(
                result_rx.try_recv().is_err(),
                "warmup must not publish a partial window"
            );
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
            if frame_idx >= warmup
                && let Some(analysis_id) = last_analysis
            {
                let age = frame_idx.saturating_sub(analysis_id);
                assert!(
                    age <= 1,
                    "expected landscape_age<=1 after warmup (frame={frame_idx}, id={analysis_id}, age={age})"
                );
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
