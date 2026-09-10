use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender};

use crate::core::landscape::{Landscape, LandscapeUpdate};
use crate::core::stream::analysis::AnalysisStream;

/// Result payload from the analysis worker:
/// `(frame_id, landscape_snapshot)`, with `None` invalidating pre-gap observations.
pub type AnalysisResult = (u64, Option<Landscape>);

#[derive(Clone, Copy)]
pub(crate) enum AnalysisDelivery {
    Latest,
    Ordered,
}

/// Analysis worker: receives time-domain hops, runs NSGT-based audio analysis,
/// and publishes either current state or every observation for listener memory.
pub(crate) fn run(
    mut stream: AnalysisStream,
    hop_rx: Receiver<(u64, Arc<[f32]>)>,
    result_tx: Sender<AnalysisResult>,
    update_rx: Receiver<LandscapeUpdate>,
    delivery: AnalysisDelivery,
) {
    let mut next_frame_id = 0;
    let mut warmup_samples = 0;
    let hop_samples = stream.hop_samples();
    let mut spectral_history = crate::core::spectral_history::SpectralHistory::new(
        stream.last().space.clone(),
        stream.sample_rate() as f64,
        stream.window_samples(),
    );
    while let Ok(first_hop) = hop_rx.recv() {
        // Preserve frame IDs through backlog draining so gaps cannot splice audio together.
        let mut hops = Vec::with_capacity(8);
        hops.push(first_hop);
        hops.extend(hop_rx.try_iter().take(hop_rx.len()));

        // Apply parameter updates (landscape params primarily; others are harmless here).
        for upd in update_rx.try_iter() {
            stream.apply_update(upd);
        }

        // Process each hop in-order to preserve the per-hop dt used by the normalizers.
        let mut analysis = None;
        for (frame_id, hop) in hops {
            assert_eq!(
                hop.len(),
                hop_samples,
                "analysis delivery requires complete NSGT hops"
            );
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
            spectral_history.observe(
                frame_id * hop_samples as u64,
                (frame_id + 1) * hop_samples as u64,
                &frame.nsgt_power,
            );
            warmup_samples = warmup_samples.saturating_sub(hop.len());
            analysis = (warmup_samples == 0).then_some((frame_id, frame));
            if matches!(delivery, AnalysisDelivery::Ordered)
                && let Some((id, mut frame)) = analysis.take()
            {
                frame.spectral_history = spectral_history.snapshot();
                if result_tx.send((id, Some(frame))).is_err() {
                    return;
                }
            }
        }
        // Only the generator's current-state delivery may discard older snapshots.
        if let Some((id, mut frame)) = analysis {
            frame.spectral_history = spectral_history.snapshot();
            let _ = result_tx.try_send((id, Some(frame)));
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
    fn listener_memory_receives_every_analysis_in_a_backlog() {
        let fs = 48_000.0;
        let hop = 128;
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
        let mut reference =
            AnalysisStream::new(params.clone(), RtNsgtKernelLog2::new(nsgt.clone()));
        let mut expected = Vec::new();
        let mut hops = Vec::new();
        for frame_id in 0..24 {
            let audio: Arc<[f32]> = (0..hop)
                .map(|i| {
                    let time = (frame_id as usize * hop + i) as f32 / fs;
                    let frequency = if frame_id < 12 { 440.0 } else { 660.0 };
                    0.1 * (2.0 * PI * frequency * time).sin()
                })
                .collect();
            expected.push(reference.process(&audio));
            hops.push((frame_id, audio));
        }
        let mut final_history = None;
        for capacity in [1, 4, 24] {
            let stream = AnalysisStream::new(params.clone(), RtNsgtKernelLog2::new(nsgt.clone()));
            let (hop_tx, hop_rx) = crossbeam_channel::unbounded();
            let (result_tx, result_rx) = crossbeam_channel::bounded(capacity);
            let (_update_tx, update_rx) = crossbeam_channel::unbounded();
            for row in &hops {
                hop_tx.send(row.clone()).unwrap();
            }
            drop(hop_tx);
            let handle = thread::spawn(move || {
                run(
                    stream,
                    hop_rx,
                    result_tx,
                    update_rx,
                    AnalysisDelivery::Ordered,
                );
            });
            for (index, reference) in expected.iter().enumerate() {
                let (frame_id, frame) = result_rx.recv_timeout(Duration::from_secs(2)).unwrap();
                assert_eq!(frame_id, index as u64);
                let frame = frame.expect("contiguous evidence");
                assert_eq!(frame.nsgt_power, reference.nsgt_power);
                assert_eq!(frame.subjective_intensity, reference.subjective_intensity);
                assert_eq!(frame.spectral_history.is_some(), index >= 3);
                if let Some(history) = frame.spectral_history {
                    assert_eq!(
                        history.observed_through_sample,
                        (index as u64 + 1) * hop as u64
                    );
                    if index == expected.len() - 1 {
                        if let Some(previous) = &final_history {
                            let previous: &Arc<
                                crate::core::spectral_history::SpectralHistorySnapshot,
                            > = previous;
                            assert_eq!(
                                history.known_rms_by_age_scan,
                                previous.known_rms_by_age_scan
                            );
                            assert_eq!(
                                history.known_coverage_by_age,
                                previous.known_coverage_by_age
                            );
                        }
                        final_history = Some(history);
                    }
                }
            }
            handle.join().unwrap();
            assert!(result_rx.try_recv().is_err());
        }

        // Coalescing may discard deliveries, but not the observations in their history.
        let (hop_tx, hop_rx) = crossbeam_channel::unbounded();
        let (result_tx, result_rx) = crossbeam_channel::unbounded();
        let (_update_tx, update_rx) = crossbeam_channel::unbounded();
        for row in &hops {
            hop_tx.send(row.clone()).unwrap();
        }
        drop(hop_tx);
        run(
            AnalysisStream::new(params.clone(), RtNsgtKernelLog2::new(nsgt.clone())),
            hop_rx,
            result_tx,
            update_rx,
            AnalysisDelivery::Latest,
        );
        let (id, frame) = result_rx.recv().unwrap();
        assert_eq!(id, 23);
        assert!(result_rx.try_recv().is_err());
        let history = frame.unwrap().spectral_history.unwrap();
        let expected_history = final_history.unwrap();
        assert_eq!(
            history.known_rms_by_age_scan,
            expected_history.known_rms_by_age_scan
        );
        assert_eq!(
            history.known_coverage_by_age,
            expected_history.known_coverage_by_age
        );

        // A blocked ordered publisher must stop when its consumer goes away.
        let stream = AnalysisStream::new(params, RtNsgtKernelLog2::new(nsgt));
        let (hop_tx, hop_rx) = crossbeam_channel::unbounded();
        let (result_tx, result_rx) = crossbeam_channel::bounded(1);
        let (_update_tx, update_rx) = crossbeam_channel::unbounded();
        for row in hops {
            hop_tx.send(row).unwrap();
        }
        let (done_tx, done_rx) = crossbeam_channel::bounded(1);
        let handle = thread::spawn(move || {
            run(
                stream,
                hop_rx,
                result_tx,
                update_rx,
                AnalysisDelivery::Ordered,
            );
            done_tx.send(()).unwrap();
        });
        result_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        drop(result_rx);
        done_rx.recv_timeout(Duration::from_secs(2)).unwrap();
        handle.join().unwrap();
        drop(hop_tx);
    }

    #[test]
    fn analysis_gap_invalidates_results_until_a_full_window_is_refilled() {
        let fs = 48_000.0;
        let hop = 128;
        for (recovery_hops, delivery) in [1, 3, 4].into_iter().flat_map(|hops| {
            [AnalysisDelivery::Latest, AnalysisDelivery::Ordered].map(|delivery| (hops, delivery))
        }) {
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
            run(stream, hop_rx, result_tx, update_rx, delivery);

            if matches!(delivery, AnalysisDelivery::Ordered) {
                let (frame_id, frame) = result_rx.recv().unwrap();
                assert_eq!(frame_id, 0);
                assert!(frame.unwrap().nsgt_power.iter().any(|&power| power > 0.0));
            }
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
        let hop = 128usize;
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

        let handle = thread::spawn(move || {
            run(
                stream,
                hop_rx,
                result_tx,
                update_rx,
                AnalysisDelivery::Latest,
            )
        });

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
