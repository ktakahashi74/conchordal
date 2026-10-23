use std::sync::Arc;

use crossbeam_channel::{Receiver, Sender, TrySendError};

use crate::core::landscape::{Landscape, LandscapeUpdate};
use crate::core::stream::analysis::AnalysisStream;

/// One frame ID binds habitat PCM and the owned source PCM captured from that mix.
pub(crate) type AnalysisHop = (u64, Arc<[f32]>, Option<crate::core::stream::loo::PcmPacket>);

/// `(frame_id, landscape_snapshot)`, with `None` invalidating pre-gap observations.
pub type AnalysisResult = (u64, Option<Landscape>);

pub(crate) enum AnalysisDelivery {
    Latest(Receiver<AnalysisResult>),
    Ordered,
}

fn publish_latest(
    sender: &Sender<AnalysisResult>,
    stale: &Receiver<AnalysisResult>,
    mut result: AnalysisResult,
) {
    loop {
        match sender.try_send(result) {
            Ok(()) | Err(TrySendError::Disconnected(_)) => return,
            Err(TrySendError::Full(value)) => {
                result = value;
                // The consumer may have freed a slot since try_send failed.
                let _ = stale.try_recv();
            }
        }
    }
}

/// Analysis worker: receives time-domain hops, runs NSGT-based audio analysis,
/// and publishes either current state or every observation for listener memory.
pub(crate) fn run(
    mut stream: AnalysisStream,
    hop_rx: Receiver<AnalysisHop>,
    result_tx: Sender<AnalysisResult>,
    update_rx: Receiver<LandscapeUpdate>,
    delivery: AnalysisDelivery,
    mut temporal_tap: Option<crate::temporal_cognition::observation::Tap>,
    raw_output: Option<crate::temporal_cognition::t2::runtime_input::Output>,
) {
    let mut loo_streams: Vec<crate::core::stream::loo::Observer> = Vec::new();
    #[cfg(test)]
    let mut l1_probe = crate::runtime::l1_probe::Probe::new("analysis");

    let mut next_frame_id = 0;
    let mut warmup_samples = 0;
    let hop_samples = stream.hop_samples();
    let mut raw = (raw_output.is_some() || temporal_tap.as_ref().is_some_and(|t| t.t2_enabled))
        .then(|| {
            crate::temporal_cognition::t2::runtime_input::Input::new(
                stream.sample_rate() as u32,
                hop_samples as u64,
                raw_output,
            )
            .expect("validated raw T2 geometry")
        });
    let mut spectral_history = crate::core::spectral_history::SpectralHistory::new(
        stream.last().space.clone(),
        stream.sample_rate() as f64,
        stream.window_samples(),
    );
    while let Ok(first_hop) = hop_rx.recv() {
        #[cfg(test)]
        let l1_begin = l1_probe.as_mut().map(|p| p.before(first_hop.0));
        #[cfg(all(test, feature = "profile-alloc"))]
        let measure = std::env::var_os("L1_ANALYSIS_ALLOC").is_some();
        #[cfg(all(test, feature = "profile-alloc"))]
        if measure {
            crate::runtime_profile::begin_allocations();
        }
        // Preserve frame IDs through backlog draining so gaps cannot splice audio together.
        let mut hops = Vec::with_capacity(8);
        hops.push(first_hop);
        hops.extend(hop_rx.try_iter().take(hop_rx.len()));

        // Apply parameter updates (landscape params primarily; others are harmless here).
        for upd in update_rx.try_iter() {
            for observer in &mut loo_streams {
                observer.update(upd);
            }
            stream.apply_update(upd);
        }

        // Process each hop in-order to preserve the per-hop dt used by the normalizers.
        let mut analysis = None;
        for (frame_id, hop, loo_pcm) in hops {
            assert_eq!(
                hop.len(),
                hop_samples,
                "analysis delivery requires complete NSGT hops"
            );
            if frame_id != next_frame_id {
                stream.reset();
                for observer in &mut loo_streams {
                    observer.reset();
                }
                warmup_samples = stream.window_samples();
                // Invalidation must arrive even when old snapshots fill the result queue.
                match &delivery {
                    AnalysisDelivery::Latest(stale) => {
                        publish_latest(&result_tx, stale, (frame_id, None));
                    }
                    AnalysisDelivery::Ordered => {
                        if result_tx.send((frame_id, None)).is_err() {
                            return;
                        }
                    }
                }
            }
            next_frame_id = frame_id.wrapping_add(1);
            let t2 = raw
                .as_mut()
                .and_then(|input| input.observe(frame_id, hop.as_ref()));
            if let Some(packet) = &loo_pcm {
                loo_streams.retain(|s| {
                    packet.sources.iter().any(|pcm| {
                        pcm.kind == s.kind && pcm.id == s.id && pcm.generation == s.generation
                    })
                });
                for source in &packet.sources {
                    let index = loo_streams
                        .iter()
                        .position(|s| {
                            s.kind == source.kind
                                && s.id == source.id
                                && s.generation == source.generation
                        })
                        .unwrap_or_else(|| {
                            let observer = match source.kind {
                                crate::core::stream::loo::SourceKind::Voice => {
                                    crate::core::stream::loo::Observer::new(
                                        source.id,
                                        source.generation,
                                        &stream,
                                    )
                                }
                                crate::core::stream::loo::SourceKind::Flora => {
                                    crate::core::stream::loo::Observer::flora(
                                        source.id,
                                        source.generation,
                                        &mut stream,
                                    )
                                }
                            };
                            loo_streams.push(observer);
                            loo_streams.len() - 1
                        });
                    loo_streams[index].process(&hop, &source.habitat);
                }
            } else {
                loo_streams.clear();
            }
            let frame = stream.process(hop.as_ref());
            if let Some(tap) = temporal_tap.as_mut() {
                let mono_energy =
                    hop.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / hop.len() as f64;
                tap.observe_frame(frame_id, &frame.nsgt_power, mono_energy, t2);
            }
            spectral_history.observe(
                frame_id * hop_samples as u64,
                (frame_id + 1) * hop_samples as u64,
                &frame.nsgt_power,
            );
            warmup_samples = warmup_samples.saturating_sub(hop.len());
            analysis = (warmup_samples == 0).then_some(frame_id);
            if matches!(&delivery, AnalysisDelivery::Ordered)
                && let Some(id) = analysis.take()
            {
                let mut frame = frame.clone();
                frame.spectral_history = spectral_history.snapshot();
                frame.loo = loo_streams
                    .iter()
                    .filter(|s| s.kind == crate::core::stream::loo::SourceKind::Voice)
                    .map(|s| s.snapshot())
                    .collect();
                frame.flora_power = loo_streams
                    .iter()
                    .filter(|s| s.kind == crate::core::stream::loo::SourceKind::Flora)
                    .map(|s| s.resource_snapshot())
                    .collect();
                if result_tx.send((id, Some(frame))).is_err() {
                    return;
                }
            }
        }
        // Only the generator's current-state delivery may discard older snapshots.
        if let Some(id) = analysis
            && let AnalysisDelivery::Latest(stale) = &delivery
        {
            let mut frame = stream.last().clone();
            frame.spectral_history = spectral_history.snapshot();
            frame.loo = loo_streams
                .iter()
                .filter(|s| s.kind == crate::core::stream::loo::SourceKind::Voice)
                .map(|s| s.snapshot())
                .collect();
            frame.flora_power = loo_streams
                .iter()
                .filter(|s| s.kind == crate::core::stream::loo::SourceKind::Flora)
                .map(|s| s.resource_snapshot())
                .collect();
            publish_latest(&result_tx, stale, (id, Some(frame)));
        }
        #[cfg(test)]
        if let (Some(p), Some(begin)) = (&mut l1_probe, l1_begin) {
            p.after(next_frame_id - 1, begin, loo_streams.len(), hop_rx.len());
        }
        #[cfg(all(test, feature = "profile-alloc"))]
        if measure {
            let counts = crate::runtime_profile::finish_allocations().unwrap();
            println!(
                "L1_ANALYSIS_ALLOC loo={} next_frame={next_frame_id} counts={counts:?}",
                !loo_streams.is_empty()
            );
        }
    }
    // Publish the final observer state before disconnecting the analysis result channel.
    drop(raw);
    drop(temporal_tap);
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
            roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.005),
            harmonicity_kernel: HarmonicityKernel::new(space, HarmonicityParams::default()),
            consonance_kernel: ConsonanceKernel::default(),
            consonance_representation: ConsonanceRepresentationParams::default(),
            consonance_density_roughness_gain: 1.0,
            roughness_aversion: 1.0,
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
    fn latest_replaces_oldest_when_production_queue_is_full() {
        let (tx, rx) = crossbeam_channel::bounded(4);
        for id in 0..4 {
            tx.send((id, Some(Landscape::default()))).unwrap();
        }
        // A gap invalidation also has to pass a full queue without blocking.
        publish_latest(&tx, &rx, (4, None));
        publish_latest(&tx, &rx, (5, Some(Landscape::default())));
        let results: Vec<_> = rx.try_iter().collect();
        assert_eq!(
            results.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            [2, 3, 4, 5]
        );
        assert!(results[2].1.is_none());
        assert!(results[3].1.is_some());
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
            expected.push(reference.process(&audio).clone());
            hops.push((frame_id, audio, None));
        }
        let mut final_history = None;
        for capacity in [1, 4, 24] {
            let stream = AnalysisStream::new(params.clone(), RtNsgtKernelLog2::new(nsgt.clone()));
            let (hop_tx, hop_rx) = crossbeam_channel::unbounded();
            let (result_tx, result_rx) = crossbeam_channel::bounded(capacity);
            let (_update_tx, update_rx) = crossbeam_channel::unbounded();
            for row in &hops {
                hop_tx.send((row.0, Arc::clone(&row.1), None)).unwrap();
            }
            drop(hop_tx);
            let handle = thread::spawn(move || {
                run(
                    stream,
                    hop_rx,
                    result_tx,
                    update_rx,
                    AnalysisDelivery::Ordered,
                    None,
                    None,
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
            hop_tx.send((row.0, Arc::clone(&row.1), None)).unwrap();
        }
        drop(hop_tx);
        let tap = crate::temporal_cognition::observation::Tap::spawn(
            0,
            fs as u32,
            hop,
            512,
            reference.last().space.clone(),
            crate::temporal_cognition::observation::Options {
                deterministic: true,
                ..Default::default()
            },
        );
        let observed = Arc::clone(&tap.snapshot);
        run(
            AnalysisStream::new(params.clone(), RtNsgtKernelLog2::new(nsgt.clone())),
            hop_rx,
            result_tx,
            update_rx,
            AnalysisDelivery::Latest(result_rx.clone()),
            Some(tap),
            None,
        );
        let observation = *observed.lock().unwrap();
        assert_eq!(
            observation.state,
            crate::temporal_cognition::observation::ObservationState::Finished
        );
        assert_eq!(observation.received_frames, 24);
        assert_eq!(observation.fully_supported_frames, 21);
        assert_eq!(observation.frame_id, Some(23));
        assert_eq!(observation.support_start_sample, 24 * 128 - 512);
        assert_eq!(observation.support_end_sample, 24 * 128);
        assert!(observation.spectral_power_sum.unwrap() > 0.0);
        assert_eq!(observation.hop_start_sample, 23 * 128);
        let expected_energy: f64 = hops[23]
            .1
            .iter()
            .map(|&value| f64::from(value) * f64::from(value) / 128.0)
            .sum();
        assert!((observation.mono_mean_square.unwrap() - expected_energy).abs() < 1e-15);
        let trajectories = observation.trajectories.unwrap();
        assert!(trajectories.peak_bins.iter().any(Option::is_some));
        assert!((trajectories.energy.iter().sum::<f64>() - expected_energy).abs() < 1e-15);
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
                None,
                None,
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
        for (recovery_hops, ordered) in [1, 3, 4]
            .into_iter()
            .flat_map(|hops| [false, true].map(|ordered| (hops, ordered)))
        {
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
            hop_tx
                .send((0, tone.clone(), Some(packet(7, 0, &tone))))
                .unwrap();
            // A gap inside the already-queued backlog must reset the complete history.
            let silence: Arc<[f32]> = vec![0.0; hop].into();
            for frame_id in 2..2 + recovery_hops {
                hop_tx
                    .send((frame_id, Arc::clone(&silence), Some(packet(7, 0, &silence))))
                    .unwrap();
            }
            drop(hop_tx);
            let delivery = if ordered {
                AnalysisDelivery::Ordered
            } else {
                AnalysisDelivery::Latest(result_rx.clone())
            };
            run(stream, hop_rx, result_tx, update_rx, delivery, None, None);

            if ordered {
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
                let frame = frame.unwrap();
                assert!(frame.nsgt_power.iter().all(|power| *power == 0.0));
                assert!(frame.loo[0].h_scan.iter().all(|power| *power == 0.0));
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
        let (hop_tx, hop_rx) = crossbeam_channel::unbounded::<AnalysisHop>();
        let (result_tx, result_rx) = crossbeam_channel::unbounded::<AnalysisResult>();
        let (_update_tx, update_rx) = crossbeam_channel::unbounded::<LandscapeUpdate>();

        let delivery = AnalysisDelivery::Latest(result_rx.clone());
        let handle =
            thread::spawn(move || run(stream, hop_rx, result_tx, update_rx, delivery, None, None));

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
                .send((frame_idx, Arc::clone(&audio), None))
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
    fn packet(id: u64, generation: u32, own: &[f32]) -> crate::core::stream::loo::PcmPacket {
        let (recycle, _) = crossbeam_channel::unbounded();
        crate::core::stream::loo::PcmPacket {
            sources: vec![crate::core::stream::loo::SourcePcm {
                kind: crate::core::stream::loo::SourceKind::Voice,
                id,
                generation,
                habitat: own.to_vec(),
            }],
            recycle,
        }
    }

    #[test]
    fn flora_and_voice_share_numeric_id_without_sharing_history() {
        use crate::core::stream::loo::{PcmPacket, SourceKind, SourcePcm};
        let (nsgt, params) = crate::runtime::body_metabolism_test_core();
        let mut stream = AnalysisStream::new(params, nsgt);
        stream.enable_resource_power();
        let hop = stream.hop_samples();
        let (tx, rx) = crossbeam_channel::unbounded();
        let (result_tx, result_rx) = crossbeam_channel::unbounded();
        let (_, updates) = crossbeam_channel::unbounded();
        let (recycle, _) = crossbeam_channel::unbounded();
        for frame in 0..40 {
            let mix: Arc<[f32]> = vec![0.03; hop].into();
            let sources = vec![
                SourcePcm {
                    kind: SourceKind::Voice,
                    id: 7,
                    generation: 0,
                    habitat: mix.to_vec(),
                },
                SourcePcm {
                    kind: SourceKind::Flora,
                    id: 7,
                    generation: u32::from(frame >= 36),
                    habitat: vec![0.0; hop],
                },
            ];
            tx.send((
                frame,
                mix,
                Some(PcmPacket {
                    sources,
                    recycle: recycle.clone(),
                }),
            ))
            .unwrap();
        }
        drop(tx);
        run(
            stream,
            rx,
            result_tx,
            updates,
            AnalysisDelivery::Ordered,
            None,
            None,
        );
        for (id, frame) in result_rx.try_iter() {
            let frame = frame.unwrap();
            assert_eq!(frame.loo.len(), 1);
            assert_eq!(frame.flora_power.len(), 1);
            assert!(frame.loo[0].h_scan.iter().all(|&x| x == 0.0));
            let observation = &frame.flora_power[0];
            assert_eq!(
                (observation.id, observation.generation),
                (7, u32::from(id >= 36))
            );
            if id >= 32 {
                assert!(
                    observation
                        .power_scan
                        .as_ref()
                        .unwrap()
                        .iter()
                        .any(|&p| p > 0.0)
                );
            }
        }
    }

    #[test]
    fn loo_birth_history_and_generation_share_publication_frame() {
        let (nsgt, params) = crate::runtime::body_metabolism_test_core();
        let template = AnalysisStream::new(params.clone(), nsgt);
        let hop = template.hop_samples();
        for ordered in [false, true] {
            let mut reference = template.clone();
            let (tx, rx) = crossbeam_channel::unbounded();
            let (result_tx, result_rx) = crossbeam_channel::unbounded();
            let (_, updates) = crossbeam_channel::unbounded();
            let mut expected = Vec::new();
            for id in 0..40 {
                let mix: Arc<[f32]> = (0..hop)
                    .map(|i| ((id * hop + i) as f32 * 0.11).sin() * 0.2)
                    .collect();
                let own: Vec<_> = mix.iter().map(|x| x * 0.5).collect();
                let active = (2..38).contains(&id);
                // Before birth the Voice contributes no PCM; copy the complete mixed history.
                let removed: Vec<_> = mix
                    .iter()
                    .zip(&own)
                    .map(|(a, b)| if active { a - b } else { *a })
                    .collect();
                let mut frame = reference.process(&removed).clone();
                frame.recompute_consonance(&params);
                expected.push(active.then_some(frame.consonance_field_score));
                tx.send((id as u64, mix, active.then(|| packet(7, 2, &own))))
                    .unwrap();
            }
            drop(tx);
            let delivery = if ordered {
                AnalysisDelivery::Ordered
            } else {
                AnalysisDelivery::Latest(result_rx.clone())
            };
            run(
                template.clone(),
                rx,
                result_tx,
                updates,
                delivery,
                None,
                None,
            );
            let frames: Vec<_> = result_rx.try_iter().collect();
            assert_eq!(frames.len(), if ordered { 40 } else { 1 });
            for (id, frame) in frames {
                let frame = frame.unwrap();
                if let Some(expected) = &expected[id as usize] {
                    assert_eq!(frame.loo.len(), 1);
                    assert_eq!((frame.loo[0].id, frame.loo[0].generation), (7, 2));
                    assert_eq!(&*frame.loo[0].score_scan, expected);
                } else {
                    assert!(frame.loo.is_empty());
                }
            }
        }
    }

    #[test]
    fn pcm_loo_latest_publishes_active_final_hop() {
        let (nsgt, params) = crate::runtime::body_metabolism_test_core();
        let template = AnalysisStream::new(params.clone(), nsgt);
        let mut reference = template.clone();
        let hop = template.hop_samples();
        let (tx, rx) = crossbeam_channel::unbounded();
        let (result_tx, result_rx) = crossbeam_channel::unbounded();
        let (_, updates) = crossbeam_channel::unbounded();
        for id in 0..40 {
            let mix: Arc<[f32]> = (0..hop)
                .map(|i| ((id * hop + i) as f32 * 0.11).sin() * 0.2)
                .collect();
            let own: Vec<_> = mix.iter().map(|x| x * 0.5).collect();
            let active = id >= 2;
            let removed: Vec<_> = mix
                .iter()
                .zip(&own)
                .map(|(mix, own)| if active { mix - own } else { *mix })
                .collect();
            reference.process(&removed);
            tx.send((id as u64, mix, active.then(|| packet(7, 2, &own))))
                .unwrap();
        }
        drop(tx);
        run(
            template,
            rx,
            result_tx,
            updates,
            AnalysisDelivery::Latest(result_rx.clone()),
            None,
            None,
        );
        let frames: Vec<_> = result_rx.try_iter().collect();
        assert_eq!(frames.len(), 1);
        let (id, frame) = &frames[0];
        assert_eq!(*id, 39);
        let frame = frame.as_ref().unwrap();
        assert_eq!(
            frame.loo.len(),
            1,
            "Latest must publish the final active observer"
        );
        let observation = &frame.loo[0];
        assert_eq!((observation.id, observation.generation), (7, 2));
        let mut expected = reference.last().clone();
        expected.recompute_consonance(&params);
        assert!(expected.harmonicity01.iter().any(|h| *h > 0.));
        assert_eq!(observation.h_scan, expected.harmonicity01);
        assert_eq!(observation.r_scan, expected.roughness01);
        assert_eq!(&*observation.score_scan, &expected.consonance_field_score);
    }

    #[test]
    fn pcm_loo_generation_starts_from_mixed_not_old_exclusion_history() {
        let (nsgt, params) = crate::runtime::body_metabolism_test_core();
        let template = AnalysisStream::new(params.clone(), nsgt);
        let hop = template.hop_samples();
        let mut mixed = template.clone();
        let mut old_reference = template.clone();
        let mut new_reference = None;
        let mut fresh_reference = template.clone();
        let (tx, rx) = crossbeam_channel::unbounded();
        let (result_tx, result_rx) = crossbeam_channel::unbounded();
        let (_, updates) = crossbeam_channel::unbounded();
        let mut expected = Vec::new();
        for id in 0..24 {
            let old: Vec<_> = (0..hop)
                .map(|i| {
                    let t = (id * hop + i) as f32;
                    if id < 2 { 0. } else { (t * 0.07).sin() * 0.2 }
                })
                .collect();
            let own: Vec<_> = (0..hop)
                .map(|i| {
                    if id < 12 {
                        old[i]
                    } else {
                        ((id * hop + i) as f32 * 0.11).sin() * 0.15
                    }
                })
                .collect();
            let mix: Arc<[f32]> = (0..hop)
                .map(|i| {
                    let other = ((id * hop + i) as f32 * 0.031).sin() * 0.07;
                    old[i] + other + if id >= 12 { own[i] } else { 0. }
                })
                .collect();
            if id == 12 {
                // The old Voice stays in the mixed history and is now another source.
                new_reference = Some(mixed.clone());
            }
            let removed: Vec<_> = mix
                .iter()
                .zip(&own)
                .map(|(mix, own)| if id >= 2 { mix - own } else { *mix })
                .collect();
            old_reference.process(&removed);
            let mut frame = if let Some(reference) = &mut new_reference {
                reference.process(&removed).clone()
            } else {
                old_reference.last().clone()
            };
            frame.recompute_consonance(&params);
            if id >= 12 {
                let mut stale = old_reference.last().clone();
                stale.recompute_consonance(&params);
                let mut fresh = fresh_reference.process(&removed).clone();
                fresh.recompute_consonance(&params);
                if id == 12 {
                    assert_ne!(
                        frame.consonance_field_score, stale.consonance_field_score,
                        "fixture must distinguish the old exclusion history"
                    );
                    assert_ne!(
                        frame.consonance_field_score, fresh.consonance_field_score,
                        "fixture must distinguish a zero-history birth"
                    );
                }
            }
            expected.push(
                (id >= 2).then_some((if id < 12 { 2 } else { 3 }, frame.consonance_field_score)),
            );
            mixed.process(&mix);
            tx.send((
                id as u64,
                mix,
                (id >= 2).then(|| packet(7, if id < 12 { 2 } else { 3 }, &own)),
            ))
            .unwrap();
        }
        drop(tx);
        run(
            template,
            rx,
            result_tx,
            updates,
            AnalysisDelivery::Ordered,
            None,
            None,
        );
        let frames: Vec<_> = result_rx.try_iter().collect();
        assert_eq!(frames.len(), 24);
        for (id, frame) in frames {
            let frame = frame.unwrap();
            if let Some((generation, score)) = &expected[id as usize] {
                assert_eq!(frame.loo.len(), 1);
                assert_eq!((frame.loo[0].id, frame.loo[0].generation), (7, *generation));
                assert_eq!(&*frame.loo[0].score_scan, score, "frame {id}");
            } else {
                assert!(frame.loo.is_empty());
            }
        }
    }

    #[cfg(feature = "profile-alloc")]
    #[test]
    fn loo_analysis_steady_state_allocation_census() {
        let (nsgt, params) = crate::runtime::body_metabolism_test_core();
        let template = AnalysisStream::new(params, nsgt);
        let hop = template.hop_samples();
        for active in [false, true] {
            let (tx, rx) = crossbeam_channel::bounded(1);
            let (result_tx, result_rx) = crossbeam_channel::bounded::<AnalysisResult>(1);
            let (_, updates) = crossbeam_channel::unbounded();
            let producer = thread::spawn(move || {
                let audio: Arc<[f32]> = (0..hop).map(|i| (i as f32 * 0.13).sin() * 0.2).collect();
                for id in 0..12 {
                    tx.send((id, audio.clone(), active.then(|| packet(7, 0, &audio))))
                        .unwrap();
                    let (actual_id, frame) = result_rx.recv().unwrap();
                    assert_eq!(actual_id, id);
                    assert_eq!(!frame.unwrap().loo.is_empty(), active);
                }
            });
            run(
                template.clone(),
                rx,
                result_tx,
                updates,
                AnalysisDelivery::Ordered,
                None,
                None,
            );
            producer.join().unwrap();
        }
    }
}
