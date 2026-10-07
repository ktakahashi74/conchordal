use super::*;
use crate::core::log2space::Log2Space;
use crate::core::nsgt_kernel::{NsgtKernelLog2, NsgtLog2Config, PowerMode};
use crate::life::sound::BodyKind;

fn analyzer() -> RtNsgtKernelLog2 {
    RtNsgtKernelLog2::new(NsgtKernelLog2::new(
        NsgtLog2Config {
            fs: 8000.,
            overlap: 0.75,
            nfft_override: Some(256),
            ..Default::default()
        },
        Log2Space::new(100., 2000., 12),
        None,
        PowerMode::Coherent,
    ))
}

fn config() -> TemporalBodyConfig {
    TemporalBodyConfig {
        means: [0.; 6],
        deviations: [1.; 6],
        accent_means: [0.; 2],
        accent_deviations: [1.; 2],
    }
}

fn recipe() -> BodySnapshot {
    BodySnapshot {
        kind: BodyKind::Sine,
        amp_scale: 1.,
        brightness: 0.,
        inharmonic: 0.,
        spread: 0.,
        unison: 1,
        motion: 0.,
        ratios: None,
    }
}

#[test]
fn offline_delivery_waits_for_publication_while_live_delivery_stays_nonblocking() {
    use std::time::Duration;
    for deterministic in [true, false] {
        let mut capture = Capture::spawn(analyzer(), config(), deterministic, None);
        capture.prepare(std::iter::once((1, 0, recipe())), 0);
        capture.begin(0);
        let snapshot = Arc::clone(&capture.snapshot);
        let held = snapshot.lock().unwrap();
        let (started_tx, started_rx) = bounded(1);
        let (done_tx, done_rx) = bounded(1);
        let sender = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            capture.end();
            done_tx.send(()).unwrap();
            capture
        });
        started_rx.recv().unwrap();
        if deterministic {
            assert_eq!(
                done_rx.recv_timeout(Duration::from_millis(20)),
                Err(crossbeam_channel::RecvTimeoutError::Timeout)
            );
        } else {
            done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
        }
        drop(held);
        if deterministic {
            done_rx.recv_timeout(Duration::from_secs(5)).unwrap();
            let published = *snapshot.lock().unwrap();
            assert_eq!((published.processed_frames, published.input_end), (1, 64));
            assert_eq!(published.version, 1);
        }
        let mut capture = sender.join().unwrap();
        if deterministic {
            // Include frames without a descriptor publication and both bus records.
            for start in (64..832).step_by(64) {
                capture.begin(start);
                capture.end();
                assert_eq!(capture.free.len(), BUFFERS);
            }
            let published = capture.snapshot();
            for record in &published.records[..2] {
                assert!(record.active);
                assert_eq!(record.end, 832);
                assert_eq!(record.available, record.end);
            }
        }
        capture.finish();
        let final_state = capture.snapshot();
        assert!(final_state.finished);
        assert_eq!(
            final_state.processed_frames,
            if deterministic { 13 } else { 1 }
        );
        assert_eq!(final_state.capture_drops, 0);
        assert_eq!(capture.free.len(), BUFFERS);
        let resources = final_state.worker_resources;
        assert_eq!(resources.frames.count, final_state.processed_frames);
        assert_eq!(resources.windows.total_ns, resources.frames.total_ns);
        assert_eq!(resources.delivery.count, final_state.processed_frames);
        assert_eq!(resources.finish.count, 1);
        assert!(resources.open_window.is_none());
        assert_eq!(
            resources.frames.histogram.iter().sum::<u64>(),
            resources.frames.count
        );
        assert!(resources.bucket_upper_us.contains(&40_000));
    }
}

#[test]
fn prototype_assignments_follow_worker_generation_and_retirement() {
    use crate::config::{TemporalBodyMedoid, TemporalBodyPrototypesConfig};
    use crate::temporal_cognition::body_model::Prototypes;
    let mut values = [0.; 6];
    values[2] = 1e-6_f64.log2();
    let model = TemporalBodyPrototypesConfig {
        model_version: "56".repeat(32),
        sample_rate: 8000,
        nfft: 256,
        hop_size: 64,
        means: [0.; 6],
        deviations: [1.; 6],
        accent_means: [0.; 2],
        accent_deviations: [1.; 2],
        medoids: vec![TemporalBodyMedoid {
            record_id: "known-silence".into(),
            raw_values: values,
            mask: 4,
        }],
    };
    let mut capture = Capture::spawn(
        analyzer(),
        config(),
        true,
        Some(Prototypes::new(&model, config())),
    );
    capture.prepare(std::iter::once((1, 0, recipe())), 0);
    let advance = |capture: &mut Capture, start: u64, hops: u64| {
        for i in 0..hops {
            capture.begin(start + i * 64);
            capture.end();
        }
    };
    // Offline delivery publishes each frame before the next generation step.
    advance(&mut capture, 0, 64);
    let before = capture.snapshot();
    assert!(
        before.prototype_assignments[..2]
            .iter()
            .all(Option::is_some)
    );
    let old_generation = before.records[0].body_generation;
    let mut replacement = recipe();
    replacement.unison = 2;
    capture.prepare(std::iter::once((1, 0, replacement)), 4096);
    advance(&mut capture, 4096, 6);
    let resetting = capture.snapshot();
    assert!(resetting.records[..2].iter().all(|r| !r.active));
    assert!(
        resetting.prototype_assignments[..2]
            .iter()
            .all(Option::is_none)
    );
    advance(&mut capture, 4480, 64);
    let after = capture.snapshot();
    assert!(after.prototype_assignments[..2].iter().all(Option::is_some));
    assert_ne!(after.records[0].body_generation, old_generation);
    capture.prepare(std::iter::empty(), 8576);
    advance(&mut capture, 8576, 6);
    capture.finish();
    let retired = capture.snapshot();
    assert!(retired.records.iter().all(|r| !r.active));
    assert!(retired.prototype_assignments.iter().all(Option::is_none));
    assert_eq!(retired.prototype_model_version, Some([0x56; 32]));
}

#[test]
fn body_window_matches_independent_pcm_and_spectral_integrals() {
    let nsgt = analyzer();
    let space = nsgt.space().clone();
    let mut reference = nsgt.clone();
    let mut lane = Lane::new(nsgt, 0, config());
    let owner = Owner {
        id: 5,
        generation: 3,
        body_generation: 1,
        born: 0,
        changed: 0,
    };
    let mut rows = Vec::new();
    let mut latest = None;
    let mut previous_energy: Option<f64> = None;
    let mut previous_scan: Option<Vec<f64>> = None;
    for start in (0..20032).step_by(64) {
        let pcm: Vec<f32> = (start..start + 64)
            .map(|t| {
                let amplitude = if start < 8000 { 0.1 } else { 0.2 };
                amplitude * (std::f32::consts::TAU * 250. * t as f32 / 8000.).sin()
            })
            .collect();
        let energy = pcm.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / 64.;
        let power = reference.process_hop(&pcm);
        let mass: f64 = power.iter().map(|v| f64::from(*v)).sum();
        let scan: Vec<_> = power
            .iter()
            .map(|v| f64::from(*v) / mass * energy)
            .collect();
        let shape = start + 64 >= 256;
        let center = scan
            .iter()
            .zip(&space.centers_log2)
            .map(|(e, f)| e * f64::from(*f))
            .sum::<f64>()
            / energy;
        let second = scan
            .iter()
            .zip(&space.centers_log2)
            .map(|(e, f)| e * f64::from(*f).powi(2))
            .sum::<f64>()
            / energy;
        let rise = previous_energy.map(|e| (energy.sqrt().log2() - e.sqrt().log2()).max(0.));
        let flux = if shape {
            previous_scan.as_ref().map(|old| {
                scan.iter()
                    .zip(old)
                    .map(|(a, b)| (0.5 * a.max(1e-12).log2() - 0.5 * b.max(1e-12).log2()).max(0.))
                    .sum::<f64>()
                    / scan.len() as f64
            })
        } else {
            None
        };
        rows.push((start as u64, energy, shape, center, second, rise, flux));
        previous_energy = Some(energy);
        previous_scan = shape.then_some(scan);
        if let Some(record) = lane.push(owner, start as u64, &pcm, true).unwrap() {
            latest = Some(record);
        }
    }
    let record = latest.unwrap();
    assert_eq!(record.mask, 0b111111);
    let mut total_energy = 0.;
    let mut spectral_energy = 0.;
    let mut center = 0.;
    let mut second = 0.;
    let mut rise = (0., 0_u64);
    let mut flux = (0., 0_u64);
    for (start, energy, shape, c, s, r, f) in rows {
        if start + 64 > record.end {
            continue;
        }
        let duration = (start + 64).saturating_sub(start.max(record.start));
        if duration == 0 {
            continue;
        }
        total_energy += duration as f64 * energy;
        if shape {
            spectral_energy += duration as f64 * energy;
            center += duration as f64 * energy * c;
            second += duration as f64 * energy * s;
        }
        if let Some(value) = r {
            rise.0 += value * duration as f64;
            rise.1 += duration;
        }
        if let Some(value) = f {
            flux.0 += value * duration as f64;
            flux.1 += duration;
        }
    }
    center /= spectral_energy;
    let expected = [
        center,
        (second / spectral_energy - center * center).max(0.).sqrt(),
        (total_energy / (record.end - record.start) as f64)
            .sqrt()
            .log2(),
        rise.0 / rise.1 as f64,
        flux.0 / flux.1 as f64,
    ];
    for (actual, expected) in record.raw_values[..5].iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-10, "{actual} != {expected}");
    }
    assert_eq!(record.end - record.start, 16000);
    assert_eq!(std::mem::size_of::<Record>(), 128);
}

#[test]
fn young_silent_gapped_and_replaced_bodies_keep_distinct_support() {
    let mut lane = Lane::new(analyzer(), 1, config());
    let mut owner = Owner {
        id: 5,
        generation: 2,
        body_generation: 9,
        born: 6400,
        changed: 6400,
    };
    let mut record = None;
    for start in (6400..10048).step_by(64) {
        if let Some(value) = lane.push(owner, start, &[0.; 64], true).unwrap() {
            record = Some(value);
        }
    }
    let silent = record.unwrap();
    assert_eq!(silent.start, 6400);
    assert!(silent.end - silent.start < 16000);
    assert_eq!(silent.mask & 0b11, 0);
    assert_eq!(
        silent.mask & 0b111100,
        0b111100,
        "{:?}; nfft={}",
        silent,
        lane.nsgt.nfft()
    );
    assert_eq!(silent.raw_values[2], 1e-6_f64.log2());
    assert_eq!(silent.raw_values[5], 0.);
    let gap = lane.push(owner, 11200, &[0.; 64], true).unwrap().unwrap();
    assert_eq!(gap.mask, 0);
    assert!(gap.coverage[2] < 0.9);
    owner.body_generation += 1;
    owner.changed = 11264;
    for start in (11264..12928).step_by(64) {
        if let Some(value) = lane.push(owner, start, &[0.25; 64], true).unwrap() {
            record = Some(value);
        }
    }
    let replaced = record.unwrap();
    assert_eq!(replaced.start, 11264);
    assert_eq!(replaced.body_generation, 10);
    assert_eq!(replaced.raw_values[2], -2.);
    assert_eq!(replaced.source_generation, 2);
}

#[test]
fn capture_generation_pool_exhaustion_and_population_limits_are_explicit() {
    let mut capture = Capture::spawn(analyzer(), config(), false, None);
    capture.prepare((0..65).map(|id| (id, 0, recipe())), 0);
    assert_eq!(capture.snapshot().outside_voice_hops, 1);
    let old = capture.token(0, 0).unwrap();
    let mut continuous = recipe();
    continuous.brightness = 0.5;
    capture.prepare(std::iter::once((0, 0, continuous.clone())), 64);
    assert_eq!(capture.token(0, 0), Some(old));
    continuous.unison = 2;
    capture.prepare(std::iter::once((0, 0, continuous)), 128);
    let new = capture.token(0, 0).unwrap();
    assert_eq!(old.0, new.0);
    assert_ne!(old.1, new.1);
    capture.begin(128);
    capture.sample(
        old,
        0,
        0.5,
        crate::scenario::control::Routing {
            to_habitat: true,
            to_presentation: false,
        },
    );
    assert!(!capture.frame.as_ref().unwrap().supported[old.0 * 2]);
    assert!(capture.frame.as_ref().unwrap().supported[old.0 * 2 + 1]);
    capture.end();
    capture.finish();
    assert!(capture.snapshot().finished);
    assert_eq!(capture.snapshot().invalid_hops, 1);
    assert!(capture.snapshot().records.iter().all(|r| !r.active));
    assert!(capture.free.len() == BUFFERS);

    let mut capture = Capture::spawn(analyzer(), config(), false, None);
    let held: Vec<_> = (0..BUFFERS).map(|_| capture.free.recv().unwrap()).collect();
    capture.begin(0);
    capture.end();
    assert_eq!(capture.snapshot().capture_drops, 1);
    assert!(capture.frame.is_none());
    capture.finish();
    drop(held);
}

#[test]
fn diagnostic_scales_are_explicit_and_shared_with_acoustic_groups() {
    let mut app = crate::config::AppConfig::default();
    app.temporal_body = Some(config());
    app.validate().unwrap();
    app.temporal_body.as_mut().unwrap().deviations[1] = -1.;
    assert!(app.validate().is_err());
    app.temporal_body = Some(config());
    app.temporal_body.as_mut().unwrap().accent_means[0] = f64::NAN;
    assert!(app.validate().is_err());
    app.temporal_body = Some(config());
    app.temporal_acoustic = Some(crate::config::TemporalAcousticConfig {
        group_means: [0.; 3],
        group_deviations: [0.05, 4., 1.],
        accent_means: [0.; 2],
        accent_deviations: [1.; 2],
        group_retirement_sec: 2.,
        inactive_energy_max: 1e-8,
        correlation_window_sec: 0.25,
        min_pairs: 4,
        min_coverage: 0.9,
        persistence_hops: 2,
    });
    app.temporal_ridge = Some(crate::config::TemporalRidgeConfig {
        means: [0.; 3],
        deviations: [1.; 3],
    });
    app.validate().unwrap();
    app.temporal_body.as_mut().unwrap().accent_deviations[0] = 2.;
    assert!(
        app.validate()
            .unwrap_err()
            .to_string()
            .contains("share frozen accent scales")
    );
}

#[test]
fn private_capture_matches_isolated_render_for_every_body_and_route() {
    use crate::core::{modulation::NeuralRhythms, timebase::Timebase};
    use crate::life::sound::RenderModulatorSpec;
    use crate::life::{
        phonation_engine::{OnsetKick, ToneCmd},
        schedule_renderer::ScheduleRenderer,
        voice::{PhonationBatch, ToneSpec},
    };
    use crate::scenario::control::Routing;
    for kind in [BodyKind::Sine, BodyKind::Harmonic, BodyKind::Modal] {
        for routing in [(true, true), (true, false), (false, true), (false, false)] {
            let mut body = recipe();
            body.kind = kind;
            body.brightness = 0.4;
            let actor = PhonationBatch {
                source_id: 5,
                source_generation: 2,
                routing: Routing {
                    to_habitat: routing.0,
                    to_presentation: routing.1,
                },
                cmds: vec![ToneCmd::On {
                    tone_id: 1,
                    kick: OnsetKick { strength: 1. },
                }],
                tones: vec![ToneSpec {
                    opportunity: None,
                    tone_id: 1,
                    onset: 0,
                    hold_ticks: Some(8000),
                    freq_hz: 250.,
                    amp: 0.2,
                    smoothing_tau_sec: 0.,
                    body: body.clone(),
                    render_modulator: RenderModulatorSpec::SeqGate { duration_sec: 2. },
                    adsr: None,
                }],
                ..Default::default()
            };
            let mut other = actor.clone();
            other.source_id = 6;
            other.tones[0].freq_hz = 1000.;
            other.tones[0].amp = 0.7;
            let all = [actor.clone(), other];
            let time = Timebase { fs: 8000., hop: 64 };
            let mut tracked = ScheduleRenderer::new(time);
            let mut capture = Capture::spawn(analyzer(), config(), true, None);
            capture.prepare(std::iter::once((5, 2, body)), 0);
            tracked.body_capture = Some(capture);
            let mut control = ScheduleRenderer::new(time);
            let mut isolated = ScheduleRenderer::new(time);
            let mut pcm = [Vec::new(), Vec::new()];
            for now in (0..6400).step_by(64) {
                let batches = if now == 0 { &all[..] } else { &[] };
                let actual = tracked.render(batches, now, &NeuralRhythms::default());
                let reference = control.render(batches, now, &NeuralRhythms::default());
                assert_eq!(actual.habitat, reference.habitat);
                assert_eq!(actual.presentation, reference.presentation);
                let private = isolated.render(
                    if now == 0 {
                        std::slice::from_ref(&actor)
                    } else {
                        &[]
                    },
                    now,
                    &NeuralRhythms::default(),
                );
                pcm[0].extend_from_slice(private.habitat);
                pcm[1].extend_from_slice(private.presentation);
            }
            let capture = tracked.body_capture.as_mut().unwrap();
            capture.finish();
            let snapshot = capture.snapshot();
            assert_eq!(snapshot.capture_drops, 0);
            assert_eq!(snapshot.invalid_hops, 0);
            assert_eq!(snapshot.records.iter().filter(|r| r.active).count(), 2);
            for record in snapshot.records.iter().filter(|r| r.active) {
                let samples = &pcm[record.bus as usize][record.start as usize..record.end as usize];
                let expected = (samples.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>()
                    / samples.len() as f64)
                    .sqrt()
                    .max(1e-6)
                    .log2();
                assert!((record.raw_values[2] - expected).abs() < 1e-12);
                assert!(record.mask & 4 != 0);
                assert_eq!(record.source_id, 5);
                assert_eq!(record.source_generation, 2);
                assert!(record.available >= record.end);
            }
        }
    }
}
