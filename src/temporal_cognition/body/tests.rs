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
fn scheduled_release_forecast_switches_at_delivery_hop_without_anticipating_audio() {
    use crate::core::modulation::NeuralRhythms;
    use crate::core::timebase::Timebase;
    use crate::life::{
        action_candidates::{OnsetOpportunity, PolicySnapshot},
        action_observation::{ActionKind, Observer},
        phonation_engine::{OnsetKick, ToneCmd},
        schedule_renderer::ScheduleRenderer,
        sound::{RenderModulatorSpec, ToneAdsr},
        voice::{PhonationBatch, ToneSpec},
    };
    for (off, hold) in [
        (13, 8000),
        (63, 8000),
        (64, 8000),
        (77, 8000),
        (160, 8000),
        (77, 40),
    ] {
        for fault in ["none", "absent", "stale", "onset", "past_release", "policy"] {
            let mut actor = PhonationBatch {
                source_id: 5,
                body_policy: Some(PolicySnapshot {
                    at: 0,
                    is_alive: true,
                    gate_allows_onset: true,
                }),
                cmds: vec![ToneCmd::On {
                    tone_id: 1,
                    kick: OnsetKick { strength: 1. },
                }],
                tones: vec![ToneSpec {
                    opportunity: Some(OnsetOpportunity {
                        issued_at: 0,
                        at: 13,
                        gate: 0,
                        intrinsic_due_at: Some(13),
                        intrinsic_period_ticks: Some(4000),
                        planned_release_at: Some(off),
                    }),
                    tone_id: 1,
                    onset: 13,
                    hold_ticks: Some(hold),
                    freq_hz: 250.,
                    amp: 0.2,
                    smoothing_tau_sec: 0.,
                    body: recipe(),
                    render_modulator: RenderModulatorSpec::SeqGate { duration_sec: 2. },
                    adsr: Some(ToneAdsr {
                        attack_sec: 0.012,
                        decay_sec: 0.,
                        sustain_level: 1.,
                        release_sec: 0.004,
                    }),
                }],
                ..Default::default()
            };
            match fault {
                "absent" => actor.tones[0].opportunity = None,
                "stale" => actor.tones[0].opportunity.as_mut().unwrap().issued_at = 1,
                "onset" => actor.tones[0].opportunity.as_mut().unwrap().at = 14,
                "past_release" => {
                    actor.tones[0]
                        .opportunity
                        .as_mut()
                        .unwrap()
                        .planned_release_at = Some(12)
                }
                "policy" => actor.body_policy.as_mut().unwrap().gate_allows_onset = false,
                _ => {}
            }
            let time = Timebase { fs: 8000., hop: 64 };
            let mut tracked = ScheduleRenderer::new(time);
            let mut capture = Capture::spawn(analyzer(), config(), true, None);
            capture.prepare(std::iter::once((5, 0, recipe())), 0);
            tracked.body_capture = Some(capture);
            let mut observer = Observer::new(8000);
            observer.enable_predictions();
            tracked.action_observer = Some(Box::new(observer));
            let mut control = ScheduleRenderer::new(time);
            let mut outcomes = Vec::new();
            for now in [0, 64, 128] {
                let mut batch = actor.clone();
                if now > 0 {
                    batch.cmds.clear();
                    batch.tones.clear();
                }
                if now <= off && off < now + 64 {
                    batch.cmds.push(ToneCmd::Off {
                        tone_id: 1,
                        off_tick: off,
                    });
                }
                let mut unobserved = batch.clone();
                for tone in &mut unobserved.tones {
                    tone.opportunity = None;
                }
                let actual =
                    tracked.render(std::slice::from_ref(&batch), now, &NeuralRhythms::default());
                let baseline = control.render(
                    std::slice::from_ref(&unobserved),
                    now,
                    &NeuralRhythms::default(),
                );
                assert_eq!(actual.habitat, baseline.habitat);
                assert_eq!(actual.presentation, baseline.presentation);
                for (_, envelope) in tracked.source_envelopes(5, 0) {
                    assert_eq!(
                        envelope.hold_end,
                        if off < now + 64 {
                            off.min(13 + hold)
                        } else {
                            13 + hold
                        }
                    );
                }
                outcomes.extend(tracked.action_observer.as_mut().unwrap().drain());
            }
            let forecast = outcomes
                .iter()
                .find(|o| o.action == ActionKind::Onset)
                .unwrap()
                .prediction
                .unwrap();
            assert_eq!(forecast.scheduled_release.is_some(), fault == "none");
            if let Some(plan) = forecast.scheduled_release {
                assert_eq!(
                    (plan.apply_at_sample, plan.off_sample),
                    (off / 64 * 64, off)
                );
            }
            assert_eq!(forecast.envelope.hold_end, 13 + hold);
            let mut changed = 0;
            for energy in forecast.source_energy {
                for (k, [left, right]) in energy.windows.into_iter().enumerate() {
                    let tick = left + (right - left) / 2;
                    let held = 13 + hold;
                    let end = if fault == "none" && tick >= off / 64 * 64 {
                        held.min(off)
                    } else {
                        held
                    };
                    let attack = 96.min(end.saturating_sub(13).max(1));
                    let gain = if tick >= end + 32 {
                        0.
                    } else {
                        let level = ((tick - 13 + 1) as f32 / attack as f32).min(1.);
                        let release = if tick < end {
                            1.
                        } else {
                            (end + 32 - tick) as f32 / 32.
                        };
                        level * release
                    };
                    let expected = f64::from(0.2_f32).powi(2) * f64::from(gain).powi(2) / 2.;
                    assert_eq!(
                        energy.incoherent_fixed_energy[k],
                        Some(expected),
                        "{off} {hold} {fault} {tick}"
                    );
                    let old = f64::from(0.2_f32).powi(2)
                        * f64::from(forecast.envelope.gain_at(tick)).powi(2)
                        / 2.;
                    changed += usize::from(expected != old);
                    assert!(
                        (energy.predictions[1][k].unwrap() - energy.predictions[0][k].unwrap())
                            .abs()
                            < 1e-15
                    );
                    assert!((energy.predictions[2][k].unwrap() - 1e-4).abs() < 1e-15);
                }
            }
            if fault == "none" && hold == 8000 {
                assert!(changed > 0);
            }
            tracked.body_capture.as_mut().unwrap().finish();
        }
    }
}

#[test]
fn source_energy_teacher_preserves_interference_routes_and_generation_support() {
    use crate::life::action_observation::{ActionKind, Observer};
    use crate::life::voice::PhonationBatch;
    use crate::scenario::control::Routing;
    for phase in [-1., 1.] {
        for failure in ["none", "missing_hop", "nonfinite_sample"] {
            let mut capture = Capture::spawn(analyzer(), config(), true, None);
            capture.prepare(std::iter::once((1, 0, recipe())), 0);
            let token = capture.token(1, 0).unwrap();
            let mut observer = Observer::new(8000);
            observer.enable_predictions();
            let command = observer
                .command(
                    &PhonationBatch {
                        source_id: 1,
                        routing: Routing::default(),
                        ..Default::default()
                    },
                    1,
                    Some(0),
                    0,
                    "accepted",
                    ActionKind::Onset,
                )
                .unwrap();
            let envelope = crate::life::sound::envelope::Envelope {
                onset: 0,
                hold_end: 1000,
                release_end: 1100,
                attack_ticks: 1,
                decay_ticks: 0,
                sustain_level: 1.,
                decay_lambda: 0.,
                release_ticks: 100,
            };
            let mut input = capture
                .prediction_input(token, 0, 0, (220., 0.1, envelope))
                .unwrap();
            input.control = Some(crate::life::sound::control_forecast::ControlForecast {
                issued_at: 0,
                valid_until: None,
                amplitude_smoothing: None,
                amplitude_updates: None,
                sample_dt: 1. / 8000.,
                starts_at: Some(0),
                kick_at: None,
                model: crate::life::sound::control_forecast::AmplitudeModel::Unmodulated {
                    gain: 1.,
                },
            });
            observer.predict(command, input, std::iter::empty());
            let waveform = |tick: u64| 0.1_f32 + (tick % 23) as f32 * 0.001;
            for now in [0, 64, 128] {
                capture.begin(now);
                observer.begin_hop(now);
                for i in 0..64 {
                    let value = waveform(now + i as u64);
                    capture.sample(token, i, value, Routing::default());
                    capture.sample(
                        token,
                        i,
                        phase * value,
                        Routing {
                            to_habitat: false,
                            to_presentation: true,
                        },
                    );
                    observer.sample(command, (1, 0, 1), now + i as u64, value, false);
                }
                if failure == "missing_hop" && now == 64 {
                    capture.frame.as_mut().unwrap().supported[token.0 * 2] = false;
                }
                if failure == "nonfinite_sample" && now == 64 {
                    capture.sample(
                        token,
                        7,
                        f32::NAN,
                        Routing {
                            to_habitat: true,
                            to_presentation: false,
                        },
                    );
                }
                assert!(capture.source_audio(token.0, (1, 0, token.1 + 1)).is_none());
                observer.observe_source_energy(&capture);
                observer.end_hop(now, now + 64);
                capture.end();
            }
            let out = observer.drain().next().unwrap();
            let energy = out.prediction.unwrap().source_energy;
            for (k, window) in energy[0].windows.iter().enumerate() {
                assert_eq!(*window, [k as u64 * 10, (k as u64 + 1) * 10]);
                let unsupported = (failure == "missing_hop" && window[0] < 128 && window[1] > 64)
                    || (failure == "nonfinite_sample" && window[0] <= 71 && window[1] > 71);
                let expected: [f64; 2] = std::array::from_fn(|bus| {
                    (window[0]..window[1])
                        .map(|tick| {
                            let value = waveform(tick);
                            f64::from(if bus == 0 {
                                value
                            } else {
                                value + phase * value
                            })
                            .powi(2)
                        })
                        .sum::<f64>()
                        / 10.
                });
                assert_eq!(energy[0].target[k], (!unsupported).then_some(expected[0]));
                assert_eq!(energy[1].target[k], Some(expected[1]));
            }
            assert!(energy.iter().all(|e| e.learned));
            let rms = ((0..160)
                .map(|tick| f64::from(waveform(tick)).powi(2))
                .sum::<f64>()
                / 160.)
                .sqrt();
            assert!((out.buses[1].rms.unwrap() - rms).abs() < 1e-15);
            assert!(
                energy[1]
                    .target
                    .iter()
                    .all(|e| *e != Some(out.buses[1].rms.unwrap().powi(2)))
            );
        }
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
fn prediction_inputs_use_only_available_fresh_same_generation_body_records() {
    let mut capture = Capture::spawn(analyzer(), config(), true, None);
    capture.prepare(std::iter::once((1, 0, recipe())), 0);
    let token = capture.token(1, 0).unwrap();
    let parameters = (
        220.,
        0.1,
        crate::life::sound::envelope::Envelope {
            onset: 400,
            hold_end: 900,
            release_end: 1000,
            attack_ticks: 10,
            decay_ticks: 0,
            sustain_level: 1.,
            decay_lambda: 0.,
            release_ticks: 100,
        },
    );
    {
        let mut state = capture.snapshot.lock().unwrap();
        state.records[0] = Record {
            source_id: 1,
            body_generation: token.1,
            end: 100,
            available: 120,
            raw_values: [2.; 6],
            mask: 63,
            active: true,
            ..Record::default()
        };
    }
    let early = capture
        .prediction_input(token, 100, 400, parameters)
        .unwrap();
    assert_eq!(early.descriptors, [[None; 6]; 2]);
    assert_eq!(early.descriptor_support, [None; 2]);
    let ready = capture
        .prediction_input(token, 120, 400, parameters)
        .unwrap();
    assert_eq!(ready.descriptors[0], [Some(2.); 6]);
    assert_eq!(ready.descriptors[1], [None; 6]);
    assert_eq!(ready.descriptor_support[0], Some([0, 100, 120]));
    assert_eq!(ready.descriptor_target_end, 1664);
    let stale = capture
        .prediction_input(token, 4101, 4101, parameters)
        .unwrap();
    assert_eq!(stale.descriptors, [[None; 6]; 2]);
    let mut changed = recipe();
    changed.unison = 2;
    capture.prepare(std::iter::once((1, 0, changed)), 128);
    assert!(
        capture
            .prediction_input(token, 128, 128, parameters)
            .is_none()
    );
    let new = capture
        .prediction_input(capture.token(1, 0).unwrap(), 128, 128, parameters)
        .unwrap();
    assert_eq!(new.descriptors, [[None; 6]; 2]);
    assert_eq!(new.descriptor_target_end, 960);
    capture.finish();
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

#[test]
fn active_candidates_include_later_off_update_and_granted_batches() {
    use crate::core::{modulation::NeuralRhythms, timebase::Timebase};
    use crate::life::{
        action_candidates::{Class, OnsetOpportunity, PolicySnapshot},
        action_observation::Observer,
        phonation_engine::{OnsetKick, ToneCmd, ToneUpdate},
        schedule_renderer::ScheduleRenderer,
        sound::{RenderModulatorSpec, ToneAdsr},
        voice::{PhonationBatch, ToneSpec},
    };
    use crate::scenario::control::Routing;
    for case in [
        "off",
        "direct_off",
        "update",
        "late_update",
        "late_pitch",
        "later_onset",
        "body_generation",
    ] {
        let time = Timebase { fs: 8000., hop: 64 };
        let mut tracked = ScheduleRenderer::new(time);
        let mut capture = Capture::spawn(analyzer(), config(), true, None);
        capture.prepare(std::iter::once((5, 2, recipe())), 0);
        let original_generation = capture.token(5, 2).unwrap().1;
        tracked.body_capture = Some(capture);
        let mut observer = Observer::new(8000);
        observer.enable_predictions();
        tracked.action_observer = Some(Box::new(observer));
        let mut control = ScheduleRenderer::new(time);
        let initial = PhonationBatch {
            source_id: 5,
            source_generation: 2,
            routing: Routing {
                to_habitat: true,
                to_presentation: false,
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
                body: recipe(),
                render_modulator: RenderModulatorSpec::SeqGate { duration_sec: 2. },
                adsr: Some(ToneAdsr {
                    attack_sec: 0.001,
                    decay_sec: 0.,
                    sustain_level: 1.,
                    release_sec: 0.1,
                }),
            }],
            ..Default::default()
        };
        for now in (0..2560).step_by(64) {
            let batches = if now == 0 {
                vec![initial.clone()]
            } else if now == 448 {
                if case == "body_generation" {
                    let mut body = recipe();
                    body.kind = BodyKind::Harmonic;
                    tracked
                        .body_capture
                        .as_mut()
                        .unwrap()
                        .prepare(std::iter::once((5, 2, body)), now);
                    assert_ne!(
                        tracked
                            .body_capture
                            .as_ref()
                            .unwrap()
                            .token(5, 2)
                            .unwrap()
                            .1,
                        original_generation
                    );
                }
                let first = PhonationBatch {
                    source_id: 5,
                    source_generation: 2,
                    intrinsic_period_sec: Some(0.5),
                    body_policy: Some(PolicySnapshot {
                        at: now,
                        is_alive: true,
                        gate_allows_onset: true,
                    }),
                    ..Default::default()
                };
                let mut later = first.clone();
                // Existing Tone routing is independent of a later command's routing.
                later.routing = Routing {
                    to_habitat: false,
                    to_presentation: true,
                };
                later.cmds = vec![if matches!(case, "update" | "late_update" | "late_pitch") {
                    ToneCmd::Update {
                        tone_id: 1,
                        at_tick: Some(if case == "update" { 800 } else { 2500 }),
                        update: ToneUpdate {
                            target_freq_hz: (case == "late_pitch").then_some(440.),
                            target_amp: Some(0.1),
                            continuous_drive: None,
                        },
                    }
                } else if case == "later_onset" {
                    ToneCmd::On {
                        tone_id: 2,
                        kick: OnsetKick { strength: 1. },
                    }
                } else {
                    ToneCmd::Off {
                        tone_id: 1,
                        off_tick: 480,
                    }
                }];
                if case == "later_onset" {
                    let mut spec = initial.tones[0].clone();
                    spec.tone_id = 2;
                    spec.onset = now;
                    spec.opportunity = Some(OnsetOpportunity {
                        issued_at: now,
                        at: now,
                        gate: 1,
                        intrinsic_due_at: Some(now),
                        intrinsic_period_ticks: Some(4000),
                        planned_release_at: None,
                    });
                    later.tones.push(spec);
                }
                if case == "direct_off" {
                    vec![later]
                } else {
                    vec![first, later]
                }
            } else {
                vec![]
            };
            let actual = tracked.render(&batches, now, &NeuralRhythms::default());
            let reference = control.render(&batches, now, &NeuralRhythms::default());
            assert_eq!(actual.habitat, reference.habitat, "{case}");
            assert_eq!(actual.presentation, reference.presentation, "{case}");
        }
        tracked.body_capture.as_mut().unwrap().finish();
        let observer = tracked.action_observer.as_mut().unwrap();
        observer.finish();
        assert_eq!(
            observer.snapshot.commands,
            if matches!(case, "update" | "late_update" | "late_pitch") {
                1
            } else {
                2
            }
        );
        let stats = observer.snapshot.body_defaults.unwrap().candidate_energy;
        let records: Vec<_> = observer.drain_candidate_energy().collect();
        if case == "body_generation" {
            assert_eq!(stats.unsupported, 1);
            assert!(records.is_empty());
            continue;
        }
        assert_eq!(stats.submitted, 1, "{case}");
        assert_eq!(stats.completed, 1, "{case}");
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert_eq!(r.body_generation, Some(original_generation));
        assert_eq!(r.default_routed, [true, case == "later_onset"]);
        let schedule = r.default_schedule.unwrap();
        match case {
            "off" | "direct_off" => {
                assert_eq!(r.tone_id, None);
                assert_eq!(schedule.first_transition.unwrap().class, Class::Release);
                assert_eq!(r.default_input.class, Class::Continue);
                assert!(r.candidates[0].buses[0][0].unwrap().mean.unwrap() > 0.);
                assert_eq!(r.candidates[0].buses[1][0].unwrap().mean, Some(0.));
            }
            "update" | "late_update" => {
                assert_eq!(schedule.unsupported_control_tones, 0);
                assert_eq!(schedule.scheduled_amplitude_tones, 1);
                assert!(
                    r.candidates[0].buses[0]
                        .iter()
                        .all(|w| w.unwrap().mean.is_some())
                );
            }
            "late_pitch" => {
                assert_eq!(schedule.unsupported_control_tones, 1);
                let default = &r.candidates[0];
                let prefix = default.buses[0][0].unwrap();
                assert_eq!(prefix.interval, [448, 2448]);
                assert!(prefix.mean.unwrap() > 0.);
                assert_eq!(prefix.difference, Some(0.));
                assert!(
                    default.buses[0][1..]
                        .iter()
                        .all(|w| w.unwrap().mean.is_none())
                );
                assert!(default.buses[1].iter().all(|w| w.unwrap().mean == Some(0.)));
            }
            "later_onset" => {
                assert_eq!(r.tone_id, Some(2));
                assert_eq!(r.retained_tones, 1);
                assert_eq!(schedule.scheduled_tones, 2);
                assert!(
                    r.candidates[0]
                        .buses
                        .iter()
                        .all(|b| b[0].unwrap().mean.unwrap() > 0.)
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn retained_source_energy_uses_real_tones_routes_and_planned_releases() {
    use crate::core::{modulation::NeuralRhythms, timebase::Timebase};
    use crate::life::{
        action_candidates::{OnsetOpportunity, PolicySnapshot},
        action_observation::{ActionKind, Observer},
        phonation_engine::{OnsetKick, ToneCmd},
        schedule_renderer::ScheduleRenderer,
        sound::{RenderModulatorSpec, ToneAdsr},
        voice::{PhonationBatch, ToneSpec},
    };
    use crate::scenario::control::Routing;
    let time = Timebase { fs: 8000., hop: 64 };
    let mut tracked = ScheduleRenderer::new(time);
    let mut capture = Capture::spawn(analyzer(), config(), true, None);
    capture.prepare(std::iter::once((5, 0, recipe())), 0);
    tracked.body_capture = Some(capture);
    let mut observer = Observer::new(8000);
    observer.enable_predictions();
    tracked.action_observer = Some(Box::new(observer));
    let mut baseline = ScheduleRenderer::new(time);
    let mut outcomes = Vec::new();
    for now in (0..16256).step_by(64) {
        let mut batches = Vec::new();
        for (id, amp, routing, source_id, generation) in if now == 0 {
            vec![
                (1, 0.2, (true, false), 5, 0),
                (2, 0.3, (false, true), 5, 0),
                (4, 0.8, (false, false), 5, 7),
                (1, 0.9, (true, true), 6, 0),
            ]
        } else if now == 128 {
            vec![(3, 0.1, (true, true), 5, 0)]
        } else {
            vec![]
        } {
            batches.push(PhonationBatch {
                source_id,
                source_generation: generation,
                routing: Routing {
                    to_habitat: routing.0,
                    to_presentation: routing.1,
                },
                body_policy: Some(PolicySnapshot {
                    at: now,
                    is_alive: true,
                    gate_allows_onset: true,
                }),
                cmds: vec![ToneCmd::On {
                    tone_id: id,
                    kick: OnsetKick { strength: 1. },
                }],
                tones: vec![ToneSpec {
                    opportunity: (id == 1 && source_id == 5).then_some(OnsetOpportunity {
                        issued_at: now,
                        at: now,
                        gate: 0,
                        intrinsic_due_at: Some(now),
                        intrinsic_period_ticks: Some(4000),
                        planned_release_at: Some(144),
                    }),
                    tone_id: id,
                    onset: now,
                    hold_ticks: Some(20000),
                    freq_hz: 250.,
                    amp,
                    smoothing_tau_sec: 0.,
                    body: recipe(),
                    render_modulator: RenderModulatorSpec::SeqGate { duration_sec: 10. },
                    adsr: Some(ToneAdsr {
                        attack_sec: 0.001,
                        decay_sec: 0.,
                        sustain_level: 1.,
                        release_sec: 0.004,
                    }),
                }],
                ..Default::default()
            });
        }
        if now == 128 {
            batches[0].cmds.push(ToneCmd::Off {
                tone_id: 1,
                off_tick: 144,
            });
        }
        if now == 192 {
            batches.push(PhonationBatch {
                source_id: 5,
                cmds: vec![ToneCmd::Off {
                    tone_id: 3,
                    off_tick: 192,
                }],
                ..Default::default()
            });
        }
        let actual = tracked.render(&batches, now, &NeuralRhythms::default());
        let unchanged = baseline.render(&batches, now, &NeuralRhythms::default());
        assert_eq!(actual.habitat, unchanged.habitat);
        assert_eq!(actual.presentation, unchanged.presentation);
        outcomes.extend(tracked.action_observer.as_mut().unwrap().drain());
    }
    for action in [ActionKind::Onset, ActionKind::Release] {
        let out = outcomes
            .iter()
            .find(|o| o.source_id == 5 && o.tone_id == 3 && o.action == action)
            .unwrap();
        let forecast = out.prediction.unwrap();
        assert!(forecast.sine.is_some());
        assert!(
            forecast
                .source_energy
                .iter()
                .all(|e| e.coherent_energy.iter().all(Option::is_some))
        );
        assert!(
            forecast
                .source_energy
                .iter()
                .any(|e| e.predictions[0] != e.incoherent_fixed_energy)
        );
        for (bus, energy) in forecast.source_energy.iter().enumerate() {
            assert!(energy.retained.complete);
            assert_eq!(
                energy.retained.scanned_entries,
                if action == ActionKind::Onset { 4 } else { 3 }
            );
            assert_eq!(
                energy.retained.included_tones,
                if bus == 0 && action == ActionKind::Release {
                    0
                } else {
                    1
                }
            );
            for (k, [left, right]) in energy.windows.into_iter().enumerate() {
                let tick = left + (right - left) / 2;
                let gain = if bus == 1 || tick < 144 {
                    1.
                } else if tick >= 176 {
                    0.
                } else {
                    (176 - tick) as f32 / 32.
                };
                let retained = f64::from(if bus == 0 { 0.2_f32 } else { 0.3_f32 }).powi(2)
                    * f64::from(gain).powi(2)
                    / 2.;
                assert_eq!(energy.retained.fixed_energy[k], retained);
                assert_eq!(
                    energy.incoherent_fixed_energy[k],
                    Some(energy.command_fixed_energy[k].unwrap() + retained)
                );
                assert!(energy.target[k].is_some());
            }
        }
        if action == ActionKind::Release {
            assert!(
                forecast.source_energy[1]
                    .command_fixed_energy
                    .iter()
                    .all(|e| *e == Some(0.))
            );
            assert!(
                forecast.source_energy[1].predictions[0]
                    .iter()
                    .all(|e| e.unwrap() > 0.04)
            );
        }
    }
    tracked.body_capture.as_mut().unwrap().finish();
}
