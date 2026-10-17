use super::*;
use std::collections::BinaryHeap;

#[test]
fn participation_context_capacity_uses_whole_hop_for_one_sample_segments() {
    use crate::core::temporal_expectation::{AcousticTemporalExpectation, OwnSoundHistory};
    for fs in [48000, 48001] {
        for rate in [190.0, 199.9, MAX_ONSET_RATE_HZ] {
            let observer = AcousticTemporalExpectation::new(fs).unwrap();
            let mut own = OwnSoundHistory::new(&observer, 0);
            let mut engine = PhonationEngine::from_config(
                &PhonationConfig {
                    clock: PhonationClockConfig::Participation {
                        coupling: 0.7,
                        base_rate_hz: rate,
                        flow_depth: 0.0,
                        steady: false,
                    },
                    onset: OnsetConfig::Always { strength: 1.0 },
                    ..Default::default()
                },
                21,
            );
            let PhonationClock::Participation(clock) = &mut engine.clock else {
                unreachable!()
            };
            clock.hop_frames = 512;
            let mut count = 0;
            for hop in 0..400 {
                let now = hop * 512;
                let forecast =
                    TemporalForecast::energy_fixture(fs, own.observed_through_frame(), |_| {
                        [0.1; 3]
                    });
                engine.set_temporal_context(Some(&forecast), [0.0; 3]);
                for (start, end) in [(now, now + 1), (now + 1, now + 512)] {
                    let ctx = CoreTickCtx {
                        now_tick: start,
                        frame_end: end,
                        fs: fs as f32,
                        rhythms: NeuralRhythms::default(),
                    };
                    let mut cmds = Vec::new();
                    let mut events = Vec::new();
                    let mut onsets = Vec::new();
                    engine.tick_with_arrival(
                        &ctx,
                        &CoreState {
                            is_alive: true,
                            onset_allowed: true,
                        },
                        None,
                        0.0,
                        1.0,
                        None,
                        None,
                        &mut cmds,
                        &mut events,
                        &mut onsets,
                    );
                    count += onsets.len();
                }
                own.process(now, &[0.0; 512], &[0.0; 512], &[0.125; 512], |_, _, _| {});
                engine.observe_participation_context(&own, |_| {});
            }
            let minimum_onsets = 400 * 512 / (f64::from(fs) / f64::from(rate)).ceil() as usize;
            assert!(
                count >= minimum_onsets - 1,
                "fs={fs}, rate={rate}, onsets={count}"
            );
            let PhonationClock::Participation(clock) = &engine.clock else {
                unreachable!()
            };
            assert_eq!(clock.hop_frames, 512);
        }
    }
}

#[test]
fn intrinsic_rate_scales_duration_rest_prediction_and_preserves_all_note_ends() {
    let fs = 48_000;
    for rate in [(1.0 / 2.128) as f32, 2.0] {
        let t0 = 1.0 / f64::from(rate);
        for (depth, length) in [(0., 1), (0.65, 3)] {
            let config = PhonationConfig {
                mode: PhonationMode::Gated,
                onset: OnsetConfig::Always { strength: 1. },
                duration: DurationConfig::FixedGate {
                    length_gates: length,
                },
                clock: PhonationClockConfig::Participation {
                    coupling: 0.8,
                    base_rate_hz: rate,
                    flow_depth: depth,
                    steady: false,
                },
                measure_accent: 0.,
            };
            let mut engine = PhonationEngine::from_config(&config, 21);
            assert_eq!(
                engine.clock.fixed_period_ticks(fs as f32),
                Some(t0 * fs as f64)
            );
            let mut cmds = Vec::new();
            let mut events = Vec::new();
            let mut onsets = Vec::new();
            let active_until = 20 * fs as u64;
            for now in (0..(active_until + 8 * fs as u64)).step_by(512) {
                let ctx = CoreTickCtx {
                    now_tick: now,
                    frame_end: now + 512,
                    fs: fs as f32,
                    rhythms: NeuralRhythms::default(),
                };
                engine.tick(
                    &ctx,
                    &CoreState {
                        is_alive: now < active_until,
                        onset_allowed: now < active_until,
                    },
                    None,
                    0.,
                    1.,
                    None,
                    &mut cmds,
                    &mut events,
                    &mut onsets,
                );
            }
            assert!(!events.is_empty());
            let PhonationClock::Participation(clock) = &engine.clock else {
                panic!()
            };
            // Prediction is f32 and Off ticks are integers: one sample bounds both roundings.
            assert!(
                (f64::from(clock.hold_theta / clock.base_rate_hz) - length as f64 * t0).abs()
                    * fs as f64
                    <= 1.
            );
            let offs: Vec<_> = cmds
                .iter()
                .filter_map(|cmd| {
                    if let ToneCmd::Off { tone_id, off_tick } = cmd {
                        Some((*tone_id, *off_tick))
                    } else {
                        None
                    }
                })
                .collect();
            for event in &events {
                let off = offs.iter().find(|(id, _)| *id == event.tone_id).unwrap().1;
                assert!(
                    ((off - event.onset_tick) as f64 - length as f64 * t0 * fs as f64).abs() <= 1.
                );
            }
            assert_eq!(events.len(), offs.len());
            engine.sustain_participation = true;
            engine.enable_rest(&DurationConfig::FixedGate { length_gates: 1 }, 1.);
            engine.rest.as_mut().unwrap().phase = RestPhase::Waiting(active_until);
            engine.rest_tail_finished(active_until + 512, fs as f32);
            let RestPhase::Gap {
                origin,
                until: Some(until),
            } = engine.rest.unwrap().phase
            else {
                panic!()
            };
            assert_eq!(until - origin, (t0 * fs as f64).round() as u64);
            engine.rest_resume(until - 1);
            assert!(matches!(engine.rest.unwrap().phase, RestPhase::Gap { .. }));
            engine.rest_resume(until);
            assert_eq!(engine.rest.unwrap().phase, RestPhase::Ready);
        }
    }
}

#[cfg(feature = "profile-alloc")]
#[test]
fn intrinsic_rate_adds_no_participation_allocations_after_clock_initialization() {
    for steady in [false, true] {
        for rate in [(1.0 / 2.128) as f32, 2.0, 200.0] {
            let config = PhonationConfig {
                mode: PhonationMode::Gated,
                onset: OnsetConfig::Always { strength: 1. },
                duration: DurationConfig::FixedGate { length_gates: 1 },
                clock: PhonationClockConfig::Participation {
                    coupling: 0.05,
                    base_rate_hz: rate,
                    flow_depth: 0.,
                    steady,
                },
                measure_accent: 0.,
            };
            let mut engine = PhonationEngine::from_config(&config, 21);
            let mut cmds = Vec::with_capacity(64);
            let mut events = Vec::with_capacity(64);
            let mut onsets = Vec::with_capacity(64);
            // Existing grid, field and Off storage first allocate at the first onset.
            let warmup_hops = (48000. / f64::from(rate)).ceil() as u64 / 512 + 2;
            let mut warmed = None;
            for hop in 0..warmup_hops {
                cmds.clear();
                events.clear();
                onsets.clear();
                let ctx = CoreTickCtx {
                    now_tick: hop * 512,
                    frame_end: (hop + 1) * 512,
                    fs: 48000.,
                    rhythms: NeuralRhythms::default(),
                };
                engine.tick(
                    &ctx,
                    &CoreState {
                        is_alive: true,
                        onset_allowed: true,
                    },
                    None,
                    0.,
                    1.,
                    None,
                    &mut cmds,
                    &mut events,
                    &mut onsets,
                );
                if !events.is_empty() {
                    warmed = Some(hop + 1);
                    break;
                }
            }
            let warmed = warmed.expect("participation must process its first onset");
            crate::runtime_profile::begin_allocations();
            for hop in warmed..warmed + 1000 {
                cmds.clear();
                events.clear();
                onsets.clear();
                let ctx = CoreTickCtx {
                    now_tick: hop * 512,
                    frame_end: (hop + 1) * 512,
                    fs: 48000.,
                    rhythms: NeuralRhythms::default(),
                };
                engine.tick(
                    &ctx,
                    &CoreState {
                        is_alive: true,
                        onset_allowed: true,
                    },
                    None,
                    0.,
                    1.,
                    None,
                    &mut cmds,
                    &mut events,
                    &mut onsets,
                );
            }
            let counts = crate::runtime_profile::finish_allocations().expect("profile-alloc");
            assert_eq!(counts.count, 0, "rate {rate}: {counts:?}");
            assert_eq!(counts.bytes, 0, "rate {rate}: {counts:?}");
        }
    }
}

#[test]
fn participation_predictions_require_issued_onsets_and_keep_the_original_forecast() {
    let forecast = TemporalForecast::energy_fixture(1000, 1000, |_| [0.4, 0.2, 0.1]);
    for allowed in [false, true] {
        let mut engine = test_engine(
            OnsetRule::Always { strength: 1.0 },
            DurationRule::fixed_gate(1),
        );
        engine.clock = PhonationClock::from_config(
            &PhonationClockConfig::Participation {
                coupling: 0.8,
                base_rate_hz: 20.0,
                flow_depth: 0.0,
                steady: false,
            },
            21,
        );
        engine.set_outcome_forecast(Some(&forecast));
        let ctx = CoreTickCtx {
            now_tick: 1000,
            frame_end: 1200,
            fs: 1000.0,
            rhythms: NeuralRhythms::default(),
        };
        let (mut cmds, mut events, mut onsets) = (Vec::new(), Vec::new(), Vec::new());
        engine.tick(
            &ctx,
            &CoreState {
                is_alive: true,
                onset_allowed: allowed,
            },
            None,
            0.0,
            1.0,
            None,
            &mut cmds,
            &mut events,
            &mut onsets,
        );
        engine.set_outcome_forecast(Some(&TemporalForecast::energy_fixture(1000, 1200, |_| {
            [9.0; 3]
        })));
        let predictions: Vec<_> = engine.drain_participation_predictions().collect();
        assert_eq!(predictions.len(), onsets.len());
        if allowed {
            assert!(predictions.len() >= 3);
        } else {
            assert!(predictions.is_empty());
        }
        for (prediction, onset) in predictions.iter().zip(onsets) {
            assert_eq!(prediction.onset_frame, onset.onset_tick);
            assert_eq!(prediction.issued_frame, 1000);
            assert_eq!(prediction.forecast_observed_frame, 1000);
            assert_eq!(
                prediction.target_start_frame,
                onset.onset_tick.div_ceil(10) * 10
            );
            assert_eq!(
                prediction.target_end_frame,
                prediction.target_start_frame + 10
            );
            assert_eq!(
                prediction.pred_continuation_habitat_band_energy,
                [0.4, 0.2, 0.1]
            );
        }
        assert_eq!(engine.drain_participation_predictions().count(), 0);
    }
}

#[test]
fn acoustic_cadence_changes_only_periodic_pacing_without_stretching_holds() {
    let fs = 48_000;
    let mut observer =
        crate::core::temporal_expectation::AcousticTemporalExpectation::new(fs).unwrap();
    for hop in 0..2000 {
        let now = hop * 480;
        let audio: [f32; 480] = std::array::from_fn(|i| {
            let age = ((now + i) % 28_800) as f32 / fs as f32;
            0.5 * (std::f32::consts::TAU * 180.0 * age).sin() * (-age / 0.012).exp()
        });
        observer.process(now as u64, &audio, |_| {});
    }
    let forecast = observer.forecast().unwrap();
    for (flow_depth, hold_cycles) in [(0.0, 1), (0.8, 3)] {
        let mut engine = test_engine(
            OnsetRule::Always { strength: 1.0 },
            DurationRule::fixed_gate(hold_cycles),
        );
        engine.clock = PhonationClock::from_config(
            &PhonationClockConfig::Participation {
                coupling: 0.8,
                base_rate_hz: 2.0,
                flow_depth,
                steady: false,
            },
            21,
        );
        engine.set_temporal_context(Some(&forecast), [0.0; 3]);
        let ctx = CoreTickCtx {
            now_tick: 20 * fs as u64,
            frame_end: 23 * fs as u64,
            fs: fs as f32,
            rhythms: NeuralRhythms::default(),
        };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        engine.tick(
            &ctx,
            &CoreState {
                is_alive: true,
                onset_allowed: true,
            },
            None,
            0.0,
            1.0,
            None,
            &mut cmds,
            &mut events,
            &mut onsets,
        );
        let PhonationClock::Participation(clock) = &engine.clock else {
            panic!("participation clock expected")
        };
        let period = clock.policy.as_ref().unwrap().period_frames();
        if flow_depth == 0.0 {
            assert!(period > 24_000.0);
        } else {
            assert_eq!(period, 24_000.0);
        }
        let mut ended = 0;
        for cmd in cmds {
            if let ToneCmd::Off { tone_id, off_tick } = cmd {
                let event = events.iter().find(|e| e.tone_id == tone_id).unwrap();
                assert_eq!(off_tick - event.onset_tick, hold_cycles as u64 * 24_000);
                ended += 1;
            }
        }
        assert!(ended > 0, "each clock must emit and finish a sound");
    }
}

#[test]
fn participation_releases_acoustic_cadence_through_silence_and_reacquires() {
    let fs = 48_000;
    let hop_size = 480;
    let mut observer =
        crate::core::temporal_expectation::AcousticTemporalExpectation::new(fs).unwrap();
    let mut own = crate::core::temporal_expectation::OwnSoundHistory::new(&observer, 0);
    let mut engine = test_engine(
        OnsetRule::Always { strength: 1.0 },
        DurationRule::fixed_gate(1),
    );
    engine.clock = PhonationClock::from_config(
        &PhonationClockConfig::Participation {
            coupling: 0.8,
            base_rate_hz: 2.0,
            flow_depth: 0.0,
            steady: false,
        },
        21,
    );
    let mut cmds = Vec::new();
    let mut events = Vec::new();
    let mut onsets = Vec::new();
    for hop in 0..10_100 {
        let now = hop * hop_size;
        let allowed = hop < 3000 || (7000..10_000).contains(&hop);
        let forecast = observer.forecast();
        engine.set_temporal_context(forecast.as_ref(), [0.0; 3]);
        engine.tick(
            &CoreTickCtx {
                now_tick: now,
                frame_end: now + hop_size,
                fs: fs as f32,
                rhythms: NeuralRhythms::default(),
            },
            &CoreState {
                is_alive: true,
                onset_allowed: allowed,
            },
            None,
            0.0,
            1.0,
            Some(1000),
            &mut cmds,
            &mut events,
            &mut onsets,
        );
        // The reference is controlled sound; actions never supply future evidence.
        let audio: [f32; 480] = std::array::from_fn(|i| {
            if !allowed {
                return 0.0;
            }
            let (origin, period) = if hop < 3000 {
                (0, 28_800)
            } else {
                (70 * fs as u64, 19_200)
            };
            let age = ((now + i as u64 - origin) % period) as f32 / fs as f32;
            0.5 * (std::f32::consts::TAU * 180.0 * age).sin() * (-age / 0.012).exp()
        });
        own.process(now, &[0.0; 480], &[0.0; 480], &audio, |_, _, _| {});
        engine.observe_participation_context(&own, |_| {});
        observer.process(now, &audio, |_| {});
        if let Some(expected) = match hop {
            2999 => Some(0.6),
            6999 => Some(0.5),
            9999 => Some(0.4),
            _ => None,
        } {
            let PhonationClock::Participation(clock) = &engine.clock else {
                unreachable!();
            };
            let period = clock.policy.as_ref().unwrap().period_frames() / fs as f64;
            assert!((period - expected).abs() < 0.01, "hop {hop}: {period}");
        }
    }
    assert!(onsets.len() > 100);
    assert!(onsets.iter().all(|e| {
        e.onset_tick < 30 * fs as u64 || (70 * fs as u64..100 * fs as u64).contains(&e.onset_tick)
    }));
    assert!(onsets.windows(2).all(|w| w[0].onset_tick < w[1].onset_tick));
    let resumed = onsets
        .iter()
        .find(|e| e.onset_tick >= 70 * fs as u64)
        .unwrap();
    assert!(resumed.onset_tick < 71 * fs as u64);
    let mut ended = 0;
    for cmd in cmds {
        if let ToneCmd::Off { tone_id, off_tick } = cmd {
            let onset = events.iter().find(|e| e.tone_id == tone_id).unwrap();
            assert_eq!(off_tick - onset.onset_tick, (fs / 2) as u64);
            ended += 1;
        }
    }
    assert_eq!(ended, onsets.len());
}

#[test]
fn participation_execution_is_independent_of_hop_size_and_shared_phase() {
    for (rate, depth) in [(40.0, 0.0), (40.0, 0.8), (200.0, 0.0)] {
        let run = |hop: u64, shared_phase: f32| {
            let mut engine = test_engine(
                OnsetRule::Always { strength: 1.0 },
                DurationRule::fixed_gate(1),
            );
            engine.clock = PhonationClock::from_config(
                &PhonationClockConfig::Participation {
                    coupling: 0.7,
                    base_rate_hz: rate,
                    flow_depth: depth,
                    steady: false,
                },
                21,
            );
            let mut cmds = Vec::new();
            let mut events = Vec::new();
            let mut onsets = Vec::new();
            for now in (0..4000).step_by(hop as usize) {
                let mut rhythms = NeuralRhythms::default();
                rhythms.theta.phase = shared_phase + now as f32 * 0.07;
                rhythms.theta.freq_hz = 7.3;
                rhythms.delta.phase = shared_phase - now as f32 * 0.03;
                rhythms.env_open = if shared_phase == 0.0 { 0.0 } else { 1.0 };
                let ctx = CoreTickCtx {
                    now_tick: now,
                    frame_end: (now + hop).min(4000),
                    fs: 1000.0,
                    rhythms,
                };
                let before = onsets.len();
                engine.tick(
                    &ctx,
                    &CoreState {
                        is_alive: true,
                        onset_allowed: true,
                    },
                    None,
                    0.0,
                    1.0,
                    Some(1000),
                    &mut cmds,
                    &mut events,
                    &mut onsets,
                );
                assert!(
                    onsets[before..]
                        .iter()
                        .all(|event| (now..ctx.frame_end).contains(&event.onset_tick))
                );
            }
            assert!(onsets.len() > 50);
            assert!(onsets.iter().all(|event| event.onset_tick >= 1000));
            assert!(
                onsets
                    .windows(2)
                    .all(|pair| pair[1].onset_tick - pair[0].onset_tick >= 5)
            );
            for cmd in &cmds {
                if let ToneCmd::Off { tone_id, off_tick } = cmd {
                    let event = events
                        .iter()
                        .find(|event| event.tone_id == *tone_id)
                        .unwrap();
                    assert_eq!(*off_tick - event.onset_tick, (1000.0 / rate) as u64);
                }
            }
            (
                cmds,
                onsets
                    .iter()
                    .map(|event| event.onset_tick)
                    .collect::<Vec<_>>(),
            )
        };
        let reference = run(1, 0.0);
        assert_eq!(reference, run(17, 1.7));
        assert_eq!(reference, run(1000, 4.2));
    }
}

fn candidate_at_gate(gate: u64) -> CandidatePoint {
    CandidatePoint { tick: gate, gate }
}

#[test]
fn a_footprint_is_representative_only_while_the_comparison_configures_it() {
    use crate::config::{FootprintSource, TemporalOnsetComparisonConfig};
    let mut engine = test_engine(
        OnsetRule::Always { strength: 1.0 },
        DurationRule::fixed_gate(1),
    );
    // A theta gate compares no onset times and has no recipe to freeze.
    assert!(engine.footprint_hold().is_none());
    engine.clock = PhonationClock::from_config(
        &PhonationClockConfig::Participation {
            coupling: 0.8,
            base_rate_hz: 2.0,
            flow_depth: 0.0,
            steady: false,
        },
        21,
    );
    assert!(
        engine.footprint_hold().is_none(),
        "unconfigured asks nothing"
    );
    let mut config = TemporalOnsetComparisonConfig {
        footprint: FootprintSource::Proxy,
        ..Default::default()
    };
    engine.set_onset_comparison(Some(config));
    let (hold_sec, adsr) = engine.footprint_hold().expect("a proxy recipe");
    assert_eq!(hold_sec, 0.5);
    assert!(adsr.is_none());
    config.footprint = FootprintSource::Body;
    engine.set_onset_comparison(Some(config));
    let (hold_sec, adsr) = engine.footprint_hold().expect("a body recipe");
    assert_eq!(hold_sec, 0.5);
    assert!(adsr.is_none());
    engine.set_sound_adsr(Some(crate::life::sound::ToneAdsr {
        attack_sec: 0.01,
        decay_sec: 0.0,
        sustain_level: 1.0,
        release_sec: 0.1,
    }));
    assert!(engine.footprint_hold().unwrap().1.is_some());
}

fn test_engine(onset_rule: OnsetRule, duration_rule: DurationRule) -> PhonationEngine {
    PhonationEngine {
        rest: None,
        #[cfg(test)]
        synchronized_counts: [0; 4],
        sustain_participation: false,
        clock: PhonationClock::ThetaGate(ThetaGateClock::default()),
        onset_rule,
        duration_rule,
        mode: PhonationMode::Gated,
        measure_accent: 0.0,
        hold: HoldCore::default(),
        next_tone_id: 0,
        last_tick: None,
        active_notes: 0,
        pending_off: BinaryHeap::new(),
        scratch_candidates: Vec::new(),
        scratch_merged: Vec::new(),
        scratch_grid: ThetaGrid::default(),
        scratch_field: TimingField::default(),
    }
}

#[test]
fn measure_accent_changes_strength_at_each_onset_without_rescheduling() {
    for (amount, confidence, ratio, expected) in [
        (1.0, 1.0, 2, [1.35, 1.0, 0.65, 1.0]),
        (1.0, 0.5, 2, [1.175, 1.0, 0.825, 1.0]),
        (0.0, 1.0, 2, [1.0; 4]),
        (1.0, 0.0, 2, [1.0; 4]),
        (1.0, 1.0, 0, [1.0; 4]),
    ] {
        let mut engine = test_engine(
            OnsetRule::Always { strength: 1.0 },
            DurationRule::Custom(Box::new(|_| DurationPlan::None)),
        );
        engine.measure_accent = amount;
        engine.clock = PhonationClock::Custom(Box::new(|ctx, out| {
            for gate in 0..4 {
                out.push(CandidatePoint {
                    tick: ctx.now_tick + gate * 250,
                    gate,
                });
            }
        }));
        let mut rhythms = NeuralRhythms::default();
        rhythms.env_open = 1.0;
        rhythms.measure_ratio = ratio;
        rhythms.measure.freq_hz = 1.0;
        rhythms.measure.alpha = confidence;
        let ctx = CoreTickCtx {
            now_tick: 1000,
            frame_end: 2000,
            fs: 1000.0,
            rhythms,
        };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        engine.tick(
            &ctx,
            &CoreState {
                is_alive: true,
                onset_allowed: true,
            },
            None,
            0.0,
            1.0,
            None,
            &mut cmds,
            &mut events,
            &mut onsets,
        );
        assert_eq!(
            events.iter().map(|e| e.onset_tick).collect::<Vec<_>>(),
            [1000, 1250, 1500, 1750]
        );
        assert_eq!(cmds.len(), 4);
        for ((command, onset), strength) in cmds.iter().zip(&onsets).zip(expected) {
            let ToneCmd::On { kick, .. } = command else {
                panic!("expected onset")
            };
            assert_eq!(kick.strength, onset.strength);
            assert!((kick.strength - strength).abs() < 1e-6);
        }
    }
}

#[test]
fn measure_feedback_does_not_invent_grouping_for_uniform_onsets() {
    use crate::core::meter::MeterNetwork;
    use crate::scenario::{CoupledTimingSpec, DurationSpec, PhonationTiming};
    let mut engine = PhonationEngine::from_spec(
        &PhonationSpec {
            sustain_participation: false,
            timing: PhonationTiming::Coupled(CoupledTimingSpec {
                coupling: 0.0,
                base_rate_hz: 2.0,
                measure_accent: 1.0,
                ..Default::default()
            }),
            duration: DurationSpec::Gates(1),
        },
        1,
    );
    let mut meter = MeterNetwork::new();
    let mut rhythms = NeuralRhythms::default();
    let mut cmds = Vec::new();
    let mut events = Vec::new();
    let mut onsets = Vec::new();
    let mut late_onsets = 0;
    for hop in 0..12000 {
        let ctx = CoreTickCtx {
            now_tick: hop * 5,
            frame_end: (hop + 1) * 5,
            fs: 1000.0,
            rhythms,
        };
        cmds.clear();
        events.clear();
        onsets.clear();
        engine.tick(
            &ctx,
            &CoreState {
                is_alive: true,
                onset_allowed: true,
            },
            None,
            0.0,
            1.0,
            None,
            &mut cmds,
            &mut events,
            &mut onsets,
        );
        let drive = onsets.iter().map(|onset| onset.strength).sum::<f32>() * 0.25;
        let state = meter.process(0.005, drive);
        rhythms = NeuralRhythms::from_meter_state(&state);
        if hop >= 8000 {
            assert_eq!(
                state.measure_ratio, 0,
                "uniform feedback created a measure at hop {hop}"
            );
            for onset in &onsets {
                assert_eq!(onset.strength, 1.0);
                late_onsets += 1;
            }
        }
    }
    assert!(late_onsets >= 39);
}

#[test]
fn accumulator_rate_zero_never_fires() {
    let mut interval = OnsetRule::Accumulator {
        rate_hz: 0.0,
        refractory_gates: 0,
        acc: 0.0,
        next_allowed_gate: 0,
    };
    let state = CoreState {
        is_alive: true,
        onset_allowed: true,
    };
    for gate in 0..10u64 {
        let c = candidate_at_gate(gate);
        let input = IntervalInput {
            gate: c.gate,
            tick: c.tick,
            dt_sec: 1.0,
            weight: 1.0,
        };
        assert!(interval.on_candidate(&input, &state).is_none());
    }
}

#[test]
fn accumulator_rate_half_fires_every_other_gate() {
    let mut interval = OnsetRule::Accumulator {
        rate_hz: 0.5,
        refractory_gates: 0,
        acc: 0.0,
        next_allowed_gate: 0,
    };
    let state = CoreState {
        is_alive: true,
        onset_allowed: true,
    };
    let mut fired = Vec::new();
    for gate in 0..10u64 {
        let c = candidate_at_gate(gate);
        let input = IntervalInput {
            gate: c.gate,
            tick: c.tick,
            dt_sec: 1.0,
            weight: 1.0,
        };
        if interval.on_candidate(&input, &state).is_some() {
            fired.push(gate);
        }
    }
    assert_eq!(fired, vec![1u64, 3, 5, 7, 9]);
}

#[test]
fn accumulator_refractory_blocks_adjacent_gates() {
    let mut interval = OnsetRule::Accumulator {
        rate_hz: 1.0,
        refractory_gates: 1,
        acc: 0.0,
        next_allowed_gate: 0,
    };
    let state = CoreState {
        is_alive: true,
        onset_allowed: true,
    };
    let mut fired = Vec::new();
    for gate in 0..6u64 {
        let c = candidate_at_gate(gate);
        let input = IntervalInput {
            gate: c.gate,
            tick: c.tick,
            dt_sec: 1.0,
            weight: 1.0,
        };
        if interval.on_candidate(&input, &state).is_some() {
            fired.push(gate);
        }
    }
    assert_eq!(fired, vec![0u64, 2, 4]);
}

#[test]
fn accumulator_weighted_steps_fire_after_sum() {
    let mut interval = OnsetRule::Accumulator {
        rate_hz: 1.0,
        refractory_gates: 0,
        acc: 0.0,
        next_allowed_gate: 0,
    };
    let state = CoreState {
        is_alive: true,
        onset_allowed: true,
    };
    let mut fired = Vec::new();
    let inputs = [
        IntervalInput {
            gate: 0,
            tick: 0,
            dt_sec: 1.0,
            weight: 0.5,
        },
        IntervalInput {
            gate: 1,
            tick: 1,
            dt_sec: 1.0,
            weight: 0.5,
        },
    ];
    for input in inputs {
        if interval.on_candidate(&input, &state).is_some() {
            fired.push(input.gate);
        }
    }
    assert_eq!(fired, vec![1u64]);
}

#[test]
fn accumulator_ignores_zero_weight() {
    let mut interval = OnsetRule::Accumulator {
        rate_hz: 1.0,
        refractory_gates: 0,
        acc: 0.0,
        next_allowed_gate: 0,
    };
    let state = CoreState {
        is_alive: true,
        onset_allowed: true,
    };
    let input = IntervalInput {
        gate: 0,
        tick: 0,
        dt_sec: 1.0,
        weight: 0.0,
    };
    assert!(interval.on_candidate(&input, &state).is_none());
}

#[test]
fn accumulator_respects_dt_sec() {
    let mut interval = OnsetRule::Accumulator {
        rate_hz: 1.0,
        refractory_gates: 0,
        acc: 0.0,
        next_allowed_gate: 0,
    };
    let state = CoreState {
        is_alive: true,
        onset_allowed: true,
    };
    let input = IntervalInput {
        gate: 0,
        tick: 0,
        dt_sec: 0.5,
        weight: 1.0,
    };
    assert!(interval.on_candidate(&input, &state).is_none());
    let input = IntervalInput {
        gate: 1,
        tick: 1,
        dt_sec: 0.5,
        weight: 1.0,
    };
    assert!(interval.on_candidate(&input, &state).is_some());
}

#[test]
fn accumulator_acc_is_clamped() {
    let mut interval = OnsetRule::Accumulator {
        rate_hz: 1.0,
        refractory_gates: 0,
        acc: 0.0,
        next_allowed_gate: 0,
    };
    let state = CoreState {
        is_alive: true,
        onset_allowed: true,
    };
    let input = IntervalInput {
        gate: 0,
        tick: 0,
        dt_sec: 1.0,
        weight: 1.0e9,
    };
    let _ = interval.on_candidate(&input, &state);
    let OnsetRule::Accumulator { acc, .. } = &interval else {
        panic!("expected Accumulator");
    };
    assert!(acc.is_finite());
    assert!(*acc <= 32.0);
}

#[test]
fn fixed_gate_connect_returns_hold_theta() {
    let mut connect = DurationRule::fixed_gate(0);
    let plan = connect.on_note_on(OnsetContext {
        tone_id: 1,
        tick: 123,
        gate: 10,
        exc_gate: 1.0,
        exc_slope: 0.0,
    });
    assert_eq!(plan, DurationPlan::HoldTheta(0.0));
}

#[test]
fn frozen_clock_opportunities_match_first_crossings_without_consuming_rng() {
    let configs = [
        PhonationClockConfig::ThetaGate,
        PhonationClockConfig::Coupling {
            coupling: 0.0,
            base_rate_hz: 30.0,
            flow_depth: 0.8,
            microtiming: 0.0,
        },
        PhonationClockConfig::Coupling {
            coupling: 0.7,
            base_rate_hz: 30.0,
            flow_depth: 0.8,
            microtiming: 0.03,
        },
    ];
    for config in configs {
        let mut clock = PhonationClock::from_config(&config, 19);
        let mut control = PhonationClock::from_config(&config, 19);
        let mut checked = 0;
        for now in (0..20_000).step_by(64) {
            let mut rhythms = NeuralRhythms::default();
            rhythms.theta.freq_hz = 30.0;
            rhythms.theta.phase = (now as f32 * 30.0 / 1000.0 * TAU).rem_euclid(TAU);
            rhythms.delta.phase = (now as f32 * 2.3 / 1000.0 * TAU).rem_euclid(TAU);
            rhythms.delta.alpha = 0.9;
            let ctx = CoreTickCtx {
                now_tick: now,
                frame_end: now + 64,
                fs: 1000.0,
                rhythms,
            };
            let frozen = clock.opportunity(&ctx).unwrap();
            for _ in 0..4 {
                assert_eq!(clock.opportunity(&ctx), Some(frozen));
            }
            let mut actual = Vec::new();
            let mut expected = Vec::new();
            clock.gather_candidates(&ctx, &mut actual);
            control.gather_candidates(&ctx, &mut expected);
            assert_eq!(
                actual.iter().map(|c| (c.tick, c.gate)).collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|c| (c.tick, c.gate))
                    .collect::<Vec<_>>()
            );
            if let Some(first) = actual.first() {
                assert_eq!(frozen.at, first.tick);
                checked += 1;
            } else {
                assert!(frozen.at >= ctx.frame_end);
            }
        }
        assert!(checked > 10);
    }
}

#[test]
fn theta_clock_preserves_crossings_on_frame_boundaries() {
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.freq_hz = 16.0;
    rhythms.theta.phase = 0.0;
    let mut clock = ThetaGateClock::default();
    for now_tick in [0, 128, 256] {
        let ctx = CoreTickCtx {
            now_tick,
            frame_end: now_tick + 128,
            fs: 1024.0,
            rhythms,
        };
        let mut candidates = Vec::new();
        clock.gather_candidates_impl(&ctx, &mut candidates);
        assert_eq!(
            candidates.iter().map(|c| c.tick).collect::<Vec<_>>(),
            vec![now_tick, now_tick + 64]
        );
        candidates.clear();
        clock.gather_candidates_impl(&ctx, &mut candidates);
        assert!(
            candidates.is_empty(),
            "repeated frames must not emit old crossings"
        );
    }
}

#[test]
fn theta_clock_near_a_crossing_does_not_repeat_the_phase_remainder() {
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.freq_hz = 8.0;
    rhythms.theta.phase = TAU * (1.0 - 10.0 / 6000.0);
    let ctx = CoreTickCtx {
        now_tick: 912_384,
        frame_end: 912_896,
        fs: 48_000.0,
        rhythms,
    };
    let mut clock = ThetaGateClock::default();
    let mut candidates = Vec::new();
    clock.gather_candidates_impl(&ctx, &mut candidates);
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].tick, ctx.now_tick + 10);
    let mut grid = ThetaGrid::from_candidates(&candidates);
    grid.ensure_boundaries_until(&ctx, 2, None);
    assert_eq!(grid.boundaries[1].tick - grid.boundaries[0].tick, 6000);
    assert_eq!(grid.boundaries[2].tick - grid.boundaries[1].tick, 6000);
}

#[test]
fn theta_clock_multiple_crossings_are_one_cycle_apart() {
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.freq_hz = 100.0;
    rhythms.theta.phase = TAU * 0.8;
    let ctx = CoreTickCtx {
        now_tick: 0,
        frame_end: 32,
        fs: 1000.0,
        rhythms,
    };
    let mut clock = ThetaGateClock::default();
    let mut candidates = Vec::new();
    clock.gather_candidates_impl(&ctx, &mut candidates);
    assert_eq!(
        candidates.iter().map(|c| c.tick).collect::<Vec<_>>(),
        vec![2, 12, 22]
    );
}

#[test]
fn theta_gate_clock_emits_multiple_gates_in_hop() {
    let mut clock = ThetaGateClock::default();
    let ctx = CoreTickCtx {
        now_tick: 0,
        frame_end: 10,
        fs: 1000.0,
        rhythms: NeuralRhythms::default(),
    };
    let mut out = Vec::new();
    let mut calls = 0;
    clock.gather_candidates_core(&ctx, &mut out, |_cursor, _ctx| {
        let tick = match calls {
            0 => Some(0),
            1 => Some(4),
            2 => Some(8),
            _ => None,
        };
        calls += 1;
        tick
    });
    assert!(out.len() >= 2, "expected multiple gates in hop");
    assert!(out[0].tick < out[1].tick);
}

#[test]
fn theta_gate_clock_emits_candidate_at_frame_start_gate() {
    let mut clock = ThetaGateClock::default();
    let rhythms = NeuralRhythms::default();
    let ctx = CoreTickCtx {
        now_tick: 0,
        frame_end: 4,
        fs: 1000.0,
        rhythms,
    };
    let mut out = Vec::new();
    clock.gather_candidates_core(&ctx, &mut out, |cursor, _ctx| {
        if cursor == 0 { Some(cursor) } else { None }
    });
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].tick, 0);
}

#[test]
fn basin_rate_preserves_independent_initial_metric_phases() {
    let mut phases = Vec::new();
    for seed in 1..=8 {
        let default = CouplingClock::new(0.95, 2.0, 0.0, 0.0, seed ^ 21);
        let basin = CouplingClock::new(0.95, 0.5 * (2.8 + 3.4), 0.0, 0.0, seed ^ 21);
        assert_eq!(default.phase, basin.phase);
        assert!((0.0..1.0).contains(&basin.phase));
        assert!(!phases.contains(&basin.phase));
        phases.push(basin.phase);
    }
}

#[test]
fn coupling_clock_free_runs_at_base_rate_with_zero_confidence() {
    // No metrical confidence -> lock collapses to 0 and the clock free-runs
    // at its intrinsic base rate (renewal/flow limit). depth = 0 keeps the
    // period regular so the assay is deterministic.
    let mut clock = CouplingClock::new(0.95, 10.0, 0.0, 0.0, 1);
    let rhythms = NeuralRhythms::default(); // delta.alpha (confidence) = 0
    let ctx = CoreTickCtx {
        now_tick: 0,
        frame_end: 350,
        fs: 1000.0,
        rhythms,
    };
    let mut out = Vec::new();
    clock.gather_candidates_impl(&ctx, &mut out);
    // base_rate 10 Hz @ fs 1000 -> 100-tick period; a few onsets per hop,
    // never one-per-tick.
    assert!(out.len() >= 2 && out.len() <= 5, "got {}", out.len());
    for pair in out.windows(2) {
        assert!(pair[1].tick - pair[0].tick >= 80);
    }
}

#[test]
fn coupling_clock_clamps_extreme_base_rate() {
    // A pathological base rate must not collapse the period to ~1 tick.
    let mut clock = CouplingClock::new(0.0, 1.0e9, 0.0, 0.0, 1);
    let rhythms = NeuralRhythms::default();
    let ctx = CoreTickCtx {
        now_tick: 0,
        frame_end: 1000,
        fs: 48_000.0,
        rhythms,
    };
    let mut out = Vec::new();
    clock.gather_candidates_impl(&ctx, &mut out);
    // At the MAX_ONSET_RATE_HZ cap, fs/200 = 240-tick period -> only a few
    // onsets in a 1000-tick hop, never one-per-tick.
    assert!(
        out.len() <= 1000 / 200 + 2,
        "clamped clock emitted too many onsets: {}",
        out.len()
    );
    for pair in out.windows(2) {
        assert!(pair[1].tick - pair[0].tick >= 200);
    }
}

#[test]
fn coupling_clock_high_coupling_locks_to_beat_band() {
    // With strong confidence and high coupling, the effective rate is pulled
    // toward the shared beat (delta) band, not the intrinsic base rate.
    let mut rhythms = NeuralRhythms::default();
    rhythms.delta.alpha = 1.0; // full confidence
    rhythms.delta.freq_hz = 2.0; // shared beat at 2 Hz
    let mut clock = CouplingClock::new(1.0, 10.0, 0.0, 0.0, 7);
    let ctx = CoreTickCtx {
        now_tick: 0,
        frame_end: 4000,
        fs: 1000.0,
        rhythms,
    };
    let mut out = Vec::new();
    clock.gather_candidates_impl(&ctx, &mut out);
    // Locked to 2 Hz (500-tick period), the free-run base of 10 Hz would have
    // produced ~40 onsets; locking yields far fewer.
    assert!(
        out.len() <= 12,
        "expected beat-locked onsets near 2 Hz, got {}",
        out.len()
    );
    for pair in out.windows(2) {
        assert!(pair[1].tick - pair[0].tick >= 300);
    }
}

#[test]
fn phonation_engine_uses_timing_field_weight() {
    let ctx = CoreTickCtx {
        now_tick: 0,
        frame_end: 8,
        fs: 1000.0,
        rhythms: NeuralRhythms::default(),
    };
    let candidates = vec![
        CandidatePoint { tick: 0, gate: 0 },
        CandidatePoint { tick: 4, gate: 1 },
    ];
    let timing_field = TimingField::from_values(0, vec![0.0, 1.0]);
    let interval = OnsetRule::Accumulator {
        rate_hz: 250.0,
        refractory_gates: 0,
        acc: 0.0,
        next_allowed_gate: 0,
    };
    let mut engine = test_engine(interval, DurationRule::fixed_gate(2));
    let state = CoreState {
        is_alive: true,
        onset_allowed: true,
    };
    let mut cmds = Vec::new();
    let mut events = Vec::new();
    let mut onsets = Vec::new();
    let mut timing_grid = ThetaGrid {
        boundaries: vec![
            GateBoundary { gate: 0, tick: 0 },
            GateBoundary { gate: 1, tick: 10 },
            GateBoundary { gate: 2, tick: 20 },
            GateBoundary { gate: 3, tick: 30 },
        ],
    };
    engine.process_candidates(
        &ctx,
        &candidates,
        &mut timing_grid,
        &timing_field,
        &state,
        None,
        &mut cmds,
        &mut events,
        &mut onsets,
    );
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].onset_tick, 4);
}

#[test]
fn timing_field_indexing_from_values() {
    let timing_field = TimingField::from_values(10, vec![0.2, 0.8]);
    assert_eq!(timing_field.e(10), 0.2);
    assert_eq!(timing_field.e(11), 0.8);
    assert_eq!(timing_field.e(12), 1.0);
}

#[test]
fn timing_field_build_from_uses_env_gate() {
    let mut rhythms = NeuralRhythms::default();
    rhythms.env_open = 0.0;
    rhythms.env_level = 1.0;
    let ctx = CoreTickCtx {
        now_tick: 0,
        frame_end: 4,
        fs: 1000.0,
        rhythms,
    };
    let grid = ThetaGrid {
        boundaries: vec![GateBoundary { gate: 0, tick: 0 }],
    };
    let timing_field = TimingField::build_from(&ctx, &grid, None, 1.0);
    assert_eq!(timing_field.e(0), 0.0);
}

#[test]
fn theta_grid_skips_duplicate_gates() {
    let candidates = vec![
        CandidatePoint { tick: 0, gate: 0 },
        CandidatePoint { tick: 1, gate: 0 },
        CandidatePoint { tick: 2, gate: 1 },
    ];
    let grid = ThetaGrid::from_candidates(&candidates);
    assert_eq!(
        grid.boundaries,
        vec![
            GateBoundary { gate: 0, tick: 0 },
            GateBoundary { gate: 1, tick: 2 },
        ]
    );
}

#[test]
#[cfg(not(debug_assertions))]
fn theta_grid_repairs_reverse_gates_without_panic() {
    // Release behaviour: a decreasing gate must not stop the performance. It
    // is dropped, leaving a grid whose gates still increase by one. Debug
    // builds trip `debug_assert!` first, hence the cfg gate.
    let candidates = vec![
        CandidatePoint { tick: 0, gate: 1 },
        CandidatePoint { tick: 1, gate: 0 },
        CandidatePoint { tick: 2, gate: 2 },
    ];
    let grid = ThetaGrid::from_candidates(&candidates);
    assert_eq!(
        grid.boundaries,
        vec![
            GateBoundary { gate: 1, tick: 0 },
            GateBoundary { gate: 2, tick: 2 },
        ]
    );
}

#[test]
fn theta_grid_rebuild_clears_previous_boundaries() {
    let mut grid = ThetaGrid::default();
    grid.rebuild_from_candidates(&[
        CandidatePoint { tick: 0, gate: 0 },
        CandidatePoint { tick: 5, gate: 1 },
    ]);
    assert_eq!(grid.boundaries.len(), 2);
    grid.rebuild_from_candidates(&[CandidatePoint { tick: 9, gate: 7 }]);
    assert_eq!(
        grid.boundaries,
        vec![GateBoundary { gate: 7, tick: 9 }],
        "rebuild must not keep stale boundaries"
    );
}

#[test]
#[cfg(debug_assertions)]
#[should_panic]
fn theta_grid_panics_on_unsorted_ticks() {
    let candidates = vec![
        CandidatePoint { tick: 2, gate: 0 },
        CandidatePoint { tick: 1, gate: 1 },
    ];
    let _ = ThetaGrid::from_candidates(&candidates);
}

#[test]
fn theta_grid_skips_noncontiguous_gates() {
    let candidates = vec![
        CandidatePoint { tick: 0, gate: 0 },
        CandidatePoint { tick: 10, gate: 2 },
    ];
    let grid = ThetaGrid::from_candidates(&candidates);
    assert_eq!(
        grid.boundaries.len(),
        1,
        "non-contiguous gate 2 should be skipped"
    );
}

#[test]
fn timing_field_fill_missing_gates() {
    let mut rhythms = NeuralRhythms::default();
    rhythms.env_open = 0.5;
    rhythms.env_level = 1.0;
    let ctx = CoreTickCtx {
        now_tick: 0,
        frame_end: 1,
        fs: 1000.0,
        rhythms,
    };
    let grid = ThetaGrid {
        boundaries: vec![
            GateBoundary { gate: 0, tick: 0 },
            GateBoundary { gate: 2, tick: 0 },
        ],
    };
    let timing_field = TimingField::build_from(&ctx, &grid, None, 1.0);
    assert_eq!(timing_field.e(0), 0.5);
    assert_eq!(timing_field.e(1), 0.5);
    assert_eq!(timing_field.e(2), 0.5);
}

#[test]
#[cfg(not(debug_assertions))]
fn timing_field_truncates_oversized_gate_gap() {
    // Release behaviour: a pathological gate gap truncates the fill instead
    // of aborting the hop, and the gates past the truncation read the 1.0
    // fallback. Debug builds trip `debug_assert!` first, hence the cfg gate.
    let mut rhythms = NeuralRhythms::default();
    rhythms.env_open = 0.5;
    rhythms.env_level = 1.0;
    let ctx = CoreTickCtx {
        now_tick: 0,
        frame_end: 16,
        fs: 1000.0,
        rhythms,
    };
    let grid = ThetaGrid {
        boundaries: vec![
            GateBoundary { gate: 0, tick: 0 },
            GateBoundary {
                gate: 100_000,
                tick: 10,
            },
        ],
    };
    let timing_field = TimingField::build_from(&ctx, &grid, None, 1.0);
    assert_eq!(
        timing_field.e(0),
        0.5,
        "gates before the gap keep their weight"
    );
    assert_eq!(
        timing_field.e(1),
        1.0,
        "gates inside the gap fall back to 1.0"
    );
    assert_eq!(
        timing_field.e(100_000),
        1.0,
        "the far boundary falls back to 1.0 instead of panicking"
    );
}

#[test]
fn timing_field_rebuild_flat_reuses_buffer_and_clears() {
    let mut field = TimingField::default();
    let grid = ThetaGrid {
        boundaries: vec![
            GateBoundary { gate: 4, tick: 0 },
            GateBoundary { gate: 5, tick: 10 },
        ],
    };
    field.rebuild_flat(&grid);
    assert_eq!(field.start_gate, 4);
    assert_eq!(field.e_gate, vec![1.0, 1.0]);
    field.rebuild_flat(&ThetaGrid::default());
    assert_eq!(field.start_gate, 0);
    assert!(field.e_gate.is_empty());
}

#[test]
fn theta_grid_tick_at_interpolates() {
    let grid = ThetaGrid {
        boundaries: vec![
            GateBoundary { gate: 0, tick: 0 },
            GateBoundary { gate: 1, tick: 100 },
        ],
    };
    assert_eq!(grid.tick_at(0, 0.5), Some(50));
}

#[test]
fn theta_grid_tick_at_avoids_boundaries() {
    let grid = ThetaGrid {
        boundaries: vec![
            GateBoundary { gate: 0, tick: 0 },
            GateBoundary { gate: 1, tick: 10 },
        ],
    };
    assert_ne!(grid.tick_at(0, 0.95), Some(10));
}

#[test]
fn theta_grid_tick_at_returns_none_when_interval_too_small() {
    let grid = ThetaGrid {
        boundaries: vec![
            GateBoundary { gate: 0, tick: 0 },
            GateBoundary { gate: 1, tick: 1 },
        ],
    };
    assert_eq!(grid.tick_at(0, 0.5), None);
}

#[test]
fn tick_at_returns_none_on_too_short_interval_for_interior_phase() {
    let grid = ThetaGrid {
        boundaries: vec![
            GateBoundary { gate: 0, tick: 0 },
            GateBoundary { gate: 1, tick: 1 },
        ],
    };
    assert_eq!(grid.tick_at(0, 0.5), None);
}

#[test]
fn tick_at_returns_boundary_ticks_for_phase_0_and_1() {
    let grid = ThetaGrid {
        boundaries: vec![
            GateBoundary { gate: 0, tick: 0 },
            GateBoundary { gate: 1, tick: 100 },
        ],
    };
    assert_eq!(grid.tick_at(0, 0.0), Some(0));
    assert_eq!(grid.tick_at(0, 1.0), Some(100));
}

#[test]
fn theta_grid_ensure_boundaries_until_adds_a_full_cycle() {
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.freq_hz = 1.0;
    rhythms.theta.phase = 0.0;
    let ctx = CoreTickCtx {
        now_tick: 0,
        frame_end: 1000,
        fs: 100.0,
        rhythms,
    };
    let t0 = 0;
    let expected = 100;
    let mut grid = ThetaGrid {
        boundaries: vec![GateBoundary { gate: 0, tick: t0 }],
    };
    grid.ensure_boundaries_until(&ctx, 1, None);
    assert_eq!(grid.boundaries.len(), 2);
    assert_eq!(grid.boundaries[1].tick, expected);
}

#[test]
fn theta_grid_ensure_boundaries_until_is_monotonic() {
    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.freq_hz = 1.0;
    rhythms.theta.phase = 0.0;
    let ctx = CoreTickCtx {
        now_tick: 0,
        frame_end: 1000,
        fs: 100.0,
        rhythms,
    };
    let mut grid = ThetaGrid {
        boundaries: vec![GateBoundary { gate: 0, tick: 0 }],
    };
    grid.ensure_boundaries_until(&ctx, 4, None);
    assert!(grid.boundaries.len() >= 5);
    for pair in grid.boundaries.windows(2) {
        assert_eq!(pair[1].gate, pair[0].gate + 1);
        assert!(pair[1].tick > pair[0].tick);
    }
}

#[test]
fn active_notes_tracks_note_on_and_off() {
    let ctx = CoreTickCtx {
        now_tick: 0,
        frame_end: 2,
        fs: 1000.0,
        rhythms: NeuralRhythms::default(),
    };
    let interval = OnsetRule::Accumulator {
        rate_hz: 1.0,
        refractory_gates: 0,
        acc: 1.0,
        next_allowed_gate: 0,
    };
    let mut engine = test_engine(interval, DurationRule::fixed_gate(0));
    let state = CoreState {
        is_alive: true,
        onset_allowed: true,
    };
    let timing_field = TimingField::from_values(0, vec![1.0]);
    let candidates = vec![CandidatePoint { tick: 0, gate: 0 }];
    let mut cmds = Vec::new();
    let mut events = Vec::new();
    let mut onsets = Vec::new();
    let mut timing_grid = ThetaGrid::from_candidates(&candidates);
    engine.process_candidates(
        &ctx,
        &candidates,
        &mut timing_grid,
        &timing_field,
        &state,
        None,
        &mut cmds,
        &mut events,
        &mut onsets,
    );
    assert_eq!(engine.active_notes, 0, "note on/off should settle to zero");
    assert!(cmds.iter().any(|cmd| matches!(cmd, ToneCmd::On { .. })));
    assert!(cmds.iter().any(|cmd| matches!(cmd, ToneCmd::Off { .. })));
}

#[test]
fn active_notes_increments_on_note_on() {
    let ctx = CoreTickCtx {
        now_tick: 0,
        frame_end: 2,
        fs: 1000.0,
        rhythms: NeuralRhythms::default(),
    };
    let interval = OnsetRule::Accumulator {
        rate_hz: 1.0,
        refractory_gates: 0,
        acc: 1.0,
        next_allowed_gate: 0,
    };
    let mut engine = test_engine(interval, DurationRule::fixed_gate(2));
    let state = CoreState {
        is_alive: true,
        onset_allowed: true,
    };
    let timing_field = TimingField::from_values(0, vec![1.0]);
    let candidates = vec![CandidatePoint { tick: 0, gate: 0 }];
    let mut cmds = Vec::new();
    let mut events = Vec::new();
    let mut onsets = Vec::new();
    let mut timing_grid = ThetaGrid {
        boundaries: vec![
            GateBoundary { gate: 0, tick: 0 },
            GateBoundary { gate: 1, tick: 10 },
            GateBoundary { gate: 2, tick: 20 },
            GateBoundary { gate: 3, tick: 30 },
        ],
    };
    engine.process_candidates(
        &ctx,
        &candidates,
        &mut timing_grid,
        &timing_field,
        &state,
        None,
        &mut cmds,
        &mut events,
        &mut onsets,
    );
    assert_eq!(engine.active_notes, 1);
    assert!(cmds.iter().any(|cmd| matches!(cmd, ToneCmd::On { .. })));
    assert!(!cmds.iter().any(|cmd| matches!(cmd, ToneCmd::Off { .. })));
}

#[test]
fn pending_off_drains_without_candidates() {
    let ctx = CoreTickCtx {
        now_tick: 0,
        frame_end: 10,
        fs: 1000.0,
        rhythms: NeuralRhythms::default(),
    };
    let mut engine = test_engine(OnsetRule::None, DurationRule::fixed_gate(1));
    engine.schedule_note_off(42, 5);
    let state = CoreState {
        is_alive: true,
        onset_allowed: true,
    };
    let timing_field = TimingField::from_values(0, Vec::new());
    let candidates: Vec<CandidatePoint> = Vec::new();
    let mut timing_grid = ThetaGrid::from_candidates(&candidates);
    let mut cmds = Vec::new();
    let mut events = Vec::new();
    let mut onsets = Vec::new();
    engine.process_candidates(
        &ctx,
        &candidates,
        &mut timing_grid,
        &timing_field,
        &state,
        None,
        &mut cmds,
        &mut events,
        &mut onsets,
    );
    assert!(cmds.iter().any(|cmd| matches!(
        cmd,
        ToneCmd::Off {
            tone_id: 42,
            off_tick: 5
        }
    )));
}

#[test]
fn note_off_emits_before_note_on_when_due() {
    let ctx = CoreTickCtx {
        now_tick: 0,
        frame_end: 2,
        fs: 1000.0,
        rhythms: NeuralRhythms::default(),
    };
    let candidates = vec![CandidatePoint { tick: 0, gate: 0 }];
    let timing_field = TimingField::from_values(0, vec![1.0]);
    let mut engine = test_engine(
        OnsetRule::Custom(Box::new(|_, _| Some(OnsetKick { strength: 1.0 }))),
        DurationRule::Custom(Box::new(|_| DurationPlan::None)),
    );
    engine.schedule_note_off(99, 0);
    let state = CoreState {
        is_alive: true,
        onset_allowed: true,
    };
    let mut cmds = Vec::new();
    let mut events = Vec::new();
    let mut onsets = Vec::new();
    let mut timing_grid = ThetaGrid::from_candidates(&candidates);
    engine.process_candidates(
        &ctx,
        &candidates,
        &mut timing_grid,
        &timing_field,
        &state,
        None,
        &mut cmds,
        &mut events,
        &mut onsets,
    );
    let off_pos = cmds
        .iter()
        .position(|cmd| matches!(cmd, ToneCmd::Off { tone_id: 99, .. }));
    let on_pos = cmds
        .iter()
        .position(|cmd| matches!(cmd, ToneCmd::On { .. }));
    assert!(off_pos.is_some());
    assert!(on_pos.is_some());
    assert!(off_pos < on_pos);
}

#[test]
fn schedule_hold_theta_fallback_never_schedules_before_onset() {
    let ctx = CoreTickCtx {
        now_tick: 0,
        frame_end: 1,
        fs: 100.0,
        rhythms: NeuralRhythms::default(),
    };
    let mut engine = test_engine(OnsetRule::None, DurationRule::fixed_gate(1));
    let onset = OnsetContext {
        tone_id: 7,
        tick: 10,
        gate: 0,
        exc_gate: 0.0,
        exc_slope: 0.0,
    };
    let mut timing_grid = ThetaGrid {
        boundaries: Vec::new(),
    };
    engine.schedule_hold_theta(onset, 1.0, &ctx, &mut timing_grid, None);
    let off_tick = engine
        .pending_off
        .peek()
        .map(|entry| entry.0.off_tick)
        .expect("pending off");
    assert!(off_tick >= onset.tick);
}

#[test]
fn schedule_hold_theta_does_not_panic_on_short_interval() {
    let ctx = CoreTickCtx {
        now_tick: 0,
        frame_end: 2,
        fs: 100.0,
        rhythms: NeuralRhythms::default(),
    };
    let mut engine = test_engine(OnsetRule::None, DurationRule::fixed_gate(1));
    let onset = OnsetContext {
        tone_id: 11,
        tick: 0,
        gate: 0,
        exc_gate: 0.0,
        exc_slope: 0.0,
    };
    let mut timing_grid = ThetaGrid {
        boundaries: vec![
            GateBoundary { gate: 0, tick: 0 },
            GateBoundary { gate: 1, tick: 1 },
        ],
    };
    engine.schedule_hold_theta(onset, 0.5, &ctx, &mut timing_grid, None);
    let off_tick = engine
        .pending_off
        .peek()
        .map(|entry| entry.0.off_tick)
        .expect("pending off");
    assert!(off_tick >= onset.tick);
}

#[test]
fn tick_at_large_tick_precision() {
    let grid = ThetaGrid {
        boundaries: vec![
            GateBoundary {
                gate: 0,
                tick: 100_000_000,
            },
            GateBoundary {
                gate: 1,
                tick: 100_000_100,
            },
        ],
    };
    assert_eq!(grid.tick_at(0, 0.5), Some(100_000_050));
}

#[test]
fn merge_candidates_keeps_distinct_gates() {
    let candidates = vec![
        CandidatePoint { tick: 10, gate: 0 },
        CandidatePoint { tick: 10, gate: 1 },
    ];
    let merged = PhonationEngine::merge_candidates(candidates);
    assert_eq!(merged.len(), 2);
    assert_ne!(merged[0].gate, merged[1].gate);
}

#[test]
fn config_constructs_onset_and_duration_rules() {
    let base = PhonationConfig {
        measure_accent: 0.0,
        mode: PhonationMode::Gated,
        onset: OnsetConfig::None,
        duration: DurationConfig::FixedGate { length_gates: 1 },
        clock: PhonationClockConfig::ThetaGate,
    };
    let changed = PhonationConfig {
        onset: OnsetConfig::Accumulator {
            rate: 2.0,
            refractory: 2,
        },
        duration: DurationConfig::Field {
            hold_min_theta: 0.2,
            hold_max_theta: 0.5,
            curve_k: 2.0,
            curve_x0: 0.4,
            drop_gain: 0.1,
        },
        ..base
    };
    let mut engine = PhonationEngine::from_config(&changed, 7);
    let state = CoreState {
        is_alive: true,
        onset_allowed: true,
    };
    let kick = engine.onset_rule.on_candidate(
        &IntervalInput {
            gate: 0,
            tick: 0,
            dt_sec: 1.0,
            weight: 1.0,
        },
        &state,
    );
    assert!(kick.is_some(), "expected accumulator interval to be active");
    let plan = engine.duration_rule.on_note_on(OnsetContext {
        tone_id: 1,
        tick: 0,
        gate: 0,
        exc_gate: 0.2,
        exc_slope: 0.1,
    });
    assert!(
        matches!(plan, DurationPlan::HoldTheta(_)),
        "expected field connect plan"
    );
}

#[test]
fn fixed_gate_connect_releases_on_next_gate_tick() {
    let mut rhythms = NeuralRhythms::default();
    rhythms.env_open = 1.0;
    rhythms.env_level = 1.0;
    let ctx = CoreTickCtx {
        now_tick: 0,
        frame_end: 10,
        fs: 1000.0,
        rhythms,
    };
    let interval = OnsetRule::Accumulator {
        rate_hz: 250.0,
        refractory_gates: 0,
        acc: 0.0,
        next_allowed_gate: 0,
    };
    let stub_ticks: Vec<Tick> = vec![0, 4, 8];
    let mut stub_index = 0usize;
    let mut engine = test_engine(interval, DurationRule::fixed_gate(1));
    engine.clock = PhonationClock::Custom(Box::new(
        move |ctx: &CoreTickCtx, out: &mut Vec<CandidatePoint>| {
            while stub_index < stub_ticks.len() {
                let tick = stub_ticks[stub_index];
                if tick < ctx.now_tick || tick >= ctx.frame_end {
                    return;
                }
                let gate = stub_index as u64;
                out.push(CandidatePoint { tick, gate });
                stub_index += 1;
            }
        },
    ));
    let state = CoreState {
        is_alive: true,
        onset_allowed: true,
    };
    let mut cmds = Vec::new();
    let mut events = Vec::new();
    let mut onsets = Vec::new();
    engine.tick(
        &ctx,
        &state,
        None,
        0.0,
        1.0,
        None,
        &mut cmds,
        &mut events,
        &mut onsets,
    );
    assert!(events.len() >= 2, "expected at least two note ons");
    let first = events[0];
    let second = events[1];
    let off_tick = cmds.iter().find_map(|cmd| match cmd {
        ToneCmd::Off { tone_id, off_tick } if *tone_id == first.tone_id => Some(*off_tick),
        _ => None,
    });
    assert_eq!(off_tick, Some(second.onset_tick));
}

#[test]
fn timing_field_applies_social_coupling() {
    let mut rhythms = NeuralRhythms::default();
    rhythms.env_open = 1.0;
    rhythms.env_level = 1.0;
    let ctx = CoreTickCtx {
        now_tick: 0,
        frame_end: 20,
        fs: 1000.0,
        rhythms,
    };
    let grid = ThetaGrid {
        boundaries: vec![
            GateBoundary { gate: 0, tick: 0 },
            GateBoundary { gate: 1, tick: 10 },
        ],
    };
    let trace = SocialDensityTrace {
        start_tick: 0,
        bin_ticks: 10,
        bins: vec![1.0, 0.0],
    };
    let timing_field = TimingField::build_from(&ctx, &grid, Some((&trace, -1.0)), 1.0);
    assert!(timing_field.e(0) < 0.5);
    assert!((timing_field.e(1) - 1.0).abs() < 1e-6);
}

#[test]
fn field_connect_holds_longer_with_high_excitation() {
    let mut connect = DurationRule::field(0.25, 1.0, 10.0, 0.5, 0.0);
    let low_plan = connect.on_note_on(OnsetContext {
        tone_id: 1,
        tick: 0,
        gate: 0,
        exc_gate: 0.0,
        exc_slope: 0.0,
    });
    let high_plan = connect.on_note_on(OnsetContext {
        tone_id: 2,
        tick: 0,
        gate: 0,
        exc_gate: 1.0,
        exc_slope: 0.0,
    });
    let hold_low = match low_plan {
        DurationPlan::HoldTheta(hold) => hold,
        _ => panic!("expected hold plan for low excitation"),
    };
    let hold_high = match high_plan {
        DurationPlan::HoldTheta(hold) => hold,
        _ => panic!("expected hold plan for high excitation"),
    };
    assert!(hold_high > hold_low);
}

#[test]
fn field_connect_schedules_off_tick_when_next_boundary_not_in_candidates() {
    let onset_tick: Tick = 0;

    let mut rhythms = NeuralRhythms::default();
    rhythms.theta.freq_hz = 1.0;
    rhythms.theta.phase = 0.0;
    let fs = 100.0;
    // Half a 1 Hz cycle at 100 samples/sec is exactly 50 ticks.
    let expected_off_tick: Tick = 50;
    let ctx = CoreTickCtx {
        now_tick: 0,
        frame_end: expected_off_tick.saturating_add(1),
        fs,
        rhythms,
    };
    let timing_field = TimingField::from_values(0, vec![1.0, 1.0]);
    let state = CoreState {
        is_alive: true,
        onset_allowed: true,
    };
    let candidates = vec![CandidatePoint { tick: 0, gate: 0 }];
    let mut engine = test_engine(
        OnsetRule::Custom(Box::new(move |c, _| {
            if c.tick == onset_tick {
                Some(OnsetKick { strength: 1.0 })
            } else {
                None
            }
        })),
        DurationRule::field(0.5, 0.5, 10.0, 0.5, 0.0),
    );
    let mut cmds = Vec::new();
    let mut events = Vec::new();
    let mut onsets = Vec::new();
    let mut timing_grid = ThetaGrid::from_candidates(&candidates);
    engine.process_candidates(
        &ctx,
        &candidates,
        &mut timing_grid,
        &timing_field,
        &state,
        None,
        &mut cmds,
        &mut events,
        &mut onsets,
    );
    assert!(cmds.iter().any(|cmd| matches!(
        cmd,
        ToneCmd::Off {
            off_tick,
            ..
        } if *off_tick == expected_off_tick
    )));
}

#[test]
fn field_connect_note_off_survives_empty_hop() {
    let onset_tick: Tick = 0;

    let timing_field = TimingField::from_values(0, vec![1.0, 1.0]);
    let state = CoreState {
        is_alive: true,
        onset_allowed: true,
    };
    let mut engine = test_engine(
        OnsetRule::Custom(Box::new(move |c, _| {
            if c.tick == onset_tick {
                Some(OnsetKick { strength: 1.0 })
            } else {
                None
            }
        })),
        DurationRule::field(1.0, 1.0, 10.0, 0.5, 0.0),
    );

    let ctx1 = CoreTickCtx {
        now_tick: 0,
        frame_end: 50,
        fs: 1000.0,
        rhythms: NeuralRhythms::default(),
    };
    let candidates = vec![CandidatePoint { tick: 0, gate: 0 }];
    let mut timing_grid = ThetaGrid {
        boundaries: vec![
            GateBoundary { gate: 0, tick: 0 },
            GateBoundary { gate: 1, tick: 100 },
            GateBoundary { gate: 2, tick: 200 },
        ],
    };
    let mut cmds = Vec::new();
    let mut events = Vec::new();
    let mut onsets = Vec::new();
    engine.process_candidates(
        &ctx1,
        &candidates,
        &mut timing_grid,
        &timing_field,
        &state,
        None,
        &mut cmds,
        &mut events,
        &mut onsets,
    );
    let note_on_id = cmds.iter().find_map(|cmd| match cmd {
        ToneCmd::On { tone_id, .. } => Some(*tone_id),
        _ => None,
    });
    assert!(note_on_id.is_some());
    assert!(!cmds.iter().any(|cmd| matches!(cmd, ToneCmd::Off { .. })));

    let ctx2 = CoreTickCtx {
        now_tick: 50,
        frame_end: 120,
        fs: 1000.0,
        rhythms: NeuralRhythms::default(),
    };
    let empty: Vec<CandidatePoint> = Vec::new();
    let mut timing_grid = ThetaGrid::from_candidates(&empty);
    let mut cmds2 = Vec::new();
    let mut events2 = Vec::new();
    let mut onsets2 = Vec::new();
    engine.process_candidates(
        &ctx2,
        &empty,
        &mut timing_grid,
        &timing_field,
        &state,
        None,
        &mut cmds2,
        &mut events2,
        &mut onsets2,
    );
    let off_tick = cmds2.iter().find_map(|cmd| match cmd {
        ToneCmd::Off { tone_id, off_tick } if Some(*tone_id) == note_on_id => Some(*off_tick),
        _ => None,
    });
    assert_eq!(off_tick, Some(100));
}

#[test]
fn timing_field_social_coupling_clamps_exponent() {
    let mut rhythms = NeuralRhythms::default();
    rhythms.env_open = 1.0;
    rhythms.env_level = 1.0;
    let ctx = CoreTickCtx {
        now_tick: 0,
        frame_end: 1,
        fs: 1000.0,
        rhythms,
    };
    let grid = ThetaGrid {
        boundaries: vec![GateBoundary { gate: 0, tick: 0 }],
    };
    let trace = SocialDensityTrace {
        start_tick: 0,
        bin_ticks: 1,
        bins: vec![1000.0],
    };
    let timing_field = TimingField::build_from(&ctx, &grid, Some((&trace, 1000.0)), 1.0);
    assert!(timing_field.e(0).is_finite());
}

#[test]
fn flow_next_ioi_depth_zero_returns_base_interval() {
    let mut cluster_remaining = 2;
    let mut rng = SmallRng::seed_from_u64(7);

    let ioi = OnsetRule::flow_next_ioi(2.5, 0.0, &mut cluster_remaining, &mut rng);

    assert!((ioi - 0.4).abs() < f32::EPSILON);
    assert_eq!(cluster_remaining, 0);
}

#[test]
fn flow_next_ioi_keeps_stationary_local_variation() {
    let mut cluster_remaining = 0;
    let mut rng = SmallRng::seed_from_u64(11);
    let mut iois = Vec::new();
    for _ in 0..96 {
        iois.push(OnsetRule::flow_next_ioi(
            2.5,
            0.9,
            &mut cluster_remaining,
            &mut rng,
        ));
    }

    let first = iois[..32].iter().sum::<f32>() / 32.0;
    let middle = iois[32..64].iter().sum::<f32>() / 32.0;
    let last = iois[64..].iter().sum::<f32>() / 32.0;
    let min = first.min(middle).min(last);
    let max = first.max(middle).max(last);
    assert!(max / min < 1.55, "block means drifted too far: {iois:?}");
    assert!(iois.iter().any(|ioi| *ioi < 0.28));
    assert!(iois.iter().any(|ioi| *ioi > 0.50));
}

// --- onset_allowed gating (Item A regression + phonation gate) ---

#[test]
fn tick_hold_defers_onset_while_disallowed_then_fires_when_allowed() {
    // Regression for the latent Hold-mode bug: gating onset *emission*
    // (rather than discarding an already-emitted onset downstream) must
    // leave `note_on_sent` unconsumed while disallowed, so the onset can
    // still fire once allowed.
    let mut engine = test_engine(OnsetRule::None, DurationRule::fixed_gate(0));
    engine.mode = PhonationMode::Hold;
    let gated = CoreState {
        is_alive: true,
        onset_allowed: false,
    };
    let ctx1 = CoreTickCtx {
        now_tick: 0,
        frame_end: 1,
        fs: 1000.0,
        rhythms: NeuralRhythms::default(),
    };
    let mut cmds = Vec::new();
    let mut events = Vec::new();
    let mut onsets = Vec::new();
    engine.tick(
        &ctx1,
        &gated,
        None,
        0.0,
        1.0,
        None,
        &mut cmds,
        &mut events,
        &mut onsets,
    );
    assert!(cmds.is_empty(), "no onset while disallowed");
    assert!(events.is_empty());
    assert!(
        !engine.hold.note_on_sent,
        "note_on_sent must not be consumed while gated"
    );

    let open = CoreState {
        is_alive: true,
        onset_allowed: true,
    };
    let ctx2 = CoreTickCtx {
        now_tick: 1,
        frame_end: 2,
        fs: 1000.0,
        rhythms: NeuralRhythms::default(),
    };
    cmds.clear();
    events.clear();
    onsets.clear();
    engine.tick(
        &ctx2,
        &open,
        None,
        0.0,
        1.0,
        None,
        &mut cmds,
        &mut events,
        &mut onsets,
    );
    assert!(cmds.iter().any(|cmd| matches!(cmd, ToneCmd::On { .. })));
    assert_eq!(events.len(), 1);
    assert!(engine.hold.note_on_sent);
    assert!(!events[0].participation_onset);
}

#[test]
fn tick_reuses_scratch_buffers_across_hops() {
    // The per-hop path must settle into its buffers: once warm, no scratch
    // buffer capacity grows again (i.e. no per-hop heap allocation).
    let mut rhythms = NeuralRhythms::default();
    rhythms.env_open = 1.0;
    rhythms.env_level = 1.0;
    rhythms.theta.freq_hz = 5.0;
    rhythms.theta.phase = 0.0;
    let mut engine = test_engine(
        OnsetRule::Always { strength: 1.0 },
        DurationRule::fixed_gate(1),
    );
    let mut gate = 0u64;
    engine.clock = PhonationClock::Custom(Box::new(
        move |ctx: &CoreTickCtx, out: &mut Vec<CandidatePoint>| {
            out.push(CandidatePoint {
                tick: ctx.now_tick,
                gate,
            });
            gate += 1;
            out.push(CandidatePoint {
                tick: ctx.now_tick + 50,
                gate,
            });
            gate += 1;
        },
    ));
    let state = CoreState {
        is_alive: true,
        onset_allowed: true,
    };
    let mut cmds = Vec::new();
    let mut events = Vec::new();
    let mut onsets = Vec::new();
    let mut caps = None;
    for hop in 0..4u64 {
        let ctx = CoreTickCtx {
            now_tick: hop * 100,
            frame_end: hop * 100 + 100,
            fs: 1000.0,
            rhythms,
        };
        cmds.clear();
        events.clear();
        onsets.clear();
        engine.tick(
            &ctx,
            &state,
            None,
            0.0,
            1.0,
            None,
            &mut cmds,
            &mut events,
            &mut onsets,
        );
        assert_eq!(events.len(), 2, "hop {hop} should emit both onsets");
        let now = (
            engine.scratch_candidates.capacity(),
            engine.scratch_merged.capacity(),
            engine.scratch_grid.boundaries.capacity(),
            engine.scratch_field.e_gate.capacity(),
        );
        if hop == 1 {
            assert!(now.0 > 0 && now.1 > 0 && now.2 > 0 && now.3 > 0);
            caps = Some(now);
        } else if hop > 1 {
            assert_eq!(caps, Some(now), "scratch buffers reallocated on hop {hop}");
        }
    }
}

#[test]
fn coupling_clock_keeps_advancing_while_onset_disallowed_then_resumes_promptly() {
    // Non-Hold (candidate-processing) suppression: the onset rule must not
    // fire while gated, but the clock (an entrained oscillator here) must
    // keep advancing its own phase so the voice can join the beat as soon
    // as the gate opens, instead of restarting cold.
    let mut engine = test_engine(
        OnsetRule::Always { strength: 1.0 },
        DurationRule::fixed_gate(0),
    );
    engine.clock = PhonationClock::Coupling(CouplingClock::new(0.0, 10.0, 0.0, 0.0, 1));
    fn phase_of(engine: &PhonationEngine) -> f64 {
        match &engine.clock {
            PhonationClock::Coupling(c) => c.phase,
            _ => panic!("expected coupling clock"),
        }
    }

    let rhythms = NeuralRhythms::default();
    let gated = CoreState {
        is_alive: true,
        onset_allowed: false,
    };
    let open = CoreState {
        is_alive: true,
        onset_allowed: true,
    };

    let phase0 = phase_of(&engine);

    // base_rate_hz=10 at fs=1000 gives a ~100-tick period; a 350-tick hop
    // would normally emit several onsets. None should fire while gated.
    let ctx1 = CoreTickCtx {
        now_tick: 0,
        frame_end: 350,
        fs: 1000.0,
        rhythms,
    };
    let mut cmds = Vec::new();
    let mut events = Vec::new();
    let mut onsets = Vec::new();
    engine.tick(
        &ctx1,
        &gated,
        None,
        0.0,
        1.0,
        None,
        &mut cmds,
        &mut events,
        &mut onsets,
    );
    assert!(onsets.is_empty(), "no onsets while gated");
    assert!(!cmds.iter().any(|cmd| matches!(cmd, ToneCmd::On { .. })));
    let phase1 = phase_of(&engine);
    assert!(
        phase1 > phase0 + 2.0,
        "clock phase must keep advancing while gated: {phase0} -> {phase1}"
    );

    // Gate opens: onsets resume in the very next hop, picking up the phase
    // already accumulated during the gated window.
    let ctx2 = CoreTickCtx {
        now_tick: 350,
        frame_end: 700,
        fs: 1000.0,
        rhythms,
    };
    cmds.clear();
    events.clear();
    onsets.clear();
    engine.tick(
        &ctx2,
        &open,
        None,
        0.0,
        1.0,
        None,
        &mut cmds,
        &mut events,
        &mut onsets,
    );
    assert!(
        !onsets.is_empty(),
        "onsets resume promptly once the gate opens"
    );
    assert!(cmds.iter().any(|cmd| matches!(cmd, ToneCmd::On { .. })));
}
