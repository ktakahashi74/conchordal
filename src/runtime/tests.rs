#[test]
fn participation_outcome_waits_for_actual_audio_and_keeps_gaps_unknown() {
    let prediction = super::ParticipationPrediction {
        issued_frame: 5,
        onset_frame: 11,
        forecast_observed_frame: 0,
        target_start_frame: 20,
        target_end_frame: 30,
        pred_continuation_habitat_band_energy: [0.4; 3],
    };
    let mut observer =
        crate::core::temporal_expectation::AcousticTemporalExpectation::new(1000).unwrap();
    let mut observations = Vec::new();
    observer.process(0, &[0.0; 30], |o| observations.push(o));
    assert!(super::participation_outcome(1, 1000, prediction, Some(observations[1])).is_none());
    let observed =
        super::participation_outcome(1, 1000, prediction, Some(observations[2])).unwrap();
    assert_eq!(observed.status, "observed");
    assert_eq!(observed.observed_habitat_band_energy, Some([0.0; 3]));
    let mut shifted = observations[2];
    shifted.start_frame += 1;
    shifted.end_frame += 1;
    let gap = super::participation_outcome(1, 1000, prediction, Some(shifted)).unwrap();
    assert_eq!(gap.status, "input_gap");
    assert_eq!(gap.observed_habitat_band_energy, None);
    let unfinished = super::participation_outcome(1, 1000, prediction, None).unwrap();
    assert_eq!(unfinished.status, "end_of_input");
    assert_eq!(unfinished.observed_habitat_band_energy, None);
}
use super::*;
use crate::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
use crate::core::roughness_kernel::{KernelParams, RoughnessKernel};
use crate::core::timebase::Timebase;

#[test]
fn analysis_ok_truth_table() {
    assert!(analysis_ok(0, None, MAX_LANDSCAPE_LAG_FRAMES));
    assert!(analysis_ok(1, Some(0), MAX_LANDSCAPE_LAG_FRAMES));
    assert!(analysis_ok(10, Some(9), MAX_LANDSCAPE_LAG_FRAMES));
    assert!(!analysis_ok(10, Some(8), MAX_LANDSCAPE_LAG_FRAMES));
    assert!(!analysis_ok(10, None, MAX_LANDSCAPE_LAG_FRAMES));
}

#[test]
fn analysis_runtime_core_uses_config_values() {
    let mut config = AppConfig::default();
    config.analysis.nfft = 2048;
    config.analysis.hop_size = 256;
    config.analysis.tau_ms = 25.0;
    config.psychoacoustics.loudness_exp = 0.31;
    config.psychoacoustics.roughness_k = 0.2;
    config.psychoacoustics.consonance.field.kernel.a = 0.9;
    config.psychoacoustics.consonance.field.level.beta = 3.0;
    config.psychoacoustics.consonance.density.roughness_gain = 1.5;

    let core = build_analysis_runtime_core(&config, 44_100);

    assert!(core.lparams.habituation.enabled);
    assert_eq!(core.fs, 44_100.0);
    assert_eq!(core.hop, 256);
    assert!((core.hop_duration.as_secs_f32() - 256.0 / 44_100.0).abs() < 1e-9);
    assert_eq!(core.lparams.tau_ms, 25.0);
    assert_eq!(core.lparams.loudness_exp, 0.31);
    assert_eq!(core.lparams.roughness_k, 0.2);
    assert_eq!(core.lparams.consonance_kernel.a, 0.9);
    assert_eq!(core.lparams.consonance_representation.beta, 3.0);
    assert_eq!(core.lparams.consonance_density_roughness_gain, 1.5);
    assert_eq!(core.landscape.space.n_bins(), core.nsgt.space().n_bins());
}

#[test]
fn footprint_only_wiring_leaves_observation_and_candidate_analysis_off() {
    use crate::config::{FootprintSource, TemporalOnsetComparisonConfig};
    let script = std::env::temp_dir().join(format!(
        "conchordal-footprint-only-{}-{}.rhai",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    ));
    std::fs::write(
        &script,
        r#"
seed(20261007);
let group = place(sine().amp(0.03).flow().cycles(3), at(440.0));
wait(0.08);
release(group);
wait(0.08);
"#,
    )
    .unwrap();
    for comparison in [
        None,
        Some(FootprintSource::Body),
        Some(FootprintSource::Proxy),
    ] {
        let mut config = AppConfig::default();
        config.analysis.nfft = 2048;
        config.dcc.coupling_strength = 0.0;
        config.temporal_onset_comparison =
            comparison.map(|footprint| TemporalOnsetComparisonConfig {
                footprint,
                ..Default::default()
            });
        config.validate().unwrap();
        let scenario = compile_scenario_from_script(&script, None).unwrap();
        assert_eq!(scenario.temporal_mode, crate::scenario::TemporalMode::Off);
        let saw_request = Arc::new(AtomicBool::new(false));
        let observed = Arc::clone(&saw_request);
        let probe: OfflineBodyProbe = Box::new(move |state, _, _, _| {
            assert!(state.schedule_renderer.body_capture.is_none());
            assert!(state.body_snapshot.is_none());
            assert!(
                state
                    .temporal_frames
                    .iter()
                    .all(|frame| frame.received_frames == 0)
            );
            assert_eq!(state.footprint_worker.is_some(), comparison.is_some());
            if let Some(worker) = &state.footprint_worker
                && worker.stats.footprint_requested > 0
            {
                observed.store(true, Ordering::Relaxed);
            }
        });
        let wiring = wire_runtime(
            &config,
            config.audio.sample_rate,
            "footprint only".into(),
            scenario,
            Arc::new(AtomicBool::new(false)),
            WiringOptions {
                offline_body_probe: Some(probe),
                #[cfg(test)]
                raw_t2_delay: Duration::ZERO,
                ui_tx: None,
                listener_forced: false,
                wait_user_exit: false,
                start_playing: true,
                audio_prod: None,
                wav_tx: None,
                reporter: None,
                deterministic_analysis: true,
                deterministic_footprints: true,
                guard_meter: None,
                underrun_frames: None,
                reserve_runtime_ids_through: 0,
                profile: None,
                audio_counters: None,
            },
        )
        .unwrap();
        assert!(wiring.listener_analysis_handle.is_none());
        join_thread("worker", wiring.worker_handle)
            .unwrap()
            .unwrap();
        join_thread("analysis", wiring.analysis_handle).unwrap();
        assert_eq!(saw_request.load(Ordering::Relaxed), comparison.is_some());
    }
    std::fs::remove_file(script).unwrap();
}

#[test]
fn instrument_ring_wiring_retains_tails_and_gui_waits_after_finish() {
    use ringbuf::traits::{Consumer, Split};
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "conchordal-instrument-renderer-{}-{stamp}",
        std::process::id()
    ));
    std::fs::create_dir(&dir).unwrap();
    for wait_user_exit in [false, true] {
        let mut native_hops = Vec::new();
        for body in ["sine", "harmonic", "modal"] {
            #[cfg(feature = "profile-alloc")]
            let mut legacy_counts = Vec::new();
            for prototype in [false, true] {
                let script = dir.join("scenario.rhai");
                std::fs::write(&script, format!(
                    "let a = place({body}().brain(\"drone\").sustain().anchor().amp(0.05).adsr(0.01, 0.0, 1.0, 0.01), at(220.0));\nwait(0.1);\nrelease(a);\nwait(0.03);\n"
                )).unwrap();
                let mut config = AppConfig {
                    render_prototype: prototype,
                    ..Default::default()
                };
                config.analysis.nfft = 2048;
                config.dcc.coupling_strength = 0.0;
                let scenario = compile_scenario_from_script(&script, Some(73)).unwrap();
                let stop = Arc::new(AtomicBool::new(false));
                let consume_stop = Arc::clone(&stop);
                let (prod, mut cons) = ringbuf::HeapRb::<f32>::new(2048).split();
                let consumer = thread::spawn(move || {
                    let mut scratch = [0.0_f32; 512];
                    let mut peak = 0.0_f32;
                    loop {
                        let count = cons.pop_slice(&mut scratch);
                        for &sample in &scratch[..count] {
                            assert!(sample.is_finite());
                            peak = peak.max(sample.abs());
                        }
                        if count == 0 && consume_stop.load(Ordering::SeqCst) {
                            return peak;
                        }
                        thread::yield_now();
                    }
                });
                let profile_path = dir.join("profile.json");
                let profile = RunProfile::create(
                    profile_path.to_str().unwrap(),
                    73,
                    false,
                    0.0,
                    wait_user_exit,
                    48_000,
                    512,
                    None,
                )
                .unwrap();
                let probe: OfflineBodyProbe = Box::new(move |state, _, _, _| {
                    assert_eq!(state.schedule_renderer.phase3_enabled(), prototype);
                    // Exclude wall-clock UI publication from the allocation control.
                    state.last_ui_update = Instant::now();
                });
                let (ui_tx, ui_rx) = bounded(UI_CHANNEL_CAPACITY);
                let wiring = wire_runtime(
                    &config,
                    48_000,
                    "instrument renderer".into(),
                    scenario,
                    Arc::clone(&stop),
                    WiringOptions {
                        offline_body_probe: Some(probe),
                        #[cfg(test)]
                        raw_t2_delay: Duration::ZERO,
                        ui_tx: Some(ui_tx),
                        listener_forced: wait_user_exit,
                        wait_user_exit,
                        start_playing: true,
                        audio_prod: Some(prod),
                        wav_tx: None,
                        reporter: None,
                        deterministic_analysis: true,
                        deterministic_footprints: false,
                        guard_meter: None,
                        underrun_frames: None,
                        reserve_runtime_ids_through: 0,
                        profile: Some(profile),
                        audio_counters: None,
                    },
                )
                .unwrap();
                loop {
                    let frame = ui_rx.recv_timeout(Duration::from_secs(30)).unwrap();
                    if frame.meta.playback_state == PlaybackState::Finished {
                        break;
                    }
                }
                if wait_user_exit {
                    assert!(!wiring.worker_handle.is_finished());
                    stop.store(true, Ordering::SeqCst);
                }
                join_thread("worker", wiring.worker_handle)
                    .unwrap()
                    .unwrap();
                join_thread("analysis", wiring.analysis_handle).unwrap();
                if let Some(handle) = wiring.listener_analysis_handle {
                    join_thread("listener", handle).unwrap();
                }
                assert!(consumer.join().unwrap() > 0.0);
                let profile: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(&profile_path).unwrap()).unwrap();
                let hops = profile["hops"].as_array().unwrap();
                if prototype {
                    native_hops.push(hops.len());
                }
                #[cfg(feature = "profile-alloc")]
                {
                    let counts: Vec<_> = hops
                        .iter()
                        .filter(|h| (0.02..0.08).contains(&h["time_sec"].as_f64().unwrap()))
                        .map(|h| {
                            (
                                h["worker_allocations"]["count"].as_u64().unwrap(),
                                h["worker_allocations"]["bytes"].as_u64().unwrap(),
                            )
                        })
                        .collect();
                    assert!(!counts.is_empty());
                    if prototype {
                        assert_eq!(counts, legacy_counts, "{body} GUI={wait_user_exit}");
                        println!(
                            "INSTRUMENT_ALLOC body={body} GUI={wait_user_exit} quiet_hops={} counts={counts:?}",
                            counts.len()
                        );
                    } else {
                        legacy_counts = counts;
                    }
                }
            }
        }
        assert!(native_hops[2] > native_hops[0] && native_hops[2] > native_hops[1]);
    }
    std::fs::remove_dir_all(dir).unwrap();
}

fn build_test_params(space: &Log2Space) -> LandscapeParams {
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
fn scenario_dispatch_after_ten_minutes_uses_audio_frame_time() {
    let fs = 48_000.0;
    let hop = 512;
    let timebase = Timebase { fs, hop };
    let cfg = WorkerConfig {
        loo_populations: Vec::new(),
        flora_metric: false,
        render_prototype: false,
        temporal_meter_t2: false,
        onset_comparison: None,
        scenario_name: "clock regression".into(),
        wait_user_exit: false,
        start_flag: Arc::new(AtomicBool::new(true)),
        exiting: Arc::new(AtomicBool::new(false)),
        scaffold: Default::default(),
        meter_shaping: Default::default(),
        guard_meter: None,
        underrun_frames: None,
        audio_counters: None,
        dcc_coupler: DccCoupler::new(Default::default()),
        hop,
        hop_duration: Duration::from_secs_f64(hop as f64 / fs as f64),
        fs,
        deterministic_analysis: true,
        deterministic_footprints: true,
    };
    let scenario = crate::scenario::Scenario {
        temporal_mode: crate::scenario::TemporalMode::Off,
        seed: 1,
        output_gain_db: 0.0,
        output_gain_position: None,
        control_update_mode: Default::default(),
        scaffold: Default::default(),
        meter_shaping: Default::default(),
        scene_markers: Vec::new(),
        events: vec![crate::scenario::TimedEvent {
            time: 605.0,
            order: 0,
            actions: vec![Action::Finish],
        }],
        duration_sec: 605.0,
    };
    let space = Log2Space::new(200.0, 4_000.0, 12);
    let mut state = WorkerState::new(
        Community::new(timebase),
        Conductor::from_scenario(scenario),
        Landscape::new(space.clone()),
        build_test_params(&space),
        DorsalStream::new(fs),
        None,
        &cfg,
    );
    for frame in [56_680, 56_718, 56_719] {
        state.frame_idx = frame;
        let now_tick = timebase.frame_start_tick(frame);
        state.generator_model.advance_to(now_tick);
        advance_population(&mut state, &cfg, now_tick, ListenerPressure::default());
        assert_eq!(state.pop.abort_requested, frame == 56_719);
    }
}

#[test]
fn explicitly_disabled_habituation_preserves_base_view_bits() {
    let config: AppConfig =
        toml::from_str("[psychoacoustics.habituation]\nenabled = false\n").unwrap();
    let mut core = build_analysis_runtime_core(&config, 48_000);
    assert!(!core.lparams.habituation.enabled);
    let mut hab = crate::core::habituation::HabituationField::new(
        &core.lparams.habituation,
        core.lparams.consonance_representation.theta,
        core.landscape.space.n_bins(),
    );
    for hop in 0..3 {
        for i in 0..core.landscape.space.n_bins() {
            core.landscape.harmonicity01[i] = ((i + hop) % 7) as f32 / 7.0;
            core.landscape.roughness01[i] = ((i + hop) % 11) as f32 / 11.0;
            core.landscape.subjective_intensity[i] = 0.25;
        }
        core.landscape.recompute_consonance(&core.lparams);
        drive_and_apply_habituation(
            &mut core.landscape,
            &mut hab,
            &core.lparams,
            core.hop as f32 / core.fs,
        );
        for (base, effective) in [
            (
                &core.landscape.consonance_field_score,
                &core.landscape.consonance_field_score_eff,
            ),
            (
                &core.landscape.consonance_field_level,
                &core.landscape.consonance_field_level_eff,
            ),
            (
                &core.landscape.consonance_density_mass,
                &core.landscape.consonance_density_mass_eff,
            ),
        ] {
            assert!(
                base.iter()
                    .zip(effective)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
        }
        assert!(hab.state().iter().all(|h| h.to_bits() == 0.0f32.to_bits()));
    }
}

#[test]
fn listener_memory_is_independent_of_receive_batching() {
    let space = Log2Space::new(200.0, 4_000.0, 12);
    let mut params = build_test_params(&space);
    params.habituation.enabled = true;
    params.habituation.satiation_sec = 0.5;
    params.habituation.recovery_sec = 1.0;
    let timebase = Timebase {
        fs: params.fs,
        hop: 4_800,
    };
    let mut outcomes = Vec::new();
    for batch_size in [1, 3, 24] {
        let mut listener = ListenerTwin::with_sample_rate(params.fs, Default::default());
        let mut hab = crate::core::habituation::HabituationField::new(
            &params.habituation,
            params.consonance_representation.theta,
            space.n_bins(),
        );
        let (tx, rx) = bounded(batch_size);
        let mut last_frame = None;
        let mut last_state = None;
        for start in (0..24).step_by(batch_size) {
            let end = (start + batch_size).min(24);
            for frame_id in start..end {
                let mut frame = Landscape::new(space.clone());
                if frame_id < 18 {
                    frame.subjective_intensity[10] = 1.0;
                    frame.harmonicity[10] = 10.0;
                }
                tx.send((frame_id as u64, Some(frame))).unwrap();
            }
            let (update, disconnected) = observe_listener_analysis_results(
                Some(&rx),
                &mut listener,
                &params,
                &mut hab,
                timebase,
                end as u64,
                0,
                &mut last_frame,
            );
            assert!(!disconnected);
            let state = update.unwrap().unwrap();
            assert_eq!(state.analysis_frame_id, end as u64 - 1);
            last_state = Some(state);
        }
        let retained = hab.state().to_vec();
        assert!(retained.iter().any(|&value| value > 0.0));
        let (update, _) = observe_listener_analysis_results(
            Some(&rx),
            &mut listener,
            &params,
            &mut hab,
            timebase,
            25,
            0,
            &mut last_frame,
        );
        assert!(update.is_none());
        assert_eq!(hab.state(), retained);
        let state = last_state.unwrap();
        outcomes.push((retained, state.stability_level, state.tension_level));
    }
    assert_eq!(outcomes[0], outcomes[1]);
    assert_eq!(outcomes[0], outcomes[2]);
}

#[test]
fn listener_analysis_gap_clears_state_instead_of_reusing_prior_pressure() {
    let space = Log2Space::new(200.0, 4_000.0, 12);
    let params = build_test_params(&space);
    let mut listener = ListenerTwin::with_sample_rate(params.fs, Default::default());
    let mut hab = crate::core::habituation::HabituationField::new(
        &params.habituation,
        params.consonance_representation.theta,
        space.n_bins(),
    );
    let timebase = Timebase {
        fs: params.fs,
        hop: 128,
    };
    let mut last_frame = Some(0);
    let (tx, rx) = bounded(2);
    let coupler = DccCoupler::new(crate::config::DccConfig {
        coupling_strength: 0.25,
        max_temperature_bonus: 0.1,
    });
    let mut evidence = Landscape::new(space.clone());
    evidence.subjective_intensity[2] = 1.0;
    evidence.harmonicity[3] = 10.0;
    tx.send((1, Some(evidence.clone()))).unwrap();
    let (update, _) = observe_listener_analysis_results(
        Some(&rx),
        &mut listener,
        &params,
        &mut hab,
        timebase,
        2,
        0,
        &mut last_frame,
    );
    let mut current = update.expect("initial evidence update");
    let initial_pressure = coupler.pressure(current);
    assert!(initial_pressure.temperature_bonus > 0.0);
    tx.send((3, None)).unwrap();
    let (update, disconnected) = observe_listener_analysis_results(
        Some(&rx),
        &mut listener,
        &params,
        &mut hab,
        timebase,
        4,
        0,
        &mut last_frame,
    );
    assert!(
        matches!(update, Some(None)),
        "a gap must invalidate the previous observation"
    );
    assert!(!disconnected);
    assert!(last_frame.is_none());
    current = update.unwrap();
    assert_eq!(coupler.pressure(current), ListenerPressure::default());
    let (update, _) = observe_listener_analysis_results(
        Some(&rx),
        &mut listener,
        &params,
        &mut hab,
        timebase,
        5,
        0,
        &mut last_frame,
    );
    assert!(
        update.is_none(),
        "an empty queue is distinct from invalidation"
    );
    assert_eq!(coupler.pressure(current), ListenerPressure::default());
    tx.send((4, Some(Landscape::new(space)))).unwrap();
    let (update, _) = observe_listener_analysis_results(
        Some(&rx),
        &mut listener,
        &params,
        &mut hab,
        timebase,
        6,
        5,
        &mut last_frame,
    );
    assert!(
        update.is_none(),
        "late pre-gap results must not restore pressure"
    );
    assert!(last_frame.is_none());
    assert_eq!(coupler.pressure(current), ListenerPressure::default());
    tx.send((5, Some(evidence))).unwrap();
    let (update, _) = observe_listener_analysis_results(
        Some(&rx),
        &mut listener,
        &params,
        &mut hab,
        timebase,
        6,
        5,
        &mut last_frame,
    );
    current = update.expect("post-gap evidence update");
    assert_eq!(last_frame, Some(5));
    assert_eq!(coupler.pressure(current), initial_pressure);
}

#[test]
fn pending_update_does_not_panic_when_analysis_channel_is_disconnected() {
    let space = Log2Space::new(80.0, 2_000.0, 96);
    let mut params = build_test_params(&space);
    let mut landscape = Landscape::new(space);
    let mut pop = Community::new(Timebase {
        fs: 48_000.0,
        hop: 256,
    });

    pop.apply_action(
        Action::UpdateLandscape {
            update: LandscapeUpdate {
                roughness_aversion: Some(2.0),
                pitch_objective_mode: None,
            },
        },
        &landscape,
    );

    let (analysis_update_tx, analysis_update_rx) = bounded::<LandscapeUpdate>(1);
    drop(analysis_update_rx);
    let changed = apply_pending_landscape_update(
        &mut pop,
        &mut params,
        &mut landscape,
        &analysis_update_tx,
        None,
    );
    assert!(changed);
}

#[test]
fn roughness_aversion_update_reaches_kernel_and_fields_without_touching_sensation() {
    let space = Log2Space::new(80.0, 2_000.0, 96);
    let mut params = build_test_params(&space);
    let mut landscape = Landscape::new(space);
    let n = landscape.roughness01.len();
    landscape.harmonicity = vec![0.6; n];
    landscape.roughness01 = vec![0.5; n];
    landscape.recompute_consonance(&params);
    let r01_before = landscape.roughness01.clone();
    let h01_before = landscape.harmonicity01.clone();
    let score_before = landscape.consonance_field_score[0];
    let mut pop = Community::new(Timebase {
        fs: 48_000.0,
        hop: 256,
    });
    let (tx, rx) = bounded::<LandscapeUpdate>(4);

    pop.apply_action(
        Action::UpdateLandscape {
            update: LandscapeUpdate {
                roughness_aversion: Some(0.0),
                pitch_objective_mode: None,
            },
        },
        &landscape,
    );
    assert!(apply_pending_landscape_update(
        &mut pop,
        &mut params,
        &mut landscape,
        &tx,
        None,
    ));

    assert_eq!(params.roughness_aversion, 0.0);
    assert_eq!(
        rx.try_recv()
            .expect("forwarded to analysis")
            .roughness_aversion,
        Some(0.0)
    );
    let kernel = params.consonance_field_kernel();
    assert_eq!(landscape.consonance_kernel.b.to_bits(), kernel.b.to_bits());
    assert_eq!(landscape.consonance_kernel.c.to_bits(), kernel.c.to_bits());
    assert_eq!(landscape.roughness01, r01_before);
    assert_eq!(landscape.harmonicity01, h01_before);
    assert_ne!(landscape.consonance_field_score[0], score_before);
}

#[test]
fn pcm_loo_late_scenario_patch_keeps_history_from_birth() {
    let dir = std::env::temp_dir().join(format!(
        "conchordal-loo-late-patch-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    ));
    let _output = test_output_directory::TestOutputDirectory::new(dir.clone());
    let script = dir.join("scenario.rhai");
    std::fs::write(
        &script,
        r#"
let own = place(sine().brain("drone").sustain().anchor().amp(0.1)
    .adsr(0.001, 0.0, 1.0, 0.01).leave_self_out(false), at(440.0));
let other = place(sine().brain("drone").sustain().anchor().amp(0.06)
    .adsr(0.001, 0.0, 1.0, 0.01), at(660.0));
wait(0.08);
own.leave_self_out(true);
wait(0.08);
release(own);
release(other);
wait(0.03);
"#,
    )
    .unwrap();
    let mut config = AppConfig::default();
    config.analysis.nfft = 2048;
    config.dcc.coupling_strength = 0.0;
    config.psychoacoustics.habituation.enabled = false;
    let scenario = compile_scenario_from_script(&script, Some(21)).unwrap();
    let (enable_time, population) = scenario
        .events
        .iter()
        .find_map(|event| {
            event.actions.iter().find_map(|action| match action {
                Action::UpdatePopulation {
                    population_id,
                    patch,
                } if patch.leave_self_out == Some(true) => Some((event.time, *population_id)),
                _ => None,
            })
        })
        .unwrap();
    let core = build_analysis_runtime_core(&config, config.audio.sample_rate);
    let mut reference = AnalysisStream::new(core.lparams.clone(), core.nsgt);
    let mut previous: Option<Landscape> = None;
    let counts = Arc::new(std::sync::Mutex::new([0_usize; 2]));
    let observed = counts.clone();
    let probe: OfflineBodyProbe = Box::new(move |state, tick, _, audio| {
        let Some(voice) = state
            .pop
            .voices
            .iter()
            .find(|voice| voice.metadata.population_id == population)
        else {
            return;
        };
        let enabled = voice.base_control.pitch.leave_self_out;
        assert_eq!(enabled, tick as f64 / 48_000.0 >= f64::from(enable_time));
        let source = state
            .schedule_renderer
            .loo_pcm
            .iter()
            .find(|source| {
                source.id == voice.id() && source.generation == voice.metadata.generation
            })
            .expect("a future self-exclusion consumer must be captured from birth");
        if let Some(expected) = &previous {
            assert_eq!(state.last_analysis_frame, Some(state.frame_idx - 1));
            assert_eq!(state.current_landscape.loo.len(), 1);
            let actual = &state.current_landscape.loo[0];
            assert_eq!(
                (actual.id, actual.generation),
                (source.id, source.generation)
            );
            assert_eq!(actual.h_scan, expected.harmonicity01);
            assert_eq!(actual.r_scan, expected.roughness01);
            assert_eq!(&*actual.score_scan, &expected.consonance_field_score);
            observed.lock().unwrap()[usize::from(enabled)] += 1;
        }
        let removed: Vec<_> = audio[0]
            .iter()
            .zip(&source.habitat)
            .map(|(mix, own)| mix - own)
            .collect();
        let mut expected = reference.process(&removed).clone();
        expected.recompute_consonance(&core.lparams);
        assert!(expected.harmonicity01.iter().any(|h| *h > 0.));
        previous = Some(expected);
    });
    let wiring = wire_runtime(
        &config,
        config.audio.sample_rate,
        "late self-exclusion".into(),
        scenario,
        Arc::new(AtomicBool::new(false)),
        WiringOptions {
            offline_body_probe: Some(probe),
            raw_t2_delay: Duration::ZERO,
            ui_tx: None,
            listener_forced: false,
            wait_user_exit: false,
            start_playing: true,
            audio_prod: None,
            wav_tx: None,
            reporter: None,
            deterministic_analysis: true,
            deterministic_footprints: true,
            guard_meter: None,
            underrun_frames: None,
            reserve_runtime_ids_through: 0,
            profile: None,
            audio_counters: None,
        },
    )
    .unwrap();
    join_thread("worker", wiring.worker_handle)
        .unwrap()
        .unwrap();
    join_thread("analysis", wiring.analysis_handle).unwrap();
    assert!(wiring.listener_analysis_handle.is_none());
    let counts = counts.lock().unwrap();
    assert!(
        counts[0] > 0 && counts[1] > 0,
        "both sides of the patch must be checked"
    );
}

#[test]
fn flora_acoustic_meter_ignores_internal_commands_for_identical_pcm() {
    for continuous in [false, true] {
        let fs = 48_000.0;
        let hop = 512;
        let timebase = Timebase { fs, hop };
        let cfg = WorkerConfig {
            loo_populations: Vec::new(),
            flora_metric: true,
            render_prototype: false,
            temporal_meter_t2: continuous,
            onset_comparison: None,
            scenario_name: "clock regression".into(),
            wait_user_exit: false,
            start_flag: Arc::new(AtomicBool::new(true)),
            exiting: Arc::new(AtomicBool::new(false)),
            scaffold: Default::default(),
            meter_shaping: Default::default(),
            guard_meter: None,
            underrun_frames: None,
            audio_counters: None,
            dcc_coupler: DccCoupler::new(Default::default()),
            hop,
            hop_duration: Duration::from_secs_f64(hop as f64 / fs as f64),
            fs,
            deterministic_analysis: true,
            deterministic_footprints: true,
        };
        let scenario = crate::scenario::Scenario {
            temporal_mode: crate::scenario::TemporalMode::Off,
            seed: 1,
            output_gain_db: 0.0,
            output_gain_position: None,
            control_update_mode: Default::default(),
            scaffold: Default::default(),
            meter_shaping: Default::default(),
            scene_markers: Vec::new(),
            events: vec![crate::scenario::TimedEvent {
                time: 605.0,
                order: 0,
                actions: vec![Action::Finish],
            }],
            duration_sec: 605.0,
        };
        let space = Log2Space::new(200.0, 4_000.0, 12);
        let state = WorkerState::new(
            Community::new(timebase),
            Conductor::from_scenario(scenario),
            Landscape::new(space.clone()),
            build_test_params(&space),
            DorsalStream::new(fs),
            None,
            &cfg,
        );

        let make_channels = |inbox| WorkerChannels {
            t2_inbox: inbox,
            temporal_snapshots: [None, None],
            ui_tx: None,
            audio_prod: None,
            wav_tx: None,
            audio_to_analysis_tx: None,
            analysis_result_rx: bounded(1).1,
            analysis_update_tx: bounded(1).0,
            presentation_to_listener_tx: None,
            listener_result_rx: None,
            listener_analysis_update_tx: None,
        };
        let scenario = crate::scenario::Scenario {
            temporal_mode: crate::scenario::TemporalMode::Off,
            seed: 1,
            output_gain_db: 0.0,
            output_gain_position: None,
            control_update_mode: Default::default(),
            scaffold: Default::default(),
            meter_shaping: Default::default(),
            scene_markers: Vec::new(),
            events: vec![crate::scenario::TimedEvent {
                time: 605.0,
                order: 0,
                actions: vec![Action::Finish],
            }],
            duration_sec: 605.0,
        };
        let mut control = WorkerState::new(
            Community::new(timebase),
            Conductor::from_scenario(scenario),
            Landscape::new(space.clone()),
            build_test_params(&space),
            DorsalStream::new(fs),
            None,
            &cfg,
        );
        let mut changed = state;
        let (output_a, inbox_a) = crate::temporal_cognition::t2::runtime_input::channel(48000, 512);
        let (output_b, inbox_b) = crate::temporal_cognition::t2::runtime_input::channel(48000, 512);
        let mut input_a = crate::temporal_cognition::t2::runtime_input::Input::new(
            48000,
            512,
            continuous.then_some(output_a),
        )
        .unwrap();
        let mut input_b = crate::temporal_cognition::t2::runtime_input::Input::new(
            48000,
            512,
            continuous.then_some(output_b),
        )
        .unwrap();
        let mut a = make_channels(continuous.then_some(inbox_a));
        let mut b = make_channels(continuous.then_some(inbox_b));
        let mut pcm = [0.; 512];
        let mut production_differs = false;
        let mut confirmed = false;
        for frame in 0..1000 {
            for (i, x) in pcm.iter_mut().enumerate() {
                let sample = frame * 512 + i as u64;
                let beat = sample % 24000;
                *x = if beat < 2400 {
                    0.15 * (-(beat as f32) / 700.).exp()
                        * (sample as f32 * std::f32::consts::TAU * 440. / 48000.).sin()
                } else {
                    0.
                };
            }
            if continuous {
                input_a.observe(frame, &pcm);
                input_b.observe(frame, &pcm);
            }
            control.frame_idx = frame;
            changed.frame_idx = frame;
            control.pop.set_test_habitat_onset_strength(0.);
            changed
                .pop
                .set_test_habitat_onset_strength(if frame % 17 == 0 { 4. } else { 0. });
            drive_production_meter(&mut control, &cfg, &mut a, &pcm);
            drive_production_meter(&mut changed, &cfg, &mut b, &pcm);
            assert_eq!(
                serde_json::to_value(control.flora_rhythm).unwrap(),
                serde_json::to_value(changed.flora_rhythm).unwrap()
            );
            assert_eq!(control.flora_rhythm.observed_end, Some((frame + 1) * 512));
            production_differs |= control.current_landscape.rhythm.delta.alpha.to_bits()
                != changed.current_landscape.rhythm.delta.alpha.to_bits();
            confirmed |= control.flora_rhythm.state.beat.confidence > 0.;
        }
        assert!(
            production_differs,
            "the command intervention must affect its real consumer"
        );
        assert!(
            confirmed,
            "the acoustic input must produce supported opportunities"
        );
    }
}
