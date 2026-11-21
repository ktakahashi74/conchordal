//! Host-driven execution with the same one-hop analysis lag as offline rendering.
use super::*;
use crate::scripting::{ScriptError, ScriptWarning};
use wiring::{PreparedRuntime, WiringOptions, prepare_runtime};

#[derive(Debug, Clone)]
pub struct ScriptDiagnostics {
    pub message: String,
    pub line: Option<usize>,
    pub column: Option<usize>,
}

impl From<ScriptError> for ScriptDiagnostics {
    fn from(error: ScriptError) -> Self {
        Self {
            message: error.message,
            line: error.position.and_then(|p| p.line()),
            column: error.position.and_then(|p| p.position()),
        }
    }
}

impl std::fmt::Display for ScriptDiagnostics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}
impl std::error::Error for ScriptDiagnostics {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepStatus {
    Running,
    Finished,
    Error(String),
}

pub struct StepEngine {
    runtime: PreparedRuntime,
    #[cfg(feature = "native")]
    presentation: Receiver<Arc<[f32]>>,
    #[cfg(feature = "native")]
    ui: Receiver<UiFrame>,
    warnings: Vec<ScriptWarning>,
    /// While presentation PCM equals habitat PCM, listener analysis reuses the field's.
    #[cfg(not(feature = "native"))]
    listener_follows: bool,
    /// Wall-clock milliseconds of the last hop: worker, ecology analysis, listener analysis.
    #[cfg(not(feature = "native"))]
    stage_ms: [f64; 3],
    last_presentation: Arc<[f32]>,
    room: ResolvedRoom,
    limiter_mode: crate::audio::limiter::LimiterMode,
}

#[cfg(any(test, not(feature = "native")))]
fn listener_can_keep_following(follows: bool, presentation: &[f32], habitat: &[f32]) -> bool {
    follows
        && presentation.len() == habitat.len()
        && presentation
            .iter()
            .zip(habitat)
            .all(|(p, h)| p.to_bits() == h.to_bits())
}

impl StepEngine {
    pub fn new(
        script: &str,
        config_toml: &str,
        seed: Option<u64>,
        sample_rate: u32,
    ) -> Result<Self, ScriptDiagnostics> {
        let error = |message| ScriptDiagnostics {
            message,
            line: None,
            column: None,
        };
        let mut config = AppConfig::from_toml(config_toml).map_err(|e| error(format!("{e:#}")))?;
        crate::life::modal::register_modal();
        let (scenario, warnings) = ScriptHost::compile_source(script, seed)?;
        validate_scenario(&scenario).map_err(error)?;
        let room = resolve_effective_room(&mut config, &scenario, "step scene")?;
        #[cfg(not(test))]
        check_site_host(&scenario)?;
        config.audio.sample_rate = sample_rate;
        if config.temporal_meter_t2
            || config.temporal_t2.is_some()
            || config.temporal_onset_comparison.is_some()
            || config.temporal_body.is_some()
            || config.temporal_body_prototypes.is_some()
            || config.temporal_ridge.is_some()
            || config.temporal_acoustic.is_some()
            || config.temporal_period.is_some()
        {
            return Err(error(
                "step execution does not support background temporal observation".into(),
            ));
        }
        if scenario.temporal_mode == crate::scenario::TemporalMode::Observe
            || scenario.harmonic_relation_sources.is_some()
            || scenario.events.iter().flat_map(|e| &e.actions).any(|a| {
                let active = |w: f64| w.is_finite() && w != 0. && (-1. ..=1.).contains(&w);
                match a {
                    Action::FollowHarmonicRelations { weight }
                    | Action::FollowHarmonicContinuation { weight } => active(*weight),
                    Action::UpdateLandscape { update } => {
                        update.harmonic_relations_weight.is_some_and(active)
                            || update.harmonic_continuation_weight.is_some_and(active)
                    }
                    _ => false,
                }
            })
        {
            return Err(error(
                "step execution does not support background temporal observation".into(),
            ));
        }
        let limiter_mode = wiring::resolve_limiter_mode(&config);
        #[cfg(feature = "native")]
        let (output, presentation) = bounded(1);
        #[cfg(not(feature = "native"))]
        let output = crate::handoff::Slot::new(1);
        #[cfg(feature = "native")]
        let (ui_tx, ui) = bounded(1);
        #[cfg(not(feature = "native"))]
        let ui_tx = crate::handoff::Slot::new(1);
        let runtime = prepare_runtime(
            &config,
            sample_rate,
            "step scene".into(),
            wiring::prepare_score(&config, sample_rate, scenario, false).map_err(error)?,
            Arc::new(AtomicBool::new(false)),
            WiringOptions {
                #[cfg(feature = "native")]
                score: None,
                end_request: Arc::new(AtomicBool::new(false)),
                duration_sec: None,
                #[cfg(test)]
                offline_body_probe: None,
                #[cfg(test)]
                raw_t2_delay: Duration::ZERO,
                ui_tx: Some(ui_tx),
                listener_forced: true,
                wait_user_exit: false,
                start_playing: true,
                #[cfg(feature = "native")]
                audio_prod: None,
                wav_tx: Some(output),
                reporter: None,
                deterministic_analysis: true,
                deterministic_footprints: true,
                guard_meter: None,
                underrun_frames: None,
                reserve_runtime_ids_through: 0,
                profile: None,
                telemetry: false,
                audio_counters: None,
            },
            true,
        )
        .map_err(error)?;
        let runtime = PreparedRuntime {
            state: (runtime.state)(&runtime.cfg),
            cfg: runtime.cfg,
            channels: runtime.channels,
            analysis_handle: runtime.analysis_handle,
            listener_analysis_handle: runtime.listener_analysis_handle,
        };
        #[cfg(not(feature = "native"))]
        let listener_follows = matches!(
            &runtime.listener_analysis_handle,
            Some(wiring::AnalysisExecution::Inline(listener)) if listener.can_follow()
        );
        Ok(Self {
            runtime,
            #[cfg(feature = "native")]
            presentation,
            #[cfg(feature = "native")]
            ui,
            warnings,
            #[cfg(not(feature = "native"))]
            listener_follows,
            #[cfg(not(feature = "native"))]
            stage_ms: [0.; 3],
            last_presentation: Arc::from([]),
            room,
            limiter_mode,
        })
    }

    /// Output conditioning is applied by the host, after listener analysis.
    pub fn limiter_mode(&self) -> crate::audio::limiter::LimiterMode {
        self.limiter_mode
    }

    pub fn hop_len(&self) -> usize {
        self.runtime.cfg.hop
    }

    pub fn output_channels(&self) -> usize {
        1 + usize::from(self.runtime.state.schedule_renderer.room.is_some())
    }

    /// Raw interleaved presentation PCM, before the host output gain and limiter.
    /// A Finished result may contain the final complete hop. Subsequent calls return silence.
    pub fn step(&mut self, presentation: &mut [f32]) -> StepStatus {
        let samples = self.hop_len() * self.output_channels();
        if presentation.len() != samples {
            presentation.fill(0.);
            return StepStatus::Error(format!("expected {samples} samples"));
        }
        presentation.fill(0.);
        if !self.runtime.state.failures.is_empty() {
            return StepStatus::Error(self.runtime.state.failures.join("; "));
        }
        if self.finished() {
            return StepStatus::Finished;
        }
        let PreparedRuntime {
            cfg,
            channels,
            state,
            analysis_handle,
            listener_analysis_handle,
        } = &mut self.runtime;
        if state.playback_state == PlaybackState::NotStarted {
            state.playback_state = PlaybackState::Playing;
        }
        if cfg.end_request.load(Ordering::SeqCst) && !state.finish_begun {
            state.request_end(state.timebase.frame_start_tick(state.frame_idx));
        }
        #[cfg(not(feature = "native"))]
        let worker_began = crate::clock::Instant::now();
        let ui_output = channels.ui_tx.take();
        let result = process_hop(cfg, channels, state);
        channels.ui_tx = ui_output;
        if let Err(err) = result {
            state.failures.push(err.clone());
            return StepStatus::Error(err);
        }
        buffer_listener_rhythm_hop(state, cfg);
        if !state.failures.is_empty() {
            return StepStatus::Error(state.failures.join("; "));
        }
        #[cfg(feature = "native")]
        let output = self.presentation.try_recv();
        #[cfg(not(feature = "native"))]
        let output = channels.wav_tx.as_ref().expect("inline output").try_recv();
        if let Ok(pcm) = output {
            presentation.copy_from_slice(&pcm);
            self.last_presentation = pcm;
        }
        #[cfg(feature = "native")]
        if !analysis_handle.step() || !listener_analysis_handle.as_mut().is_some_and(|a| a.step()) {
            state
                .failures
                .push("inline analysis failed to consume its hop".into());
        }
        #[cfg(not(feature = "native"))]
        {
            let ms = |began: crate::clock::Instant| began.elapsed().as_secs_f64() * 1000.;
            self.stage_ms[0] = ms(worker_began);
            let ecology_began = crate::clock::Instant::now();
            let wiring::AnalysisExecution::Inline(field) = analysis_handle;
            for update in channels.analysis_update_tx.try_iter() {
                field.update(update);
            }
            let (id, habitat, packet) = channels
                .audio_to_analysis_tx
                .as_ref()
                .expect("inline habitat")
                .try_recv()
                .expect("worker produced a habitat hop");
            assert!(!field.begin_hop(id), "inline field must be contiguous");
            // The listener's hop is taken first: a split adopts the field's pre-hop state.
            let listener_began = crate::clock::Instant::now();
            let mut listener = match listener_analysis_handle {
                Some(wiring::AnalysisExecution::Inline(listener)) => {
                    for update in channels
                        .listener_analysis_update_tx
                        .as_ref()
                        .expect("inline listener updates")
                        .try_iter()
                    {
                        listener.update(update);
                    }
                    let (listener_id, pcm, listener_packet) = channels
                        .presentation_to_listener_tx
                        .as_ref()
                        .expect("inline listener input")
                        .try_recv()
                        .expect("worker produced presentation");
                    assert!(
                        !listener.begin_hop(listener_id),
                        "inline listener must be contiguous"
                    );
                    if self.listener_follows && !listener_can_keep_following(true, &pcm, &habitat) {
                        listener.split_from(field);
                        self.listener_follows = false;
                    }
                    Some((listener, listener_id, pcm, listener_packet))
                }
                None => None,
            };
            let listener_ms = ms(listener_began);
            if field.observe(id, &habitat, packet.as_ref()) {
                channels
                    .analysis_result_rx
                    .send((id, Some(field.snapshot())))
                    .expect("field consumed once per hop");
            }
            let listener_began = crate::clock::Instant::now();
            self.stage_ms[1] = ms(ecology_began) - listener_ms;
            if let Some((listener, id, pcm, packet)) = listener.as_mut() {
                let observed = if self.listener_follows {
                    listener.follow(*id, pcm, field)
                } else {
                    listener.observe(*id, pcm, packet.as_ref())
                };
                if observed {
                    channels
                        .listener_result_rx
                        .as_ref()
                        .expect("inline listener result")
                        .send((*id, Some(listener.snapshot())))
                        .expect("listener consumed once per hop");
                }
            }
            self.stage_ms[2] = listener_ms + ms(listener_began);
            if let Some(packet) = packet {
                state.loo_storage = Some((packet.sources, packet.sites));
            }
        }
        if !state.failures.is_empty() {
            StepStatus::Error(state.failures.join("; "))
        } else if state.finished {
            StepStatus::Finished
        } else {
            StepStatus::Running
        }
    }

    /// Build a snapshot when the host requests a GUI update.
    pub fn ui_frame(&mut self) -> Option<UiFrame> {
        if !self.runtime.state.failures.is_empty() {
            return None;
        }
        let left = if self.output_channels() == 2 {
            self.last_presentation
                .iter()
                .step_by(2)
                .copied()
                .collect::<Arc<[f32]>>()
        } else {
            self.last_presentation.clone()
        };
        let runtime = &mut self.runtime;
        #[cfg(feature = "native")]
        for _ in self.ui.try_iter() {}
        #[cfg(not(feature = "native"))]
        for _ in runtime.channels.ui_tx.as_ref()?.try_iter() {}
        let peak = left.iter().fold(0.0f32, |p, x| p.max(x.abs()));
        let now = runtime
            .state
            .timebase
            .frame_start_tick(runtime.state.frame_idx);
        send_runtime_ui_frame(
            &mut runtime.state,
            &runtime.channels,
            &runtime.cfg,
            left,
            peak,
            [peak; 2],
            now,
        );
        #[cfg(feature = "native")]
        {
            self.ui.try_recv().ok()
        }
        #[cfg(not(feature = "native"))]
        {
            runtime.channels.ui_tx.as_ref()?.try_recv().ok()
        }
    }

    pub fn request_end(&mut self) {
        self.runtime.cfg.end_request.store(true, Ordering::SeqCst);
    }
    pub fn finished(&self) -> bool {
        self.runtime.state.finished
    }
    /// Wall-clock milliseconds of the last hop's worker, ecology and listener stages,
    /// for the browser host's performance counters.
    #[cfg(not(feature = "native"))]
    pub fn stage_ms(&self) -> [f64; 3] {
        self.stage_ms
    }
    pub fn warnings(&self) -> &[ScriptWarning] {
        &self.warnings
    }
    pub fn output_gain(&self) -> f32 {
        self.room.output_gain
    }

    pub fn room_diagnostic(&self) -> &str {
        &self.room.diagnostic
    }
}

#[cfg(all(test, feature = "native"))]
mod tests {
    use super::*;

    fn compare_offline(script: &str, end_hop: usize, config_toml: &str) {
        let config = AppConfig::from_toml(config_toml).unwrap();
        let (scenario, _) = ScriptHost::compile_source(script, Some(21)).unwrap();
        let (tx, rx) = bounded(16);
        let wiring = wiring::wire_runtime(
            &config,
            48_000,
            "reference".into(),
            scenario,
            Arc::new(AtomicBool::new(false)),
            WiringOptions {
                score: None,
                end_request: Arc::new(AtomicBool::new(false)),
                duration_sec: Some(end_hop as f64 * config.analysis.hop_size as f64 / 48_000.),
                offline_body_probe: None,
                raw_t2_delay: Duration::ZERO,
                ui_tx: None,
                listener_forced: true,
                wait_user_exit: false,
                start_playing: true,
                audio_prod: None,
                wav_tx: Some(tx),
                reporter: None,
                deterministic_analysis: true,
                deterministic_footprints: true,
                guard_meter: None,
                underrun_frames: None,
                reserve_runtime_ids_through: 0,
                profile: None,
                telemetry: false,
                audio_counters: None,
            },
        )
        .unwrap();
        let mut engine = StepEngine::new(script, config_toml, Some(21), 48_000).unwrap();
        let mut actual = vec![0.; engine.hop_len() * engine.output_channels()];
        let mut count = 0;
        let mut audible = false;
        while let Ok(expected) = rx.recv() {
            if count == end_hop || count == end_hop + 1 {
                engine.request_end();
            }
            let status = engine.step(&mut actual);
            assert!(!matches!(status, StepStatus::Error(_)), "{status:?}");
            if count >= end_hop && status == StepStatus::Running {
                assert_eq!(engine.runtime.state.playback_state, PlaybackState::Ending);
            }
            let mismatch = expected
                .iter()
                .zip(&actual)
                .position(|(a, b)| a.to_bits() != b.to_bits());
            assert!(
                mismatch.is_none(),
                "PCM diverges at hop {count}, sample {mismatch:?}"
            );
            audible |= actual.iter().any(|x| *x != 0.);
            if count % 37 == 0 {
                let frame = engine.ui_frame().unwrap();
                assert!(
                    frame
                        .wave
                        .samples
                        .iter()
                        .eq(actual.iter().step_by(engine.output_channels()))
                );
            }
            count += 1;
        }
        wiring::join_runtime(
            Some(wiring.worker_handle),
            [
                ("analysis", Some(wiring.analysis_handle)),
                ("listener", wiring.listener_analysis_handle),
            ],
        )
        .unwrap();
        assert!(audible);
        assert!(count >= end_hop);
        assert!(engine.finished());
        assert_eq!(engine.step(&mut actual), StepStatus::Finished);
        assert!(actual.iter().all(|x| *x == 0.));
    }

    #[test]
    fn demo_presentation_matches_ordered_offline_and_graceful_end() {
        compare_offline(include_str!("../../tests/fixtures/step_demo.rhai"), 384, "");
    }

    #[test]
    fn voice_and_flora_self_exclusion_match_offline() {
        compare_offline(
            "place(sine().sustain().leave_self_out(true).amp(0.05), at(220)); place(flora().amp(0.05).fade(0.1), random(220,880)); wait(4);",
            64,
            "",
        );
    }

    #[test]
    fn rain_without_its_drops_matches_offline() {
        compare_offline(
            "place(rain().amp(0.05).rate(80).hit(0.3).ring(2).fade(0.1).leave_drops_out(true), consonance(200, 2000)); wait(4);",
            64,
            "",
        );
    }

    #[test]
    fn wet_requested_end_matches_native_duration_through_the_final_pcm() {
        compare_offline(
            "place(sine().anchor().sustain().adsr(0,0,1,0.08), at(220)); wait(4);",
            12,
            "[room]\nmode='both'",
        );
    }

    #[test]
    fn drops_only_rain_observes_the_silent_field() {
        let mut engine = StepEngine::new(
            "wait(0.5); place(rain().objects(0).rate(100).leave_drops_out(true), consonance(200, 2000)); wait(2);",
            "",
            Some(21),
            48_000,
        )
        .unwrap();
        let mut output = vec![0.; engine.hop_len()];
        let mut silent: Option<(Vec<f32>, Vec<f32>)> = None;
        let (mut checked, mut heard) = (0, false);
        for hop in 0..200 {
            assert!(!matches!(engine.step(&mut output), StepStatus::Error(_)));
            let landscape = &engine.runtime.state.current_landscape;
            // Placement is at hop 47; hop 45 still holds the field of silence.
            if hop == 45 {
                assert!(landscape.ecology.rain_observations.is_empty());
                silent = Some((
                    landscape.harmonicity01.clone(),
                    landscape.roughness01.clone(),
                ));
            }
            if let (Some((h, r)), Some(observation)) =
                (&silent, landscape.ecology.rain_observations.first())
            {
                assert_eq!(
                    (&observation.h_scan, &observation.r_scan),
                    (h, r),
                    "hop {hop}"
                );
                heard |= landscape.harmonicity01 != *h || landscape.roughness01 != *r;
                checked += 1;
            }
        }
        assert!(checked > 100 && heard, "{checked} {heard}");
    }

    #[test]
    fn diagnostics_are_positioned_and_endless_is_rejected_before_loop() {
        let error = StepEngine::new("\nunknown_word();", "", Some(21), 48_000)
            .err()
            .unwrap();
        assert_eq!(error.line, Some(2));
        let error = StepEngine::new("endless(); loop { wait(1); }", "", Some(21), 48_000)
            .err()
            .unwrap();
        assert!(error.message.contains("endless"));
        assert!(error.line.is_some());
        assert!(StepEngine::new("wait(1);", "", Some(21), 8_000).is_err());
    }

    #[test]
    fn invalid_buffer_does_not_advance_engine() {
        let mut engine = StepEngine::new(
            "place(sine().sustain(),at(220)); wait(1);",
            "",
            Some(21),
            48_000,
        )
        .unwrap();
        let mut output = [1.; 3];
        assert!(matches!(engine.step(&mut output), StepStatus::Error(_)));
        assert_eq!(engine.runtime.state.frame_idx, 0);
        assert_eq!(output, [0.; 3]);
    }
}

#[cfg(all(test, feature = "native"))]
#[path = "site_acoustic_tests.rs"]
mod site_acoustic_tests;

#[cfg(all(test, feature = "native"))]
mod strict_room_readers {
    use super::*;

    const CONFIG: &str = "[room]\nmode='both'";

    #[test]
    fn sites_validate_coordinates_and_distances_before_preparation() {
        use crate::scenario::Site;
        let (mut scene, _) = ScriptHost::compile_source("wait(0.01);", Some(1)).unwrap();
        let mut cfg = AppConfig::default();
        for (x, y, expected) in [
            (f64::NAN, 0., "coordinates must be finite"),
            (0., f64::INFINITY, "coordinates must be finite"),
            (f64::MAX, f64::MAX, "distance is not representable"),
            (-f64::MAX, 0., "distance is not representable"),
        ] {
            scene.sites = vec![
                Site {
                    x: f64::MAX,
                    y: 0.,
                    position: rhai::Position::new(2, 3),
                },
                Site {
                    x,
                    y,
                    position: rhai::Position::new(5, 7),
                },
            ];
            let err = resolve_effective_room(&mut cfg, &scene, "site test").unwrap_err();
            assert!(err.message.contains(expected), "{err}");
            let pos = (5, 7);
            assert_eq!((err.line.unwrap(), err.column.unwrap()), pos);
        }
    }

    #[test]
    fn late_site_on_a_mixed_voice_starts_its_field_at_score_start() {
        let script = "place(sine().brain(\"seq\").sustain().amp(0.04),at(375));
            wait(0.45);place(sine().sustain().amp(0).leave_self_out(false),at(500).site(0,36));wait(1);";
        let mut step = StepEngine::new(script, "", Some(21), 48000).unwrap();
        let (kernel, params) = crate::runtime::body_metabolism_test_core();
        let mut reference =
            crate::core::stream::analysis::AnalysisStream::new(params.clone(), kernel);
        let mut differs = false;
        for n in 0..95 {
            assert_eq!(step.step(&mut [0.; 512]), StepStatus::Running);
            let state = &step.runtime.state;
            let actual = state.current_landscape.at_site(1);
            if n > 0 {
                let mut expected = reference.last().clone();
                expected.recompute_consonance(&params);
                assert_eq!(actual.subjective_intensity, expected.subjective_intensity);
                assert_eq!(
                    actual.consonance_density_mass,
                    expected.consonance_density_mass
                );
                differs |=
                    actual.subjective_intensity != state.current_landscape.subjective_intensity;
            }
            let sites = state.schedule_renderer.sites.as_ref().unwrap();
            let local: [f32; 512] = std::array::from_fn(|i| sites.dry_habitat(0)[i] * (1. / 24.));
            reference.process(&local);
        }
        assert!(differs);
    }

    #[test]
    fn site_late_flora_hears_fixed_line_at_its_distance() {
        fn scene(x: i32) -> (Vec<f32>, f64, f32, f64) {
            let script = format!(
                r#"let v=sine().brain("seq").anchor().sustain().amp(0.03).adsr(0.02,0,1,0.01);
                let first=place(v,at(500).site({x},0)); wait(0.21);
                place(flora().amp(0).fade(0).respond_to_onsets(2),random(100,2200).site(12,0));
                wait(0.4);release(first);wait(1.5);
                for n in 0..8 {{
                    let hz=if n%2==0 {{330}} else {{800}};
                    let a=place(v,at(hz).site({x},0)); wait(0.24);release(a);wait(0.46);
                }} wait(0.4);"#
            );
            let mut engine = StepEngine::new(&script, "", Some(21), 48000).unwrap();
            let mut hop = vec![0.; engine.hop_len()];
            let mut audience = Vec::new();
            let mut onset = 0.;
            let mut power = 0.;
            let mut rise = 0.;
            let mut fixed_moss = None;
            for n in 0..729 {
                assert_eq!(engine.step(&mut hop), StepStatus::Running);
                audience.extend_from_slice(&hop);
                let state = &engine.runtime.state;
                if let Some(flora) = state.pop.flora.first() {
                    let local = state.current_landscape.at_site(flora.site_id());
                    let probe =
                        fixed_moss.get_or_insert_with(|| flora.fixed_lane_response_probe(1500.));
                    probe.observe_onsets(local, n * 512);
                    if n == 728 {
                        rise=serde_json::to_value(probe.report(0.,local)).unwrap()["respond"]["rise_sum"].as_f64().unwrap();
                    }
                    assert!(local.ecology.site_end.is_some_and(|end| end <= n * 512));
                    if let Some(owner) = local.ecology.flora_observations.first() {
                        if let Some(scan) = &owner.power_scan {
                            power += scan.iter().sum::<f32>();
                        }
                        let obs = owner.perception.as_ref().unwrap();
                        let scan = obs.onset_habituation_state_scan.as_ref().unwrap();
                        assert_eq!(
                            scan.as_ref(),
                            local.onset_habituation_state_scan.as_ref(),
                            "silent moss leaves the same local field"
                        );
                        onset += scan.iter().map(|x| f64::from(*x)).sum::<f64>();
                    }
                }
            }
            (audience, onset, power, rise)
        }
        let near = scene(12);
        let far = scene(-12);
        assert_eq!(near.0, far.0);
        assert!(
            near.2 > 100. * far.2 && far.2 > 0.,
            "resource powers {} {}",
            near.2,
            far.2
        );
        assert_ne!(near.1, far.1, "local attack evidence");
        assert_ne!(near.3, far.3, "moss response to external attacks");
    }

    #[test]
    fn site_solo_flora_rests_on_its_own_attacks_and_does_not_answer_them() {
        use crate::life::span_gap::Phase;
        let mut script = String::from(
            "room(\"both\"); let f = place(flora().amp(0.12).pace(8).fade(0.03).pitch_apply_mode(\"rearticulate\").rest_on_habituation(64).respond_to_onsets(16),random(220,880).site(0,1));",
        );
        for _ in 0..20 {
            script.push_str("wait(0.25); f.amp(0); wait(0.25); f.amp(0.12);");
        }
        script.push_str("wait(1);");
        let mut step = StepEngine::new(&script, "", Some(21), 48000).unwrap();
        let mut pcm = vec![0.; step.hop_len() * 2];
        let (mut peak, mut heard, mut integral) = (0f32, 0f32, 0f64);
        let mut phases = Vec::new();
        for _ in 0..930 {
            assert_eq!(step.step(&mut pcm), StepStatus::Running);
            peak = pcm.iter().fold(peak, |a, x| a.max(x.abs()));
            let state = &step.runtime.state;
            let f = &state.pop.flora[0];
            let local = state.current_landscape.at_site(f.site_id());
            if let Some(o) = local.ecology.flora_observations.first() {
                let p = o.perception.as_ref().unwrap();
                assert!(
                    p.onset_habituation_state_scan
                        .as_ref()
                        .unwrap()
                        .iter()
                        .all(|&x| x == 0.)
                );
                assert!(
                    o.power_scan
                        .as_ref()
                        .is_none_or(|scan| scan.iter().all(|&x| x == 0.))
                );
            }
            let gap = f.span_gap.as_ref().unwrap();
            heard = heard.max(gap.heard());
            integral = integral.max(gap.exit_integral);
            if phases.last() != Some(&gap.phase) {
                phases.push(gap.phase);
            }
            if phases.contains(&Phase::GapActive) {
                let report = serde_json::to_value(f.report(0., local)).unwrap();
                assert_eq!(report["respond"]["rise_sum"], 0.);
                assert_eq!(report["respond"]["extra_sec"], 0.);
                break;
            }
        }
        assert!(
            peak > 0. && heard > 0. && integral > 0. && phases.contains(&Phase::GapActive),
            "{peak} {heard} {integral} {phases:?}"
        );
    }

    #[test]
    fn site_off_different_sends_split_the_following_listener() {
        let mut step = StepEngine::new(
            "place(sine().sustain().send(presentation_bus),at(375).site(0,1)); wait(1);",
            "",
            Some(21),
            48000,
        )
        .unwrap();
        let mut out = vec![0.; step.hop_len()];
        let mut follows = true;
        for _ in 0..12 {
            assert_eq!(step.step(&mut out), StepStatus::Running);
            follows = listener_can_keep_following(
                follows,
                &out,
                step.runtime.state.schedule_renderer.habitat_output(),
            );
        }
        assert!(!follows);
        assert!(!listener_can_keep_following(follows, &[0.], &[0.]));
        assert!(!listener_can_keep_following(true, &[-0.], &[0.]));
    }

    #[test]
    fn site_last_reader_releases_histories_after_a_later_placement() {
        let script = "let a=place(flora().fade(0.01).respond_to_onsets(2),random(220,880).site(0,0));wait(0.25);release(a);
            wait(0.6);let b=place(rain().fade(0.01).objects(0),consonance(200,900).site(0,0)); wait(0.2);release(b);wait(2);";
        let mut step = StepEngine::new(script, "", Some(2), 48000).unwrap();
        let mut out = vec![0.; step.hop_len()];
        for n in 0..170 {
            assert_eq!(step.step(&mut out), StepStatus::Running);
            let state = &step.runtime.state;
            if n == 10 {
                assert!(
                    !state
                        .current_landscape
                        .at_site(1)
                        .onset_habituation_state_scan
                        .is_empty()
                );
            }
            if n == 68 {
                assert!(state.pop.flora.is_empty() && state.pop.rain.is_empty());
                assert_eq!(state.current_landscape.ecology.sites.len(), 1);
                let local = state.current_landscape.at_site(1);
                assert!(
                    local.onset_habituation_state_scan.is_empty()
                        && local.ecology.site_resource_power.is_none()
                );
            }
            if n == 169 {
                assert!(state.current_landscape.ecology.sites.is_empty());
            }
        }
    }

    #[test]
    fn site_harmonic_conflicts_are_positioned_for_both_orders() {
        for control in ["follow_harmonic_relations", "follow_harmonic_continuation"] {
            for first in [true, false] {
                let site = "place(sine(), at(220).site(1,2));";
                let call = format!("{control}(-0.5);");
                let script = if first {
                    format!("{site}\n{call} wait(1);")
                } else {
                    format!("{call}\n{site} wait(1);")
                };
                let err = StepEngine::new(&script, "", Some(1), 48000).err().unwrap();
                assert!(
                    err.line.is_some() && err.column.is_some() && err.message.contains(control),
                    "{err:?}"
                );
            }
        }
    }

    #[test]
    fn site_gain_includes_explicit_base_and_score_gain_in_every_mode() {
        let (mut scene, _) =
            ScriptHost::compile_source("output_gain(2); wait(0.1);", Some(21)).unwrap();
        crate::scenario::Site::register(&mut scene.sites, Some(crate::scenario::Site::DEFAULT));
        for (mode, default) in [("off", 4.0), ("presentation", -5.0), ("both", -5.0)] {
            for (audio, base) in [("", default), ("[audio]\noutput_gain_db=-3", -3.)] {
                let mut config =
                    AppConfig::from_toml(&format!("[room]\nmode='{mode}'\n{audio}")).unwrap();
                let resolved = resolve_effective_room(&mut config, &scene, "site").unwrap();
                assert_eq!(resolved.output_gain, 10f32.powf((base + 2.) / 20.));
            }
        }
    }

    #[test]
    fn one_missing_slot_stops_step_before_the_next_publication() {
        let script = "place(sine().sustain().leave_self_out(true),at(330)); wait(0.021); place(sine().sustain().leave_self_out(true),at(440)); wait(1);";
        wiring::ROOM_OWNER_OVERRIDE.set(Some(1));
        let step = StepEngine::new(script, CONFIG, Some(21), 48000);
        wiring::ROOM_OWNER_OVERRIDE.set(None);
        let mut step = step.unwrap();
        let mut pcm = vec![0.; step.hop_len() * 2];
        for _ in 0..2 {
            assert_eq!(step.step(&mut pcm), StepStatus::Running);
        }
        let last = step.last_presentation.clone();
        for _ in 0..2 {
            pcm.fill(1.);
            let StepStatus::Error(e) = step.step(&mut pcm) else {
                panic!("missing owner slot")
            };
            assert!(
                e.contains("tick 1024: need 2 owner slots, reserved 1"),
                "{e}"
            );
            assert!(pcm.iter().all(|s| *s == 0.));
            assert_eq!(step.runtime.state.frame_idx, 2);
            assert!(Arc::ptr_eq(&last, &step.last_presentation));
            assert!(step.presentation.is_empty());
            assert!(step.ui_frame().is_none());
        }
    }

    #[test]
    fn consecutive_respawns_include_both_sides_of_cleanup_in_the_room_bound() {
        let script = r#"place(sine().brain("drone").sustain().anchor().leave_self_out(true)
            .respawn_random().respawn_capacity(1).respawn_background_death_rate(100000.0),at(330));
            wait(1);"#;
        let mut step = StepEngine::new(script, CONFIG, Some(21), 48000).unwrap();
        let mut pcm = vec![0.; step.hop_len() * step.output_channels()];
        let mut seen = std::collections::BTreeSet::new();
        for _ in 0..15 {
            assert_eq!(step.step(&mut pcm), StepStatus::Running);
            seen.extend(step.runtime.state.pop.voices.iter().map(|v| v.id()));
        }
        let counts = step
            .runtime
            .state
            .schedule_renderer
            .room
            .as_ref()
            .unwrap()
            .counts();
        assert!(seen.len() > 10);
        assert_eq!((counts.reserved, counts.peak), (6, 6));
    }

    #[test]
    fn flora_resource_and_growth_receive_external_power_and_fields() {
        let mut step = StepEngine::new(
            "place(flora().tone(\"sine\").amp(0.2).leave_self_out(true),random(220,880)); wait(4);",
            CONFIG,
            Some(21),
            48000,
        )
        .unwrap();
        let mut pcm = vec![0.; step.hop_len() * step.output_channels()];
        let mut heard = false;
        for _ in 0..150 {
            assert_eq!(step.step(&mut pcm), StepStatus::Running);
            heard |= pcm.iter().any(|&x| x != 0.);
        }
        let state = &mut step.runtime.state;
        let field = &state.current_landscape;
        assert!(heard && field.harmonicity.iter().any(|&x| x > 0.));
        let observation = &field.ecology.flora_observations[0];
        assert!(
            observation
                .power_scan
                .as_ref()
                .unwrap()
                .iter()
                .all(|&x| x == 0.)
        );
        assert!(
            observation
                .perception
                .as_ref()
                .unwrap()
                .h_scan
                .iter()
                .all(|&x| x == 0.)
        );
        let report = serde_json::to_value(state.pop.flora[0].report(1.6, field)).unwrap();
        assert_eq!(report["external_current"], true);
        for name in ["external_consumption_scan", "harmonicity_state_scan"] {
            assert!(
                report[name]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|x| x.as_f64() == Some(0.))
            );
        }
        assert_eq!(
            report["growth_level_scan"],
            serde_json::to_value(&observation.level_scan).unwrap()
        );
        state.pop.flora[0].observe_external(field, None);
        let missing = serde_json::to_value(state.pop.flora[0].report(1.6, field)).unwrap();
        assert_eq!(missing["external_current"], false);
        assert_eq!(
            missing["external_observed_frame"],
            report["external_observed_frame"]
        );
    }
}
