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
    output_gain: f32,
    limiter_mode: crate::audio::limiter::LimiterMode,
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
        config.audio.sample_rate = sample_rate;
        config
            .validate_runtime()
            .map_err(|e| error(format!("{e:#}")))?;
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
        crate::life::modal::register_modal();
        let (scenario, warnings) = ScriptHost::compile_source(script, seed)?;
        validate_scenario(&scenario).map_err(error)?;
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
        let output_gain = config.output_gain(scenario.output_gain_db);
        if !output_gain.is_finite() || output_gain <= 0. {
            return Err(error(
                "combined output gain must be finite and positive".into(),
            ));
        }
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
            scenario,
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
            output_gain,
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
        process_hop(cfg, channels, state);
        channels.ui_tx = ui_output;
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
                    if self.listener_follows
                        && pcm
                            .iter()
                            .zip(habitat.iter())
                            .any(|(a, b)| a.to_bits() != b.to_bits())
                    {
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
                state.loo_storage = Some(packet.sources);
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
        self.output_gain
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
                assert!(landscape.rain_observations.is_empty());
                silent = Some((
                    landscape.harmonicity01.clone(),
                    landscape.roughness01.clone(),
                ));
            }
            if let (Some((h, r)), Some(observation)) =
                (&silent, landscape.rain_observations.first())
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
mod strict_room_readers {
    use super::*;

    const CONFIG: &str = "[room]\nmode='both'";

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
        let observation = &field.flora_observations[0];
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
