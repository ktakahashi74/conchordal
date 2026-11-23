use super::temporal_ports::{self, TemporalChannels, TemporalConfig, TemporalObservation};
use super::*;

/// Immutable per-run settings and shared control flags for the worker thread.
pub(super) struct WorkerConfig {
    pub(super) felt: crate::core::felt_config::FeltConfig,
    pub(super) harmonic_relations: Option<(Arc<crate::core::harmonic_support::Geometry>, usize)>,
    /// Reserve the habitat audio-only meter (Flora, harmonic rhythm, Rain follow_meter).
    pub(super) habitat_meter: bool,
    pub(super) render_prototype: crate::life::sound::render_prototype::RenderPrototype,
    pub(super) temporal_config: TemporalConfig,
    #[cfg(feature = "research")]
    pub(super) onset_comparison: Option<crate::config::TemporalOnsetComparisonConfig>,
    pub(super) scenario_name: String,
    pub(super) wait_user_exit: bool,
    pub(super) start_flag: Arc<AtomicBool>,
    pub(super) exiting: Arc<AtomicBool>,
    pub(super) end_request: Arc<AtomicBool>,
    pub(super) duration_tick: Option<Tick>,
    pub(super) scaffold: ScaffoldConfig,
    pub(super) harmonic_rhythm: u64,
    pub(super) meter_shaping: crate::core::meter::MeterShaping,
    pub(super) guard_meter: Option<Arc<LimiterMeter>>,
    pub(super) underrun_frames: Option<Arc<AtomicU64>>,
    pub(super) telemetry: bool,
    pub(super) audio_counters: Option<Arc<crate::audio::telemetry::AudioCallbackCounters>>,
    pub(super) dcc_coupler: DccCoupler,
    pub(super) hop: usize,
    pub(super) hop_duration: Duration,
    pub(super) fs: f32,
    /// Offline/report paths consume analysis with a fixed lag so results do not
    /// depend on worker scheduling. Real-time leaves the listener best-effort.
    pub(super) deterministic_analysis: bool,
    #[cfg(feature = "research")]
    pub(super) deterministic_footprints: bool,
}

/// Channel endpoints and the audio ring-buffer producer owned by the worker.
pub(super) struct WorkerChannels {
    pub(super) observation: TemporalChannels,
    pub(super) ui_tx: Option<Sender<UiFrame>>,
    #[cfg(feature = "native")]
    pub(super) audio_prod: Option<ringbuf::HeapProd<f32>>,
    pub(super) wav_tx: Option<Sender<Arc<[f32]>>>,
    pub(super) audio_to_analysis_tx: Option<Sender<analysis_worker::AnalysisHop>>,
    pub(super) analysis_result_rx: Receiver<analysis_worker::AnalysisResult>,
    pub(super) analysis_update_tx: Sender<LandscapeUpdate>,
    pub(super) presentation_to_listener_tx: Option<Sender<analysis_worker::AnalysisHop>>,
    pub(super) listener_result_rx: Option<Receiver<analysis_worker::AnalysisResult>>,
    pub(super) listener_analysis_update_tx: Option<Sender<LandscapeUpdate>>,
}

/// Mutable worker state. Built once before the loop; hop phases take it `&mut`.
pub(super) struct WorkerState {
    pub(super) lifecycle: PlaybackLifecycle,
    pub(super) ui: UiDelivery,
    pub(super) listener: ListenerSync,
    pub(super) temporal: TemporalState,
    pub(super) loo_populations: Vec<u64>,
    #[cfg(feature = "native")]
    pub(super) loo_recycle: (
        Sender<crate::core::stream::loo::CaptureStorage>,
        Receiver<crate::core::stream::loo::CaptureStorage>,
    ),
    #[cfg(not(feature = "native"))]
    pub(super) loo_storage: Option<crate::core::stream::loo::CaptureStorage>,
    #[cfg(test)]
    pub(super) offline_body_probe: Option<OfflineBodyProbe>,
    pub(super) pop: Community,
    pub(super) conductor: Conductor,
    pub(super) current_landscape: Landscape,
    pub(super) lparams: LandscapeParams,
    pub(super) dorsal: DorsalStream,
    pub(super) reporter: Option<JsonlReporter>,
    pub(super) profile: Option<RunProfile>,
    pub(super) timebase: crate::core::timebase::Timebase,
    pub(super) log_space: Log2Space,
    pub(super) frame_idx: u64,
    pub(super) generator_model: crate::life::generator_model::GeneratorModel,
    pub(super) schedule_renderer: ScheduleRenderer,
    pub(super) phonation_batches_buf: Vec<PhonationBatch>,
}

pub(super) struct PlaybackLifecycle {
    pub(super) playback_state: PlaybackState,
    pub(super) finish_begun: bool,
    pub(super) end_sequence: Option<EndSequence>,
    pub(super) finish_logged: bool,
    pub(super) finished: bool,
    pub(super) finalized: bool,
    pub(super) scenario_end_tick: Option<Tick>,
    pub(super) script_error_reported: bool,
    pub(super) failures: Vec<String>,
}

pub(super) struct UiDelivery {
    pub(super) pending_final_ui_frame: Option<UiFrame>,
    pub(super) ui_rhythm_hops: std::collections::VecDeque<crate::viewdata::ListenerRhythmHop>,
    pub(super) last_ui_update: Instant,
    pub(super) last_guard_log: Instant,
    pub(super) last_tick_log: Instant,
    pub(super) monitor: AudioMonitor,
}

pub(super) struct ListenerSync {
    pub(super) last_analysis_frame: Option<u64>,
    pub(super) last_listener_analysis_frame: Option<u64>,
    pub(super) listener_min_valid_frame: u64,
    pub(super) listener_twin: ListenerTwin,
    pub(super) latest_listener_fast_state: Option<ListenerFastState>,
    pub(super) latest_listener_state: Option<ListenerState>,
    /// Real-time listener update that met a full queue; merged and retried next hop.
    pub(super) pending_listener_update: Option<LandscapeUpdate>,
    pub(super) hab_ecology: crate::core::habituation::HabituationField,
    pub(super) hab_listener: crate::core::habituation::HabituationField,
}

pub(super) struct TemporalState {
    pub(super) observation: TemporalObservation,
    pub(super) harmonic_relations:
        Option<crate::temporal_cognition::harmonic_relations::runtime::Runtime>,
    // Production-side meter core (auditory-motor coupling), separate from the
    // perception meter inside ListenerTwin. See "Perception vs Production".
    // Composer-set terrain priors (meter_stability / temporal_basin) seed it.
    pub(super) prod_meter: MeterNetwork,
    pub(super) harmonic_clock: Option<crate::life::harmonic_rhythm::HarmonicRhythm>,
    pub(super) flora_meter: Option<MeterNetwork>,
    pub(super) flora_rhythm: crate::life::flora::TemporalObservation,
    pub(super) temporal_expectation:
        Option<crate::core::temporal_expectation::AcousticTemporalExpectation>,
    #[cfg(feature = "research")]
    pub(super) footprint_worker: Option<crate::life::onset_footprint::Worker>,
    pub(super) participation_predictions: Vec<(u64, u32, ParticipationPrediction)>,
    /// Traced decisions wait here so their encoding is charged to the report phase.
    #[cfg(feature = "research")]
    pub(super) participation_decisions: Vec<(
        u64,
        crate::life::temporal_participation::ParticipationDecision,
    )>,
}

pub(super) struct EndSequence {
    pub(super) finish_tick: Tick,
}

impl WorkerState {
    pub(super) fn begin_finish(&mut self, now_tick: Tick) {
        if self.lifecycle.finish_begun {
            return;
        }
        self.lifecycle.finish_begun = true;
        if self.schedule_renderer.room.is_some() {
            self.lifecycle.playback_state = PlaybackState::Ending;
        }
        self.pop
            .apply_action(Action::Finish, &self.current_landscape);
        self.lifecycle.scenario_end_tick = Some(now_tick);
        self.pop.voices.clear();
        for flora in &mut self.pop.flora {
            flora.release_at_scenario_end();
        }
        for rain in &mut self.pop.rain {
            rain.release_at_scenario_end();
        }
        self.schedule_renderer.shutdown_at(now_tick);
    }

    pub(super) fn request_end(&mut self, now_tick: Tick) {
        if self.lifecycle.finish_begun || self.lifecycle.end_sequence.is_some() {
            return;
        }
        self.conductor.discard_pending();
        self.pop.release_for_end();
        let voice_ticks = self
            .pop
            .voices
            .iter()
            .filter(|v| v.release.remove_pending)
            .map(|v| {
                (f64::from(v.release.gain) * f64::from(v.release.sec) * f64::from(self.timebase.fs))
                    .ceil() as Tick
            })
            .max()
            .unwrap_or(0);
        let flora_ticks = self
            .pop
            .flora
            .iter()
            .map(|f| f.remaining_release_ticks())
            .max()
            .unwrap_or(0);
        let rain_ticks = self
            .pop
            .rain
            .iter()
            .map(|r| r.remaining_release_ticks())
            .max()
            .unwrap_or(0);
        let hop = self.timebase.hop as Tick;
        let finish_tick = now_tick
            .saturating_add(voice_ticks.max(flora_ticks).max(rain_ticks))
            .div_ceil(hop)
            .saturating_mul(hop);
        self.lifecycle.end_sequence = Some(EndSequence { finish_tick });
        self.lifecycle.playback_state = PlaybackState::Ending;
    }

    fn exit_during_recovery(&self, cfg: &WorkerConfig) -> bool {
        cfg.audio_counters
            .as_ref()
            .is_some_and(|c| c.recovering.load(Ordering::Acquire))
            && (cfg.end_request.load(Ordering::SeqCst) || self.lifecycle.end_sequence.is_some())
    }

    pub(super) fn current_time(&self) -> f64 {
        self.timebase
            .tick_to_sec(self.timebase.frame_start_tick(self.frame_idx))
    }

    pub(super) fn new(
        mut pop: Community,
        conductor: Conductor,
        landscape: Landscape,
        lparams: LandscapeParams,
        dorsal: DorsalStream,
        reporter: Option<JsonlReporter>,
        cfg: &WorkerConfig,
    ) -> Self {
        let expression = conductor.expression;
        pop.expression = expression;
        pop.harmonic_rhythm = cfg.harmonic_rhythm > 0;
        let mut current_landscape: Landscape = landscape;
        #[cfg(feature = "native")]
        let harmonic_relations = cfg.harmonic_relations.as_ref().map(|(geometry, sources)| {
            let runtime = crate::temporal_cognition::harmonic_relations::runtime::Runtime::new(
                geometry.clone(),
                cfg.fs as u32,
                cfg.hop,
                *sources,
            );
            current_landscape.harmonic.recorder = Some(runtime.recorder.clone());
            runtime
        });
        #[cfg(not(feature = "native"))]
        let harmonic_relations = None;
        let log_space = current_landscape.space.clone();
        let timebase = crate::core::timebase::Timebase {
            fs: cfg.fs,
            hop: cfg.hop,
        };
        let new_meter = || {
            cfg.temporal_config
                .make_meter(cfg.fs as u32, cfg.hop as u64)
        };
        let mut prod_meter = new_meter();
        prod_meter.set_shaping(cfg.meter_shaping);
        let needs_flora_meter =
            cfg.habitat_meter || cfg.harmonic_rhythm > 0 || cfg.felt.requested();
        let flora_meter = needs_flora_meter.then(|| {
            let mut meter = new_meter();
            meter.set_shaping(cfg.meter_shaping);
            cfg.felt.setup_meter(&mut meter, cfg.fs, cfg.hop);
            meter
        });
        let mut generator_model =
            crate::life::generator_model::GeneratorModel::new(timebase, log_space.clone());
        generator_model.advance_to(timebase.frame_start_tick(0));
        let hab_bins = current_landscape.space.n_bins();
        let hab_ecology = crate::core::habituation::HabituationField::new(
            &lparams.habituation,
            lparams.consonance_representation.theta,
            hab_bins,
        );
        let hab_listener = crate::core::habituation::HabituationField::new(
            &lparams.habituation,
            lparams.consonance_representation.theta,
            hab_bins,
        );
        let mut listener_twin = ListenerTwin::with_sample_rate(
            cfg.fs,
            crate::listener_twin::ListenerTwinConfig::default(),
        );
        if reporter.is_some() && !listener_twin.enable_contour_reporting(cfg.fs) {
            warn!("listener contour reporting unavailable below 2400 Hz sample rate");
        }
        let now = Instant::now();
        let monitor = AudioMonitor::new();
        let harmonic_clock = (cfg.harmonic_rhythm > 0).then(|| {
            crate::life::harmonic_rhythm::HarmonicRhythm::new(
                cfg.harmonic_rhythm,
                cfg.meter_shaping,
            )
        });
        let temporal_expectation =
            crate::core::temporal_expectation::AcousticTemporalExpectation::new(cfg.fs as u32);
        let mut schedule_renderer = ScheduleRenderer::new(timebase).with_expression(expression);
        schedule_renderer.prototype = cfg.render_prototype;
        Self {
            #[cfg(test)]
            offline_body_probe: None,
            pop,
            loo_populations: Vec::new(),
            conductor,
            current_landscape,
            lparams,
            dorsal,
            reporter,
            profile: None,
            timebase,
            log_space,
            frame_idx: 0,
            generator_model,
            #[cfg(feature = "native")]
            loo_recycle: {
                let (tx, rx) = bounded(1);
                tx.send(Default::default())
                    .expect("initial LOO capture storage");
                (tx, rx)
            },
            #[cfg(not(feature = "native"))]
            loo_storage: Some(Default::default()),
            schedule_renderer,
            phonation_batches_buf: Vec::new(),
            lifecycle: PlaybackLifecycle {
                playback_state: PlaybackState::NotStarted,
                finish_begun: false,
                end_sequence: None,
                finish_logged: false,
                finished: false,
                finalized: false,
                scenario_end_tick: None,
                script_error_reported: false,
                failures: Vec::new(),
            },
            ui: UiDelivery {
                pending_final_ui_frame: None,
                ui_rhythm_hops: Default::default(),
                last_ui_update: now,
                last_guard_log: now - Duration::from_millis(200),
                last_tick_log: now,
                monitor,
            },
            listener: ListenerSync {
                last_analysis_frame: None,
                last_listener_analysis_frame: None,
                listener_min_valid_frame: 0,
                listener_twin,
                latest_listener_fast_state: None,
                latest_listener_state: None,
                pending_listener_update: None,
                hab_ecology,
                hab_listener,
            },
            temporal: TemporalState {
                observation: Default::default(),
                harmonic_relations,
                prod_meter,
                harmonic_clock,
                flora_meter,
                flora_rhythm: Default::default(),
                temporal_expectation,
                #[cfg(feature = "native")]
                #[cfg(feature = "research")]
                footprint_worker: cfg.onset_comparison.map(|_| {
                    let mut worker = crate::life::onset_footprint::Worker::new();
                    if cfg.deterministic_footprints {
                        worker.deliver_footprints_deterministically();
                    }
                    worker
                }),
                #[cfg(not(feature = "native"))]
                #[cfg(feature = "research")]
                footprint_worker: None,
                participation_predictions: Vec::new(),
                #[cfg(feature = "research")]
                participation_decisions: Vec::new(),
            },
        }
    }
}

#[cfg(feature = "native")]
pub(super) fn worker_loop(
    cfg: WorkerConfig,
    mut channels: WorkerChannels,
    mut state: WorkerState,
) -> Result<(), String> {
    state.schedule_renderer.profile = state.profile.as_ref().map(|_| Default::default());

    if cfg.start_flag.load(Ordering::SeqCst) {
        state.lifecycle.playback_state = PlaybackState::Playing;
    }
    let samples_per_hop = cfg.hop * (1 + usize::from(state.schedule_renderer.room.is_some()));
    let idle_silence = vec![0.0f32; samples_per_hop];
    #[cfg(test)]
    let mut l1_probe = super::l1_probe::Probe::new("audio");

    temporal_ports::read_initial(&mut state, &channels);
    if let Some(tx) = &channels.ui_tx {
        let mut initial_frame = build_initial_ui_frame(
            &cfg.scenario_name,
            &state.conductor,
            &state.pop,
            &state.current_landscape,
            state.timebase.frame_start_tick(state.frame_idx),
            state.lifecycle.playback_state,
            cfg.fs,
        );
        state
            .temporal
            .observation
            .set_initial_ui(&mut initial_frame);
        let _ = tx.try_send(initial_frame);
    }

    loop {
        if state.exit_during_recovery(&cfg) {
            cfg.exiting.store(true, Ordering::SeqCst);
        }
        if cfg
            .audio_counters
            .as_ref()
            .is_some_and(|counters| counters.failed.load(Ordering::Acquire))
        {
            cfg.exiting.store(true, Ordering::SeqCst);
        }
        if cfg.exiting.load(Ordering::SeqCst) || state.lifecycle.finished {
            if !state.lifecycle.finalized {
                #[cfg(test)]
                drop(l1_probe.take());
                finish_run(&cfg, &mut channels, &mut state);
            }
            if cfg.exiting.load(Ordering::SeqCst) || !cfg.wait_user_exit {
                break;
            }
        }

        if cfg.end_request.load(Ordering::SeqCst) && !state.lifecycle.finish_begun {
            let now = state.timebase.frame_start_tick(state.frame_idx);
            state.request_end(now);
        }
        if state.lifecycle.playback_state == PlaybackState::NotStarted
            && !cfg.start_flag.load(Ordering::SeqCst)
        {
            thread::sleep(Duration::from_millis(10));
            continue;
        } else if state.lifecycle.playback_state == PlaybackState::NotStarted {
            state.lifecycle.playback_state = PlaybackState::Playing;
        }

        if state.lifecycle.finished && cfg.wait_user_exit {
            if let Some(frame) = state.ui.pending_final_ui_frame.take()
                && let Some(tx) = &channels.ui_tx
                && let Err(crate::handoff::TrySendError::Full(frame)) = tx.try_send(frame)
            {
                state.ui.pending_final_ui_frame = Some(frame);
            }
            if let Some(prod) = channels.audio_prod.as_mut() {
                while prod.vacant_len() >= samples_per_hop {
                    AudioOutput::push_samples(prod, &idle_silence, None);
                }
            }
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        let mut produced_any = false;
        if channels.audio_prod.is_some() {
            while channels
                .audio_prod
                .as_ref()
                .is_some_and(|prod| prod.vacant_len() >= samples_per_hop)
                && !cfg.exiting.load(Ordering::SeqCst)
                && !state.lifecycle.finished
                && !state.exit_during_recovery(&cfg)
            {
                process_hop(&cfg, &mut channels, &mut state)?;
                produced_any = true;
            }
        } else {
            // Offline/render-only mode: progress even without audio output.
            #[cfg(test)]
            let l1_begin = l1_probe
                .as_mut()
                .map(|p| (state.frame_idx, p.before(state.frame_idx)));
            process_hop(&cfg, &mut channels, &mut state)?;
            #[cfg(test)]
            if let (Some(p), Some((frame, begin))) = (&mut l1_probe, l1_begin) {
                let voices = state
                    .pop
                    .voices
                    .iter()
                    .filter(|v| {
                        v.is_alive()
                            && (v.effective_control.pitch.leave_self_out || v.selection_environment)
                    })
                    .count();
                p.after(
                    frame,
                    begin,
                    voices,
                    channels
                        .audio_to_analysis_tx
                        .as_ref()
                        .map_or(0, |tx| tx.len()),
                );
            }
            produced_any = true;
        }

        if !produced_any {
            thread::sleep(Duration::from_millis(1));
        }
    }
    if cfg
        .audio_counters
        .as_ref()
        .is_some_and(|counters| counters.failed.load(Ordering::Acquire))
    {
        state
            .lifecycle
            .failures
            .push("audio stream configuration unsupported during recovery".into());
    }
    if state.lifecycle.failures.is_empty() {
        Ok(())
    } else {
        Err(state.lifecycle.failures.join("; "))
    }
}

#[cfg(feature = "native")]
pub(super) fn finish_run(
    cfg: &WorkerConfig,
    channels: &mut WorkerChannels,
    state: &mut WorkerState,
) {
    #[cfg(test)]
    if let Some(relations) = &state.temporal.harmonic_relations
        && let Some(directory) = std::env::var_os("L1_OUTPUT")
    {
        let path = std::path::Path::new(&directory).join("harmonic-relations-at-audio-end.json");
        serde_json::to_writer(
            std::fs::File::create(path).expect("L1 audio-end harmonic statistics"),
            &relations.summary(),
        )
        .expect("L1 audio-end harmonic statistics serialization");
    }
    if state.temporal.harmonic_relations.is_some() && state.frame_idx > 0 {
        wait_for_analysis(state, &channels.analysis_result_rx, &cfg.exiting);
    }
    if let Some(relations) = &mut state.temporal.harmonic_relations {
        if let Some(support) = state.current_landscape.harmonic.support.clone() {
            relations.advance(&support, &mut state.current_landscape);
        }
        relations.finish();
        let summary = relations.summary();
        tracing::info!(summary = %summary, "harmonic relations runtime");
        #[cfg(test)]
        {
            eprintln!("HV7_RUNTIME {summary}");
            if let Some(directory) = std::env::var_os("L1_OUTPUT") {
                let path = std::path::Path::new(&directory).join("harmonic-relations.json");
                serde_json::to_writer(
                    std::fs::File::create(path).expect("L1 harmonic statistics"),
                    &summary,
                )
                .expect("L1 harmonic statistics serialization");
            }
        }
    }
    if state.lifecycle.scenario_end_tick.is_none() && state.reporter.is_some() {
        for voice in &state.pop.voices {
            if let Some(mut record) = voice.pitch_ctl.dwell_state {
                record.event = "end";
                record.at = Some(state.timebase.frame_start_tick(state.frame_idx));
                report_try(&mut state.reporter, "dwell end", |writer| {
                    writer.write_dwell(
                        voice.id(),
                        voice.metadata.generation,
                        cfg.fs as u32,
                        &record,
                    )
                });
            }
        }
    }
    #[cfg(feature = "research")]
    if let Some(worker) = state.temporal.footprint_worker.as_mut() {
        worker.finish();
        route_body_footprints(
            &mut state.pop.voices,
            worker,
            &mut state.reporter,
            state.timebase.frame_start_tick(state.frame_idx),
        );
        report_try(
            &mut state.reporter,
            "final body footprint worker",
            |writer| writer.write_body_footprint_worker(&worker.stats),
        );
    }
    temporal_ports::finish_capture(state);

    temporal_ports::finish_analysis(state, channels, cfg);
    for (voice_id, _, prediction) in state.temporal.participation_predictions.drain(..) {
        let sample = participation_outcome(voice_id, cfg.fs as u32, prediction, None)
            .expect("end of input closes every pending observation");
        report_try(&mut state.reporter, "participation outcome", |writer| {
            writer.write_participation_outcome(&sample)
        });
    }

    report_try(&mut state.reporter, "rhythm summary", |writer| {
        writer.write_rhythm_summary()
    });
    report_try(
        &mut state.reporter,
        "listener confidence summary",
        |writer| writer.write_listener_confidence_summary(),
    );
    report_try(&mut state.reporter, "flush", |writer| writer.flush());
    let mut failure = state
        .reporter
        .as_mut()
        .and_then(|writer| writer.failure.take());
    if let Some(profile) = state.profile.as_mut() {
        profile.background = Some(crate::runtime_profile::BackgroundProfile {
            scope: "terminal worker counters after drain/join; null means disabled, not zero cost; resource distributions cover all received frames and sample-clock 100ms windows, not the selected hop measurement interval; wall processing includes publication lock/copy, delivery includes queue wait, neither is thread CPU time; private startup is worker lane initialization; table/rejected are unused for private body; excludes final profile serialization",
            observation: state
                .temporal
                .observation
                .background_profile(&channels.observation),
        });
    }
    if let Some(profile) = state.profile.take()
        && let Err(err) = profile.write()
    {
        failure = Some(match failure {
            Some(previous) => format!("{previous}; {err}"),
            None => err,
        });
    }
    if let Some(err) = failure {
        state.lifecycle.failures.push(err);
    }
    state.lifecycle.finalized = true;
    if !state.lifecycle.failures.is_empty() {
        cfg.exiting.store(true, Ordering::SeqCst);
    }
}

/// One generated hop: analysis merge, ecology step, render, routing, telemetry.
pub(super) fn process_hop(
    cfg: &WorkerConfig,
    channels: &mut WorkerChannels,
    state: &mut WorkerState,
) -> Result<(), String> {
    let t_start = Instant::now();
    if state.profile.is_some() {
        begin_allocations();
    }
    let strict = state
        .schedule_renderer
        .room
        .as_ref()
        .is_some_and(|room| room.strict());
    if strict && let Some(reporter) = &mut state.reporter {
        reporter.begin_room_hop();
    }
    let frame_idx = state.frame_idx;
    #[cfg(feature = "native")]
    let output_channels = 1 + usize::from(state.schedule_renderer.room.is_some());
    #[cfg(feature = "native")]
    let buffer_capacity = channels
        .audio_prod
        .as_ref()
        .map(|prod| prod.capacity().get() / output_channels)
        .unwrap_or(cfg.hop);
    #[cfg(feature = "native")]
    let occupancy = channels
        .audio_prod
        .as_ref()
        .map(|prod| buffer_capacity.saturating_sub(prod.vacant_len() / output_channels))
        .unwrap_or(0);

    #[cfg(not(feature = "native"))]
    let (buffer_capacity, occupancy) = (cfg.hop, 0);
    let now_tick = state.timebase.frame_start_tick(state.frame_idx);
    let now_sec = state.timebase.tick_to_sec(now_tick);
    state.generator_model.advance_to(now_tick);
    if state.frame_idx == 0 || state.ui.last_tick_log.elapsed() >= Duration::from_secs(1) {
        debug!(
            "[tick] frame_idx={} now_tick={} now_sec={:.6}",
            state.frame_idx, now_tick, now_sec
        );
        state.ui.last_tick_log = Instant::now();
    }
    state.pop.set_current_frame(state.frame_idx);

    let analysis_wait_start = Instant::now();
    let analysis_updated = wait_for_analysis(state, &channels.analysis_result_rx, &cfg.exiting);
    let analysis_wait = analysis_wait_start.elapsed();
    let landscape_start = state.profile.as_ref().map(|_| Instant::now());
    apply_landscape_updates(state, channels, cfg, analysis_updated, now_tick);
    let landscape_update = landscape_start
        .map(|start| start.elapsed())
        .unwrap_or_default();

    let listener_wait_start = Instant::now();
    let listener_state_update = wait_for_listener(state, channels, cfg);
    let listener_wait = listener_wait_start.elapsed();
    if let Some(listener_state) = listener_state_update {
        state.listener.latest_listener_state = listener_state;
    }
    let listener_pressure = cfg
        .dcc_coupler
        .pressure(state.listener.latest_listener_state);
    let listener_pressure_update =
        listener_state_update.map(|listener_state| cfg.dcc_coupler.pressure(listener_state));

    temporal_ports::read_live(state, channels);

    let population_start = state.profile.as_ref().map(|_| Instant::now());
    state.current_landscape.harmonic.now_sample = now_tick;
    if let Some(relations) = &mut state.temporal.harmonic_relations
        && let Some(support) = state.current_landscape.harmonic.support.clone()
    {
        relations.advance(&support, &mut state.current_landscape);
    }
    let phonation_count = advance_population(state, cfg, now_tick, listener_pressure);
    let population_elapsed = population_start
        .map(|start| start.elapsed())
        .unwrap_or_default();
    let reports_start = state.profile.as_ref().map(|_| Instant::now());
    emit_hop_reports(
        state,
        cfg,
        now_sec,
        phonation_count,
        listener_state_update.flatten(),
        listener_pressure_update,
    );
    let reports_elapsed = reports_start
        .map(|start| start.elapsed())
        .unwrap_or_default();

    let render_start = state.profile.as_ref().map(|_| Instant::now());
    let (presentation_chunk, habitat_chunk, max_abs, synthesis_elapsed) =
        render_and_route_audio(state, channels, cfg, now_tick, phonation_count).map_err(|e| {
            if state.profile.is_some() {
                finish_allocations();
            }
            cfg.exiting.store(true, Ordering::SeqCst);
            let prefix = format!(
                "{e}; {frame_idx} complete hops published (end tick {now_tick}, exclusive)"
            );
            match &state.reporter {
                Some(writer) => format!("{prefix}; partial report {}", writer.path),
                None => prefix,
            }
        })?;
    if strict {
        report_try(
            &mut state.reporter,
            "room hop",
            JsonlReporter::publish_room_hop,
        );
    }
    if let Some(relations) = &mut state.temporal.harmonic_relations {
        relations.collect(now_tick + cfg.hop as u64);
    }
    let render_route = render_start
        .map(|start| start.elapsed())
        .unwrap_or_default();
    let post_render_start = state.profile.as_ref().map(|_| Instant::now());
    let second_ticks = cfg.fs as Tick;
    if state.reporter.is_some() {
        for flora in &state.pop.flora {
            if flora.lifecycle_changed
                || state
                    .temporal
                    .harmonic_clock
                    .as_ref()
                    .is_some_and(|c| !c.ticks.is_empty())
                || now_tick == 0
                || now_tick / second_ticks
                    != now_tick.saturating_sub(cfg.hop as Tick) / second_ticks
            {
                report_try(&mut state.reporter, "flora", |writer| {
                    writer.write_flora(
                        flora,
                        now_sec,
                        state.current_landscape.at_site(flora.site_id()),
                    )
                });
            }
        }
    }
    let mut loo_chunk = (state.schedule_renderer.sites.is_some()
        || !state.loo_populations.is_empty()
        || state.pop.has_placed_flora
        || state.pop.has_placed_rain_observer)
        .then(|| crate::core::stream::loo::PcmPacket {
            sites: state
                .schedule_renderer
                .sites
                .as_mut()
                .map(|s| std::mem::take(&mut s.full))
                .unwrap_or_default(),
            sources: std::mem::take(&mut state.schedule_renderer.loo_pcm),
            #[cfg(feature = "native")]
            recycle: state.loo_recycle.0.clone(),
        });
    let channel_peak = [max_abs, max_abs];

    temporal_ports::send_before_meter(state, channels, cfg, &habitat_chunk, &mut loo_chunk);
    drive_production_meter(
        state,
        cfg,
        &mut channels.observation,
        habitat_chunk.as_ref(),
    );
    state.listener.latest_listener_fast_state =
        Some(state.listener.listener_twin.observe_presentation_audio(
            now_tick,
            state.frame_idx,
            presentation_chunk.as_ref(),
        ));
    state.listener.listener_twin.observe_contour_audio(
        state.timebase.frame_start_tick(state.frame_idx),
        presentation_chunk.as_ref(),
        |event| {
            let sample = ListenerContourSample {
                generated_frame_id: state.frame_idx,
                time_sec: event.available_sec,
                event: event.kind,
                onset_sec: event.onset_sec,
                periodic_frequency_hz: event.periodic_frequency_hz,
                periodicity: event.periodicity,
                delta_log2: event.score.and_then(|score| score.delta_log2),
                gain_bits: event.score.and_then(|score| score.gain_bits),
                loss_bits: event.score.and_then(|score| score.loss_bits),
                context_support: event.score.map(|score| score.context_support),
                error_threshold_bits: event.score.and_then(|score| score.error_threshold_bits),
                calibration_events: event.score.map(|score| score.calibration_events),
                error_candidate: event.score.is_some_and(|score| score.error_candidate),
                gap_threshold_sec: event.gap_threshold_sec,
                missing_start_sec: event.missing_start_sec,
            };
            report_try(&mut state.reporter, "listener contour", |writer| {
                writer.write_listener_contour(&sample)
            });
        },
    );

    // Feed every habitat hop to analysis. Dropping hops breaks time continuity.
    let send_after_meter = !cfg.temporal_config.t2_enabled();
    if send_after_meter
        && let Err(err) = channels
            .audio_to_analysis_tx
            .as_ref()
            .expect("analysis input open during generation")
            .send((
                state.frame_idx,
                Arc::clone(&habitat_chunk),
                loo_chunk.take(),
            ))
    {
        state
            .lifecycle
            .failures
            .push(format!("analysis worker disconnected: {err}"));
        cfg.exiting.store(true, Ordering::SeqCst);
    }

    let room = state.schedule_renderer.room.as_ref();
    let normal_wet_finish = state.lifecycle.finish_begun
        && room.is_some()
        && state.lifecycle.failures.is_empty()
        && !cfg.exiting.load(Ordering::SeqCst);
    let abort_without_tails = state.pop.abort_requested;
    let abort_without_tails = abort_without_tails && !cfg.render_prototype.enabled();
    let finished_now = if let Some(room) = room {
        state.lifecycle.finish_begun
            && room.drained()
            && state
                .schedule_renderer
                .is_idle_at(state.timebase.frame_end_tick(frame_idx))
            && state.pop.flora.iter().all(|flora| flora.is_finished())
            && state.pop.rain.iter().all(|rain| rain.is_finished())
    } else if abort_without_tails && !state.pop.has_placed_flora && !state.pop.has_placed_rain {
        true
    } else {
        state.lifecycle.scenario_end_tick.is_some()
            && state.schedule_renderer.is_idle()
            && state.pop.flora.is_empty()
            && state.pop.rain.is_empty()
    };
    if finished_now {
        state.lifecycle.playback_state = PlaybackState::Finished;
    }
    #[cfg(feature = "native")]
    if finished_now && channels.wav_tx.is_some() {
        channels.wav_tx.take();
        info!("[t={:.6}] WAV closed.", state.current_time());
    }

    let should_send_ui = channels.ui_tx.is_some()
        && (finished_now || state.ui.last_ui_update.elapsed() >= UI_MIN_INTERVAL);

    if channels.ui_tx.is_some() {
        buffer_listener_rhythm_hop(state, cfg);
    }

    let peak_level = state.ui.monitor.max_peak.max(max_abs);

    log_guard_meter(cfg, &mut state.ui.last_guard_log, now_sec);

    if should_send_ui {
        send_runtime_ui_frame(
            state,
            channels,
            cfg,
            presentation_chunk,
            peak_level,
            channel_peak,
            now_tick,
        );
    }

    if finished_now {
        state.lifecycle.finished = true;
        if !state.lifecycle.finish_logged {
            let note = if cfg.wait_user_exit {
                "Waiting for user exit."
            } else {
                "Exiting."
            };
            info!("[t={:.6}] Scenario finished. {note}", state.current_time());
            state.lifecycle.finish_logged = true;
        }
        if !cfg.wait_user_exit {
            // `exiting` doubles as the stop flag that releases the output stream, so a
            // natural end first lets the device play the queued tail. The wait is
            // bounded by twice the ring duration; stop, recovery and failure end it.
            #[cfg(feature = "native")]
            if state.lifecycle.failures.is_empty()
                && let Some(prod) = channels.audio_prod.as_ref()
            {
                let deadline = Instant::now()
                    + Duration::from_secs_f64(2.0 * buffer_capacity as f64 / f64::from(cfg.fs));
                while prod.occupied_len() > 0
                    && Instant::now() < deadline
                    && !cfg.exiting.load(Ordering::SeqCst)
                    && !cfg.audio_counters.as_ref().is_some_and(|c| {
                        c.recovering.load(Ordering::Acquire) || c.failed.load(Ordering::Acquire)
                    })
                {
                    thread::sleep(Duration::from_millis(1));
                }
            }
            cfg.exiting.store(true, Ordering::SeqCst);
        }
    }

    state.ui.monitor.update(
        state.current_time(),
        state.frame_idx,
        cfg.hop,
        cfg.hop_duration,
        buffer_capacity,
        occupancy,
        max_abs,
        t_start.elapsed(),
        analysis_wait,
        listener_wait,
        cfg.underrun_frames.as_deref(),
        state.lifecycle.finish_begun && !normal_wet_finish,
        cfg.telemetry,
        #[cfg(feature = "native")]
        state
            .conductor
            .score
            .as_ref()
            .map(|session| session.stream.as_ref()),
        &mut state.pop,
        cfg.audio_counters.as_deref(),
    );

    if !state.lifecycle.finished {
        state.frame_idx += 1;
    }
    // Include preceding telemetry in the hop cost, but exclude this record's own write.
    report_try(&mut state.reporter, "hop timing", |writer| {
        writer.write_hop_timing(&HopTimingSample {
            room_streams: state.schedule_renderer.room.as_ref().map(|r| r.counts()),
            frame_idx,
            time_sec: now_sec,
            elapsed_us: t_start.elapsed().as_secs_f64() * 1_000_000.0,
            analysis_wait_us: analysis_wait.as_secs_f64() * 1_000_000.0,
            listener_wait_us: listener_wait.as_secs_f64() * 1_000_000.0,
            hop_budget_us: cfg.hop_duration.as_secs_f64() * 1_000_000.0,
            audio_output: if cfg.underrun_frames.is_some() {
                "device"
            } else {
                "no_device"
            },
            underrun_frames_total: cfg
                .underrun_frames
                .as_ref()
                .map(|count| count.load(Ordering::Relaxed)),
        })
    });
    if let Some(profile) = state.profile.as_mut() {
        let alive_voice_count = state
            .pop
            .voices
            .iter()
            .filter(|voice| voice.is_alive())
            .count();
        let post_render_us = post_render_start
            .map(|start| start.elapsed().as_secs_f64() * 1_000_000.0)
            .unwrap_or_default();
        let elapsed_us = t_start.elapsed().as_secs_f64() * 1_000_000.0;
        let worker_allocations = finish_allocations();
        profile.record(HopProfile {
            frame_idx,
            time_sec: now_sec,
            alive_voice_count,
            elapsed_us,
            analysis_wait_us: analysis_wait.as_secs_f64() * 1_000_000.0,
            listener_wait_us: listener_wait.as_secs_f64() * 1_000_000.0,
            landscape_update_us: landscape_update.as_secs_f64() * 1_000_000.0,
            population_us: population_elapsed.as_secs_f64() * 1_000_000.0,
            reports_us: reports_elapsed.as_secs_f64() * 1_000_000.0,
            render_route_us: render_route.as_secs_f64() * 1_000_000.0,
            synthesis_us: synthesis_elapsed.as_secs_f64() * 1_000_000.0,
            rendering: state.schedule_renderer.profile,
            rendered_tone_count: state.schedule_renderer.active_tone_count(),
            post_render_us,
            worker_allocations,
            temporal_meter: state.temporal.observation.meter_profile(),
            underrun_frames_total: cfg
                .underrun_frames
                .as_ref()
                .map(|count| count.load(Ordering::Relaxed)),
        });
    }
    Ok(())
}
