use super::*;

/// Immutable per-run settings and shared control flags for the worker thread.
pub(super) struct WorkerConfig {
    pub(super) loo_populations: Vec<u64>,
    pub(super) render_prototype: bool,
    pub(super) temporal_meter_t2: bool,
    pub(super) onset_comparison: Option<crate::config::TemporalOnsetComparisonConfig>,
    pub(super) scenario_name: String,
    pub(super) wait_user_exit: bool,
    pub(super) start_flag: Arc<AtomicBool>,
    pub(super) exiting: Arc<AtomicBool>,
    pub(super) scaffold: ScaffoldConfig,
    pub(super) meter_shaping: crate::core::meter::MeterShaping,
    pub(super) guard_meter: Option<Arc<LimiterMeter>>,
    pub(super) underrun_frames: Option<Arc<AtomicU64>>,
    pub(super) audio_counters: Option<Arc<crate::audio::output::AudioCallbackCounters>>,
    pub(super) dcc_coupler: DccCoupler,
    pub(super) hop: usize,
    pub(super) hop_duration: Duration,
    pub(super) fs: f32,
    /// Offline/report paths consume analysis with a fixed lag so results do not
    /// depend on worker scheduling. Real-time leaves the listener best-effort.
    pub(super) deterministic_analysis: bool,
    pub(super) deterministic_footprints: bool,
}

/// Channel endpoints and the audio ring-buffer producer owned by the worker.
pub(super) struct WorkerChannels {
    pub(super) t2_inbox: Option<crate::temporal_cognition::t2::runtime_input::Inbox>,
    pub(super) temporal_snapshots:
        [Option<Arc<std::sync::Mutex<crate::temporal_cognition::observation::Snapshot>>>; 2],
    pub(super) ui_tx: Option<Sender<UiFrame>>,
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
    pub(super) loo_recycle: (
        Sender<Vec<crate::core::stream::loo::SourcePcm>>,
        Receiver<Vec<crate::core::stream::loo::SourcePcm>>,
    ),
    #[cfg(test)]
    pub(super) offline_body_probe: Option<OfflineBodyProbe>,
    pub(super) body_snapshot: Option<Box<crate::temporal_cognition::body::Snapshot>>,
    pub(super) footprint_worker: Option<crate::life::onset_footprint::Worker>,
    pub(super) temporal_frames: Box<[crate::temporal_cognition::observation::Snapshot; 2]>,
    pub(super) pop: Community,
    pub(super) conductor: Conductor,
    pub(super) current_landscape: Landscape,
    pub(super) lparams: LandscapeParams,
    pub(super) dorsal: DorsalStream,
    pub(super) reporter: Option<JsonlReporter>,
    pub(super) profile: Option<RunProfile>,
    pub(super) timebase: crate::core::timebase::Timebase,
    pub(super) playback_state: PlaybackState,
    pub(super) finish_logged: bool,
    pub(super) finished: bool,
    pub(super) finalized: bool,
    pub(super) pending_final_ui_frame: Option<UiFrame>,
    pub(super) failures: Vec<String>,
    pub(super) log_space: Log2Space,
    pub(super) frame_idx: u64,
    pub(super) monitor: AudioMonitor,
    pub(super) last_guard_log: Instant,
    pub(super) last_ui_update: Instant,
    pub(super) last_tick_log: Instant,
    pub(super) last_analysis_frame: Option<u64>,
    pub(super) last_listener_analysis_frame: Option<u64>,
    pub(super) listener_min_valid_frame: u64,
    pub(super) listener_twin: ListenerTwin,
    pub(super) latest_listener_fast_state: Option<ListenerFastState>,
    pub(super) latest_listener_state: Option<ListenerState>,
    // Production-side meter core (auditory-motor coupling), separate from the
    // perception meter inside ListenerTwin. See "Perception vs Production".
    // Composer-set terrain priors (meter_stability / temporal_basin) seed it.
    pub(super) prod_meter: MeterNetwork,
    pub(super) t2_sample: Option<crate::life::report::TemporalMeterSample>,
    pub(super) temporal_expectation:
        Option<crate::core::temporal_expectation::AcousticTemporalExpectation>,
    pub(super) participation_predictions: Vec<(u64, ParticipationPrediction)>,
    /// Traced decisions wait here so their encoding is charged to the report phase.
    pub(super) participation_decisions: Vec<(
        u64,
        crate::life::temporal_participation::ParticipationDecision,
    )>,
    pub(super) generator_model: crate::life::generator_model::GeneratorModel,
    pub(super) hab_ecology: crate::core::habituation::HabituationField,
    pub(super) hab_listener: crate::core::habituation::HabituationField,
    pub(super) schedule_renderer: ScheduleRenderer,
    pub(super) scenario_end_tick: Option<Tick>,
    pub(super) phonation_batches_buf: Vec<PhonationBatch>,
}

impl WorkerState {
    pub(super) fn current_time(&self) -> f32 {
        self.timebase
            .tick_to_sec(self.timebase.frame_start_tick(self.frame_idx))
    }

    pub(super) fn new(
        pop: Community,
        conductor: Conductor,
        landscape: Landscape,
        lparams: LandscapeParams,
        dorsal: DorsalStream,
        reporter: Option<JsonlReporter>,
        cfg: &WorkerConfig,
    ) -> Self {
        let current_landscape: Landscape = landscape;
        let log_space = current_landscape.space.clone();
        let timebase = crate::core::timebase::Timebase {
            fs: cfg.fs,
            hop: cfg.hop,
        };
        let mut prod_meter = if cfg.temporal_meter_t2 {
            MeterNetwork::new_continuous(cfg.fs as u32, cfg.hop as u64)
        } else {
            MeterNetwork::new()
        };
        prod_meter.set_shaping(cfg.meter_shaping);
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
        Self {
            #[cfg(test)]
            offline_body_probe: None,
            pop,
            temporal_frames: Default::default(),
            conductor,
            current_landscape,
            lparams,
            dorsal,
            reporter,
            profile: None,
            timebase,
            playback_state: PlaybackState::NotStarted,
            finish_logged: false,
            finished: false,
            finalized: false,
            pending_final_ui_frame: None,
            failures: Vec::new(),
            log_space,
            frame_idx: 0,
            monitor: AudioMonitor::new(),
            last_guard_log: now - Duration::from_millis(200),
            last_ui_update: now,
            last_tick_log: now,
            last_analysis_frame: None,
            last_listener_analysis_frame: None,
            listener_min_valid_frame: 0,
            listener_twin,
            latest_listener_fast_state: None,
            latest_listener_state: None,
            prod_meter,
            t2_sample: None,
            temporal_expectation:
                crate::core::temporal_expectation::AcousticTemporalExpectation::new(cfg.fs as u32),
            participation_predictions: Vec::new(),
            participation_decisions: Vec::new(),
            generator_model,
            hab_ecology,
            hab_listener,
            loo_recycle: {
                let (tx, rx) = bounded(1);
                tx.send(Vec::new()).expect("initial LOO capture storage");
                (tx, rx)
            },
            schedule_renderer: ScheduleRenderer::new(timebase).with_prototype(cfg.render_prototype),
            scenario_end_tick: None,
            phonation_batches_buf: Vec::new(),
            body_snapshot: None,
            footprint_worker: cfg.onset_comparison.map(|_| {
                let mut worker = crate::life::onset_footprint::Worker::new();
                if cfg.deterministic_footprints {
                    worker.deliver_footprints_deterministically();
                }
                worker
            }),
        }
    }
}

pub(super) fn worker_loop(
    cfg: WorkerConfig,
    mut channels: WorkerChannels,
    mut state: WorkerState,
) -> Result<(), String> {
    state.schedule_renderer.profile = state.profile.as_ref().map(|_| Default::default());

    if cfg.start_flag.load(Ordering::SeqCst) {
        state.playback_state = PlaybackState::Playing;
    }
    let idle_silence = vec![0.0f32; cfg.hop];
    #[cfg(test)]
    let mut l1_probe = super::l1_probe::Probe::new("audio");

    for (target, snapshot) in state
        .temporal_frames
        .iter_mut()
        .zip(&channels.temporal_snapshots)
    {
        if let Some(snapshot) = snapshot {
            *target = *snapshot.lock().expect("initial observation snapshot");
        }
    }
    if let Some(tx) = &channels.ui_tx {
        let mut initial_frame = build_initial_ui_frame(
            &cfg.scenario_name,
            &state.conductor,
            &state.pop,
            &state.current_landscape,
            state.current_time(),
            state.playback_state,
            cfg.fs,
        );
        initial_frame.temporal = Arc::new(*state.temporal_frames);
        let _ = tx.try_send(initial_frame);
    }

    loop {
        if cfg
            .audio_counters
            .as_ref()
            .is_some_and(|counters| counters.callback_errors_total.load(Ordering::Relaxed) > 0)
        {
            cfg.exiting.store(true, Ordering::SeqCst);
        }
        if cfg.exiting.load(Ordering::SeqCst) || state.finished {
            if !state.finalized {
                finish_run(&cfg, &mut channels, &mut state);
            }
            if cfg.exiting.load(Ordering::SeqCst) || !cfg.wait_user_exit {
                break;
            }
        }

        if state.playback_state == PlaybackState::NotStarted
            && !cfg.start_flag.load(Ordering::SeqCst)
        {
            thread::sleep(Duration::from_millis(10));
            continue;
        } else if state.playback_state == PlaybackState::NotStarted {
            state.playback_state = PlaybackState::Playing;
        }

        if state.finished && cfg.wait_user_exit {
            if let Some(frame) = state.pending_final_ui_frame.take()
                && let Some(tx) = &channels.ui_tx
                && let Err(crossbeam_channel::TrySendError::Full(frame)) = tx.try_send(frame)
            {
                state.pending_final_ui_frame = Some(frame);
            }
            if let Some(prod) = channels.audio_prod.as_mut() {
                while prod.vacant_len() >= cfg.hop {
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
                .is_some_and(|prod| prod.vacant_len() >= cfg.hop)
                && !cfg.exiting.load(Ordering::SeqCst)
                && !state.finished
            {
                process_hop(&cfg, &mut channels, &mut state);
                produced_any = true;
            }
        } else {
            // Offline/render-only mode: progress even without audio output.
            #[cfg(test)]
            let l1_begin = l1_probe
                .as_mut()
                .map(|p| (state.frame_idx, p.before(state.frame_idx)));
            process_hop(&cfg, &mut channels, &mut state);
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
    if let Some(counters) = &cfg.audio_counters {
        let errors = counters.callback_errors_total.load(Ordering::Relaxed);
        if errors > 0 {
            state
                .failures
                .push(format!("audio callback failed: {errors} stream errors"));
        }
    }
    if state.failures.is_empty() {
        Ok(())
    } else {
        Err(state.failures.join("; "))
    }
}

pub(super) fn finish_run(
    cfg: &WorkerConfig,
    channels: &mut WorkerChannels,
    state: &mut WorkerState,
) {
    if let Some(worker) = state.footprint_worker.as_mut() {
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
    if let Some(capture) = state.schedule_renderer.body_capture.as_mut() {
        capture.finish();
        let snapshot = capture.snapshot();
        if snapshot.state == crate::temporal_cognition::observation::ObservationState::Failed {
            state
                .failures
                .push("private body analysis worker failed".into());
        }

        if let Some(current) = state.body_snapshot.as_mut() {
            **current = snapshot;
        }
        report_try(&mut state.reporter, "final body observation", |writer| {
            writer.write_body_observation(&snapshot)
        });
    }

    if (channels.temporal_snapshots.iter().any(Option::is_some) || channels.t2_inbox.is_some())
        && channels.audio_to_analysis_tx.is_some()
    {
        // Close input and drain publishers before reading their final observer state.
        // This runs after generation has stopped, never in a live hop or callback.
        channels.audio_to_analysis_tx.take();
        channels.presentation_to_listener_tx.take();
        if let Some(inbox) = channels.t2_inbox.as_mut() {
            loop {
                for _ in inbox.rx.try_iter() {}
                match channels.analysis_result_rx.try_recv() {
                    Ok(_) => {}
                    Err(crossbeam_channel::TryRecvError::Disconnected) => break,
                    Err(crossbeam_channel::TryRecvError::Empty) => {
                        thread::sleep(Duration::from_micros(200))
                    }
                }
            }
            inbox.finish();
            if let Some(sample) = state.t2_sample.as_mut() {
                sample.input = inbox.diagnostics;
                report_try(&mut state.reporter, "final T2 meter", |writer| {
                    writer.write_temporal_meter(sample)
                });
            }
        } else {
            for _ in &channels.analysis_result_rx {}
        }
        if let Some(rx) = &channels.listener_result_rx {
            for _ in rx {}
        }
        let now_sec = state.current_time();

        for (target, snapshot) in state
            .temporal_frames
            .iter_mut()
            .zip(&channels.temporal_snapshots)
        {
            if let Some(snapshot) = snapshot {
                *target = *snapshot.lock().expect("final observation snapshot");
                report_try(
                    &mut state.reporter,
                    "final temporal observation",
                    |writer| writer.write_temporal_observation(now_sec, target),
                );
            }
        }
        state.last_ui_update = Instant::now() - UI_MIN_INTERVAL;
        let now_tick = state.timebase.frame_start_tick(state.frame_idx);
        send_runtime_ui_frame(state, channels, cfg, Arc::from([]), 0.0, [0.0; 2], now_tick);
    }
    for (voice_id, prediction) in state.participation_predictions.drain(..) {
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
        use crate::runtime_profile::{BackgroundProfile, BodyWorkerProfile, SharedWorkerProfile};
        profile.background = Some(BackgroundProfile {
            scope: "terminal worker counters after drain/join; null means disabled, not zero cost; resource distributions cover all received frames and sample-clock 100ms windows, not the selected hop measurement interval; wall processing includes publication lock/copy, delivery includes queue wait, neither is thread CPU time; private startup is worker lane initialization; table/rejected are unused for private body; excludes final profile serialization",
            shared: std::array::from_fn(|bus| {
                channels.temporal_snapshots[bus].as_ref().map(|_| {
                    let snapshot = &state.temporal_frames[bus];
                    SharedWorkerProfile {
                        bus: snapshot.bus,
                        state: snapshot.state,
                        received_frames: snapshot.received_frames,
                        source_missing_samples: snapshot.source_missing_samples,
                        delivery_dropped_frames: snapshot.delivery_dropped_frames,
                        rejected_frames: snapshot.rejected_frames,
                    }
                })
            }),
            body: state
                .body_snapshot
                .as_ref()
                .map(|snapshot| BodyWorkerProfile {
                    finished: snapshot.finished,
                    input_end: snapshot.input_end,
                    processed_frames: snapshot.processed_frames,
                    invalid_hops: snapshot.invalid_hops,
                    capture_drops: snapshot.capture_drops,
                    outside_voice_hops: snapshot.outside_voice_hops,
                    worker_resources: snapshot.worker_resources,
                }),
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
        state.failures.push(err);
    }
    state.finalized = true;
    if !state.failures.is_empty() {
        cfg.exiting.store(true, Ordering::SeqCst);
    }
}

/// One generated hop: analysis merge, ecology step, render, routing, telemetry.
pub(super) fn process_hop(
    cfg: &WorkerConfig,
    channels: &mut WorkerChannels,
    state: &mut WorkerState,
) {
    let t_start = Instant::now();
    if state.profile.is_some() {
        begin_allocations();
    }
    let frame_idx = state.frame_idx;
    let buffer_capacity = channels
        .audio_prod
        .as_ref()
        .map(|prod| prod.capacity().get())
        .unwrap_or(cfg.hop);
    let occupancy = channels
        .audio_prod
        .as_ref()
        .map(|prod| buffer_capacity.saturating_sub(prod.vacant_len()))
        .unwrap_or(0);

    let now_tick = state.timebase.frame_start_tick(state.frame_idx);
    let now_sec = state.timebase.tick_to_sec(now_tick);
    state.generator_model.advance_to(now_tick);
    if state.frame_idx == 0 || state.last_tick_log.elapsed() >= Duration::from_secs(1) {
        debug!(
            "[tick] frame_idx={} now_tick={} now_sec={:.6}",
            state.frame_idx, now_tick, now_sec
        );
        state.last_tick_log = Instant::now();
    }
    state.pop.set_current_frame(state.frame_idx);

    let analysis_wait_start = Instant::now();
    let analysis_updated = wait_for_analysis(state, &channels.analysis_result_rx, &cfg.exiting);
    let analysis_wait = analysis_wait_start.elapsed();
    let landscape_start = state.profile.as_ref().map(|_| Instant::now());
    apply_landscape_updates(state, channels, cfg, analysis_updated, now_tick, now_sec);
    let landscape_update = landscape_start
        .map(|start| start.elapsed())
        .unwrap_or_default();

    let listener_wait_start = Instant::now();
    let listener_state_update = wait_for_listener(state, channels, cfg);
    let listener_wait = listener_wait_start.elapsed();
    if let Some(listener_state) = listener_state_update {
        state.latest_listener_state = listener_state;
    }
    let listener_pressure = cfg.dcc_coupler.pressure(state.latest_listener_state);
    let listener_pressure_update =
        listener_state_update.map(|listener_state| cfg.dcc_coupler.pressure(listener_state));

    for (target, snapshot) in state
        .temporal_frames
        .iter_mut()
        .zip(&channels.temporal_snapshots)
    {
        if let Some(snapshot) = snapshot
            && let Ok(current) = snapshot.try_lock()
        {
            *target = *current;
        }
    }

    let population_start = state.profile.as_ref().map(|_| Instant::now());
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
        render_and_route_audio(state, channels, cfg, now_tick, phonation_count);
    let render_route = render_start
        .map(|start| start.elapsed())
        .unwrap_or_default();
    let post_render_start = state.profile.as_ref().map(|_| Instant::now());
    let second_ticks = cfg.fs as Tick;
    if state.reporter.is_some()
        && (now_tick == 0
            || now_tick / second_ticks != now_tick.saturating_sub(cfg.hop as Tick) / second_ticks)
    {
        for flora in &state.pop.flora {
            report_try(&mut state.reporter, "flora", |writer| {
                writer.write_flora(flora, f64::from(now_sec))
            });
        }
    }
    let mut loo_chunk =
        (!cfg.loo_populations.is_empty()).then(|| crate::core::stream::loo::PcmPacket {
            sources: std::mem::take(&mut state.schedule_renderer.loo_pcm),
            recycle: state.loo_recycle.0.clone(),
        });
    let channel_peak = [max_abs, max_abs];

    if cfg.temporal_meter_t2
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
            .failures
            .push(format!("analysis worker disconnected: {err}"));
        cfg.exiting.store(true, Ordering::SeqCst);
    }
    drive_production_meter(state, cfg, channels, habitat_chunk.as_ref());
    state.latest_listener_fast_state = Some(state.listener_twin.observe_presentation_audio(
        now_sec,
        state.frame_idx,
        presentation_chunk.as_ref(),
    ));
    state.listener_twin.observe_contour_audio(
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

    // Feed every habitat hop to NSGT-RT. Dropping hops breaks time continuity.
    if !cfg.temporal_meter_t2
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
            .failures
            .push(format!("analysis worker disconnected: {err}"));
        cfg.exiting.store(true, Ordering::SeqCst);
    }

    let finished_now =
        if state.pop.abort_requested && !cfg.render_prototype && !state.pop.has_placed_flora {
            true
        } else {
            state.scenario_end_tick.is_some()
                && state.schedule_renderer.is_idle()
                && state.pop.flora.is_empty()
        };
    if finished_now {
        state.playback_state = PlaybackState::Finished;
    }
    if finished_now && channels.wav_tx.is_some() {
        channels.wav_tx.take();
        info!("[t={:.6}] WAV closed.", state.current_time());
    }

    let should_send_ui = channels.ui_tx.is_some()
        && (finished_now || state.last_ui_update.elapsed() >= UI_MIN_INTERVAL);

    let peak_level = state.monitor.max_peak.max(max_abs);

    log_guard_meter(cfg, &mut state.last_guard_log, now_sec);

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
        state.finished = true;
        if !state.finish_logged {
            let note = if cfg.wait_user_exit {
                "Waiting for user exit."
            } else {
                "Exiting."
            };
            info!("[t={:.6}] Scenario finished. {note}", state.current_time());
            state.finish_logged = true;
        }
        if !cfg.wait_user_exit {
            cfg.exiting.store(true, Ordering::SeqCst);
        }
    }

    state.monitor.update(
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
        state.conductor.is_done(),
    );

    if !state.finished {
        state.frame_idx += 1;
    }
    // Include preceding telemetry in the hop cost, but exclude this record's own write.
    report_try(&mut state.reporter, "hop timing", |writer| {
        writer.write_hop_timing(&HopTimingSample {
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
            t2_meter: state.t2_sample,
            underrun_frames_total: cfg
                .underrun_frames
                .as_ref()
                .map(|count| count.load(Ordering::Relaxed)),
        });
    }
}
