use super::wiring::AnalysisRuntimeCore;
use super::*;
use crate::core::analysis_worker::AnalysisTemporalInput;
use crate::core::stream::loo::PcmPacket;
use crate::runtime_profile::{
    BackgroundTemporalProfile, BodyWorkerProfile, SharedWorkerProfile, TemporalMeterProfile,
};
use crate::temporal_cognition::{capture::BodyCapture, ui_observation::TemporalUi};

#[derive(Default)]
pub(super) struct TemporalConfig(bool);

impl TemporalConfig {
    pub(super) fn from_config(config: &AppConfig) -> Self {
        Self(config.temporal_meter_t2)
    }

    #[cfg(test)]
    pub(super) fn from_enabled(enabled: bool) -> Self {
        Self(enabled)
    }

    pub(super) fn t2_enabled(&self) -> bool {
        self.0
    }

    pub(super) fn make_meter(&self, fs: u32, hop: u64) -> MeterNetwork {
        if self.0 {
            MeterNetwork::new_continuous(fs, hop)
        } else {
            MeterNetwork::new()
        }
    }

    pub(super) fn observing(scenario: &Scenario) -> bool {
        scenario.temporal_mode == crate::scenario::TemporalMode::Observe
    }

    pub(super) fn supports_observe() -> bool {
        true
    }

    pub(super) fn validate_step(
        config: &AppConfig,
        scenario: &Scenario,
    ) -> Result<(), &'static str> {
        if config.temporal_meter_t2
            || config.temporal_t2.is_some()
            || config.temporal_onset_comparison.is_some()
            || config.temporal_body.is_some()
            || config.temporal_body_prototypes.is_some()
            || config.temporal_ridge.is_some()
            || config.temporal_acoustic.is_some()
            || config.temporal_period.is_some()
        {
            return Err("step execution does not support background temporal observation");
        }
        if scenario.temporal_mode == crate::scenario::TemporalMode::Observe {
            return Err("step execution does not support background temporal observation");
        }
        Ok(())
    }
}

#[derive(Default)]
pub(super) struct TemporalChannels {
    pub(in crate::runtime) t2_inbox: Option<crate::temporal_cognition::t2::runtime_input::Inbox>,
    pub(in crate::runtime) temporal_snapshots:
        [Option<Arc<std::sync::Mutex<crate::temporal_cognition::observation::Snapshot>>>; 2],
}

#[derive(Default)]
pub(super) struct TemporalObservation {
    pub(in crate::runtime) t2_sample: Option<crate::life::report::TemporalMeterSample>,
    pub(in crate::runtime) body_snapshot: Option<Box<crate::temporal_cognition::body::Snapshot>>,
    pub(in crate::runtime) temporal_frames:
        Box<[crate::temporal_cognition::observation::Snapshot; 2]>,
}

pub(super) struct TemporalWiring {
    pub(super) habitat: (bool, AnalysisTemporalInput),
    pub(super) presentation: (bool, AnalysisTemporalInput),
    pub(super) channels: TemporalChannels,
    pub(super) capture: BodyCapture,
}

impl TemporalWiring {
    #[cfg(feature = "native")]
    pub(super) fn prepare(
        config: &AppConfig,
        scenario: &Scenario,
        core: &AnalysisRuntimeCore,
        runtime_sample_rate: u32,
        deterministic_analysis: bool,
        _raw_t2_delay: Duration,
    ) -> Self {
        let hop = core.hop;
        let (raw_output, t2_inbox) = if config.temporal_meter_t2 {
            let (output, inbox) = crate::temporal_cognition::t2::runtime_input::channel(
                runtime_sample_rate,
                hop as u64,
            );
            let output = {
                #[cfg(test)]
                {
                    let mut output = output;
                    output.delay = _raw_t2_delay;
                    output
                }
                #[cfg(not(test))]
                output
            };
            (Some(output), Some(inbox))
        } else {
            (None, None)
        };
        let observing = TemporalConfig::observing(scenario);
        let arrival_enabled = config.temporal_onset_comparison.is_some_and(|c| c.arrival);
        let mut temporal_snapshots = [None, None];
        let taps = std::array::from_fn::<_, 2, _>(|bus| {
            (observing || ((arrival_enabled || config.temporal_t2.is_some()) && bus == 0)).then(
                || {
                    let tap = crate::temporal_cognition::observation::Tap::spawn(
                        bus as u8,
                        runtime_sample_rate,
                        hop,
                        core.nsgt.nfft(),
                        core.landscape.space.clone(),
                        crate::temporal_cognition::observation::Options {
                            body_prototypes: config.temporal_body_prototypes.as_ref().map(
                                |model| {
                                    let scales = config
                                        .temporal_body
                                        .expect("prototype model requires scales");
                                    (
                                        scales,
                                        crate::temporal_cognition::body_model::Prototypes::new(
                                            model, scales,
                                        ),
                                    )
                                },
                            ),
                            deterministic: deterministic_analysis,
                            ridge: config.temporal_ridge,
                            acoustic: config.temporal_acoustic,
                            period: config.temporal_period,
                            arrival_payload: config
                                .temporal_onset_comparison
                                .is_some_and(|c| c.arrival),
                            t2: config.temporal_t2,
                        },
                    );
                    temporal_snapshots[bus] = Some(Arc::clone(&tap.snapshot));

                    tap
                },
            )
        });
        let body_capture = if observing || arrival_enabled {
            config.temporal_body.map(|body| {
                crate::temporal_cognition::body::Capture::spawn(
                    core.nsgt.clone(),
                    body,
                    deterministic_analysis,
                    config.temporal_body_prototypes.as_ref().map(|model| {
                        crate::temporal_cognition::body_model::Prototypes::new(model, body)
                    }),
                )
            })
        } else {
            None
        };
        let [habitat, presentation] = taps;
        Self {
            habitat: (
                habitat.is_some(),
                AnalysisTemporalInput::new(habitat, raw_output),
            ),
            presentation: (
                presentation.is_some(),
                AnalysisTemporalInput::new(presentation, None),
            ),
            channels: TemporalChannels {
                t2_inbox,
                temporal_snapshots,
            },
            capture: BodyCapture::new(body_capture),
        }
    }

    #[cfg(not(feature = "native"))]
    pub(super) fn prepare(
        _: &AppConfig,
        _: &Scenario,
        _: &AnalysisRuntimeCore,
        _: u32,
        _: bool,
        _: Duration,
    ) -> Self {
        Self {
            habitat: Default::default(),
            presentation: Default::default(),
            channels: Default::default(),
            capture: Default::default(),
        }
    }
}

impl TemporalObservation {
    pub(super) fn initialize_capture(&mut self, capture: &BodyCapture) {
        self.body_snapshot = capture.as_ref().map(|capture| Box::new(capture.snapshot()));
    }

    #[cfg(feature = "native")]
    pub(super) fn set_initial_ui(&self, frame: &mut UiFrame) {
        frame.observation = TemporalUi::from_snapshots(None, &self.temporal_frames);
    }

    pub(super) fn ui_snapshot(&self) -> TemporalUi {
        TemporalUi::from_snapshots(self.body_snapshot.as_deref(), &self.temporal_frames)
    }

    pub(super) fn meter_profile(&self) -> TemporalMeterProfile {
        TemporalMeterProfile {
            t2_meter: self.t2_sample,
        }
    }

    #[cfg(feature = "native")]
    pub(super) fn background_profile(
        &self,
        channels: &TemporalChannels,
    ) -> BackgroundTemporalProfile {
        BackgroundTemporalProfile {
            shared: std::array::from_fn(|bus| {
                channels.temporal_snapshots[bus].as_ref().map(|_| {
                    let snapshot = &self.temporal_frames[bus];
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
            body: self
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
        }
    }
}

#[cfg(feature = "native")]
pub(super) fn read_initial(state: &mut WorkerState, channels: &WorkerChannels) {
    for (target, snapshot) in state
        .temporal
        .observation
        .temporal_frames
        .iter_mut()
        .zip(&channels.observation.temporal_snapshots)
    {
        if let Some(snapshot) = snapshot {
            *target = *snapshot.lock().expect("initial observation snapshot");
        }
    }
}

pub(super) fn read_live(state: &mut WorkerState, channels: &WorkerChannels) {
    for (target, snapshot) in state
        .temporal
        .observation
        .temporal_frames
        .iter_mut()
        .zip(&channels.observation.temporal_snapshots)
    {
        if let Some(snapshot) = snapshot
            && let Ok(current) = snapshot.try_lock()
        {
            *target = *current;
        }
    }
}

pub(super) fn refresh_body(state: &mut WorkerState) {
    if let Some(capture) = state.schedule_renderer.body_capture.as_ref() {
        let snapshot = capture.snapshot();

        if let Some(current) = state.temporal.observation.body_snapshot.as_mut()
            && (current.version != snapshot.version
                || current.capture_drops != snapshot.capture_drops
                || current.outside_voice_hops != snapshot.outside_voice_hops)
        {
            **current = snapshot;
            report_try(&mut state.reporter, "body observation", |writer| {
                writer.write_body_observation(&snapshot)
            });
        }
    }
}

pub(super) fn emit_observations(state: &mut WorkerState, now_sec: f64) {
    for observation in state.temporal.observation.temporal_frames.iter() {
        if observation.state != crate::temporal_cognition::observation::ObservationState::Off {
            report_try(&mut state.reporter, "temporal observation", |writer| {
                writer.write_temporal_observation(now_sec, observation)
            });
        }
    }
}

#[cfg(feature = "native")]
pub(super) fn finish_capture(state: &mut WorkerState) {
    if let Some(capture) = state.schedule_renderer.body_capture.as_mut() {
        capture.finish();
        let snapshot = capture.snapshot();
        if snapshot.state == crate::temporal_cognition::observation::ObservationState::Failed {
            state
                .lifecycle
                .failures
                .push("private body analysis worker failed".into());
        }

        if let Some(current) = state.temporal.observation.body_snapshot.as_mut() {
            **current = snapshot;
        }
        report_try(&mut state.reporter, "final body observation", |writer| {
            writer.write_body_observation(&snapshot)
        });
    }
}

#[cfg(feature = "native")]
pub(super) fn finish_analysis(
    state: &mut WorkerState,
    channels: &mut WorkerChannels,
    cfg: &WorkerConfig,
) {
    if (channels
        .observation
        .temporal_snapshots
        .iter()
        .any(Option::is_some)
        || channels.observation.t2_inbox.is_some())
        && channels.audio_to_analysis_tx.is_some()
    {
        // Close input and drain publishers before reading their final observer state.
        // This runs after generation has stopped, never in a live hop or callback.
        channels.audio_to_analysis_tx.take();
        channels.presentation_to_listener_tx.take();
        if let Some(inbox) = channels.observation.t2_inbox.as_mut() {
            loop {
                for _ in inbox.rx.try_iter() {}
                match channels.analysis_result_rx.try_recv() {
                    Ok(_) => {}
                    Err(crate::handoff::TryRecvError::Disconnected) => break,
                    Err(crate::handoff::TryRecvError::Empty) => {
                        thread::sleep(Duration::from_micros(200))
                    }
                }
            }
            inbox.finish();
            if let Some(sample) = state.temporal.observation.t2_sample.as_mut() {
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
            .temporal
            .observation
            .temporal_frames
            .iter_mut()
            .zip(&channels.observation.temporal_snapshots)
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
        state.ui.last_ui_update = Instant::now() - UI_MIN_INTERVAL;
        let now_tick = state.timebase.frame_start_tick(state.frame_idx);
        send_runtime_ui_frame(state, channels, cfg, Arc::from([]), 0.0, [0.0; 2], now_tick);
    }
}

pub(super) fn send_before_meter(
    state: &mut WorkerState,
    channels: &WorkerChannels,
    cfg: &WorkerConfig,
    habitat_chunk: &Arc<[f32]>,
    loo_chunk: &mut Option<PcmPacket>,
) {
    if cfg.temporal_config.t2_enabled()
        && let Err(err) = channels
            .audio_to_analysis_tx
            .as_ref()
            .expect("analysis input open during generation")
            .send((state.frame_idx, Arc::clone(habitat_chunk), loo_chunk.take()))
    {
        state
            .lifecycle
            .failures
            .push(format!("analysis worker disconnected: {err}"));
        cfg.exiting.store(true, Ordering::SeqCst);
    }
}

pub(super) fn update_meter(
    state: &mut WorkerState,
    channels: &mut TemporalChannels,
    cfg: &WorkerConfig,
    fast_known: bool,
    flux_drive: f32,
    drive: f32,
) -> Option<crate::core::meter::MeterState> {
    if let Some(inbox) = channels.t2_inbox.as_mut() {
        let start = state.timebase.frame_start_tick(state.frame_idx);
        let end = state.timebase.frame_end_tick(state.frame_idx);
        if let Some(meter) = state.temporal.flora_meter.as_mut() {
            if meter.needs_felt_bands() {
                let m = state.dorsal.last_metrics();
                meter.set_felt_bands(end, fast_known.then_some([m.e_low, m.e_mid, m.e_high]));
            }
            meter.continuous_fast(
                start,
                end,
                cfg.hop_duration.as_secs_f32(),
                fast_known.then_some(flux_drive),
            );
        }
        state.temporal.prod_meter.continuous_fast(
            start,
            end,
            cfg.hop_duration.as_secs_f32(),
            fast_known.then_some(drive),
        );
        let mut remaining = inbox.rx.len();
        let mut failure = None;
        loop {
            if !cfg.deterministic_analysis && remaining == 0 {
                break;
            }
            match inbox.rx.try_recv() {
                Ok(delivery) => {
                    let packet = inbox.received(delivery, end);
                    if let Some(meter) = state.temporal.flora_meter.as_mut()
                        && let Err(reason) = meter.continuous_ingest(packet)
                    {
                        failure = Some(reason);
                        break;
                    }
                    if let Err(reason) = state.temporal.prod_meter.continuous_ingest(packet) {
                        failure = Some(reason);
                        break;
                    }
                    remaining = remaining.saturating_sub(1);
                    if cfg.deterministic_analysis && packet.end == end {
                        break;
                    }
                }
                Err(crate::handoff::TryRecvError::Empty) => {
                    if !cfg.deterministic_analysis || cfg.exiting.load(Ordering::SeqCst) {
                        break;
                    }
                    #[cfg(feature = "native")]
                    thread::sleep(Duration::from_micros(200));
                    #[cfg(not(feature = "native"))]
                    {
                        failure = Some("inline T2 delivery is unsupported");
                        break;
                    }
                }
                Err(crate::handoff::TryRecvError::Disconnected) => {
                    failure = Some("raw T2 producer disconnected");
                    break;
                }
            }
        }
        if let Some(reason) = failure {
            cfg.exiting.store(true, Ordering::SeqCst);
            state.lifecycle.failures.push(reason.to_owned());
        }
        if let Some(meter) = state.temporal.flora_meter.as_mut() {
            state.temporal.flora_rhythm = crate::life::flora::TemporalObservation {
                observed_end: Some(end),
                known: fast_known && failure.is_none(),
                flux: flux_drive,
                state: meter.continuous_finish(),
            };
        }
        let result = state.temporal.prod_meter.continuous_finish();
        if state.reporter.is_some() || state.profile.is_some() {
            let sample = crate::life::report::TemporalMeterSample {
                sample_rate: cfg.fs as u32,
                hop: cfg.hop as u64,
                use_cut: end,
                input: inbox.diagnostics,
                recurrence: state
                    .temporal
                    .prod_meter
                    .continuous_diagnostics()
                    .expect("continuous mode"),
                state: result,
            };
            report_try(&mut state.reporter, "T2 meter", |writer| {
                writer.write_temporal_meter(&sample)
            });
            state.temporal.observation.t2_sample = Some(sample);
        }
        Some(result)
    } else {
        None
    }
}
