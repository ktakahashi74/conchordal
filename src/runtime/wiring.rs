use super::*;

#[cfg(feature = "native")]
pub(crate) struct RuntimeInit {
    pub(crate) ui_frame_rx: Option<Receiver<UiFrame>>,
    pub(crate) worker_handle: Option<std::thread::JoinHandle<Result<(), String>>>,
    pub(crate) analysis_handle: Option<std::thread::JoinHandle<()>>,
    pub(crate) listener_analysis_handle: Option<std::thread::JoinHandle<()>>,
    pub(crate) start_flag: Arc<AtomicBool>,
    pub(crate) audio_out: Option<AudioOutput>,
    pub(crate) audio_init_error: Option<String>,
}

pub(super) struct AnalysisRuntimeCore {
    pub(super) fs: f32,
    pub(super) hop: usize,
    pub(super) hop_duration: Duration,
    pub(super) lparams: LandscapeParams,
    pub(super) nsgt: RtNsgtKernelLog2,
    pub(super) landscape: Landscape,
    pub(super) dorsal: DorsalStream,
}

pub(super) fn build_analysis_runtime_core(
    config: &AppConfig,
    runtime_sample_rate: u32,
) -> AnalysisRuntimeCore {
    let fs = runtime_sample_rate as f32;
    let space = crate::core::log2space::production_space(runtime_sample_rate)
        .expect("validated production sample rate");
    let lparams = LandscapeParams {
        fs,
        roughness_kernel: RoughnessKernel::new(KernelParams::default(), 0.005),
        harmonicity_kernel: HarmonicityKernel::new(&space, HarmonicityParams::default()),
        consonance_kernel: ConsonanceKernel {
            a: config.psychoacoustics.consonance.field.kernel.a,
            b: config.psychoacoustics.consonance.field.kernel.b,
            c: config.psychoacoustics.consonance.field.kernel.c,
            d: config.psychoacoustics.consonance.field.kernel.d,
        },
        consonance_representation: ConsonanceRepresentationParams {
            beta: config.psychoacoustics.consonance.field.level.beta,
            theta: config.psychoacoustics.consonance.field.level.theta,
        },
        consonance_density_roughness_gain: config.psychoacoustics.consonance.density.roughness_gain,
        roughness_aversion: 1.0,
        harmonicity_preference: 1.0,
        habituation: crate::core::habituation::HabituationParams {
            enabled: config.psychoacoustics.habituation.enabled,
            satiation_sec: config.psychoacoustics.habituation.satiation_sec,
            recovery_sec: config.psychoacoustics.habituation.recovery_sec,
            ref_drive: config.psychoacoustics.habituation.ref_drive,
        },
        loudness_exp: config.psychoacoustics.loudness_exp,
        tau_ms: config.analysis.tau_ms,
        ref_power: 1e-4,
        roughness_k: config.psychoacoustics.roughness_k,
        roughness_ref_f0_hz: 1000.0,
        roughness_ref_sep_erb: 0.25,
        roughness_ref_mass_split: 0.5,
        roughness_ref_eps: 1e-12,
    };
    let nfft = config.analysis.nfft;
    let hop = config.analysis.hop_size;
    let overlap = 1.0 - (hop as f32 / nfft as f32);
    let power_mode = if config.psychoacoustics.use_incoherent_power {
        PowerMode::Incoherent
    } else {
        PowerMode::Coherent
    };
    let nsgt_kernel = NsgtKernelLog2::new(
        NsgtLog2Config {
            fs,
            overlap,
            nfft_override: Some(nfft),
            kernel_align: config.analysis.kernel_align,
        },
        space,
        None,
        power_mode,
    );
    let nsgt = RtNsgtKernelLog2::with_config(nsgt_kernel.clone(), RtConfig::default());
    let hop_duration = Duration::from_secs_f32(hop as f32 / fs);
    let mut landscape = Landscape::new(nsgt.space().clone());
    landscape.consonance_representation = lparams.consonance_representation;
    let dorsal = DorsalStream::new(fs);

    AnalysisRuntimeCore {
        fs,
        hop,
        hop_duration,
        lparams,
        nsgt,
        landscape,
        dorsal,
    }
}

#[cfg(feature = "native")]
pub(super) fn spawn_analysis_worker(
    name: &'static str,
    analysis_stream: AnalysisStream,
    audio_to_analysis_rx: Receiver<analysis_worker::AnalysisHop>,
    analysis_result_tx: Sender<analysis_worker::AnalysisResult>,
    analysis_update_rx: Receiver<LandscapeUpdate>,
    delivery: analysis_worker::AnalysisDelivery,
    temporal_inputs: (
        Option<crate::temporal_cognition::observation::Tap>,
        Option<crate::temporal_cognition::t2::runtime_input::Output>,
    ),
) -> thread::JoinHandle<()> {
    thread::Builder::new()
        .name(name.into())
        .spawn(move || {
            let (temporal_tap, raw_output) = temporal_inputs;
            analysis_worker::run(
                analysis_stream,
                audio_to_analysis_rx,
                analysis_result_tx,
                analysis_update_rx,
                delivery,
                temporal_tap,
                raw_output,
            )
        })
        .expect("spawn analysis worker")
}

pub(super) fn resolve_limiter_mode(config: &AppConfig) -> LimiterMode {
    match config.audio.limiter {
        crate::config::LimiterSetting::None => LimiterMode::None,
        crate::config::LimiterSetting::SoftClip => {
            LimiterMode::SoftClip(crate::audio::limiter::SoftClipParams::default())
        }
        crate::config::LimiterSetting::PeakLimiter => {
            LimiterMode::PeakLimiter(crate::audio::limiter::PeakLimiterParams::default())
        }
    }
}

/// Path-specific wiring knobs. Every field is an actual difference between the
/// GUI/headless path (`init_runtime`) and the offline render path (`run_render`).
pub(super) struct WiringOptions {
    #[cfg(feature = "native")]
    pub(super) score: Option<crate::scripting::score::ScriptSession>,
    pub(super) end_request: Arc<AtomicBool>,
    pub(super) duration_sec: Option<f64>,
    #[cfg(test)]
    pub(super) offline_body_probe: Option<OfflineBodyProbe>,
    #[cfg(test)]
    pub(super) raw_t2_delay: Duration,
    pub(super) ui_tx: Option<Sender<UiFrame>>,
    /// Enables listener analysis even without DCC coupling (GUI display / report).
    pub(super) listener_forced: bool,
    pub(super) wait_user_exit: bool,
    pub(super) start_playing: bool,
    #[cfg(feature = "native")]
    pub(super) audio_prod: Option<ringbuf::HeapProd<f32>>,
    pub(super) wav_tx: Option<Sender<Arc<[f32]>>>,
    pub(super) reporter: Option<JsonlReporter>,
    pub(super) deterministic_analysis: bool,
    /// Offline render only: footprints return at the hop that asked, whatever the worker's
    /// pace. The instrument keeps wall-clock delivery, even with `--play=false`.
    pub(super) deterministic_footprints: bool,
    pub(super) guard_meter: Option<Arc<LimiterMeter>>,
    pub(super) underrun_frames: Option<Arc<AtomicU64>>,
    pub(super) reserve_runtime_ids_through: u64,
    pub(super) profile: Option<RunProfile>,
    pub(super) telemetry: bool,
    pub(super) audio_counters: Option<Arc<crate::audio::telemetry::AudioCallbackCounters>>,
}

#[cfg(feature = "native")]
pub(super) struct RuntimeWiring {
    pub(super) start_flag: Arc<AtomicBool>,
    pub(super) worker_handle: thread::JoinHandle<Result<(), String>>,
    pub(super) analysis_handle: thread::JoinHandle<()>,
    pub(super) listener_analysis_handle: Option<thread::JoinHandle<()>>,
}

#[cfg(test)]
thread_local! {
    pub(super) static DISABLE_GUI_SUPPORT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Wiring shared by both runtime paths: analysis core, channels, analysis
/// worker threads, Community/Conductor setup, and the worker thread spawn.
#[cfg(feature = "native")]
pub(super) fn wire_runtime(
    config: &AppConfig,
    runtime_sample_rate: u32,
    scenario_label: String,
    scenario: Scenario,
    stop_flag: Arc<AtomicBool>,
    opts: WiringOptions,
) -> Result<RuntimeWiring, String> {
    let prepared = prepare_runtime(
        config,
        runtime_sample_rate,
        scenario_label,
        scenario,
        stop_flag,
        opts,
        false,
    )?;
    let PreparedRuntime {
        cfg,
        channels,
        state,
        analysis_handle,
        listener_analysis_handle,
    } = prepared;
    let start_flag = cfg.start_flag.clone();
    let worker_handle = thread::Builder::new()
        .name("worker".into())
        .spawn(move || {
            // Observer identities and update phases belong to the worker thread.
            let state = state(&cfg);
            worker_loop(cfg, channels, state)
        })
        .expect("spawn worker");
    Ok(RuntimeWiring {
        start_flag,
        worker_handle,
        analysis_handle: analysis_handle.into_thread(),
        listener_analysis_handle: listener_analysis_handle.map(AnalysisExecution::into_thread),
    })
}

pub(super) struct PreparedRuntime<S = WorkerState> {
    pub(super) cfg: WorkerConfig,
    pub(super) channels: WorkerChannels,
    pub(super) state: S,
    pub(super) analysis_handle: AnalysisExecution,
    pub(super) listener_analysis_handle: Option<AnalysisExecution>,
}

pub(super) enum AnalysisExecution {
    #[cfg(feature = "native")]
    Thread(thread::JoinHandle<()>),
    #[cfg(feature = "native")]
    Inline(Box<dyn FnMut() -> bool>),
    #[cfg(not(feature = "native"))]
    Inline(Box<analysis_worker::AnalysisState>),
}

impl AnalysisExecution {
    #[cfg(feature = "native")]
    fn into_thread(self) -> thread::JoinHandle<()> {
        match self {
            Self::Thread(handle) => handle,
            Self::Inline(_) => unreachable!("native wiring"),
        }
    }
    #[cfg(feature = "native")]
    pub(super) fn step(&mut self) -> bool {
        match self {
            Self::Inline(step) => step(),
            #[cfg(feature = "native")]
            Self::Thread(_) => unreachable!("step wiring"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prepare_runtime(
    config: &AppConfig,
    runtime_sample_rate: u32,
    scenario_label: String,
    mut scenario: Scenario,
    stop_flag: Arc<AtomicBool>,
    opts: WiringOptions,
    inline: bool,
) -> Result<PreparedRuntime<impl FnOnce(&WorkerConfig) -> WorkerState + use<>>, String> {
    let WiringOptions {
        #[cfg(feature = "native")]
        score,
        end_request,
        duration_sec,
        #[cfg(test)]
        offline_body_probe,
        #[cfg(test)]
        raw_t2_delay,
        ui_tx,
        listener_forced,
        wait_user_exit,
        start_playing,
        #[cfg(feature = "native")]
        audio_prod,
        wav_tx,
        mut reporter,
        deterministic_analysis,
        deterministic_footprints,
        guard_meter,
        underrun_frames,
        reserve_runtime_ids_through,
        mut profile,
        telemetry,
        audio_counters,
    } = opts;

    let mut device_config = config.clone();
    device_config.audio.sample_rate = runtime_sample_rate;
    device_config.validate().map_err(|err| format!("{err:#}"))?;
    let core = build_analysis_runtime_core(config, runtime_sample_rate);
    let fs = core.fs;
    let hop = core.hop;
    let hop_duration = core.hop_duration;
    let expression_markers = scenario.expression.is_some();
    scenario.prepare_expression(fs, hop);
    if expression_markers {
        report_try(&mut reporter, "scene markers", |writer| {
            writer.write_scene_markers(&scenario.scene_markers)
        });
    }
    let lparams_runtime = core.lparams.clone();

    #[cfg(feature = "native")]
    let (analysis_update_tx, analysis_update_rx) = bounded::<LandscapeUpdate>(8);
    #[cfg(feature = "native")]
    let (audio_to_analysis_tx, audio_to_analysis_rx) = bounded::<analysis_worker::AnalysisHop>(64);
    #[cfg(feature = "native")]
    let (analysis_result_tx, analysis_result_rx) = bounded::<analysis_worker::AnalysisResult>(4);

    #[cfg(not(feature = "native"))]
    let analysis_update_tx = crate::handoff::Slot::new(8);
    #[cfg(not(feature = "native"))]
    let audio_to_analysis_tx = crate::handoff::Slot::new(1);
    #[cfg(not(feature = "native"))]
    let analysis_result_rx = crate::handoff::Slot::new(4);

    #[cfg(feature = "native")]
    let (raw_output, t2_inbox) = if config.temporal_meter_t2 {
        let (output, inbox) =
            crate::temporal_cognition::t2::runtime_input::channel(runtime_sample_rate, hop as u64);
        #[cfg(test)]
        let output = {
            let mut output = output;
            output.delay = raw_t2_delay;
            output
        };
        (Some(output), Some(inbox))
    } else {
        (None, None)
    };
    #[cfg(not(feature = "native"))]
    let t2_inbox = None;
    // NSGT-RT based audio analysis thread.
    let observing = scenario.temporal_mode == crate::scenario::TemporalMode::Observe;
    let arrival_enabled = config.temporal_onset_comparison.is_some_and(|c| c.arrival);
    let mut temporal_snapshots = [None, None];
    #[cfg(feature = "native")]
    let mut taps = std::array::from_fn::<_, 2, _>(|bus| {
        (observing || ((arrival_enabled || config.temporal_t2.is_some()) && bus == 0)).then(|| {
            let tap = crate::temporal_cognition::observation::Tap::spawn(
                bus as u8,
                runtime_sample_rate,
                hop,
                core.nsgt.nfft(),
                core.landscape.space.clone(),
                crate::temporal_cognition::observation::Options {
                    body_prototypes: config.temporal_body_prototypes.as_ref().map(|model| {
                        let scales = config
                            .temporal_body
                            .expect("prototype model requires scales");
                        (
                            scales,
                            crate::temporal_cognition::body_model::Prototypes::new(model, scales),
                        )
                    }),
                    deterministic: deterministic_analysis,
                    ridge: config.temporal_ridge,
                    acoustic: config.temporal_acoustic,
                    period: config.temporal_period,
                    arrival_payload: config.temporal_onset_comparison.is_some_and(|c| c.arrival),
                    t2: config.temporal_t2,
                },
            );
            temporal_snapshots[bus] = Some(Arc::clone(&tap.snapshot));

            tap
        })
    });
    #[cfg(feature = "native")]
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
    #[cfg(not(feature = "native"))]
    let taps: [Option<crate::temporal_cognition::observation::Tap>; 2] = [None, None];
    #[cfg(not(feature = "native"))]
    let body_capture: Option<crate::temporal_cognition::body::Capture> = None;
    let resource_power = scenario.end == crate::scenario::ScenarioEnd::Open
        || scenario
            .events
            .iter()
            .flat_map(|event| &event.actions)
            .any(|action| matches!(action, crate::scenario::Action::PlaceFlora { .. }));
    let display_or_report = ui_tx.is_some() || reporter.is_some();
    let mut analysis_stream = AnalysisStream::new(core.lparams.clone(), core.nsgt.clone())
        .with_nsgt_power(display_or_report || taps[0].is_some() || resource_power);
    let harmonic_relations_enabled = scenario.harmonic_relation_sources.is_some()
        || scenario
            .events
            .iter()
            .flat_map(|event| &event.actions)
            .any(|action| {
                let valid = |w: f64| w.is_finite() && w != 0. && (-1. ..=1.).contains(&w);
                match action {
                    Action::FollowHarmonicRelations { weight }
                    | Action::FollowHarmonicContinuation { weight } => valid(*weight),
                    Action::UpdateLandscape { update } => {
                        update.harmonic_relations_weight.is_some_and(valid)
                            || update.harmonic_continuation_weight.is_some_and(valid)
                    }
                    _ => false,
                }
            });
    let gui_support = ui_tx.is_some();
    #[cfg(test)]
    let gui_support = gui_support && !DISABLE_GUI_SUPPORT.get();
    let support_enabled = gui_support || scenario.end == crate::scenario::ScenarioEnd::Open || harmonic_relations_enabled || scenario.events.iter().flat_map(|event| &event.actions).any(|action| {
        matches!(action, crate::scenario::Action::SetSupportPreference { axis, value } if *axis < 3 && value.is_finite() && *value != 0. && (*axis != 2 || *value > 0.))
        || matches!(action, crate::scenario::Action::UpdateLandscape { update } if update.support_preference.iter().enumerate().any(|(axis, value)| value.is_some_and(|w| w.is_finite() && w != 0. && (axis != 2 || w > 0.))))
    });
    let support_geometry = support_enabled.then(|| {
        crate::core::harmonic_support::Geometry::new(
            core.nsgt.space(),
            &core.lparams.harmonicity_kernel,
        )
    });
    if let Some(geometry) = &support_geometry {
        analysis_stream.enable_support(
            geometry.clone(),
            harmonic_relations_enabled || scenario.end == crate::scenario::ScenarioEnd::Open,
        );
        if gui_support {
            analysis_stream.publish_support_for_display();
        }
        analysis_stream.apply_update(crate::core::landscape::LandscapeUpdate {
            support_preference: [Some(0.); 3],
            ..Default::default()
        });
    }

    let tonal_enabled = scenario.observe_tonal_context || scenario.events.iter().flat_map(|e| &e.actions).any(
        |a| matches!(a, Action::Spawn { spec, .. } if spec.control.pitch.follow_tonal_context > 0.),
    );
    let tonal_decoder = (tonal_enabled && core.fs == 48000.)
        .then(|| std::sync::Arc::new(crate::core::tonal_context::Decoder::new()));
    if let Some(decoder) = &tonal_decoder {
        analysis_stream.enable_tonal_context(decoder.clone());
    }
    if scenario.remember_pitch_set > 0. {
        analysis_stream.enable_pitch_set(scenario.remember_pitch_set);
    }
    if tonal_enabled && tonal_decoder.is_none() {
        tracing::warn!("follow_tonal_context requires 48000 Hz; queries remain unknown");
    }
    let onset_habituation_enabled = scenario.end == crate::scenario::ScenarioEnd::Open
        || scenario
            .events
            .iter()
            .flat_map(|event| &event.actions)
            .any(|action| match action {
                Action::Spawn { spec, .. } => spec.control.phonation.rest_on_habituation > 0.0,
                Action::PlaceFlora { spec, .. } => spec.rest_on_habituation > 0.0,
                _ => false,
            });
    let onset_period = scenario.meter_shaping.basin_hz.map_or(0.5, |(lo, hi)| {
        let (lo, hi) = crate::core::meter::sane_basin(lo, hi);
        2.0 / f64::from(lo + hi)
    });
    if onset_habituation_enabled {
        analysis_stream.enable_onset_habituation(onset_period);
        if !core.lparams.habituation.enabled {
            tracing::warn!(
                "rest_on_habituation is enabled but habituation is disabled; spans will not exit"
            );
        }
    }
    if resource_power {
        analysis_stream.enable_resource_power();
    }
    #[cfg(feature = "native")]
    let analysis_handle = start_analysis(
        inline,
        "field-analysis",
        analysis_stream,
        audio_to_analysis_rx,
        analysis_result_tx,
        analysis_update_rx,
        analysis_worker::AnalysisDelivery::Latest(analysis_result_rx.clone()),
        (taps[0].take(), raw_output),
    );

    #[cfg(not(feature = "native"))]
    let analysis_handle = AnalysisExecution::Inline(Box::new(analysis_worker::AnalysisState::new(
        analysis_stream,
        None,
        None,
    )));

    let dcc_coupler = DccCoupler::new(config.dcc);
    let listener_analysis_enabled =
        observing || listener_forced || dcc_coupler.coupling_strength() > 0.0;
    if let Some(profile) = profile.as_mut() {
        profile.listener_enabled = listener_analysis_enabled;
    }
    let (
        presentation_to_listener_tx,
        listener_result_rx,
        listener_analysis_update_tx,
        listener_analysis_handle,
    ) = if listener_analysis_enabled {
        // The listener-side research tap also consumes its bus's NSGT power.
        let mut listener_stream = AnalysisStream::new(core.lparams.clone(), core.nsgt.clone())
            .with_nsgt_power(display_or_report || taps[1].is_some());
        if let Some(decoder) = &tonal_decoder {
            listener_stream.enable_tonal_context(decoder.clone());
        }
        #[cfg(feature = "native")]
        {
            let (presentation_tx, presentation_rx) = bounded::<analysis_worker::AnalysisHop>(64);
            let (result_tx, result_rx) = bounded::<analysis_worker::AnalysisResult>(4);
            let (update_tx, update_rx) = bounded::<LandscapeUpdate>(8);
            let handle = start_analysis(
                inline,
                "listener-analysis",
                listener_stream,
                presentation_rx,
                result_tx,
                update_rx,
                analysis_worker::AnalysisDelivery::Ordered,
                (taps[1].take(), None),
            );
            (
                Some(presentation_tx),
                Some(result_rx),
                Some(update_tx),
                Some(handle),
            )
        }
        #[cfg(not(feature = "native"))]
        {
            (
                Some(crate::handoff::Slot::new(1)),
                Some(crate::handoff::Slot::new(4)),
                Some(crate::handoff::Slot::new(8)),
                Some(AnalysisExecution::Inline(Box::new(
                    analysis_worker::AnalysisState::new(listener_stream, None, None),
                ))),
            )
        }
    } else {
        (None, None, None, None)
    };

    let mut landscape = core.landscape;
    landscape.support_planned = support_enabled;
    landscape.support_ref_power = core.lparams.ref_power;
    landscape.support_loudness_exp = core.lparams.loudness_exp;
    if support_enabled {
        tracing::info!(
            "harmonic support: heard first-order fields integrated over cached body partials; no self-report uses a labeled single partial"
        );
    }
    let dorsal = core.dorsal;

    let mut pop = Community::new(crate::core::timebase::Timebase { fs, hop });
    if support_enabled {
        pop.prepare_support(core.nsgt.space().n_bins());
    }
    if let Some(settings) = config.body_metabolism.filter(|settings| settings.enabled) {
        pop.body_metabolism = Some(
            crate::life::body_metabolism::BodyMetabolism::new(
                core.nsgt.clone(),
                &core.lparams,
                settings,
            )
            .with_renderer_phase3(config.render_prototype),
        );
    }
    if config.birth_surrogate {
        pop.enable_birth_surrogate(
            core.lparams.loudness_exp,
            core.lparams.ref_power,
            core.lparams.consonance_representation,
            config.render_prototype,
        );
    }
    let scenario_max_id = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .filter_map(|action| match action {
            Action::Spawn { ids, .. } => ids.assigned().iter().copied().max(),
            _ => None,
        })
        .max()
        .unwrap_or(0);
    pop.reserve_runtime_ids_through(scenario_max_id.max(reserve_runtime_ids_through));
    pop.set_seed(scenario.seed);
    pop.onset_habituation_config =
        onset_habituation_enabled.then_some((core.lparams.habituation, onset_period));
    pop.set_control_update_mode(scenario.control_update_mode);
    if reporter.is_some() {
        pop.enable_auto_observe();
    }
    let scaffold = scenario.scaffold;
    let harmonic_rhythm = scenario.harmonic_rhythm;
    let meter_shaping = scenario.meter_shaping;
    // Reserve observers from birth, including populations enabled by a later patch.
    let mut loo_populations = Vec::new();
    for action in scenario.events.iter().flat_map(|event| &event.actions) {
        let population = match action {
            Action::Spawn {
                population_id,
                spec,
                ..
            } if spec.control.pitch.follow_tonal_context > 0.
                || spec.control.pitch.leave_self_out
                || matches!(
                    &spec.articulation,
                    crate::scenario::ArticulationCoreConfig::Entrain {
                        lifecycle: crate::scenario::lifecycle::LifecycleConfig::Sustain {
                            selection_environment: true,
                            ..
                        },
                        ..
                    }
                ) =>
            {
                Some(*population_id)
            }
            Action::UpdatePopulation {
                population_id,
                patch,
            } if patch.leave_self_out == Some(true) => Some(*population_id),
            _ => None,
        };
        if let Some(id) = population
            && !loo_populations.contains(&id)
        {
            loo_populations.push(id);
        }
    }
    // Accumulate acoustic history from run start, even for a later placement.
    let has_flora = scenario.end == crate::scenario::ScenarioEnd::Open
        || scenario
            .events
            .iter()
            .flat_map(|event| &event.actions)
            .any(|action| matches!(action, Action::PlaceFlora { .. }));
    let harmonic_relations = harmonic_relations_enabled.then(|| {
        let sources = scenario.harmonic_relation_sources.unwrap_or_else(|| {
            scenario
                .events
                .iter()
                .flat_map(|event| &event.actions)
                .map(|action| match action {
                    Action::Spawn { ids, .. } => ids.len(),
                    Action::PlaceFlora {
                        spec, extent_hz, ..
                    } => crate::life::flora::harmonic_source_bound(
                        &landscape.space,
                        *extent_hz,
                        spec.residual,
                    ),
                    Action::SetRespawnPolicy { capacity, .. } => *capacity,
                    _ => 0,
                })
                .sum::<usize>()
        });
        (
            support_geometry.clone().expect("temporal support geometry"),
            sources,
        )
    });
    let mut conductor =
        Conductor::from_scenario(scenario, crate::core::timebase::Timebase { fs, hop });

    #[cfg(feature = "native")]
    {
        conductor.score = score;
    }
    let start_flag = Arc::new(AtomicBool::new(start_playing));

    let cfg = WorkerConfig {
        harmonic_relations,
        has_flora,
        render_prototype: config.render_prototype,
        temporal_meter_t2: config.temporal_meter_t2,
        onset_comparison: config.temporal_onset_comparison,
        scenario_name: scenario_label,
        wait_user_exit,
        start_flag: start_flag.clone(),
        exiting: stop_flag,
        end_request,
        duration_tick: duration_sec
            .map(|sec| crate::core::timebase::Timebase { fs, hop }.sec_to_tick(sec)),
        scaffold,
        meter_shaping,
        harmonic_rhythm,
        guard_meter,
        underrun_frames,
        telemetry,
        audio_counters,
        dcc_coupler,
        hop,
        hop_duration,
        fs,
        deterministic_analysis,
        deterministic_footprints,
    };
    let channels = WorkerChannels {
        t2_inbox,
        temporal_snapshots,
        ui_tx,
        #[cfg(feature = "native")]
        audio_prod,
        wav_tx,
        audio_to_analysis_tx: Some(audio_to_analysis_tx),
        analysis_result_rx,
        analysis_update_tx,
        presentation_to_listener_tx,
        listener_result_rx,
        listener_analysis_update_tx,
    };

    let state = move |cfg: &WorkerConfig| {
        let mut state = WorkerState::new(
            pop,
            conductor,
            landscape,
            lparams_runtime,
            dorsal,
            reporter,
            cfg,
        );
        state.loo_populations = loo_populations;
        state.body_snapshot = body_capture.as_ref().map(|c| Box::new(c.snapshot()));
        state.schedule_renderer.body_capture = body_capture;
        state.profile = profile;
        #[cfg(test)]
        {
            state.offline_body_probe = offline_body_probe;
        }
        state
    };
    Ok(PreparedRuntime {
        cfg,
        channels,
        state,
        analysis_handle,
        listener_analysis_handle,
    })
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "native")]
fn start_analysis(
    inline: bool,
    name: &'static str,
    stream: AnalysisStream,
    input: Receiver<analysis_worker::AnalysisHop>,
    output: Sender<analysis_worker::AnalysisResult>,
    updates: Receiver<LandscapeUpdate>,
    delivery: analysis_worker::AnalysisDelivery,
    temporal: (
        Option<crate::temporal_cognition::observation::Tap>,
        Option<crate::temporal_cognition::t2::runtime_input::Output>,
    ),
) -> AnalysisExecution {
    #[cfg(feature = "native")]
    if !inline {
        return AnalysisExecution::Thread(spawn_analysis_worker(
            name, stream, input, output, updates, delivery, temporal,
        ));
    }
    {
        let mut step = analysis_worker::make_step(
            stream,
            input.clone(),
            output,
            updates,
            analysis_worker::AnalysisDelivery::Ordered,
            temporal.0,
            temporal.1,
        );
        AnalysisExecution::Inline(Box::new(move || input.try_recv().is_ok_and(&mut step)))
    }
}

#[cfg(feature = "native")]
pub(crate) fn init_runtime(
    args: crate::cli::Args,
    config: AppConfig,
    stop_flag: Arc<AtomicBool>,
) -> RuntimeInit {
    if let Err(err) = config.validate() {
        eprintln!("{err:#}");
        std::process::exit(1);
    }
    let path = Path::new(&args.scenario_path);
    let (scenario_label, scenario, output_gain, score) = load_scenario(
        path,
        args.seed,
        &config.audio,
        args.duration,
        &stop_flag,
        &args.end_request,
    )
    .unwrap_or_else(|err| {
        eprintln!("{err}");
        std::process::exit(1);
    });
    crate::life::modal::register_modal();
    let latency_ms = config.audio.latency_ms;
    let guard_mode = resolve_limiter_mode(&config);
    let guard_meter = if args.play {
        Some(Arc::new(LimiterMeter::default()))
    } else {
        None
    };

    // Complete the optional probe table before starting device callbacks.
    let prepare_tonal = scenario.observe_tonal_context || scenario.events.iter().flat_map(|e| &e.actions).any(
        |a| matches!(a, Action::Spawn { spec, .. } if spec.control.pitch.follow_tonal_context > 0.),
    );
    // Audio
    let (audio_out, audio_prod, mut audio_init_error) = if args.play {
        match AudioOutput::new(
            latency_ms,
            config.analysis.hop_size,
            guard_mode,
            guard_meter.clone(),
            output_gain,
            prepare_tonal,
        ) {
            Ok((out, prod)) => (Some(out), Some(prod), None),
            Err(e) => {
                let msg = e.to_string();
                eprintln!("Audio init failed: {msg}");
                if args.nogui {
                    std::process::exit(1);
                }
                (None, None, Some(msg))
            }
        }
    } else {
        (None, None, None)
    };

    // Decide runtime sample rate: use actual stream rate when playing, otherwise config value.
    let runtime_sample_rate: u32 = if args.play {
        let device_rate = audio_out
            .as_ref()
            .map(|out| out.config.sample_rate)
            .unwrap_or(config.audio.sample_rate);
        if device_rate != config.audio.sample_rate {
            debug!(
                "Runtime sample rate overridden by device: {} (config {})",
                device_rate, config.audio.sample_rate
            );
        }
        device_rate
    } else {
        config.audio.sample_rate
    };
    let (ui_tx, ui_frame_rx) = if args.nogui {
        (None, None)
    } else {
        let (tx, rx) = bounded(UI_CHANNEL_CAPACITY);
        (Some(tx), Some(rx))
    };

    if let Err(err) = validate_run_bounds(
        &scenario,
        args.duration,
        runtime_sample_rate,
        config.analysis.hop_size,
        false,
        args.report.is_some(),
        args.profile.is_some(),
    ) {
        eprintln!("{err}");
        std::process::exit(1);
    }
    let mut reporter = args
        .report
        .as_deref()
        .map(JsonlReporter::create)
        .transpose()
        .unwrap_or_else(|err| {
            eprintln!("{err}");
            std::process::exit(1);
        });
    report_try(&mut reporter, "meta", |writer| {
        writer.write_meta(scenario.seed)
    });
    if scenario.expression.is_none() {
        report_try(&mut reporter, "scene markers", |writer| {
            writer.write_scene_markers(&scenario.scene_markers)
        });
    }
    let observing = scenario.temporal_mode == crate::scenario::TemporalMode::Observe;
    let deterministic_analysis = (reporter.is_some() || observing) && !args.play;
    let profile = args.profile.as_deref().map(|path| {
        if let Some(report_path) = args.report.as_deref()
            && (Path::new(path) == Path::new(report_path)
                || Path::new(path)
                    .canonicalize()
                    .ok()
                    .is_some_and(|profile_path| {
                        Path::new(report_path).canonicalize().ok().as_ref() == Some(&profile_path)
                    }))
        {
            eprintln!("profile and report paths must differ");
            std::process::exit(1);
        }
        let coupling = config.dcc.coupling_strength;
        RunProfile::create(
            path,
            scenario.seed,
            reporter.is_some(),
            coupling,
            false,
            runtime_sample_rate,
            config.analysis.hop_size,
            audio_out
                .as_ref()
                .map(|out| (out.device_info(), out.counters())),
        )
        .unwrap_or_else(|err| {
            eprintln!("{err}");
            std::process::exit(1);
        })
    });

    let wiring = wire_runtime(
        &config,
        runtime_sample_rate,
        scenario_label,
        scenario,
        stop_flag.clone(),
        WiringOptions {
            score,
            end_request: args.end_request,
            duration_sec: args.duration,
            #[cfg(test)]
            offline_body_probe: None,
            #[cfg(test)]
            raw_t2_delay: Duration::ZERO,
            ui_tx,
            listener_forced: !args.nogui || args.report.is_some(),
            wait_user_exit: config.playback.wait_user_exit,
            start_playing: !config.playback.wait_user_start,
            #[cfg(feature = "native")]
            audio_prod,
            wav_tx: None,
            reporter,
            deterministic_analysis,
            deterministic_footprints: false,
            guard_meter,
            underrun_frames: audio_out.as_ref().map(AudioOutput::underrun_frames),
            reserve_runtime_ids_through: 0,
            profile,
            telemetry: args.telemetry,
            audio_counters: audio_out.as_ref().map(AudioOutput::counters),
        },
    )
    .unwrap_or_else(|error| {
        eprintln!("{error}");
        std::process::exit(1);
    });

    if prepare_tonal
        && let Some(out) = &audio_out
        && let Err(error) = out.start_prepared()
    {
        let message = error.to_string();
        eprintln!("Audio start failed: {message}");
        stop_flag.store(true, Ordering::SeqCst);
        if args.nogui {
            std::process::exit(1);
        }
        audio_init_error = Some(message);
    }

    RuntimeInit {
        ui_frame_rx,
        worker_handle: Some(wiring.worker_handle),
        analysis_handle: Some(wiring.analysis_handle),
        listener_analysis_handle: wiring.listener_analysis_handle,
        start_flag: wiring.start_flag,
        audio_out,
        audio_init_error,
    }
}

#[cfg(feature = "native")]
pub fn run_headless(args: crate::cli::Args, config: AppConfig, stop_flag: Arc<AtomicBool>) {
    let mut rt = init_runtime(args, config, stop_flag.clone());
    rt.start_flag.store(true, Ordering::SeqCst);
    while rt
        .worker_handle
        .as_ref()
        .is_some_and(|worker| !worker.is_finished())
        && !stop_flag.load(Ordering::SeqCst)
    {
        if let Some(audio) = rt.audio_out.as_mut()
            && let Err(err) = audio.supervise(Duration::from_millis(50))
        {
            eprintln!("{err}");
            stop_flag.store(true, Ordering::SeqCst);
        }
        thread::sleep(Duration::from_millis(50));
    }
    if let Err(err) = join_runtime(
        rt.worker_handle,
        [
            ("analysis", rt.analysis_handle),
            ("listener-analysis", rt.listener_analysis_handle),
        ],
    ) {
        eprintln!("{err}");
        std::process::exit(1);
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(feature = "native")]
pub fn run_render(
    scenario_path: &str,
    wav_path: String,
    config: AppConfig,
    stop_flag: Arc<AtomicBool>,
    seed: Option<u64>,
    report_path: Option<&str>,
    reserve_runtime_ids_through: u64,
    duration_sec: Option<f64>,
    end_request: Arc<AtomicBool>,
) -> Result<(), String> {
    if reserve_runtime_ids_through == u64::MAX {
        return Err("reserve-runtime-ids-through must be less than u64::MAX".to_string());
    }
    config.validate().map_err(|err| format!("{err:#}"))?;
    crate::life::modal::register_modal();

    let guard_mode = resolve_limiter_mode(&config);
    let guard_meter = Some(Arc::new(LimiterMeter::default()));

    let path = Path::new(scenario_path);
    let (scenario_label, scenario, output_gain, score) = load_scenario(
        path,
        seed,
        &config.audio,
        duration_sec,
        &stop_flag,
        &end_request,
    )?;
    validate_run_bounds(
        &scenario,
        duration_sec,
        config.audio.sample_rate,
        config.analysis.hop_size,
        true,
        report_path.is_some(),
        false,
    )?;
    let mut reporter = report_path.map(JsonlReporter::create).transpose()?;
    if let Some(writer) = reporter.as_mut() {
        writer.write_meta(scenario.seed)?;
        if scenario.expression.is_none() {
            writer.write_scene_markers(&scenario.scene_markers)?;
        }
    }

    // Render mode never opens an audio device: use configured sample-rate directly.
    let runtime_sample_rate = config.audio.sample_rate;
    let (wav_tx, wav_rx) = bounded::<Arc<[f32]>>(16);
    let wav_handle = crate::audio::writer::WavOutput::run(
        wav_rx,
        wav_path,
        runtime_sample_rate,
        guard_mode,
        guard_meter.clone(),
        output_gain,
    );

    let wiring = wire_runtime(
        &config,
        runtime_sample_rate,
        scenario_label,
        scenario,
        stop_flag,
        WiringOptions {
            score,
            end_request,
            duration_sec,
            #[cfg(test)]
            offline_body_probe: None,
            #[cfg(test)]
            raw_t2_delay: Duration::ZERO,
            ui_tx: None,
            listener_forced: reporter.is_some(),
            wait_user_exit: false,
            start_playing: true,
            audio_prod: None,
            wav_tx: Some(wav_tx),
            reporter,
            deterministic_analysis: true,
            deterministic_footprints: true,
            guard_meter,
            underrun_frames: None,
            reserve_runtime_ids_through,
            profile: None,
            telemetry: false,
            audio_counters: None,
        },
    )?;

    let runtime_result = join_runtime(
        Some(wiring.worker_handle),
        [
            ("analysis", Some(wiring.analysis_handle)),
            ("listener-analysis", wiring.listener_analysis_handle),
        ],
    );
    let wav_result = join_thread("wav", wav_handle).and_then(|result| result);
    match (runtime_result, wav_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(err), Ok(())) | (Ok(()), Err(err)) => Err(err),
        (Err(runtime), Err(wav)) => Err(format!("{runtime}; {wav}")),
    }
}

#[cfg(feature = "native")]
pub(crate) fn join_runtime(
    worker: Option<thread::JoinHandle<Result<(), String>>>,
    observers: impl IntoIterator<Item = (&'static str, Option<thread::JoinHandle<()>>)>,
) -> Result<(), String> {
    let mut failures = Vec::new();
    if let Some(handle) = worker
        && let Err(err) = join_thread("worker", handle).and_then(|result| result)
    {
        failures.push(err);
    }
    for (name, handle) in observers {
        if let Some(handle) = handle
            && let Err(err) = join_thread(name, handle)
        {
            failures.push(err);
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; "))
    }
}

#[cfg(feature = "native")]
pub(crate) fn join_thread<T>(name: &str, handle: thread::JoinHandle<T>) -> Result<T, String> {
    handle.join().map_err(|payload| {
        let detail = if let Some(msg) = payload.downcast_ref::<&str>() {
            (*msg).to_string()
        } else if let Some(msg) = payload.downcast_ref::<String>() {
            msg.clone()
        } else {
            "unknown panic payload".to_string()
        };
        format!("{name} thread failed: {detail}")
    })
}
