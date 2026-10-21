use super::*;

pub(super) fn analysis_ok(frame_idx: u64, last_analysis: Option<u64>, max_lag: u64) -> bool {
    if frame_idx == 0 {
        return true;
    }
    match last_analysis {
        Some(id) => frame_idx.saturating_sub(id) <= max_lag,
        None => false,
    }
}

/// Returns `(updated, unavailable, space_changed)`. `unavailable` means the analysis thread is
/// gone or invalidated: fixed-lag field analysis cannot generate the recovery
/// window while waiting, so the caller must end the run instead of waiting.
pub(super) fn merge_latest_analysis_results(
    analysis_result_rx: &Receiver<analysis_worker::AnalysisResult>,
    current_landscape: &mut Landscape,
    log_space: &mut Log2Space,
    generator_model: &mut crate::life::generator_model::GeneratorModel,
    lparams: &LandscapeParams,
    last_analysis_frame: &mut Option<u64>,
    frame_idx: u64,
) -> (bool, bool, bool) {
    let mut latest_audio: Option<(u64, Landscape)> = None;
    let mut unavailable = false;
    loop {
        match analysis_result_rx.try_recv() {
            Ok((analyzed_id, Some(frame))) => {
                *last_analysis_frame = Some(analyzed_id);
                latest_audio = Some((analyzed_id, frame));
            }
            Ok((_, None)) => {
                unavailable = true;
                current_landscape.spectral_history = None;
                *last_analysis_frame = None;
                latest_audio = None;
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                unavailable = true;
                break;
            }
        }
    }
    let Some((_analysis_id, frame)) = latest_audio else {
        return (false, unavailable, false);
    };

    let space_changed = current_landscape.space.n_bins() != frame.space.n_bins()
        || current_landscape.space.fmin != frame.space.fmin
        || current_landscape.space.fmax != frame.space.fmax
        || current_landscape.space.bins_per_oct != frame.space.bins_per_oct;
    if space_changed {
        current_landscape.resize_to_space(frame.space.clone());
        *log_space = current_landscape.space.clone();
        generator_model.set_space(log_space.clone());
    }
    // Keep frame perception metadata, but use the worker's current valuation.
    current_landscape.set_analysis_metadata(
        frame.roughness_kernel_params,
        frame.harmonicity_params,
        lparams.consonance_field_kernel(),
        frame.roughness_k,
        frame.roughness_ref_peak,
        frame.roughness_ref_eps,
    );
    current_landscape.roughness = frame.roughness;
    current_landscape.roughness_shape_raw = frame.roughness_shape_raw;
    current_landscape.roughness01 = frame.roughness01;
    current_landscape.harmonicity = frame.harmonicity;
    current_landscape.roughness01_scalar = frame.roughness01_scalar;
    current_landscape.loudness_mass = frame.loudness_mass;
    current_landscape.pitch_objective_mode = frame.pitch_objective_mode;
    current_landscape.subjective_intensity = frame.subjective_intensity;
    current_landscape.nsgt_power = frame.nsgt_power;
    current_landscape.spectral_history = frame.spectral_history;
    current_landscape.recompute_consonance(lparams);

    if cfg!(debug_assertions) && frame_idx.is_multiple_of(30) {
        let mut max_r = 0.0f32;
        let mut max_i = 0usize;
        for (i, &r) in current_landscape.roughness01.iter().enumerate() {
            if r.is_finite() && r > max_r {
                max_r = r;
                max_i = i;
            }
        }
        let h = current_landscape
            .harmonicity01
            .get(max_i)
            .copied()
            .unwrap_or(0.0);
        let r = current_landscape
            .roughness01
            .get(max_i)
            .copied()
            .unwrap_or(0.0);
        let c_score = current_landscape
            .consonance_field_score_eff
            .get(max_i)
            .copied()
            .unwrap_or(0.0);
        let c_level = current_landscape
            .consonance_field_level_eff
            .get(max_i)
            .copied()
            .unwrap_or(0.0);
        let (c_score_pred, c_level_pred) =
            compose_consonance_field_score_level_with_params(h, r, lparams);
        debug!(
            "c_score_check bin={} h={:.4} r={:.4} c_score={:.4} c_score_pred={:.4} c_level={:.4} c_level_pred={:.4}",
            max_i, h, r, c_score, c_score_pred, c_level, c_level_pred
        );
    }

    (true, unavailable, space_changed)
}

/// Advance a habituation field one hop from the landscape's raw views, then
/// write the effective views. No-op when disabled: `recompute_consonance`
/// already seeded the effective views bit-exact to raw.
pub(super) fn drive_and_apply_habituation(
    landscape: &mut Landscape,
    hab: &mut crate::core::habituation::HabituationField,
    lparams: &LandscapeParams,
    dt_sec: f32,
) {
    if !hab.is_enabled() {
        return;
    }
    hab.ensure_len(landscape.space.n_bins());
    let (proj, _max) = lparams
        .harmonicity_kernel
        .potential_h_from_log2_spectrum(&landscape.subjective_intensity, &landscape.space);
    hab.advance_from_parts(&landscape.consonance_field_level, &proj, dt_sec);
    landscape.apply_habituation(hab.state(), hab.theta(), &lparams.consonance_representation);
}

#[allow(clippy::too_many_arguments)]
pub(super) fn observe_listener_analysis_results(
    listener_result_rx: Option<&Receiver<analysis_worker::AnalysisResult>>,
    listener_twin: &mut ListenerTwin,
    lparams: &LandscapeParams,
    hab_listener: &mut crate::core::habituation::HabituationField,
    timebase: crate::core::timebase::Timebase,
    generated_frame_id: u64,
    min_valid_frame: u64,
    last_listener_analysis_frame: &mut Option<u64>,
) -> (Option<Option<ListenerState>>, bool) {
    let Some(rx) = listener_result_rx else {
        return (None, false);
    };
    let mut latest_state = None;
    let mut disconnected = false;
    // Finish the queued batch without chasing an active producer on the audio thread.
    let pending = rx.len().max(1);
    for _ in 0..pending {
        match rx.try_recv() {
            Ok((analyzed_id, Some(_))) if analyzed_id < min_valid_frame => continue,
            Ok((analyzed_id, frame)) => {
                *last_listener_analysis_frame = frame.as_ref().map(|_| analyzed_id);
                let Some(mut frame) = frame else {
                    listener_twin.spectral_history = None;
                    latest_state = Some(None);
                    continue;
                };
                frame.recompute_consonance(lparams);
                // Each observed hop advances memory once, including observed silence.
                drive_and_apply_habituation(
                    &mut frame,
                    hab_listener,
                    lparams,
                    timebase.hop as f32 / timebase.fs,
                );
                let analysis_time_sec = timebase.tick_to_sec(timebase.frame_end_tick(analyzed_id));
                latest_state = Some(Some(listener_twin.observe_presentation_landscape(
                    analysis_time_sec,
                    generated_frame_id,
                    analyzed_id,
                    &frame,
                )));
            }
            Err(TryRecvError::Empty) => break,
            // Same hazard as the main analysis path: under deterministic render the
            // caller waits on `analysis_ok`, so a dead listener thread would hang it.
            Err(TryRecvError::Disconnected) => {
                disconnected = true;
                break;
            }
        }
    }
    (latest_state, disconnected)
}

pub(super) fn apply_scaffold(
    rhythms: &mut crate::core::modulation::NeuralRhythms,
    scaffold: ScaffoldConfig,
    time_sec: f32,
    frame_idx: u64,
) {
    let Some(phase_0_1) = scaffold_phase_0_1(scaffold, time_sec, frame_idx) else {
        return;
    };
    let freq_hz = match scaffold {
        ScaffoldConfig::Off => return,
        ScaffoldConfig::Shared { freq_hz } | ScaffoldConfig::Scrambled { freq_hz, .. } => {
            freq_hz.max(0.0)
        }
    };
    rhythms.theta.freq_hz = freq_hz;
    rhythms.theta.phase = wrap_pm_pi(std::f32::consts::TAU * phase_0_1);
    rhythms.theta.mag = 1.0;
    rhythms.theta.alpha = 1.0;
    rhythms.theta.beta = 0.0;
    rhythms.env_open = 1.0;
    rhythms.env_level = 1.0;
}

/// Drain analysis results into the landscape.
/// Spec: landscape may lag analysis by <= MAX_LANDSCAPE_LAG_FRAMES; block until
/// the landscape is fresh enough or the run is ending.
pub(super) fn wait_for_analysis(
    state: &mut WorkerState,
    analysis_result_rx: &Receiver<analysis_worker::AnalysisResult>,
    exiting: &AtomicBool,
) -> bool {
    let mut analysis_updated = false;
    loop {
        let (merged, analysis_unavailable, space_changed) = merge_latest_analysis_results(
            analysis_result_rx,
            &mut state.current_landscape,
            &mut state.log_space,
            &mut state.generator_model,
            &state.lparams,
            &mut state.last_analysis_frame,
            state.frame_idx,
        );
        if space_changed {
            for flora in &mut state.pop.flora {
                flora.on_space_change(&state.log_space);
            }
        }
        if merged {
            analysis_updated = true;
        }

        if analysis_ok(
            state.frame_idx,
            state.last_analysis_frame,
            MAX_LANDSCAPE_LAG_FRAMES,
        ) {
            break;
        }
        if analysis_unavailable {
            state
                .failures
                .push("Field analysis disconnected or invalidated".to_owned());
            exiting.store(true, Ordering::SeqCst);
            break;
        }
        if exiting.load(Ordering::SeqCst) {
            break;
        }
        thread::sleep(Duration::from_micros(200));
    }
    analysis_updated
}

/// Apply pending kernel-parameter updates, advance ecology habituation, and
/// refresh the generator model's observed field and rhythm gate.
pub(super) fn apply_landscape_updates(
    state: &mut WorkerState,
    channels: &WorkerChannels,
    cfg: &WorkerConfig,
    analysis_updated: bool,
    now_tick: Tick,
    now_sec: f32,
) {
    let landscape_params_updated = apply_pending_landscape_update(
        &mut state.pop,
        &mut state.lparams,
        &mut state.current_landscape,
        &channels.analysis_update_tx,
        channels.listener_analysis_update_tx.as_ref(),
    );
    drive_and_apply_habituation(
        &mut state.current_landscape,
        &mut state.hab_ecology,
        &state.lparams,
        cfg.hop_duration.as_secs_f32(),
    );
    if (analysis_updated || landscape_params_updated)
        && let Some(analysis_id) = state.last_analysis_frame
    {
        // NSGT is right-aligned; the observed scan is stamped at the analysis frame end.
        let obs_tick = state.timebase.frame_end_tick(analysis_id);
        state.generator_model.observe_consonance_field_level(
            obs_tick,
            Arc::from(&state.current_landscape.consonance_field_level_eff[..]),
        );
    }

    apply_scaffold(
        &mut state.current_landscape.rhythm,
        cfg.scaffold,
        now_sec,
        state.frame_idx,
    );
    state
        .generator_model
        .update_gate_from_rhythm(now_tick, &state.current_landscape.rhythm);
}

/// Drain listener analysis results. Deterministic render mirrors the
/// main-analysis fixed-lag wait so the listener frame consumed at render-frame
/// N does not depend on worker scheduling. Real-time keeps the single
/// best-effort batch to avoid stalling the generator on listener-analysis
/// latency. Memory processes every received observation before publishing the
/// latest state; input backpressure still invalidates dropped presentation audio.
pub(super) fn wait_for_listener(
    state: &mut WorkerState,
    channels: &WorkerChannels,
    cfg: &WorkerConfig,
) -> Option<Option<ListenerState>> {
    let mut listener_state_update: Option<Option<ListenerState>> = None;
    loop {
        let (listener_state, listener_disconnected) = observe_listener_analysis_results(
            channels.listener_result_rx.as_ref(),
            &mut state.listener_twin,
            &state.lparams,
            &mut state.hab_listener,
            state.timebase,
            state.frame_idx,
            state.listener_min_valid_frame,
            &mut state.last_listener_analysis_frame,
        );
        if let Some(listener_state) = listener_state {
            listener_state_update = Some(listener_state);
        }
        if !cfg.deterministic_analysis || channels.listener_result_rx.is_none() {
            break;
        }
        if matches!(listener_state, Some(None)) {
            state
                .failures
                .push("Deterministic listener analysis invalidated".to_owned());
            cfg.exiting.store(true, Ordering::SeqCst);
            break;
        }
        if analysis_ok(
            state.frame_idx,
            state.last_listener_analysis_frame,
            MAX_LANDSCAPE_LAG_FRAMES,
        ) {
            break;
        }
        if listener_disconnected {
            state
                .failures
                .push("Listener analysis thread stopped delivering results".to_owned());
            cfg.exiting.store(true, Ordering::SeqCst);
            break;
        }
        if cfg.exiting.load(Ordering::SeqCst) {
            break;
        }
        thread::sleep(Duration::from_micros(200));
    }
    listener_state_update
}

pub(super) fn apply_pending_landscape_update(
    pop: &mut Community,
    params: &mut LandscapeParams,
    current_landscape: &mut Landscape,
    analysis_update_tx: &Sender<LandscapeUpdate>,
    listener_analysis_update_tx: Option<&Sender<LandscapeUpdate>>,
) -> bool {
    if let Some(update) = pop.take_pending_update() {
        let valuation_changed = update
            .roughness_aversion
            .is_some_and(|w| params.set_roughness_aversion(w));
        if let Some(mode) = update.pitch_objective_mode {
            current_landscape.pitch_objective_mode = mode;
        }
        if valuation_changed {
            current_landscape.consonance_kernel = params.consonance_field_kernel();
            current_landscape.recompute_consonance(params);
        }
        if let Err(err) = analysis_update_tx.send(update) {
            warn!("analysis update disconnected: {err}");
        }
        if let Some(tx) = listener_analysis_update_tx
            && let Err(err) = tx.send(update)
        {
            warn!("listener analysis update disconnected: {err}");
        }
        return valuation_changed;
    }
    false
}

pub(super) fn compose_consonance_field_score_level_with_params(
    h_state01: f32,
    r_state01: f32,
    params: &LandscapeParams,
) -> (f32, f32) {
    let c_score = params.consonance_field_kernel().score(h_state01, r_state01);
    let c_level = params.consonance_representation.level(c_score);
    (c_score, c_level)
}
