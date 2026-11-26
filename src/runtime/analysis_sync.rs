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
                if !frame.ecology.sites.is_empty() && analyzed_id >= frame_idx {
                    continue;
                }
                *last_analysis_frame = Some(analyzed_id);
                latest_audio = Some((analyzed_id, frame));
            }
            Ok((_, None)) => {
                unavailable = true;
                current_landscape.harmonic.support = None;
                for (_, site) in &mut current_landscape.ecology.sites {
                    let site = Arc::make_mut(site);
                    site.ecology.loo.clear();
                    site.ecology.flora_observations.clear();
                    site.ecology.rain_observations.clear();
                    site.ecology.tonal_observations.clear();
                    site.ecology.tonal_context = None;
                    site.acoustic.onset_habituation_known = false;
                    site.harmonic.support = None;
                    site.ecology.site_resource_power = None;
                }
                current_landscape.acoustic.onset_habituation_known = false;
                current_landscape.ecology.tonal_context = None;
                current_landscape.ecology.tonal_observations.clear();
                current_landscape.ecology.spectral_history = None;
                current_landscape.ecology.loo.clear();
                current_landscape.ecology.flora_observations.clear();
                current_landscape.ecology.rain_observations.clear();
                *last_analysis_frame = None;
                latest_audio = None;
            }
            Err(TryRecvError::Empty) => break,
            Err(TryRecvError::Disconnected) => {
                unavailable = true;
                current_landscape.harmonic.support = None;
                for (_, site) in &mut current_landscape.ecology.sites {
                    let site = Arc::make_mut(site);
                    site.ecology.loo.clear();
                    site.ecology.flora_observations.clear();
                    site.ecology.rain_observations.clear();
                    site.ecology.tonal_observations.clear();
                    site.ecology.tonal_context = None;
                    site.acoustic.onset_habituation_known = false;
                    site.harmonic.support = None;
                    site.ecology.site_resource_power = None;
                }
                current_landscape.acoustic.onset_habituation_known = false;
                current_landscape.ecology.tonal_context = None;
                current_landscape.ecology.tonal_observations.clear();
                break;
            }
        }
    }
    let Some((_analysis_id, mut frame)) = latest_audio else {
        return (false, unavailable, false);
    };

    let current_space = &current_landscape.acoustic.space;
    let incoming_space = &frame.acoustic.space;
    let space_changed = current_space.n_bins() != incoming_space.n_bins()
        || current_space.fmin != incoming_space.fmin
        || current_space.fmax != incoming_space.fmax
        || current_space.bins_per_oct != incoming_space.bins_per_oct;
    if space_changed {
        current_landscape.resize_to_space(incoming_space.clone());
        *log_space = current_landscape.acoustic.space.clone();
        generator_model.set_space(log_space.clone());
    }
    // Keep frame perception metadata, but use the worker's current valuation.
    frame.set_analysis_metadata(
        frame.acoustic.roughness_kernel_params,
        frame.acoustic.harmonicity_params,
        lparams.consonance_field_kernel(),
        frame.acoustic.roughness_k,
        frame.acoustic.roughness_ref_peak,
        frame.acoustic.roughness_ref_eps,
    );
    // Keep the active grid and worker-owned values. Revaluation below overwrites
    // H01 and the incoming consonance buffers before any consumer reads them.
    let acoustic = &mut current_landscape.acoustic;
    std::mem::swap(&mut acoustic.space, &mut frame.acoustic.space);
    *acoustic = crate::core::landscape::AcousticField {
        perc_habituation_state_scan: std::mem::take(&mut acoustic.perc_habituation_state_scan),
        consonance_theta: acoustic.consonance_theta,
        loudness_exp: acoustic.loudness_exp,
        ..frame.acoustic
    };
    current_landscape.pitch_objective_mode = frame.pitch_objective_mode;
    current_landscape.harmonic.support = frame.harmonic.support;
    current_landscape.ecology.sites = frame.ecology.sites;
    current_landscape.ecology.loo = frame.ecology.loo;
    current_landscape.ecology.tonal_context = frame.ecology.tonal_context;
    current_landscape.ecology.pitch_set = frame.ecology.pitch_set;
    current_landscape.ecology.pitch_set_weight = frame.ecology.pitch_set_weight;
    current_landscape.ecology.tonal_observations = frame.ecology.tonal_observations;
    current_landscape.ecology.flora_observations = frame.ecology.flora_observations;
    current_landscape.ecology.rain_observations = frame.ecology.rain_observations;
    current_landscape.ecology.spectral_history = frame.ecology.spectral_history;
    current_landscape.recompute_consonance(lparams);

    if cfg!(debug_assertions) && frame_idx.is_multiple_of(30) {
        let mut max_r = 0.0f32;
        let mut max_i = 0usize;
        for (i, &r) in current_landscape.acoustic.roughness01.iter().enumerate() {
            if r.is_finite() && r > max_r {
                max_r = r;
                max_i = i;
            }
        }
        let h = current_landscape
            .acoustic
            .harmonicity01
            .get(max_i)
            .copied()
            .unwrap_or(0.0);
        let r = current_landscape
            .acoustic
            .roughness01
            .get(max_i)
            .copied()
            .unwrap_or(0.0);
        let c_score = current_landscape
            .acoustic
            .consonance_field_score_eff
            .get(max_i)
            .copied()
            .unwrap_or(0.0);
        let c_level = current_landscape
            .acoustic
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
    hab.ensure_len(landscape.acoustic.space.n_bins());
    match hab.drive() {
        crate::core::habituation::HabituationDrive::RootProjection => {
            let (proj, _max) = lparams.harmonicity_kernel.potential_h_from_log2_spectrum(
                &landscape.acoustic.subjective_intensity,
                &landscape.acoustic.space,
            );
            hab.advance_from_parts(&landscape.acoustic.consonance_field_level, &proj, dt_sec);
        }
        _ => hab.advance_from_channels(
            &landscape.acoustic.subjective_intensity,
            &landscape.acoustic.space,
            &lparams.harmonicity_kernel.params,
            dt_sec,
        ),
    }
    for (_, local) in &mut landscape.ecology.sites {
        Arc::make_mut(local).apply_habituation(
            hab.state(),
            hab.theta(),
            &lparams.consonance_representation,
        );
    }
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
                    listener_twin.invalidate_prediction();
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
    tick: Tick,
    fs: f32,
    frame_idx: u64,
) {
    let Some(phase_0_1) = scaffold_phase_0_1(scaffold, tick, fs, frame_idx) else {
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
            &mut state.listener.last_analysis_frame,
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
            state.listener.last_analysis_frame,
            MAX_LANDSCAPE_LAG_FRAMES,
        ) {
            break;
        }
        if analysis_unavailable {
            state
                .lifecycle
                .failures
                .push("Field analysis disconnected or invalidated".to_owned());
            exiting.store(true, Ordering::SeqCst);
            break;
        }
        if exiting.load(Ordering::SeqCst) {
            break;
        }
        // Without threads the analysis ran inline, so a missing result is final.
        if !cfg!(feature = "native") {
            state
                .lifecycle
                .failures
                .push("inline analysis result is missing".into());
            exiting.store(true, Ordering::SeqCst);
            break;
        }
        #[cfg(feature = "native")]
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
) {
    let landscape_params_updated = apply_pending_landscape_update(
        &mut state.pop,
        &mut state.lparams,
        &mut state.current_landscape,
        &channels.analysis_update_tx,
        channels.listener_analysis_update_tx.as_ref().map(|tx| {
            (
                tx,
                &mut state.listener.pending_listener_update,
                cfg.deterministic_analysis,
            )
        }),
    );
    drive_and_apply_habituation(
        &mut state.current_landscape,
        &mut state.listener.hab_ecology,
        &state.lparams,
        cfg.hop_duration.as_secs_f32(),
    );
    if (analysis_updated || landscape_params_updated)
        && let Some(analysis_id) = state.listener.last_analysis_frame
    {
        // NSGT is right-aligned; the observed scan is stamped at the analysis frame end.
        let obs_tick = state.timebase.frame_end_tick(analysis_id);
        state.generator_model.observe_consonance_field_level(
            obs_tick,
            Arc::from(&state.current_landscape.acoustic.consonance_field_level_eff[..]),
        );
    }

    apply_scaffold(
        &mut state.current_landscape.rhythm,
        cfg.scaffold,
        now_tick,
        cfg.fs,
        state.frame_idx,
    );
    let shared = &mut state.current_landscape;
    for (_, local) in &mut shared.ecology.sites {
        let local = Arc::make_mut(local);
        local.rhythm = shared.rhythm;
        local.ecology.pitch_set = shared.ecology.pitch_set.clone();
        local.ecology.pitch_set_weight = shared.ecology.pitch_set_weight;
        local.harmonic.support_weights = shared.harmonic.support_weights;
    }
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
            &mut state.listener.listener_twin,
            &state.lparams,
            &mut state.listener.hab_listener,
            state.timebase,
            state.frame_idx,
            state.listener.listener_min_valid_frame,
            &mut state.listener.last_listener_analysis_frame,
        );
        if let Some(listener_state) = listener_state {
            listener_state_update = Some(listener_state);
        }
        if !cfg.deterministic_analysis || channels.listener_result_rx.is_none() {
            break;
        }
        if matches!(listener_state, Some(None)) {
            state
                .lifecycle
                .failures
                .push("Deterministic listener analysis invalidated".to_owned());
            cfg.exiting.store(true, Ordering::SeqCst);
            break;
        }
        if analysis_ok(
            state.frame_idx,
            state.listener.last_listener_analysis_frame,
            MAX_LANDSCAPE_LAG_FRAMES,
        ) {
            break;
        }
        if listener_disconnected {
            state
                .lifecycle
                .failures
                .push("Listener analysis thread stopped delivering results".to_owned());
            cfg.exiting.store(true, Ordering::SeqCst);
            break;
        }
        if cfg.exiting.load(Ordering::SeqCst) {
            break;
        }
        // Without threads the analysis ran inline, so a missing result is final.
        if !cfg!(feature = "native") {
            state
                .lifecycle
                .failures
                .push("inline listener result is missing".into());
            cfg.exiting.store(true, Ordering::SeqCst);
            break;
        }
        #[cfg(feature = "native")]
        thread::sleep(Duration::from_micros(200));
    }
    listener_state_update
}

pub(super) fn apply_pending_landscape_update(
    pop: &mut Community,
    params: &mut LandscapeParams,
    current_landscape: &mut Landscape,
    analysis_update_tx: &Sender<LandscapeUpdate>,
    listener: Option<(&Sender<LandscapeUpdate>, &mut Option<LandscapeUpdate>, bool)>,
) -> bool {
    let update = pop.take_pending_update();
    let mut valuation_changed = false;
    if let Some(update) = update {
        if let Some(weight) = update.harmonic_relations_weight
            && weight.is_finite()
            && (-1. ..=1.).contains(&weight)
        {
            current_landscape.harmonic.relations_weight = weight;
        }
        if let Some(weight) = update.harmonic_continuation_weight
            && weight.is_finite()
            && (-1. ..=1.).contains(&weight)
        {
            current_landscape.harmonic.continuation_weight = weight;
        }
        for (axis, value) in update.support_preference.iter().enumerate() {
            if let Some(value) = value {
                current_landscape.harmonic.support_weights.set(axis, *value);
            }
        }
        let roughness_changed = update
            .roughness_aversion
            .is_some_and(|w| params.set_roughness_aversion(w));
        let harmonicity_changed = update
            .harmonicity_preference
            .is_some_and(|w| params.set_harmonicity_preference(w));
        valuation_changed = roughness_changed || harmonicity_changed;
        if let Some(mode) = update.pitch_objective_mode {
            current_landscape.pitch_objective_mode = mode;
        }
        if valuation_changed {
            current_landscape.acoustic.consonance_kernel = params.consonance_field_kernel();
            current_landscape.recompute_consonance(params);
        }
        if let Err(err) = analysis_update_tx.send(update) {
            warn!("analysis update disconnected: {err}");
        }
    }
    // The real-time listener thread may be blocked sending results this worker has
    // not read yet, so a full queue keeps the merged update for the next hop.
    if let Some((tx, pending, deterministic)) = listener {
        if let Some(update) = update {
            pending.get_or_insert_default().merge(update);
        }
        if let Some(update) = pending.take() {
            if deterministic {
                if let Err(err) = tx.send(update) {
                    warn!("listener analysis update disconnected: {err}");
                }
            } else {
                match tx.try_send(update) {
                    Ok(()) => {}
                    Err(TrySendError::Full(update)) => *pending = Some(update),
                    Err(TrySendError::Disconnected(_)) => {
                        warn!("listener analysis update disconnected");
                    }
                }
            }
        }
    }
    valuation_changed
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

#[cfg(test)]
mod habituation_tests {
    use super::*;
    use crate::core::habituation::HabituationDrive;
    use crate::core::habituation::HabituationField;

    #[test]
    fn root_projection_drive_preserves_legacy_bits() {
        let core = super::super::wiring::build_analysis_runtime_core(&AppConfig::default(), 48000);
        let mut params = core.lparams;
        assert_eq!(params.habituation.drive, HabituationDrive::ChannelP);
        params.habituation.drive = HabituationDrive::RootProjection;
        let mut actual = core.landscape;
        let mut legacy = actual.clone();
        let n = actual.acoustic.space.n_bins();
        let mut field = HabituationField::new(&params.habituation, 0.0, n);
        let mut old_h = vec![0.0_f32; n];
        for step in 0..20 {
            for i in 0..n {
                let v = if step < 10 {
                    ((i * 17 + step) % 41) as f32 / 40.0
                } else {
                    0.0
                };
                actual.acoustic.subjective_intensity[i] = v;
                actual.acoustic.harmonicity01[i] = v;
                actual.acoustic.roughness01[i] = 0.2;
            }
            actual.recompute_consonance(&params);
            legacy.clone_from(&actual);
            let (proj, _) = params.harmonicity_kernel.potential_h_from_log2_spectrum(
                &legacy.acoustic.subjective_intensity,
                &legacy.acoustic.space,
            );
            let dt = 0.05_f32;
            let ae = (-dt / (5.0 / std::f32::consts::LN_10)).exp();
            let ar = (-dt / (8.0 / std::f32::consts::LN_10)).exp();
            for i in 0..n {
                let raw = (legacy.acoustic.consonance_field_level[i] * proj[i]).max(0.0);
                let d = (raw / (raw + 0.25)).clamp(0.0, 1.0);
                let a = if d > old_h[i] { ae } else { ar };
                old_h[i] = (a * old_h[i] + (1.0 - a) * d).clamp(0.0, 1.0);
            }
            legacy.apply_habituation(&old_h, 0.0, &params.consonance_representation);
            drive_and_apply_habituation(&mut actual, &mut field, &params, dt);
            for (got, expected) in [
                (
                    &actual.acoustic.perc_habituation_state_scan,
                    &legacy.acoustic.perc_habituation_state_scan,
                ),
                (
                    &actual.acoustic.consonance_field_score_eff,
                    &legacy.acoustic.consonance_field_score_eff,
                ),
                (
                    &actual.acoustic.consonance_field_level_eff,
                    &legacy.acoustic.consonance_field_level_eff,
                ),
                (
                    &actual.acoustic.consonance_density_mass_eff,
                    &legacy.acoustic.consonance_density_mass_eff,
                ),
            ] {
                assert!(
                    got.iter()
                        .zip(expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "profile-alloc")]
    fn channel_habituation_runtime_reuses_storage() {
        for drive in [HabituationDrive::ChannelP, HabituationDrive::ChannelQ] {
            let mut config = AppConfig::default();
            config.psychoacoustics.habituation.drive = drive;
            let core = super::super::wiring::build_analysis_runtime_core(&config, 48000);
            assert_eq!(core.lparams.habituation.drive, drive);
            let mut landscape = core.landscape;
            landscape.acoustic.subjective_intensity[100] = 1.0;
            landscape.recompute_consonance(&core.lparams);
            let mut field = HabituationField::new(
                &core.lparams.habituation,
                0.0,
                landscape.acoustic.space.n_bins(),
            );
            crate::runtime_profile::begin_allocations();
            for _ in 0..64 {
                drive_and_apply_habituation(
                    &mut landscape,
                    &mut field,
                    &core.lparams,
                    512.0 / 48000.0,
                );
            }
            let counts = crate::runtime_profile::finish_allocations().unwrap();
            assert_eq!(counts.count, 0);
            assert_eq!(counts.bytes, 0);
            assert!(
                landscape
                    .acoustic
                    .perc_habituation_state_scan
                    .iter()
                    .any(|&h| h > 0.0)
            );
        }
    }
}
