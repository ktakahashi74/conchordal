use super::*;

pub(super) struct RuntimeUiFrameInput<'a> {
    pub(super) body: Option<&'a crate::temporal_cognition::body::Snapshot>,
    pub(super) temporal: &'a [crate::temporal_cognition::observation::Snapshot; 2],
    pub(super) scenario_name: &'a str,
    pub(super) conductor: &'a Conductor,
    pub(super) pop: &'a Community,
    pub(super) generator_model: &'a crate::life::generator_model::GeneratorModel,
    pub(super) current_landscape: &'a LandscapeFrame,
    pub(super) log_space: &'a Log2Space,
    pub(super) presentation_chunk: Arc<[f32]>,
    pub(super) listener_fast_state: Option<ListenerFastState>,
    pub(super) listener_state: Option<ListenerState>,
    pub(super) current_time: f32,
    pub(super) playback_state: PlaybackState,
    pub(super) peak_level: f32,
    pub(super) channel_peak: [f32; 2],
    pub(super) now_tick: Tick,
    pub(super) hop: usize,
    pub(super) fs: f32,
}

pub(super) fn build_initial_ui_frame(
    scenario_name: &str,
    conductor: &Conductor,
    pop: &Community,
    current_landscape: &LandscapeFrame,
    current_time: f32,
    playback_state: PlaybackState,
    fs: f32,
) -> UiFrame {
    #[cfg(test)]
    UI_FRAME_BUILDS.with(|count| count.set(count.get() + 1));
    UiFrame {
        body: None,
        wave: WaveFrame {
            fs,
            samples: Arc::from(Vec::<f32>::new()),
        },
        spec: SpecFrame {
            spec_hz: current_landscape.space.centers_hz.clone(),
            amps: vec![0.0; current_landscape.space.n_bins()],
        },
        listener: ListenerFrame::default(),
        temporal: Default::default(),
        landscape: current_landscape.clone(),
        meta: SimulationMeta {
            time_sec: current_time,
            duration_sec: conductor.total_duration(),
            voice_count: pop.voices.len(),
            event_queue_len: conductor.remaining_events(),
            peak_level: 0.0,
            scenario_name: scenario_name.to_string(),
            scene_name: conductor.current_scene_name(current_time),
            playback_state,
            channel_peak: [0.0; 2],
            window_peak: [0.0; 2],
            kuramoto_order_r: None,
            kuramoto_active_count: 0,
            entrain_phases: Vec::new(),
        },
        prediction: PredictionFrame::default(),
        voices: Vec::new(),
    }
}

pub(super) fn build_runtime_ui_frame(input: RuntimeUiFrameInput<'_>) -> UiFrame {
    #[cfg(test)]
    UI_FRAME_BUILDS.with(|count| count.set(count.get() + 1));
    let wave = WaveFrame {
        fs: input.fs,
        samples: input.presentation_chunk,
    };
    let spec = SpecFrame {
        spec_hz: input.log_space.centers_hz.clone(),
        amps: input
            .current_landscape
            .nsgt_power
            .iter()
            .map(|&x| x.sqrt())
            .collect(),
    };
    let voices: Vec<VoiceStateInfo> = input
        .pop
        .voices
        .iter()
        .map(|agent| {
            let freq_hz = agent.body.base_freq_hz();
            VoiceStateInfo {
                id: agent.id,
                freq_hz,
                target_freq: 2.0f32.powf(agent.target_pitch_log2()),
                integration_window: agent.integration_window(),
                consonance: input.current_landscape.evaluate_pitch_level(freq_hz),
            }
        })
        .collect();
    let pred_stats = input.pop.last_pred_gate_stats();
    let (entrain_phases, kuramoto_order_r) = input.pop.entrain_phases_and_order();
    let theta_hz = positive_hz(input.current_landscape.rhythm.theta.freq_hz);
    let delta_hz = positive_hz(input.current_landscape.rhythm.delta.freq_hz);
    let (pred_tau_tick, pred_horizon_tick) = input
        .generator_model
        .predictor_tau_horizon_ticks(&input.current_landscape.rhythm);
    let frame_end = input.now_tick.saturating_add((input.hop as Tick).max(1));
    let pred_next_gate = input.generator_model.last_pred_next_gate();
    let pred_available_in_hop = pred_next_gate
        .as_ref()
        .is_some_and(|(gate_tick, _)| *gate_tick >= input.now_tick && *gate_tick < frame_end);
    let pred_c_field_level_next_gate = pred_next_gate.map(|(_, scan)| scan);

    UiFrame {
        wave,
        body: input.body.map(|b| Arc::new(*b)),
        temporal: Arc::new(*input.temporal),
        spec,
        listener: listener_frame_from_states(input.listener_fast_state, input.listener_state),
        landscape: input.current_landscape.clone(),
        meta: SimulationMeta {
            time_sec: input.current_time,
            duration_sec: input.conductor.total_duration(),
            voice_count: input.pop.voices.len(),
            event_queue_len: input.conductor.remaining_events(),
            peak_level: input.peak_level,
            scenario_name: input.scenario_name.to_string(),
            scene_name: input.conductor.current_scene_name(input.current_time),
            playback_state: input.playback_state,
            channel_peak: input.channel_peak,
            window_peak: [0.0; 2],
            kuramoto_order_r,
            kuramoto_active_count: entrain_phases.len(),
            entrain_phases,
        },
        prediction: PredictionFrame {
            next_gate_tick_est: input.generator_model.next_gate_tick_est,
            theta_hz,
            delta_hz,
            pred_n_theta_per_delta: Some(
                input
                    .generator_model
                    .predictor_n_theta_per_delta(&input.current_landscape.rhythm),
            ),
            pred_tau_tick: Some(pred_tau_tick),
            pred_horizon_tick: Some(pred_horizon_tick),
            pred_c_field_level_next_gate,
            pred_gain_raw_mean: pred_stats.map(|stats| stats.raw_mean),
            pred_gain_raw_min: pred_stats.map(|stats| stats.raw_min),
            pred_gain_raw_max: pred_stats.map(|stats| stats.raw_max),
            pred_gain_mixed_mean: pred_stats.map(|stats| stats.mixed_mean),
            pred_gain_mixed_min: pred_stats.map(|stats| stats.mixed_min),
            pred_gain_mixed_max: pred_stats.map(|stats| stats.mixed_max),
            pred_sync_mean: pred_stats.map(|stats| stats.sync_mean),
            gate_boundary_in_hop: input.pop.last_gate_boundary_in_hop(),
            pred_available_in_hop: Some(pred_available_in_hop),
            phonation_onsets_in_hop: input.pop.last_phonation_onsets_in_hop(),
        },
        voices,
    }
}

pub(super) fn positive_hz(hz: f32) -> Option<f32> {
    (hz.is_finite() && hz > 0.0).then_some(hz)
}

pub(super) fn listener_frame_from_states(
    fast_state: Option<ListenerFastState>,
    state: Option<ListenerState>,
) -> ListenerFrame {
    let mut frame = ListenerFrame::default();
    if let Some(fast) = fast_state {
        frame.time_sec = fast.time_sec;
        frame.generated_frame_id = fast.generated_frame_id;
        frame.attention_level = fast.attention_level;
        frame.attention = DorsalFrame {
            e_low: fast.attention_metrics.e_low,
            e_mid: fast.attention_metrics.e_mid,
            e_high: fast.attention_metrics.e_high,
            flux: fast.attention_metrics.flux,
        };
        frame.meter = fast.meter_state;
        frame.has_fast_state = fast.has_state;
    }
    if let Some(state) = state {
        frame.time_sec = state.time_sec;
        frame.generated_frame_id = state.generated_frame_id;
        frame.analysis_frame_id = state.analysis_frame_id;
        frame.analysis_lag_frames = state.analysis_lag_frames;
        frame.stability_level = state.stability_level;
        frame.resolvability_level = state.resolvability_level;
        frame.tension_level = state.tension_level;
        frame.has_state = true;
    }
    frame
}

pub(super) fn send_runtime_ui_frame(
    state: &mut WorkerState,
    channels: &WorkerChannels,
    cfg: &WorkerConfig,
    presentation_chunk: Arc<[f32]>,
    peak_level: f32,
    channel_peak: [f32; 2],
    now_tick: Tick,
) {
    let Some(tx) = &channels.ui_tx else {
        return;
    };
    let _ = tx.try_send(build_runtime_ui_frame(RuntimeUiFrameInput {
        body: state.body_snapshot.as_deref(),
        temporal: &state.temporal_frames,
        scenario_name: &cfg.scenario_name,
        conductor: &state.conductor,
        pop: &state.pop,
        generator_model: &state.generator_model,
        current_landscape: &state.current_landscape,
        log_space: &state.log_space,
        presentation_chunk,
        listener_fast_state: state.latest_listener_fast_state,
        listener_state: state.latest_listener_state,
        current_time: state.current_time(),
        playback_state: state.playback_state,
        peak_level,
        channel_peak,
        now_tick,
        hop: cfg.hop,
        fs: cfg.fs,
    }));
    state.last_ui_update = Instant::now();
}
