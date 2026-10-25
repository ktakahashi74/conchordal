use super::*;

/// Dispatch scenario events, collect phonation batches, and step the ecology.
/// Returns the number of valid batches in `state.phonation_batches_buf`.
pub(super) fn advance_population(
    state: &mut WorkerState,
    cfg: &WorkerConfig,
    now_tick: Tick,
    listener_pressure: ListenerPressure,
) -> usize {
    #[cfg(test)]
    crate::life::body_metabolism::assay::begin_hop(now_tick);
    state.conductor.dispatch_until(
        state.current_time(),
        &state.current_landscape,
        &mut state.pop,
    );
    for flora in &mut state.pop.flora {
        flora.observe_rhythm(state.flora_rhythm, now_tick, cfg.meter_shaping);
        flora.observe_external(&state.current_landscape, state.last_analysis_frame);
        flora.advance(&state.current_landscape);
    }
    if state.pop.rest_enabled {
        state.schedule_renderer.prepare_rest_sources(
            state
                .pop
                .voices
                .iter()
                .filter(|voice| voice.phonation_engine.tracks_tails())
                .map(|voice| (voice.id(), voice.metadata.generation)),
        );
    }
    if let Some(runtime) = state.pop.body_metabolism.as_mut() {
        runtime.prepare(&state.pop.voices, now_tick, &state.lparams);
    }
    if let Some(capture) = state.schedule_renderer.body_capture.as_mut() {
        capture.prepare(
            state
                .pop
                .voices
                .iter()
                .map(|v| (v.id(), v.metadata.generation, v.body_snapshot())),
            now_tick,
        );
    }
    if cfg.onset_comparison.is_some() {
        for voice in &mut state.pop.voices {
            voice
                .phonation_engine
                .set_onset_comparison(cfg.onset_comparison);
            voice
                .phonation_engine
                .set_decision_trace(state.reporter.is_some());
        }
        request_body_footprints(state, cfg.fs, now_tick);
    }
    if let Some(observer) = state.temporal_expectation.as_mut() {
        // Track from birth, including before a later switch into participation.
        state.schedule_renderer.prepare_self_sound(
            state.pop.voices.iter().map(|voice| voice.id()),
            now_tick,
            observer,
        );
        for voice in &mut state.pop.voices {
            let id = voice.id();
            if let Some(own) = state.schedule_renderer.self_sound_history(id) {
                voice
                    .phonation_engine
                    .observe_participation_context(own, |update| {
                        report_try(&mut state.reporter, "participation context", |writer| {
                            writer.write_participation_context(id, cfg.fs as u32, &update)
                        });
                    });
            }
        }
        let frame_end = now_tick.saturating_add(cfg.hop as Tick);
        let needs_forecast = state.pop.voices.iter().any(|voice| {
            voice
                .phonation_engine
                .needs_temporal_context(now_tick, frame_end)
        });
        let shared = needs_forecast.then(|| observer.forecast()).flatten();
        for voice in &mut state.pop.voices {
            if voice
                .phonation_engine
                .needs_temporal_context(now_tick, frame_end)
            {
                let mut local = shared;
                // Audit the unmodified habitat forecast; leave-self-out is a decision view.
                voice
                    .phonation_engine
                    .set_outcome_forecast(state.reporter.as_ref().and(shared.as_ref()));
                let mut own = state.schedule_renderer.self_sound_history(voice.id());
                if let (Some(local), Some(own)) = (local.as_mut(), own.as_deref_mut()) {
                    observer.use_external_energy(local, &mut own.external);
                }
                voice
                    .phonation_engine
                    .set_temporal_context(local.as_ref(), own.map_or([0.0; 3], |own| own.profile));
            }
        }
    }
    let phonation_count = if state.scenario_end_tick.is_none() {
        state.pop.collect_phonation_batches_into(
            &mut state.generator_model,
            &state.current_landscape,
            now_tick,
            cfg.onset_comparison
                .filter(|c| c.arrival)
                .map(|_| &state.temporal_frames[0]),
            state.body_snapshot.as_deref(),
            state.schedule_renderer.body_capture.as_ref(),
            &mut state.phonation_batches_buf,
        )
    } else {
        0
    };

    if state.reporter.is_some() {
        for voice in &mut state.pop.voices {
            let id = voice.id();
            if let Some(gap) = &mut voice.phonation_engine.span_gap {
                if state.scenario_end_tick.is_none() && state.conductor.is_done() {
                    gap.finish(now_tick);
                }
                for transition in gap.trace.drain(..) {
                    report_try(&mut state.reporter, "span gap", |writer| {
                        writer.write_span_gap(
                            id,
                            voice.metadata.generation,
                            cfg.fs as u32,
                            &transition,
                        )
                    });
                }
            }
            state.participation_predictions.extend(
                voice
                    .phonation_engine
                    .drain_participation_predictions()
                    .map(|prediction| (id, prediction)),
            );
            state.participation_decisions.extend(
                voice
                    .phonation_engine
                    .drain_participation_decisions()
                    .map(|decision| (id, decision)),
            );
        }
    }

    if state.scenario_end_tick.is_none() && state.conductor.is_done() {
        state.scenario_end_tick = Some(now_tick);
        state.pop.voices.clear();
        for flora in &mut state.pop.flora {
            flora.release_at_scenario_end();
        }
        state.schedule_renderer.shutdown_at(now_tick);
    }

    state.pop.advance_with_listener_pressure(
        cfg.hop,
        state.frame_idx,
        cfg.hop_duration.as_secs_f32(),
        &state.current_landscape,
        listener_pressure,
    );
    state.pop.flora.retain(|flora| !flora.is_finished());
    state.pop.cleanup_dead(
        state.frame_idx,
        cfg.hop_duration.as_secs_f32(),
        state.conductor.is_done(),
        &state.current_landscape,
    );
    phonation_count
}

/// I11-1 §4.2: at most one representative-onset footprint request per Voice per hop.
/// The hop path carries the identity, the request and the reply; the projection
/// itself runs on its independent worker.
pub(super) fn request_body_footprints(state: &mut WorkerState, fs: f32, now: Tick) {
    let phase3 = state.schedule_renderer.phase3_enabled();
    let voices = &mut state.pop.voices;
    let Some(worker) = state.footprint_worker.as_mut() else {
        return;
    };
    worker.prepare(
        voices
            .iter()
            .map(|voice| (voice.id(), voice.metadata.generation, voice.body_snapshot())),
    );
    for voice in voices {
        if !voice.is_alive() {
            continue;
        }
        let Some(mut recipe) = voice.footprint_recipe(fs) else {
            continue;
        };
        recipe.renderer_phase3 = phase3;
        let Some(body_generation) = worker.body_generation((voice.id(), voice.metadata.generation))
        else {
            continue;
        };
        let identity =
            crate::life::onset_footprint::Identity::new(voice.id(), body_generation, &recipe);
        let resent = voice
            .phonation_engine
            .footprint_hop(identity, now, &recipe, |request| {
                worker.request_footprint(request)
            });
        if resent {
            worker.stats.footprint_resent += 1;
        }
    }
}

/// Route returned footprints to the Voice that asked for them (I11-1 §4.2).
/// A reply the full queue dropped releases the request, and the Voice resends it next hop.
pub(super) fn route_body_footprints(
    voices: &mut [crate::life::voice::Voice],
    worker: &mut crate::life::onset_footprint::Worker,
    reporter: &mut Option<JsonlReporter>,
    now: Tick,
) {
    let mut superseded = 0;
    for record in worker.drain_footprints(now) {
        let discarded = voices
            .iter_mut()
            .find(|voice| voice.id() == record.identity.source_id)
            .is_some_and(|voice| voice.phonation_engine.footprint_receive(&record));
        superseded += u64::from(discarded);
        report_try(reporter, "body footprint", |writer| {
            writer.write_body_footprint(&record, discarded)
        });
    }
    worker.stats.footprint_superseded += superseded;
    let mut released = 0;
    for identity in worker.drain_released_footprints() {
        if let Some(voice) = voices
            .iter_mut()
            .find(|voice| voice.id() == identity.source_id)
            && voice.phonation_engine.footprint_release(identity)
        {
            released += 1;
        }
    }
    worker.stats.footprint_released += released;
}

/// Emit per-hop JSONL report records. No-op without a reporter.
pub(super) fn emit_hop_reports(
    state: &mut WorkerState,
    cfg: &WorkerConfig,
    now_sec: f32,
    phonation_count: usize,
    listener_state_update: Option<ListenerState>,
    listener_pressure_update: Option<ListenerPressure>,
) {
    if state.reporter.is_none() {
        return;
    }
    for (voice_id, decision) in state.participation_decisions.drain(..) {
        report_try(&mut state.reporter, "participation decision", |writer| {
            writer.write_participation_decision(voice_id, cfg.fs as u32, &decision)
        });
    }
    // Only reporting is decimated; each analysis hop has already advanced history.
    let report_hops = (0.1 * cfg.fs / cfg.hop as f32).ceil().max(1.0) as u64;
    if state.frame_idx.is_multiple_of(report_hops) {
        for observation in state.temporal_frames.iter() {
            if observation.state != crate::temporal_cognition::observation::ObservationState::Off {
                report_try(&mut state.reporter, "temporal observation", |writer| {
                    writer.write_temporal_observation(now_sec, observation)
                });
            }
        }
        for (bus, history) in [
            (
                "habitat",
                state.current_landscape.spectral_history.as_deref(),
            ),
            (
                "presentation",
                state.listener_twin.spectral_history.as_deref(),
            ),
        ] {
            if let Some(snapshot) = history {
                report_try(&mut state.reporter, "spectral history", |writer| {
                    writer.write_spectral_history(bus, state.frame_idx * cfg.hop as u64, snapshot)
                });
            }
        }
    }
    let onset_samples = onset_samples_from_batches(
        &state.pop.voices,
        &state.phonation_batches_buf[..phonation_count],
        cfg.fs,
        cfg.scaffold,
        state.frame_idx,
    );
    let runtime_events = state.pop.drain_runtime_events();
    let phonation_gate_open_events = state.pop.drain_phonation_gate_open_events();
    let death_records = state.pop.take_death_records();
    let active_population_ids = state.pop.active_population_ids();
    let population_steps = summarize_populations(
        &state.pop.voices,
        &active_population_ids,
        &state.current_landscape,
        now_sec,
    );
    let kuramoto = state.pop.kuramoto_order_parameter();
    let rhythm_observation = RhythmObservation {
        time_sec: now_sec,
        kuramoto_order_r: kuramoto.map(|(r, _)| r),
        kuramoto_active_count: kuramoto.map(|(_, n)| n).unwrap_or(0),
        onsets_in_hop: onset_samples.len(),
        theta_hz: positive_hz(state.current_landscape.rhythm.theta.freq_hz),
        delta_hz: positive_hz(state.current_landscape.rhythm.delta.freq_hz),
        env_open: state.current_landscape.rhythm.env_open.clamp(0.0, 1.0),
        env_level: state.current_landscape.rhythm.env_level.clamp(0.0, 1.0),
        measure_hz: positive_hz(state.current_landscape.rhythm.measure.freq_hz),
        measure_phase: state.current_landscape.rhythm.measure.phase,
        measure_confidence: state.current_landscape.rhythm.measure.alpha,
        measure_ratio: state.current_landscape.rhythm.measure_ratio,
    };
    report_try(&mut state.reporter, "runtime events", |writer| {
        writer.write_runtime_events(&runtime_events)
    });
    report_try(&mut state.reporter, "phonation gate open", |writer| {
        writer.write_phonation_gate_opens(&phonation_gate_open_events)
    });
    report_try(&mut state.reporter, "deaths", |writer| {
        writer.write_deaths(&death_records, cfg.hop, cfg.fs)
    });
    report_try(&mut state.reporter, "onsets", |writer| {
        writer.write_onsets(&onset_samples)
    });
    report_try(&mut state.reporter, "population steps", |writer| {
        writer.write_population_steps(&population_steps)
    });
    report_try(&mut state.reporter, "rhythm observation", |writer| {
        writer.write_rhythm_observation(rhythm_observation)
    });
    let hstate = &state.current_landscape.perc_habituation_state_scan;
    let n = hstate.len().max(1);
    let mean_h = hstate.iter().sum::<f32>() / n as f32;
    let max_h = hstate.iter().copied().fold(0.0f32, f32::max);
    let mut mean_erosion = 0.0f32;
    for i in 0..state.current_landscape.consonance_field_score.len() {
        mean_erosion += state.current_landscape.consonance_field_score[i]
            - state.current_landscape.consonance_field_score_eff[i];
    }
    mean_erosion /= n as f32;
    let tracked_bin = state
        .current_landscape
        .consonance_field_score
        .iter()
        .enumerate()
        .fold(
            (0usize, f32::MIN),
            |(bi, bv), (i, &v)| {
                if v > bv { (i, v) } else { (bi, bv) }
            },
        )
        .0;
    let tracked_state = state.current_landscape.perc_habituation_state_scan[tracked_bin];
    let tracked_score = state.current_landscape.consonance_field_score[tracked_bin];
    let tracked_score_eff = state.current_landscape.consonance_field_score_eff[tracked_bin];
    report_try(&mut state.reporter, "habituation", |w| {
        w.write_habituation(
            now_sec,
            mean_h,
            max_h,
            mean_erosion,
            tracked_bin,
            tracked_state,
            tracked_score,
            tracked_score_eff,
        )
    });
    let now_tick = state.timebase.frame_start_tick(state.frame_idx);
    let second_ticks = state.timebase.sec_to_tick(1.0).max(1);
    if state.frame_idx == 0
        || now_tick / second_ticks != now_tick.saturating_sub(cfg.hop as Tick) / second_ticks
    {
        report_try(&mut state.reporter, "habituation scan", |writer| {
            writer.write_habituation_scan(now_sec, &state.current_landscape)
        });
    }
    if let Some(listener_state) = listener_state_update {
        let sample = ListenerStateSample {
            time_sec: listener_state.time_sec,
            generated_frame_id: listener_state.generated_frame_id,
            analysis_frame_id: listener_state.analysis_frame_id,
            analysis_lag_frames: listener_state.analysis_lag_frames,
            stability_level: listener_state.stability_level,
            resolvability_level: listener_state.resolvability_level,
            tension_level: listener_state.tension_level,
            attention_level: listener_state.attention_level,
            beat_hz: listener_state.beat_hz,
            beat_phase: listener_state.beat_phase,
            beat_confidence: listener_state.beat_confidence,
            subdivision_ratio: listener_state.subdivision_ratio,
            subdivision_confidence: listener_state.subdivision_confidence,
            measure_hz: listener_state.measure_hz,
            measure_ratio: listener_state.measure_ratio,
            measure_confidence: listener_state.measure_confidence,
        };
        report_try(&mut state.reporter, "listener state", |writer| {
            writer.write_listener_state(&sample)
        });
    }
    if let Some(pressure) = listener_pressure_update {
        report_try(&mut state.reporter, "dcc pressure", |writer| {
            writer.write_dcc_pressure(now_sec, pressure)
        });
    }
}

/// Render one hop and route it to the two mono buses:
///   presentation_chunk -> cpal output + wav + UI metering
///   habitat_chunk -> generator rhythm + auditory analysis
/// Returns both buses, their peak, and renderer time (including own observation and match reports).
pub(super) fn render_and_route_audio(
    state: &mut WorkerState,
    channels: &mut WorkerChannels,
    cfg: &WorkerConfig,
    now_tick: Tick,
    phonation_count: usize,
) -> (Arc<[f32]>, Arc<[f32]>, f32, Duration) {
    if let Some(runtime) = state.pop.body_metabolism.as_mut() {
        runtime.prepare(&state.pop.voices, now_tick, &state.lparams);
    }
    let synthesis_start = state.profile.as_ref().map(|_| Instant::now());
    if !cfg.loo_populations.is_empty() || state.pop.has_placed_flora {
        // Reuse one owned packet. Backpressure is included in the audio deadline.
        state.schedule_renderer.loo_pcm =
            match state.loo_recycle.1.recv_timeout(Duration::from_secs(1)) {
                Ok(sources) => sources,
                Err(error) => {
                    state
                        .failures
                        .push(format!("LOO capture recycling failed: {error}"));
                    cfg.exiting.store(true, Ordering::SeqCst);
                    Vec::new()
                }
            };
        let sources = &mut state.schedule_renderer.loo_pcm;
        sources.retain(|source| match source.kind {
            crate::core::stream::loo::SourceKind::Voice => state
                .pop
                .voices
                .iter()
                .any(|v| v.id() == source.id && v.metadata.generation == source.generation),
            crate::core::stream::loo::SourceKind::Flora => {
                state.pop.flora.iter().any(|f| f.id == source.id)
            }
        });
        for voice in &state.pop.voices {
            if cfg.loo_populations.contains(&voice.metadata.population_id)
                && !sources.iter().any(|s| {
                    s.kind == crate::core::stream::loo::SourceKind::Voice
                        && s.id == voice.id()
                        && s.generation == voice.metadata.generation
                })
            {
                sources.push(crate::core::stream::loo::SourcePcm {
                    kind: crate::core::stream::loo::SourceKind::Voice,
                    analyze_hr: true,
                    id: voice.id(),
                    generation: voice.metadata.generation,
                    habitat: vec![0.; cfg.hop],
                });
            }
        }
        for flora in &state.pop.flora {
            if !sources
                .iter()
                .any(|s| s.kind == crate::core::stream::loo::SourceKind::Flora && s.id == flora.id)
            {
                sources.push(crate::core::stream::loo::SourcePcm {
                    kind: crate::core::stream::loo::SourceKind::Flora,
                    analyze_hr: flora.needs_hr_observer(),
                    id: flora.id,
                    generation: 0,
                    habitat: vec![0.0; cfg.hop],
                });
            }
        }
    }
    let sources = state
        .pop
        .body_metabolism
        .as_mut()
        .map_or(&mut [][..], |runtime| runtime.pcm.as_mut_slice());
    let frame = state.schedule_renderer.render_with_source_pcm(
        &state.phonation_batches_buf[..phonation_count],
        now_tick,
        &state.current_landscape.rhythm,
        sources,
        &mut state.pop.flora,
        |id, start, window, matched| {
            report_try(&mut state.reporter, "local prediction match", |writer| {
                writer.write_local_prediction_match(id, cfg.fs as u32, start, window, matched)
            });
        },
    );
    if let Some(runtime) = state.pop.body_metabolism.as_mut() {
        runtime.observe(now_tick, frame.habitat, &state.lparams);
    }
    let synthesis_elapsed = synthesis_start
        .map(|start| start.elapsed())
        .unwrap_or_default();

    let mut max_p = 0.0f32;
    for &s in frame.presentation {
        let abs_s = s.abs();
        if abs_s > max_p {
            max_p = abs_s;
        }
    }

    if let Some(prod) = channels.audio_prod.as_mut() {
        let reset_underrun_frames = if state.frame_idx == 0 {
            cfg.underrun_frames.as_deref()
        } else {
            None
        };
        AudioOutput::push_samples(prod, frame.presentation, reset_underrun_frames);
    }

    let presentation_chunk: Arc<[f32]> = Arc::from(frame.presentation);
    let habitat_chunk: Arc<[f32]> = Arc::from(frame.habitat);
    if state.pop.rest_enabled {
        for source in &state.schedule_renderer.rest_sources {
            if let Some(at) = source.quiet_at
                && let Some(voice) = state
                    .pop
                    .voices
                    .iter_mut()
                    .find(|v| v.id() == source.id && v.metadata.generation == source.generation)
            {
                voice.phonation_engine.rest_tail_finished(at, cfg.fs);
                if let Some(gap) = &mut voice.phonation_engine.span_gap {
                    for transition in gap.trace.drain(..) {
                        report_try(&mut state.reporter, "span gap", |writer| {
                            writer.write_span_gap(
                                source.id,
                                source.generation,
                                cfg.fs as u32,
                                &transition,
                            )
                        });
                    }
                }
            }
        }
    }
    #[cfg(test)]
    if let Some(mut probe) = state.offline_body_probe.take() {
        probe(
            state,
            now_tick,
            phonation_count,
            [&habitat_chunk, &presentation_chunk],
        );
        state.offline_body_probe = Some(probe);
    }
    if let Some(capture) = state.schedule_renderer.body_capture.as_ref() {
        let snapshot = capture.snapshot();

        if let Some(current) = state.body_snapshot.as_mut()
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
    if let Some(worker) = state.footprint_worker.as_mut() {
        route_body_footprints(&mut state.pop.voices, worker, &mut state.reporter, now_tick);
    }

    if state.reporter.is_some() {
        // Drain after actual rendering, before a later hop can retire a Voice.
        state
            .schedule_renderer
            .drain_prediction_errors(|id, from, through, window, errors| {
                report_try(&mut state.reporter, "local prediction errors", |writer| {
                    writer.write_local_prediction_errors(
                        id,
                        cfg.fs as u32,
                        from,
                        through,
                        window,
                        errors,
                    )
                });
            });
    }
    if let Some(tx) = channels.wav_tx.as_ref()
        && let Err(err) = tx.send(Arc::clone(&presentation_chunk))
    {
        state
            .failures
            .push(format!("WAV render output disconnected: {err}"));
        cfg.exiting.store(true, Ordering::SeqCst);
    }
    if let Some(tx) = channels.presentation_to_listener_tx.as_ref() {
        if cfg.deterministic_analysis {
            // Never drop a presentation hop: the listener worker must
            // observe the same hop sequence every run. Blocking applies
            // backpressure but cannot deadlock, because the fixed-lag
            // consume above keeps the worker at most one frame behind.
            if let Err(err) = tx.send((state.frame_idx, Arc::clone(&presentation_chunk), None)) {
                state
                    .failures
                    .push(format!("listener analysis worker disconnected: {err}"));
                cfg.exiting.store(true, Ordering::SeqCst);
                channels.presentation_to_listener_tx = None;
            }
        } else {
            match tx.try_send((state.frame_idx, Arc::clone(&presentation_chunk), None)) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    state.latest_listener_state = None;
                    state.listener_twin.spectral_history = None;
                    state.last_listener_analysis_frame = None;
                    state.listener_min_valid_frame = state.frame_idx.saturating_add(1);
                    debug!("listener analysis backlog full; dropped presentation hop");
                }
                Err(TrySendError::Disconnected(_)) => {
                    warn!("listener analysis worker disconnected");
                    channels.presentation_to_listener_tx = None;
                    state.latest_listener_state = None;
                    state.listener_twin.spectral_history = None;
                }
            }
        }
    }

    (presentation_chunk, habitat_chunk, max_p, synthesis_elapsed)
}

pub(super) fn participation_outcome(
    voice_id: u64,
    sample_rate: u32,
    prediction: ParticipationPrediction,
    observation: Option<TemporalObservation>,
) -> Option<ParticipationOutcomeSample> {
    if observation.is_some_and(|o| o.end_frame <= prediction.target_start_frame) {
        return None;
    }
    let matched = observation.is_some_and(|o| {
        o.start_frame == prediction.target_start_frame && o.end_frame == prediction.target_end_frame
    });
    Some(ParticipationOutcomeSample {
        voice_id,
        sample_rate,
        prediction,
        status: if matched {
            "observed"
        } else if observation.is_some() {
            "input_gap"
        } else {
            "end_of_input"
        },
        observed_start_frame: observation.map(|o| o.start_frame),
        observed_end_frame: observation.map(|o| o.end_frame),
        observed_habitat_band_energy: observation.filter(|_| matched).map(|o| o.band_energy),
    })
}

/// Drive the production meter from habitat flux and weak onset reinforcement.
pub(super) fn drive_production_meter(
    state: &mut WorkerState,
    cfg: &WorkerConfig,
    channels: &mut WorkerChannels,
    habitat_chunk: &[f32],
) {
    if state.finished {
        return;
    }
    if let Some(observer) = state.temporal_expectation.as_mut() {
        let now = state.timebase.frame_start_tick(state.frame_idx);
        observer.process(now, habitat_chunk, |observation| {
            state
                .participation_predictions
                .retain(|(voice_id, prediction)| {
                    let Some(sample) = participation_outcome(
                        *voice_id,
                        cfg.fs as u32,
                        *prediction,
                        Some(observation),
                    ) else {
                        return true;
                    };
                    report_try(&mut state.reporter, "participation outcome", |writer| {
                        writer.write_participation_outcome(&sample)
                    });
                    false
                });
        });
    }
    let fast_known = !cfg.temporal_meter_t2 || habitat_chunk.iter().all(|v| v.is_finite());
    if fast_known {
        state.dorsal.process(habitat_chunk);
    }
    let flux_drive = (state.dorsal.last_metrics().flux.max(0.0) * 500.0)
        .tanh()
        .clamp(0.0, 1.0);
    const ONSET_DRIVE_GAIN: f32 = 0.25;
    let onset_drive = (state
        .pop
        .last_habitat_onset_strength_in_hop()
        .unwrap_or(0.0)
        * ONSET_DRIVE_GAIN)
        .clamp(0.0, 1.0);
    let drive = flux_drive.max(onset_drive);
    let meter_state = if let Some(inbox) = channels.t2_inbox.as_mut() {
        let start = state.timebase.frame_start_tick(state.frame_idx);
        let end = state.timebase.frame_end_tick(state.frame_idx);
        if let Some(meter) = state.flora_meter.as_mut() {
            meter.continuous_fast(
                start,
                end,
                cfg.hop_duration.as_secs_f32(),
                fast_known.then_some(flux_drive),
            );
        }
        state.prod_meter.continuous_fast(
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
                    if let Some(meter) = state.flora_meter.as_mut()
                        && let Err(reason) = meter.continuous_ingest(packet)
                    {
                        failure = Some(reason);
                        break;
                    }
                    if let Err(reason) = state.prod_meter.continuous_ingest(packet) {
                        failure = Some(reason);
                        break;
                    }
                    remaining = remaining.saturating_sub(1);
                    if cfg.deterministic_analysis && packet.end == end {
                        break;
                    }
                }
                Err(crossbeam_channel::TryRecvError::Empty) => {
                    if !cfg.deterministic_analysis || cfg.exiting.load(Ordering::SeqCst) {
                        break;
                    }
                    thread::sleep(Duration::from_micros(200));
                }
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    failure = Some("raw T2 producer disconnected");
                    break;
                }
            }
        }
        if let Some(reason) = failure {
            cfg.exiting.store(true, Ordering::SeqCst);
            state.failures.push(reason.to_owned());
        }
        if let Some(meter) = state.flora_meter.as_mut() {
            state.flora_rhythm = crate::life::flora::TemporalObservation {
                observed_end: Some(end),
                known: fast_known && failure.is_none(),
                flux: flux_drive,
                state: meter.continuous_finish(),
            };
        }
        let result = state.prod_meter.continuous_finish();
        if state.reporter.is_some() || state.profile.is_some() {
            let sample = crate::life::report::TemporalMeterSample {
                sample_rate: cfg.fs as u32,
                hop: cfg.hop as u64,
                use_cut: end,
                input: inbox.diagnostics,
                recurrence: state
                    .prod_meter
                    .continuous_diagnostics()
                    .expect("continuous mode"),
                state: result,
            };
            report_try(&mut state.reporter, "T2 meter", |writer| {
                writer.write_temporal_meter(&sample)
            });
            state.t2_sample = Some(sample);
        }
        result
    } else {
        if let Some(meter) = state.flora_meter.as_mut() {
            let known = habitat_chunk.iter().all(|v| v.is_finite());
            state.flora_rhythm = crate::life::flora::TemporalObservation {
                observed_end: Some(state.timebase.frame_end_tick(state.frame_idx)),
                known,
                flux: flux_drive,
                state: meter
                    .process_evidence(cfg.hop_duration.as_secs_f32(), known.then_some(flux_drive)),
            };
        }
        state
            .prod_meter
            .process(cfg.hop_duration.as_secs_f32(), drive)
    };
    state.current_landscape.rhythm = NeuralRhythms::from_meter_state(&meter_state);
}
