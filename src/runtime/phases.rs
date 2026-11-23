use super::temporal_ports;
#[cfg(feature = "research")]
#[path = "phases/felt_reports/active.rs"]
mod felt_reports;
#[cfg(not(feature = "research"))]
#[path = "phases/felt_reports/inactive.rs"]
mod felt_reports;

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
    #[cfg(feature = "research")]
    crate::life::body_metabolism::assay::begin_hop(now_tick);
    #[cfg(test)]
    if now_tick == 0 && std::env::var_os("HRY_EXECUTION").is_some() {
        state.schedule_renderer.harmonic_execution = Some(Vec::new());
    }
    if cfg.end_request.load(Ordering::SeqCst) {
        state.request_end(now_tick);
    }
    #[cfg(feature = "native")]
    if let Some(session) = &state.conductor.score {
        let stream = session.stream.clone();
        stream.publish_pace(now_tick, state.timebase);
        if let Err(error) = stream.wait_ready(
            now_tick,
            state.timebase,
            &cfg.exiting,
            &cfg.end_request,
            cfg.audio_counters.is_some(),
        ) {
            tracing::error!("{error}");
            state.lifecycle.failures.push(error);
        }
        if !state.lifecycle.script_error_reported {
            let error = match &stream.state.lock().unwrap().terminal {
                Some(crate::scripting::score::Terminal::Failed(error)) => Some(error.clone()),
                _ => None,
            };
            if let Some(error) = error {
                eprintln!("script: {:?}: {error}", error.position);
                state.lifecycle.script_error_reported = true;
            }
        }
        while let Some((item, charge)) = stream.pop_due(now_tick, state.timebase) {
            match item {
                crate::scripting::score::ScoreItem::Event(event) => {
                    for action in event.actions {
                        info!("[t={:.3}] Event: {action}", event.time);
                        if let Action::Spawn {
                            population_id,
                            spec,
                            ..
                        } = &action
                            && (spec.control.pitch.follow_tonal_context > 0.0
                                || spec.control.pitch.leave_self_out
                                || matches!(
                                    &spec.articulation,
                                    crate::scenario::ArticulationCoreConfig::Entrain {
                                        lifecycle:
                                            crate::scenario::lifecycle::LifecycleConfig::Sustain {
                                                selection_environment: true,
                                                ..
                                            },
                                        ..
                                    }
                                ))
                            && !state.loo_populations.contains(population_id)
                        {
                            state.loo_populations.push(*population_id);
                        }
                        crate::core::felt_config::FeltConfig::before_action(
                            &action,
                            state.temporal.flora_meter.as_mut(),
                        );
                        state.pop.apply_action(action, &state.current_landscape);
                    }
                }
                crate::scripting::score::ScoreItem::Marker(marker) => {
                    report_try(&mut state.reporter, "scene marker", |writer| {
                        writer.write_scene_markers(std::slice::from_ref(&marker))
                    });
                    state.conductor.current_scene = Some(marker.name);
                }
            }
            stream.return_credit(charge);
        }
    }
    let finish_dispatched =
        state
            .conductor
            .dispatch_until(now_tick, &state.current_landscape, &mut state.pop);
    if !finish_dispatched && cfg.duration_tick.is_some_and(|end| now_tick >= end) {
        state.request_end(now_tick);
    }
    let finish_due = finish_dispatched
        || state
            .lifecycle
            .end_sequence
            .as_ref()
            .is_some_and(|end| now_tick >= end.finish_tick);
    let wet_finish =
        state.schedule_renderer.room.is_some() && (finish_due || state.lifecycle.finish_begun);
    for receipt in state.schedule_renderer.phrase_receipts.drain(..) {
        if let Some(voice) = state.pop.voices.iter_mut().find(|v| {
            v.id() == receipt.record.voice_id && v.metadata.generation == receipt.record.generation
        }) && let Some(phrase) = &mut voice.phonation_engine.phrase_boundary
        {
            phrase.harmonic_recorder = state.current_landscape.harmonic.recorder.clone();
            phrase.trace_enabled = state.reporter.is_some();
            phrase.observe(&receipt, now_tick);
        }
    }
    for voice in &mut state.pop.voices {
        if crate::life::support_action::active(state.current_landscape.harmonic.support_weights) {
            voice.support.renderer_phase3 = cfg.render_prototype;
            voice.support.body = state.schedule_renderer.support_body(
                voice.id(),
                voice.metadata.generation,
                now_tick,
                &state.current_landscape.rhythm,
                &voice.body_snapshot(),
            );
        }
        if let Some(phrase) = &mut voice.phonation_engine.phrase_boundary {
            phrase.finish_publication(now_tick);
            for boundary in phrase.trace.drain(..) {
                report_try(&mut state.reporter, "phrase boundary", |writer| {
                    writer.write_phrase_boundary(&boundary)
                });
            }
        }
    }
    if let Some(clock) = &mut state.temporal.harmonic_clock {
        clock.advance(
            cfg.fs,
            now_tick,
            now_tick + cfg.hop as u64,
            state.temporal.flora_rhythm,
        );
        for voice in &mut state.pop.voices {
            voice.harmonic.temperature_bonus = listener_pressure.temperature_bonus;
            if voice.harmonic.member && finish_due {
                // Terminal shutdown must veto proposals before collection.
                voice.start_remove_fade(0.0);
            }
        }
        state.pop.harmonic_ticks.clear();
        state.pop.harmonic_ticks.extend_from_slice(&clock.ticks);
        report_try(&mut state.reporter, "harmonic clock", |writer| {
            writer.write_harmonic_clock(
                now_tick,
                state.temporal.flora_rhythm.observed_end,
                clock.observed,
                &clock.ticks,
            )
        });
    }
    for flora in &mut state.pop.flora {
        flora.harmonic.ticks = state.temporal.harmonic_clock.as_ref().map(|clock| {
            let mut ticks = flora.harmonic.ticks.take().unwrap_or_default();
            ticks.clear();
            ticks.extend_from_slice(&clock.ticks);
            ticks
        });
        let local = state.current_landscape.at_site(flora.site_id());
        flora.observe_onsets(local, now_tick);
        flora.observe_rhythm(state.temporal.flora_rhythm, now_tick, cfg.meter_shaping);
        flora.observe_external(local, state.listener.last_analysis_frame);
        if let Some(gap) = &mut flora.span_gap {
            gap.observe(local, now_tick);
        }
        flora.advance(local);
        if flora.is_finished()
            && let Some(gap) = &mut flora.span_gap
        {
            gap.finish(now_tick);
            for transition in gap.trace.drain(..) {
                report_try(&mut state.reporter, "Flora span gap", |writer| {
                    writer.write_flora_span_gap(flora.id, cfg.fs as u32, &transition)
                });
            }
        }
    }
    for rain in &mut state.pop.rain {
        rain.observe_meter(state.temporal.flora_rhythm, now_tick);
        if !wet_finish {
            rain.advance(state.current_landscape.at_site(rain.site_id()));
        }
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
    state
        .pop
        .body_metabolism
        .prepare(&state.pop.voices, now_tick, &state.lparams);
    state
        .schedule_renderer
        .body_capture
        .prepare(&state.pop.voices, now_tick);
    if cfg
        .onset
        .configure(&mut state.pop.voices, state.reporter.is_some())
    {
        state.temporal.onset.request(
            &mut state.pop.voices,
            cfg.fs,
            now_tick,
            state.schedule_renderer.prototype.enabled(),
        );
    }
    if let Some(observer) = state.temporal.temporal_expectation.as_mut() {
        if let Some(sites) = &mut state.schedule_renderer.sites {
            sites.align_energy(now_tick);
        }
        if state
            .schedule_renderer
            .room
            .as_ref()
            .is_some_and(|room| room.discontinuous(now_tick))
            || (state.schedule_renderer.sites.is_some() && observer.discontinuous(now_tick))
        {
            observer.align_to(now_tick);
            state
                .schedule_renderer
                .restart_self_histories(observer, now_tick);
        }
        // Track from birth, including before a later switch into participation.
        state.schedule_renderer.prepare_self_sound(
            state.pop.voices.iter().map(|voice| {
                (
                    voice.id(),
                    voice.metadata.generation,
                    voice.effective_control.body.site_id,
                )
            }),
            now_tick,
            observer,
        );
        for voice in &mut state.pop.voices {
            let id = voice.id();
            if let Some(own) = state
                .schedule_renderer
                .self_sound_history(id, voice.metadata.generation)
            {
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
                let spatial = state.schedule_renderer.sites.is_some();
                let mut own = state
                    .schedule_renderer
                    .self_sound_history(voice.id(), voice.metadata.generation);
                if own.as_ref().is_some_and(|h| !observer.aligned_history(h)) {
                    local = None;
                }
                if let (Some(local), Some(own)) = (local.as_mut(), own.as_deref_mut()) {
                    observer.use_external_energy(local, &mut own.external);
                }
                let outcome = if spatial {
                    local.as_ref()
                } else {
                    shared.as_ref()
                };
                voice
                    .phonation_engine
                    .set_outcome_forecast(state.reporter.as_ref().and(outcome));
                voice
                    .phonation_engine
                    .set_temporal_context(local.as_ref(), own.map_or([0.0; 3], |own| own.profile));
            }
        }
    }
    crate::life::phonation_engine::FeltVoice::set_hop_input(
        &mut state.pop.voices,
        state.temporal.flora_meter.as_ref(),
        &state.temporal.flora_rhythm,
        now_tick,
        state.reporter.is_some(),
    );
    let phonation_count = if !state.lifecycle.finish_begun && !wet_finish {
        state.pop.collect_phonation_batches_into(
            &mut state.generator_model,
            &state.current_landscape,
            now_tick,
            &cfg.onset.arrival(
                &state.temporal.observation,
                &state.schedule_renderer.body_capture,
            ),
            &mut state.phonation_batches_buf,
        )
    } else {
        0
    };

    if state.reporter.is_some() {
        for voice in &mut state.pop.voices {
            let id = voice.id();
            felt_reports::write_rhythm(&mut state.reporter, voice);
            for event in voice.harmonic.events.drain(..) {
                report_try(&mut state.reporter, "harmonic voice", |writer| {
                    writer.write_harmonic_voice(id, voice.metadata.generation, &event)
                });
            }
            for record in voice.pitch_ctl.pitch_set_trace.drain(..) {
                report_try(&mut state.reporter, "pitch set", |writer| {
                    writer.write_pitch_set(id, voice.metadata.generation, cfg.fs as u32, &record)
                });
            }
            for record in voice.pitch_ctl.tonal_trace.drain(..) {
                report_try(&mut state.reporter, "tonal context", |writer| {
                    writer.write_tonal_context(
                        id,
                        voice.metadata.generation,
                        cfg.fs as u32,
                        &record,
                    )
                });
            }
            if let Some(sequence) = &mut voice.pitch_ctl.own_sequence {
                report_try(&mut state.reporter, "own sequence", |writer| {
                    writer.write_own_sequence(
                        id,
                        voice.metadata.generation,
                        cfg.fs as u32,
                        sequence,
                        &mut voice.pitch_ctl.sequence_decisions,
                    )
                });
            }
            if let Some(me) = &mut voice.pitch_ctl.melodic {
                report_try(&mut state.reporter, "melodic continuity", |writer| {
                    writer.write_melodic_continuity(
                        id,
                        voice.metadata.generation,
                        cfg.fs as u32,
                        me,
                        voice.pitch_ctl.own_sequence.is_none(),
                    )
                });
            }
            if state.lifecycle.scenario_end_tick.is_none()
                && finish_due
                && let Some(mut record) = voice.pitch_ctl.dwell_state
            {
                record.event = "end";
                record.at = Some(now_tick);
                record.phase = record.remainder;
                report_try(&mut state.reporter, "dwell end", |writer| {
                    writer.write_dwell(id, voice.metadata.generation, cfg.fs as u32, &record)
                });
            }
            if let Some(gap) = &mut voice.phonation_engine.span_gap {
                if state.lifecycle.scenario_end_tick.is_none() && finish_due {
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
            state.temporal.participation_predictions.extend(
                voice
                    .phonation_engine
                    .drain_participation_predictions()
                    .map(|prediction| (id, voice.metadata.generation, prediction)),
            );
            state.temporal.onset.collect(id, voice);
        }
    }

    if finish_due {
        state.begin_finish(now_tick);
    }

    state.pop.advance_with_listener_pressure(
        cfg.hop,
        state.frame_idx,
        cfg.hop_duration.as_secs_f32(),
        &state.current_landscape,
        listener_pressure,
    );
    if state.reporter.is_some() {
        for voice in &mut state.pop.voices {
            let id = voice.id();
            for record in voice.pitch_ctl.dwell_trace.drain(..) {
                report_try(&mut state.reporter, "dwell", |writer| {
                    writer.write_dwell(id, voice.metadata.generation, cfg.fs as u32, &record)
                });
            }
            for record in voice.pitch_ctl.respond_trace.drain(..) {
                report_try(&mut state.reporter, "respond", |writer| {
                    writer.write_respond(id, voice.metadata.generation, cfg.fs as u32, &record)
                });
            }
        }
    }
    state.pop.flora.retain(|flora| !flora.is_finished());
    state.pop.rain.retain(|rain| !rain.is_finished());
    state.pop.cleanup_dead(
        state.frame_idx,
        cfg.hop_duration.as_secs_f32(),
        state.lifecycle.finish_begun,
        &state.current_landscape,
    );
    state.loo_populations.retain(|id| {
        state.pop.contains_population(*id) || state.conductor.has_pending_population(*id)
    });
    phonation_count
}

/// Emit per-hop JSONL report records. No-op without a reporter.
pub(super) fn emit_hop_reports(
    state: &mut WorkerState,
    cfg: &WorkerConfig,
    now_sec: f64,
    phonation_count: usize,
    listener_state_update: Option<ListenerState>,
    listener_pressure_update: Option<ListenerPressure>,
) {
    if state.reporter.is_none() {
        return;
    }
    state
        .temporal
        .onset
        .emit(&mut state.reporter, cfg.fs as u32);
    // Only reporting is decimated; each analysis hop has already advanced history.
    let report_hops = (0.1 * cfg.fs / cfg.hop as f32).ceil().max(1.0) as u64;
    if state.frame_idx.is_multiple_of(report_hops) {
        report_try(&mut state.reporter, "estimator status", |writer| {
            writer.write_estimator_status()
        });
        temporal_ports::emit_observations(state, now_sec);
        for (bus, history) in [
            (
                "habitat",
                state.current_landscape.ecology.spectral_history.as_deref(),
            ),
            (
                "presentation",
                state.listener.listener_twin.spectral_history.as_deref(),
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
    report_try(&mut state.reporter, "habitat sensation", |writer| {
        writer.write_habitat_sensation(now_sec, &state.current_landscape)
    });
    let now_tick = state.timebase.frame_start_tick(state.frame_idx);
    let second_ticks = state.timebase.sec_to_tick(1.0).max(1);
    if state.frame_idx == 0
        || now_tick / second_ticks != now_tick.saturating_sub(cfg.hop as Tick) / second_ticks
    {
        report_try(&mut state.reporter, "habituation scan", |writer| {
            writer.write_habituation_scan(now_sec, &state.current_landscape)
        });
        for rain in &state.pop.rain {
            report_try(&mut state.reporter, "rain field", |writer| {
                writer.write_rain_field(now_sec, rain, &state.current_landscape.space)
            });
        }
    }
    if let Some(listener_state) = listener_state_update {
        let sample = ListenerStateSample {
            prediction: listener_state.prediction,
            tonal_context: listener_state.tonal_context,
            tonal_context_end: listener_state.tonal_context_end,
            time_sec: listener_state.time_sec,
            generated_frame_id: listener_state.generated_frame_id,
            analysis_frame_id: listener_state.analysis_frame_id,
            analysis_lag_frames: listener_state.analysis_lag_frames,
            stability_level: listener_state.stability_level,
            nearby_improvement_level: listener_state.nearby_improvement_level,
            static_pull_level: listener_state.static_pull_level,
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
) -> Result<(Arc<[f32]>, Arc<[f32]>, f32, Duration), String> {
    state.schedule_renderer.harmonic_recorder = state.current_landscape.harmonic.recorder.clone();
    for flora in &mut state.pop.flora {
        flora.harmonic.recorder = state.current_landscape.harmonic.recorder.clone();
    }
    state
        .schedule_renderer
        .retain_phrase_sources(|id, generation| {
            state.pop.voices.iter().any(|v| {
                v.id() == id
                    && v.metadata.generation == generation
                    && v.phonation_engine.phrase_boundary.is_some()
            })
        });
    state
        .pop
        .body_metabolism
        .prepare(&state.pop.voices, now_tick, &state.lparams);
    let synthesis_start = state.profile.as_ref().map(|_| Instant::now());
    if state.schedule_renderer.sites.is_some()
        || !state.loo_populations.is_empty()
        || state.pop.has_placed_flora
        || state.pop.has_placed_rain_observer
    {
        // Reuse one owned packet. Backpressure is included in the audio deadline.
        #[cfg(feature = "native")]
        let recycled = state
            .loo_recycle
            .1
            .recv_timeout(Duration::from_secs(1))
            .map_err(|e| e.to_string());
        #[cfg(not(feature = "native"))]
        let recycled = state
            .loo_storage
            .take()
            .ok_or_else(|| "inline LOO storage is missing".to_string());
        state.schedule_renderer.loo_pcm = match recycled {
            Ok((sources, full)) => {
                state
                    .schedule_renderer
                    .receive_site_buffers(now_tick, &state.pop, full);
                sources
            }
            Err(error) => {
                state
                    .lifecycle
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
            crate::core::stream::loo::SourceKind::Rain => {
                state.pop.rain.iter().any(|r| r.id() == source.id)
            }
        });
        for voice in &state.pop.voices {
            if state
                .loo_populations
                .contains(&voice.metadata.population_id)
                && !sources.iter().any(|s| {
                    s.kind == crate::core::stream::loo::SourceKind::Voice
                        && s.id == voice.id()
                        && s.generation == voice.metadata.generation
                })
            {
                let pitch = &voice.effective_control.pitch;
                sources.push(crate::core::stream::loo::SourcePcm {
                    site_id: voice.effective_control.body.site_id,
                    kind: crate::core::stream::loo::SourceKind::Voice,
                    analyze_hr: true,
                    onset_history: pitch.respond_to_onsets > 0.0 && pitch.leave_self_out,
                    tonal_context: None,
                    id: voice.id(),
                    generation: voice.metadata.generation,
                    habitat: vec![0.; cfg.hop],
                });
            }
        }
        for source in sources
            .iter_mut()
            .filter(|s| s.kind == crate::core::stream::loo::SourceKind::Voice)
        {
            source.tonal_context = state
                .pop
                .voices
                .iter()
                .find(|v| v.id() == source.id && v.metadata.generation == source.generation)
                .filter(|v| v.effective_control.pitch.follow_tonal_context > 0.)
                .map(|v| v.effective_control.pitch.leave_self_out);
        }
        for flora in &state.pop.flora {
            if !sources
                .iter()
                .any(|s| s.kind == crate::core::stream::loo::SourceKind::Flora && s.id == flora.id)
            {
                sources.push(crate::core::stream::loo::SourcePcm {
                    site_id: flora.site_id(),
                    kind: crate::core::stream::loo::SourceKind::Flora,
                    analyze_hr: flora.needs_hr_observer(),
                    onset_history: state.schedule_renderer.sites.is_some()
                        && flora.needs_external_onsets(),
                    tonal_context: None,
                    id: flora.id,
                    generation: 0,
                    habitat: vec![0.0; cfg.hop],
                });
            }
        }
        for rain in state.pop.rain.iter().filter(|rain| rain.needs_observer()) {
            if !sources
                .iter()
                .any(|s| s.kind == crate::core::stream::loo::SourceKind::Rain && s.id == rain.id())
            {
                sources.push(crate::core::stream::loo::SourcePcm {
                    site_id: rain.site_id(),
                    kind: crate::core::stream::loo::SourceKind::Rain,
                    analyze_hr: true,
                    onset_history: false,
                    tonal_context: None,
                    id: rain.id(),
                    generation: 0,
                    habitat: vec![0.0; cfg.hop],
                });
            }
        }
    }
    let frame = state.schedule_renderer.render_with_source_pcm(
        &state.phonation_batches_buf[..phonation_count],
        now_tick,
        &state.current_landscape.rhythm,
        state.pop.body_metabolism.source_pcm(),
        &mut state.pop.flora,
        &mut state.pop.rain,
        |id, start, window, matched| {
            report_try(&mut state.reporter, "local prediction match", |writer| {
                writer.write_local_prediction_match(id, cfg.fs as u32, start, window, matched)
            });
        },
    )?;
    for flora in &mut state.pop.flora {
        if let Some(gap) = &mut flora.span_gap {
            for transition in gap.trace.drain(..) {
                report_try(&mut state.reporter, "Flora span gap", |writer| {
                    writer.write_flora_span_gap(flora.id, cfg.fs as u32, &transition)
                });
            }
        }
    }
    for rain in &mut state.pop.rain {
        let id = rain.id();
        for impact in rain.impact_trace.drain(..) {
            report_try(&mut state.reporter, "rain impact", |writer| {
                writer.write_rain_impact(
                    id,
                    now_tick + impact.offset as u64,
                    cfg.fs as u32,
                    &impact,
                )
            });
        }
    }
    state.pop.body_metabolism.observe_at_sites(
        now_tick,
        frame.habitat,
        frame.sites,
        &state.lparams,
    );
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

    #[cfg(feature = "native")]
    if let Some(prod) = channels.audio_prod.as_mut() {
        let reset_underrun_frames = if state.frame_idx == 0 {
            cfg.underrun_frames.as_deref()
        } else {
            None
        };
        AudioOutput::push_samples(
            prod,
            frame.stereo.unwrap_or(frame.presentation),
            reset_underrun_frames,
        );
    }

    let presentation_chunk: Arc<[f32]> = Arc::from(frame.presentation);
    let output_chunk = channels.wav_tx.as_ref().map(|_| {
        frame
            .stereo
            .map_or_else(|| Arc::clone(&presentation_chunk), Arc::from)
    });
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
    temporal_ports::refresh_body(state);

    #[cfg(feature = "native")]
    state
        .temporal
        .onset
        .route(&mut state.pop.voices, &mut state.reporter, now_tick);

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
    if let (Some(tx), Some(pcm)) = (channels.wav_tx.as_ref(), output_chunk)
        && let Err(err) = tx.send(pcm)
    {
        state
            .lifecycle
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
                    .lifecycle
                    .failures
                    .push(format!("listener analysis worker disconnected: {err}"));
                cfg.exiting.store(true, Ordering::SeqCst);
                channels.presentation_to_listener_tx = None;
            }
        } else {
            match tx.try_send((state.frame_idx, Arc::clone(&presentation_chunk), None)) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    state.listener.latest_listener_state = None;
                    state.listener.listener_twin.spectral_history = None;
                    state.listener.listener_twin.invalidate_prediction();
                    state.listener.last_listener_analysis_frame = None;
                    state.listener.listener_min_valid_frame = state.frame_idx.saturating_add(1);
                    debug!("listener analysis backlog full; dropped presentation hop");
                }
                Err(TrySendError::Disconnected(_)) => {
                    warn!("listener analysis worker disconnected");
                    channels.presentation_to_listener_tx = None;
                    state.listener.latest_listener_state = None;
                    state.listener.listener_twin.spectral_history = None;
                    state.listener.listener_twin.invalidate_prediction();
                }
            }
        }
    }

    for receipt in &state.schedule_renderer.phrase_receipts {
        report_try(&mut state.reporter, "phrase execution", |writer| {
            writer.write_phrase_execution(receipt)
        });
    }
    Ok((presentation_chunk, habitat_chunk, max_p, synthesis_elapsed))
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
    channels: &mut temporal_ports::TemporalChannels,
    habitat_chunk: &[f32],
) {
    if state.lifecycle.finished {
        return;
    }
    if let Some(observer) = state.temporal.temporal_expectation.as_mut() {
        let now = state.timebase.frame_start_tick(state.frame_idx);
        observer.process(now, habitat_chunk, |observation| {
            state.temporal.participation_predictions.retain(
                |(voice_id, generation, prediction)| {
                    let mut sample_observation = observation;
                    let local_energy = if state.schedule_renderer.sites.is_some() {
                        state
                            .schedule_renderer
                            .self_sound_history(*voice_id, *generation)
                            .and_then(|h| {
                                h.observed_external_energy(
                                    observation.start_frame,
                                    observation.end_frame,
                                )
                            })
                    } else {
                        Some(observation.band_energy)
                    };
                    if let Some(energy) = local_energy {
                        sample_observation.band_energy = energy;
                    }
                    let Some(mut sample) = participation_outcome(
                        *voice_id,
                        cfg.fs as u32,
                        *prediction,
                        Some(sample_observation),
                    ) else {
                        return true;
                    };
                    if local_energy.is_none() {
                        sample.status = "input_gap";
                        sample.observed_habitat_band_energy = None;
                    }
                    report_try(&mut state.reporter, "participation outcome", |writer| {
                        writer.write_participation_outcome(&sample)
                    });
                    false
                },
            );
        });
    }
    let fast_known =
        !cfg.temporal_config.t2_enabled() || habitat_chunk.iter().all(|v| v.is_finite());
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
    let meter_state = if let Some(state) =
        temporal_ports::update_meter(state, channels, cfg, fast_known, flux_drive, drive)
    {
        state
    } else {
        if let Some(meter) = state.temporal.flora_meter.as_mut() {
            let known = habitat_chunk.iter().all(|v| v.is_finite());
            if meter.needs_felt_bands() {
                let m = state.dorsal.last_metrics();
                meter.set_felt_bands(
                    state.timebase.frame_end_tick(state.frame_idx),
                    known.then_some([m.e_low, m.e_mid, m.e_high]),
                );
            }
            state.temporal.flora_rhythm = crate::life::flora::TemporalObservation {
                observed_end: Some(state.timebase.frame_end_tick(state.frame_idx)),
                known,
                flux: flux_drive,
                state: meter.process_at(
                    state.timebase.frame_start_tick(state.frame_idx),
                    cfg.hop,
                    cfg.fs,
                    known.then_some(flux_drive),
                ),
            };
        }
        state.temporal.prod_meter.process_at(
            state.timebase.frame_start_tick(state.frame_idx),
            cfg.hop,
            cfg.fs,
            Some(drive),
        )
    };
    felt_reports::write_beat(state, cfg);

    state.current_landscape.rhythm = NeuralRhythms::from_meter_state(&meter_state);
}
