use super::*;
use crate::life::social_density::SocialDensityTrace;

// The configured policy is the current source of the per-adoption decision.
pub(super) fn select_pitch_application(
    pitch: &crate::scenario::control::PitchControl,
    movement: Option<(&Log2Space, f32, f32)>,
) -> PitchApplyMode {
    if pitch.mode == PitchMode::Lock && pitch.pitch_apply_mode == PitchApplyMode::Rearticulate {
        PitchApplyMode::GateSnap
    } else if pitch.pitch_apply_mode == PitchApplyMode::Rearticulate
        && pitch.rearticulate_above_cents > 0.0
        && let Some((space, reference, target)) = movement
        && onset_distance_cents(space, reference, target) <= pitch.rearticulate_above_cents as f64
    {
        PitchApplyMode::Glide
    } else {
        pitch.pitch_apply_mode
    }
}

pub(super) fn onset_distance_cents(space: &Log2Space, a: f32, b: f32) -> f64 {
    if let (Some(i), Some(j)) = (space.index_of_log2(a), space.index_of_log2(b))
        && space.centers_log2[i].to_bits() == a.to_bits()
        && space.centers_log2[j].to_bits() == b.to_bits()
    {
        i.abs_diff(j) as f64 * 1200.0 / space.bins_per_oct as f64
    } else {
        1200.0 * (b as f64 - a as f64).abs()
    }
}

impl Voice {
    pub(super) fn glide_pitch(pitch: f32, target: f32, dt: f32, tau: f32) -> f32 {
        let alpha = if tau <= 1e-6 {
            1.0
        } else {
            1.0 - (-dt.max(0.0) / tau).exp()
        };
        pitch + (target - pitch) * alpha.clamp(0.0, 1.0)
    }

    pub(super) fn surviving_tone(
        tone: Option<ToneId>,
        resting: bool,
        cmds: &[ToneCmd],
        at: Tick,
    ) -> Option<ToneId> {
        tone.filter(|tone_id| !resting || !cmds.iter().any(|cmd| {
            matches!(cmd, ToneCmd::Off { tone_id: off_id, off_tick } if off_id == tone_id && *off_tick <= at)
        }))
    }

    pub(super) fn prepare_span_segment(&mut self, ctx: &CoreTickCtx, space: Option<&Log2Space>) {
        self.phonation_engine.advance_span(ctx.now_tick);
        let deferred = self.defers_participation_pitch();
        let Some(gap) = &mut self.phonation_engine.span_gap else {
            return;
        };
        for adoption in &self.participation_pitch.rearticulations {
            if adoption.at <= ctx.now_tick
                && gap.last_adoption.is_none_or(|last| adoption.at > last)
            {
                if deferred {
                    let application = select_pitch_application(
                        &self.effective_control.pitch,
                        space
                            .map(|space| (space, gap.last_onset_pitch.log2(), adoption.pitch_log2)),
                    );
                    if application == PitchApplyMode::Glide {
                        gap.glide_target = Some(adoption.pitch_log2);
                        gap.pending_pitch = None;
                    } else {
                        gap.pending_pitch = Some(adoption.pitch_log2);
                        gap.glide_target = None;
                    }
                }
                gap.last_adoption = Some(adoption.at);
            }
        }
        if let Some(step) = self
            .participation_pitch
            .steps
            .iter()
            .find(|step| step.at == ctx.now_tick)
            && gap.last_control != Some(step.at)
        {
            let target = if deferred {
                gap.glide_target
            } else {
                Some(step.target_log2)
            };
            if let Some(target) = target {
                gap.pitch = Self::glide_pitch(
                    gap.pitch.log2(),
                    target,
                    step.dt,
                    self.effective_control.pitch.pitch_glide_tau_sec.max(0.0),
                )
                .exp2();
            }
            gap.last_control = Some(step.at);
        }
        gap.next_pitch = gap.pending_pitch.map_or(gap.pitch, f32::exp2);
        if !deferred
            && let Some(adoption) = self
                .participation_pitch
                .rearticulations
                .iter()
                .find(|m| m.at == ctx.now_tick)
        {
            gap.next_pitch = adoption.pitch_log2.exp2();
        }
    }

    pub(super) fn prepare_participation_hop(
        &mut self,
        ctx: &CoreTickCtx,
        state: &CoreState,
        out: &mut PhonationBatch,
    ) {
        if self.phonation_engine.span_gap.is_some() {
            let pitch = self
                .participation_pitch
                .steps
                .first()
                .map_or(self.body.base_freq_hz(), |step| step.before_log2.exp2());
            self.phonation_engine
                .set_span_pitch(ctx.now_tick, pitch, pitch);
        }
        self.phonation_engine.prepare_synchronized_hop(ctx);
        self.phonation_engine.initial_participating_hold(
            ctx,
            state,
            &mut out.cmds,
            &mut self.phonation_scratch.events,
            &mut out.onsets,
        );
    }

    pub(super) fn tick_participation_segment(
        &mut self,
        ctx: &CoreTickCtx,
        state: &CoreState,
        arrival: Option<&crate::life::arrival_cost::ArrivalContext<'_>>,
        out: &mut PhonationBatch,
    ) {
        self.phonation_engine.tick_with_arrival(
            ctx,
            state,
            None,
            0.0,
            1.0,
            None,
            arrival,
            &mut out.cmds,
            &mut self.phonation_scratch.events,
            &mut out.onsets,
        );
        self.phonation_engine.advance_span(ctx.frame_end);
    }

    pub(super) fn apply_phonation_onsets(&mut self, consonance: f32, out: &PhonationBatch) {
        for onset in &out.onsets {
            self.articulation
                .apply_phonation_onset(consonance, onset.strength);
        }
    }

    pub(super) fn phonation_segment(
        &self,
        ctx: &CoreTickCtx,
        start: Tick,
        end: Tick,
    ) -> CoreTickCtx {
        let mut segment = CoreTickCtx {
            now_tick: start,
            frame_end: end,
            ..*ctx
        };
        if matches!(self.phonation_engine.clock, PhonationClock::Coupling(_)) {
            let elapsed = start.saturating_sub(ctx.now_tick) as f32 / ctx.fs;
            segment.rhythms.delta.phase = (ctx.rhythms.delta.phase
                + std::f32::consts::TAU * ctx.rhythms.delta.freq_hz * elapsed)
                .rem_euclid(std::f32::consts::TAU);
            segment.rhythms.measure.phase = (ctx.rhythms.measure.phase
                + std::f32::consts::TAU * ctx.rhythms.measure.freq_hz * elapsed)
                .rem_euclid(std::f32::consts::TAU);
        }
        segment
    }

    pub(crate) fn configure_control_hop(&mut self, hop: usize, control_step: usize, fs: f32) {
        self.phonation_engine.reserve_felt_hop(hop, fs);
        self.support.clock_fs = fs;
        self.support.clock_step = control_step.max(1) as u64;
        if (self.effective_control.pitch.mode == PitchMode::Free || self.sustain_participation)
            && matches!(self.phonation_engine.mode, PhonationMode::Hold)
        {
            let mut capacity = (hop / control_step).max(1);
            if self.sustain_participation
                && matches!(self.phonation_engine.clock, PhonationClock::Coupling(_))
            {
                capacity = capacity.max(self.phonation_engine.reserve_synchronized_hop(hop, fs));
            }
            if let Some(deferred) = &mut self.sequence_control {
                deferred.reserve(capacity + 1);
            }
            self.participation_pitch.rearticulations.reserve(capacity);
            self.support.held_opportunities.reserve(capacity);
            if let Some(shape) = self.phonation_engine.clock.shape_mut() {
                shape.reserve(capacity, hop);
            }
            if self.sustain_participation {
                self.participation_pitch.steps.reserve(capacity);
            }
            self.phonation_scratch.events.reserve(capacity + 1);
            self.active_render_notes.reserve(capacity + 1);
        }
    }

    pub(crate) fn prepare_phonation_batch(&self, out: &mut PhonationBatch) {
        let capacity = self.participation_pitch.rearticulations.capacity();
        if capacity > 0 {
            // Batch storage is initialized alongside its Voice, then reused.
            out.cmds
                .reserve((3 * (capacity + 1)).saturating_sub(out.cmds.len()));
            out.tones
                .reserve((capacity + 1).saturating_sub(out.tones.len()));
            out.onsets
                .reserve((capacity + 1).saturating_sub(out.onsets.len()));
        }
    }

    pub(crate) fn reports_onset_pitch(&self) -> bool {
        self.participation_pitch.application == PitchApplyMode::Rearticulate
            || self.sustain_participation
    }

    pub fn force_set_pitch_log2(&mut self, log_freq: f32) {
        let log_freq = log_freq.max(0.0);
        self.body.set_pitch_log2(log_freq);
        self.articulation.set_gate(1.0);
        self.pitch_ctl.force_set_target_pitch_log2(log_freq);
    }

    /// Update pitch targets at control rate (hop-sized steps).
    pub fn update_pitch_target(
        &mut self,
        rhythms: &NeuralRhythms,
        dt_sec: f32,
        landscape: &Landscape,
        neighbor_pitch_log2: &[f32],
        neighbor_salience: &[f32],
    ) {
        self.update_pitch_target_with_listener_pressure(
            rhythms,
            dt_sec,
            landscape,
            neighbor_pitch_log2,
            neighbor_salience,
            ListenerPressure::default(),
        );
    }

    pub(crate) fn update_pitch_target_with_listener_pressure(
        &mut self,
        rhythms: &NeuralRhythms,
        dt_sec: f32,
        landscape: &Landscape,
        neighbor_pitch_log2: &[f32],
        neighbor_salience: &[f32],
        listener_pressure: ListenerPressure,
    ) {
        if self.harmonic.member {
            return;
        }
        if self.effective_control.supports_own_sequence()
            && let Some(deferred) = &mut self.sequence_control
        {
            deferred.push(
                self.control_tick,
                self.body.base_freq_hz(),
                rhythms,
                dt_sec,
                landscape,
                &self.effective_control.pitch,
                neighbor_pitch_log2,
                neighbor_salience,
                listener_pressure,
            );
            self.participation_pitch.adopted = false;
            return;
        }
        self.decide_pitch_from_frequency(
            self.body.base_freq_hz(),
            rhythms,
            dt_sec,
            landscape,
            neighbor_pitch_log2,
            neighbor_salience,
            listener_pressure,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decide_pitch_from_frequency(
        &mut self,
        current_freq: f32,
        rhythms: &NeuralRhythms,
        dt_sec: f32,
        landscape: &Landscape,
        neighbor_pitch_log2: &[f32],
        neighbor_salience: &[f32],
        listener_pressure: ListenerPressure,
    ) {
        #[cfg(test)]
        crate::core::tonal_context::acquisition::owner(
            self.id,
            self.metadata.generation,
            self.control_tick.unwrap_or(0),
            current_freq,
            self.effective_control.pitch.dwell,
        );
        #[cfg(feature = "loo-diagnostic")]
        pitch_core::diagnostic::set_owner(
            self.id,
            self.metadata.population_id,
            self.metadata.generation,
            self.control_tick.unwrap_or(0),
        );
        if self.effective_control.pitch.follow_tonal_context > 0. {
            let excluded = self.effective_control.pitch.leave_self_out;
            self.pitch_ctl.loo_input_mut().tonal_context = landscape
                .ecology
                .tonal_observations
                .iter()
                .find(|s| s.id == self.id && s.generation == self.metadata.generation)
                .map(|s| {
                    if s.excluded != excluded {
                        None
                    } else if s.shared {
                        landscape.ecology.tonal_context.clone()
                    } else {
                        s.snapshot.clone()
                    }
                })
                .unwrap_or_else(|| {
                    if excluded
                        && !landscape
                            .ecology
                            .site_end
                            .is_some_and(|end| end <= self.birth_tick.unwrap_or(0))
                    {
                        None
                    } else {
                        landscape.ecology.tonal_context.clone()
                    }
                });
        } else {
            self.pitch_ctl.loo_input_mut().tonal_context = None;
        }
        let previous = self.target_pitch_log2();
        // Deferred participation adopts at an independent opportunity, not at this proposal.
        let rearticulation = (self.effective_control.pitch.tail_roughness_aversion > 0.0
            && self.effective_control.pitch.leave_self_out
            && select_pitch_application(&self.effective_control.pitch, None)
                == PitchApplyMode::Rearticulate
            && !self.sustain_participation
            && matches!(self.phonation_engine.mode, PhonationMode::Hold)
            && self.phonation_engine.has_active_notes()
            && self.phonation_gate_open
            && self.compute_target_amp() > Self::AMP_EPS
            && self.is_alive()
            && !self.release.pending
            && !self.release.remove_pending
            && self.control_tick.is_some())
        .then(|| pitch_core::RearticulationView {
            weight: self.effective_control.pitch.tail_roughness_aversion,
            id: self.id,
            generation: self.metadata.generation,
            reference_log2: self
                .participation_pitch
                .rearticulations
                .last()
                .map(|r| r.pitch_log2.exp2().log2())
                .or(self.participation_pitch.last_onset_log2)
                .unwrap_or(previous),
            threshold_cents: self.effective_control.pitch.rearticulate_above_cents,
            previous_target: previous,
            bounds: PitchController::target_bounds(
                &self.effective_control.pitch,
                current_freq.max(1.).log2(),
                landscape,
            ),
        });
        self.pitch_ctl.loo_input_mut().rearticulation = rearticulation;
        self.pitch_ctl.control_tick = self.control_tick;
        self.participation_pitch.adopted = self.pitch_ctl.update_pitch_target(
            current_freq,
            rhythms,
            dt_sec,
            landscape,
            &self.effective_control.pitch,
            neighbor_pitch_log2,
            neighbor_salience,
            listener_pressure.temperature_bonus,
        );
        if crate::life::support_action::active(landscape.harmonic.support_weights)
            && self.phonation_engine.mode == PhonationMode::Hold
            && !self.sustain_participation
            && let Some(at) = self.control_tick
        {
            if self.pitch_ctl.harmonic_evaluated || self.phonation_engine.held_next.is_none() {
                self.phonation_engine.held_next = self.pitch_ctl.next_proposal_tick(
                    at,
                    &self.effective_control.pitch,
                    rhythms,
                    self.support.clock_fs,
                    self.support.clock_step,
                );
            }
            if self.pitch_ctl.harmonic_evaluated {
                self.support
                    .held_opportunities
                    .push((at, self.phonation_engine.held_next));
            }
        }
        if self.participation_pitch.adopted {
            self.participation_pitch.application =
                select_pitch_application(&self.effective_control.pitch, None);
            self.select_adoption_application(&landscape.space, previous);
            if (self.participation_pitch.adoption_application == PitchApplyMode::Rearticulate
                || self.sustain_participation)
                && (self.defers_participation_pitch()
                    || (self.phonation_engine.has_active_notes() && self.phonation_gate_open))
                && self.is_alive()
                && !self.release.pending
                && !self.release.remove_pending
            {
                // Release grows the buffer if the birth capacity is insufficient.
                debug_assert!(
                    self.participation_pitch.rearticulations.len()
                        < self.participation_pitch.rearticulations.capacity(),
                    "control hop capacity must be configured at birth"
                );
                debug_assert!(
                    self.control_tick.is_some(),
                    "control must timestamp pitch adoption"
                );
                // Release drops an untimed adoption instead of inventing its sample.
                if let Some(at) = self.control_tick {
                    let pitch_log2 = self.target_pitch_log2();
                    self.participation_pitch
                        .rearticulations
                        .push(Rearticulation { at, pitch_log2 });
                    tracing::debug!(target: "pitch::adoption", id = self.id, at, pitch_log2);
                }
            }
        }
    }

    pub(super) fn defers_participation_pitch(&self) -> bool {
        self.sustain_participation
            && self.effective_control.pitch.mode == PitchMode::Free
            && matches!(
                self.participation_pitch.application,
                PitchApplyMode::GateSnap | PitchApplyMode::Rearticulate
            )
    }

    pub(super) fn select_adoption_application(&mut self, space: &Log2Space, previous: f32) {
        // Earlier legal requests in this output hop precede the last materialized onset.
        let reference = self
            .participation_pitch
            .rearticulations
            .last()
            .map(|r| r.pitch_log2.exp2().log2())
            .or(self.participation_pitch.last_onset_log2)
            .unwrap_or(previous);
        self.participation_pitch.adoption_application = select_pitch_application(
            &self.effective_control.pitch,
            Some((space, reference, self.target_pitch_log2())),
        );
    }

    /// Decide phase.
    ///
    /// Contract: this must not mutate body/articulation/lifecycle fields that
    /// other agents can observe in the same substep. Only local pitch-controller
    /// decision state (target/adaptation memory) may change here.
    #[allow(clippy::too_many_arguments)]
    pub fn decide_pitch_target(
        &mut self,
        dt_sec: f32,
        rhythms: &NeuralRhythms,
        landscape: &Landscape,
        neighbor_pitch_log2: &[f32],
        neighbor_salience: &[f32],
    ) {
        self.decide_pitch_target_with_listener_pressure(
            dt_sec,
            rhythms,
            landscape,
            neighbor_pitch_log2,
            neighbor_salience,
            ListenerPressure::default(),
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn decide_pitch_target_with_listener_pressure(
        &mut self,
        dt_sec: f32,
        rhythms: &NeuralRhythms,
        landscape: &Landscape,
        neighbor_pitch_log2: &[f32],
        neighbor_salience: &[f32],
        listener_pressure: ListenerPressure,
    ) {
        #[cfg(all(test, feature = "research"))]
        {
            self.pitch_ctl.trace_owner = Some((self.id(), self.metadata.generation));
        }
        self.update_pitch_target_with_listener_pressure(
            rhythms,
            dt_sec,
            landscape,
            neighbor_pitch_log2,
            neighbor_salience,
            listener_pressure,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn tick_phonation(
        &mut self,
        tb: &Timebase,
        now: Tick,
        rhythms: &NeuralRhythms,
        social: Option<&SocialDensityTrace>,
        social_coupling: f32,
        extra_gate_gain: f32,
        consonance: f32,
    ) -> PhonationBatch {
        let mut batch = PhonationBatch::default();
        self.tick_phonation_into(
            tb,
            now,
            rhythms,
            social,
            social_coupling,
            extra_gate_gain,
            consonance,
            &mut batch,
        );
        batch
    }

    #[allow(clippy::too_many_arguments)]
    pub fn tick_phonation_into(
        &mut self,
        tb: &Timebase,
        now: Tick,
        rhythms: &NeuralRhythms,
        social: Option<&SocialDensityTrace>,
        social_coupling: f32,
        extra_gate_gain: f32,
        consonance: f32,
        out: &mut PhonationBatch,
    ) {
        self.tick_phonation_into_with_arrival(
            tb,
            now,
            rhythms,
            social,
            social_coupling,
            extra_gate_gain,
            consonance,
            None,
            None,
            out,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tick_phonation_into_with_arrival(
        &mut self,
        tb: &Timebase,
        now: Tick,
        rhythms: &NeuralRhythms,
        social: Option<&SocialDensityTrace>,
        social_coupling: f32,
        extra_gate_gain: f32,
        consonance: f32,
        arrival: Option<&crate::life::arrival_cost::ArrivalContext<'_>>,
        space: Option<&Log2Space>,
        out: &mut PhonationBatch,
    ) {
        out.source_id = self.id;
        out.source_generation = self.metadata.generation;
        out.routing = self.effective_control.body.routing;
        out.clear();
        out.motion_clock = self.generation_motion_clock();
        out.motion_update = self.pending_motion_amount.take();
        self.phonation_scratch.events.clear();
        let hop_tick = (tb.hop as Tick).max(1);
        let frame_end = now.saturating_add(hop_tick);
        let ctx = CoreTickCtx {
            now_tick: now,
            frame_end,
            fs: tb.fs,
            rhythms: self.phonation_engine.felt.rhythms(*rhythms, now, tb.fs),
        };
        // Target amp and the phonation gate are both evaluated up front and fed
        // into `onset_allowed`, which the engine uses to gate onset *emission*.
        // This must happen before the engine ticks: gating emission (instead of
        // discarding an already-emitted onset afterward) is what keeps Hold's
        // one-shot latch (`note_on_sent`) from being consumed while the onset
        // cannot be heard, so it can still fire once amp/gate allow it later.
        let target_amp = self.compute_target_amp();
        self.update_phonation_gate(consonance);
        let onset_allowed = target_amp > Self::AMP_EPS && self.phonation_gate_open;
        let state = CoreState {
            is_alive: self.is_alive()
                && !self.release.remove_pending
                && !((self.reports_onset_pitch() || self.phonation_engine.expression_active)
                    && self.release.pending),
            onset_allowed,
        };
        out.body_policy = Some(crate::life::action_candidates::PolicySnapshot {
            site_id: self.effective_control.body.site_id,
            at: now,
            is_alive: state.is_alive,
            gate_allows_onset: state.onset_allowed,
        });
        if let PhonationClock::Participation(clock) = &mut self.phonation_engine.clock {
            clock.hop_frames = hop_tick;
        }
        self.phonation_engine.set_sound_adsr(self.voice_adsr);
        if self.sequence_control.is_some() && self.effective_control.supports_own_sequence() {
            self.emit_sequence_participation(
                &ctx, &state, consonance, target_amp, arrival, space, out,
            );
            return;
        }
        if self.sustain_participation
            && let Some(shape) = self.phonation_engine.clock.shape_mut()
        {
            // Only tied Voices instantiate the extra opportunity guard.
            if let Some(tie) = shape.tie.as_mut() {
                tie.enabled = self.effective_control.supports_tie();
                shape.enabled = self.effective_control.supports_articulation_persistence()
                    && shape.retention_active;
                if shape.enabled || tie.enabled {
                    self.emit_shape_participation(
                        &ctx, &state, consonance, target_amp, arrival, space, out,
                    );
                    return;
                }
            } else {
                shape.enabled = self.effective_control.supports_articulation_persistence();
                if shape.enabled {
                    self.emit_shape_participation(
                        &ctx, &state, consonance, target_amp, arrival, space, out,
                    );
                    return;
                }
            }
        }
        if self.sustain_participation {
            self.emit_sustained_participation(
                &ctx, &state, consonance, target_amp, arrival, space, out,
            );
            return;
        }
        if !self.phonation_engine.support_held_active() {
            self.support.held_opportunities.clear();
        }
        if self.phonation_engine.support_held_active()
            && let Some(&(at, next)) = self.support.held_opportunities.first()
        {
            if at == now
                && !self
                    .participation_pitch
                    .rearticulations
                    .iter()
                    .any(|r| r.at == now)
            {
                self.phonation_engine.held_window = Some((at, next));
            } else {
                self.phonation_engine.held_next = Some(at);
            }
        }
        self.phonation_engine.tick_with_arrival(
            &ctx,
            &state,
            social,
            social_coupling,
            extra_gate_gain,
            None,
            arrival,
            &mut out.cmds,
            &mut self.phonation_scratch.events,
            &mut out.onsets,
        );
        if self.phonation_engine.support_held_active() && !self.sustain_participation {
            for &(at, next) in &self.support.held_opportunities {
                self.phonation_engine.held_window = Some((at, next));
                let request_ctx = CoreTickCtx {
                    now_tick: at,
                    ..ctx
                };
                let request = self
                    .participation_pitch
                    .rearticulations
                    .iter()
                    .find(|r| r.at == at);
                if let Some(request) = request {
                    if let Some(context) = self.phonation_engine.support_action.as_mut() {
                        context.pitch = request.pitch_log2;
                    }
                    let emitted = self.phonation_engine.rearticulate_hold(
                        &request_ctx,
                        &state,
                        &mut out.cmds,
                        &mut self.phonation_scratch.events,
                        &mut out.onsets,
                    );
                    if !emitted
                        && let crate::life::support_action::Body::One { pitch, .. } =
                            self.support.body
                    {
                        self.body.set_pitch_log2(pitch);
                    }
                } else {
                    self.phonation_engine.tick_with_arrival(
                        &request_ctx,
                        &state,
                        social,
                        social_coupling,
                        extra_gate_gain,
                        None,
                        arrival,
                        &mut out.cmds,
                        &mut self.phonation_scratch.events,
                        &mut out.onsets,
                    );
                }
            }
            self.support.held_opportunities.clear();
            self.phonation_engine.held_window = None;
            self.participation_pitch.rearticulations.clear();
        }
        if self.reports_onset_pitch() {
            for request in &self.participation_pitch.rearticulations {
                let in_hop = request.at >= now && request.at < frame_end;
                debug_assert!(
                    in_hop,
                    "rearticulation timestamp must belong to the next output hop"
                );
                // Release drops requests outside this hop without replaying them later.
                if !in_hop {
                    continue;
                }
                let request_ctx = CoreTickCtx {
                    now_tick: request.at,
                    ..ctx
                };
                let previous_pitch =
                    self.phonation_engine
                        .support_action
                        .as_mut()
                        .and_then(|context| {
                            context.pitch = request.pitch_log2;
                            match self.support.body {
                                crate::life::support_action::Body::One { pitch, .. } => Some(pitch),
                                _ => None,
                            }
                        });
                let emitted = self.phonation_engine.rearticulate_hold(
                    &request_ctx,
                    &state,
                    &mut out.cmds,
                    &mut self.phonation_scratch.events,
                    &mut out.onsets,
                );
                if !emitted && let Some(pitch) = previous_pitch {
                    self.body.set_pitch_log2(pitch);
                }
            }
            // Released tails must retain their pitch before surviving-note updates.
            self.prune_tracked_render_notes(&out.cmds);
        }
        let phonation_mode = self.phonation_engine.mode;
        let had_tone_on = out.cmds.iter().any(|cmd| matches!(cmd, ToneCmd::On { .. }));
        if self.release.remove_pending {
            self.emit_release_note_offs(now, &mut out.cmds);
            Self::discard_pending_render_tone_ons(&mut out.cmds);
            self.phonation_scratch.events.clear();
            out.onsets.clear();
            self.prune_tracked_render_notes(&out.cmds);
            self.participation_pitch.rearticulations.clear();
            return;
        }
        if matches!(phonation_mode, PhonationMode::Gated) || self.reports_onset_pitch() {
            self.apply_phonation_onsets(consonance, out);
            // Snap-and-hold: a GateSnap voice adopts the current consonance
            // target at the note onset, then holds that pitch for the note (no
            // mid-note glide). Successive notes step toward consonance discretely.
            if !out.onsets.is_empty()
                && matches!(
                    self.participation_pitch.application,
                    PitchApplyMode::GateSnap
                )
                && !matches!(self.effective_control.pitch.mode, PitchMode::Lock)
            {
                self.body.set_pitch_log2(self.target_pitch_log2());
            }
        }
        let freq_hz = self.body.base_freq_hz();
        let continuous_drive = self.effective_control.body.continuous_drive;
        self.emit_phonation_updates(now, target_amp, freq_hz, continuous_drive, &mut out.cmds);
        if self.phonation_scratch.events.is_empty() {
            self.prune_tracked_render_notes(&out.cmds);
            if had_tone_on {
                Self::discard_pending_render_tone_ons(&mut out.cmds);
            }
            self.participation_pitch.rearticulations.clear();
            return;
        }
        if !freq_hz.is_finite() || freq_hz <= 0.0 {
            self.prune_tracked_render_notes(&out.cmds);
            if had_tone_on {
                Self::discard_pending_render_tone_ons(&mut out.cmds);
            }
            self.phonation_scratch.events.clear();
            self.participation_pitch.rearticulations.clear();
            return;
        }
        // `onset_allowed` already prevented the engine from emitting any onset
        // event while `target_amp <= AMP_EPS`, so a non-empty events list here
        // guarantees `target_amp` is materializable.
        debug_assert!(
            target_amp > Self::AMP_EPS,
            "onset gating must prevent tone events at/below AMP_EPS"
        );
        let render_modulator =
            self.articulation
                .render_modulator_spec(if self.reports_onset_pitch() {
                    PhonationMode::Gated
                } else {
                    phonation_mode
                });
        let body = self.body_snapshot();
        let hold_ticks = if matches!(phonation_mode, PhonationMode::Hold) {
            Some(Tick::MAX)
        } else {
            None
        };
        for event in self.phonation_scratch.events.drain(..) {
            out.tones.push(ToneSpec::from_event(
                event,
                self.participation_pitch
                    .rearticulations
                    .iter()
                    .find(|request| request.at == event.onset_tick)
                    .map_or(freq_hz, |request| request.pitch_log2.exp2()),
                target_amp,
                hold_ticks,
                body.clone(),
                render_modulator.clone(),
                self.voice_adsr,
            ));
        }
        self.participation_pitch.rearticulations.clear();
        self.track_new_render_notes(&out.tones);
        // An onset and its note-off can share a hop. Track first, then remove
        // every note already scheduled to end, including the newly added ones.
        self.prune_tracked_render_notes(&out.cmds);
        debug_assert!(
            !out.cmds.iter().any(|cmd| matches!(cmd, ToneCmd::On { .. })) || !out.tones.is_empty(),
            "ToneCmd::On emitted without tone specs"
        );
    }

    /// Advance the one-way phonation gate (`phonate_when_viable()`): once
    /// perceived consonance reaches the low bound of the viability window, the
    /// gate opens and never closes again. Later consonance drops are the
    /// energy/viability system's business, not the gate's.
    pub(super) fn update_phonation_gate(&mut self, consonance: f32) {
        if self.phonation_gate_open {
            return;
        }
        if consonance >= self.articulation.consonance_viability_low() {
            self.phonation_gate_open = true;
        }
    }
}
