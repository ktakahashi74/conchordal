use super::*;
use crate::life::shape_policy::Shape;

impl Voice {
    fn articulation_persistence_clear_before(&mut self, at: Tick) {
        let PhonationClock::Participation(clock) = &mut self.phonation_engine.clock else {
            unreachable!()
        };
        if clock.shape.as_mut().unwrap().clear_before(at) {
            self.pending_participation_pitch = None;
        }
    }

    fn articulation_persistence_drain_ons(
        &mut self,
        index: &mut usize,
        pitch: &mut f32,
        amp: f32,
        until: Tick,
        out: &mut PhonationBatch,
    ) {
        while *index < self.phonation_scratch.events.len() {
            let at = self.phonation_scratch.events[*index].onset_tick;
            self.articulation_persistence_clear_before(at);
            self.emit_bounded_on(*index, pitch, amp, out);
            *index += 1;
        }
        self.articulation_persistence_clear_before(until);
        let PhonationClock::Participation(clock) = &mut self.phonation_engine.clock else {
            unreachable!()
        };
        clock.shape.as_mut().unwrap().finish_segment();
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_shape_participation(
        &mut self,
        ctx: &CoreTickCtx,
        state: &CoreState,
        consonance: f32,
        target_amp: f32,
        arrival: Option<&crate::life::arrival_cost::ArrivalContext<'_>>,
        space: Option<&Log2Space>,
        out: &mut PhonationBatch,
    ) {
        let previous_tone = self.active_render_notes.last().map(|n| n.tone_id);
        out.body_opportunity = self.phonation_engine.clock.opportunity(ctx);
        out.intrinsic_period_sec = self.phonation_engine.clock.intrinsic_period_sec();
        self.phonation_engine.initial_participating_hold(
            ctx,
            state,
            &mut out.cmds,
            &mut self.phonation_scratch.events,
            &mut out.onsets,
        );
        let mut pitch = self.body.base_freq_hz().log2();
        let mut on_index = 0;
        self.articulation_persistence_drain_ons(
            &mut on_index,
            &mut pitch,
            target_amp,
            ctx.now_tick,
            out,
        );
        let mut m_index = 0;
        let mut cursor = ctx.now_tick;
        let tau = self.effective_control.pitch.pitch_glide_tau_sec.max(0.0);
        if state.is_alive {
            for step_index in 0..self.participation_pitch_steps.len() {
                let step = self.participation_pitch_steps[step_index];
                if cursor < step.at {
                    let segment = CoreTickCtx {
                        now_tick: cursor,
                        frame_end: step.at,
                        ..*ctx
                    };
                    self.phonation_engine.set_adoption_candidate(None);
                    self.phonation_engine.tick_with_arrival(
                        &segment,
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
                    self.articulation_persistence_drain_ons(
                        &mut on_index,
                        &mut pitch,
                        target_amp,
                        step.at - 1,
                        out,
                    );
                }
                while m_index < self.rearticulations.len()
                    && self.rearticulations[m_index].at <= step.at
                {
                    let m = self.rearticulations[m_index];
                    let reference = self.last_onset_pitch_log2.unwrap_or(pitch);
                    let base = select_pitch_application(
                        &self.effective_control.pitch,
                        Some((
                            space.expect("adoption needs the current grid"),
                            reference,
                            m.pitch_log2,
                        )),
                    );
                    let PhonationClock::Participation(clock) = &mut self.phonation_engine.clock
                    else {
                        unreachable!()
                    };
                    let locked = clock
                        .policy
                        .as_ref()
                        .is_some_and(|p| p.articulation_persistence_planned());
                    let shape = clock.shape.as_mut().unwrap();
                    shape.propose(
                        if base == PitchApplyMode::Glide {
                            Shape::Connect
                        } else {
                            Shape::Base
                        },
                        m.at,
                        locked,
                    );
                    let glide = shape.state == Shape::Connect || base == PitchApplyMode::Glide;
                    self.articulation_persistence_clear_before(m.at);
                    if glide {
                        self.pending_participation_pitch = None;
                        self.participation_glide_target = Some(m.pitch_log2);
                        self.adoption_application = PitchApplyMode::Glide;
                    } else {
                        self.pending_participation_pitch = Some(m.pitch_log2);
                        self.participation_glide_target = None;
                        self.adoption_application = base;
                    }
                    m_index += 1;
                }
                let segment = CoreTickCtx {
                    now_tick: step.at,
                    frame_end: step.at.saturating_add(1),
                    ..*ctx
                };
                self.phonation_engine.set_adoption_candidate(None);
                self.phonation_engine.tick_with_arrival(
                    &segment,
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
                // M and glide precede a sounded opportunity at exactly the same sample.
                if let Some(target) = self.participation_glide_target {
                    let alpha = if tau <= 1e-6 {
                        1.0
                    } else {
                        1.0 - (-step.dt.max(0.0) / tau).exp()
                    };
                    pitch += (target - pitch) * alpha.clamp(0.0, 1.0);
                    let same_on = self
                        .phonation_scratch
                        .events
                        .get(on_index)
                        .is_some_and(|e| e.onset_tick == step.at);
                    let tone = on_index
                        .checked_sub(1)
                        .map(|i| self.phonation_scratch.events[i].tone_id)
                        .or(previous_tone);
                    let tone = tone.filter(|tone_id| self.phonation_engine.rest.is_none()
                        || !out.cmds.iter().any(|cmd| matches!(cmd, ToneCmd::Off { tone_id: id, off_tick } if id == tone_id && *off_tick <= step.at)));
                    if !same_on && let Some(tone_id) = tone {
                        out.cmds.push(ToneCmd::Update {
                            tone_id,
                            at_tick: Some(step.at),
                            update: ToneUpdate {
                                target_freq_hz: Some(pitch.exp2()),
                                ..Default::default()
                            },
                        });
                    }
                }
                self.articulation_persistence_drain_ons(
                    &mut on_index,
                    &mut pitch,
                    target_amp,
                    step.at,
                    out,
                );
                cursor = step.at.saturating_add(1);
            }
            if cursor < ctx.frame_end {
                let segment = CoreTickCtx {
                    now_tick: cursor,
                    ..*ctx
                };
                self.phonation_engine.tick_with_arrival(
                    &segment,
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
                self.articulation_persistence_drain_ons(
                    &mut on_index,
                    &mut pitch,
                    target_amp,
                    ctx.frame_end - 1,
                    out,
                );
            }
            // Release keeps the processed pitch; the hop cleanup drops unmatched adoptions.
            debug_assert_eq!(m_index, self.rearticulations.len());
            self.body.set_pitch_log2(pitch);
        } else {
            self.pending_participation_pitch = None;
            self.participation_glide_target = None;
        }
        for onset in &out.onsets {
            self.articulation
                .apply_phonation_onset(consonance, onset.strength);
        }
        self.prune_tracked_render_notes(&out.cmds);
        for tracked in &mut self.active_render_notes {
            tracked.last_target_freq_hz = self.body.base_freq_hz();
        }
        self.emit_phonation_updates(
            ctx.now_tick,
            target_amp,
            self.body.base_freq_hz(),
            self.effective_control.body.continuous_drive,
            &mut out.cmds,
        );
        self.track_new_render_notes(&out.tones);
        self.prune_tracked_render_notes(&out.cmds);
        self.phonation_scratch.events.clear();
        self.rearticulations.clear();
        self.participation_pitch_steps.clear();
    }
}
