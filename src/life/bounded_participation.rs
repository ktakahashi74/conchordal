use super::*;

impl Voice {
    pub(super) fn emit_bounded_on(
        &mut self,
        index: usize,
        pitch: &mut f32,
        target_amp: f32,
        out: &mut PhonationBatch,
    ) {
        let event = &self.phonation_scratch.events[index];
        if event.opportunity.is_some()
            && let Some(target) = self.pending_participation_pitch.take()
        {
            *pitch = target;
        }
        let tone = ToneSpec {
            opportunity: event.opportunity,
            tone_id: event.tone_id,
            onset: event.onset_tick,
            hold_ticks: Some(Tick::MAX),
            freq_hz: pitch.exp2(),
            amp: target_amp,
            smoothing_tau_sec: Self::PHONATION_UPDATE_SMOOTH_TAU_SEC,
            body: self.body_snapshot(),
            render_modulator: self
                .articulation
                .render_modulator_spec(PhonationMode::Gated),
            adsr: self.voice_adsr,
        };
        self.last_onset_pitch_log2 = Some(tone.freq_hz.log2());
        out.tones.push(tone);
    }

    pub(super) fn emit_bounded_participation(
        &mut self,
        state: &CoreState,
        consonance: f32,
        target_amp: f32,
        now: Tick,
        space: Option<&Log2Space>,
        out: &mut PhonationBatch,
    ) {
        let mut pitch = self.body.base_freq_hz().log2();
        let mut on_index = 0;
        let mut m_index = 0;
        let previous_tone = self.active_render_notes.last().map(|n| n.tone_id);
        let tau = self.effective_control.pitch.pitch_glide_tau_sec.max(0.0);
        if state.is_alive {
            for step_index in 0..self.participation_pitch_steps.len() {
                let step = self.participation_pitch_steps[step_index];
                while on_index < self.phonation_scratch.events.len()
                    && self.phonation_scratch.events[on_index].onset_tick < step.at
                {
                    self.emit_bounded_on(on_index, &mut pitch, target_amp, out);
                    on_index += 1;
                }
                while m_index < self.rearticulations.len()
                    && self.rearticulations[m_index].at <= step.at
                {
                    let m = self.rearticulations[m_index];
                    let reference = self.last_onset_pitch_log2.unwrap_or(pitch);
                    self.adoption_application = select_pitch_application(
                        &self.effective_control.pitch,
                        Some((
                            space.expect("pitch adoption needs the current grid"),
                            reference,
                            m.pitch_log2,
                        )),
                    );
                    if self.adoption_application == PitchApplyMode::Glide {
                        self.pending_participation_pitch = None;
                        self.participation_glide_target = Some(m.pitch_log2);
                    } else {
                        self.pending_participation_pitch = Some(m.pitch_log2);
                        self.participation_glide_target = None;
                    }
                    m_index += 1;
                }
                if let Some(target) = self.participation_glide_target {
                    let alpha = if tau <= 1e-6 {
                        1.0
                    } else {
                        1.0 - (-step.dt.max(0.0) / tau).exp()
                    };
                    pitch += (target - pitch) * alpha.clamp(0.0, 1.0);
                    let onset = self
                        .phonation_scratch
                        .events
                        .get(on_index)
                        .is_some_and(|e| e.onset_tick == step.at);
                    let tone = on_index
                        .checked_sub(1)
                        .map(|i| self.phonation_scratch.events[i].tone_id)
                        .or(previous_tone);
                    let tone = tone.filter(|tone_id| {
                        self.phonation_engine.rest.is_none()
                            || !out.cmds.iter().any(|cmd| {
                                matches!(cmd, ToneCmd::Off { tone_id: off_id, off_tick }
                                    if off_id == tone_id && *off_tick <= step.at)
                            })
                    });
                    if !onset && let Some(tone_id) = tone {
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
                while on_index < self.phonation_scratch.events.len()
                    && self.phonation_scratch.events[on_index].onset_tick <= step.at
                {
                    self.emit_bounded_on(on_index, &mut pitch, target_amp, out);
                    on_index += 1;
                }
            }
            while on_index < self.phonation_scratch.events.len() {
                self.emit_bounded_on(on_index, &mut pitch, target_amp, out);
                on_index += 1;
            }
            assert_eq!(
                m_index,
                self.rearticulations.len(),
                "each adoption has a control sample"
            );
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
            now,
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
