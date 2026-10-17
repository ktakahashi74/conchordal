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
        if event.participation_onset
            && let Some(target) = self.participation_pitch.pending.take()
        {
            *pitch = target;
        }
        let tone = ToneSpec::from_event(
            *event,
            pitch.exp2(),
            target_amp,
            Some(Tick::MAX),
            self.body_snapshot(),
            self.articulation
                .render_modulator_spec(PhonationMode::Gated),
            self.voice_adsr,
        );
        self.participation_pitch.last_onset_log2 = Some(tone.freq_hz.log2());
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
            for step_index in 0..self.participation_pitch.steps.len() {
                let step = self.participation_pitch.steps[step_index];
                while on_index < self.phonation_scratch.events.len()
                    && self.phonation_scratch.events[on_index].onset_tick < step.at
                {
                    self.emit_bounded_on(on_index, &mut pitch, target_amp, out);
                    on_index += 1;
                }
                while m_index < self.participation_pitch.rearticulations.len()
                    && self.participation_pitch.rearticulations[m_index].at <= step.at
                {
                    let m = self.participation_pitch.rearticulations[m_index];
                    debug_assert!(space.is_some(), "pitch adoption needs the current grid");
                    // Release drops an adoption without a grid and keeps the processed pitch.
                    let Some(space) = space else {
                        m_index += 1;
                        continue;
                    };
                    let reference = self.participation_pitch.last_onset_log2.unwrap_or(pitch);
                    self.participation_pitch.adoption_application = select_pitch_application(
                        &self.effective_control.pitch,
                        Some((space, reference, m.pitch_log2)),
                    );
                    if self.participation_pitch.adoption_application == PitchApplyMode::Glide {
                        self.participation_pitch.pending = None;
                        self.participation_pitch.glide_target = Some(m.pitch_log2);
                    } else {
                        self.participation_pitch.pending = Some(m.pitch_log2);
                        self.participation_pitch.glide_target = None;
                    }
                    m_index += 1;
                }
                if let Some(target) = self.participation_pitch.glide_target {
                    pitch = Self::glide_pitch(pitch, target, step.dt, tau);
                    let onset = self
                        .phonation_scratch
                        .events
                        .get(on_index)
                        .is_some_and(|e| e.onset_tick == step.at);
                    let tone = on_index
                        .checked_sub(1)
                        .map(|i| self.phonation_scratch.events[i].tone_id)
                        .or(previous_tone);
                    let tone = Self::surviving_tone(
                        tone,
                        self.phonation_engine.rest.is_some(),
                        &out.cmds,
                        step.at,
                    );
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
            // Release keeps the processed pitch; the hop cleanup drops unmatched adoptions.
            debug_assert_eq!(
                m_index,
                self.participation_pitch.rearticulations.len(),
                "each adoption has a control sample"
            );
            self.body.set_pitch_log2(pitch);
        } else {
            self.participation_pitch.pending = None;
            self.participation_pitch.glide_target = None;
        }
        self.apply_phonation_onsets(consonance, out);
        self.update_surviving_render_notes(now, target_amp, out);
        self.finish_participation_batch(out);
    }
}
