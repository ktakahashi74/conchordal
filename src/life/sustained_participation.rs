use super::*;

impl Voice {
    fn emit_deferred_participation(
        &mut self,
        state: &CoreState,
        consonance: f32,
        target_amp: f32,
        now: Tick,
        out: &mut PhonationBatch,
    ) {
        for onset in &out.onsets {
            self.articulation
                .apply_phonation_onset(consonance, onset.strength);
        }
        let mut pitch = self.body.base_freq_hz().log2();
        let mut index = 0;
        let body = self.body_snapshot();
        let modulator = self
            .articulation
            .render_modulator_spec(PhonationMode::Gated);
        for event in &self.phonation_scratch.events {
            // Same-sample adoption precedes the opportunity; future adoptions stay pending.
            while index < self.rearticulations.len()
                && self.rearticulations[index].at <= event.onset_tick
            {
                if state.is_alive {
                    self.pending_participation_pitch = Some(self.rearticulations[index].pitch_log2);
                }
                index += 1;
            }
            if event.opportunity.is_some()
                && let Some(target) = self.pending_participation_pitch.take()
            {
                pitch = target;
            }
            out.tones.push(ToneSpec {
                opportunity: event.opportunity,
                tone_id: event.tone_id,
                onset: event.onset_tick,
                hold_ticks: Some(Tick::MAX),
                freq_hz: pitch.exp2(),
                amp: target_amp,
                smoothing_tau_sec: Self::PHONATION_UPDATE_SMOOTH_TAU_SEC,
                body: body.clone(),
                render_modulator: modulator.clone(),
                adsr: self.voice_adsr,
            });
        }
        if state.is_alive {
            if let Some(adoption) = self.rearticulations.get(index..).and_then(|ms| ms.last()) {
                self.pending_participation_pitch = Some(adoption.pitch_log2);
            }
            self.body.set_pitch_log2(pitch);
        } else {
            self.pending_participation_pitch = None;
        }
        self.prune_tracked_render_notes(&out.cmds);
        // Held tones and their release tails never receive pitch updates in this mode.
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

    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_sustained_participation(
        &mut self,
        ctx: &CoreTickCtx,
        state: &CoreState,
        consonance: f32,
        target_amp: f32,
        arrival: Option<&crate::life::arrival_cost::ArrivalContext<'_>>,
        space: Option<&Log2Space>,
        out: &mut PhonationBatch,
    ) {
        let deferred = self.defers_participation_pitch();
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
        let mut cursor = ctx.now_tick;
        if state.is_alive {
            for step in &self.participation_pitch_steps {
                assert!(step.at >= ctx.now_tick && step.at < ctx.frame_end);
                if cursor < step.at {
                    let segment = CoreTickCtx {
                        now_tick: cursor,
                        frame_end: step.at,
                        ..*ctx
                    };
                    self.phonation_engine.set_adoption_candidate(
                        self.rearticulations
                            .iter()
                            .filter(|_| !deferred)
                            .find(|m| m.at >= cursor)
                            .map(|m| m.at),
                    );
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
                }
                let segment = CoreTickCtx {
                    now_tick: step.at,
                    frame_end: step.at.saturating_add(1),
                    ..*ctx
                };
                self.phonation_engine.set_adoption_candidate(
                    self.rearticulations
                        .iter()
                        .filter(|_| !deferred)
                        .find(|m| m.at >= step.at)
                        .map(|m| m.at),
                );
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
            }
        }
        if deferred
            && self.pitch_application == PitchApplyMode::Rearticulate
            && (self.effective_control.pitch.rearticulate_above_cents > 0.0
                || self.participation_glide_target.is_some())
        {
            self.emit_bounded_participation(
                state,
                consonance,
                target_amp,
                ctx.now_tick,
                space,
                out,
            );
            return;
        }
        if deferred {
            self.emit_deferred_participation(state, consonance, target_amp, ctx.now_tick, out);
            return;
        }
        // Resolve sound production at sample times; never apply a future target to an earlier On.
        let mut pitch = self
            .participation_pitch_steps
            .first()
            .map_or(self.body.base_freq_hz().log2(), |s| s.before_log2);
        let initial_pitch = pitch;
        let tau = self.effective_control.pitch.pitch_glide_tau_sec.max(0.0);
        for step in &mut self.participation_pitch_steps {
            let onset = self
                .phonation_scratch
                .events
                .iter()
                .any(|e| e.onset_tick == step.at);
            let adoption = self.rearticulations.iter().find(|m| m.at == step.at);
            if onset && adoption.is_some() {
                pitch = step.target_log2;
            } else {
                let alpha = if tau <= 1e-6 {
                    1.0
                } else {
                    1.0 - (-step.dt.max(0.0) / tau).exp()
                };
                pitch += (step.target_log2 - pitch) * alpha.clamp(0.0, 1.0);
            }
            step.before_log2 = pitch;
            // Updates follow only the tone that survives at this sample. Released tails keep pitch.
            if state.is_alive {
                let tone = self
                    .phonation_scratch
                    .events
                    .iter()
                    .rev()
                    .find(|e| e.onset_tick <= step.at)
                    .map(|e| e.tone_id)
                    .or(previous_tone);
                let tone = tone.filter(|tone_id| {
                    self.phonation_engine.rest.is_none()
                        || !out.cmds.iter().any(|cmd| {
                            matches!(cmd, ToneCmd::Off { tone_id: off_id, off_tick }
                        if off_id == tone_id && *off_tick <= step.at)
                        })
                });
                if let Some(tone_id) = tone.filter(|_| !onset) {
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
        }
        if state.is_alive {
            self.body.set_pitch_log2(pitch);
        }
        for onset in &out.onsets {
            self.articulation
                .apply_phonation_onset(consonance, onset.strength);
        }
        self.prune_tracked_render_notes(&out.cmds);
        // Frequency updates above already carry their causal sample. Ordinary amp/drive changes stay here.
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
        let body = self.body_snapshot();
        let modulator = self
            .articulation
            .render_modulator_spec(PhonationMode::Gated);
        for event in self.phonation_scratch.events.drain(..) {
            let pitch = self
                .participation_pitch_steps
                .iter()
                .rev()
                .find(|s| s.at <= event.onset_tick)
                .map_or(initial_pitch, |s| s.before_log2);
            out.tones.push(ToneSpec {
                opportunity: event.opportunity,
                tone_id: event.tone_id,
                onset: event.onset_tick,
                hold_ticks: Some(Tick::MAX),
                freq_hz: pitch.exp2(),
                amp: target_amp,
                smoothing_tau_sec: Self::PHONATION_UPDATE_SMOOTH_TAU_SEC,
                body: body.clone(),
                render_modulator: modulator.clone(),
                adsr: self.voice_adsr,
            });
        }
        self.track_new_render_notes(&out.tones);
        self.prune_tracked_render_notes(&out.cmds);
        self.rearticulations.clear();
        self.participation_pitch_steps.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::{DurationConfig, OnsetConfig, PhonationClockConfig, PhonationConfig};
    fn setup(entrain: bool) -> (Voice, Timebase, NeuralRhythms, PhonationBatch) {
        let mut control = VoiceControl::default();
        control.pitch.mode = PitchMode::Free;
        control.pitch.freq = 330.0;
        control.pitch.pitch_apply_mode = PitchApplyMode::Glide;
        control.phonation.spec = crate::scenario::PhonationSpec {
            timing: crate::scenario::PhonationTiming::Coupled(crate::scenario::CoupledTimingSpec {
                relation: crate::scenario::RhythmRelation::Participating,
                coupling: 0.8,
                ..Default::default()
            }),
            duration: crate::scenario::DurationSpec::WhileAlive,
            sustain_participation: true,
        };
        let core = if entrain {
            ArticulationCoreConfig::default()
        } else {
            ArticulationCoreConfig::Drone {
                sway: None,
                breath_gain_init: None,
            }
        };
        let mut v = Voice::spawn_from_control(
            control,
            core,
            2,
            0,
            VoiceMetadata::default(),
            48_000.0,
            None,
            21,
        );
        v.phonation_engine = PhonationEngine::from_config(
            &PhonationConfig {
                mode: PhonationMode::Hold,
                clock: PhonationClockConfig::Participation {
                    coupling: 0.8,
                    base_rate_hz: 2.0,
                    flow_depth: 0.0,
                },
                onset: OnsetConfig::Always { strength: 1.0 },
                duration: DurationConfig::FixedGate { length_gates: 1 },
                ..Default::default()
            },
            21,
        );
        v.articulation.set_autonomous_attack_enabled(false);
        v.configure_control_hop(512, 64);
        let mut out = PhonationBatch::default();
        v.prepare_phonation_batch(&mut out);
        (
            v,
            Timebase {
                fs: 48_000.0,
                hop: 512,
            },
            NeuralRhythms::default(),
            out,
        )
    }
    fn emit(v: &mut Voice, tb: &Timebase, r: &NeuralRhythms, at: Tick, out: &mut PhonationBatch) {
        v.tick_phonation_into(tb, at, r, None, 0.0, 1.0, 0.8, out);
    }
    fn deferred() -> (Voice, Timebase, NeuralRhythms, PhonationBatch) {
        let (mut v, tb, r, out) = setup(false);
        v.apply_patch(&ControlUpdate {
            pitch_apply_mode: Some(PitchApplyMode::GateSnap),
            ..Default::default()
        })
        .unwrap();
        (v, tb, r, out)
    }
    fn adoption(v: &mut Voice, at: Tick, hz: f32) {
        v.rearticulations.push(Rearticulation {
            at,
            pitch_log2: hz.log2(),
        });
        v.participation_pitch_steps.push(ParticipationPitchStep {
            at,
            before_log2: v.body.base_freq_hz().log2(),
            target_log2: hz.log2(),
            dt: 64.0 / 48_000.0,
        });
    }
    #[test]
    fn pending_participation_pitch_replaces_then_consumes_at_one_clock_on() {
        let (mut v, tb, r, mut out) = deferred();
        emit(&mut v, &tb, &r, 0, &mut out);
        let initial = out.tones[0].freq_hz;
        adoption(&mut v, 512, 440.0);
        emit(&mut v, &tb, &r, 512, &mut out);
        assert!(out.tones.is_empty());
        assert_eq!(v.body.base_freq_hz().to_bits(), initial.to_bits());
        adoption(&mut v, 1024, 550.0);
        emit(&mut v, &tb, &r, 1024, &mut out);
        assert_eq!(
            v.pending_participation_pitch.unwrap().to_bits(),
            550.0f32.log2().to_bits()
        );
        for frame in 3..100 {
            emit(&mut v, &tb, &r, frame * 512, &mut out);
            if !out.tones.is_empty() {
                assert_eq!(out.tones.len(), 1);
                assert_eq!(out.onsets.len(), 1);
                assert_eq!(
                    out.cmds
                        .iter()
                        .filter(|c| matches!(c, ToneCmd::On { .. }))
                        .count(),
                    1
                );
                assert_eq!(
                    out.tones[0].freq_hz.to_bits(),
                    550.0f32.log2().exp2().to_bits()
                );
                assert!(v.pending_participation_pitch.is_none());
                assert_eq!(v.active_render_notes.len(), 1);
                return;
            }
            assert_eq!(v.body.base_freq_hz().to_bits(), initial.to_bits());
            assert!(v.pending_participation_pitch.is_some());
        }
        panic!("clock must produce an onset");
    }
    #[test]
    fn deferred_participation_multiple_onsets_follow_only_past_adoptions() {
        let (mut v, tb, r, mut out) = deferred();
        emit(&mut v, &tb, &r, 0, &mut out);
        out.clear();
        adoption(&mut v, 562, 440.0);
        adoption(&mut v, 712, 550.0);
        adoption(&mut v, 912, 660.0);
        let opportunity = crate::life::action_candidates::OnsetOpportunity {
            issued_at: 512,
            at: 612,
            gate: 1,
            intrinsic_due_at: None,
            intrinsic_period_ticks: None,
            planned_release_at: None,
        };
        v.phonation_scratch.events.push(ToneOnEvent {
            tone_id: 7,
            onset_tick: 612,
            opportunity: Some(opportunity),
        });
        v.phonation_scratch.events.push(ToneOnEvent {
            tone_id: 8,
            onset_tick: 812,
            opportunity: Some(crate::life::action_candidates::OnsetOpportunity {
                at: 812,
                ..opportunity
            }),
        });
        let ctx = CoreTickCtx {
            now_tick: 512,
            frame_end: 1024,
            fs: tb.fs,
            rhythms: r,
        };
        v.emit_deferred_participation(
            &CoreState {
                is_alive: true,
                onset_allowed: true,
            },
            0.8,
            0.1,
            ctx.now_tick,
            &mut out,
        );
        assert_eq!(out.tones.len(), 2);
        assert_eq!(
            out.tones[0].freq_hz.to_bits(),
            440.0f32.log2().exp2().to_bits()
        );
        assert_eq!(
            out.tones[1].freq_hz.to_bits(),
            550.0f32.log2().exp2().to_bits()
        );
        assert_eq!(
            v.body.base_freq_hz().to_bits(),
            out.tones[1].freq_hz.to_bits()
        );
        assert_eq!(
            v.pending_participation_pitch.unwrap().to_bits(),
            660.0f32.log2().to_bits()
        );
    }
    #[test]
    fn deferred_participation_body_rejection_keeps_pending_until_a_later_clock() {
        let (mut v, tb, r, mut out) = deferred();
        emit(&mut v, &tb, &r, 0, &mut out);
        let initial = v.body.base_freq_hz();
        adoption(&mut v, 512, 440.0);
        out.clear();
        let ctx = CoreTickCtx {
            now_tick: 512,
            frame_end: 1024,
            fs: tb.fs,
            rhythms: r,
        };
        v.emit_sustained_participation(
            &ctx,
            &CoreState {
                is_alive: true,
                onset_allowed: false,
            },
            0.8,
            0.1,
            None,
            None,
            &mut out,
        );
        assert!(out.tones.is_empty() && out.onsets.is_empty());
        assert_eq!(v.body.base_freq_hz().to_bits(), initial.to_bits());
        assert_eq!(
            v.pending_participation_pitch.unwrap().to_bits(),
            440.0f32.log2().to_bits()
        );
    }
    #[test]
    fn deferred_participation_control_holds_body_while_target_changes() {
        let (mut v, _, r, _) = deferred();
        v.pitch_ctl.force_set_target_pitch_log2(440.0f32.log2());
        v.control_tick = Some(0);
        let initial = v.body.base_freq_hz();
        v.update_articulation_autonomous(64.0 / 48_000.0, &r);
        assert_eq!(v.body.base_freq_hz().to_bits(), initial.to_bits());
        assert_eq!(
            v.participation_pitch_steps[0].target_log2.to_bits(),
            440.0f32.log2().to_bits()
        );
    }
    #[test]
    fn deferred_participation_future_adoption_does_not_change_initial_on_in_same_hop() {
        let (mut v, tb, r, mut out) = deferred();
        adoption(&mut v, 128, 440.0);
        emit(&mut v, &tb, &r, 0, &mut out);
        assert_eq!(out.tones.len(), 1);
        assert_eq!(out.tones[0].onset, 0);
        assert_eq!(
            out.tones[0].freq_hz.to_bits(),
            330.0f32.log2().exp2().to_bits()
        );
        assert_eq!(
            v.body.base_freq_hz().to_bits(),
            out.tones[0].freq_hz.to_bits()
        );
        assert_eq!(
            v.pending_participation_pitch.unwrap().to_bits(),
            440.0f32.log2().to_bits()
        );
        assert!(!out.cmds.iter().any(
            |c| matches!(c, ToneCmd::Update { update, .. } if update.target_freq_hz.is_some())
        ));
    }
    #[test]
    fn deferred_participation_skip_preserves_pending_pitch_and_tone() {
        use crate::core::temporal_expectation::TemporalForecast;
        let (mut v, tb, r, mut out) = deferred();
        emit(&mut v, &tb, &r, 0, &mut out);
        let freq = v.body.base_freq_hz();
        adoption(&mut v, 512, 440.0);
        v.phonation_engine
            .set_onset_comparison(Some(crate::config::TemporalOnsetComparisonConfig::default()));
        v.phonation_engine.set_decision_trace(true);
        let forecast = TemporalForecast::energy_fixture(48_000, 0, |_| [10.0, 0.0, 0.0]);
        v.phonation_engine
            .set_temporal_context(Some(&forecast), [1.0, 0.0, 0.0]);
        for frame in 1..100 {
            emit(&mut v, &tb, &r, frame * 512, &mut out);
            if v.phonation_engine
                .drain_participation_decisions()
                .any(|d| d.skipped)
            {
                assert!(out.onsets.is_empty());
                assert!(!out.cmds.iter().any(|c| matches!(c, ToneCmd::Off { .. })));
                assert_eq!(v.body.base_freq_hz().to_bits(), freq.to_bits());
                assert_eq!(
                    v.pending_participation_pitch.unwrap().to_bits(),
                    440.0f32.log2().to_bits()
                );
                assert_eq!(v.active_render_notes.len(), 1);
                return;
            }
        }
        panic!("fixture must skip");
    }
    #[test]
    fn deferred_participation_lifecycle_cancels_pending_and_future_adoptions() {
        for reason in 0..3 {
            let (mut v, tb, r, mut out) = setup(true);
            v.apply_patch(&ControlUpdate {
                pitch_apply_mode: Some(PitchApplyMode::GateSnap),
                ..Default::default()
            })
            .unwrap();
            emit(&mut v, &tb, &r, 0, &mut out);
            v.pending_participation_pitch = Some(440.0f32.log2());
            adoption(&mut v, 576, 550.0);
            match reason {
                0 => v.release_pending = true,
                1 => v.remove_pending = true,
                _ => {
                    let crate::life::articulation_core::AnyArticulationCore::Entrain(core) =
                        &mut v.articulation.core
                    else {
                        unreachable!()
                    };
                    core.energy = 0.0;
                    core.state = crate::life::articulation_core::ArticulationState::Idle;
                }
            }
            emit(&mut v, &tb, &r, 512, &mut out);
            assert!(out.onsets.is_empty() && out.tones.is_empty());
            assert!(v.pending_participation_pitch.is_none());
            assert!(v.rearticulations.is_empty());
            assert!(out.cmds.iter().any(|c| matches!(c, ToneCmd::Off { .. })));
        }
    }
    #[test]
    #[cfg(feature = "profile-alloc")]
    fn deferred_participation_emission_reuses_birth_storage() {
        let (mut v, tb, r, mut out) = deferred();
        emit(&mut v, &tb, &r, 0, &mut out);
        v.phonation_engine.clock = crate::life::phonation_engine::PhonationClock::from_config(
            &PhonationClockConfig::Participation {
                coupling: 0.8,
                base_rate_hz: crate::life::phonation_engine::MAX_ONSET_RATE_HZ,
                flow_depth: 0.0,
            },
            21,
        );
        emit(&mut v, &tb, &r, 512, &mut out);
        adoption(&mut v, 1024, 440.0);
        adoption(&mut v, 1088, 550.0);
        crate::runtime_profile::begin_allocations();
        emit(&mut v, &tb, &r, 1024, &mut out);
        let counts = crate::runtime_profile::finish_allocations().unwrap();
        assert!(!out.onsets.is_empty());
        assert_eq!((counts.count, counts.bytes), (0, 0));
    }

    #[test]
    fn deferred_participation_live_switch_retains_tone_and_controller_destination() {
        let (mut v, tb, r, mut out) = deferred();
        assert!(v.pending_participation_pitch.is_none());
        emit(&mut v, &tb, &r, 0, &mut out);
        let initial = v.body.base_freq_hz();
        let tone = v.active_render_notes[0].tone_id;
        v.pitch_ctl.force_set_target_pitch_log2(440.0f32.log2());
        adoption(&mut v, 576, 440.0);
        emit(&mut v, &tb, &r, 512, &mut out);
        assert!(v.pending_participation_pitch.is_some());
        v.apply_patch(&ControlUpdate {
            pitch_apply_mode: Some(PitchApplyMode::Glide),
            pitch_glide_tau_sec: Some(0.03),
            ..Default::default()
        })
        .unwrap();
        assert!(v.pending_participation_pitch.is_none());
        assert_eq!(v.active_render_notes[0].tone_id, tone);
        assert_eq!(v.body.base_freq_hz(), initial);
        emit(&mut v, &tb, &r, 1024, &mut out);
        assert!(out.tones.is_empty() && out.onsets.is_empty());
        v.set_control_tick(1536);
        v.update_articulation_autonomous(64.0 / 48_000.0, &r);
        assert!(v.body.base_freq_hz() > initial && v.body.base_freq_hz() < 440.0);
        emit(&mut v, &tb, &r, 1536, &mut out);
        v.apply_patch(&ControlUpdate {
            pitch_apply_mode: Some(PitchApplyMode::GateSnap),
            ..Default::default()
        })
        .unwrap();
        assert!(v.pending_participation_pitch.is_none());
        let held = v.body.base_freq_hz();
        v.set_control_tick(2048);
        v.update_articulation_autonomous(64.0 / 48_000.0, &r);
        emit(&mut v, &tb, &r, 2048, &mut out);
        assert_eq!(v.body.base_freq_hz(), held);
        assert!(out.tones.is_empty() && out.onsets.is_empty());
        assert_eq!(v.active_render_notes[0].tone_id, tone);
    }

    #[test]
    fn deferred_participation_same_sample_adoption_precedes_opportunity() {
        let (mut v, tb, r, mut out) = deferred();
        emit(&mut v, &tb, &r, 0, &mut out);
        out.clear();
        adoption(&mut v, 576, 440.0);
        v.phonation_scratch.events.push(ToneOnEvent {
            tone_id: 7,
            onset_tick: 576,
            opportunity: Some(crate::life::action_candidates::OnsetOpportunity {
                issued_at: 512,
                at: 576,
                gate: 1,
                intrinsic_due_at: None,
                intrinsic_period_ticks: None,
                planned_release_at: None,
            }),
        });
        v.emit_deferred_participation(
            &CoreState {
                is_alive: true,
                onset_allowed: true,
            },
            0.8,
            0.1,
            512,
            &mut out,
        );
        assert_eq!(out.tones.len(), 1);
        assert_eq!(
            out.tones[0].freq_hz.to_bits(),
            440.0f32.log2().exp2().to_bits()
        );
        assert!(v.pending_participation_pitch.is_none());
    }

    #[test]
    fn sustained_participation_clock_restarts_same_pitch_with_one_on_and_unbounded_hold() {
        let (mut v, tb, r, mut out) = setup(false);
        let mut ons = 0;
        for frame in 0..100 {
            emit(&mut v, &tb, &r, frame * 512, &mut out);
            ons += out.onsets.len();
            for t in &out.tones {
                assert_eq!(t.hold_ticks, Some(Tick::MAX));
                assert_eq!(t.freq_hz.to_bits(), 330.0f32.log2().exp2().to_bits());
            }
            assert!(out.tones.len() <= 1);
            assert_eq!(v.active_render_notes.len(), 1);
        }
        assert!(ons > 1);
    }
    #[test]
    fn sustained_participation_m_and_o_same_sample_merge_and_off_precedes_new_pitch() {
        let (mut v, tb, r, mut out) = setup(false);
        emit(&mut v, &tb, &r, 0, &mut out);
        let old = out.tones[0].tone_id;
        let mut at = None;
        for frame in 1..100 {
            emit(&mut v, &tb, &r, frame * 512, &mut out);
            if let Some(t) = out.tones.first() {
                at = Some(t.onset + 24_000);
                break;
            }
        }
        let at = at.unwrap();
        let start = at / 512 * 512;
        // Reach the identical future clock sample with an adoption in that control slot.
        v.rearticulations.push(Rearticulation {
            at,
            pitch_log2: 440.0f32.log2(),
        });
        v.participation_pitch_steps.push(ParticipationPitchStep {
            at,
            before_log2: 330.0f32.log2(),
            target_log2: 440.0f32.log2(),
            dt: 64.0 / 48_000.0,
        });
        emit(&mut v, &tb, &r, start, &mut out);
        assert_eq!(out.onsets.iter().filter(|o| o.onset_tick == at).count(), 1);
        assert_eq!(
            out.tones[0].freq_hz.to_bits(),
            440.0f32.log2().exp2().to_bits()
        );
        assert!(
            !out.cmds
                .iter()
                .any(|c| matches!(c,ToneCmd::Update{tone_id,..} if *tone_id==old))
        );
    }
    #[test]
    fn sustained_participation_release_death_remove_cancel_pending_m_and_clock() {
        for reason in 0..3 {
            let (mut v, tb, r, mut out) = setup(true);
            emit(&mut v, &tb, &r, 0, &mut out);
            v.rearticulations.push(Rearticulation {
                at: 576,
                pitch_log2: 440.0f32.log2(),
            });
            v.participation_pitch_steps.push(ParticipationPitchStep {
                at: 576,
                before_log2: 330.0f32.log2(),
                target_log2: 440.0f32.log2(),
                dt: 64.0 / 48_000.0,
            });
            match reason {
                0 => v.release_pending = true,
                1 => v.remove_pending = true,
                _ => {
                    let crate::life::articulation_core::AnyArticulationCore::Entrain(core) =
                        &mut v.articulation.core
                    else {
                        unreachable!()
                    };
                    core.energy = 0.0;
                    core.state = crate::life::articulation_core::ArticulationState::Idle;
                }
            }
            emit(&mut v, &tb, &r, 512, &mut out);
            assert!(out.onsets.is_empty());
            assert!(out.tones.is_empty());
            assert!(out.cmds.iter().any(|c| matches!(c, ToneCmd::Off { .. })));
            assert!(v.rearticulations.is_empty());
        }
    }
    #[test]
    fn sustained_participation_skip_keeps_the_current_tone() {
        use crate::core::temporal_expectation::TemporalForecast;
        let (mut v, tb, r, mut out) = setup(false);
        v.phonation_engine
            .set_onset_comparison(Some(crate::config::TemporalOnsetComparisonConfig::default()));
        v.phonation_engine.set_decision_trace(true);
        let forecast = TemporalForecast::energy_fixture(48_000, 0, |_| [10.0, 0.0, 0.0]);
        v.phonation_engine
            .set_temporal_context(Some(&forecast), [1.0, 0.0, 0.0]);
        let mut saw_skip = false;
        for frame in 0..100 {
            emit(&mut v, &tb, &r, frame * 512, &mut out);
            if v.phonation_engine
                .drain_participation_decisions()
                .any(|d| d.skipped)
            {
                saw_skip = true;
                assert!(out.onsets.is_empty());
                assert!(!out.cmds.iter().any(|c| matches!(c, ToneCmd::Off { .. })));
                assert!(v.phonation_engine.has_active_notes());
                assert_eq!(v.active_render_notes.len(), 1);
                break;
            }
        }
        assert!(saw_skip);
    }
    #[test]
    fn sustained_participation_energy_changes_once_per_on_and_autonomous_pulse_is_absent() {
        let (mut v, tb, r, mut out) = setup(true);
        let crate::life::articulation_core::AnyArticulationCore::Entrain(core) =
            &mut v.articulation.core
        else {
            unreachable!()
        };
        core.energy = 0.8;
        core.attack_cost_fraction = 0.1;
        core.attack_recharge_fraction = 0.0;
        let mut expected = 0.8f32;
        for frame in 0..100 {
            emit(&mut v, &tb, &r, frame * 512, &mut out);
            for _ in &out.onsets {
                expected -= 0.1;
            }
            let crate::life::articulation_core::AnyArticulationCore::Entrain(core) =
                &v.articulation.core
            else {
                unreachable!()
            };
            assert_eq!(core.energy.to_bits(), expected.to_bits());
            for tone in &out.tones {
                assert!(matches!(
                    tone.render_modulator,
                    RenderModulatorSpec::EntrainPulse {
                        autonomous_pulse: None,
                        ..
                    }
                ));
            }
        }
    }
    #[test]
    #[cfg(feature = "profile-alloc")]
    fn sustained_participation_emission_reuses_birth_storage() {
        let (mut v, tb, r, mut out) = setup(false);
        emit(&mut v, &tb, &r, 0, &mut out);
        // Warm the existing clock scratch, then measure a hop containing real clock On events.
        v.phonation_engine.clock = crate::life::phonation_engine::PhonationClock::from_config(
            &PhonationClockConfig::Participation {
                coupling: 0.8,
                base_rate_hz: crate::life::phonation_engine::MAX_ONSET_RATE_HZ,
                flow_depth: 0.0,
            },
            21,
        );
        emit(&mut v, &tb, &r, 512, &mut out);
        assert!(!out.onsets.is_empty());
        for i in 0..(tb.hop / 64) as u64 {
            v.participation_pitch_steps.push(ParticipationPitchStep {
                at: 1024 + i * 64,
                before_log2: 330.0f32.log2(),
                target_log2: 350.0f32.log2(),
                dt: 64.0 / 48_000.0,
            });
        }
        v.rearticulations.push(Rearticulation {
            at: 1024,
            pitch_log2: 350.0f32.log2(),
        });
        crate::runtime_profile::begin_allocations();
        emit(&mut v, &tb, &r, 1024, &mut out);
        let counts = crate::runtime_profile::finish_allocations().unwrap();
        assert!(!out.onsets.is_empty());
        assert_eq!(counts.count, 0);
        assert_eq!(counts.bytes, 0);
    }
    #[test]
    fn sustained_participation_earlier_clock_on_does_not_use_future_target() {
        let (mut v, tb, r, mut out) = setup(false);
        emit(&mut v, &tb, &r, 0, &mut out);
        let at = (1..100)
            .find_map(|frame| {
                emit(&mut v, &tb, &r, frame * 512, &mut out);
                out.tones.first().map(|t| t.onset + 24_000)
            })
            .unwrap();
        let later = at + 1;
        v.rearticulations.push(Rearticulation {
            at: later,
            pitch_log2: 440.0f32.log2(),
        });
        v.participation_pitch_steps.push(ParticipationPitchStep {
            at: later,
            before_log2: 330.0f32.log2(),
            target_log2: 440.0f32.log2(),
            dt: 64.0 / 48_000.0,
        });
        emit(&mut v, &tb, &r, at / 512 * 512, &mut out);
        let earlier = out.tones.iter().find(|t| t.onset == at).unwrap();
        assert_eq!(earlier.freq_hz.to_bits(), 330.0f32.log2().exp2().to_bits());
        assert!(v.body.base_freq_hz() > earlier.freq_hz);
        assert_eq!(out.onsets.len(), 1);
    }

    #[test]
    fn sustained_participation_live_glide_preserves_requests_and_clock() {
        let (mut v, tb, r, mut out) = setup(false);
        emit(&mut v, &tb, &r, 0, &mut out);
        let tone = out.tones[0].tone_id;
        v.rearticulations.push(Rearticulation {
            at: 576,
            pitch_log2: 350.0f32.log2(),
        });
        v.apply_patch(&crate::scenario::control::ControlUpdate {
            pitch_apply_mode: Some(PitchApplyMode::Glide),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(v.rearticulations.len(), 1);
        assert_eq!(v.active_render_notes[0].tone_id, tone);
        assert!(v.phonation_engine.sustain_participation);
        emit(&mut v, &tb, &r, 512, &mut out);
        assert!(v.rearticulations.is_empty());
        assert!(out.tones.is_empty());
        v.base_control.pitch.mode = PitchMode::Lock;
        v.effective_control.pitch.mode = PitchMode::Lock;
        for mode in [
            PitchApplyMode::GateSnap,
            PitchApplyMode::Glide,
            PitchApplyMode::GateSnap,
        ] {
            v.apply_patch(&crate::scenario::control::ControlUpdate {
                pitch_apply_mode: Some(mode),
                ..Default::default()
            })
            .unwrap();
            assert_eq!(v.active_render_notes[0].tone_id, tone);
            assert!(v.phonation_engine.sustain_participation);
        }
    }

    #[test]
    #[cfg(feature = "profile-alloc")]
    fn sustained_participation_capacities_follow_control_hop() {
        for hop in [257, 1024] {
            let (mut v, mut tb, r, mut out) = setup(false);
            tb.hop = hop;
            v.configure_control_hop(hop, 64);
            v.prepare_phonation_batch(&mut out);
            emit(&mut v, &tb, &r, 0, &mut out);
            for frame in 1..4 {
                emit(&mut v, &tb, &r, frame * hop as u64, &mut out);
            }
            let start = 4 * hop as u64;
            for i in 0..(hop / 64).max(1) {
                v.participation_pitch_steps.push(ParticipationPitchStep {
                    at: start + (i * 64) as u64,
                    before_log2: v.body.base_freq_hz().log2(),
                    target_log2: 350.0f32.log2(),
                    dt: 64.0 / 48_000.0,
                });
            }
            crate::runtime_profile::begin_allocations();
            emit(&mut v, &tb, &r, start, &mut out);
            let counts = crate::runtime_profile::finish_allocations().unwrap();
            assert_eq!((counts.count, counts.bytes), (0, 0));
        }
    }
    fn rest_setup(periods: f32) -> (Voice, Timebase, NeuralRhythms, PhonationBatch) {
        let (mut v, tb, r, out) = setup(false);
        v.phonation_engine.enable_rest(
            &DurationConfig::Field {
                hold_min_theta: 0.2,
                hold_max_theta: 0.2,
                curve_k: 2.,
                curve_x0: 0.5,
                drop_gain: 0.5,
            },
            periods,
        );
        (v, tb, r, out)
    }
    #[test]
    fn rest_finite_release_g1_waits_then_consumes_without_forced_on() {
        use crate::life::phonation_engine::RestPhase;
        let (mut v, tb, r, mut out) = rest_setup(1.);
        emit(&mut v, &tb, &r, 0, &mut out);
        assert_eq!(out.tones.len(), 1);
        let off = match v.phonation_engine.rest.unwrap().phase {
            RestPhase::Continue(at) => at,
            _ => panic!(),
        };
        assert_eq!(off, 4800);
        let mut off_seen = 0;
        for at in (512..60000).step_by(512) {
            emit(&mut v, &tb, &r, at, &mut out);
            assert!(out.tones.is_empty());
            for cmd in &out.cmds {
                if matches!(cmd, ToneCmd::Off {off_tick, ..} if *off_tick == off) {
                    off_seen += 1;
                }
            }
        }
        assert_eq!(off_seen, 1);
        assert!(matches!(
            v.phonation_engine.rest.unwrap().phase,
            RestPhase::Waiting(4800)
        ));
        assert!(!v.phonation_engine.has_active_notes());
        assert_eq!(
            v.phonation_engine.drain_participation_predictions().count(),
            0
        );
        v.phonation_engine.rest_tail_finished(61000, tb.fs);
        assert_eq!(
            v.phonation_engine.rest.unwrap().phase,
            RestPhase::Gap {
                origin: 61000,
                until: Some(85000)
            }
        );
        for at in (60416..85000).step_by(512) {
            emit(&mut v, &tb, &r, at, &mut out);
            assert!(out.tones.is_empty());
        }
        v.phonation_engine.rest_resume(85000);
        assert_eq!(v.phonation_engine.rest.unwrap().phase, RestPhase::Ready);
        let mut found = false;
        for at in (85504..160000).step_by(512) {
            emit(&mut v, &tb, &r, at, &mut out);
            if !out.tones.is_empty() {
                assert!(out.tones[0].onset > 85000);
                found = true;
                break;
            }
        }
        assert!(found);
    }
    #[test]
    #[cfg(feature = "profile-alloc")]
    fn rest_warmed_continue_gap_and_receipt_allocate_zero() {
        let (mut v, tb, r, mut out) = rest_setup(1.);
        for at in (0..24000).step_by(512) {
            emit(&mut v, &tb, &r, at, &mut out);
        }
        for at in (24064..90000).step_by(512) {
            crate::runtime_profile::begin_allocations();
            if at == 24064 {
                v.phonation_engine.rest_tail_finished(at, tb.fs);
            }
            emit(&mut v, &tb, &r, at, &mut out);
            let counts = crate::runtime_profile::finish_allocations().unwrap();
            assert_eq!((counts.count, counts.bytes), (0, 0), "at {at}");
        }
    }

    #[test]
    fn rest_zero_fraction_and_unknown_period_preserve_definition() {
        use crate::life::phonation_engine::RestPhase;
        for (periods, until) in [(0., 6000), (0.5, 18000), (2., 54000)] {
            let (mut v, tb, _, _) = rest_setup(periods);
            v.phonation_engine.rest.as_mut().unwrap().phase = RestPhase::Waiting(4800);
            v.phonation_engine.rest_tail_finished(6000, tb.fs);
            assert_eq!(
                v.phonation_engine.rest.unwrap().phase,
                RestPhase::Gap {
                    origin: 6000,
                    until: Some(until)
                }
            );
            v.phonation_engine.rest_resume(until.saturating_sub(1));
            assert!(matches!(
                v.phonation_engine.rest.unwrap().phase,
                RestPhase::Gap { .. }
            ));
            v.phonation_engine.rest_resume(until);
            assert_eq!(v.phonation_engine.rest.unwrap().phase, RestPhase::Ready);
        }
        let (mut v, tb, _, _) = rest_setup(1.);
        v.phonation_engine.rest.as_mut().unwrap().phase = RestPhase::Waiting(4800);
        v.phonation_engine.clock = crate::life::phonation_engine::PhonationClock::from_config(
            &PhonationClockConfig::ThetaGate,
            21,
        );
        v.phonation_engine.rest_tail_finished(6000, tb.fs);
        v.phonation_engine.rest_resume(Tick::MAX);
        assert_eq!(
            v.phonation_engine.rest.unwrap().phase,
            RestPhase::Gap {
                origin: 6000,
                until: None
            }
        );
    }

    #[test]
    fn rest_intent_consumes_opportunities_without_cost_skip_or_prediction() {
        let (mut v, tb, r, mut out) = rest_setup(1.);
        v.phonation_engine
            .set_onset_comparison(Some(crate::config::TemporalOnsetComparisonConfig::default()));
        v.phonation_engine.set_decision_trace(true);
        let forecast =
            crate::core::temporal_expectation::TemporalForecast::energy_fixture(48000, 0, |_| {
                [10., 0., 0.]
            });
        v.phonation_engine
            .set_temporal_context(Some(&forecast), [1., 0., 0.]);
        let mut suppressed = 0;
        for at in (0..120000).step_by(512) {
            emit(&mut v, &tb, &r, at, &mut out);
            for d in v.phonation_engine.drain_participation_decisions() {
                if !d.onset_allowed {
                    suppressed += 1;
                    assert!(!d.skipped);
                }
            }
            assert_eq!(
                v.phonation_engine.drain_participation_predictions().count(),
                0
            );
        }
        assert!(suppressed > 0);
    }

    #[test]
    fn rest_gate_snap_keeps_latest_target_through_continue_tail_and_gap() {
        let (mut v, tb, r, mut out) = rest_setup(1.);
        v.apply_patch(&ControlUpdate {
            pitch_apply_mode: Some(PitchApplyMode::GateSnap),
            ..Default::default()
        })
        .unwrap();
        adoption(&mut v, 64, 440.);
        emit(&mut v, &tb, &r, 0, &mut out);
        assert_eq!(out.tones.len(), 1);
        assert_eq!(
            out.tones[0].freq_hz.to_bits(),
            330.0f32.log2().exp2().to_bits()
        );
        for at in (512..24576).step_by(512) {
            if at == 512 {
                adoption(&mut v, 576, 550.);
            }
            emit(&mut v, &tb, &r, at, &mut out);
            assert!(out.tones.is_empty());
            assert_eq!(
                v.body.base_freq_hz().to_bits(),
                330.0f32.log2().exp2().to_bits()
            );
        }
        v.phonation_engine.rest_tail_finished(24576, tb.fs);
        for at in (24576..48640).step_by(512) {
            if at == 24576 {
                adoption(&mut v, 24640, 660.);
            }
            emit(&mut v, &tb, &r, at, &mut out);
            assert!(out.tones.is_empty());
        }
        assert_eq!(v.pending_participation_pitch, Some(660.0f32.log2()));
        let mut on = None;
        for at in (48640..180224).step_by(512) {
            emit(&mut v, &tb, &r, at, &mut out);
            if let Some(tone) = out.tones.first() {
                on = Some(tone.clone());
                break;
            }
        }
        let tone = on.expect("normal participation resumes");
        assert!(tone.onset >= 48576);
        assert!(tone.opportunity.is_some());
        assert_eq!(tone.freq_hz.to_bits(), 660.0f32.log2().exp2().to_bits());
        assert!(v.pending_participation_pitch.is_none());
    }

    #[test]
    fn rest_lifecycle_cancels_finite_off_pending_target_and_future_on() {
        use crate::life::phonation_engine::RestPhase;
        for reason in 0..3 {
            let (mut v, tb, r, mut out) = setup(true);
            v.phonation_engine
                .enable_rest(&DurationConfig::FixedGate { length_gates: 2 }, 1.);
            v.apply_patch(&ControlUpdate {
                pitch_apply_mode: Some(PitchApplyMode::GateSnap),
                ..Default::default()
            })
            .unwrap();
            emit(&mut v, &tb, &r, 0, &mut out);
            adoption(&mut v, 576, 440.);
            match reason {
                0 => v.release_pending = true,
                1 => v.remove_pending = true,
                _ => {
                    let crate::life::articulation_core::AnyArticulationCore::Entrain(core) =
                        &mut v.articulation.core
                    else {
                        panic!()
                    };
                    core.energy = 0.;
                    core.state = crate::life::articulation_core::ArticulationState::Idle;
                }
            }
            emit(&mut v, &tb, &r, 512, &mut out);
            assert!(out.onsets.is_empty() && out.tones.is_empty());
            assert_eq!(
                out.cmds
                    .iter()
                    .filter(|c| matches!(c, ToneCmd::Off { .. }))
                    .count(),
                1
            );
            assert!(v.pending_participation_pitch.is_none());
            assert_eq!(v.phonation_engine.rest.unwrap().phase, RestPhase::Finished);
            v.phonation_engine.rest_tail_finished(6000, tb.fs);
            for at in (1024..60000).step_by(512) {
                emit(&mut v, &tb, &r, at, &mut out);
                assert!(out.tones.is_empty());
                assert!(!out.cmds.iter().any(|c| matches!(c, ToneCmd::Off { .. })));
            }
        }
    }

    #[test]
    fn rest_fresh_birth_keeps_config_without_parent_gap_or_pending_pitch() {
        use crate::life::phonation_engine::RestPhase;
        let (mut parent, _, _, _) = rest_setup(0.5);
        parent.effective_control.phonation.rest_periods = Some(0.5);
        parent.effective_control.phonation.finite_duration =
            Some(crate::scenario::DurationSpec::Gates(2));
        parent.phonation_engine.rest.as_mut().unwrap().phase = RestPhase::Gap {
            origin: 6000,
            until: None,
        };
        parent.pending_participation_pitch = Some(660.0f32.log2());
        let child = Voice::spawn_from_control(
            parent.effective_control.clone(),
            ArticulationCoreConfig::default(),
            3,
            5,
            VoiceMetadata {
                generation: 1,
                parent_id: Some(parent.id()),
                ..Default::default()
            },
            48000.,
            None,
            21,
        );
        assert_eq!(child.phonation_engine.rest.unwrap().periods, 0.5);
        assert_eq!(child.phonation_engine.rest.unwrap().phase, RestPhase::Ready);
        assert!(child.pending_participation_pitch.is_none());
    }

    #[test]
    fn rest_glide_adoption_after_off_cannot_update_tail_or_create_on() {
        let (mut v, tb, r, mut out) = rest_setup(1.);
        emit(&mut v, &tb, &r, 0, &mut out);
        let first = out.tones[0].tone_id;
        adoption(&mut v, 576, 440.);
        emit(&mut v, &tb, &r, 512, &mut out);
        assert!(out.cmds.iter().any(|c|matches!(c,ToneCmd::Update { tone_id,at_tick:Some(576),update } if *tone_id==first && update.target_freq_hz.is_some())));
        for at in (1024..4608).step_by(512) {
            emit(&mut v, &tb, &r, at, &mut out);
        }
        adoption(&mut v, 4864, 550.);
        emit(&mut v, &tb, &r, 4608, &mut out);
        assert!(
            out.cmds
                .iter()
                .any(|c| matches!(c,ToneCmd::Off {tone_id,off_tick:4800} if *tone_id==first))
        );
        assert!(out.tones.is_empty());
        assert!(!out.cmds.iter().any(|c|matches!(c,ToneCmd::Update {tone_id,at_tick:Some(at),update} if *tone_id==first && *at>=4800 && update.target_freq_hz.is_some())));
        assert!(v.rearticulations.is_empty());
    }
    fn bounded_space() -> &'static Log2Space {
        static SPACE: std::sync::LazyLock<Log2Space> =
            std::sync::LazyLock::new(|| Log2Space::new(55., 8000., 96));
        &SPACE
    }
    fn bounded_emit(
        v: &mut Voice,
        tb: &Timebase,
        r: &NeuralRhythms,
        at: Tick,
        out: &mut PhonationBatch,
    ) {
        v.tick_phonation_into_with_arrival(
            tb,
            at,
            r,
            None,
            0.,
            1.,
            0.8,
            None,
            Some(bounded_space()),
            out,
        );
    }
    fn bounded_on(id: ToneId, at: Tick) -> ToneOnEvent {
        ToneOnEvent {
            tone_id: id,
            onset_tick: at,
            opportunity: Some(crate::life::action_candidates::OnsetOpportunity {
                issued_at: at,
                at,
                gate: 1,
                intrinsic_due_at: None,
                intrinsic_period_ticks: None,
                planned_release_at: None,
            }),
        }
    }
    fn bounded_fixture(entrain: bool) -> (Voice, Timebase, NeuralRhythms, PhonationBatch) {
        let (mut v, tb, r, out) = setup(entrain);
        v.apply_patch(&ControlUpdate {
            pitch_apply_mode: Some(PitchApplyMode::Rearticulate),
            rearticulate_above_cents: Some(50.0),
            pitch_glide_tau_sec: Some(0.03),
            ..Default::default()
        })
        .unwrap();
        (v, tb, r, out)
    }
    #[test]
    fn bounded_small_glides_without_attack_and_large_freezes() {
        let (mut v, tb, r, mut out) = bounded_fixture(false);
        bounded_emit(&mut v, &tb, &r, 0, &mut out);
        let onset = v.last_onset_pitch_log2;
        let initial = v.body.base_freq_hz();
        adoption(&mut v, 512, 335.0);
        bounded_emit(&mut v, &tb, &r, 512, &mut out);
        assert!(out.onsets.is_empty() && out.tones.is_empty());
        assert!(v.body.base_freq_hz() > initial && v.body.base_freq_hz() < 335.0);
        assert_eq!(v.last_onset_pitch_log2, onset);
        let before = v.body.base_freq_hz();
        adoption(&mut v, 1024, 440.0);
        bounded_emit(&mut v, &tb, &r, 1024, &mut out);
        assert!(out.onsets.is_empty());
        assert_eq!(v.body.base_freq_hz().to_bits(), before.to_bits());
        assert_eq!(v.pending_participation_pitch, Some(440.0f32.log2()));
    }
    #[test]
    fn bounded_crossing_cancels_the_old_pending_target() {
        let (mut v, tb, r, mut out) = bounded_fixture(false);
        bounded_emit(&mut v, &tb, &r, 0, &mut out);
        adoption(&mut v, 512, 440.0);
        bounded_emit(&mut v, &tb, &r, 512, &mut out);
        adoption(&mut v, 1024, 335.0);
        bounded_emit(&mut v, &tb, &r, 1024, &mut out);
        assert!(v.pending_participation_pitch.is_none());
        out.clear();
        v.phonation_scratch.events.push(bounded_on(70, 1536));
        v.emit_bounded_participation(
            &CoreState {
                is_alive: true,
                onset_allowed: true,
            },
            0.8,
            0.1,
            1536,
            Some(bounded_space()),
            &mut out,
        );
        assert_eq!(out.tones.len(), 1);
        assert!(out.tones[0].freq_hz < 335.0);
        assert!(v.pending_participation_pitch.is_none());
    }
    #[test]
    fn bounded_actual_earlier_on_changes_reference_for_later_m() {
        let (mut v, tb, r, mut out) = bounded_fixture(false);
        bounded_emit(&mut v, &tb, &r, 0, &mut out);
        out.clear();
        adoption(&mut v, 562, 440.0);
        adoption(&mut v, 712, 450.0);
        adoption(&mut v, 912, 660.0);
        let op = crate::life::action_candidates::OnsetOpportunity {
            issued_at: 512,
            at: 612,
            gate: 1,
            intrinsic_due_at: None,
            intrinsic_period_ticks: None,
            planned_release_at: None,
        };
        for (id, at) in [(7, 612), (8, 812)] {
            v.phonation_scratch.events.push(ToneOnEvent {
                tone_id: id,
                onset_tick: at,
                opportunity: Some(crate::life::action_candidates::OnsetOpportunity { at, ..op }),
            });
        }
        v.emit_bounded_participation(
            &CoreState {
                is_alive: true,
                onset_allowed: true,
            },
            0.8,
            0.1,
            512,
            Some(bounded_space()),
            &mut out,
        );
        assert_eq!(out.tones.len(), 2);
        assert_eq!(
            out.tones[0].freq_hz.to_bits(),
            440.0f32.log2().exp2().to_bits()
        );
        assert!(out.tones[1].freq_hz > 440.0 && out.tones[1].freq_hz < 450.0);
        assert_eq!(v.pending_participation_pitch, Some(660.0f32.log2()));
        assert!(v.participation_glide_target.is_none());
    }
    #[test]
    fn bounded_same_sample_m_precedes_exactly_one_opportunity() {
        let (mut v, tb, r, mut out) = bounded_fixture(false);
        bounded_emit(&mut v, &tb, &r, 0, &mut out);
        out.clear();
        adoption(&mut v, 576, 440.0);
        v.phonation_scratch.events.push(ToneOnEvent {
            tone_id: 7,
            onset_tick: 576,
            opportunity: Some(crate::life::action_candidates::OnsetOpportunity {
                issued_at: 512,
                at: 576,
                gate: 1,
                intrinsic_due_at: None,
                intrinsic_period_ticks: None,
                planned_release_at: None,
            }),
        });
        v.emit_bounded_participation(
            &CoreState {
                is_alive: true,
                onset_allowed: true,
            },
            0.8,
            0.1,
            512,
            Some(bounded_space()),
            &mut out,
        );
        assert_eq!(out.tones.len(), 1);
        assert_eq!(
            out.tones[0].freq_hz.to_bits(),
            440.0f32.log2().exp2().to_bits()
        );
        assert!(v.pending_participation_pitch.is_none());
    }
    #[test]
    fn bounded_body_rejection_keeps_pending_and_actual_reference() {
        let (mut v, tb, r, mut out) = bounded_fixture(false);
        bounded_emit(&mut v, &tb, &r, 0, &mut out);
        let actual = v.last_onset_pitch_log2;
        adoption(&mut v, 512, 440.0);
        out.clear();
        let ctx = CoreTickCtx {
            now_tick: 512,
            frame_end: 1024,
            fs: tb.fs,
            rhythms: r,
        };
        v.emit_sustained_participation(
            &ctx,
            &CoreState {
                is_alive: true,
                onset_allowed: false,
            },
            0.8,
            0.1,
            None,
            Some(bounded_space()),
            &mut out,
        );
        assert!(out.onsets.is_empty());
        assert_eq!(v.last_onset_pitch_log2, actual);
        assert_eq!(v.pending_participation_pitch, Some(440.0f32.log2()));
    }
    #[test]
    fn bounded_lifecycle_cancels_and_does_not_change_actual_reference() {
        for reason in 0..3 {
            let (mut v, tb, r, mut out) = bounded_fixture(true);
            bounded_emit(&mut v, &tb, &r, 0, &mut out);
            let actual = v.last_onset_pitch_log2;
            adoption(&mut v, 512, 440.0);
            bounded_emit(&mut v, &tb, &r, 512, &mut out);
            adoption(&mut v, 1088, 550.0);
            match reason {
                0 => v.release_pending = true,
                1 => v.remove_pending = true,
                _ => {
                    let crate::life::articulation_core::AnyArticulationCore::Entrain(core) =
                        &mut v.articulation.core
                    else {
                        unreachable!()
                    };
                    core.energy = 0.0;
                    core.state = crate::life::articulation_core::ArticulationState::Idle;
                }
            }
            bounded_emit(&mut v, &tb, &r, 1024, &mut out);
            assert!(out.onsets.is_empty() && out.tones.is_empty());
            assert!(v.pending_participation_pitch.is_none());
            assert!(v.rearticulations.is_empty() && v.participation_glide_target.is_none());
            assert_eq!(v.last_onset_pitch_log2, actual);
        }
    }
    #[test]
    #[cfg(feature = "profile-alloc")]
    fn bounded_boundary_glide_and_actual_clock_on_allocate_zero() {
        let (mut v, tb, r, mut out) = bounded_fixture(false);
        bounded_emit(&mut v, &tb, &r, 0, &mut out);
        v.phonation_engine.clock = crate::life::phonation_engine::PhonationClock::from_config(
            &PhonationClockConfig::Participation {
                coupling: 0.8,
                base_rate_hz: crate::life::phonation_engine::MAX_ONSET_RATE_HZ,
                flow_depth: 0.0,
            },
            21,
        );
        bounded_emit(&mut v, &tb, &r, 512, &mut out);
        adoption(&mut v, 1024, 440.0);
        adoption(&mut v, 1088, 335.0);
        crate::runtime_profile::begin_allocations();
        bounded_emit(&mut v, &tb, &r, 1024, &mut out);
        let counts = crate::runtime_profile::finish_allocations().unwrap();
        assert!(!out.onsets.is_empty());
        assert_eq!((counts.count, counts.bytes), (0, 0));
    }
    #[test]
    fn bounded_skip_preserves_pending_pitch_and_tone() {
        use crate::core::temporal_expectation::TemporalForecast;
        let (mut v, tb, r, mut out) = bounded_fixture(false);
        bounded_emit(&mut v, &tb, &r, 0, &mut out);
        let freq = v.body.base_freq_hz();
        adoption(&mut v, 512, 440.0);
        v.phonation_engine
            .set_onset_comparison(Some(crate::config::TemporalOnsetComparisonConfig::default()));
        v.phonation_engine.set_decision_trace(true);
        let forecast = TemporalForecast::energy_fixture(48_000, 0, |_| [10.0, 0.0, 0.0]);
        v.phonation_engine
            .set_temporal_context(Some(&forecast), [1.0, 0.0, 0.0]);
        for frame in 1..100 {
            bounded_emit(&mut v, &tb, &r, frame * 512, &mut out);
            if v.phonation_engine
                .drain_participation_decisions()
                .any(|d| d.skipped)
            {
                assert!(out.onsets.is_empty());
                assert!(!out.cmds.iter().any(|c| matches!(c, ToneCmd::Off { .. })));
                assert_eq!(v.body.base_freq_hz().to_bits(), freq.to_bits());
                assert_eq!(
                    v.pending_participation_pitch.unwrap().to_bits(),
                    440.0f32.log2().to_bits()
                );
                assert_eq!(v.active_render_notes.len(), 1);
                return;
            }
        }
        panic!("fixture must skip");
    }
    #[test]
    fn bounded_live_modes_cancel_unrendered_requests_without_extra_on() {
        for source in [
            PitchApplyMode::Glide,
            PitchApplyMode::GateSnap,
            PitchApplyMode::Rearticulate,
        ] {
            for destination in [
                PitchApplyMode::Glide,
                PitchApplyMode::GateSnap,
                PitchApplyMode::Rearticulate,
            ] {
                let (mut v, tb, r, mut out) = bounded_fixture(false);
                v.apply_patch(&ControlUpdate {
                    pitch_apply_mode: Some(source),
                    ..Default::default()
                })
                .unwrap();
                bounded_emit(&mut v, &tb, &r, 0, &mut out);
                let actual = v.last_onset_pitch_log2;
                let tones = v.active_render_notes.len();
                v.pitch_ctl.force_set_target_pitch_log2(440.0f32.log2());
                adoption(&mut v, 576, 440.);
                v.pending_participation_pitch = Some(550.0f32.log2());
                v.participation_glide_target = Some(335.0f32.log2());
                v.apply_patch(&ControlUpdate {
                    pitch_apply_mode: Some(destination),
                    ..Default::default()
                })
                .unwrap();
                assert_eq!(v.active_render_notes.len(), tones);
                assert_eq!(v.last_onset_pitch_log2, actual);
                assert_eq!(v.target_pitch_log2(), 440.0f32.log2());
                if source != destination {
                    assert!(v.rearticulations.is_empty());
                    assert!(v.participation_pitch_steps.is_empty());
                    assert!(v.pending_participation_pitch.is_none());
                    assert!(v.participation_glide_target.is_none());
                    bounded_emit(&mut v, &tb, &r, 512, &mut out);
                    assert!(out.onsets.is_empty());
                } else {
                    assert!(v.pending_participation_pitch.is_some());
                    assert_eq!(v.rearticulations.len(), 1);
                }
            }
        }
    }
    #[test]
    fn bounded_j_changes_preserve_selection_until_the_next_adoption() {
        let (mut v, tb, r, mut out) = bounded_fixture(false);
        bounded_emit(&mut v, &tb, &r, 0, &mut out);
        adoption(&mut v, 512, 335.);
        bounded_emit(&mut v, &tb, &r, 512, &mut out);
        let before = v.body.base_freq_hz();
        v.apply_patch(&ControlUpdate {
            rearticulate_above_cents: Some(0.),
            ..Default::default()
        })
        .unwrap();
        v.participation_pitch_steps.push(ParticipationPitchStep {
            at: 1024,
            before_log2: before.log2(),
            target_log2: 335.0f32.log2(),
            dt: 64. / 48000.,
        });
        bounded_emit(&mut v, &tb, &r, 1024, &mut out);
        assert!(v.body.base_freq_hz() > before);
        assert!(out.onsets.is_empty());
        adoption(&mut v, 1536, 334.);
        bounded_emit(&mut v, &tb, &r, 1536, &mut out);
        assert!(v.participation_glide_target.is_none());
        assert_eq!(v.pending_participation_pitch, Some(334.0f32.log2()));
        let held = v.body.base_freq_hz();
        v.apply_patch(&ControlUpdate {
            rearticulate_above_cents: Some(100.),
            ..Default::default()
        })
        .unwrap();
        bounded_emit(&mut v, &tb, &r, 2048, &mut out);
        assert_eq!(v.body.base_freq_hz().to_bits(), held.to_bits());
        assert_eq!(v.pending_participation_pitch, Some(334.0f32.log2()));
    }
    #[test]
    fn bounded_zero_matches_gate_snap_including_zero_distance_adoptions() {
        let (mut a, tb, r, mut x) = deferred();
        let (mut b, _, _, mut y) = bounded_fixture(false);
        b.apply_patch(&ControlUpdate {
            rearticulate_above_cents: Some(0.),
            ..Default::default()
        })
        .unwrap();
        for at in (0..144000).step_by(512) {
            if at == 512 || at == 1024 || at == 24576 {
                let hz = if at == 1024 { 330. } else { 440. };
                adoption(&mut a, at + 64, hz);
                adoption(&mut b, at + 64, hz);
            }
            bounded_emit(&mut a, &tb, &r, at, &mut x);
            bounded_emit(&mut b, &tb, &r, at, &mut y);
            assert_eq!(
                serde_json::to_value(&x).unwrap(),
                serde_json::to_value(&y).unwrap(),
                "at {at}"
            );
            assert_eq!(
                a.body.base_freq_hz().to_bits(),
                b.body.base_freq_hz().to_bits()
            );
            assert_eq!(a.pending_participation_pitch, b.pending_participation_pitch);
        }
    }
    #[test]
    fn bounded_current_grid_and_newborn_state_are_used() {
        let grid = Log2Space::new(65., 8000., 48);
        let reference = grid.centers_log2[100];
        let target = grid.centers_log2[102];
        let (mut v, _, _, mut out) = bounded_fixture(false);
        assert!(v.last_onset_pitch_log2.is_none());
        assert!(v.pending_participation_pitch.is_none());
        assert!(v.participation_glide_target.is_none());
        v.last_onset_pitch_log2 = Some(reference);
        v.body.set_pitch_log2(reference);
        adoption(&mut v, 576, target.exp2());
        v.rearticulations[0].pitch_log2 = target;
        v.emit_bounded_participation(
            &CoreState {
                is_alive: true,
                onset_allowed: true,
            },
            0.8,
            0.1,
            512,
            Some(&grid),
            &mut out,
        );
        assert_eq!(v.adoption_application, PitchApplyMode::Glide);
        assert!(v.pending_participation_pitch.is_none());
        let other = Log2Space::new(55., 8000., 96);
        v.last_onset_pitch_log2 = Some(other.centers_log2[100]);
        adoption(&mut v, 1088, other.centers_log2[105].exp2());
        v.rearticulations[0].pitch_log2 = other.centers_log2[105];
        v.emit_bounded_participation(
            &CoreState {
                is_alive: true,
                onset_allowed: true,
            },
            0.8,
            0.1,
            1024,
            Some(&other),
            &mut out,
        );
        assert_eq!(v.adoption_application, PitchApplyMode::Rearticulate);
        assert!(v.pending_participation_pitch.is_some());
    }
    #[test]
    fn bounded_rest_preserves_actual_reference_and_resumes_with_latest_pending() {
        let (mut v, tb, r, mut out) = rest_setup(1.);
        v.apply_patch(&ControlUpdate {
            pitch_apply_mode: Some(PitchApplyMode::Rearticulate),
            rearticulate_above_cents: Some(50.),
            pitch_glide_tau_sec: Some(0.03),
            ..Default::default()
        })
        .unwrap();
        bounded_emit(&mut v, &tb, &r, 0, &mut out);
        let actual = v.last_onset_pitch_log2;
        let mut off_seen = false;
        for at in (512..24576).step_by(512) {
            if at == 512 {
                adoption(&mut v, 576, 440.);
            }
            bounded_emit(&mut v, &tb, &r, at, &mut out);
            off_seen |= out
                .cmds
                .iter()
                .any(|c| matches!(c, ToneCmd::Off { off_tick: 4800, .. }));
            assert!(out.tones.is_empty());
        }
        assert!(off_seen);
        assert_eq!(v.last_onset_pitch_log2, actual);
        v.phonation_engine.rest_tail_finished(24576, tb.fs);
        adoption(&mut v, 24640, 335.);
        bounded_emit(&mut v, &tb, &r, 24576, &mut out);
        assert!(out.tones.is_empty());
        assert!(v.pending_participation_pitch.is_none());
        assert!(out.cmds.iter().all(|c| !matches!(
            c,
            ToneCmd::Update {
                update: ToneUpdate {
                    target_freq_hz: Some(_),
                    ..
                },
                ..
            }
        )));
        assert_eq!(v.last_onset_pitch_log2, actual);
        adoption(&mut v, 25152, 660.);
        bounded_emit(&mut v, &tb, &r, 25088, &mut out);
        assert_eq!(v.pending_participation_pitch, Some(660.0f32.log2()));
        let mut found = false;
        for at in (25600..180224).step_by(512) {
            bounded_emit(&mut v, &tb, &r, at, &mut out);
            if let Some(tone) = out.tones.first() {
                assert!(tone.onset >= 48576);
                assert!(tone.opportunity.is_some());
                assert_eq!(tone.freq_hz.to_bits(), 660.0f32.log2().exp2().to_bits());
                assert_eq!(v.last_onset_pitch_log2, Some(tone.freq_hz.log2()));
                assert!(v.pending_participation_pitch.is_none());
                found = true;
                break;
            }
            assert_eq!(v.last_onset_pitch_log2, actual);
        }
        assert!(found);
    }
}
