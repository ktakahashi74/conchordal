use super::*;

impl Voice {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_sustained_participation(
        &mut self,
        ctx: &CoreTickCtx,
        state: &CoreState,
        consonance: f32,
        target_amp: f32,
        arrival: Option<&crate::life::arrival_cost::ArrivalContext<'_>>,
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
}
