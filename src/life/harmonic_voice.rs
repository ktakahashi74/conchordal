//! Chronological pitch integration and held-tone materialization for HRY.
use super::*;

impl Voice {
    pub(crate) fn tick_harmonic(
        &mut self,
        tb: &Timebase,
        now: Tick,
        landscape: &Landscape,
        ticks: &[Tick],
        out: &mut PhonationBatch,
    ) {
        out.clear();
        self.harmonic_events.clear();
        out.source_id = self.id;
        out.source_generation = self.metadata.generation;
        out.routing = self.effective_control.body.routing;
        self.phonation_scratch.events.clear();
        self.update_phonation_gate(landscape.evaluate_pitch_level(self.body.base_freq_hz()));
        let target_amp = self.compute_target_amp();
        let terminal = !self.is_alive() || self.release.pending || self.release.remove_pending;
        let state = CoreState {
            is_alive: !terminal,
            onset_allowed: target_amp > Self::AMP_EPS && self.phonation_gate_open,
        };
        out.body_policy = Some(super::super::action_candidates::PolicySnapshot {
            at: now,
            is_alive: state.is_alive,
            gate_allows_onset: state.onset_allowed,
        });
        let end = now + tb.hop as Tick;
        let mut ctx = CoreTickCtx {
            now_tick: now,
            frame_end: end,
            fs: tb.fs,
            rhythms: landscape.rhythm,
        };
        self.phonation_engine.set_sound_adsr(self.voice_adsr);
        let pitch = &self.effective_control.pitch;
        let (field_lo, field_hi) = landscape.freq_bounds_log2();
        let center = pitch.freq.log2();
        let half = pitch.range_oct.clamp(0., 6.) * 0.5;
        let lo = field_lo.max(center - half);
        let hi = field_hi.min(center + half);
        let current = self.body.base_freq_hz();
        let supported = current.is_finite()
            && current > 0.
            && lo <= hi
            && current.log2() >= lo
            && current.log2() <= hi;
        if terminal || !supported {
            if self.pitch_ctl.dwell_trace_enabled && self.phonation_engine.has_active_notes() {
                self.harmonic_events
                    .push(super::super::harmonic_rhythm::VoiceEvent {
                        tick: now,
                        action: if terminal { "terminal" } else { "support_lost" },
                        frequency_hz: current,
                    });
            }
            // Preserve Once's latch while releasing the current tone exactly once.
            self.phonation_engine.tick_with_arrival(
                &ctx,
                &CoreState {
                    is_alive: false,
                    onset_allowed: false,
                },
                None,
                0.0,
                1.0,
                None,
                None,
                &mut out.cmds,
                &mut self.phonation_scratch.events,
                &mut out.onsets,
            );
            if !terminal {
                self.harmonic_waiting = true;
            }
        }
        self.prune_tracked_render_notes(&out.cmds);
        self.emit_phonation_updates(
            now,
            target_amp,
            current,
            self.effective_control.body.continuous_drive,
            &mut out.cmds,
        );
        let (neighbors, salience) = std::mem::take(&mut self.harmonic_neighbors);
        let mut previous = now;
        let mut at = now;
        let mut opportunity = 0;
        loop {
            ctx.now_tick = at;
            let shared = ticks.get(opportunity) == Some(&at) && at < end;
            if shared {
                opportunity += 1;
            }
            self.phonation_engine.advance_span(at);
            let legal = !terminal
                && state.onset_allowed
                && lo <= hi
                && self
                    .phonation_engine
                    .rest
                    .is_none_or(|r| r.phase == crate::life::phonation_engine::RestPhase::Ready)
                && self
                    .phonation_engine
                    .span_gap
                    .as_ref()
                    .is_none_or(|g| g.allows_on());
            let initial =
                at == now && !self.phonation_engine.has_active_notes() && !self.harmonic_waiting;
            self.pitch_ctl.control_tick = Some(at);
            self.pitch_ctl.harmonic_gate = Some(shared && legal && !initial);
            ctx.rhythms.advance_in_place((at - previous) as f32 / tb.fs);
            let changed = self.pitch_ctl.update_pitch_target(
                self.body.base_freq_hz(),
                &ctx.rhythms,
                (at - previous) as f32 / tb.fs,
                landscape,
                &self.effective_control.pitch,
                &neighbors,
                &salience,
                self.harmonic_temperature_bonus,
            );
            if at == now && !terminal && supported && !self.harmonic_waiting {
                self.phonation_engine.tick_with_arrival(
                    &ctx,
                    &state,
                    None,
                    0.0,
                    1.0,
                    None,
                    None,
                    &mut out.cmds,
                    &mut self.phonation_scratch.events,
                    &mut out.onsets,
                );
            }
            let resuming = self.harmonic_waiting;
            if shared
                && legal
                && self.pitch_ctl.harmonic_evaluated
                && (changed || self.harmonic_waiting)
            {
                self.phonation_engine.rearticulate_hold(
                    &ctx,
                    &state,
                    &mut out.cmds,
                    &mut self.phonation_scratch.events,
                    &mut out.onsets,
                );
                self.body.set_pitch_log2(self.target_pitch_log2());
                self.harmonic_waiting = false;
            }
            if shared && self.pitch_ctl.dwell_trace_enabled {
                self.harmonic_events
                    .push(super::super::harmonic_rhythm::VoiceEvent {
                        tick: at,
                        action: if !legal {
                            "veto"
                        } else if !self.pitch_ctl.harmonic_evaluated {
                            "not_ready"
                        } else if resuming {
                            "resume"
                        } else if changed {
                            "adopt"
                        } else {
                            "unchanged"
                        },
                        frequency_hz: self.body.base_freq_hz(),
                    });
            }
            if !self.phonation_scratch.events.is_empty() {
                let freq = self.body.base_freq_hz();
                let body = self.body_snapshot();
                let modulator = self
                    .articulation
                    .render_modulator_spec(PhonationMode::Gated);
                for event in self.phonation_scratch.events.drain(..) {
                    out.tones.push(ToneSpec::from_event(
                        event,
                        freq,
                        target_amp,
                        Some(Tick::MAX),
                        body.clone(),
                        modulator.clone(),
                        self.voice_adsr,
                    ));
                }
            }
            if at == end {
                break;
            }
            previous = at;
            let control = now + ((at - now) / 64 + 1) * 64;
            at = control
                .min(end)
                .min(ticks.get(opportunity).copied().unwrap_or(end));
        }
        self.harmonic_neighbors = (neighbors, salience);
        self.pitch_ctl.harmonic_gate = None;
        self.apply_phonation_onsets(
            landscape.evaluate_pitch_level(self.body.base_freq_hz()),
            out,
        );
        self.track_new_render_notes(&out.tones);
        self.prune_tracked_render_notes(&out.cmds);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (Voice, Landscape, Timebase, PhonationBatch) {
        let land = Landscape::new(Log2Space::new(55., 2000., 96));
        let mut c = VoiceControl::default();
        c.pitch.mode = PitchMode::Free;
        c.pitch.freq = 220.;
        c.pitch.range_oct = 4.;
        c.pitch.pitch_apply_mode = PitchApplyMode::Rearticulate;
        c.pitch.proposal_interval_sec = Some(0.001);
        let mut v = Voice::spawn_from_control(
            c,
            ArticulationCoreConfig::Drone {
                sway: None,
                breath_gain_init: None,
            },
            1,
            0,
            VoiceMetadata::default(),
            48000.,
            Some(&land),
            21,
        );
        v.harmonic_member = true;
        v.pitch_ctl.dwell_trace_enabled = true;
        v.configure_control_hop(512, 64, 48000.);
        (
            v,
            land,
            Timebase {
                fs: 48000.,
                hop: 512,
            },
            PhonationBatch::default(),
        )
    }

    #[test]
    fn harmonic_voice_waits_for_readiness_and_resumes_once_after_support_loss() {
        let (mut v, land, tb, mut out) = fixture();
        v.tick_harmonic(&tb, 0, &land, &[47, 49], &mut out);
        assert_eq!(out.tones[0].onset, 0);
        assert_eq!(v.harmonic_events[0].action, "not_ready");
        assert_ne!(v.harmonic_events[1].action, "not_ready");
        let held = v.body.base_freq_hz();
        v.effective_control.pitch.freq = 440.;
        v.effective_control.pitch.range_oct = 0.;
        v.tick_harmonic(&tb, 512, &land, &[], &mut out);
        assert!(out.tones.is_empty());
        assert!(
            out.cmds
                .iter()
                .any(|c| matches!(c, ToneCmd::Off { off_tick: 512, .. }))
        );
        v.tick_harmonic(&tb, 1024, &land, &[], &mut out);
        assert!(out.tones.is_empty());
        assert!(!out.cmds.iter().any(|c| matches!(c, ToneCmd::Off { .. })));
        v.effective_control.pitch.freq = held;
        v.tick_harmonic(&tb, 1536, &land, &[1700, 1800], &mut out);
        assert_eq!(out.tones.len(), 1);
        assert_eq!(out.tones[0].onset, 1700);
        assert_eq!(out.tones[0].freq_hz, held);
        assert_eq!(v.harmonic_events[0].action, "resume");
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
        v.start_remove_fade(1.);
        v.tick_harmonic(&tb, 2048, &land, &[2100], &mut out);
        assert!(out.tones.is_empty());
        assert_eq!(v.harmonic_events.last().unwrap().action, "veto");
    }

    #[test]
    fn harmonic_voice_dwell_stop_and_empty_range_preserve_readiness() {
        let (mut v, mut land, tb, mut out) = fixture();
        v.effective_control.pitch.dwell = 2.0;
        land.consonance_field_score_eff.fill(100.);
        v.tick_harmonic(&tb, 0, &land, &[100, 400], &mut out);
        assert!(v.harmonic_events.iter().all(|e| e.action == "not_ready"));
        let state = v.pitch_ctl.dwell_state.unwrap();
        assert_eq!(state.phase, 0.);
        assert!((state.elapsed_sec - 512. / 48000.).abs() < 1e-6);
        v.effective_control.pitch.freq = 10000.;
        v.effective_control.pitch.range_oct = 0.;
        v.tick_harmonic(&tb, 512, &land, &[600, 700], &mut out);
        assert!(out.tones.is_empty());
        assert!(
            v.harmonic_events
                .iter()
                .filter(|e| e.tick >= 600)
                .all(|e| e.action == "veto")
        );
    }

    #[test]
    fn harmonic_voice_no_opportunity_preserves_pitch_and_does_not_double_integrate() {
        let (mut v, land, tb, mut out) = fixture();
        let pitch = v.body.base_freq_hz();
        for i in 0..10 {
            v.tick_harmonic(&tb, i * 512, &land, &[], &mut out);
        }
        assert_eq!(v.body.base_freq_hz(), pitch);
        v.update_pitch_target(&NeuralRhythms::default(), 100., &land, &[], &[]);
        v.tick_harmonic(&tb, 5120, &land, &[5130], &mut out);
        let r = v.pitch_ctl.dwell_trace.last().unwrap();
        assert!((r.elapsed_sec - 5130. / 48000.).abs() < 1e-6);
    }
}
