use super::*;
use crate::scenario::control::BodyControl;

#[derive(Clone, Copy, Debug)]
pub(super) struct TrackedRenderNote {
    pub(super) tone_id: ToneId,
    pub(super) last_target_amp: f32,
    pub(super) last_target_freq_hz: f32,
    pub(super) last_continuous_drive: f32,
}

#[derive(Clone, Copy, Debug)]
struct BodyRuntime {
    amp: f32,
    brightness: f32,
    inharmonic: f32,
    motion: f32,
    spread: f32,
    unison: usize,
}

impl BodyRuntime {
    fn from_control(body: &BodyControl) -> Self {
        let timbre = &body.timbre;
        Self {
            amp: body.amp,
            brightness: timbre.brightness,
            inharmonic: timbre.inharmonic,
            motion: timbre.motion,
            spread: timbre.spread,
            unison: timbre.unison,
        }
    }
}

impl Voice {
    const PHONATION_AMP_UPDATE_EPS: f32 = 0.01;
    pub(super) const PHONATION_UPDATE_SMOOTH_TAU_SEC: f32 = 0.02;

    pub(crate) fn refresh_support_template(&mut self, landscape: &Landscape) {
        let space = &landscape.space;
        let key = (
            space.fmin,
            space.fmax,
            space.bins_per_oct,
            landscape.harmonic.support_ref_power,
            landscape.harmonic.support_loudness_exp,
        );
        if self.support.template_key == Some(key) {
            return;
        }
        let signal = ArticulationSignal {
            amplitude: 1.,
            is_active: true,
            ..Default::default()
        };
        let density = self
            .body
            .project_support_density(space, key.3, key.4, &signal);
        self.support.template = density.map(Into::into);
        self.support.template_key = Some(key);
    }

    pub(crate) fn refresh_support_actions(
        &mut self,
        landscape: &Landscape,
        tb: &Timebase,
        now: Tick,
    ) {
        use crate::life::support_action::{Context, active};
        self.phonation_engine.support_action = None;
        self.phonation_engine.support_held_enabled = active(landscape.harmonic.support_weights)
            && self.effective_control.pitch.leave_self_out
            && self.effective_control.body.routing.to_habitat;
        if !self.phonation_engine.support_held_enabled {
            return;
        }
        let Some(snapshot) = landscape
            .ecology
            .loo
            .iter()
            .find(|s| s.id == self.id && s.generation == self.metadata.generation)
            .and_then(|s| s.support.clone())
        else {
            return;
        };
        self.refresh_support_template(landscape);
        let shape = self.body_snapshot();
        let spec = self
            .articulation
            .render_modulator_spec(if self.harmonic.member {
                PhonationMode::Gated
            } else {
                self.phonation_engine.mode
            });
        if self
            .support
            .onset
            .as_ref()
            .is_none_or(|(cached, _)| cached != &shape)
        {
            self.support.onset = crate::life::sound::Tone::from_parts(
                *tb,
                now,
                Tick::MAX,
                self.body.base_freq_hz(),
                self.body.amp(),
                Some(shape.clone()),
                Some(spec.clone()),
                self.voice_adsr,
            )
            .map(|tone| (shape.clone(), tone));
        }
        let amp = self.compute_target_amp();
        let onset = self
            .support
            .onset
            .as_mut()
            .filter(|_| self.phonation_engine.phrase_boundary.is_none())
            .and_then(|(_, tone)| {
                tone.preview_new_energy(
                    *tb,
                    now,
                    Tick::MAX,
                    amp,
                    shape.amp_scale,
                    self.voice_adsr,
                    spec,
                    &landscape.rhythm,
                    #[cfg(feature = "research")]
                    self.support.renderer_phase3,
                )
            });
        self.phonation_engine.support_action = Some(Context {
            source: Some((self.id, self.metadata.generation)),
            snapshot,
            template: self.support.template.clone(),
            single_density: landscape.support_single_density(self.body.amp()),
            weights: landscape.harmonic.support_weights,
            loudness_exp: landscape.harmonic.support_loudness_exp,
            nominal_amp: self.body.amp(),
            pitch: self.body.base_freq_hz().log2(),
            onset,
            body: self.support.body,
            retention_bonus: 0.,
            allow_release: true,
            replace: self.phonation_engine.mode == PhonationMode::Hold
                || self.phonation_engine.sustain_participation,
        });
    }

    pub(super) fn apply_body_runtime(&mut self) {
        self.support.template_key = None;
        let runtime = BodyRuntime::from_control(&self.effective_control.body);
        self.amp_update_pending |= runtime.amp != self.body.amp();
        self.body.set_amp(runtime.amp);
        self.body.apply_timbre_controls(
            runtime.brightness,
            runtime.inharmonic,
            runtime.motion,
            runtime.spread,
            runtime.unison,
        );
    }

    pub(super) fn update_surviving_render_notes(
        &mut self,
        now: Tick,
        target_amp: f32,
        out: &mut PhonationBatch,
    ) {
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
    }

    pub(super) fn finish_participation_batch(&mut self, out: &PhonationBatch) {
        self.track_new_render_notes(&out.tones);
        self.prune_tracked_render_notes(&out.cmds);
        self.phonation_scratch.events.clear();
        self.participation_pitch.rearticulations.clear();
        self.participation_pitch.steps.clear();
    }

    pub(super) fn discard_pending_render_tone_ons(out_cmds: &mut Vec<ToneCmd>) {
        // Keep Off/Update commands intact when a new render tone cannot be materialized.
        out_cmds.retain(|cmd| !matches!(cmd, ToneCmd::On { .. }));
    }

    pub(super) fn emit_release_note_offs(&self, now: Tick, out_cmds: &mut Vec<ToneCmd>) {
        for tracked in &self.active_render_notes {
            let already_off = out_cmds.iter().any(
                |cmd| matches!(cmd, ToneCmd::Off { tone_id, .. } if *tone_id == tracked.tone_id),
            );
            if !already_off {
                out_cmds.push(ToneCmd::Off {
                    tone_id: tracked.tone_id,
                    off_tick: now,
                });
            }
        }
    }

    pub(super) fn emit_phonation_updates(
        &mut self,
        now: Tick,
        target_amp: f32,
        freq_hz: f32,
        continuous_drive: f32,
        out_cmds: &mut Vec<ToneCmd>,
    ) {
        let autonomous_pulse = if self.participation_pitch.pulse_update_pending {
            Some(
                match self
                    .articulation
                    .render_modulator_spec(if self.reports_onset_pitch() {
                        PhonationMode::Gated
                    } else {
                        self.phonation_engine.mode
                    }) {
                    RenderModulatorSpec::EntrainPulse {
                        autonomous_pulse, ..
                    } => autonomous_pulse,
                    _ => None,
                },
            )
        } else {
            None
        };
        for tracked in &mut self.active_render_notes {
            // Explicit amplitude controls bypass the threshold for autonomous vitality drift.
            let amp_changed = self.amp_update_pending
                || (target_amp - tracked.last_target_amp).abs() >= Self::PHONATION_AMP_UPDATE_EPS;
            let freq_changed = !self.harmonic.member
                && freq_hz.is_finite()
                && freq_hz > 0.0
                && (freq_hz - tracked.last_target_freq_hz).abs() > 0.01;
            let drive_changed = (continuous_drive - tracked.last_continuous_drive).abs() > 1e-4;
            if !amp_changed && !freq_changed && !drive_changed && autonomous_pulse.is_none() {
                continue;
            }
            out_cmds.push(ToneCmd::Update {
                tone_id: tracked.tone_id,
                at_tick: Some(now),
                update: ToneUpdate {
                    autonomous_pulse,
                    target_freq_hz: if freq_changed { Some(freq_hz) } else { None },
                    target_amp: if amp_changed { Some(target_amp) } else { None },
                    continuous_drive: if drive_changed {
                        Some(continuous_drive)
                    } else {
                        None
                    },
                },
            });
            if amp_changed {
                tracked.last_target_amp = target_amp;
            }
            if freq_changed {
                tracked.last_target_freq_hz = freq_hz;
            }
            if drive_changed {
                tracked.last_continuous_drive = continuous_drive;
            }
        }
        self.amp_update_pending = false;
        self.participation_pitch.pulse_update_pending = false;
    }

    pub(super) fn prune_tracked_render_notes(&mut self, cmds: &[ToneCmd]) {
        self.active_render_notes.retain(|tracked| {
            !cmds.iter().any(
                |cmd| matches!(cmd, ToneCmd::Off { tone_id, .. } if *tone_id == tracked.tone_id),
            )
        });
    }

    pub(super) fn track_new_render_notes(&mut self, tones: &[ToneSpec]) {
        if let Some(tone) = tones.last() {
            self.participation_pitch.last_onset_log2 = Some(tone.freq_hz.log2());
        }
        self.active_render_notes
            .extend(tones.iter().map(|tone| TrackedRenderNote {
                tone_id: tone.tone_id,
                last_target_amp: tone.amp,
                last_target_freq_hz: tone.freq_hz,
                last_continuous_drive: 0.0,
            }));
    }

    pub(crate) fn compute_target_amp(&self) -> f32 {
        let mut amp =
            self.body.amp() * self.compute_output_gain() * self.articulation.vitality_scalar();
        if !amp.is_finite() {
            amp = 0.0;
        }
        amp.max(0.0)
    }

    pub(crate) fn body_snapshot(&self) -> BodySnapshot {
        self.body.snapshot()
    }

    #[cfg(feature = "research")]
    pub(crate) fn supports_body_metabolism(&self, renderer_phase3: bool) -> bool {
        sound_body::supports_body_model(
            &self.effective_control,
            &self.body.snapshot(),
            renderer_phase3,
        )
    }

    #[cfg(any(test, feature = "research"))]
    pub(crate) fn representative_body_recipe(
        &self,
        fs: f32,
        hold: Tick,
    ) -> crate::life::onset_footprint::Recipe {
        let modulator = crate::life::onset_footprint::representative_modulator(
            self.articulation
                .render_modulator_spec(self.phonation_engine.mode),
        );
        let modulator = match modulator {
            crate::life::sound::RenderModulatorSpec::DroneSway { sway_rate, .. } => {
                crate::life::sound::RenderModulatorSpec::DroneSway {
                    phase: 0.0,
                    sway_rate,
                }
            }
            other => other,
        };
        crate::life::onset_footprint::Recipe {
            #[cfg(feature = "research")]
            renderer_phase3: false,
            body: self.body_snapshot(),
            freq_hz: self.body.base_freq_hz(),
            hold,
            adsr: self.voice_adsr,
            modulator,
            smoothing_tau_sec: Self::PHONATION_UPDATE_SMOOTH_TAU_SEC,
            fs,
        }
    }

    /// The representative recipe of I11-1 §4.2, frozen from what this Voice would
    /// emit now. `None` unless a participation clock configures onset comparison.
    #[cfg(feature = "research")]
    pub(crate) fn footprint_recipe(&self, fs: f32) -> Option<crate::life::onset_footprint::Recipe> {
        let (hold_sec, adsr) = self.phonation_engine.footprint_hold()?;
        let phonation_mode = self.phonation_engine.mode;
        Some(crate::life::onset_footprint::Recipe {
            #[cfg(feature = "research")]
            renderer_phase3: false,
            body: self.body_snapshot(),
            freq_hz: self.body.base_freq_hz(),
            hold: if matches!(phonation_mode, PhonationMode::Hold) {
                Tick::MAX
            } else {
                (f64::from(hold_sec) * f64::from(fs)).round() as Tick
            },
            adsr,
            modulator: crate::life::onset_footprint::representative_modulator(
                self.articulation.render_modulator_spec(phonation_mode),
            ),
            smoothing_tau_sec: Self::PHONATION_UPDATE_SMOOTH_TAU_SEC,
            fs,
        })
    }

    #[cfg(test)]
    pub(crate) fn project_loo_power(&mut self, space: &Log2Space) -> &[f32] {
        let input = self.pitch_ctl.loo_input_mut();
        input.computed_score = None;
        input.own_power_scan.resize(space.n_bins(), 0.0);
        input.own_power_scan.fill(0.0);
        if self.effective_control.body.routing.to_habitat {
            self.body
                .project_spectral_body(&mut input.own_power_scan, space, &self.last_signal);
        }
        for value in &mut input.own_power_scan {
            *value *= *value;
        }
        &input.own_power_scan
    }

    pub fn render_spectrum(&mut self, amps: &mut [f32], space: &Log2Space) {
        let signal = self.last_signal;
        if !signal.is_active || signal.amplitude <= 0.0 {
            return;
        }
        self.body.project_spectral_body(amps, space, &signal);
    }
}
