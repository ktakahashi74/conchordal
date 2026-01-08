use crate::core::landscape::Landscape;
use crate::core::log2space::{Log2Space, sample_scan_linear_log2};
use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::intent::{BodySnapshot, Intent};
use crate::life::intent_planner::choose_best_gesture_tf_by_pred_c;
use crate::life::perceptual::{FeaturesNow, PerceptualContext};
use crate::life::phonation_engine::{
    CandidatePoint, CoreState, CoreTickCtx, NoteId, OnsetEvent, PhonationClock, PhonationCmd,
    PhonationEngine, PhonationKick, PhonationNoteEvent, PhonationUpdate, ThetaGateClock,
};
use crate::life::scenario::{
    OnBirthPhonation, PhonationConfig, PlanningConfig, SustainUpdateCadence, SustainUpdateTarget,
};
use crate::life::social_density::SocialDensityTrace;
use rand::rngs::SmallRng;
use std::collections::VecDeque;

#[path = "articulation_core.rs"]
pub mod articulation_core;
#[path = "pitch_core.rs"]
pub mod pitch_core;
#[path = "sound_body.rs"]
pub mod sound_body;

pub use articulation_core::{
    AnyArticulationCore, ArticulationCore, ArticulationSignal, ArticulationState, ArticulationStep,
    ArticulationWrapper, DroneCore, ErrorState, KuramotoCore, PinkNoise, PlannedGate, PlannedPitch,
    Sensitivity, SequencedCore,
};
pub use pitch_core::{AnyPitchCore, PitchCore, PitchHillClimbPitchCore, TargetProposal};
pub use sound_body::{AnySoundBody, HarmonicBody, SineBody, SoundBody};

pub trait AudioAgent {
    fn id(&self) -> u64;
    fn metadata(&self) -> &AgentMetadata;
    fn render_wave(
        &mut self,
        buffer: &mut [f32],
        fs: f32,
        current_frame: u64,
        dt_sec: f32,
        landscape: &Landscape,
        global_coupling: f32,
    );
    fn render_spectrum(&mut self, amps: &mut [f32], space: &Log2Space);
    fn is_alive(&self) -> bool;
}

#[derive(Debug, Clone, Default)]
pub struct AgentMetadata {
    pub id: u64,
    pub tag: Option<String>,
    pub group_idx: usize,
    pub member_idx: usize,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AudioSample {
    pub sample: f32,
}

#[derive(Debug)]
pub struct Individual {
    pub id: u64,
    pub metadata: AgentMetadata,
    pub articulation: ArticulationWrapper,
    pub pitch: AnyPitchCore,
    pub perceptual: PerceptualContext,
    pub planning: PlanningConfig,
    pub phonation: PhonationConfig,
    pub phonation_engine: PhonationEngine,
    pub body: AnySoundBody,
    pub last_signal: ArticulationSignal,
    pub release_gain: f32,
    pub release_sec: f32,
    pub release_pending: bool,
    pub birth_pending: bool,
    pub birth_fired: bool,
    pub birth_onset_tick: Option<Tick>,
    pub birth_onset_gate: Option<u64>,
    pub sustain_note_id: Option<NoteId>,
    pub sustain_onset_tick: Option<Tick>,
    pub target_pitch_log2: f32,
    pub integration_window: f32,
    pub accumulated_time: f32,
    pub last_theta_sample: f32,
    pub last_target_salience: f32,
    pub last_error_state: ErrorState,
    pub last_error_cents: f32,
    pub error_initialized: bool,
    pub last_chosen_freq_hz: f32,
    pub next_intent_tick: Tick,
    pub intent_seq: u64,
    pub self_confidence: f32,
    pub pred_intent_records: VecDeque<PredIntentRecord>,
    pub pred_intent_records_cap: usize,
    pub rng: SmallRng,
}

#[derive(Clone, Debug)]
pub struct PredIntentRecord {
    pub intent_id: u64,
    pub onset: Tick,
    pub end: Tick,
    pub freq_hz: f32,
    pub pred_c_statepm1: f32,
    pub created_at: Tick,
    pub eval_tick: Tick,
}

#[derive(Clone, Debug)]
pub struct PhonationNoteSpec {
    pub note_id: NoteId,
    pub onset: Tick,
    pub hold_ticks: Option<Tick>,
    pub freq_hz: f32,
    pub amp: f32,
    pub body: BodySnapshot,
    pub articulation: ArticulationWrapper,
}

#[derive(Clone, Debug, Default)]
pub struct PhonationBatch {
    pub source_id: u64,
    pub cmds: Vec<PhonationCmd>,
    pub notes: Vec<PhonationNoteSpec>,
    pub onsets: Vec<OnsetEvent>,
}

impl Individual {
    const AMP_EPS: f32 = 1e-6;

    pub fn birth_pending(&self) -> bool {
        self.birth_pending && !self.birth_fired
    }

    pub fn set_birth_onset_tick(&mut self, onset: Option<Tick>) {
        self.birth_onset_tick = onset;
        self.birth_onset_gate = None;
    }

    pub fn should_retain(&self) -> bool {
        self.is_alive() || self.birth_pending() || self.phonation_engine.has_active_notes()
    }
    pub fn metadata(&self) -> &AgentMetadata {
        &self.metadata
    }

    pub fn force_set_pitch_log2(&mut self, log_freq: f32) {
        let log_freq = log_freq.max(0.0);
        self.body.set_pitch_log2(log_freq);
        self.target_pitch_log2 = log_freq;
        self.last_chosen_freq_hz = self.body.base_freq_hz();
        self.articulation.set_gate(1.0);
        self.accumulated_time = 0.0;
        self.last_theta_sample = 0.0;
        self.last_target_salience = 0.0;
        self.last_error_state = ErrorState::default();
        self.last_error_cents = 0.0;
        self.error_initialized = false;
    }

    pub fn update_pitch_target(&mut self, rhythms: &NeuralRhythms, dt: f32, landscape: &Landscape) {
        let dt = dt.max(0.0);
        let current_freq = self.body.base_freq_hz().max(1.0);
        let current_pitch_log2 = current_freq.log2();
        if self.target_pitch_log2 <= 0.0 {
            self.target_pitch_log2 = current_pitch_log2;
        }
        self.integration_window = 2.0 + 10.0 / current_freq.max(1.0);
        self.accumulated_time += dt;

        let theta_signal = rhythms.theta.mag * rhythms.theta.phase.sin();
        let theta_cross = self.last_theta_sample <= 0.0 && theta_signal > 0.0;
        self.last_theta_sample = theta_signal;

        if theta_cross && self.accumulated_time >= self.integration_window {
            let elapsed = self.accumulated_time;
            self.accumulated_time = 0.0;
            let features = FeaturesNow::from_subjective_intensity(&landscape.subjective_intensity);
            debug_assert_eq!(features.distribution.len(), landscape.space.n_bins());
            self.perceptual.ensure_len(features.distribution.len());
            let proposal = self.pitch.propose_target(
                current_pitch_log2,
                self.target_pitch_log2,
                current_freq,
                self.integration_window,
                landscape,
                &self.perceptual,
                &features,
                &mut self.rng,
            );
            self.target_pitch_log2 = proposal.target_pitch_log2;
            self.last_target_salience = proposal.salience;
            if let Some(idx) = landscape.space.index_of_log2(self.target_pitch_log2) {
                self.perceptual.update(idx, &features, elapsed);
            }
        }

        let (fmin, fmax) = landscape.freq_bounds_log2();
        self.target_pitch_log2 = self.target_pitch_log2.clamp(fmin, fmax);
    }

    pub fn update_articulation(
        &mut self,
        dt_per_sample: f32,
        rhythms: &NeuralRhythms,
        landscape: &Landscape,
        global_coupling: f32,
    ) -> ArticulationSignal {
        let current_freq = self.body.base_freq_hz().max(1.0);
        let current_pitch_log2 = current_freq.log2();
        let planned = PlannedPitch {
            target_pitch_log2: self.target_pitch_log2,
            jump_cents_abs: 1200.0 * (self.target_pitch_log2 - current_pitch_log2).abs(),
            salience: self.last_target_salience,
        };
        let pitch_error_cents = 1200.0 * (planned.target_pitch_log2 - current_pitch_log2);
        let d_pitch_error_cents_per_sec = if self.error_initialized && dt_per_sample > 0.0 {
            (pitch_error_cents - self.last_error_cents) / dt_per_sample
        } else {
            0.0
        };
        self.last_error_state = ErrorState {
            pitch_error_cents,
            abs_pitch_error_cents: pitch_error_cents.abs(),
            d_pitch_error_cents_per_sec,
        };
        self.last_error_cents = pitch_error_cents;
        self.error_initialized = true;
        let apply_planned_pitch =
            self.articulation
                .update_gate(&planned, &self.last_error_state, rhythms, dt_per_sample);
        if apply_planned_pitch {
            self.body.set_pitch_log2(planned.target_pitch_log2);
        }
        let consonance = landscape.evaluate_pitch01(self.body.base_freq_hz());
        let step: ArticulationStep = self.articulation.process(
            consonance,
            rhythms,
            dt_per_sample,
            global_coupling,
            &planned,
            &self.last_error_state,
        );
        debug_assert_eq!(step.apply_planned_pitch, apply_planned_pitch);
        let mut signal = step.signal;
        signal.amplitude *= self.articulation.gate();
        if self.release_pending {
            let step = dt_per_sample / self.release_sec.max(1e-6);
            self.release_gain = (self.release_gain - step).max(0.0);
        }
        signal.amplitude *= self.release_gain;
        signal.is_active = signal.is_active && signal.amplitude > 0.0;
        self.last_signal = signal;
        signal
    }

    pub fn start_release(&mut self, release_sec: f32) {
        if self.release_pending {
            return;
        }
        self.release_pending = true;
        self.release_sec = release_sec.max(1e-4);
        self.release_gain = self.release_gain.clamp(0.0, 1.0);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn plan_intents(
        &mut self,
        tb: &Timebase,
        now: Tick,
        gate_tick: Tick,
        perc_tick: Tick,
        pred_eval_tick: Option<Tick>,
        hop: usize,
        landscape: &Landscape,
        intents: &[Intent],
        pred_c_scan_at: &mut dyn FnMut(Tick) -> Option<std::sync::Arc<[f32]>>,
    ) -> Vec<Intent> {
        self.update_self_confidence_from_perc(&landscape.space, landscape, perc_tick);
        let hop_tick = hop as Tick;
        if self.next_intent_tick != 0 && now < self.next_intent_tick {
            return Vec::new();
        }
        let frame_end = now.saturating_add(hop_tick.max(1));
        if gate_tick < now || gate_tick >= frame_end {
            return Vec::new();
        }

        let use_pred_c = self.planning.pitch_mode == crate::life::scenario::PlanPitchMode::PredC
            && pred_eval_tick.is_some();
        let pred_tick = if use_pred_c {
            Some(pred_eval_tick.expect("pred_eval_tick must be Some when use_pred_c"))
        } else {
            None
        };
        let theta_hz = landscape.rhythm.theta.freq_hz;
        let dur_sec = gate_duration_sec_from_theta(theta_hz, &self.planning);
        let dur_tick = sec_to_tick_at_least_one(tb, dur_sec);
        let amp = self.release_gain.clamp(0.0, 1.0);
        if amp <= Self::AMP_EPS {
            return Vec::new();
        }
        let base_freq_hz = if self.last_chosen_freq_hz > 0.0 && self.last_chosen_freq_hz.is_finite()
        {
            self.last_chosen_freq_hz
        } else {
            self.body.base_freq_hz()
        };
        let mut freq_hz = base_freq_hz;
        if let Some(pred_tick) = pred_tick {
            let mut freq_eps = tb.sec_to_tick(0.01);
            if freq_eps == 0 {
                freq_eps = 1;
            }
            let intent_refs = intents;
            let mut make_freq_candidates = |onset: Tick| {
                let min = onset.saturating_sub(freq_eps);
                let max = onset.saturating_add(freq_eps);
                let neighbors: Vec<f32> = intent_refs
                    .iter()
                    .filter(|intent| intent.onset >= min && intent.onset <= max)
                    .filter_map(|intent| {
                        if intent.freq_hz.is_finite() && intent.freq_hz > 0.0 {
                            Some(intent.freq_hz)
                        } else {
                            None
                        }
                    })
                    .collect();
                self.pitch
                    .propose_freqs_hz_with_neighbors(base_freq_hz, &neighbors, 16, 8, 12.0)
            };
            let candidates = [pred_tick];
            if let Some(choice) = choose_best_gesture_tf_by_pred_c(
                &landscape.space,
                &candidates,
                base_freq_hz,
                &mut make_freq_candidates,
                &mut *pred_c_scan_at,
            ) {
                freq_hz = choice.freq_hz.clamp(20.0, 20_000.0);
            }
        }
        self.last_chosen_freq_hz = freq_hz;
        let snapshot = self.body_snapshot();
        let kind = snapshot.kind.clone();
        let intent = Intent {
            source_id: self.id,
            intent_id: self.intent_seq,
            kind: crate::life::intent::IntentKind::Normal,
            onset: gate_tick,
            duration: dur_tick,
            freq_hz,
            amp,
            tag: Some(format!("agent:{} {}", self.id, kind)),
            confidence: 1.0,
            body: Some(snapshot),
            articulation: Some(self.articulation.clone()),
        };
        if let Some(pred_tick) = pred_tick {
            self.record_pred_intent(&intent, pred_c_scan_at, now, pred_tick, landscape);
        }
        self.intent_seq = self.intent_seq.wrapping_add(1);
        self.next_intent_tick = now.saturating_add(hop_tick.max(1));
        vec![intent]
    }

    pub fn take_birth_intent_if_due(
        &mut self,
        tb: &Timebase,
        now: Tick,
        frame_end: Tick,
        landscape: &Landscape,
    ) -> Option<Intent> {
        if !self.birth_pending() || !matches!(self.phonation.on_birth, OnBirthPhonation::Once) {
            return None;
        }
        let onset = self.birth_onset_tick?;
        if onset < now || onset >= frame_end {
            return None;
        }
        let theta_hz = landscape.rhythm.theta.freq_hz;
        let dur_sec = gate_duration_sec_from_theta(theta_hz, &self.planning);
        let dur_tick = sec_to_tick_at_least_one(tb, dur_sec);
        let amp = self.body.amp().clamp(0.0, 1.0);
        let base_freq_hz = if self.last_chosen_freq_hz > 0.0 && self.last_chosen_freq_hz.is_finite()
        {
            self.last_chosen_freq_hz
        } else {
            self.body.base_freq_hz()
        };
        let snapshot = self.body_snapshot();
        let kind = snapshot.kind.clone();
        let intent = Intent {
            source_id: self.id,
            intent_id: self.intent_seq,
            kind: crate::life::intent::IntentKind::BirthOnce,
            onset,
            duration: dur_tick,
            freq_hz: base_freq_hz.max(1.0),
            amp,
            tag: Some(format!("agent:{} birth {}", self.id, kind)),
            confidence: 1.0,
            body: Some(snapshot),
            articulation: Some(self.articulation.clone()),
        };
        self.intent_seq = self.intent_seq.wrapping_add(1);
        self.birth_fired = true;
        self.birth_pending = false;
        Some(intent)
    }

    pub fn flush_sustain_note_off(
        &mut self,
        off_tick: Tick,
        out_cmds: &mut Vec<PhonationCmd>,
    ) -> bool {
        let Some(note_id) = self.sustain_note_id else {
            return false;
        };
        let onset = self.sustain_onset_tick.unwrap_or(off_tick);
        let off_tick = off_tick.max(onset);
        out_cmds.push(PhonationCmd::NoteOff { note_id, off_tick });
        self.phonation_engine.register_external_note_off();
        self.sustain_note_id = None;
        self.sustain_onset_tick = None;
        true
    }

    pub fn tick_phonation(
        &mut self,
        tb: &Timebase,
        now: Tick,
        rhythms: &NeuralRhythms,
        social: Option<&SocialDensityTrace>,
        social_coupling: f32,
    ) -> PhonationBatch {
        let hop_tick = (tb.hop as Tick).max(1);
        let frame_end = now.saturating_add(hop_tick);
        let birth_onset_tick = if self.birth_pending() {
            self.birth_onset_tick
                .filter(|tick| *tick >= now && *tick < frame_end)
        } else {
            None
        };
        let ctx = CoreTickCtx {
            now_tick: now,
            frame_end,
            fs: tb.fs,
            rhythms: *rhythms,
        };
        let state = CoreState {
            is_alive: self.is_alive(),
        };
        let mut cmds = Vec::new();
        let mut events = Vec::new();
        let mut onsets = Vec::new();
        let birth = self.phonation_engine.tick(
            &ctx,
            &state,
            birth_onset_tick,
            social,
            social_coupling,
            &mut cmds,
            &mut events,
            &mut onsets,
        );
        if let Some(gate) = birth.gate {
            self.birth_onset_gate = Some(gate);
        }
        let gate_hint = self.phonation_engine.next_gate_index_hint();
        if birth_onset_tick.is_some() && !birth.applied {
            self.phonation_engine.notify_birth_onset(gate_hint);
        }
        let mut notes = Vec::new();
        for event in events {
            if let Some(note) = self.build_phonation_note_spec(event, None) {
                notes.push(note);
            }
        }
        if let (OnBirthPhonation::Sustain, Some(onset_tick), None) = (
            self.phonation.on_birth,
            birth_onset_tick,
            self.sustain_note_id,
        ) {
            let event = PhonationNoteEvent {
                note_id: self.phonation_engine.next_note_id,
                onset_tick,
            };
            if let Some(note) = self.build_phonation_note_spec(event, Some(Tick::MAX)) {
                let note_id = event.note_id;
                self.phonation_engine.next_note_id =
                    self.phonation_engine.next_note_id.wrapping_add(1);
                self.phonation_engine.register_external_note_on();
                cmds.push(PhonationCmd::NoteOn {
                    note_id,
                    kick: PhonationKick::Birth,
                });
                onsets.push(OnsetEvent {
                    gate: self.birth_onset_gate.unwrap_or(gate_hint),
                    onset_tick,
                    strength: 1.0,
                });
                notes.push(note);
                self.sustain_note_id = Some(note_id);
                self.sustain_onset_tick = Some(onset_tick);
                self.birth_fired = true;
                self.birth_pending = false;
            } else {
                // No retrigger: if the note spec is invalid (e.g., amp too low),
                // sustain stays silent and we consume the birth.
                self.birth_fired = true;
                self.birth_pending = false;
            }
        }
        self.emit_sustain_updates(now, frame_end, tb, rhythms, &mut cmds);
        if self.sustain_note_id.is_some() && !self.is_alive() {
            self.flush_sustain_note_off(now, &mut cmds);
        }
        PhonationBatch {
            source_id: self.id,
            cmds,
            notes,
            onsets,
        }
    }

    pub fn update_self_confidence_from_perc(
        &mut self,
        space: &Log2Space,
        perc_landscape: &Landscape,
        perc_tick: Tick,
    ) {
        if self.pred_intent_records.is_empty() {
            return;
        }
        let mut next_records = VecDeque::with_capacity(self.pred_intent_records.len());
        for record in self.pred_intent_records.drain(..) {
            if perc_tick >= record.end {
                continue;
            }
            if record.eval_tick <= perc_tick && perc_tick < record.end {
                let perc_c =
                    sample_scan_linear_log2(space, &perc_landscape.consonance, record.freq_hz);
                if !perc_c.is_finite() || !record.pred_c_statepm1.is_finite() {
                    continue;
                }
                let err_c = perc_c - record.pred_c_statepm1;
                if !err_c.is_finite() {
                    continue;
                }
                let agreement01 = 1.0 - (err_c.abs() / 2.0).clamp(0.0, 1.0);
                let lr = 0.05;
                self.self_confidence = lerp(self.self_confidence, agreement01, lr).clamp(0.0, 1.0);
                continue;
            }
            next_records.push_back(record);
        }
        self.pred_intent_records = next_records;
    }

    fn record_pred_intent(
        &mut self,
        intent: &Intent,
        pred_c_scan_at: &mut dyn FnMut(Tick) -> Option<std::sync::Arc<[f32]>>,
        now: Tick,
        eval_tick: Tick,
        landscape: &Landscape,
    ) {
        let pred_c_statepm1 = pred_c_scan_at(eval_tick)
            .map(|scan| sample_scan_linear_log2(&landscape.space, scan.as_ref(), intent.freq_hz))
            .unwrap_or(0.0);
        let pred_c_statepm1 = if pred_c_statepm1.is_finite() {
            pred_c_statepm1
        } else {
            0.0
        };
        let record = PredIntentRecord {
            intent_id: intent.intent_id,
            onset: intent.onset,
            end: intent.onset.saturating_add(intent.duration),
            freq_hz: intent.freq_hz,
            pred_c_statepm1,
            created_at: now,
            eval_tick,
        };
        if self.pred_intent_records.len() >= self.pred_intent_records_cap {
            let _ = self.pred_intent_records.pop_front();
        }
        self.pred_intent_records.push_back(record);
    }

    fn build_phonation_note_spec(
        &mut self,
        event: PhonationNoteEvent,
        hold_ticks: Option<Tick>,
    ) -> Option<PhonationNoteSpec> {
        let amp = self.release_gain.clamp(0.0, 1.0);
        if amp <= Self::AMP_EPS {
            return None;
        }
        let freq_hz = self.body.base_freq_hz();
        if !freq_hz.is_finite() || freq_hz <= 0.0 {
            return None;
        }
        self.last_chosen_freq_hz = freq_hz;
        Some(PhonationNoteSpec {
            note_id: event.note_id,
            onset: event.onset_tick,
            hold_ticks,
            freq_hz,
            amp,
            body: self.body_snapshot(),
            articulation: self.articulation.clone(),
        })
    }

    fn emit_sustain_updates(
        &self,
        now: Tick,
        frame_end: Tick,
        tb: &Timebase,
        rhythms: &NeuralRhythms,
        out_cmds: &mut Vec<PhonationCmd>,
    ) {
        let note_id = match self.sustain_note_id {
            Some(note_id) => note_id,
            None => return,
        };
        if !self.is_alive() {
            return;
        }
        let cadence = self.phonation.sustain_update.cadence;
        if cadence == SustainUpdateCadence::Off {
            return;
        }
        let mut update = PhonationUpdate::default();
        for target in &self.phonation.sustain_update.what {
            match target {
                SustainUpdateTarget::Pitch => {
                    let freq_hz = self.body.base_freq_hz();
                    if freq_hz.is_finite() && freq_hz > 0.0 {
                        update.freq_hz = Some(freq_hz);
                    }
                }
                SustainUpdateTarget::Gain => {
                    let amp = self.body.amp();
                    if amp.is_finite() {
                        update.amp = Some(amp);
                    }
                }
            }
        }
        if update.is_empty() {
            return;
        }
        match cadence {
            SustainUpdateCadence::Hop => {
                out_cmds.push(PhonationCmd::Update {
                    note_id,
                    at_tick: Some(now),
                    update,
                });
            }
            SustainUpdateCadence::Gate => {
                for tick in Self::gate_ticks_for_window(now, frame_end, tb, rhythms) {
                    out_cmds.push(PhonationCmd::Update {
                        note_id,
                        at_tick: Some(tick),
                        update,
                    });
                }
            }
            SustainUpdateCadence::Off => {}
        }
    }

    pub(crate) fn gate_ticks_for_window(
        now: Tick,
        frame_end: Tick,
        tb: &Timebase,
        rhythms: &NeuralRhythms,
    ) -> Vec<Tick> {
        let ctx = CoreTickCtx {
            now_tick: now,
            frame_end,
            fs: tb.fs,
            rhythms: *rhythms,
        };
        let mut clock = ThetaGateClock::default();
        let mut candidates = Vec::new();
        clock.gather_candidates(&ctx, &mut candidates);
        Self::ticks_from_candidates(&candidates, now, frame_end)
    }

    pub(crate) fn ticks_from_candidates(
        candidates: &[CandidatePoint],
        now: Tick,
        frame_end: Tick,
    ) -> Vec<Tick> {
        let mut ticks = Vec::new();
        let mut last_tick = None;
        for candidate in candidates {
            let tick = candidate.tick;
            if tick < now || tick >= frame_end {
                continue;
            }
            if let Some(prev) = last_tick
                && tick <= prev
            {
                if tick < prev {
                    debug_assert!(tick > prev, "gate ticks must be strictly increasing");
                }
                continue;
            }
            ticks.push(tick);
            last_tick = Some(tick);
        }
        ticks
    }

    fn body_snapshot(&self) -> BodySnapshot {
        match &self.body {
            AnySoundBody::Sine(body) => BodySnapshot {
                kind: "sine".to_string(),
                amp_scale: body.amp.clamp(0.0, 1.0),
                brightness: 0.0,
                noise_mix: 0.0,
            },
            AnySoundBody::Harmonic(body) => BodySnapshot {
                kind: "harmonic".to_string(),
                amp_scale: body.amp.clamp(0.0, 1.0),
                brightness: body.genotype.brightness.clamp(0.0, 1.0),
                noise_mix: body.genotype.jitter.clamp(0.0, 1.0),
            },
        }
    }
}

fn gate_duration_sec_from_theta(theta_hz: f32, planning: &PlanningConfig) -> f32 {
    let fallback = 0.08;
    if !theta_hz.is_finite() || theta_hz <= 0.0 {
        return fallback;
    }

    let mut min_s = planning.gate_dur_min_sec;
    let mut max_s = planning.gate_dur_max_sec;

    if !min_s.is_finite() || min_s <= 0.0 {
        min_s = 0.010;
    }
    if !max_s.is_finite() || max_s <= 0.0 {
        max_s = 0.50;
    }
    if min_s > max_s {
        std::mem::swap(&mut min_s, &mut max_s);
    }

    let scale = if planning.gate_dur_scale.is_finite() {
        planning.gate_dur_scale
    } else {
        0.90
    };

    let period_sec = 1.0 / theta_hz;
    let dur_sec = (period_sec * scale).clamp(min_s, max_s);
    if dur_sec.is_finite() {
        dur_sec
    } else {
        fallback
    }
}

fn sec_to_tick_at_least_one(tb: &Timebase, sec: f32) -> Tick {
    let t = tb.sec_to_tick(sec);
    if t < 1 { 1 } else { t }
}

#[cfg(test)]
mod tests {
    use super::AgentMetadata;
    use super::{gate_duration_sec_from_theta, sec_to_tick_at_least_one};
    use crate::core::modulation::NeuralRhythms;
    use crate::core::timebase::{Tick, Timebase};
    use crate::life::phonation_engine::{
        CandidatePoint, ClockSource, CoreTickCtx, PhonationClock, PhonationCmd, ThetaGateClock,
    };
    use crate::life::scenario::{
        IndividualConfig, LifeConfig, OnBirthPhonation, PhonationConfig, PlanPitchMode,
        PlanningConfig, SustainUpdateCadence, SustainUpdateConfig, SustainUpdateTarget,
    };

    #[test]
    fn gate_duration_tracks_theta() {
        let planning = PlanningConfig::default();
        let d6 = gate_duration_sec_from_theta(6.0, &planning);
        assert!((d6 - 0.15).abs() < 1e-6);

        let d3 = gate_duration_sec_from_theta(3.0, &planning);
        assert!((d3 - 0.3).abs() < 1e-6);

        let d0 = gate_duration_sec_from_theta(0.0, &planning);
        assert!((d0 - 0.08).abs() < 1e-6);
    }

    #[test]
    fn gate_duration_handles_min_max_swap() {
        let mut planning = PlanningConfig::default();
        planning.gate_dur_min_sec = 0.5;
        planning.gate_dur_max_sec = 0.01;
        let dur = gate_duration_sec_from_theta(3.0, &planning);
        assert!(dur.is_finite());
        assert!(dur >= 0.01 && dur <= 0.5);
    }

    #[test]
    fn gate_duration_tick_is_at_least_one() {
        let planning = PlanningConfig {
            gate_dur_scale: 0.001,
            gate_dur_min_sec: 0.0001,
            gate_dur_max_sec: 0.0002,
            pitch_mode: PlanPitchMode::Off,
        };
        let dur_sec = gate_duration_sec_from_theta(1000.0, &planning);
        let tb = Timebase { fs: 10.0, hop: 64 };
        let raw = tb.sec_to_tick(dur_sec);
        assert!(raw < 1);
        let clamped = sec_to_tick_at_least_one(&tb, dur_sec);
        assert_eq!(clamped, 1);
    }

    #[test]
    fn sustain_update_hop_emits_updates_without_candidates() {
        let tb = Timebase { fs: 1000.0, hop: 4 };
        let mut life = LifeConfig::default();
        life.phonation = PhonationConfig {
            on_birth: OnBirthPhonation::Sustain,
            sustain_update: SustainUpdateConfig {
                cadence: SustainUpdateCadence::Hop,
                what: vec![SustainUpdateTarget::Pitch],
                smoothing: 0.0,
            },
            ..PhonationConfig::default()
        };
        let cfg = IndividualConfig {
            freq: 220.0,
            amp: 0.5,
            life,
            tag: None,
        };
        let metadata = AgentMetadata {
            id: 1,
            tag: None,
            group_idx: 0,
            member_idx: 0,
        };
        let mut agent = cfg.spawn(1, 0, metadata, tb.fs, 0);
        agent.set_birth_onset_tick(Some(0));

        let rhythms = NeuralRhythms::default();
        let batch0 = agent.tick_phonation(&tb, 0, &rhythms, None, 0.0);
        let (note_on0, update0, note_off0) = count_cmds(&batch0.cmds);
        assert_eq!(note_on0, 1);
        assert_eq!(update0, 1);
        assert_eq!(note_off0, 0);
        let note_id = batch0.cmds.iter().find_map(|cmd| match cmd {
            PhonationCmd::NoteOn { note_id, .. } => Some(*note_id),
            _ => None,
        });
        let Some(note_id) = note_id else {
            panic!("expected sustain note on");
        };

        let hop = tb.hop as u64;
        let batch1 = agent.tick_phonation(&tb, hop, &rhythms, None, 0.0);
        let (note_on1, update1, note_off1) = count_cmds(&batch1.cmds);
        assert_eq!(note_on1, 0);
        assert_eq!(update1, 1);
        assert_eq!(note_off1, 0);
        assert!(
            batch1.cmds.iter().all(|cmd| match cmd {
                PhonationCmd::Update { note_id: id, .. } => *id == note_id,
                _ => true,
            }),
            "update should target sustain note id"
        );

        agent.release_gain = 0.0;
        let now2 = hop.saturating_mul(2);
        let batch2 = agent.tick_phonation(&tb, now2, &rhythms, None, 0.0);
        let (note_on2, update2, note_off2) = count_cmds(&batch2.cmds);
        assert_eq!(note_on2, 0);
        assert_eq!(update2, 0);
        assert_eq!(note_off2, 1);
        let off = batch2.cmds.iter().find_map(|cmd| match cmd {
            PhonationCmd::NoteOff {
                note_id: id,
                off_tick,
            } => Some((*id, *off_tick)),
            _ => None,
        });
        assert_eq!(off, Some((note_id, now2)));
    }

    #[test]
    fn sustain_update_gate_emits_updates_on_gate_ticks() {
        let tb = Timebase {
            fs: 1000.0,
            hop: 64,
        };
        let mut life = LifeConfig::default();
        life.phonation = PhonationConfig {
            on_birth: OnBirthPhonation::Sustain,
            sustain_update: SustainUpdateConfig {
                cadence: SustainUpdateCadence::Gate,
                what: vec![SustainUpdateTarget::Pitch],
                smoothing: 0.0,
            },
            ..PhonationConfig::default()
        };
        let cfg = IndividualConfig {
            freq: 220.0,
            amp: 0.5,
            life,
            tag: None,
        };
        let metadata = AgentMetadata {
            id: 2,
            tag: None,
            group_idx: 0,
            member_idx: 0,
        };
        let mut agent = cfg.spawn(2, 0, metadata, tb.fs, 0);
        agent.set_birth_onset_tick(Some(0));

        let mut rhythms = NeuralRhythms::default();
        rhythms.theta.freq_hz = 40.0;
        rhythms.theta.phase = 0.0;
        let now: Tick = 0;
        let frame_end = now.saturating_add(tb.hop as u64);
        let batch = agent.tick_phonation(&tb, now, &rhythms, None, 0.0);
        let updates: Vec<u64> = batch
            .cmds
            .iter()
            .filter_map(|cmd| match cmd {
                PhonationCmd::Update { at_tick, .. } => *at_tick,
                _ => None,
            })
            .collect();
        assert!(!updates.is_empty(), "expected at least one gate update");
        assert!(
            updates.iter().all(|tick| *tick >= now && *tick < frame_end),
            "updates must be within frame window"
        );
        assert!(
            updates.windows(2).all(|pair| pair[0] < pair[1]),
            "update ticks must be strictly increasing"
        );

        let ctx = CoreTickCtx {
            now_tick: now,
            frame_end,
            fs: tb.fs,
            rhythms,
        };
        let mut clock = ThetaGateClock::default();
        let mut candidates = Vec::new();
        clock.gather_candidates(&ctx, &mut candidates);
        let expected: Vec<u64> = candidates
            .iter()
            .map(|c| c.tick)
            .filter(|tick| *tick >= now && *tick < frame_end)
            .collect();
        assert_eq!(updates, expected);
    }

    #[test]
    fn gate_ticks_skip_non_monotonic_candidates() {
        let candidates = vec![
            CandidatePoint {
                tick: 10,
                gate: 0,
                theta_pos: 0.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
            CandidatePoint {
                tick: 10,
                gate: 1,
                theta_pos: 1.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
            CandidatePoint {
                tick: 12,
                gate: 2,
                theta_pos: 2.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
            CandidatePoint {
                tick: 12,
                gate: 3,
                theta_pos: 3.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
            CandidatePoint {
                tick: 14,
                gate: 4,
                theta_pos: 4.0,
                phase_in_gate: 0.0,
                sources: vec![ClockSource::GateBoundary],
            },
        ];
        let ticks = super::Individual::ticks_from_candidates(&candidates, 10, 20);
        assert_eq!(ticks, vec![10, 12, 14]);
    }

    fn count_cmds(cmds: &[PhonationCmd]) -> (usize, usize, usize) {
        let mut note_on = 0;
        let mut update = 0;
        let mut note_off = 0;
        for cmd in cmds {
            match cmd {
                PhonationCmd::NoteOn { .. } => note_on += 1,
                PhonationCmd::Update { .. } => update += 1,
                PhonationCmd::NoteOff { .. } => note_off += 1,
            }
        }
        (note_on, update, note_off)
    }
}

#[cfg(test)]
impl Individual {
    pub fn debug_last_error_state(&self) -> ErrorState {
        self.last_error_state
    }
}

impl AudioAgent for Individual {
    fn id(&self) -> u64 {
        self.id
    }

    fn metadata(&self) -> &AgentMetadata {
        self.metadata()
    }

    fn render_wave(
        &mut self,
        buffer: &mut [f32],
        fs: f32,
        _current_frame: u64,
        dt_sec: f32,
        landscape: &Landscape,
        global_coupling: f32,
    ) {
        if buffer.is_empty() {
            return;
        }
        let dt_per_sample = dt_sec / buffer.len() as f32;
        let mut rhythms = landscape.rhythm;
        for sample in buffer.iter_mut() {
            self.update_pitch_target(&rhythms, dt_per_sample, landscape);
            let signal =
                self.update_articulation(dt_per_sample, &rhythms, landscape, global_coupling);
            if signal.is_active {
                self.body
                    .articulate_wave(sample, fs, dt_per_sample, &signal);
            }
            rhythms.advance_in_place(dt_per_sample);
        }
    }

    fn render_spectrum(&mut self, amps: &mut [f32], space: &Log2Space) {
        let signal = self.last_signal;
        if !signal.is_active || signal.amplitude <= 0.0 {
            return;
        }
        self.body.project_spectral_body(amps, space, &signal);
    }

    fn is_alive(&self) -> bool {
        self.articulation.is_alive() && self.release_gain > 0.0
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
