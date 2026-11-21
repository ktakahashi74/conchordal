use crate::core::landscape::Landscape;
use crate::core::log2space::Log2Space;
use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::dcc_coupler::ListenerPressure;
use crate::life::control_adapters::adaptation_config_from_control;
use crate::life::phonation_engine::{
    CoreState, CoreTickCtx, OnsetEvent, PhonationClock, PhonationEngine, ToneCmd, ToneId,
    ToneOnEvent, ToneUpdate,
};
pub use crate::life::phrase_boundary::PhraseTone;
use crate::life::sound::{BodySnapshot, RenderModulatorSpec};
#[cfg(test)]
use crate::scenario::control::ControlUpdate;
use crate::scenario::control::{
    BodyMethod, PhonationGate, PitchApplyMode, PitchMode, VoiceControl,
};
use crate::scenario::lifecycle::LifecycleConfig;
use crate::scenario::{
    ArticulationCoreConfig, PhonationMode, PhonationTimingUpdateKind, VoiceSpec,
};
use rand::SeedableRng;
#[cfg(test)]
use voice_control::Dirty;
use voice_phonation::{onset_distance_cents, select_pitch_application};
use voice_render::TrackedRenderNote;

#[path = "articulation_core.rs"]
pub(crate) mod articulation_core;
#[path = "harmonic_voice.rs"]
mod harmonic_voice;
#[path = "pitch_controller.rs"]
pub(crate) mod pitch_controller;
#[path = "pitch_core.rs"]
pub(crate) mod pitch_core;
#[path = "sequence_execution.rs"]
pub(crate) mod sequence_execution;
#[path = "sound_body.rs"]
pub(crate) mod sound_body;

#[path = "voice_control.rs"]
mod voice_control;
#[path = "voice_phonation.rs"]
mod voice_phonation;
#[path = "voice_render.rs"]
mod voice_render;

pub use articulation_core::{
    AnyArticulationCore, ArticulationCore, ArticulationSignal, ArticulationState,
    ArticulationWrapper, DroneCore, KuramotoCore, PinkNoise, PlannedGate, PlannedPitch,
    Sensitivity, SequencedCore,
};
pub use pitch_controller::PitchController;
pub use pitch_core::{
    AnyPitchCore, PitchCore, PitchHillClimbPitchCore, PitchPeakSamplerCore, TargetProposal,
};
pub use sound_body::{AnySoundBody, HarmonicBody, SineBody, SoundBody};

#[derive(Debug, Clone, Default)]
pub struct VoiceMetadata {
    pub population_id: u64,
    pub member_idx: usize,
    pub generation: u32,
    pub parent_id: Option<u64>,
}

#[derive(Debug)]
pub(crate) struct VoiceSupportState {
    pub(crate) body: super::support_action::Body,
    #[cfg(feature = "research")]
    pub(crate) renderer_phase3: bool,
    onset: Option<(BodySnapshot, crate::life::sound::Tone)>,
    held_opportunities: Vec<(Tick, Option<Tick>)>,
    clock_fs: f32,
    clock_step: u64,
    pub(crate) template: Option<std::sync::Arc<[(isize, f32)]>>,
    template_key: Option<(f32, f32, u32, f32, f32)>,
}

#[derive(Debug)]
pub(crate) struct VoiceHarmonicState {
    pub(crate) member: bool,
    pub(crate) temperature_bonus: f32,
    pub(crate) events: Vec<super::harmonic_rhythm::VoiceEvent>,
    waiting: bool,
    pub(crate) neighbors: (Vec<f32>, Vec<f32>),
}

#[derive(Debug)]
pub struct Voice {
    pub(crate) support: VoiceSupportState,
    pub(crate) harmonic: VoiceHarmonicState,
    pub(crate) birth_tick: Option<Tick>,
    pub(super) sequence_control: Option<sequence_execution::DeferredPitch>,
    pub id: u64,
    pub metadata: VoiceMetadata,
    fixed_body_method: BodyMethod,
    fixed_phonation_timing: PhonationTimingUpdateKind,
    pub base_control: VoiceControl,
    pub effective_control: VoiceControl,
    pub articulation: ArticulationWrapper,
    pub(crate) pitch_ctl: PitchController,
    pub phonation_engine: PhonationEngine,
    pub(crate) social_coupling: f32,
    pub body: AnySoundBody,
    pub last_signal: ArticulationSignal,
    pub(crate) selection_environment: bool,
    pub(crate) phonation_scratch: PhonationScratch,
    active_render_notes: Vec<TrackedRenderNote>,
    amp_update_pending: bool,
    sustain_participation: bool,
    participation_pitch: ParticipationPitchState,
    pub(crate) release: ReleaseState,
    control_tick: Option<Tick>,
    pub(crate) life_accumulator: Option<super::telemetry::LifeAccumulator>,
    voice_adsr: Option<super::sound::ToneAdsr>,
    /// One-way latch for `phonate_when_viable()`: starts open for
    /// `PhonationGate::Immediate` (the default), starts closed for
    /// `PhonationGate::WhenViable` until perceived consonance reaches the low
    /// bound of the viability window, then never re-closes.
    phonation_gate_open: bool,
}

#[derive(Debug)]
struct ParticipationPitchState {
    steps: Vec<ParticipationPitchStep>,
    application: PitchApplyMode,
    adopted: bool,
    adoption_application: PitchApplyMode,
    last_onset_log2: Option<f32>,
    pending: Option<f32>,
    glide_target: Option<f32>,
    rearticulations: Vec<Rearticulation>,
    pulse_update_pending: bool,
}

#[derive(Debug)]
pub(crate) struct ReleaseState {
    pub(crate) gain: f32,
    pub(crate) sec: f32,
    pub(crate) pending: bool,
    pub(crate) remove_pending: bool,
}

#[derive(Clone, Debug)]
#[cfg_attr(test, derive(serde::Serialize, serde::Deserialize))]
pub struct ToneSpec {
    #[cfg_attr(test, serde(default, skip_serializing_if = "Option::is_none"))]
    pub phrase: Option<PhraseTone>,
    /// Whether this onset came from a clock candidate.
    pub participation_onset: bool,
    pub tone_id: ToneId,
    pub onset: Tick,
    pub hold_ticks: Option<Tick>,
    pub freq_hz: f32,
    pub amp: f32,
    pub smoothing_tau_sec: f32,
    pub body: BodySnapshot,
    pub render_modulator: RenderModulatorSpec,
    pub adsr: Option<super::sound::ToneAdsr>,
}

impl ToneSpec {
    fn from_event(
        event: ToneOnEvent,
        freq_hz: f32,
        amp: f32,
        hold_ticks: Option<Tick>,
        body: BodySnapshot,
        render_modulator: RenderModulatorSpec,
        adsr: Option<super::sound::ToneAdsr>,
    ) -> Self {
        Self {
            phrase: None,
            participation_onset: event.participation_onset,
            tone_id: event.tone_id,
            onset: event.onset_tick,
            hold_ticks,
            freq_hz,
            amp,
            smoothing_tau_sec: Voice::PHONATION_UPDATE_SMOOTH_TAU_SEC,
            body,
            render_modulator,
            adsr,
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct PhonationScratch {
    events: Vec<ToneOnEvent>,
}

#[derive(Clone, Copy, Debug)]
struct ParticipationPitchStep {
    at: Tick,
    before_log2: f32,
    target_log2: f32,
    dt: f32,
}

#[derive(Clone, Copy, Debug)]
struct Rearticulation {
    at: Tick,
    pitch_log2: f32,
}

#[derive(Clone, Debug, Default)]
#[cfg_attr(test, derive(serde::Serialize, serde::Deserialize))]
pub struct PhonationBatch {
    pub body_policy: Option<super::action_candidates::PolicySnapshot>,
    pub source_id: u64,
    pub source_generation: u32,
    pub routing: crate::scenario::control::Routing,
    pub cmds: Vec<ToneCmd>,
    pub tones: Vec<ToneSpec>,
    pub onsets: Vec<OnsetEvent>,
}

impl PhonationBatch {
    pub(crate) fn clear(&mut self) {
        self.body_policy = None;
        self.cmds.clear();
        self.tones.clear();
        self.onsets.clear();
    }
}

impl Voice {
    pub(crate) const AMP_EPS: f32 = 1e-6;

    #[allow(clippy::too_many_arguments)]
    pub fn spawn_from_control(
        control: VoiceControl,
        articulation_config: ArticulationCoreConfig,
        assigned_id: u64,
        start_frame: u64,
        metadata: VoiceMetadata,
        fs: f32,
        landscape: Option<&Landscape>,
        seed_offset: u64,
    ) -> Self {
        let seed = seed_offset ^ assigned_id ^ start_frame.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);

        control
            .validate_pitch_application()
            .expect("valid voice pitch application");
        let effective_control = control.clone();
        let pitch_application = select_pitch_application(&effective_control.pitch, None);
        let target_freq = effective_control.pitch.freq.max(1.0);
        let target_pitch_log2 = target_freq.log2();
        let integration_window = PitchController::integration_window_for_freq(target_freq);
        let selection_environment = matches!(
            &articulation_config,
            ArticulationCoreConfig::Entrain {
                lifecycle: LifecycleConfig::Sustain {
                    selection_environment: true,
                    ..
                },
                ..
            }
        );

        let core =
            AnyArticulationCore::from_config(&articulation_config, fs, assigned_id, &mut rng);

        let mut phonation_engine =
            PhonationEngine::from_spec(&effective_control.phonation.spec, seed);
        #[cfg(feature = "research")]
        if effective_control.phonation.syncopate > 0.0 {
            phonation_engine.felt = Some(Box::new(super::phonation_engine::FeltRhythm::new(
                effective_control.phonation.syncopate,
                seed_offset
                    ^ assigned_id.rotate_left(17)
                    ^ u64::from(metadata.generation).rotate_left(39)
                    ^ 0x4645_4c54_5f42_4541,
            )));
        }
        if effective_control.phonation.phrase_boundary > 0.0
            && effective_control.supports_phrase_boundary()
            && let Some(period) = phonation_engine.clock.intrinsic_period_sec()
        {
            // Author mapping: one initial bodily cycle is a quarter note.
            phonation_engine.phrase_boundary =
                Some(crate::life::phrase_boundary::PhraseBoundary::new(
                    assigned_id,
                    metadata.generation,
                    fs,
                    effective_control.phonation.phrase_boundary,
                    [12., 4. * period, 4. * period],
                ));
        }
        if effective_control.phonation.rest_on_habituation > 0.0 {
            phonation_engine.span_gap = Some(crate::life::span_gap::SpanGap::new(
                effective_control.phonation.rest_on_habituation,
                seed_offset,
                assigned_id,
                metadata.generation,
                fs,
                target_freq,
            ));
        }
        if effective_control.phonation.sway > 0.0 {
            phonation_engine.sway_rng = Some(rand::SeedableRng::seed_from_u64(
                seed_offset
                    ^ assigned_id.rotate_left(17)
                    ^ u64::from(metadata.generation).rotate_left(39)
                    ^ 0x5357_4159_5f56_4f49,
            ));
        }
        if let Some(periods) = effective_control.phonation.rest_periods {
            let mut finite = effective_control.phonation.spec.clone();
            finite.duration = effective_control
                .phonation
                .finite_duration
                .clone()
                .expect("validated finite rest duration");
            finite.sustain_participation = false;
            let config = crate::life::control_adapters::phonation_config_from_spec(&finite);
            phonation_engine.enable_rest(&config.duration, periods);
        }
        let sustain_participation = effective_control
            .phonation
            .spec
            .is_sustained_participation();
        let tie = effective_control.phonation.tie;
        #[cfg(test)]
        let force_tie = effective_control.phonation.force_tie_attack;
        #[cfg(not(test))]
        let force_tie = false;
        let persistence = effective_control.phonation.articulation_persistence;
        #[cfg(test)]
        let force_base = effective_control.phonation.force_articulation_base;
        #[cfg(not(test))]
        let force_base = false;
        // An inherited policy starts in Base even while mode or J suspends its use.
        if effective_control.pitch.mode == PitchMode::Free
            && sustain_participation
            && ((persistence > 0.0 && persistence < 1.0) || force_base || tie > 0.0 || force_tie)
            && let Some(shape_slot) = phonation_engine.clock.shape_slot()
        {
            *shape_slot = Some(super::shape_policy::ShapePolicy::new(
                if force_base || !(persistence > 0.0 && persistence < 1.0) {
                    1.0
                } else {
                    persistence
                },
                seed,
                fs,
            ));
            shape_slot.as_mut().unwrap().retention_active =
                force_base || (persistence > 0.0 && persistence < 1.0);
            shape_slot.as_mut().unwrap().enabled = effective_control
                .supports_articulation_persistence()
                && shape_slot.as_ref().unwrap().retention_active;
            if tie > 0.0 || force_tie {
                shape_slot.as_mut().unwrap().tie =
                    Some(super::shape_policy::TiePolicy::new(tie, seed));
            }
            #[cfg(test)]
            if effective_control.phonation.observe_articulation {
                shape_slot.as_mut().unwrap().observations =
                    Some(vec![(start_frame, super::shape_policy::Shape::Base)]);
            }
        }
        let social_coupling = effective_control.phonation.spec.social_coupling();

        let mut pitch = AnyPitchCore::from_control(&effective_control.pitch, target_pitch_log2);
        let adaptation_config = adaptation_config_from_control(&effective_control.adaptation);
        let adaptation =
            crate::life::adaptation::AdaptationContext::from_config(&adaptation_config, 0);

        let body = sound_body::build_sound_body_from_control(
            &effective_control,
            target_freq,
            fs,
            landscape,
            &mut rng,
        );

        // Perceive with your own body: the LOO pitch search subtracts the voice's
        // own partials. Ratios are fixed at spawn (timbre patches never change the
        // base ratios), so this one-time wiring is sufficient.
        pitch.set_body_ratios(body.snapshot().ratios);

        let initial_dwell = (effective_control.pitch.dwell > 0.0).then(|| {
            let (level, excluded) = landscape.map_or((0.5, false), |landscape| {
                pitch.dwell_level(
                    landscape,
                    target_freq,
                    effective_control.pitch.leave_self_out,
                )
            });
            pitch_controller::DwellRecord {
                at: None,
                event: "state",
                elapsed_sec: 0.0,
                phase: 0.0,
                remainder: 0.0,
                level,
                source: if excluded { "self_excluded" } else { "mixed" },
                freq_hz: target_freq,
                weight: effective_control.pitch.dwell,
            }
        });
        let mut pitch_ctl = PitchController::new(
            pitch,
            adaptation,
            target_pitch_log2,
            integration_window,
            rng,
        );
        pitch_ctl.set_adaptation_enabled(effective_control.adaptation.enabled);
        pitch_ctl.dwell_state = initial_dwell;
        let respond = effective_control.pitch.respond_to_onsets;
        if respond.is_finite() && respond > 0.0 {
            pitch_ctl.respond = Some(pitch_controller::OnsetResponse::new(
                respond,
                landscape.map_or(0, |landscape| landscape.space.n_bins()),
            ));
        }
        if effective_control.pitch.follow_own_sequence > 0.0
            && effective_control.supports_own_sequence()
        {
            let half = effective_control.pitch.range_oct.clamp(0.0, 6.0) * 0.5;
            pitch_ctl.own_sequence = Some(crate::life::own_sequence::OwnSequence::new(
                seed_offset,
                assigned_id,
                metadata.generation,
                fs,
                effective_control.pitch.follow_own_sequence,
                (target_pitch_log2 - half, target_pitch_log2 + half),
            ));
        }

        if effective_control.pitch.melodic_continuity > 0.0
            && effective_control.supports_own_sequence()
        {
            let half = effective_control.pitch.range_oct.clamp(0.0, 6.0) * 0.5;
            pitch_ctl.melodic = Some(crate::life::melodic_continuity::MelodicContinuity::new(
                fs,
                effective_control.pitch.melodic_continuity,
                (target_pitch_log2 - half, target_pitch_log2 + half),
            ));
        }
        let body_envelope = &effective_control.body.envelope;
        let voice_adsr = Some(super::sound::ToneAdsr {
            attack_sec: body_envelope.attack_sec,
            decay_sec: body_envelope.decay_sec,
            sustain_level: body_envelope.sustain_level,
            release_sec: body_envelope.release_sec,
        });

        let (articulation_core, lifecycle_label, default_by_articulation, breath_gain_init) =
            match &articulation_config {
                ArticulationCoreConfig::Entrain {
                    lifecycle,
                    breath_gain_init,
                    ..
                } => {
                    let life_label = match lifecycle {
                        LifecycleConfig::Decay { .. } => "decay",
                        LifecycleConfig::Sustain { .. } => "sustain",
                    };
                    ("entrain", life_label, 1.0, *breath_gain_init)
                }
                ArticulationCoreConfig::Seq {
                    breath_gain_init, ..
                } => ("seq", "none", 1.0, *breath_gain_init),
                ArticulationCoreConfig::Drone {
                    breath_gain_init, ..
                } => ("drone", "none", 1.0, *breath_gain_init),
            };
        let breath_gain = breath_gain_init
            .unwrap_or(default_by_articulation)
            .clamp(0.0, 1.0);
        tracing::debug!(
            target: "rhythm::spawn",
            id = assigned_id,
            population_id = metadata.population_id,
            member_idx = metadata.member_idx,
            articulation = articulation_core,
            lifecycle = lifecycle_label,
            breath_gain_init,
            breath_gain
        );

        let phonation_gate_open =
            !matches!(effective_control.phonation.gate, PhonationGate::WhenViable);

        let mut voice = Voice {
            support: VoiceSupportState {
                body: Default::default(),
                #[cfg(feature = "research")]
                renderer_phase3: false,
                onset: None,
                held_opportunities: Vec::new(),
                clock_fs: 0.,
                clock_step: 1,
                template: None,
                template_key: None,
            },
            harmonic: VoiceHarmonicState {
                member: false,
                temperature_bonus: 0.0,
                events: Vec::new(),
                waiting: false,
                neighbors: Default::default(),
            },
            birth_tick: None,
            sequence_control: ((effective_control.pitch.follow_own_sequence > 0.0
                || effective_control.pitch.melodic_continuity > 0.0)
                && effective_control.supports_own_sequence())
            .then(sequence_execution::DeferredPitch::default),
            participation_pitch: ParticipationPitchState {
                steps: Vec::new(),
                application: pitch_application,
                adopted: false,
                adoption_application: pitch_application,
                last_onset_log2: None,
                pending: None,
                glide_target: None,
                rearticulations: Vec::new(),
                pulse_update_pending: false,
            },
            release: ReleaseState {
                gain: 1.0,
                sec: 0.03,
                pending: false,
                remove_pending: false,
            },
            sustain_participation,
            id: assigned_id,
            metadata,
            fixed_body_method: effective_control.body.method,
            fixed_phonation_timing: effective_control.phonation.spec.timing_update_kind(),
            base_control: control,
            effective_control,
            articulation: ArticulationWrapper::new(
                core,
                breath_gain,
                Self::autonomous_attack_enabled(
                    phonation_engine.mode,
                    pitch_application,
                    sustain_participation,
                ),
            ),
            pitch_ctl,
            phonation_engine,
            social_coupling,
            body,
            last_signal: Default::default(),
            selection_environment,
            phonation_scratch: Default::default(),
            active_render_notes: Vec::new(),
            amp_update_pending: false,
            control_tick: None,
            life_accumulator: None,
            voice_adsr,
            phonation_gate_open,
        };
        if let Some(landscape) = landscape.filter(|l| l.harmonic.support_weights.active())
            && voice.effective_control.pitch.leave_self_out
        {
            voice.refresh_support_template(landscape);
        }
        voice
    }

    pub fn metadata(&self) -> &VoiceMetadata {
        &self.metadata
    }

    pub fn id(&self) -> u64 {
        self.id
    }

    /// Whether the `phonate_when_viable()` gate has latched open. Always
    /// `true` for the default `Immediate` gate; a one-way latch for
    /// `WhenViable` (see `update_phonation_gate`).
    pub fn phonation_gate_open(&self) -> bool {
        self.phonation_gate_open
    }

    pub fn target_pitch_log2(&self) -> f32 {
        self.pitch_ctl.target_pitch_log2()
    }

    pub fn integration_window(&self) -> f32 {
        self.pitch_ctl.integration_window()
    }

    pub fn release_gain(&self) -> f32 {
        self.release.gain
    }

    #[cfg(test)]
    pub(crate) fn set_accumulated_time_for_test(&mut self, value: f32) {
        self.pitch_ctl.set_accumulated_time_for_test(value);
    }

    #[cfg(test)]
    pub(crate) fn pitch_core_for_test(&self) -> &AnyPitchCore {
        self.pitch_ctl.core_for_test()
    }

    #[cfg(test)]
    pub(crate) fn accumulated_time_for_test(&self) -> f32 {
        self.pitch_ctl.accumulated_time_for_test()
    }

    #[cfg(test)]
    pub(crate) fn set_theta_phase_state_for_test(&mut self, last_phase: f32, initialized: bool) {
        self.pitch_ctl
            .set_theta_phase_state_for_test(last_phase, initialized);
    }
}

// Spec -> Voice construction lives here so the IR layer (`scenario`) never
// depends on the domain layer: life consumes the IR, not the reverse.
impl VoiceSpec {
    pub fn spawn(
        &self,
        assigned_id: u64,
        start_frame: u64,
        metadata: VoiceMetadata,
        fs: f32,
        seed_offset: u64,
    ) -> Voice {
        self.spawn_with_landscape(assigned_id, start_frame, metadata, fs, None, seed_offset)
    }

    pub fn spawn_with_landscape(
        &self,
        assigned_id: u64,
        start_frame: u64,
        metadata: VoiceMetadata,
        fs: f32,
        landscape: Option<&Landscape>,
        seed_offset: u64,
    ) -> Voice {
        Voice::spawn_from_control(
            self.control.clone(),
            self.articulation.clone(),
            assigned_id,
            start_frame,
            metadata,
            fs,
            landscape,
            seed_offset,
        )
    }
}

#[cfg(test)]
#[path = "voice_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "pitch_application_tests.rs"]
mod pitch_application_tests;

#[path = "shape_participation.rs"]
mod shape_participation;
#[path = "sustained_participation.rs"]
mod sustained_participation;

#[cfg(test)]
#[path = "rearticulation_boundary_tests.rs"]
mod rearticulation_boundary_tests;

#[path = "bounded_participation.rs"]
mod bounded_participation;

#[cfg(test)]
#[path = "tie_tests.rs"]
mod tie_tests;

#[cfg(test)]
#[path = "synchronized_tests.rs"]
mod synchronized_tests;
