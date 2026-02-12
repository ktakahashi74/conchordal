use std::fmt;

use crate::core::landscape::LandscapeUpdate;
use crate::life::control::{AgentControl, ControlUpdate};
use crate::life::individual::{AgentMetadata, Individual};
use crate::life::lifecycle::LifecycleConfig;

#[derive(Debug, Clone)]
pub struct Scenario {
    pub seed: u64,
    pub scene_markers: Vec<SceneMarker>,
    pub events: Vec<TimedEvent>,
    pub duration_sec: f32,
}

#[derive(Debug, Clone)]
pub struct SceneMarker {
    pub name: String,
    pub time: f32,
    pub order: u64,
}

#[derive(Debug, Clone)]
pub struct TimedEvent {
    pub time: f32,
    pub order: u64,
    pub actions: Vec<Action>,
}

#[derive(Debug, Clone)]
pub struct EnvelopeConfig {
    pub attack_sec: f32,
    pub decay_sec: f32,
    pub sustain_level: f32,
}

impl Default for EnvelopeConfig {
    fn default() -> Self {
        Self {
            attack_sec: 0.01,
            decay_sec: 0.1,
            sustain_level: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HarmonicMode {
    Harmonic, // Integer multiples (1, 2, 3...)
    Metallic, // Non-integer ratios (e.g., k^1.4)
}

#[derive(Debug, Clone)]
pub struct TimbreGenotype {
    pub mode: HarmonicMode,

    // --- Structure ---
    pub stiffness: f32, // Inharmonicity coefficient

    // --- Color ---
    pub brightness: f32, // Spectral slope decay
    pub comb: f32,       // Even harmonic attenuation

    // --- Physics (Time-variant) ---
    pub damping: f32, // High-frequency decay factor based on energy level

    // --- Fluctuation & Texture ---
    pub vibrato_rate: f32,
    pub vibrato_depth: f32,
    pub jitter: f32, // 1/f Pink Noise FM strength
    pub unison: f32, // Detune amount (0.0 = single)
}

impl Default for TimbreGenotype {
    fn default() -> Self {
        Self {
            mode: HarmonicMode::Harmonic,
            stiffness: 0.0,
            brightness: 0.6,
            comb: 0.0,
            damping: 0.5,
            vibrato_rate: 5.0,
            vibrato_depth: 0.0,
            jitter: 0.0,
            unison: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhonationIntervalConfig {
    None,
    Accumulator { rate: f32, refractory: u32 },
}

impl Default for PhonationIntervalConfig {
    fn default() -> Self {
        PhonationIntervalConfig::Accumulator {
            rate: 1.0,
            refractory: 1,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SubdivisionClockConfig {
    pub divisions: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct InternalPhaseClockConfig {
    pub ratio: f32,
    pub phase0: f32,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum PhonationClockConfig {
    #[default]
    ThetaGate,
    Composite {
        subdivision: Option<SubdivisionClockConfig>,
        internal_phase: Option<InternalPhaseClockConfig>,
    },
}

#[derive(Debug, Clone, PartialEq, Default)]
pub enum SubThetaModConfig {
    #[default]
    None,
    Cosine {
        n: u32,
        depth: f32,
        phase0: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PhonationConnectConfig {
    FixedGate {
        length_gates: u32,
    },
    Field {
        hold_min_theta: f32,
        hold_max_theta: f32,
        curve_k: f32,
        curve_x0: f32,
        drop_gain: f32,
    },
}

impl Default for PhonationConnectConfig {
    fn default() -> Self {
        PhonationConnectConfig::FixedGate { length_gates: 8 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SocialConfig {
    pub coupling: f32,
    pub bin_ticks: u32,
    pub smooth: f32,
}

impl Default for SocialConfig {
    fn default() -> Self {
        Self {
            coupling: 0.0,
            bin_ticks: 0,
            smooth: 0.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PhonationMode {
    #[default]
    Gated,
    Hold,
}

#[derive(Debug, Clone, Default)]
pub struct PhonationConfig {
    pub mode: PhonationMode,
    pub interval: PhonationIntervalConfig,
    pub connect: PhonationConnectConfig,
    pub clock: PhonationClockConfig,
    pub sub_theta_mod: SubThetaModConfig,
    pub social: SocialConfig,
}

#[derive(Debug, Clone)]
pub struct AgentSpec {
    pub control: AgentControl,
    pub articulation: ArticulationCoreConfig,
}

pub type IndividualConfig = AgentSpec;
pub type SpawnSpec = AgentSpec;

#[derive(Debug, Clone)]
pub enum SoundBodyConfig {
    Sine {
        phase: Option<f32>,
    },
    Harmonic {
        genotype: TimbreGenotype,
        partials: Option<usize>,
    },
}

impl Default for SoundBodyConfig {
    fn default() -> Self {
        SoundBodyConfig::Sine { phase: None }
    }
}

#[derive(Debug, Clone)]
pub enum ArticulationCoreConfig {
    Entrain {
        lifecycle: LifecycleConfig,
        rhythm_freq: Option<f32>,
        rhythm_sensitivity: Option<f32>,
        breath_gain_init: Option<f32>,
    },
    Seq {
        duration: f32,
        breath_gain_init: Option<f32>,
    },
    Drone {
        sway: Option<f32>,
        breath_gain_init: Option<f32>,
    },
}

impl Default for ArticulationCoreConfig {
    fn default() -> Self {
        ArticulationCoreConfig::Entrain {
            lifecycle: LifecycleConfig::default(),
            rhythm_freq: None,
            rhythm_sensitivity: None,
            breath_gain_init: None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum PitchCoreConfig {
    PitchHillClimb {
        neighbor_step_cents: Option<f32>,
        tessitura_gravity: Option<f32>,
        move_cost_coeff: Option<f32>,
        move_cost_exp: Option<u8>,
        improvement_threshold: Option<f32>,
        exploration: Option<f32>,
        persistence: Option<f32>,
    },
    PitchPeakSampler {
        neighbor_step_cents: Option<f32>,
        window_cents: Option<f32>,
        top_k: Option<usize>,
        temperature: Option<f32>,
        sigma_cents: Option<f32>,
        random_candidates: Option<usize>,
        tessitura_gravity: Option<f32>,
        exploration: Option<f32>,
        persistence: Option<f32>,
    },
}

impl Default for PitchCoreConfig {
    fn default() -> Self {
        PitchCoreConfig::PitchHillClimb {
            neighbor_step_cents: None,
            tessitura_gravity: None,
            move_cost_coeff: None,
            move_cost_exp: None,
            improvement_threshold: None,
            exploration: None,
            persistence: None,
        }
    }
}

// Scenes are represented by SceneMarker and do not own events.

impl AgentSpec {
    pub fn spawn(
        &self,
        assigned_id: u64,
        start_frame: u64,
        metadata: AgentMetadata,
        fs: f32,
        seed_offset: u64,
    ) -> Individual {
        Individual::spawn_from_control(
            self.control.clone(),
            self.articulation.clone(),
            assigned_id,
            start_frame,
            metadata,
            fs,
            seed_offset,
        )
    }
}

impl fmt::Display for AgentSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Agent(control={:?})", self.control)
    }
}

#[derive(Debug, Clone)]
pub enum SpawnStrategy {
    Consonance {
        root_freq: f32,
        min_mul: f32,
        max_mul: f32,
        min_dist_erb: f32,
    },
    ConsonanceDensity {
        min_freq: f32,
        max_freq: f32,
        min_dist_erb: f32,
    },
    RandomLog {
        min_freq: f32,
        max_freq: f32,
    },
    Linear {
        start_freq: f32,
        end_freq: f32,
    },
}

impl SpawnStrategy {
    pub fn freq_range_hz(&self) -> (f32, f32) {
        match self {
            SpawnStrategy::Consonance {
                root_freq,
                min_mul,
                max_mul,
                ..
            } => (root_freq * min_mul, root_freq * max_mul),
            SpawnStrategy::ConsonanceDensity {
                min_freq, max_freq, ..
            }
            | SpawnStrategy::RandomLog { min_freq, max_freq }
            | SpawnStrategy::Linear {
                start_freq: min_freq,
                end_freq: max_freq,
            } => (*min_freq, *max_freq),
        }
    }

    pub fn min_dist_erb(&self) -> f32 {
        match self {
            SpawnStrategy::Consonance { min_dist_erb, .. }
            | SpawnStrategy::ConsonanceDensity { min_dist_erb, .. } => *min_dist_erb,
            _ => 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Action {
    Spawn {
        group_id: u64,
        ids: Vec<u64>,
        spec: AgentSpec,
        strategy: Option<SpawnStrategy>,
    },
    Update {
        group_id: u64,
        ids: Vec<u64>,
        update: ControlUpdate,
    },
    Release {
        group_id: u64,
        ids: Vec<u64>,
        fade_sec: f32,
    },
    SetHarmonicityParams {
        update: LandscapeUpdate,
    },
    SetGlobalCoupling {
        value: f32,
    },
    SetRoughnessTolerance {
        value: f32,
    },
    Finish,
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Action::Spawn { group_id, ids, .. } => {
                write!(f, "Spawn group={} count={}", group_id, ids.len())
            }
            Action::Update { group_id, .. } => write!(f, "Update group={}", group_id),
            Action::Release {
                group_id, fade_sec, ..
            } => write!(f, "Release group={} fade={:.3}", group_id, fade_sec),
            Action::SetHarmonicityParams { update } => write!(
                f,
                "SetHarmonicityParams mirror={:?} roughness_k={:?}",
                update.mirror, update.roughness_k
            ),
            Action::SetGlobalCoupling { value } => {
                write!(f, "SetGlobalCoupling value={:.3}", value)
            }
            Action::SetRoughnessTolerance { value } => {
                write!(f, "SetRoughnessTolerance value={:.3}", value)
            }
            Action::Finish => write!(f, "Finish"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::life::individual::{AgentMetadata, SoundBody};

    #[test]
    fn agent_spec_aliases_behave_identically() {
        let mut control = AgentControl::default();
        control.pitch.freq = 330.0;
        let base = AgentSpec {
            control,
            articulation: ArticulationCoreConfig::default(),
        };

        let as_individual: IndividualConfig = base.clone();
        let as_spawn: SpawnSpec = base.clone();
        assert_eq!(format!("{as_individual}"), format!("{as_spawn}"));

        let a = as_individual.spawn(10, 0, AgentMetadata::default(), 48_000.0, 99);
        let b = as_spawn.spawn(10, 0, AgentMetadata::default(), 48_000.0, 99);
        assert_eq!(a.id(), b.id());
        assert!((a.body.base_freq_hz() - b.body.base_freq_hz()).abs() <= 1e-6);
    }
}
