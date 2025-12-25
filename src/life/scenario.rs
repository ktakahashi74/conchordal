use schemars::JsonSchema;
use serde::{
    Deserialize, Serialize,
    de::{self, Deserializer},
};
use std::fmt;

use crate::life::individual::{
    AgentMetadata, AnyCore, ArticulationState, DroneCore, Harmonic, HarmonicBody,
    IndividualWrapper, KuramotoCore, PinkNoise, PureTone, Sensitivity, SequencedCore, SineBody,
};
use crate::life::lifecycle::LifecycleConfig;
use rand::{Rng as _, rng};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    #[serde(alias = "episodes")]
    pub scenes: Vec<Scene>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EnvelopeConfig {
    pub attack_sec: f32,
    pub decay_sec: f32,
    pub sustain_level: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum HarmonicMode {
    Harmonic, // Integer multiples (1, 2, 3...)
    Metallic, // Non-integer ratios (e.g., k^1.4)
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
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

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(tag = "brain", rename_all = "snake_case")]
pub enum BrainConfig {
    Entrain {
        #[serde(flatten)]
        lifecycle: LifecycleConfig,
    },
    Seq {
        duration: f32,
    },
    Drone {
        #[serde(default)]
        sway: f32,
    },
}

impl Default for BrainConfig {
    fn default() -> Self {
        BrainConfig::Entrain {
            lifecycle: LifecycleConfig::Decay {
                initial_energy: 1.0,
                half_life_sec: 0.5,
                attack_sec: crate::life::lifecycle::default_decay_attack(),
            },
        }
    }
}

fn default_brain() -> BrainConfig {
    BrainConfig::default()
}

fn deserialize_brain_config<'de, D>(deserializer: D) -> Result<BrainConfig, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Debug, Clone, Deserialize)]
    #[serde(tag = "brain", rename_all = "snake_case")]
    enum Tagged {
        Entrain {
            #[serde(flatten)]
            lifecycle: LifecycleConfig,
        },
        Seq {
            duration: f32,
        },
        Drone {
            #[serde(default)]
            sway: f32,
        },
    }

    let value = Value::deserialize(deserializer)?;
    if let Ok(tagged) = Tagged::deserialize(value.clone()) {
        return Ok(match tagged {
            Tagged::Entrain { lifecycle } => BrainConfig::Entrain { lifecycle },
            Tagged::Seq { duration } => BrainConfig::Seq { duration },
            Tagged::Drone { sway } => BrainConfig::Drone { sway },
        });
    }
    if let Ok(lifecycle) = LifecycleConfig::deserialize(value.clone()) {
        return Ok(BrainConfig::Entrain { lifecycle });
    }
    Err(de::Error::custom(
        "failed to parse brain config or legacy lifecycle",
    ))
}

impl<'de> Deserialize<'de> for BrainConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserialize_brain_config(deserializer)
    }
}

impl fmt::Display for BrainConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BrainConfig::Entrain { lifecycle } => write!(f, "brain=entrain {}", lifecycle),
            BrainConfig::Seq { duration } => write!(f, "brain=seq(duration={duration:.3}s)"),
            BrainConfig::Drone { sway } => write!(f, "brain=drone(sway={sway:.3})"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene {
    #[serde(default)]
    pub name: Option<String>,
    pub start_time: f32,
    pub events: Vec<Event>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepeatConfig {
    pub count: usize,           // 繰り返し回数
    pub interval: f32,          // 実行間隔（秒）
    pub id_offset: Option<u64>, // 回ごとに base_id をずらす量（SpawnAgents用）
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub time: f32,
    pub repeat: Option<RepeatConfig>,
    pub actions: Vec<Action>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum IndividualConfig {
    #[serde(rename = "pure_tone")]
    PureTone {
        freq: f32,
        amp: f32,
        phase: Option<f32>,
        #[serde(default)]
        rhythm_freq: Option<f32>,
        #[serde(default)]
        rhythm_sensitivity: Option<f32>,
        #[serde(default)]
        commitment: Option<f32>,
        #[serde(default)]
        habituation_sensitivity: Option<f32>,
        #[serde(
            default = "default_brain",
            alias = "lifecycle",
            deserialize_with = "deserialize_brain_config"
        )]
        brain: BrainConfig,
        tag: Option<String>,
    }, // future variants
    #[serde(rename = "harmonic")]
    Harmonic {
        freq: f32,
        amp: f32,
        genotype: TimbreGenotype,
        #[serde(
            default = "default_brain",
            alias = "lifecycle",
            deserialize_with = "deserialize_brain_config"
        )]
        brain: BrainConfig, // Controls the global amplitude envelope
        tag: Option<String>,
        #[serde(default)]
        rhythm_freq: Option<f32>,
        #[serde(default)]
        rhythm_sensitivity: Option<f32>,
        #[serde(default)]
        commitment: Option<f32>,
        #[serde(default)]
        habituation_sensitivity: Option<f32>,
    },
}

impl IndividualConfig {
    pub fn id(&self) -> Option<u64> {
        match self {
            IndividualConfig::PureTone { .. } => None,
            IndividualConfig::Harmonic { .. } => None,
        }
    }

    pub fn tag(&self) -> Option<&String> {
        match self {
            IndividualConfig::PureTone { tag, .. } => tag.as_ref(),
            IndividualConfig::Harmonic { tag, .. } => tag.as_ref(),
        }
    }

    fn envelope_from_lifecycle(
        lifecycle: &LifecycleConfig,
        fs: f32,
    ) -> (
        f32,
        f32,
        f32,
        f32,
        f32,
        ArticulationState,
        Sensitivity,
        bool,
        f32,
    ) {
        match lifecycle {
            LifecycleConfig::Decay {
                initial_energy,
                half_life_sec,
                attack_sec,
            } => {
                let atk = attack_sec.max(0.0005);
                let attack_step = 1.0 / (fs * atk);
                let decay_sec = half_life_sec.max(0.01);
                let decay_factor = (-1.0f32 / (fs * decay_sec)).exp();
                let basal = 0.0;
                (
                    *initial_energy,
                    basal,
                    0.0,
                    attack_step,
                    decay_factor,
                    ArticulationState::Attack,
                    Sensitivity::default(), // fire-and-forget
                    false,                  // one-shot
                    0.02,
                )
            }
            LifecycleConfig::Sustain {
                initial_energy,
                metabolism_rate,
                recharge_rate,
                action_cost,
                envelope,
            } => {
                let atk = envelope.attack_sec.max(0.0005);
                let attack_step = 1.0 / (fs * atk);
                let decay_sec = envelope.decay_sec.max(0.01);
                let decay_factor = (-1.0f32 / (fs * decay_sec)).exp();
                (
                    *initial_energy,
                    *metabolism_rate,
                    recharge_rate.unwrap_or(0.5), // configurable recharge
                    attack_step,
                    decay_factor,
                    ArticulationState::Idle,
                    Sensitivity {
                        delta: 1.0,
                        theta: 1.0,
                        alpha: 0.5,
                        beta: 0.5,
                    },
                    true, // can retrigger rhythmically
                    action_cost.unwrap_or(0.02),
                )
            }
        }
    }

    fn core_from_brain(
        brain: &BrainConfig,
        fs: f32,
        assigned_id: u64,
        rhythm_freq: &Option<f32>,
        rhythm_sensitivity: &Option<f32>,
        noise_seed: u64,
    ) -> AnyCore {
        match brain {
            BrainConfig::Entrain { lifecycle } => {
                let (
                    energy,
                    basal_cost,
                    recharge_rate,
                    attack_step,
                    decay_factor,
                    state,
                    sensitivity,
                    retrigger,
                    action_cost,
                ) = Self::envelope_from_lifecycle(lifecycle, fs);
                let mut rng = rng();
                AnyCore::Entrain(KuramotoCore {
                    energy,
                    basal_cost,
                    action_cost,
                    recharge_rate,
                    sensitivity: Sensitivity {
                        beta: rhythm_sensitivity.unwrap_or(sensitivity.beta),
                        ..sensitivity
                    },
                    rhythm_phase: 0.0,
                    rhythm_freq: rhythm_freq.unwrap_or_else(|| rng.random_range(0.5..3.0)),
                    env_level: 0.0,
                    state,
                    attack_step,
                    decay_factor,
                    retrigger,
                    noise_1f: PinkNoise::new(noise_seed, 0.001),
                    confidence: 1.0,
                    gate_threshold: 0.02,
                })
            }
            BrainConfig::Seq { duration } => AnyCore::Seq(SequencedCore {
                timer: 0.0,
                duration: duration.max(0.0),
                env_level: 0.0,
            }),
            BrainConfig::Drone { sway } => {
                let mut rng = rng();
                let sway_rate = if *sway <= 0.0 { 0.05 } else { *sway };
                AnyCore::Drone(DroneCore {
                    phase: rng.random_range(0.0..std::f32::consts::TAU),
                    sway_rate,
                })
            }
        }
    }

    pub fn spawn(
        &self,
        assigned_id: u64,
        start_frame: u64,
        mut metadata: AgentMetadata,
    ) -> IndividualWrapper {
        metadata.id = assigned_id;
        if metadata.tag.is_none() {
            metadata.tag = self.tag().cloned();
        }
        match self {
            IndividualConfig::PureTone {
                freq,
                amp,
                phase,
                rhythm_freq,
                rhythm_sensitivity,
                commitment,
                habituation_sensitivity,
                brain,
                ..
            } => {
                let fs = 48_000.0f32;
                let target_freq = freq.max(1.0);
                let target_pitch_log2 = target_freq.log2();
                let integration_window = 0.05 + 6.0 / target_freq;
                let commitment = commitment.unwrap_or(0.5).clamp(0.0, 1.0);
                let habituation = habituation_sensitivity.unwrap_or(1.0).max(0.0);

                IndividualWrapper::PureTone(PureTone {
                    id: assigned_id,
                    metadata,
                    core: Self::core_from_brain(
                        brain,
                        fs,
                        assigned_id,
                        rhythm_freq,
                        rhythm_sensitivity,
                        assigned_id,
                    ),
                    body: SineBody {
                        freq_hz: *freq,
                        amp: *amp,
                        audio_phase: phase.unwrap_or(0.0),
                    },
                    last_signal: Default::default(),
                    release_gain: 1.0,
                    release_sec: 0.03,
                    release_pending: false,
                    target_pitch_log2,
                    tessitura_center: target_pitch_log2,
                    tessitura_gravity: 0.1,
                    integration_window,
                    accumulated_time: 0.0,
                    breath_gain: 1.0,
                    commitment,
                    habituation_sensitivity: habituation,
                    last_theta_sample: 0.0,
                })
            }
            IndividualConfig::Harmonic {
                freq,
                amp,
                genotype,
                rhythm_freq,
                rhythm_sensitivity,
                commitment,
                habituation_sensitivity,
                brain,
                ..
            } => {
                let fs = 48_000.0f32;
                let mut rng = rng();
                let partials = 16;
                let mut phases = Vec::with_capacity(partials);
                let mut detune_phases = Vec::with_capacity(partials);
                for _ in 0..partials {
                    phases.push(rng.random_range(0.0..std::f32::consts::TAU));
                    detune_phases.push(rng.random_range(0.0..std::f32::consts::TAU));
                }
                let target_freq = freq.max(1.0);
                let target_pitch_log2 = target_freq.log2();
                let integration_window = 0.05 + 6.0 / target_freq;
                let commitment = commitment.unwrap_or(0.5).clamp(0.0, 1.0);
                let habituation = habituation_sensitivity.unwrap_or(1.0).max(0.0);
                IndividualWrapper::Harmonic(Harmonic {
                    id: assigned_id,
                    metadata,
                    core: Self::core_from_brain(
                        brain,
                        fs,
                        assigned_id,
                        rhythm_freq,
                        rhythm_sensitivity,
                        assigned_id.wrapping_add(start_frame),
                    ),
                    body: HarmonicBody {
                        base_freq_hz: *freq,
                        amp: *amp,
                        genotype: genotype.clone(),
                        lfo_phase: 0.0,
                        phases,
                        detune_phases,
                        jitter_gen: PinkNoise::new(
                            assigned_id.wrapping_add(start_frame ^ 0xdead_beef),
                            0.001,
                        ),
                    },
                    last_signal: Default::default(),
                    release_gain: 1.0,
                    release_sec: 0.03,
                    release_pending: false,
                    target_pitch_log2,
                    tessitura_center: target_pitch_log2,
                    tessitura_gravity: 0.1,
                    integration_window,
                    accumulated_time: 0.0,
                    breath_gain: 1.0,
                    commitment,
                    habituation_sensitivity: habituation,
                    last_theta_sample: 0.0,
                })
            }
        }
    }
}

impl fmt::Display for IndividualConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IndividualConfig::PureTone {
                freq,
                amp,
                tag,
                brain,
                ..
            } => {
                let tag_str = tag.as_deref().unwrap_or("-");
                write!(
                    f,
                    "PureTone(tag={}, freq={:.1} Hz, amp={:.3}, {})",
                    tag_str, freq, amp, brain
                )
            }
            IndividualConfig::Harmonic {
                freq,
                amp,
                tag,
                brain,
                genotype,
                ..
            } => {
                let tag_str = tag.as_deref().unwrap_or("-");
                write!(
                    f,
                    "Harmonic(tag={}, mode={:?}, freq={:.1} Hz, amp={:.3}, {})",
                    tag_str, genotype.mode, freq, amp, brain
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spawn_carries_envelope_and_params() {
        let cfg = IndividualConfig::PureTone {
            freq: 220.0,
            amp: 0.3,
            phase: Some(0.25),
            rhythm_freq: None,
            rhythm_sensitivity: None,
            commitment: None,
            habituation_sensitivity: None,
            brain: BrainConfig::Entrain {
                lifecycle: LifecycleConfig::Decay {
                    initial_energy: 1.0,
                    half_life_sec: 0.5,
                    attack_sec: crate::life::lifecycle::default_decay_attack(),
                },
            },
            tag: Some("test".into()),
        };

        let agent = cfg.spawn(
            7,
            5,
            AgentMetadata {
                id: 7,
                tag: Some("test".into()),
                group_idx: 0,
                member_idx: 0,
            },
        );
        match agent {
            IndividualWrapper::PureTone(ind) => {
                assert_eq!(ind.id, 7);
                assert_eq!(ind.body.freq_hz, 220.0);
                assert_eq!(ind.body.amp, 0.3);
            }
            IndividualWrapper::Harmonic(_) => panic!("expected pure tone"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Action {
    AddAgent {
        agent: IndividualConfig,
    },
    SpawnAgents {
        method: SpawnMethod,
        count: usize,
        amp: f32,
        #[serde(
            default = "default_brain",
            alias = "lifecycle",
            deserialize_with = "deserialize_brain_config"
        )]
        brain: BrainConfig,
        tag: Option<String>,
    },
    RemoveAgent {
        target: String,
    },
    ReleaseAgent {
        target: String,
        release_sec: f32,
    },
    SetFreq {
        target: String,
        freq_hz: f32,
    },
    SetAmp {
        target: String,
        amp: f32,
    },
    SetRhythmVitality {
        value: f32,
    },
    SetGlobalCoupling {
        value: f32,
    },
    SetRoughnessTolerance {
        value: f32,
    },
    SetHarmonicity {
        mirror: Option<f32>,
        limit: Option<u32>,
    },
    SetCommitment {
        target: String,
        value: f32,
    },
    SetDrift {
        target: String,
        value: f32,
    },
    SetHabituationSensitivity {
        target: String,
        value: f32,
    },
    SetHabituationParams {
        weight: f32,
        tau: f32,
        max_depth: f32,
    },
    Finish,
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Action::AddAgent { agent } => write!(f, "AddAgent {}", agent),
            Action::SpawnAgents {
                method,
                count,
                amp,
                brain,
                tag,
            } => {
                let tag_str = tag.as_deref().unwrap_or("-");
                write!(
                    f,
                    "SpawnAgents tag={} count={} amp={:.3} {} {}",
                    tag_str, count, amp, method, brain
                )
            }
            Action::RemoveAgent { target } => write!(f, "RemoveAgent target={}", target),
            Action::ReleaseAgent {
                target,
                release_sec,
            } => write!(f, "ReleaseAgent target={} sec={:.3}", target, release_sec),
            Action::SetFreq { target, freq_hz } => {
                write!(f, "SetFreq target={} freq={:.2} Hz", target, freq_hz)
            }
            Action::SetAmp { target, amp } => {
                write!(f, "SetAmp target={} amp={:.3}", target, amp)
            }
            Action::SetRhythmVitality { value } => {
                write!(f, "SetRhythmVitality value={:.3}", value)
            }
            Action::SetGlobalCoupling { value } => {
                write!(f, "SetGlobalCoupling value={:.3}", value)
            }
            Action::SetRoughnessTolerance { value } => {
                write!(f, "SetRoughnessTolerance value={:.3}", value)
            }
            Action::SetHarmonicity { mirror, limit } => {
                let m = mirror
                    .map(|v| format!("{:.3}", v))
                    .unwrap_or_else(|| "-".into());
                let l = limit.map(|v| v.to_string()).unwrap_or_else(|| "-".into());
                write!(f, "SetHarmonicity mirror={} limit={}", m, l)
            }
            Action::SetCommitment { target, value } => {
                write!(f, "SetCommitment target={} value={:.3}", target, value)
            }
            Action::SetDrift { target, value } => {
                write!(f, "SetDrift target={} value={:.3}", target, value)
            }
            Action::SetHabituationSensitivity { target, value } => {
                write!(
                    f,
                    "SetHabituationSensitivity target={} value={:.3}",
                    target, value
                )
            }
            Action::SetHabituationParams {
                weight,
                tau,
                max_depth,
            } => write!(
                f,
                "SetHabituation weight={:.3} tau={:.3} max={:.3}",
                weight, tau, max_depth
            ),
            Action::Finish => write!(f, "Finish"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum SpawnMethod {
    /// Deterministically search a frequency that maximizes H = C - R.
    Harmonicity {
        min_freq: f32,
        max_freq: f32,
        /// Minimum ERB distance between newborn fundamentals at spawn time. Default: 1.0
        min_dist_erb: Option<f32>,
    },
    /// Deterministically search a frequency that minimizes H = C - R.
    LowHarmonicity {
        min_freq: f32,
        max_freq: f32,
        /// Minimum ERB distance between newborn fundamentals at spawn time. Default: 1.0
        min_dist_erb: Option<f32>,
    },
    /// Stochastically sample using H as a density (temperature controls sharpness).
    HarmonicDensity {
        min_freq: f32,
        max_freq: f32,
        temperature: Option<f32>,
        /// Minimum ERB distance between newborn fundamentals at spawn time. Default: 1.0
        min_dist_erb: Option<f32>,
    },
    /// Search for a region where H ≈ 0.5 (midpoint of normalized consonance).
    ZeroCrossing {
        min_freq: f32,
        max_freq: f32,
        /// Minimum ERB distance between newborn fundamentals at spawn time. Default: 1.0
        min_dist_erb: Option<f32>,
    },
    /// Search for an energy gap (a spectral valley).
    SpectralGap {
        min_freq: f32,
        max_freq: f32,
        /// Minimum ERB distance between newborn fundamentals at spawn time. Default: 1.0
        min_dist_erb: Option<f32>,
    },
    /// Random log-uniform sampling between `min_freq` and `max_freq`.
    RandomLogUniform {
        min_freq: f32,
        max_freq: f32,
        /// Minimum ERB distance between newborn fundamentals at spawn time. Default: 1.0
        min_dist_erb: Option<f32>,
    },
}

impl SpawnMethod {
    pub fn freq_range_hz(&self) -> (f32, f32) {
        match self {
            SpawnMethod::Harmonicity {
                min_freq, max_freq, ..
            }
            | SpawnMethod::LowHarmonicity {
                min_freq, max_freq, ..
            }
            | SpawnMethod::HarmonicDensity {
                min_freq, max_freq, ..
            }
            | SpawnMethod::ZeroCrossing {
                min_freq, max_freq, ..
            }
            | SpawnMethod::SpectralGap {
                min_freq, max_freq, ..
            }
            | SpawnMethod::RandomLogUniform {
                min_freq, max_freq, ..
            } => (*min_freq, *max_freq),
        }
    }

    pub fn min_dist_erb_or_default(&self) -> f32 {
        let min_dist_erb = match self {
            SpawnMethod::Harmonicity { min_dist_erb, .. }
            | SpawnMethod::LowHarmonicity { min_dist_erb, .. }
            | SpawnMethod::HarmonicDensity { min_dist_erb, .. }
            | SpawnMethod::ZeroCrossing { min_dist_erb, .. }
            | SpawnMethod::SpectralGap { min_dist_erb, .. }
            | SpawnMethod::RandomLogUniform { min_dist_erb, .. } => *min_dist_erb,
        };
        min_dist_erb.unwrap_or(1.0)
    }
}

impl fmt::Display for SpawnMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpawnMethod::Harmonicity {
                min_freq,
                max_freq,
                min_dist_erb,
            } => write!(
                f,
                "method=harmonicity({:.1}-{:.1} Hz, min_dist_erb={})",
                min_freq,
                max_freq,
                min_dist_erb.unwrap_or(1.0)
            ),
            SpawnMethod::LowHarmonicity {
                min_freq,
                max_freq,
                min_dist_erb,
            } => write!(
                f,
                "method=low_harmonicity({:.1}-{:.1} Hz, min_dist_erb={})",
                min_freq,
                max_freq,
                min_dist_erb.unwrap_or(1.0)
            ),
            SpawnMethod::HarmonicDensity {
                min_freq,
                max_freq,
                temperature,
                min_dist_erb,
            } => {
                let temp = temperature.unwrap_or(1.0);
                write!(
                    f,
                    "method=harmonic_density({:.1}-{:.1} Hz, temp={}, min_dist_erb={})",
                    min_freq,
                    max_freq,
                    temp,
                    min_dist_erb.unwrap_or(1.0)
                )
            }
            SpawnMethod::ZeroCrossing {
                min_freq,
                max_freq,
                min_dist_erb,
            } => {
                write!(
                    f,
                    "method=zero_crossing({:.1}-{:.1} Hz, min_dist_erb={})",
                    min_freq,
                    max_freq,
                    min_dist_erb.unwrap_or(1.0)
                )
            }
            SpawnMethod::SpectralGap {
                min_freq,
                max_freq,
                min_dist_erb,
            } => write!(
                f,
                "method=spectral_gap({:.1}-{:.1} Hz, min_dist_erb={})",
                min_freq,
                max_freq,
                min_dist_erb.unwrap_or(1.0)
            ),
            SpawnMethod::RandomLogUniform {
                min_freq,
                max_freq,
                min_dist_erb,
            } => write!(
                f,
                "method=random_log_uniform({:.1}-{:.1} Hz, min_dist_erb={})",
                min_freq,
                max_freq,
                min_dist_erb.unwrap_or(1.0)
            ),
        }
    }
}
