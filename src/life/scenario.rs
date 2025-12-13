use serde::{Deserialize, Serialize};
use std::fmt;

use crate::life::individual::{
    AgentMetadata, ArticulationState, Harmonic, Individual, PinkNoise, PureTone, Sensitivity,
};
use crate::life::lifecycle::LifecycleConfig;
use rand::{Rng as _, rng};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    pub episodes: Vec<Episode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvelopeConfig {
    pub attack_sec: f32,
    pub decay_sec: f32,
    pub sustain_level: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HarmonicMode {
    Harmonic, // Integer multiples (1, 2, 3...)
    Metallic, // Non-integer ratios (e.g., k^1.4)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
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
        lifecycle: LifecycleConfig,
        tag: Option<String>,
    }, // future variants
    #[serde(rename = "harmonic")]
    Harmonic {
        freq: f32,
        amp: f32,
        genotype: TimbreGenotype,
        lifecycle: LifecycleConfig, // Controls the global amplitude envelope
        tag: Option<String>,
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
                )
            }
            LifecycleConfig::Sustain {
                initial_energy,
                metabolism_rate,
                envelope,
            } => {
                let atk = envelope.attack_sec.max(0.0005);
                let attack_step = 1.0 / (fs * atk);
                let decay_sec = envelope.decay_sec.max(0.01);
                let decay_factor = (-1.0f32 / (fs * decay_sec)).exp();
                (
                    *initial_energy,
                    *metabolism_rate,
                    0.5, // simple default recharge
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
                )
            }
        }
    }

    pub fn spawn(
        &self,
        assigned_id: u64,
        start_frame: u64,
        mut metadata: AgentMetadata,
    ) -> Individual {
        metadata.id = assigned_id;
        if metadata.tag.is_none() {
            metadata.tag = self.tag().cloned();
        }
        match self {
            IndividualConfig::PureTone {
                freq,
                amp,
                lifecycle,
                ..
            } => {
                let fs = 48_000.0f32;
                let (
                    energy,
                    basal_cost,
                    recharge_rate,
                    attack_step,
                    decay_factor,
                    state,
                    sensitivity,
                    retrigger,
                ) = Self::envelope_from_lifecycle(lifecycle, fs);

                Individual::PureTone(PureTone {
                    id: assigned_id,
                    metadata,
                    freq_hz: *freq,
                    amp: *amp,
                    energy,
                    basal_cost,
                    action_cost: 0.02,
                    recharge_rate,
                    sensitivity,
                    rhythm_phase: 0.0,
                    rhythm_freq: rng().random_range(0.5..3.0),
                    audio_phase: 0.0,
                    env_level: 0.0,
                    state,
                    attack_step,
                    decay_factor,
                    retrigger,
                    omega: 0.0,
                    noise_1f: PinkNoise::new(assigned_id, 0.001),
                    confidence: 1.0,
                    gate_threshold: 0.02,
                })
            }
            IndividualConfig::Harmonic {
                freq,
                amp,
                genotype,
                lifecycle,
                ..
            } => {
                let fs = 48_000.0f32;
                let (
                    energy,
                    basal_cost,
                    recharge_rate,
                    attack_step,
                    decay_factor,
                    state,
                    sensitivity,
                    retrigger,
                ) = Self::envelope_from_lifecycle(lifecycle, fs);
                let mut rng = rng();
                let partials = 16;
                let mut phases = Vec::with_capacity(partials);
                let mut detune_phases = Vec::with_capacity(partials);
                for _ in 0..partials {
                    phases.push(rng.random_range(0.0..std::f32::consts::TAU));
                    detune_phases.push(rng.random_range(0.0..std::f32::consts::TAU));
                }
                Individual::Harmonic(Harmonic {
                    id: assigned_id,
                    metadata,
                    base_freq_hz: *freq,
                    amp: *amp,
                    genotype: genotype.clone(),
                    energy,
                    basal_cost,
                    action_cost: 0.02,
                    recharge_rate,
                    sensitivity,
                    rhythm_phase: 0.0,
                    rhythm_freq: rng.random_range(0.5..3.0),
                    lfo_phase: 0.0,
                    env_level: 0.0,
                    state,
                    attack_step,
                    decay_factor,
                    retrigger,
                    confidence: 1.0,
                    gate_threshold: 0.02,
                    phases,
                    detune_phases,
                    jitter_gen: PinkNoise::new(assigned_id.wrapping_add(start_frame), 0.001),
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
                lifecycle,
                ..
            } => {
                let tag_str = tag.as_deref().unwrap_or("-");
                write!(
                    f,
                    "PureTone(tag={}, freq={:.1} Hz, amp={:.3}, {})",
                    tag_str, freq, amp, lifecycle
                )
            }
            IndividualConfig::Harmonic {
                freq,
                amp,
                tag,
                lifecycle,
                genotype,
            } => {
                let tag_str = tag.as_deref().unwrap_or("-");
                write!(
                    f,
                    "Harmonic(tag={}, mode={:?}, freq={:.1} Hz, amp={:.3}, {})",
                    tag_str, genotype.mode, freq, amp, lifecycle
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
            lifecycle: LifecycleConfig::Decay {
                initial_energy: 1.0,
                half_life_sec: 0.5,
                attack_sec: crate::life::lifecycle::default_decay_attack(),
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
            Individual::PureTone(ind) => {
                assert_eq!(ind.id, 7);
                assert_eq!(ind.freq_hz, 220.0);
                assert_eq!(ind.amp, 0.3);
            }
            Individual::Harmonic(_) => panic!("expected pure tone"),
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
        lifecycle: LifecycleConfig,
        tag: Option<String>,
    },
    RemoveAgent {
        target: String,
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
                lifecycle,
                tag,
            } => {
                let tag_str = tag.as_deref().unwrap_or("-");
                write!(
                    f,
                    "SpawnAgents tag={} count={} amp={:.3} {} {}",
                    tag_str, count, amp, method, lifecycle
                )
            }
            Action::RemoveAgent { target } => write!(f, "RemoveAgent target={}", target),
            Action::SetFreq { target, freq_hz } => {
                write!(f, "SetFreq target={} freq={:.2} Hz", target, freq_hz)
            }
            Action::SetAmp { target, amp } => {
                write!(f, "SetAmp target={} amp={:.3}", target, amp)
            }
            Action::SetRhythmVitality { value } => {
                write!(f, "SetRhythmVitality value={:.3}", value)
            }
            Action::Finish => write!(f, "Finish"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum SpawnMethod {
    /// H = C - R を最大化する周波数を探索（決定論的）
    Harmonicity { min_freq: f32, max_freq: f32 },
    /// H = C - R を最小化する周波数を探索（決定論的）
    LowHarmonicity { min_freq: f32, max_freq: f32 },
    /// H の値を確率密度としてサンプリング（確率的・群生）
    HarmonicDensity {
        min_freq: f32,
        max_freq: f32,
        temperature: Option<f32>,
    },
    /// H ≈ 0 となる領域を探索
    ZeroCrossing { min_freq: f32, max_freq: f32 },
    /// エネルギーの空白域（スペクトルの谷）を探索
    SpectralGap { min_freq: f32, max_freq: f32 },
    /// 単純なランダム（オクターブ等間隔分布）
    RandomLogUniform { min_freq: f32, max_freq: f32 },
}

impl fmt::Display for SpawnMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpawnMethod::Harmonicity { min_freq, max_freq } => {
                write!(f, "method=harmonicity({:.1}-{:.1} Hz)", min_freq, max_freq)
            }
            SpawnMethod::LowHarmonicity { min_freq, max_freq } => write!(
                f,
                "method=low_harmonicity({:.1}-{:.1} Hz)",
                min_freq, max_freq
            ),
            SpawnMethod::HarmonicDensity {
                min_freq,
                max_freq,
                temperature,
            } => write!(
                f,
                "method=harmonic_density({:.1}-{:.1} Hz, temp={})",
                min_freq,
                max_freq,
                temperature.unwrap_or(1.0)
            ),
            SpawnMethod::ZeroCrossing { min_freq, max_freq } => {
                write!(
                    f,
                    "method=zero_crossing({:.1}-{:.1} Hz)",
                    min_freq, max_freq
                )
            }
            SpawnMethod::SpectralGap { min_freq, max_freq } => {
                write!(f, "method=spectral_gap({:.1}-{:.1} Hz)", min_freq, max_freq)
            }
            SpawnMethod::RandomLogUniform { min_freq, max_freq } => write!(
                f,
                "method=random_log_uniform({:.1}-{:.1} Hz)",
                min_freq, max_freq
            ),
        }
    }
}
