use schemars::JsonSchema;
use serde::{
    Deserialize, Serialize,
    de::{self, Deserializer},
};
use std::fmt;
use tracing::debug;

use crate::life::individual::{
    AgentMetadata, AnyFieldCore, AnyModulationCore, AnyTemporalCore, Individual,
};
use crate::life::lifecycle::LifecycleConfig;
use crate::life::perceptual::PerceptualConfig;
use rand::SeedableRng;
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    pub scenes: Vec<Scene>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
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

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LifeConfig {
    #[serde(default)]
    pub body: SoundBodyConfig,
    #[serde(default)]
    pub temporal: TemporalCoreConfig,
    #[serde(default)]
    pub field: FieldCoreConfig,
    #[serde(default)]
    pub modulation: ModulationCoreConfig,
    #[serde(default)]
    pub perceptual: PerceptualConfig,
    #[serde(default)]
    pub breath_gain_init: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "core", rename_all = "snake_case", deny_unknown_fields)]
pub enum SoundBodyConfig {
    Sine {
        #[serde(default)]
        phase: Option<f32>,
    },
    Harmonic {
        #[serde(default, flatten)]
        genotype: TimbreGenotype,
        #[serde(default)]
        partials: Option<usize>,
    },
}

impl Default for SoundBodyConfig {
    fn default() -> Self {
        SoundBodyConfig::Sine { phase: None }
    }
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(tag = "core", rename_all = "snake_case", deny_unknown_fields)]
pub enum TemporalCoreConfig {
    Entrain {
        #[serde(flatten)]
        lifecycle: LifecycleConfig,
        #[serde(default)]
        rhythm_freq: Option<f32>,
        #[serde(default)]
        rhythm_sensitivity: Option<f32>,
    },
    Seq {
        duration: f32,
    },
    Drone {
        #[serde(default)]
        sway: Option<f32>,
    },
}

impl Default for TemporalCoreConfig {
    fn default() -> Self {
        TemporalCoreConfig::Entrain {
            lifecycle: LifecycleConfig::default(),
            rhythm_freq: None,
            rhythm_sensitivity: None,
        }
    }
}

impl<'de> Deserialize<'de> for TemporalCoreConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let obj = value
            .as_object()
            .ok_or_else(|| de::Error::custom("temporal core must be a map"))?;
        let core = obj
            .get("core")
            .and_then(|v| v.as_str())
            .ok_or_else(|| de::Error::custom("temporal core missing `core`"))?;
        match core {
            "entrain" => {
                let rhythm_freq = obj
                    .get("rhythm_freq")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32);
                let rhythm_sensitivity = obj
                    .get("rhythm_sensitivity")
                    .and_then(|v| v.as_f64())
                    .map(|v| v as f32);
                let mut lifecycle_obj = obj.clone();
                lifecycle_obj.remove("core");
                lifecycle_obj.remove("rhythm_freq");
                lifecycle_obj.remove("rhythm_sensitivity");
                let lifecycle_value = Value::Object(lifecycle_obj);
                let lifecycle =
                    LifecycleConfig::deserialize(lifecycle_value).map_err(de::Error::custom)?;
                Ok(TemporalCoreConfig::Entrain {
                    lifecycle,
                    rhythm_freq,
                    rhythm_sensitivity,
                })
            }
            "seq" => {
                for key in obj.keys() {
                    if key != "core" && key != "duration" {
                        return Err(de::Error::unknown_field(key, &["core", "duration"]));
                    }
                }
                let duration = obj
                    .get("duration")
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| de::Error::custom("seq core requires `duration`"))?
                    as f32;
                Ok(TemporalCoreConfig::Seq { duration })
            }
            "drone" => {
                for key in obj.keys() {
                    if key != "core" && key != "sway" {
                        return Err(de::Error::unknown_field(key, &["core", "sway"]));
                    }
                }
                let sway = obj.get("sway").and_then(|v| v.as_f64()).map(|v| v as f32);
                Ok(TemporalCoreConfig::Drone { sway })
            }
            other => Err(de::Error::unknown_variant(
                other,
                &["entrain", "seq", "drone"],
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "core", rename_all = "snake_case", deny_unknown_fields)]
pub enum FieldCoreConfig {
    PitchHillClimb {
        #[serde(default)]
        neighbor_step_cents: Option<f32>,
        #[serde(default)]
        tessitura_gravity: Option<f32>,
        #[serde(default)]
        improvement_threshold: Option<f32>,
    },
}

impl Default for FieldCoreConfig {
    fn default() -> Self {
        FieldCoreConfig::PitchHillClimb {
            neighbor_step_cents: None,
            tessitura_gravity: None,
            improvement_threshold: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "core", rename_all = "snake_case", deny_unknown_fields)]
pub enum ModulationCoreConfig {
    Static {
        #[serde(default)]
        exploration: Option<f32>,
        #[serde(default)]
        persistence: Option<f32>,
    },
}

impl Default for ModulationCoreConfig {
    fn default() -> Self {
        ModulationCoreConfig::Static {
            exploration: None,
            persistence: None,
        }
    }
}

impl fmt::Display for LifeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "life[body={:?}, temporal={:?}, field={:?}, modulation={:?}, perceptual={:?}, breath_gain_init={:?}]",
            self.body,
            self.temporal,
            self.field,
            self.modulation,
            self.perceptual,
            self.breath_gain_init
        )
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
#[serde(deny_unknown_fields)]
pub struct IndividualConfig {
    pub freq: f32,
    pub amp: f32,
    pub life: LifeConfig,
    pub tag: Option<String>,
}

impl IndividualConfig {
    pub fn id(&self) -> Option<u64> {
        None
    }

    pub fn tag(&self) -> Option<&String> {
        self.tag.as_ref()
    }

    pub fn spawn(
        &self,
        assigned_id: u64,
        start_frame: u64,
        mut metadata: AgentMetadata,
        fs: f32,
    ) -> Individual {
        metadata.id = assigned_id;
        if metadata.tag.is_none() {
            metadata.tag = self.tag().cloned();
        }
        let target_freq = self.freq.max(1.0);
        let target_pitch_log2 = target_freq.log2();
        let integration_window = 2.0 + 10.0 / target_freq;
        let seed = assigned_id ^ start_frame.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let temporal = AnyTemporalCore::from_config(&self.life.temporal, fs, assigned_id, &mut rng);
        let field = AnyFieldCore::from_config(&self.life.field, target_pitch_log2, &mut rng);
        let modulation = AnyModulationCore::from_config(&self.life.modulation);
        let perceptual =
            crate::life::perceptual::PerceptualContext::from_config(&self.life.perceptual, 0);
        let body = crate::life::individual::AnySoundBody::from_config(
            &self.life.body,
            self.freq,
            self.amp,
            &mut rng,
        );
        let (temporal_core, lifecycle_label, default_by_temporal) = match &self.life.temporal {
            TemporalCoreConfig::Entrain { lifecycle, .. } => {
                let life_label = match lifecycle {
                    LifecycleConfig::Decay { .. } => "decay",
                    LifecycleConfig::Sustain { .. } => "sustain",
                };
                ("entrain", life_label, 1.0)
            }
            TemporalCoreConfig::Seq { .. } => ("seq", "none", 1.0),
            TemporalCoreConfig::Drone { .. } => ("drone", "none", 0.0),
        };
        let breath_gain = self
            .life
            .breath_gain_init
            .unwrap_or(default_by_temporal)
            .clamp(0.0, 1.0);
        debug!(
            target: "rhythm::spawn",
            id = assigned_id,
            tag = ?metadata.tag,
            temporal = temporal_core,
            lifecycle = lifecycle_label,
            breath_gain_init = self.life.breath_gain_init,
            breath_gain
        );
        Individual {
            id: assigned_id,
            metadata,
            temporal,
            field,
            modulation,
            perceptual,
            body,
            last_signal: Default::default(),
            release_gain: 1.0,
            release_sec: 0.03,
            release_pending: false,
            target_pitch_log2,
            integration_window,
            accumulated_time: 0.0,
            breath_gain,
            last_theta_sample: 0.0,
            last_target_salience: 0.0,
            rng,
        }
    }
}

impl fmt::Display for IndividualConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tag_str = self.tag.as_deref().unwrap_or("-");
        write!(
            f,
            "Agent(tag={}, freq={:.1} Hz, amp={:.3}, {})",
            tag_str, self.freq, self.amp, self.life
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::life::individual::SoundBody;

    #[test]
    fn spawn_carries_life_config() {
        let cfg = IndividualConfig {
            freq: 220.0,
            amp: 0.3,
            life: LifeConfig {
                body: SoundBodyConfig::Sine { phase: Some(0.25) },
                temporal: TemporalCoreConfig::Entrain {
                    lifecycle: LifecycleConfig::Decay {
                        initial_energy: 1.0,
                        half_life_sec: 0.5,
                        attack_sec: crate::life::lifecycle::default_decay_attack(),
                    },
                    rhythm_freq: None,
                    rhythm_sensitivity: None,
                },
                field: FieldCoreConfig::PitchHillClimb {
                    neighbor_step_cents: None,
                    tessitura_gravity: None,
                    improvement_threshold: None,
                },
                modulation: ModulationCoreConfig::Static {
                    exploration: None,
                    persistence: None,
                },
                perceptual: PerceptualConfig {
                    tau_fast: None,
                    tau_slow: None,
                    w_boredom: None,
                    w_familiarity: None,
                    rho_self: None,
                    boredom_gamma: None,
                    self_smoothing_radius: None,
                    silence_mass_epsilon: None,
                },
                breath_gain_init: None,
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
            48_000.0,
        );
        assert_eq!(agent.id, 7);
        assert_eq!(agent.body.base_freq_hz(), 220.0);
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
        life: LifeConfig,
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
                life,
                tag,
            } => {
                let tag_str = tag.as_deref().unwrap_or("-");
                write!(
                    f,
                    "SpawnAgents tag={} count={} amp={:.3} {} {}",
                    tag_str, count, amp, method, life
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

impl Default for SpawnMethod {
    fn default() -> Self {
        SpawnMethod::RandomLogUniform {
            min_freq: 110.0,
            max_freq: 440.0,
            min_dist_erb: Some(0.0),
        }
    }
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
