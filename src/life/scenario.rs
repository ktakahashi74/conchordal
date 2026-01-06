use schemars::JsonSchema;
use serde::{
    Deserialize, Serialize,
    de::{self, Deserializer},
    ser::{SerializeMap, Serializer},
};
use std::collections::VecDeque;
use std::fmt;
use tracing::debug;

use crate::life::individual::{
    AgentMetadata, AnyArticulationCore, AnyPitchCore, ArticulationWrapper, Individual,
};
use crate::life::lifecycle::LifecycleConfig;
use crate::life::perceptual::PerceptualConfig;
use crate::life::phonation_engine::PhonationEngine;
use rand::SeedableRng;
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    pub seed: u64,
    pub scene_markers: Vec<SceneMarker>,
    pub events: Vec<TimedEvent>,
    pub duration_sec: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneMarker {
    pub name: String,
    pub time: f32,
    pub order: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimedEvent {
    pub time: f32,
    pub order: u64,
    pub actions: Vec<Action>,
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

fn default_gate_dur_scale() -> f32 {
    0.90
}

fn default_gate_dur_min_sec() -> f32 {
    0.010
}

fn default_gate_dur_max_sec() -> f32 {
    10_000.0
}

fn default_plan_pitch_mode() -> PlanPitchMode {
    PlanPitchMode::Off
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum OnBirthPhonation {
    Off,
    #[default]
    Once,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum BirthTiming {
    #[default]
    Gate,
    Immediate,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, JsonSchema, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PhonationIntervalConfig {
    #[default]
    None,
    Accumulator {
        rate: f32,
        #[serde(default)]
        refractory: u32,
    },
}

impl<'de> Deserialize<'de> for PhonationIntervalConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(s) => match s.as_str() {
                "none" => Ok(PhonationIntervalConfig::None),
                _ => Err(de::Error::custom(format!(
                    "phonation interval must be \"none\" or a map (got \"{s}\")"
                ))),
            },
            Value::Object(map) => {
                let ty = map
                    .get("type")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| de::Error::custom("phonation interval missing `type`"))?;
                match ty {
                    "none" => Ok(PhonationIntervalConfig::None),
                    "accumulator" => {
                        let rate = map.get("rate").and_then(|v| v.as_f64()).ok_or_else(|| {
                            de::Error::custom("phonation interval accumulator requires `rate`")
                        })? as f32;
                        let refractory =
                            map.get("refractory").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
                        Ok(PhonationIntervalConfig::Accumulator { rate, refractory })
                    }
                    _ => Err(de::Error::custom(format!(
                        "phonation interval type must be \"none\" or \"accumulator\" (got \"{ty}\")"
                    ))),
                }
            }
            _ => Err(de::Error::custom(
                "phonation interval must be a string or map",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SubdivisionClockConfig {
    pub divisions: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct InternalPhaseClockConfig {
    pub ratio: f32,
    #[serde(default)]
    pub phase0: f32,
}

#[derive(Debug, Clone, PartialEq, JsonSchema, Default)]
pub enum PhonationClockConfig {
    #[default]
    ThetaGate,
    Composite {
        subdivision: Option<SubdivisionClockConfig>,
        internal_phase: Option<InternalPhaseClockConfig>,
    },
}

impl Serialize for PhonationClockConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            PhonationClockConfig::ThetaGate => serializer.serialize_str("theta_gate"),
            PhonationClockConfig::Composite {
                subdivision,
                internal_phase,
            } => {
                let mut map = serializer.serialize_map(None)?;
                map.serialize_entry("type", "composite")?;
                if let Some(config) = subdivision {
                    map.serialize_entry("subdivision", config)?;
                }
                if let Some(config) = internal_phase {
                    map.serialize_entry("internal_phase", config)?;
                }
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for PhonationClockConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(s) => {
                let lowered = s.to_ascii_lowercase();
                match lowered.as_str() {
                    "theta_gate" | "thetagate" => Ok(Self::ThetaGate),
                    _ => Err(de::Error::custom(format!(
                        "phonation clock must be \"theta_gate\" (got \"{s}\")"
                    ))),
                }
            }
            Value::Object(map) => {
                let ty = map
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("composite");
                match ty {
                    "composite" => {
                        let mut subdivision = None;
                        let mut internal_phase = None;
                        for (k, v) in map {
                            match k.as_str() {
                                "type" => {}
                                "subdivision" => {
                                    subdivision = Some(
                                        SubdivisionClockConfig::deserialize(v)
                                            .map_err(de::Error::custom)?,
                                    );
                                }
                                "internal_phase" => {
                                    internal_phase = Some(
                                        InternalPhaseClockConfig::deserialize(v)
                                            .map_err(de::Error::custom)?,
                                    );
                                }
                                _ => {
                                    return Err(de::Error::custom(format!(
                                        "phonation clock has unknown key: {k}"
                                    )));
                                }
                            }
                        }
                        Ok(Self::Composite {
                            subdivision,
                            internal_phase,
                        })
                    }
                    _ => Err(de::Error::custom(format!(
                        "phonation clock type must be \"composite\" (got \"{ty}\")"
                    ))),
                }
            }
            _ => Err(de::Error::custom("phonation clock must be a string or map")),
        }
    }
}

#[derive(Debug, Clone, PartialEq, JsonSchema, Default)]
pub enum SubThetaModConfig {
    #[default]
    None,
    Cosine {
        n: u32,
        depth: f32,
        phase0: f32,
    },
}

impl Serialize for SubThetaModConfig {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            SubThetaModConfig::None => serializer.serialize_str("none"),
            SubThetaModConfig::Cosine { n, depth, phase0 } => {
                let mut map = serializer.serialize_map(None)?;
                map.serialize_entry("type", "cosine")?;
                map.serialize_entry("n", n)?;
                map.serialize_entry("depth", depth)?;
                map.serialize_entry("phase0", phase0)?;
                map.end()
            }
        }
    }
}

impl<'de> Deserialize<'de> for SubThetaModConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(s) => {
                let lowered = s.to_ascii_lowercase();
                match lowered.as_str() {
                    "none" => Ok(Self::None),
                    _ => Err(de::Error::custom(format!(
                        "sub_theta_mod must be \"none\" (got \"{s}\")"
                    ))),
                }
            }
            Value::Object(map) => {
                let ty = map.get("type").and_then(|v| v.as_str()).unwrap_or("cosine");
                match ty {
                    "cosine" => {
                        let n = map.get("n").and_then(|v| v.as_u64()).unwrap_or(1) as u32;
                        let depth = map.get("depth").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                        let phase0 =
                            map.get("phase0").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
                        Ok(Self::Cosine { n, depth, phase0 })
                    }
                    _ => Err(de::Error::custom(format!(
                        "sub_theta_mod type must be \"cosine\" (got \"{ty}\")"
                    ))),
                }
            }
            _ => Err(de::Error::custom("sub_theta_mod must be a string or map")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PhonationConnectConfig {
    FixedGate { length_gates: u32 },
}

impl Default for PhonationConnectConfig {
    fn default() -> Self {
        PhonationConnectConfig::FixedGate { length_gates: 1 }
    }
}

impl<'de> Deserialize<'de> for PhonationConnectConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(s) => match s.as_str() {
                "fixed_gate" => Ok(PhonationConnectConfig::FixedGate { length_gates: 1 }),
                _ => Err(de::Error::custom(format!(
                    "phonation connect must be \"fixed_gate\" or a map (got \"{s}\")"
                ))),
            },
            Value::Object(map) => {
                let ty = map
                    .get("type")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| de::Error::custom("phonation connect missing `type`"))?;
                match ty {
                    "fixed_gate" => {
                        let length_gates = map
                            .get("length_gates")
                            .or_else(|| map.get("length"))
                            .and_then(|v| v.as_u64())
                            .unwrap_or(1) as u32;
                        Ok(PhonationConnectConfig::FixedGate { length_gates })
                    }
                    _ => Err(de::Error::custom(format!(
                        "phonation connect type must be \"fixed_gate\" (got \"{ty}\")"
                    ))),
                }
            }
            _ => Err(de::Error::custom(
                "phonation connect must be a string or map",
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, JsonSchema)]
pub struct PhonationConfig {
    pub on_birth: OnBirthPhonation,
    pub timing: BirthTiming,
    pub interval: PhonationIntervalConfig,
    pub connect: PhonationConnectConfig,
    pub clock: PhonationClockConfig,
    pub sub_theta_mod: SubThetaModConfig,
}

impl Default for PhonationConfig {
    fn default() -> Self {
        Self {
            on_birth: OnBirthPhonation::Once,
            timing: BirthTiming::Gate,
            interval: PhonationIntervalConfig::None,
            connect: PhonationConnectConfig::default(),
            clock: PhonationClockConfig::default(),
            sub_theta_mod: SubThetaModConfig::default(),
        }
    }
}

impl<'de> Deserialize<'de> for PhonationConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::String(s) => match s.as_str() {
                "once" => Ok(Self {
                    on_birth: OnBirthPhonation::Once,
                    timing: BirthTiming::Gate,
                    interval: PhonationIntervalConfig::None,
                    connect: PhonationConnectConfig::default(),
                    clock: PhonationClockConfig::default(),
                    sub_theta_mod: SubThetaModConfig::default(),
                }),
                "immediate" => Ok(Self {
                    on_birth: OnBirthPhonation::Once,
                    timing: BirthTiming::Immediate,
                    interval: PhonationIntervalConfig::None,
                    connect: PhonationConnectConfig::default(),
                    clock: PhonationClockConfig::default(),
                    sub_theta_mod: SubThetaModConfig::default(),
                }),
                "off" => Ok(Self {
                    on_birth: OnBirthPhonation::Off,
                    timing: BirthTiming::Gate,
                    interval: PhonationIntervalConfig::None,
                    connect: PhonationConnectConfig::default(),
                    clock: PhonationClockConfig::default(),
                    sub_theta_mod: SubThetaModConfig::default(),
                }),
                _ => Err(de::Error::custom(format!(
                    "phonation must be \"once\", \"immediate\", or \"off\" (got \"{s}\")"
                ))),
            },
            Value::Object(map) => {
                let mut on_birth = OnBirthPhonation::Once;
                let mut timing = BirthTiming::Gate;
                let mut interval = PhonationIntervalConfig::None;
                let mut connect = PhonationConnectConfig::default();
                let mut clock = PhonationClockConfig::default();
                let mut sub_theta_mod = SubThetaModConfig::default();
                for (k, v) in map {
                    match k.as_str() {
                        "on_birth" => {
                            let s = v.as_str().ok_or_else(|| {
                                de::Error::custom("phonation.on_birth must be a string")
                            })?;
                            on_birth = match s {
                                "once" => OnBirthPhonation::Once,
                                "off" => OnBirthPhonation::Off,
                                _ => {
                                    return Err(de::Error::custom(format!(
                                        "phonation.on_birth must be \"once\" or \"off\" (got \"{s}\")"
                                    )));
                                }
                            };
                        }
                        "timing" => {
                            let s = v.as_str().ok_or_else(|| {
                                de::Error::custom("phonation.timing must be a string")
                            })?;
                            timing = match s {
                                "gate" => BirthTiming::Gate,
                                "immediate" => BirthTiming::Immediate,
                                _ => {
                                    return Err(de::Error::custom(format!(
                                        "phonation.timing must be \"gate\" or \"immediate\" (got \"{s}\")"
                                    )));
                                }
                            };
                        }
                        "interval" | "after_birth" => {
                            interval = PhonationIntervalConfig::deserialize(v)
                                .map_err(de::Error::custom)?;
                        }
                        "connect" => {
                            connect = PhonationConnectConfig::deserialize(v)
                                .map_err(de::Error::custom)?;
                        }
                        "clock" => {
                            clock =
                                PhonationClockConfig::deserialize(v).map_err(de::Error::custom)?;
                        }
                        "sub_theta_mod" => {
                            sub_theta_mod =
                                SubThetaModConfig::deserialize(v).map_err(de::Error::custom)?;
                        }
                        _ => {
                            return Err(de::Error::custom(format!(
                                "phonation has unknown key: {k}"
                            )));
                        }
                    }
                }
                Ok(Self {
                    on_birth,
                    timing,
                    interval,
                    connect,
                    clock,
                    sub_theta_mod,
                })
            }
            _ => Err(de::Error::custom("phonation must be a string or map")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, JsonSchema, Default)]
#[serde(rename_all = "snake_case")]
pub enum PlanPitchMode {
    #[default]
    #[serde(alias = "none")]
    Off,
    PredC,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PlanningConfig {
    #[serde(default = "default_gate_dur_scale")]
    pub gate_dur_scale: f32,
    #[serde(default = "default_gate_dur_min_sec")]
    pub gate_dur_min_sec: f32,
    #[serde(default = "default_gate_dur_max_sec")]
    pub gate_dur_max_sec: f32,
    #[serde(default = "default_plan_pitch_mode")]
    pub pitch_mode: PlanPitchMode,
}

impl Default for PlanningConfig {
    fn default() -> Self {
        Self {
            gate_dur_scale: default_gate_dur_scale(),
            gate_dur_min_sec: default_gate_dur_min_sec(),
            gate_dur_max_sec: default_gate_dur_max_sec(),
            pitch_mode: default_plan_pitch_mode(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct LifeConfig {
    #[serde(default)]
    pub body: SoundBodyConfig,
    #[serde(default)]
    pub articulation: ArticulationCoreConfig,
    #[serde(default)]
    pub pitch: PitchCoreConfig,
    #[serde(default)]
    pub perceptual: PerceptualConfig,
    #[serde(default)]
    pub planning: PlanningConfig,
    #[serde(default)]
    pub phonation: PhonationConfig,
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
pub enum ArticulationCoreConfig {
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

impl Default for ArticulationCoreConfig {
    fn default() -> Self {
        ArticulationCoreConfig::Entrain {
            lifecycle: LifecycleConfig::default(),
            rhythm_freq: None,
            rhythm_sensitivity: None,
        }
    }
}

impl<'de> Deserialize<'de> for ArticulationCoreConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        let obj = value
            .as_object()
            .ok_or_else(|| de::Error::custom("articulation core must be a map"))?;
        let core = obj
            .get("core")
            .and_then(|v| v.as_str())
            .ok_or_else(|| de::Error::custom("articulation core missing `core`"))?;
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
                Ok(ArticulationCoreConfig::Entrain {
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
                Ok(ArticulationCoreConfig::Seq { duration })
            }
            "drone" => {
                for key in obj.keys() {
                    if key != "core" && key != "sway" {
                        return Err(de::Error::unknown_field(key, &["core", "sway"]));
                    }
                }
                let sway = obj.get("sway").and_then(|v| v.as_f64()).map(|v| v as f32);
                Ok(ArticulationCoreConfig::Drone { sway })
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
pub enum PitchCoreConfig {
    PitchHillClimb {
        #[serde(default)]
        neighbor_step_cents: Option<f32>,
        #[serde(default)]
        tessitura_gravity: Option<f32>,
        #[serde(default)]
        improvement_threshold: Option<f32>,
        #[serde(default)]
        exploration: Option<f32>,
        #[serde(default)]
        persistence: Option<f32>,
    },
}

impl Default for PitchCoreConfig {
    fn default() -> Self {
        PitchCoreConfig::PitchHillClimb {
            neighbor_step_cents: None,
            tessitura_gravity: None,
            improvement_threshold: None,
            exploration: None,
            persistence: None,
        }
    }
}

impl fmt::Display for LifeConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "life[body={:?}, articulation={:?}, pitch={:?}, perceptual={:?}, planning={:?}, phonation={:?}, breath_gain_init={:?}]",
            self.body,
            self.articulation,
            self.pitch,
            self.perceptual,
            self.planning,
            self.phonation,
            self.breath_gain_init
        )
    }
}

// Scenes are represented by SceneMarker and do not own events.

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
        seed_offset: u64,
    ) -> Individual {
        metadata.id = assigned_id;
        if metadata.tag.is_none() {
            metadata.tag = self.tag().cloned();
        }
        let target_freq = self.freq.max(1.0);
        let target_pitch_log2 = target_freq.log2();
        let integration_window = 2.0 + 10.0 / target_freq;
        let seed = seed_offset ^ assigned_id ^ start_frame.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let mut rng = rand::rngs::SmallRng::seed_from_u64(seed);
        let core =
            AnyArticulationCore::from_config(&self.life.articulation, fs, assigned_id, &mut rng);
        let phonation_engine = PhonationEngine::from_config(&self.life.phonation, seed);
        let pitch = AnyPitchCore::from_config(&self.life.pitch, target_pitch_log2, &mut rng);
        let perceptual =
            crate::life::perceptual::PerceptualContext::from_config(&self.life.perceptual, 0);
        let body = crate::life::individual::AnySoundBody::from_config(
            &self.life.body,
            self.freq,
            self.amp,
            &mut rng,
        );
        let (articulation_core, lifecycle_label, default_by_articulation) =
            match &self.life.articulation {
                ArticulationCoreConfig::Entrain { lifecycle, .. } => {
                    let life_label = match lifecycle {
                        LifecycleConfig::Decay { .. } => "decay",
                        LifecycleConfig::Sustain { .. } => "sustain",
                    };
                    ("entrain", life_label, 1.0)
                }
                ArticulationCoreConfig::Seq { .. } => ("seq", "none", 1.0),
                ArticulationCoreConfig::Drone { .. } => ("drone", "none", 0.0),
            };
        let breath_gain = self
            .life
            .breath_gain_init
            .unwrap_or(default_by_articulation)
            .clamp(0.0, 1.0);
        debug!(
            target: "rhythm::spawn",
            id = assigned_id,
            tag = ?metadata.tag,
            articulation = articulation_core,
            lifecycle = lifecycle_label,
            breath_gain_init = self.life.breath_gain_init,
            breath_gain
        );
        Individual {
            id: assigned_id,
            metadata,
            articulation: ArticulationWrapper::new(core, breath_gain),
            pitch,
            perceptual,
            planning: self.life.planning.clone(),
            phonation: self.life.phonation.clone(),
            phonation_engine,
            body,
            last_signal: Default::default(),
            release_gain: 1.0,
            release_sec: 0.03,
            release_pending: false,
            birth_pending: matches!(self.life.phonation.on_birth, OnBirthPhonation::Once),
            birth_fired: false,
            birth_onset_tick: None,
            target_pitch_log2,
            integration_window,
            accumulated_time: 0.0,
            last_theta_sample: 0.0,
            last_target_salience: 0.0,
            last_error_state: Default::default(),
            last_error_cents: 0.0,
            error_initialized: false,
            last_chosen_freq_hz: target_freq,
            next_intent_tick: 0,
            intent_seq: 0,
            self_confidence: 0.5,
            pred_intent_records: VecDeque::new(),
            pred_intent_records_cap: 256,
            rng,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CohortHandle {
    pub tag: String,
    pub group_id: u64,
    pub base_id: u64,
    pub count: u32,
}

impl CohortHandle {
    pub fn len(&self) -> u32 {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

#[derive(Debug, Clone)]
pub struct CohortIter {
    next: u32,
    base_id: u64,
    count: u32,
    tag: Option<String>,
}

impl Iterator for CohortIter {
    type Item = AgentHandle;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.count {
            return None;
        }
        let id = self.base_id + u64::from(self.next);
        self.next += 1;
        Some(AgentHandle {
            id,
            tag: self.tag.clone(),
        })
    }
}

impl IntoIterator for CohortHandle {
    type Item = AgentHandle;
    type IntoIter = CohortIter;

    fn into_iter(self) -> Self::IntoIter {
        CohortIter {
            next: 0,
            base_id: self.base_id,
            count: self.count,
            tag: Some(self.tag),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AgentHandle {
    pub id: u64,
    pub tag: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TagSelector {
    pub tag: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TargetRef {
    AgentId { id: u64 },
    Range { base_id: u64, count: u32 },
    Tag { tag: String },
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
    use serde_json::json;

    #[test]
    fn spawn_carries_life_config() {
        let cfg = IndividualConfig {
            freq: 220.0,
            amp: 0.3,
            life: LifeConfig {
                body: SoundBodyConfig::Sine { phase: Some(0.25) },
                articulation: ArticulationCoreConfig::Entrain {
                    lifecycle: LifecycleConfig::Decay {
                        initial_energy: 1.0,
                        half_life_sec: 0.5,
                        attack_sec: crate::life::lifecycle::default_decay_attack(),
                    },
                    rhythm_freq: None,
                    rhythm_sensitivity: None,
                },
                pitch: PitchCoreConfig::PitchHillClimb {
                    neighbor_step_cents: None,
                    tessitura_gravity: None,
                    improvement_threshold: None,
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
                ..Default::default()
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
            0,
        );
        assert_eq!(agent.id, 7);
        assert_eq!(agent.body.base_freq_hz(), 220.0);
    }

    #[test]
    fn phonation_sugar_parses() {
        let once: PhonationConfig = serde_json::from_value(json!("once")).expect("once");
        assert_eq!(once.on_birth, OnBirthPhonation::Once);
        assert_eq!(once.timing, BirthTiming::Gate);
        assert_eq!(once.interval, PhonationIntervalConfig::None);

        let immediate: PhonationConfig =
            serde_json::from_value(json!("immediate")).expect("immediate");
        assert_eq!(immediate.on_birth, OnBirthPhonation::Once);
        assert_eq!(immediate.timing, BirthTiming::Immediate);
        assert_eq!(immediate.interval, PhonationIntervalConfig::None);

        let off: PhonationConfig = serde_json::from_value(json!("off")).expect("off");
        assert_eq!(off.on_birth, OnBirthPhonation::Off);
        assert_eq!(off.interval, PhonationIntervalConfig::None);
    }

    #[test]
    fn phonation_map_parses() {
        let cfg: PhonationConfig = serde_json::from_value(json!({
            "on_birth": "once",
            "timing": "gate",
            "interval": {
                "type": "accumulator",
                "rate": 0.25,
                "refractory": 1
            },
            "connect": {
                "type": "fixed_gate",
                "length_gates": 2
            }
        }))
        .expect("map");
        assert_eq!(cfg.on_birth, OnBirthPhonation::Once);
        assert_eq!(cfg.timing, BirthTiming::Gate);
        assert_eq!(
            cfg.interval,
            PhonationIntervalConfig::Accumulator {
                rate: 0.25,
                refractory: 1
            }
        );
        assert_eq!(
            cfg.connect,
            PhonationConnectConfig::FixedGate { length_gates: 2 }
        );
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Action {
    AddAgent {
        id: u64,
        agent: IndividualConfig,
    },
    SpawnAgents {
        group_id: u64,
        base_id: u64,
        count: u32,
        method: SpawnMethod,
        amp: f32,
        life: LifeConfig,
        tag: Option<String>,
    },
    RemoveAgent {
        target: TargetRef,
    },
    ReleaseAgent {
        target: TargetRef,
        release_sec: f32,
    },
    SetFreq {
        target: TargetRef,
        freq_hz: f32,
    },
    SetAmp {
        target: TargetRef,
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
        target: TargetRef,
        value: f32,
    },
    SetDrift {
        target: TargetRef,
        value: f32,
    },
    PostIntent {
        source_id: u64,
        onset_sec: f32,
        duration_sec: f32,
        freq_hz: f32,
        amp: f32,
        tag: Option<String>,
        confidence: f32,
    },
    Finish,
}

impl fmt::Display for TargetRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TargetRef::AgentId { id } => write!(f, "id={id}"),
            TargetRef::Range { base_id, count } => {
                write!(f, "range={base_id}..{}", base_id + u64::from(*count))
            }
            TargetRef::Tag { tag } => write!(f, "tag={tag}"),
        }
    }
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Action::AddAgent { id, agent } => write!(f, "AddAgent id={id} {agent}"),
            Action::SpawnAgents {
                method,
                count,
                amp,
                life,
                tag,
                group_id,
                base_id,
            } => {
                let tag_str = tag.as_deref().unwrap_or("-");
                write!(
                    f,
                    "SpawnAgents tag={} group_id={} base_id={} count={} amp={:.3} {} {}",
                    tag_str, group_id, base_id, count, amp, method, life
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
            Action::PostIntent {
                source_id,
                onset_sec,
                duration_sec,
                freq_hz,
                amp,
                tag,
                confidence,
            } => write!(
                f,
                "PostIntent src={} onset={:.3} dur={:.3} freq={:.1} amp={:.3} tag={:?} conf={:.2}",
                source_id, onset_sec, duration_sec, freq_hz, amp, tag, confidence
            ),
            Action::Finish => write!(f, "Finish"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
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
