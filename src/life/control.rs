use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

const AMP_MIN: f32 = 0.0;
const AMP_MAX: f32 = 1.0;
const TIMBRE_MIN: f32 = 0.0;
const TIMBRE_MAX: f32 = 1.0;
const RANGE_OCT_MIN: f32 = 0.0;
const RANGE_OCT_MAX: f32 = 6.0;
const FREQ_MIN_HZ: f32 = 1.0;
const FREQ_MAX_HZ: f32 = 20_000.0;

fn de_f32<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: Deserializer<'de>,
{
    let v = f64::deserialize(deserializer)?;
    Ok(v as f32)
}

fn de_f32_clamp_amp<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: Deserializer<'de>,
{
    let v = de_f32(deserializer)?;
    Ok(v.clamp(AMP_MIN, AMP_MAX))
}

fn de_f32_clamp_timbre<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: Deserializer<'de>,
{
    let v = de_f32(deserializer)?;
    Ok(v.clamp(TIMBRE_MIN, TIMBRE_MAX))
}

fn de_f32_clamp_range_oct<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: Deserializer<'de>,
{
    let v = de_f32(deserializer)?;
    Ok(v.clamp(RANGE_OCT_MIN, RANGE_OCT_MAX))
}

fn de_f32_clamp_unit<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: Deserializer<'de>,
{
    let v = de_f32(deserializer)?;
    Ok(v.clamp(0.0, 1.0))
}

fn de_f32_clamp_freq<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: Deserializer<'de>,
{
    let v = de_f32(deserializer)?;
    Ok(v.clamp(FREQ_MIN_HZ, FREQ_MAX_HZ))
}

fn de_opt_f32_clamp_amp<'de, D>(deserializer: D) -> Result<Option<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    let v = Option::<f64>::deserialize(deserializer)?;
    Ok(v.map(|v| (v as f32).clamp(AMP_MIN, AMP_MAX)))
}

fn de_opt_f32_clamp_timbre<'de, D>(deserializer: D) -> Result<Option<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    let v = Option::<f64>::deserialize(deserializer)?;
    Ok(v.map(|v| (v as f32).clamp(TIMBRE_MIN, TIMBRE_MAX)))
}

fn de_opt_f32_clamp_range_oct<'de, D>(deserializer: D) -> Result<Option<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    let v = Option::<f64>::deserialize(deserializer)?;
    Ok(v.map(|v| (v as f32).clamp(RANGE_OCT_MIN, RANGE_OCT_MAX)))
}

fn de_opt_f32_clamp_unit<'de, D>(deserializer: D) -> Result<Option<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    let v = Option::<f64>::deserialize(deserializer)?;
    Ok(v.map(|v| (v as f32).clamp(0.0, 1.0)))
}

fn de_opt_f32_clamp_freq<'de, D>(deserializer: D) -> Result<Option<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    let v = Option::<f64>::deserialize(deserializer)?;
    Ok(v.map(|v| (v as f32).clamp(FREQ_MIN_HZ, FREQ_MAX_HZ)))
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct WorldControl {}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct AgentControl {
    #[serde(default)]
    pub body: BodyControl,
    #[serde(default)]
    pub pitch: PitchControl,
    #[serde(default)]
    pub phonation: PhonationControl,
    #[serde(default)]
    pub perceptual: PerceptualControl,
}

impl AgentControl {
    pub fn validate(&self) -> Result<(), String> {
        let constraint = &self.pitch.constraint;
        if !matches!(constraint.mode, PitchConstraintMode::Free) && constraint.freq_hz.is_none() {
            return Err("pitch.constraint.freq_hz is required when mode != free".to_string());
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<Value, String> {
        serde_json::to_value(self).map_err(|e| format!("serialize AgentControl: {e}"))
    }

    pub fn from_json(value: Value) -> Result<Self, String> {
        let control: AgentControl =
            serde_json::from_value(value).map_err(|e| format!("parse AgentControl: {e}"))?;
        control.validate()?;
        Ok(control)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum BodyMethod {
    Sine,
    Harmonic,
}

impl Default for BodyMethod {
    fn default() -> Self {
        BodyMethod::Sine
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TimbreControl {
    #[serde(default, deserialize_with = "de_f32_clamp_timbre")]
    pub brightness: f32,
    #[serde(default, deserialize_with = "de_f32_clamp_timbre")]
    pub inharmonic: f32,
    #[serde(default, deserialize_with = "de_f32_clamp_timbre")]
    pub width: f32,
    #[serde(default, deserialize_with = "de_f32_clamp_timbre")]
    pub motion: f32,
}

impl Default for TimbreControl {
    fn default() -> Self {
        Self {
            brightness: 0.6,
            inharmonic: 0.0,
            width: 0.0,
            motion: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BodyControl {
    #[serde(default)]
    pub method: BodyMethod,
    #[serde(default, deserialize_with = "de_f32_clamp_amp")]
    pub amp: f32,
    #[serde(default)]
    pub timbre: TimbreControl,
}

impl Default for BodyControl {
    fn default() -> Self {
        Self {
            method: BodyMethod::default(),
            amp: 0.18,
            timbre: TimbreControl::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PitchConstraintMode {
    Free,
    Attractor,
    Lock,
}

impl Default for PitchConstraintMode {
    fn default() -> Self {
        PitchConstraintMode::Free
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PitchConstraint {
    #[serde(default)]
    pub mode: PitchConstraintMode,
    #[serde(default, deserialize_with = "de_opt_f32_clamp_freq")]
    pub freq_hz: Option<f32>,
    #[serde(default, deserialize_with = "de_f32_clamp_unit")]
    pub strength: f32,
}

impl Default for PitchConstraint {
    fn default() -> Self {
        Self {
            mode: PitchConstraintMode::Free,
            freq_hz: None,
            strength: 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PitchControl {
    #[serde(default, deserialize_with = "de_f32_clamp_freq")]
    pub center_hz: f32,
    #[serde(default, deserialize_with = "de_f32_clamp_range_oct")]
    pub range_oct: f32,
    #[serde(default, deserialize_with = "de_f32_clamp_unit")]
    pub gravity: f32,
    #[serde(default, deserialize_with = "de_f32_clamp_unit")]
    pub exploration: f32,
    #[serde(default, deserialize_with = "de_f32_clamp_unit")]
    pub persistence: f32,
    #[serde(default)]
    pub constraint: PitchConstraint,
}

impl Default for PitchControl {
    fn default() -> Self {
        Self {
            center_hz: 220.0,
            range_oct: RANGE_OCT_MAX,
            gravity: 0.5,
            exploration: 0.0,
            persistence: 0.5,
            constraint: PitchConstraint::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PhonationType {
    Interval,
    Clock,
    Drone,
    Field,
    None,
}

impl Default for PhonationType {
    fn default() -> Self {
        PhonationType::Interval
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PhonationControl {
    #[serde(default, rename = "type")]
    pub r#type: PhonationType,
    #[serde(default, deserialize_with = "de_f32_clamp_unit")]
    pub density: f32,
    #[serde(default, deserialize_with = "de_f32_clamp_unit")]
    pub sync: f32,
    #[serde(default, deserialize_with = "de_f32_clamp_unit")]
    pub legato: f32,
    #[serde(default, deserialize_with = "de_f32_clamp_unit")]
    pub sociality: f32,
}

impl Default for PhonationControl {
    fn default() -> Self {
        Self {
            r#type: PhonationType::default(),
            density: 0.5,
            sync: 0.5,
            legato: 0.5,
            sociality: 0.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PerceptualControl {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default, deserialize_with = "de_f32_clamp_unit")]
    pub adaptation: f32,
    #[serde(default, deserialize_with = "de_f32_clamp_unit")]
    pub novelty_bias: f32,
    #[serde(default, deserialize_with = "de_f32_clamp_unit")]
    pub self_focus: f32,
}

impl Default for PerceptualControl {
    fn default() -> Self {
        Self {
            enabled: true,
            adaptation: 0.5,
            novelty_bias: 1.0,
            self_focus: 0.15,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct AgentPatch {
    #[serde(default)]
    pub body: Option<BodyPatch>,
    #[serde(default)]
    pub pitch: Option<PitchPatch>,
    #[serde(default)]
    pub phonation: Option<PhonationPatch>,
    #[serde(default)]
    pub perceptual: Option<PerceptualPatch>,
}

impl AgentPatch {
    pub fn contains_type_switch(&self) -> bool {
        self.body
            .as_ref()
            .and_then(|body| body.method.as_ref())
            .is_some()
            || self
                .phonation
                .as_ref()
                .and_then(|phonation| phonation.r#type.as_ref())
                .is_some()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct BodyPatch {
    #[serde(default)]
    pub method: Option<BodyMethod>,
    #[serde(default, deserialize_with = "de_opt_f32_clamp_amp")]
    pub amp: Option<f32>,
    #[serde(default)]
    pub timbre: Option<TimbrePatch>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct TimbrePatch {
    #[serde(default, deserialize_with = "de_opt_f32_clamp_timbre")]
    pub brightness: Option<f32>,
    #[serde(default, deserialize_with = "de_opt_f32_clamp_timbre")]
    pub inharmonic: Option<f32>,
    #[serde(default, deserialize_with = "de_opt_f32_clamp_timbre")]
    pub width: Option<f32>,
    #[serde(default, deserialize_with = "de_opt_f32_clamp_timbre")]
    pub motion: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct PitchPatch {
    #[serde(default, deserialize_with = "de_opt_f32_clamp_freq")]
    pub center_hz: Option<f32>,
    #[serde(default, deserialize_with = "de_opt_f32_clamp_range_oct")]
    pub range_oct: Option<f32>,
    #[serde(default, deserialize_with = "de_opt_f32_clamp_unit")]
    pub gravity: Option<f32>,
    #[serde(default, deserialize_with = "de_opt_f32_clamp_unit")]
    pub exploration: Option<f32>,
    #[serde(default, deserialize_with = "de_opt_f32_clamp_unit")]
    pub persistence: Option<f32>,
    #[serde(default)]
    pub constraint: Option<PitchConstraintPatch>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct PitchConstraintPatch {
    #[serde(default)]
    pub mode: Option<PitchConstraintMode>,
    #[serde(default, deserialize_with = "de_opt_f32_clamp_freq")]
    pub freq_hz: Option<f32>,
    #[serde(default, deserialize_with = "de_opt_f32_clamp_unit")]
    pub strength: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct PhonationPatch {
    #[serde(default, rename = "type")]
    pub r#type: Option<PhonationType>,
    #[serde(default, deserialize_with = "de_opt_f32_clamp_unit")]
    pub density: Option<f32>,
    #[serde(default, deserialize_with = "de_opt_f32_clamp_unit")]
    pub sync: Option<f32>,
    #[serde(default, deserialize_with = "de_opt_f32_clamp_unit")]
    pub legato: Option<f32>,
    #[serde(default, deserialize_with = "de_opt_f32_clamp_unit")]
    pub sociality: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
#[serde(deny_unknown_fields)]
pub struct PerceptualPatch {
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default, deserialize_with = "de_opt_f32_clamp_unit")]
    pub adaptation: Option<f32>,
    #[serde(default, deserialize_with = "de_opt_f32_clamp_unit")]
    pub novelty_bias: Option<f32>,
    #[serde(default, deserialize_with = "de_opt_f32_clamp_unit")]
    pub self_focus: Option<f32>,
}

pub fn merge_json(base: Value, patch: Value) -> Value {
    match (base, patch) {
        (Value::Object(mut base_map), Value::Object(patch_map)) => {
            for (k, v) in patch_map {
                let base_val = base_map.remove(&k).unwrap_or(Value::Null);
                base_map.insert(k, merge_json(base_val, v));
            }
            Value::Object(base_map)
        }
        (_, patch_val) => patch_val,
    }
}

pub fn remove_json_path(value: &mut Value, path: &str) -> bool {
    let mut parts = path.split('.').filter(|p| !p.is_empty()).peekable();
    let mut cursor = value;
    while let Some(part) = parts.next() {
        let is_last = parts.peek().is_none();
        match cursor {
            Value::Object(map) => {
                if is_last {
                    return map.remove(part).is_some();
                }
                if let Some(next) = map.get_mut(part) {
                    cursor = next;
                } else {
                    return false;
                }
            }
            _ => return false,
        }
    }
    false
}
