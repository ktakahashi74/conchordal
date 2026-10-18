mod context;
pub mod defs_gen;
pub(crate) mod docs;
mod engine;
mod placement;
mod spec;

use context::PopulationStatus;
use context::ScriptWarnings;
pub use context::{ScriptContext, ScriptWarning};
pub use placement::{Bus, BusSet, Placement};
use spec::{BrainKind, PhonationKind, PopulationSpec};
pub use spec::{PopulationHandle, PopulationSpecHandle};

use std::collections::BTreeMap;
use std::fs;
use std::sync::{Arc, Mutex};

use rand::random;
use rhai::{Array, Dynamic, Engine, EvalAltResult, FLOAT, FnPtr, INT, NativeCallContext, Position};

use crate::core::float::sanitize01;
use crate::core::landscape::PitchObjectiveMode;
use crate::core::mode_pattern::ModePattern;

use crate::core::meter::{MeterShaping, sane_basin};
use crate::scenario::PhonationTiming;
use crate::scenario::control::{
    BodyMethod, ControlUpdate, MAX_FREQ_HZ, MIN_FREQ_HZ, MoveCostTimeScale, PhonationGate,
    PitchApplyMode, PitchCoreKind, PitchMode, Routing, VoiceControl,
};
use crate::scenario::lifecycle::LifecycleConfig;
use crate::scenario::{
    Action, ArticulationCoreConfig, ControlUpdateMode, CoupledTimingSpec, DurationSpec,
    EnvelopeConfig, FieldDurationSpec, FieldSampling, FieldTarget, MetabolismRhythmReward,
    PhonationSpec, RespawnPeakBiasConfig, RespawnPolicy, RhythmCouplingMode, RhythmRelation,
    RhythmRewardMetric, RhythmRole, ScaffoldConfig, Scenario, SceneMarker, SpawnStrategy,
    TimedEvent, VoiceSpec,
};

fn parse_enum<T: Copy>(
    label: &str,
    name: &str,
    position: Position,
    values: &[(&str, T)],
) -> Result<T, Box<EvalAltResult>> {
    values
        .iter()
        .find(|(key, _)| *key == name)
        .map(|(_, value)| *value)
        .ok_or_else(|| {
            let expected = values
                .iter()
                .map(|(key, _)| *key)
                .collect::<Vec<_>>()
                .join(" or ");
            Box::new(EvalAltResult::ErrorRuntime(
                format!("{label} expects {expected}; got '{name}' (names are case-sensitive)")
                    .into(),
                position,
            ))
        })
}

const DEFAULT_SEQ_DURATION_SEC: f32 = 1.0;
const DEFAULT_PULSE_RATE: f32 = 2.25;
const DEFAULT_PULSE_SYNC: f32 = 0.5;
const DEFAULT_GATE_COUNT: u32 = 5;
const DEFAULT_CONSONANCE_MOVEMENT_GLIDE_TAU_SEC: f32 = 0.30;

fn rhai_array_to_f32(
    values: Array,
    label: &str,
    diagnostics: &ScriptWarnings,
    position: Position,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(values.len());
    for value in values {
        if let Some(v) = value.clone().try_cast::<FLOAT>() {
            out.push(v as f32);
            continue;
        }
        if let Some(v) = value.try_cast::<INT>() {
            out.push(v as f32);
            continue;
        }
        diagnostics
            .lock()
            .expect("lock script warnings")
            .push(ScriptWarning {
                message: format!("{label} expects numeric array elements"),
                position,
            });
    }
    out
}

pub struct ScriptHost;

#[derive(Debug, Clone)]
pub struct ScriptError {
    pub message: String,
    pub position: Option<Position>,
}

impl ScriptError {
    pub fn new(message: impl Into<String>, position: Option<Position>) -> Self {
        Self {
            message: message.into(),
            position,
        }
    }

    pub fn with_context(mut self, context: impl AsRef<str>) -> Self {
        let ctx = context.as_ref();
        self.message = format!("{ctx}: {}", self.message);
        self
    }

    pub fn from_eval(err: Box<EvalAltResult>, context: Option<&str>) -> Self {
        let pos = err.position();
        let position = if pos == Position::NONE {
            None
        } else {
            Some(pos)
        };
        let mut err = ScriptError::new(err.to_string(), position);
        if let Some(ctx) = context {
            err = err.with_context(ctx);
        }
        err
    }
}

impl std::fmt::Display for ScriptError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for ScriptError {}

#[cfg(test)]
mod tests;
