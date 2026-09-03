pub(crate) mod articulation_envelope;
pub(crate) mod conductor;
pub(crate) mod constants;
mod control_adapters;
pub mod gate_clock;
pub mod voice;
pub(crate) use voice::articulation_core;
pub(crate) mod adaptation;
pub mod community;
pub mod generator_model;
pub(crate) mod metabolism_policy;
pub(crate) mod modal;
pub mod phonation_engine;
pub(crate) mod report;
pub mod schedule_renderer;
pub(crate) mod social_density;
pub(crate) mod telemetry;

pub mod sound;
#[cfg(test)]
mod tests;
