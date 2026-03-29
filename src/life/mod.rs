pub(crate) mod articulation_envelope;
pub mod conductor;
pub mod constants;
pub mod control;
mod control_adapters;
pub mod gate_clock;
pub mod voice;
pub use voice::articulation_core;
pub mod adaptation;
pub mod lifecycle;
pub mod meta;
pub mod metabolism_policy;
pub mod modal;
pub mod phonation_engine;
pub mod population;
pub mod report;
pub mod scenario;
pub mod schedule_renderer;
pub mod scripting;
pub mod social_density;
pub mod telemetry;
pub mod world_model;

pub mod sound;
#[cfg(test)]
mod tests;
