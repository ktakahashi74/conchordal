pub(crate) mod action_candidates;
pub(crate) mod arrival_cost;
pub(crate) mod articulation_envelope;
pub(crate) mod conductor;
pub(crate) mod constants;
mod control_adapters;
pub(crate) mod energy_window;
pub(crate) mod flora;
pub mod gate_clock;
pub(crate) mod onset_footprint;
pub(crate) mod tone_energy;
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
pub(crate) mod shape_policy;
pub(crate) mod social_density;
pub(crate) mod telemetry;
pub(crate) mod temporal_participation;

pub(crate) mod body_fitness;
pub(crate) mod body_metabolism;
pub mod sound;
#[cfg(test)]
mod tests;

pub(crate) mod own_sequence;
pub(crate) mod ppm_decay;
pub(crate) mod span_gap;
