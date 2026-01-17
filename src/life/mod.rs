pub mod conductor;
pub mod control;
mod control_adapters;
pub mod gate_clock;
pub mod individual;
pub mod intent;
pub mod lifecycle;
pub mod meta;
pub mod perceptual;
pub mod phonation_engine;
pub mod population;
pub mod predictive_spectrum;
pub mod scenario;
pub mod schedule_renderer;
pub mod scripting;
pub mod social_density;
pub mod world_model;

pub mod sound;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_organic;
