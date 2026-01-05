pub mod api;
pub mod conductor;
pub mod gate_clock;
pub mod individual;
pub mod intent;
pub mod intent_planner;
pub mod lifecycle;
pub mod meta;
pub mod perceptual;
pub mod phonation_engine;
pub mod plan;
pub mod population;
pub mod predictive_spectrum;
pub mod scenario;
pub mod schedule_renderer;
pub mod scripting;
pub mod sound_voice;
pub mod world_model;

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_organic;
