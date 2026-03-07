pub mod any_backend;
pub mod control;
pub mod events;
pub mod harmonic_resonator_backend;
pub mod modal_engine;
pub(crate) mod mode_utils;
pub mod render_modulator;
pub mod sine_osc_backend;
pub(crate) mod spectral;
pub mod voice;

pub use any_backend::AnyBackend;
pub use control::{ControlRamp, VoiceControlBlock};
pub use events::{AudioCommand, BodyKind, BodySnapshot, IndividualId, VoiceTarget};
pub use modal_engine::{ModalEngine, ModalMode, ModeShape};
pub(crate) use render_modulator::RenderModulator;
pub use render_modulator::{AutonomousPulseSpec, RenderModulatorSpec, RenderModulatorStateKind};
pub use voice::{Voice, default_release_ticks};
