pub mod any_backend;
pub mod control;
pub mod events;
pub mod modal_engine;
pub(crate) mod spectral;
pub mod voice;

pub use any_backend::AnyBackend;
pub use control::{ControlRamp, VoiceControlBlock};
pub use events::{AudioCommand, BodyKind, BodySnapshot, IndividualId, VoiceTarget};
pub use modal_engine::{ModalEngine, ModalMode, ModeShape};
pub use voice::{Voice, default_release_ticks};
