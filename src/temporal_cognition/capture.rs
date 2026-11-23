#[cfg(feature = "research")]
#[path = "capture/active.rs"]
mod implementation;
#[cfg(not(feature = "research"))]
#[path = "capture/inactive.rs"]
mod implementation;

pub(crate) use implementation::{BodyCapture, BodyCaptureSlot};
