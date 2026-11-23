#[cfg(feature = "research")]
#[path = "ui_observation/active.rs"]
mod implementation;
#[cfg(not(feature = "research"))]
#[path = "ui_observation/inactive.rs"]
mod implementation;

pub(crate) use implementation::TemporalUi;
