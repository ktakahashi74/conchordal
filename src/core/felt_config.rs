#[cfg(feature = "research")]
#[path = "felt_config/implementation/active.rs"]
mod implementation;
#[cfg(not(feature = "research"))]
#[path = "felt_config/implementation/inactive.rs"]
mod implementation;

pub(crate) use implementation::FeltConfig;
