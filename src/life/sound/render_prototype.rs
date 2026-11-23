#[cfg(feature = "research")]
mod active;
#[cfg(not(feature = "research"))]
mod inactive;

#[cfg(feature = "research")]
pub(crate) use active::RenderPrototype;
#[cfg(not(feature = "research"))]
pub(crate) use inactive::RenderPrototype;
