#[cfg(feature = "research")]
mod active;
#[cfg(not(feature = "research"))]
mod inactive;

#[cfg(feature = "research")]
pub(crate) use active::BodyMetabolismPort;
#[cfg(all(test, feature = "research"))]
pub(crate) use active::assay;
#[cfg(not(feature = "research"))]
pub(crate) use inactive::BodyMetabolismPort;
