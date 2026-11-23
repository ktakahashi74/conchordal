//! Ordered auditory evidence, group ownership and periodic forecasts.

#[cfg(feature = "research")]
mod accents;
#[cfg(feature = "research")]
pub(crate) mod arrival;
#[cfg(feature = "research")]
pub(crate) mod body;
#[cfg(feature = "research")]
pub(crate) mod body_model;
pub(crate) mod capture;
#[cfg(feature = "research")]
pub(crate) mod context;
#[cfg(feature = "research")]
mod feature_projection;
#[cfg(feature = "research")]
mod features;
#[cfg(feature = "research")]
mod group;
#[cfg(feature = "research")]
mod grouping;
pub(crate) mod harmonic_relations;
#[cfg(feature = "research")]
mod observables;
#[cfg(feature = "research")]
pub(crate) mod observation;
#[cfg(feature = "research")]
pub(crate) mod proposals;
#[cfg(feature = "research")]
pub(crate) mod resources;
#[cfg(feature = "research")]
mod ridge;
#[cfg(feature = "research")]
pub(crate) mod t2;
#[cfg(feature = "research")]
mod trajectory;
pub(crate) mod ui_observation;

#[cfg(all(test, feature = "research"))]
mod tests;

#[cfg(all(test, feature = "research"))]
pub(crate) use accents::Summary as AccentSummary;
#[cfg(feature = "research")]
pub(crate) use ridge::Handle;
