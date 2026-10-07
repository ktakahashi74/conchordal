//! Ordered auditory evidence, group ownership and periodic forecasts.

mod accents;
pub(crate) mod arrival;
pub(crate) mod body;
pub(crate) mod body_model;
pub(crate) mod context;
mod feature_projection;
mod features;
mod group;
mod grouping;
mod observables;
pub(crate) mod observation;
pub(crate) mod proposals;
pub(crate) mod resources;
mod ridge;
mod trajectory;

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use accents::Summary as AccentSummary;
pub(crate) use ridge::Handle;
