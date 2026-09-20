//! Ordered auditory evidence and M0 numerical model preparation.

mod accents;
pub(crate) mod action_profiles;
pub(crate) mod arrival;
mod auditory_timing;
pub(crate) mod body;
pub(crate) mod body_model;
#[cfg(test)]
mod consequence;
mod descriptor;
mod feature_projection;
mod features;
pub(crate) mod gesture;
pub(crate) mod groove;
mod group;
mod grouping;
mod hazard;
#[cfg(test)]
mod joint;
#[cfg(test)]
mod long_form;
mod matcher;
mod memory;
mod observables;
pub(crate) mod observation;
pub(crate) mod phrase;
pub(crate) mod private_trace;
pub(crate) mod proposals;
mod query;
mod ratings;
pub(crate) mod recall;
pub(crate) mod reference_inventory;
pub(crate) mod resources;
mod ridge;
pub(crate) mod section;
mod trajectory;
mod transport;
pub(crate) mod whole;

#[cfg(test)]
mod tests;
