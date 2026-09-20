//! Ordered auditory evidence and M0 numerical model preparation.

mod accents;
pub(crate) mod action_profiles;
pub(crate) mod arrival;
mod auditory_timing;
pub(crate) mod body;
pub(crate) mod body_model;
pub(crate) mod context;
mod descriptor;
mod feature_projection;
mod features;
pub(crate) mod gesture;
mod group;
mod grouping;
mod matcher;
mod memory;
mod observables;
pub(crate) mod observation;
pub(crate) mod private_trace;
pub(crate) mod proposals;
mod query;
pub(crate) mod recall;
pub(crate) mod reference_inventory;
pub(crate) mod resources;
mod ridge;
mod trajectory;
mod transport;

#[cfg(test)]
mod tests;
