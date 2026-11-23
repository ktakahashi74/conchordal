//! I11 arrival evidence borrowed by one phonation decision.

#[cfg(feature = "research")]
mod active;
#[cfg(not(feature = "research"))]
mod inactive;

#[cfg(all(test, feature = "research"))]
pub(crate) use active::tests;
#[cfg(feature = "research")]
pub(crate) use active::{ArrivalEvidence, ArrivalProvenance, ArrivalSet, ArrivalSources};
#[cfg(not(feature = "research"))]
pub(crate) use inactive::ArrivalSources;

#[derive(Clone, Copy)]
pub(crate) struct ArrivalContext<'a> {
    lifetime: std::marker::PhantomData<&'a ()>,
    #[cfg(feature = "research")]
    pub(crate) evidence: ArrivalEvidence<'a>,
}
