//! Frozen bodily inputs, not acoustic predictions or permission to execute them.

use crate::core::timebase::Tick;

/// Facts passed to the phonation engine, not permission for a future candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[cfg_attr(test, derive(serde::Deserialize))]
pub struct PolicySnapshot {
    pub at: Tick,
    pub is_alive: bool,
    pub gate_allows_onset: bool,
}

/// Frozen clock state, not a reservation or future permission to emit sound.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
#[cfg_attr(test, derive(serde::Deserialize))]
pub enum OpportunityBasis {
    ParticipationDue,
    ParticipationPlanned,
    CouplingProjection,
    ThetaProjection,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[cfg_attr(test, derive(serde::Deserialize))]
pub struct Opportunity {
    pub issued_at: Tick,
    pub at: Tick,
    pub basis: OpportunityBasis,
}

/// A granted clock candidate, captured before onset emission.
/// Participation's intrinsic due time is read before resolving the selected plan.
/// Attached to its one materialized ToneSpec; later reports are receipts, not pending actions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[cfg_attr(test, derive(serde::Deserialize))]
pub struct OnsetOpportunity {
    pub issued_at: Tick,
    pub at: Tick,
    pub gate: u64,
    pub intrinsic_due_at: Option<Tick>,
    pub intrinsic_period_ticks: Option<Tick>,
    pub planned_release_at: Option<Tick>,
}
