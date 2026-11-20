//! Frozen bodily inputs, not acoustic predictions or permission to execute them.

use crate::core::timebase::Tick;

/// Facts passed to the phonation engine, not permission for a future candidate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[cfg_attr(test, derive(serde::Deserialize))]
pub struct PolicySnapshot {
    /// Placement at batch creation, retained by sounding notes through release.
    #[serde(default, skip_serializing_if = "crate::scenario::implicit_site")]
    pub site_id: u32,
    pub at: Tick,
    pub is_alive: bool,
    pub gate_allows_onset: bool,
}

#[cfg(test)]
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

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Opportunity {
    pub issued_at: Tick,
    pub at: Tick,
    pub basis: OpportunityBasis,
}
