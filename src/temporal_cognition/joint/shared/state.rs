//! Lagged conditional summaries owned by complete shared paths, never merged by labels.

use crate::temporal_cognition::ridge::Handle;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::temporal_cognition::joint) struct Support {
    pub interval: [u64; 2],
    pub source_start: u64,
    pub available: u64,
    pub assignment_seconds: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::temporal_cognition::joint) struct Section {
    pub context: u64,
    pub mass: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::temporal_cognition::joint) struct Group {
    pub group: Handle,
    pub support: Option<Support>,
    pub last_observed_end: Option<u64>,
    pub explicit_unknown: f64,
    pub pruned_mass: f64,
    pub represented_mass: f64,
    pub articulation: [f64; 4],
    // Index 0 is known absence/inactivity; index 1 is presence/activity.
    pub grouping: [f64; 2],
    pub phrase: [f64; 2],
    pub correspondence: [f64; 2],
    pub sections: [Option<Section>; 15],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(in crate::temporal_cognition::joint) struct State {
    pub key: u64,
    pub admission: Option<super::producer::Origin>,
    pub end: u64,
    pub groups: [Option<Group>; 8],
}
