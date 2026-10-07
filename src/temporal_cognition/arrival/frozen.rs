//! Frozen owner and support boundary for conditional arrival forecasts.

use super::*;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Frozen {
    pub group: Handle,
    pub issued_at: u64,
    pub sample_rate: u32,
    pub(super) engine: Engine,
}

impl Engine {
    pub(in crate::temporal_cognition) fn freeze(
        &self,
        group: Handle,
        issue: u64,
        rate: u32,
    ) -> Option<Frozen> {
        let last = self.last?;
        (rate > 0
            && self.cut == Some(issue)
            && last.group == group
            && last.event_end <= issue
            && last.at_cut(issue)
            && self.context.available <= issue
            && self.context.source_end <= self.context.available)
            .then_some(Frozen {
                group,
                issued_at: issue,
                sample_rate: rate,
                engine: *self,
            })
    }
}
