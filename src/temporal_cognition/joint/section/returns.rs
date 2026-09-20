//! One earlier-context proposal per selected correspondence, from sealed memory provenance.

use super::Return;
use crate::temporal_cognition::{joint::correspondence, recall::Recall, ridge::Handle};

pub(in crate::temporal_cognition) struct Returns {
    pub(in crate::temporal_cognition::joint) group: Handle,
    cut: u64,
    entries: [Option<Return>; 16],
}

impl Returns {
    pub fn new(group: Handle) -> Self {
        Self {
            group,
            cut: 0,
            entries: [None; 16],
        }
    }

    pub fn refresh(
        &mut self,
        cache: &correspondence::Cache,
        memory: &Recall,
    ) -> Result<(), &'static str> {
        if cache.group != self.group || cache.cut < self.cut {
            return Err("invalid section-return cache owner or cut");
        }
        let mut entries = [None; 16];
        if let Some((query, _)) = memory.matches_for(self.group).filter(|(q, _)| {
            !q.cutoff_tie
                && q.received_at <= cache.cut
                && cache.cut <= q.deadline
                && q.cue.is_some_and(|c| {
                    c.occurrence_id > 0
                        && c.start_sample == q.support_start_sample
                        && c.support_end_sample == q.support_end_sample
                        && c.selected_at <= q.issued_at
                        && c.weighted_seconds > 0.
                })
                && q.supporting_audio_end.is_some_and(|end| {
                    end <= cache.cut && u128::from(cache.cut - end) * 2 < u128::from(cache.rate)
                })
        }) {
            for (slot, state) in cache.states[..16]
                .iter()
                .enumerate()
                .filter_map(|(i, s)| s.map(|s| (i, s)))
            {
                if state.support.query != query.query_id {
                    continue;
                }
                let Some(matched) = state.matched.filter(|m| {
                    !m.ambiguous
                        && m.cost <= 1.
                        && m.transformation
                            .iter()
                            .all(|v| v.is_some_and(|v| v.abs() < 2.))
                }) else {
                    continue;
                };
                let Some(origin) = memory
                    .occurrence(matched.episode_id, matched.episode_generation)
                    .filter(|o| {
                        o.group.bus == self.group.bus
                            && o.group.epoch == self.group.epoch
                            && o.start <= matched.support_start_sample
                            && matched.support_end_sample <= o.end
                            && o.end <= query.support_start_sample
                            && o.sealed_at <= query.issued_at
                            && o.support > 0.
                            && o.retained_mass > 0.
                    })
                else {
                    continue;
                };
                let strongest = origin
                    .contexts
                    .iter()
                    .flatten()
                    .filter(|c| {
                        c.context_id > 0
                            && c.mass.is_finite()
                            && c.mass > 0.
                            && c.mass <= origin.retained_mass
                    })
                    .max_by(|a, b| {
                        a.mass
                            .total_cmp(&b.mass)
                            .then(b.context_id.cmp(&a.context_id))
                    });
                if let Some(context) = strongest {
                    entries[slot] = Some(Return {
                        cut: cache.cut,
                        rate: cache.rate,
                        correspondence: state,
                        query,
                        origin,
                        context: *context,
                    });
                }
            }
        }
        self.cut = cache.cut;
        self.entries = entries;
        Ok(())
    }

    pub(in crate::temporal_cognition::joint) fn entries(&self) -> impl Iterator<Item = &Return> {
        self.entries.iter().flatten()
    }

    pub fn for_correspondence(&self, state: &correspondence::State) -> Option<&Return> {
        self.entries.iter().flatten().find(|r| {
            r.correspondence.group == state.group
                && r.correspondence.support == state.support
                && r.correspondence.matched == state.matched
        })
    }
}
