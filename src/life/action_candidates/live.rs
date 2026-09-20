//! Read-only, sampled mapping of existing bodily transitions. No clock is advanced here.

use super::{BodyState, Class, Input, OnsetOpportunity, Opportunity, PolicySnapshot};
use crate::life::voice::PhonationBatch;
use std::collections::VecDeque;

const CAPACITY: usize = crate::temporal_cognition::body::VOICES;
pub(crate) const VERSION: u32 = 3;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct Record {
    pub version: u32,
    pub source_id: u64,
    pub source_generation: u32,
    pub body_generation: Option<u32>,
    pub issued_at: u64,
    pub policy: Option<PolicySnapshot>,
    pub opportunity: Option<Opportunity>,
    pub onset_opportunity: Option<OnsetOpportunity>,
    pub onset_recipe_tone_id: Option<u64>,
    pub onset_recipe_count: usize,
    pub intrinsic_period_sec: Option<f64>,
    pub candidate_times: [u64; 16],
    pub candidate_time_count: usize,
    pub active_tones: usize,
    /// Habitat, presentation. Envelope activity is not measured audibility.
    pub active_bus_tones: [usize; 2],
    pub queued_tones: usize,
    pub next_queued_onset: Option<u64>,
    pub other_source_generation_tones: usize,
    pub unmatched_body_tones: usize,
    pub accepted_transitions: usize,
    pub default_tone_id: Option<u64>,
    pub default_body_generation: Option<u32>,
    pub default_routed: Option<[bool; 2]>,
    pub default_input: Option<Input>,
    pub status: &'static str,
}

impl Record {
    pub(crate) fn finish_facts(&mut self, sample_rate: u32) {
        let (class, at, status) = if self.active_tones > 0 {
            (Class::Continue, self.issued_at, "active_body")
        } else if let Some(at) = self.next_queued_onset {
            (Class::DelayedOnset, at, "queued_onset")
        } else {
            return;
        };
        self.default_input = class.input(
            self.issued_at,
            at,
            None,
            sample_rate,
            BodyState {
                permits_action: Some(true),
                active_at_candidate: Some(self.active_tones > 0),
                pending_opportunity: false,
                due_unconsumed: false,
            },
        );
        self.status = status;
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Transition {
    pub tone_id: u64,
    pub at: u64,
    pub class: Class,
    pub active_at_candidate: bool,
    pub body_generation: Option<u32>,
    pub routed: [bool; 2],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub struct Stats {
    pub(crate) candidate_energy: super::energy::Stats,
    pub sampled: u64,
    pub mapped: u64,
    pub unknown: u64,
    pub rate_limited: u64,
    pub capacity_dropped: u64,
    pub output_dropped: u64,
    pub retired: u64,
    pub latest: Option<Record>,
}

struct Slot {
    owner: (u64, u32),
    last_sample: u64,
    record: Option<Record>,
}

pub(crate) struct Bank {
    pub(crate) energy: super::energy::Worker,
    pub(crate) pending: VecDeque<Box<super::energy::Packet>>,
    pub(crate) shared: [Option<
        std::sync::Arc<crate::temporal_cognition::action_profiles::consumer::Publication>,
    >; 2],
    pub(crate) bindings: Vec<crate::temporal_cognition::action_profiles::consumer::Binding>,
    slots: Vec<Option<Slot>>,
    ready: Vec<Record>,
    sample_rate: u32,
    pub(crate) stats: Stats,
}

impl Bank {
    pub(crate) fn new(sample_rate: u32) -> Self {
        assert!(sample_rate > 0);
        Self {
            energy: super::energy::Worker::new(),
            pending: VecDeque::with_capacity(CAPACITY),
            shared: [None, None],
            bindings: Vec::with_capacity(crate::temporal_cognition::body::VOICES * 2),
            slots: (0..CAPACITY).map(|_| None).collect(),
            ready: Vec::with_capacity(CAPACITY),
            sample_rate,
            stats: Stats::default(),
        }
    }

    pub(crate) fn begin_hop(&mut self, batches: &[PhonationBatch]) {
        for slot in &mut self.slots {
            if let Some(s) = slot
                && !batches
                    .iter()
                    .any(|b| s.owner == (b.source_id, b.source_generation))
            {
                if self
                    .stats
                    .latest
                    .is_some_and(|r| (r.source_id, r.source_generation) == s.owner)
                {
                    self.stats.latest = None;
                }
                *slot = None;
                self.stats.retired += 1;
            }
        }
    }

    pub(crate) fn sample_due(&self, owner: (u64, u32), now: u64) -> bool {
        self.slots
            .iter()
            .flatten()
            .find(|s| s.owner == owner)
            .is_none_or(|s| {
                now.saturating_sub(s.last_sample) >= u64::from(self.sample_rate).div_ceil(20)
            })
    }

    pub(crate) fn begin_source(
        &mut self,
        batch: &PhonationBatch,
        now: u64,
        body_generation: Option<u32>,
    ) -> Option<&mut Record> {
        let owner = (batch.source_id, batch.source_generation);
        let existing = self
            .slots
            .iter()
            .position(|s| s.as_ref().is_some_and(|s| s.owner == owner));
        let same_hop = existing.is_some_and(|i| {
            self.slots[i]
                .as_ref()
                .unwrap()
                .record
                .is_some_and(|r| r.issued_at == now)
        });
        if let Some(i) = existing {
            let s = self.slots[i].as_ref().unwrap();
            if now.saturating_sub(s.last_sample) < u64::from(self.sample_rate).div_ceil(20) {
                self.stats.rate_limited += 1;
                if !same_hop {
                    return None;
                }
            }
        }
        let Some(i) = existing.or_else(|| self.slots.iter().position(Option::is_none)) else {
            self.stats.capacity_dropped += 1;
            return None;
        };
        let mut recipes = batch.tones.iter().filter(|tone| {
            tone.opportunity.is_some_and(|o| o.issued_at == now && o.at == tone.onset && o.at >= now)
                && batch.body_policy.is_some_and(|p| p.at == now && p.is_alive && p.gate_allows_onset)
                && batch.cmds.iter().any(|cmd| matches!(cmd,
                    crate::life::phonation_engine::ToneCmd::On { tone_id, .. } if *tone_id == tone.tone_id))
        });
        let first_recipe = recipes.next();
        let onset_recipe_count = usize::from(first_recipe.is_some()) + recipes.count();
        if same_hop {
            let record = self.slots[i].as_mut().unwrap().record.as_mut().unwrap();
            record.onset_recipe_count += onset_recipe_count;
            if record.onset_recipe_tone_id.is_none() {
                record.onset_recipe_tone_id = first_recipe.map(|t| t.tone_id);
                record.onset_opportunity = first_recipe.and_then(|t| t.opportunity);
            }
            return None;
        }
        self.slots[i] = Some(Slot {
            owner,
            last_sample: now,
            record: Some(Record {
                version: VERSION,
                source_id: owner.0,
                source_generation: owner.1,
                body_generation,
                issued_at: now,
                policy: batch.body_policy.filter(|p| p.at == now),
                opportunity: batch
                    .body_opportunity
                    .filter(|p| p.issued_at == now && p.at >= now),
                onset_opportunity: first_recipe.and_then(|t| t.opportunity),
                onset_recipe_tone_id: first_recipe.map(|t| t.tone_id),
                onset_recipe_count,
                intrinsic_period_sec: batch
                    .intrinsic_period_sec
                    .filter(|p| p.is_finite() && *p > 0.0),
                candidate_times: [0; 16],
                candidate_time_count: 0,
                active_tones: 0,
                active_bus_tones: [0; 2],
                queued_tones: 0,
                next_queued_onset: None,
                other_source_generation_tones: 0,
                unmatched_body_tones: 0,
                accepted_transitions: 0,
                default_tone_id: None,
                default_body_generation: None,
                default_routed: None,
                default_input: None,
                status: "no_planned_transition",
            }),
        });
        self.slots[i].as_mut().unwrap().record.as_mut()
    }

    pub(crate) fn transition(&mut self, owner: (u64, u32), now: u64, transition: Transition) {
        let Some(record) = self
            .slots
            .iter_mut()
            .flatten()
            .find(|s| s.owner == owner)
            .and_then(|s| s.record.as_mut())
            .filter(|r| r.issued_at == now)
        else {
            return;
        };
        record.accepted_transitions += 1;
        if record.accepted_transitions != 1 {
            return;
        }
        record.default_tone_id = Some(transition.tone_id);
        record.default_body_generation = transition.body_generation;
        record.default_routed = Some(transition.routed);
        record.default_input = transition.class.input(
            now,
            transition.at,
            None,
            self.sample_rate,
            BodyState {
                permits_action: Some(true),
                active_at_candidate: Some(transition.active_at_candidate),
                pending_opportunity: false,
                due_unconsumed: false,
            },
        );
        record.status = if record.default_input.is_some() {
            "accepted_transition"
        } else {
            "accepted_transition_unmapped"
        };
    }

    pub(crate) fn end_hop(&mut self) {
        self.energy.poll();
        self.stats.candidate_energy = self.energy.stats;
        for slot in self.slots.iter_mut().flatten() {
            let Some(mut record) = slot.record.take() else {
                continue;
            };
            // Clock points remain diagnostic: they do not make an action legal.
            if let Some(at) = record
                .default_input
                .map(|i| i.at)
                .or(record.opportunity.map(|o| o.at))
            {
                let span = record
                    .intrinsic_period_sec
                    .map(|p| ((2.0 * p).min(4.0) * f64::from(self.sample_rate)).floor() as u64);
                if let Some((times, count)) = super::times(record.issued_at, span, at, [None; 3]) {
                    record.candidate_times = times;
                    record.candidate_time_count = count;
                }
            }
            self.stats.sampled += 1;
            if record.default_input.is_some() {
                self.stats.mapped += 1;
            } else {
                self.stats.unknown += 1;
            }
            self.stats.latest = Some(record);
            if self.ready.len() < self.ready.capacity() {
                self.ready.push(record);
            } else {
                self.stats.output_dropped += 1;
            }
        }
    }

    pub(crate) fn drain(&mut self) -> impl Iterator<Item = Record> + '_ {
        self.ready.drain(..)
    }

    pub(crate) fn sampled_record(&self, owner: (u64, u32), now: u64) -> Option<Record> {
        self.slots.iter().flatten().find_map(|slot| {
            slot.record
                .filter(|r| slot.owner == owner && r.issued_at == now)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clock_grid_keeps_far_default_without_granting_permission_or_reusing_stale_inputs() {
        let mut bank = Bank::new(1000);
        let mut batch = PhonationBatch {
            body_policy: Some(PolicySnapshot {
                at: 10,
                is_alive: false,
                gate_allows_onset: false,
            }),
            body_opportunity: Some(Opportunity {
                issued_at: 10,
                at: 10_000,
                basis: super::super::OpportunityBasis::ParticipationDue,
            }),
            intrinsic_period_sec: Some(0.1),
            ..Default::default()
        };
        bank.begin_source(&batch, 10, None)
            .unwrap()
            .finish_facts(1000);
        bank.end_hop();
        let first = bank.drain().next().unwrap();
        assert_eq!(first.default_input, None);
        assert_eq!(first.candidate_time_count, 13);
        assert_eq!(first.candidate_times[12], 10_000);
        assert_eq!(first.candidate_times[11], 210);
        assert!(first.candidate_times[..13].windows(2).all(|p| p[0] < p[1]));
        // Sampling a reused batch at a later issue must not recycle the old clock point.
        bank.begin_source(&batch, 60, None)
            .unwrap()
            .finish_facts(1000);
        bank.end_hop();
        let stale = bank.drain().next().unwrap();
        assert_eq!(stale.opportunity, None);
        assert_eq!(stale.candidate_time_count, 0);
        batch.body_opportunity.as_mut().unwrap().issued_at = 110;
        let record = bank.begin_source(&batch, 110, None).unwrap();
        record.active_tones = 1;
        record.finish_facts(1000);
        bank.end_hop();
        let active = bank.drain().next().unwrap();
        assert_eq!(active.default_input.unwrap().class, Class::Continue);
        assert_eq!(active.candidate_time_count, 12);
        assert_eq!(active.candidate_times[0], 110);
        assert!(!active.candidate_times[..12].contains(&10_000));
    }

    #[test]
    fn sampling_keeps_unknown_policy_and_first_unsupported_transition() {
        let mut bank = Bank::new(1001);
        let mut batch = PhonationBatch {
            source_id: 7,
            body_policy: Some(PolicySnapshot {
                at: 99,
                is_alive: true,
                gate_allows_onset: true,
            }),
            intrinsic_period_sec: Some(f64::NAN),
            ..Default::default()
        };
        assert!(bank.sample_due((7, 0), 100));
        let record = bank.begin_source(&batch, 100, Some(9)).unwrap();
        assert_eq!(record.policy, None);
        assert_eq!(record.intrinsic_period_sec, None);
        record.active_tones = 1;
        record.finish_facts(1001);
        assert_eq!(record.default_input.unwrap().class, Class::Continue);
        assert!(!bank.sample_due((7, 0), 99));
        assert!(!bank.sample_due((7, 0), 100));
        assert!(!bank.sample_due((7, 0), 150));
        assert!(bank.sample_due((7, 0), 151));
        assert!(bank.sample_due((7, 1), 100));
        let transition = Transition {
            tone_id: 4,
            at: 101,
            class: Class::Release,
            active_at_candidate: false,
            body_generation: Some(8),
            routed: [false, true],
        };
        bank.transition((7, 1), 100, transition);
        bank.transition((7, 0), 99, transition);
        bank.transition((7, 0), 100, transition);
        bank.transition(
            (7, 0),
            100,
            Transition {
                class: Class::DelayedOnset,
                tone_id: 5,
                ..transition
            },
        );
        assert!(bank.begin_source(&batch, 100, Some(9)).is_none());
        bank.end_hop();
        let result = bank.drain().next().unwrap();
        assert_eq!(result.accepted_transitions, 2);
        assert_eq!(result.default_tone_id, Some(4));
        assert_eq!(result.default_body_generation, Some(8));
        assert_eq!(result.default_input, None);
        assert_eq!(result.status, "accepted_transition_unmapped");
        assert!(bank.begin_source(&batch, 99, Some(9)).is_none());
        assert!(bank.begin_source(&batch, 150, Some(9)).is_none());
        batch.body_policy.as_mut().unwrap().at = 151;
        assert!(
            bank.begin_source(&batch, 151, Some(9))
                .unwrap()
                .policy
                .is_some()
        );
        bank.end_hop();
        assert_eq!(bank.stats.unknown, 2);
    }

    #[test]
    fn owners_capacity_output_and_retirement_are_bounded_and_separate() {
        let mut bank = Bank::new(1000);
        let mut batches: Vec<_> = (0..=CAPACITY)
            .map(|i| PhonationBatch {
                source_id: i as u64,
                ..Default::default()
            })
            .collect();
        for batch in &batches {
            bank.begin_source(batch, 0, Some(3));
        }
        bank.end_hop();
        assert_eq!(bank.stats.capacity_dropped, 1);
        for batch in &batches {
            bank.begin_source(batch, 50, Some(3));
        }
        bank.end_hop();
        assert_eq!(bank.stats.capacity_dropped, 2);
        assert_eq!(bank.stats.output_dropped, CAPACITY as u64);
        assert_eq!(bank.drain().count(), CAPACITY);
        batches.truncate(1);
        batches[0].source_generation = 1;
        bank.begin_hop(&batches);
        assert_eq!(bank.stats.latest, None);
        assert_eq!(bank.stats.retired, CAPACITY as u64);
        // A new generation must not inherit its predecessor's sampling clock or activity.
        let record = bank.begin_source(&batches[0], 51, Some(0)).unwrap();
        assert_eq!(record.source_generation, 1);
        assert_eq!(record.active_tones, 0);
        assert_eq!(record.default_input, None);
        bank.end_hop();
        assert_eq!(bank.drain().count(), 1);
    }
}
