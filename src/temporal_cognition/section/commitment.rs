//! Bounded endpoint commitments over retained phrase ancestry, in observation time.

use std::sync::Arc;

use crate::temporal_cognition::{descriptor, phrase, ridge::Handle};

pub(super) const PENDING: usize = 512;

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct ContextSupport {
    pub context_id: u64,
    pub mass: f64,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Evidence {
    pub occurrence_id: u64,
    pub sequence: u64,
    pub ongoing_credit: u64,
    pub group: Handle,
    pub start: u64,
    pub end: u64,
    pub deadline: u64,
    pub sealed_at: u64,
    pub assignment_seconds: f64,
    pub retained_mass: f64,
    pub support: f64,
    pub lag_known_seconds: f64,
    pub lag_missing_seconds: f64,
    pub ending: Option<crate::temporal_cognition::observables::WindowDescriptor>,
    pub contexts: [Option<ContextSupport>; 16],
}

pub(super) struct Sealed {
    pub evidence: Evidence,
    pub descriptor: Arc<descriptor::Frozen>,
    members: u16,
    revised: bool,
    latest_mass: f64,
}

struct Pending {
    evidence: Evidence,
    descriptor: Arc<descriptor::Frozen>,
    members: u16,
}

pub(super) struct Ledger {
    rate: u32,
    next: u64,
    capacity: usize,
    pending: Vec<Pending>,
    pub sealed: Vec<Sealed>,
    pub total: u64,
    pub lost: u64,
    pub revisions: u64,
}

impl Ledger {
    pub(super) fn retains_prefix(&self, group: Handle, credit: u64, start: u64) -> bool {
        self.pending.iter().any(|p| {
            p.evidence.group == group
                && p.evidence.ongoing_credit == credit
                && p.evidence.start == start
        })
    }

    pub fn new(rate: u32, capacity: usize) -> Self {
        Self {
            rate,
            // Ongoing credits and sealed endpoints occupy disjoint query namespaces.
            next: 1 << 63,
            capacity,
            pending: Vec::with_capacity(PENDING),
            sealed: Vec::with_capacity(capacity),
            total: 0,
            lost: 0,
            revisions: 0,
        }
    }

    pub fn advance(
        &mut self,
        group: &phrase::GroupSnapshot,
        end: u64,
        observed: Option<(u64, u64)>,
    ) -> Result<(), &'static str> {
        let inherit = |members: u16| -> Result<(u16, f64), &'static str> {
            let mut next = 0;
            let mut mass = 0.;
            for (i, p) in group
                .candidates
                .iter()
                .enumerate()
                .filter_map(|(i, p)| p.map(|p| (i, p)))
            {
                if p.parent_index >= 15 || !p.mass.is_finite() || !(0. ..=1.).contains(&p.mass) {
                    return Err("invalid commitment parent or retained mass");
                }
                if members & (1 << p.parent_index) != 0 {
                    next |= 1 << i;
                    mass += p.mass;
                }
            }
            if mass > 1. + 1e-12 {
                return Err("commitment retained mass exceeds one");
            }
            Ok((next, mass))
        };
        for p in self
            .pending
            .iter_mut()
            .filter(|p| p.evidence.group == group.group)
        {
            // A hop arriving past the deadline cannot revise the last admissible support.
            if u128::from(end) * 2 <= u128::from(p.evidence.end) * 2 + u128::from(self.rate) {
                let (members, mass) = inherit(p.members)?;
                p.members = members;
                p.evidence.retained_mass = mass;
                if let Some((a, b)) = observed.filter(|(_, b)| *b <= p.evidence.deadline) {
                    p.evidence.lag_known_seconds +=
                        b.saturating_sub(a.max(p.evidence.end)) as f64 / f64::from(self.rate);
                }
            }
        }
        for p in self
            .sealed
            .iter_mut()
            .filter(|p| p.evidence.group == group.group)
        {
            let (members, mass) = inherit(p.members)?;
            p.members = members;
            p.latest_mass = mass;
        }
        Ok(())
    }

    pub fn stage(
        &mut self,
        group: &phrase::GroupSnapshot,
        span: phrase::Foreground,
        descriptor: descriptor::Frozen,
        assignment_seconds: f64,
        end: u64,
        observed: Option<(u64, u64)>,
    ) -> Result<(), &'static str> {
        if self.sealed.iter().any(|p| {
            p.evidence.group == group.group
                && p.evidence.ongoing_credit == span.credit
                && p.evidence.end == span.heard_end
        }) {
            return Ok(());
        }
        let rate = f64::from(self.rate);
        if span.start >= span.heard_end || assignment_seconds <= 0. {
            return Ok(());
        }
        if !assignment_seconds.is_finite()
            || assignment_seconds > (span.heard_end - span.start) as f64 / rate + 1e-12
            || descriptor.group != group.group
            || (descriptor.start * rate).round() as u64 != span.start
            || (descriptor.end * rate).round() as u64 != span.heard_end
        {
            return Err("commitment must preserve original endpoint and acoustic support");
        }
        let deadline = span
            .heard_end
            .checked_add(u64::from(self.rate).div_ceil(2))
            .ok_or("commitment deadline overflow")?;
        if u128::from(end) * 2 > u128::from(span.heard_end) * 2 + u128::from(self.rate) {
            self.lost += 1;
            return Ok(());
        }
        let mut members = 0;
        let mut mass = 0.;
        let mut ending = None;
        for (i, p) in group
            .candidates
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.map(|p| (i, p)))
        {
            if p.completed_foreground.is_some_and(|f| {
                f.credit == span.credit && f.start == span.start && f.heard_end == span.heard_end
            }) {
                members |= 1 << i;
                mass += p.mass;
                if ending.is_some() && ending != p.completed_ending {
                    return Err("endpoint aliases changed their original ending descriptor");
                }
                ending = p.completed_ending;
            }
        }
        if mass > 1. + 1e-12 {
            return Err("endpoint alternatives exceed retained mass");
        }
        if let Some(pending) = self.pending.iter_mut().find(|p| {
            p.evidence.group == group.group
                && p.evidence.ongoing_credit == span.credit
                && p.evidence.end == span.heard_end
        }) {
            if pending.evidence.start != span.start
                || pending.evidence.ending != ending
                || pending.descriptor.blocks != descriptor.blocks
            {
                return Err("pending endpoint alias changed original evidence");
            }
            pending.members |= members;
            pending.evidence.retained_mass = group
                .candidates
                .iter()
                .enumerate()
                .filter_map(|(i, c)| {
                    c.filter(|_| pending.members & (1 << i) != 0)
                        .map(|c| c.mass)
                })
                .sum();
            return Ok(());
        }
        if self.pending.len() == PENDING {
            self.lost += 1;
            return Ok(());
        }
        self.next = self
            .next
            .checked_add(1)
            .ok_or("sealed occurrence IDs exhausted")?;
        // Groups can deliver different original endpoints at the same observation cut.
        let position = self.pending.partition_point(|p| {
            (p.evidence.end, p.evidence.start, p.evidence.occurrence_id)
                < (span.heard_end, span.start, self.next)
        });
        self.pending.insert(
            position,
            Pending {
                evidence: Evidence {
                    occurrence_id: self.next,
                    sequence: 0,
                    ongoing_credit: span.credit,
                    group: group.group,
                    start: span.start,
                    end: span.heard_end,
                    deadline,
                    sealed_at: 0,
                    assignment_seconds,
                    retained_mass: mass,
                    support: 0.,
                    lag_known_seconds: observed
                        .filter(|(_, b)| {
                            u128::from(*b) * 2
                                <= u128::from(span.heard_end) * 2 + u128::from(self.rate)
                        })
                        .map_or(0., |(a, b)| {
                            b.saturating_sub(a.max(span.heard_end)) as f64 / rate
                        }),
                    lag_missing_seconds: 0.,
                    ending,
                    contexts: [None; 16],
                },
                descriptor: Arc::new(descriptor),
                members,
            },
        );
        Ok(())
    }

    pub fn seal(&mut self, end: u64, retained: &[Option<phrase::GroupSnapshot>; 7]) {
        let mut i = 0;
        while i < self.pending.len() {
            if !retained
                .iter()
                .flatten()
                .any(|g| g.group == self.pending[i].evidence.group)
            {
                self.pending.remove(i);
                self.lost += 1;
                continue;
            }
            if self.pending[i].evidence.deadline > end {
                i += 1;
                continue;
            }
            let mut p = self.pending.remove(i);
            p.evidence.sealed_at = end;
            p.evidence.support =
                p.evidence.retained_mass * p.evidence.assignment_seconds * f64::from(self.rate)
                    / (p.evidence.end - p.evidence.start) as f64;
            p.evidence.lag_missing_seconds = (0.5 - p.evidence.lag_known_seconds).max(0.);
            if p.evidence.support <= 0. {
                continue;
            }
            p.evidence.sequence = self.total + 1;
            if self.sealed.len() == self.capacity {
                self.sealed.remove(0);
            }
            self.sealed.push(Sealed {
                evidence: p.evidence,
                descriptor: p.descriptor,
                members: p.members,
                revised: false,
                latest_mass: p.evidence.retained_mass,
            });
            self.total += 1;
        }
    }

    pub fn project(
        &mut self,
        group: Handle,
        end: u64,
        rows: &[(u64, u64, u64, f64)],
    ) -> Result<(), &'static str> {
        for pending in self.pending.iter_mut().filter(|p| {
            p.evidence.group == group
                && u128::from(end) * 2 <= u128::from(p.evidence.end) * 2 + u128::from(self.rate)
        }) {
            let mut contexts: Vec<ContextSupport> = Vec::with_capacity(240);
            for &(credit, endpoint, context, mass) in rows {
                if credit != pending.evidence.ongoing_credit || endpoint != pending.evidence.end {
                    continue;
                }
                if !mass.is_finite() || mass < 0. {
                    return Err("invalid joint phrase/section support");
                }
                if let Some(c) = contexts.iter_mut().find(|c| c.context_id == context) {
                    c.mass += mass;
                } else {
                    contexts.push(ContextSupport {
                        context_id: context,
                        mass,
                    });
                }
            }
            let total: f64 = contexts.iter().map(|c| c.mass).sum();
            if total > pending.evidence.retained_mass + 1e-10 {
                return Err("joint section support exceeds its phrase ancestry");
            }
            contexts.sort_by(|a, b| {
                b.mass
                    .total_cmp(&a.mass)
                    .then(a.context_id.cmp(&b.context_id))
            });
            pending.evidence.contexts = std::array::from_fn(|i| contexts.get(i).copied());
        }
        for sealed in self
            .sealed
            .iter_mut()
            .filter(|p| p.evidence.group == group && !p.revised)
        {
            let rows: Vec<_> = rows
                .iter()
                .filter(|r| r.0 == sealed.evidence.ongoing_credit && r.1 == sealed.evidence.end)
                .collect();
            // Truncated ancestry is not a new interpretation of a sealed occurrence.
            if rows.is_empty() {
                continue;
            }
            let fraction = sealed.evidence.assignment_seconds * f64::from(self.rate)
                / (sealed.evidence.end - sealed.evidence.start) as f64;
            let mut distance = (sealed.latest_mass - sealed.evidence.retained_mass).abs();
            let old_known: f64 = sealed
                .evidence
                .contexts
                .iter()
                .flatten()
                .map(|c| c.mass)
                .sum();
            let new_known: f64 = rows.iter().map(|r| r.3).sum();
            distance = distance.max(
                ((sealed.latest_mass - new_known) - (sealed.evidence.retained_mass - old_known))
                    .abs(),
            );
            for context in sealed.evidence.contexts.iter().flatten() {
                let mass: f64 = rows
                    .iter()
                    .filter(|r| r.2 == context.context_id)
                    .map(|r| r.3)
                    .sum();
                distance = distance.max((mass - context.mass).abs());
            }
            for row in &rows {
                if !sealed
                    .evidence
                    .contexts
                    .iter()
                    .flatten()
                    .any(|c| c.context_id == row.2)
                {
                    let mass: f64 = rows.iter().filter(|r| r.2 == row.2).map(|r| r.3).sum();
                    distance = distance.max(mass);
                }
            }
            if fraction * distance > 0.25 {
                sealed.revised = true;
                self.revisions += 1;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn different_group_delivery_order_cannot_reorder_original_commitment_time() {
        use crate::temporal_cognition::phrase::tests as fixture;
        let acoustic = fixture::input(1, 0.);
        let mut gesture = fixture::gesture_model();
        gesture.advance(&acoustic, &fixture::ridges(), 100).unwrap();
        let mut phrase = phrase::Phrase::new(1, 0, 0, 1000, 100, fixture::config()).unwrap();
        phrase
            .advance(&acoustic, &gesture, None, None, 100)
            .unwrap();
        let template = phrase.snapshot().groups[0].unwrap();
        let mut ledger = Ledger::new(1000, 8);
        let mut retained = [None; 7];
        for steps in [3, 1, 2] {
            let mut group = template;
            group.group.generation = steps;
            let span = phrase::Foreground {
                id: steps,
                credit: steps,
                start: 0,
                heard_end: steps * 100,
            };
            let mut candidate = group.candidates[0].unwrap();
            candidate.completed_foreground = Some(span);
            candidate.mass = 1.;
            group.candidates.fill(None);
            group.candidates[0] = Some(candidate);
            let mut descriptor = descriptor::Span::new(descriptor::Config {
                group: group.group,
                sample_rate: 1000,
                first_hop_start: 0,
                hop: 100,
                cadence: 2,
                span_start: 0.,
                span_end: None,
                scales: [1.; 10],
                capacity: 128,
            })
            .unwrap();
            for step in 1..=steps {
                let mut raw = fixture::input(step, 0.).features[0].unwrap().raw;
                raw.group = group.group;
                descriptor
                    .push(&raw, &[((step - 1) * 100, step * 100)], step * 100)
                    .unwrap();
            }
            ledger
                .stage(
                    &group,
                    span,
                    descriptor.prefix(steps * 100).unwrap(),
                    steps as f64 / 10.,
                    400,
                    None,
                )
                .unwrap();
            retained[steps as usize - 1] = Some(group);
        }
        ledger.seal(1000, &retained);
        assert_eq!(
            ledger
                .sealed
                .iter()
                .map(|p| (p.evidence.sequence, p.evidence.end))
                .collect::<Vec<_>>(),
            [(1, 100), (2, 200), (3, 300)]
        );
        assert_eq!(
            ledger
                .sealed
                .iter()
                .map(|p| p.evidence.ongoing_credit)
                .collect::<Vec<_>>(),
            [1, 2, 3]
        );
        assert_eq!(ledger.total, 3);
        assert_eq!(ledger.lost, 0);
    }
    use crate::temporal_cognition::phrase::tests as fixture;

    #[test]
    fn sealed_joint_cell_revisions_use_strict_threshold_without_rewriting_support() {
        let a = fixture::input(1, 0.);
        let mut gesture = fixture::gesture_model();
        gesture.advance(&a, &fixture::ridges(), 100).unwrap();
        let mut phrase = phrase::Phrase::new(1, 0, 0, 1000, 100, fixture::config()).unwrap();
        phrase.advance(&a, &gesture, None, None, 100).unwrap();
        let mut g = phrase.snapshot().groups[0].unwrap();
        let mut child = g.candidates[0].unwrap();
        let span = phrase::Foreground {
            id: 11,
            credit: 11,
            start: 0,
            heard_end: 100,
        };
        child.completed_foreground = Some(span);
        child.mass = 1.;
        child.parent_index = 0;
        g.candidates.fill(None);
        g.candidates[0] = Some(child);
        let mut descriptor = descriptor::Span::new(descriptor::Config {
            group: g.group,
            sample_rate: 1000,
            first_hop_start: 0,
            hop: 100,
            cadence: 2,
            span_start: 0.,
            span_end: None,
            scales: [1.; 10],
            capacity: 128,
        })
        .unwrap();
        descriptor
            .push(&a.features[0].unwrap().raw, &[(0, 100)], 100)
            .unwrap();
        let mut ledger = Ledger::new(1000, 2);
        ledger
            .stage(
                &g,
                span,
                descriptor.prefix(100).unwrap(),
                0.1,
                100,
                Some((0, 100)),
            )
            .unwrap();
        ledger.advance(&g, 600, Some((100, 600))).unwrap();
        ledger
            .project(g.group, 600, &[(11, 100, 7, 0.5), (11, 100, 8, 0.5)])
            .unwrap();
        ledger.seal(600, &[Some(g), None, None, None, None, None, None]);
        let bytes = ledger.sealed[0].descriptor.packed();
        ledger.advance(&g, 700, Some((600, 700))).unwrap();
        ledger
            .project(g.group, 700, &[(11, 100, 7, 0.25), (11, 100, 8, 0.75)])
            .unwrap();
        assert_eq!(ledger.revisions, 0);
        ledger
            .project(g.group, 700, &[(11, 100, 7, 0.249), (11, 100, 8, 0.751)])
            .unwrap();
        assert_eq!(ledger.revisions, 1);
        ledger
            .project(g.group, 700, &[(11, 100, 7, 0.1), (11, 100, 8, 0.9)])
            .unwrap();
        assert_eq!(ledger.revisions, 1);
        assert_eq!(ledger.sealed[0].evidence.support, 1.);
        assert_eq!(ledger.sealed[0].evidence.contexts[0].unwrap().mass, 0.5);
        assert_eq!(ledger.sealed[0].descriptor.packed(), bytes);
    }
}
