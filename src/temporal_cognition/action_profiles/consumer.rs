//! Whole-cell prototype diagnostics; descriptor association is not transfer validation.

use super::{CLASSES, Cell, Profiles, Snapshot, Table};
use crate::life::action_candidates::{Class, Input};
use crate::temporal_cognition::{body, ridge::Handle};
use serde::Serialize;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct Key {
    pub lookup_version: u8,
    pub routed_prototypes: u8,
    pub profile_sha256: [u8; 32],
    pub body_model_version: [u8; 32],
    pub bus: u8,
    pub epoch: u64,
    pub issued_at: u64,
    pub assignment_end: u64,
    pub evaluation_delay_samples: Option<u64>,
    pub groups: [Option<Handle>; 8],
}

impl From<&Snapshot> for Key {
    fn from(s: &Snapshot) -> Self {
        Self {
            lookup_version: 3,
            routed_prototypes: s.routed_prototypes,
            profile_sha256: s.profile_sha256,
            body_model_version: s.body_model_version,
            bus: s.bus,
            epoch: s.epoch,
            issued_at: s.issued_at,
            assignment_end: s.assignment_end,
            evaluation_delay_samples: s.evaluation_delay_samples,
            groups: s.groups,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct Binding {
    pub source_id: u64,
    pub source_generation: u32,
    pub body_generation: u32,
    pub bus: u8,
    pub end: u64,
    pub available: u64,
    pub model_version: [u8; 32],
    pub prototype: usize,
    pub distance: f64,
    pub common_coordinates: usize,
}

pub(crate) fn bindings(snapshot: &body::Snapshot) -> impl Iterator<Item = Binding> + '_ {
    snapshot
        .records
        .iter()
        .zip(&snapshot.prototype_assignments)
        .filter_map(|(r, a)| {
            let a = a.as_ref()?;
            (r.active && a.key.1 == 0).then_some(Binding {
                source_id: r.source_id,
                source_generation: r.source_generation,
                body_generation: r.body_generation,
                bus: r.bus,
                end: r.end,
                available: r.available,
                model_version: snapshot.prototype_model_version?,
                prototype: usize::try_from(a.key.0).ok()?,
                distance: a.distance,
                common_coordinates: a.common_coordinates,
            })
        })
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct Entry {
    pub prototype: usize,
    pub class: Class,
    pub group: Handle,
    pub action_at: u64,
    pub evaluation_at: u64,
    pub window_start: u64,
}

impl From<&Cell> for Entry {
    fn from(c: &Cell) -> Self {
        Self {
            prototype: c.prototype,
            class: c.class,
            group: c.group,
            action_at: c.action_at,
            evaluation_at: c.evaluation_at,
            window_start: c.window_start,
        }
    }
}

pub(crate) struct Publication {
    pub key: Key,
    offsets: [u64; 32],
    control_aliases: [u32; 8],
    cells: Box<[Option<Entry>]>,
}

impl Table {
    pub(crate) fn publication(&self, profiles: &Profiles) -> Option<Arc<Publication>> {
        let snapshot = self.snapshot.as_ref().as_ref()?;
        if snapshot.profile_sha256 != profiles.sha256
            || snapshot.body_model_version != profiles.body_model_version
            || profiles
                .profiles
                .iter()
                .enumerate()
                .any(|(i, p)| p.source_routed != (snapshot.routed_prototypes & (1 << i) != 0))
        {
            return None;
        }
        Some(Arc::new(Publication {
            key: Key::from(snapshot),
            offsets: *profiles.offsets(),
            control_aliases: std::array::from_fn(|i| {
                let Some(profile) = profiles.profiles.get(i) else {
                    return 0;
                };
                let baseline = profile.trajectories[4][0];
                if baseline.is_none() || profile.trajectories[3][0] != baseline {
                    return 0;
                }
                (1..32).fold(1, |mask, cell| {
                    mask | (u32::from(profile.trajectories[2][cell] == baseline) << cell)
                })
            }),
            cells: self
                .cells
                .iter()
                .map(|c| c.as_ref().map(Entry::from))
                .collect(),
        }))
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
// Keep bounded whole-cell lookups allocation-free.
#[allow(clippy::large_enum_variant)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum Pair {
    NoTable,
    QueuedDefault,
    CompositeDefault,
    UnroutedDefault,
    UnroutedPrototype,
    NoBodyBinding,
    WrongOwner,
    WrongBus,
    IncompatibleModel,
    FutureBody,
    StaleBody,
    FutureTable,
    StaleTable,
    NoGroup,
    OutsideHorizon,
    UnknownCell,
    Paired {
        candidate_index: usize,
        default_index: usize,
        candidate: Entry,
        body_default: Entry,
    },
}

impl Publication {
    pub(crate) fn bytes(&self) -> usize {
        std::mem::size_of::<Self>() + std::mem::size_of_val(&*self.cells)
    }

    pub(crate) fn pair(
        &self,
        binding: Option<Binding>,
        owner: (u64, u32, Option<u32>),
        bus: usize,
        now: u64,
        candidate: Input,
        body_default: Input,
    ) -> Pair {
        let Some(b) = binding else {
            return Pair::NoBodyBinding;
        };
        if b.source_id != owner.0
            || b.source_generation != owner.1
            || Some(b.body_generation) != owner.2
        {
            return Pair::WrongOwner;
        }
        if bus > 1 || self.key.bus as usize != bus || b.bus as usize != bus {
            return Pair::WrongBus;
        }
        if self.key.lookup_version != 3
            || b.model_version != self.key.body_model_version
            || b.prototype >= self.key.groups.len()
        {
            return Pair::IncompatibleModel;
        }
        if b.end > b.available || b.available > now {
            return Pair::FutureBody;
        }
        if now - b.end >= 24_000 {
            return Pair::StaleBody;
        }
        if self.key.issued_at > now || self.key.assignment_end > self.key.issued_at {
            return Pair::FutureTable;
        }
        if now - self.key.assignment_end >= 24_000 {
            return Pair::StaleTable;
        }
        // A silent acquisition lane cannot describe a routed actual-body action.
        if self.key.routed_prototypes & (1 << b.prototype) == 0 {
            return Pair::UnroutedPrototype;
        }
        let Some(group) = self.key.groups[b.prototype] else {
            return Pair::NoGroup;
        };
        if group.bus != self.key.bus || group.epoch != self.key.epoch {
            return Pair::NoGroup;
        }
        let index = |at: u64| {
            let offset = at.checked_sub(self.key.issued_at)?;
            if offset > self.offsets[31] {
                return None;
            }
            Some(match self.offsets.binary_search(&offset) {
                Ok(i) => i,
                Err(i) if i > 0 && i < 32 => {
                    if offset - self.offsets[i - 1] <= self.offsets[i] - offset {
                        i - 1
                    } else {
                        i
                    }
                }
                _ => return None,
            })
        };
        let (Some(ci), Some(di)) = (index(candidate.at), index(body_default.at)) else {
            return Pair::OutsideHorizon;
        };
        let cell = |input: Input, i: usize| {
            // Class legality uses the Voice decision clock; templates use the shared issue.
            // Change only the acoustic lookup, never the caller's opportunity bookkeeping.
            let projected_class = match (input.class, i) {
                (Class::OnsetNow, 1..) => Class::DelayedOnset,
                (Class::DelayedOnset, 0) => Class::OnsetNow,
                (Class::Continue | Class::Skip, 1..) | (Class::Wait, 0) => {
                    if self.control_aliases[b.prototype] & (1 << i) == 0 {
                        return None;
                    }
                    if i == 0 { Class::Skip } else { Class::Wait }
                }
                _ => input.class,
            };
            let class = CLASSES.iter().position(|c| *c == projected_class)?;
            let entry = self
                .cells
                .get((b.prototype * 7 + class) * 32 + i)?
                .as_ref()?;
            (entry.prototype == b.prototype
                && entry.class == projected_class
                && entry.group == group
                && entry.action_at == self.key.issued_at.checked_add(self.offsets[i])?)
            .then_some(*entry)
        };
        let (Some(candidate), Some(body_default)) = (cell(candidate, ci), cell(body_default, di))
        else {
            return Pair::UnknownCell;
        };
        Pair::Paired {
            candidate_index: ci,
            default_index: di,
            candidate,
            body_default,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::life::action_candidates::BodyState;

    fn fixture() -> (Publication, Binding, Input) {
        let group = Handle {
            bus: 0,
            epoch: 3,
            generation: 9,
        };
        let offsets = std::array::from_fn(|i| (i as u64 * 192000 + 15) / 31);
        let key = Key {
            lookup_version: 3,
            routed_prototypes: 1,
            profile_sha256: [1; 32],
            body_model_version: [2; 32],
            bus: 0,
            epoch: 3,
            issued_at: 1000,
            assignment_end: 1000,
            evaluation_delay_samples: None,
            groups: [Some(group), None, None, None, None, None, None, None],
        };
        let mut cells = vec![None; 8 * 7 * 32];
        for (ci, class) in CLASSES.into_iter().enumerate() {
            for (i, offset) in offsets.into_iter().enumerate() {
                cells[ci * 32 + i] = Some(Entry {
                    prototype: 0,
                    class,
                    group,
                    action_at: 1000 + offset,
                    evaluation_at: 1000 + offset,
                    window_start: 0,
                });
            }
        }
        let binding = Binding {
            source_id: 7,
            source_generation: 2,
            body_generation: 4,
            bus: 0,
            end: 1000,
            available: 1000,
            model_version: [2; 32],
            prototype: 0,
            distance: 0.1,
            common_coordinates: 6,
        };
        let input = Class::OnsetNow
            .input(
                1000,
                1000,
                Some(24000),
                48000,
                BodyState {
                    permits_action: Some(true),
                    active_at_candidate: Some(true),
                    pending_opportunity: true,
                    due_unconsumed: true,
                },
            )
            .unwrap();
        (
            Publication {
                key,
                offsets,
                control_aliases: [u32::MAX; 8],
                cells: cells.into_boxed_slice(),
            },
            binding,
            input,
        )
    }

    #[test]
    fn candidate_worker_keeps_actual_and_prototype_routing_separate() {
        use crate::life::action_candidates::energy::{ScheduledRequest, Worker};
        use crate::life::self_prediction::ToneEnergy;
        use crate::life::sound::envelope::Envelope;
        for bus in 0..2 {
            for prototype_routed in [false, true] {
                for actual_routed in [false, true] {
                    let (mut table, mut binding, _) = fixture();
                    table.key.bus = bus as u8;
                    table.key.groups[0].as_mut().unwrap().bus = bus as u8;
                    table.key.routed_prototypes = u8::from(prototype_routed);
                    for cell in table.cells.iter_mut().flatten() {
                        cell.group.bus = bus as u8;
                    }
                    binding.bus = bus as u8;
                    let mut worker = Worker::new();
                    let mut packet = worker.acquire().unwrap();
                    packet.scheduled = Some(ScheduledRequest {
                        source_id: 7,
                        source_generation: 2,
                        body_generation: 4,
                        issued_at: 1000,
                        sample_rate: 48000,
                        hop: 512,
                        period: Some(24000),
                    });
                    packet.shared[bus] = Some(Arc::new(table));
                    packet.bindings[bus] = Some(binding);
                    packet.retained.push((
                        1,
                        [actual_routed; 2],
                        ToneEnergy {
                            amplitude: 0.1,
                            control: None,
                            sine: None,
                            bank: None,
                            scheduled_release: None,
                            envelope: Envelope {
                                onset: 0,
                                hold_end: 200000,
                                release_end: 200000,
                                attack_ticks: 0,
                                decay_ticks: 0,
                                sustain_level: 1.,
                                decay_lambda: 0.,
                                release_ticks: 0,
                            },
                        },
                    ));
                    worker.submit(packet, true);
                    worker.finish();
                    let records: Vec<_> = worker.drain().collect();
                    assert_eq!(records.len(), 1);
                    assert_eq!(records[0].default_routed[bus], actual_routed);
                    for c in &records[0].candidates {
                        assert!(matches!(c.shared[1 - bus], Pair::NoTable));
                        assert!(match c.shared[bus] {
                            Pair::UnroutedDefault => !actual_routed,
                            Pair::UnroutedPrototype => actual_routed && !prototype_routed,
                            Pair::Paired { .. } => actual_routed && prototype_routed,
                            _ => false,
                        });
                    }
                }
            }
        }
        let (mut table, binding, input) = fixture();
        table.key.routed_prototypes = 1 << 7;
        assert!(matches!(
            table.pair(Some(binding), (7, 2, Some(4)), 0, 1000, input, input),
            Pair::UnroutedPrototype
        ));
        table.key.lookup_version = 2;
        assert!(matches!(
            table.pair(Some(binding), (7, 2, Some(4)), 0, 1000, input, input),
            Pair::IncompatibleModel
        ));
    }

    #[test]
    fn queued_schedule_cannot_consume_a_fresh_excitation_profile() {
        use crate::life::action_candidates::energy::{ScheduledRequest, Worker};
        use crate::life::self_prediction::ToneEnergy;
        use crate::life::sound::envelope::Envelope;
        let (table, _, _) = fixture();
        let mut worker = Worker::new();
        let mut packet = worker.acquire().unwrap();
        packet.scheduled = Some(ScheduledRequest {
            source_id: 7,
            source_generation: 2,
            body_generation: 4,
            issued_at: 1000,
            sample_rate: 48000,
            hop: 512,
            period: Some(24000),
        });
        packet.shared[0] = Some(Arc::new(table));
        packet.retained.push((
            1,
            [true, false],
            ToneEnergy {
                amplitude: 0.1,
                control: None,
                sine: None,
                bank: None,
                scheduled_release: None,
                envelope: Envelope {
                    onset: 2000,
                    hold_end: 8000,
                    release_end: 8000,
                    attack_ticks: 0,
                    decay_ticks: 0,
                    sustain_level: 1.,
                    decay_lambda: 0.,
                    release_ticks: 0,
                },
            },
        ));
        worker.submit(packet, true);
        worker.finish();
        let records: Vec<_> = worker.drain().collect();
        assert_eq!(records.len(), 1);
        assert!(
            records[0]
                .candidates
                .iter()
                .all(|c| matches!(c.shared[0], Pair::QueuedDefault)
                    && matches!(c.shared[1], Pair::NoTable))
        );
    }

    #[test]
    fn decision_classes_map_to_the_shared_clock_without_changing_inputs() {
        let (mut table, binding, original) = fixture();
        // Match the real sparse class layout instead of the exhaustive lookup fixture.
        for (ci, class) in CLASSES.into_iter().enumerate() {
            for i in 0..32 {
                if matches!(class, Class::OnsetNow | Class::Skip | Class::Continue) && i > 0
                    || matches!(class, Class::DelayedOnset | Class::Wait) && i == 0
                {
                    table.cells[ci * 32 + i] = None;
                }
            }
        }
        let now = 1000 + table.offsets[1];
        let body = BodyState {
            permits_action: Some(true),
            active_at_candidate: Some(true),
            pending_opportunity: true,
            due_unconsumed: true,
        };
        let default = Class::Continue
            .input(now, now, Some(24000), 48000, body)
            .unwrap();
        for class in [
            Class::OnsetNow,
            Class::Continue,
            Class::Skip,
            Class::Release,
            Class::Gap,
        ] {
            let candidate = class.input(now, now, Some(24000), 48000, body).unwrap();
            let Pair::Paired {
                candidate: selected,
                body_default,
                candidate_index,
                default_index,
            } = table.pair(Some(binding), (7, 2, Some(4)), 0, now, candidate, default)
            else {
                panic!("shifted {class:?}")
            };
            assert_eq!((candidate_index, default_index), (1, 1));
            assert_eq!(body_default.class, Class::Wait);
            assert_eq!(
                selected.class,
                match class {
                    Class::OnsetNow => Class::DelayedOnset,
                    Class::Continue | Class::Skip => Class::Wait,
                    _ => class,
                }
            );
            assert_eq!(candidate.class, class);
            assert_eq!(candidate.consumes_due_opportunity, class == Class::Skip);
            assert_eq!(candidate.excitation_at.is_some(), class == Class::OnsetNow);
        }
        let delayed = Class::DelayedOnset
            .input(1000, 3400, Some(24000), 48000, body)
            .unwrap();
        let wait = Class::Wait
            .input(1000, 3400, Some(24000), 48000, body)
            .unwrap();
        for (candidate, expected) in [(delayed, Class::OnsetNow), (wait, Class::Skip)] {
            let Pair::Paired {
                candidate: selected,
                candidate_index,
                ..
            } = table.pair(Some(binding), (7, 2, Some(4)), 0, 1000, candidate, original)
            else {
                panic!("first-cell mapping")
            };
            assert_eq!(candidate_index, 0);
            assert_eq!(selected.class, expected);
        }
        let release = Class::Release
            .input(now, 1000 + table.offsets[3], Some(24000), 48000, body)
            .unwrap();
        let Pair::Paired { candidate, .. } =
            table.pair(Some(binding), (7, 2, Some(4)), 0, now, release, default)
        else {
            panic!("supported shifted pair")
        };
        assert_eq!(candidate.action_at, 1000 + table.offsets[3]);
    }

    #[test]
    fn clock_mapping_requires_control_identity_and_the_selected_whole_cell() {
        let (mut table, binding, original) = fixture();
        let at = 1000 + table.offsets[1];
        let default = Input {
            class: Class::Continue,
            issued_at: at,
            at,
            excitation_at: None,
            ..original
        };
        table.control_aliases[0] &= !(1 << 1);
        assert!(matches!(
            table.pair(Some(binding), (7, 2, Some(4)), 0, at, default, default),
            Pair::UnknownCell
        ));
        table.control_aliases[0] |= 1 << 1;
        table.cells[2 * 32 + 1] = None;
        assert!(matches!(
            table.pair(Some(binding), (7, 2, Some(4)), 0, at, default, default),
            Pair::UnknownCell
        ));
        // An original-class cell or supported neighbor cannot repair the selected mapped cell.
        assert!(table.cells[4 * 32 + 1].is_some() && table.cells[2 * 32 + 2].is_some());
        let onset = Input {
            issued_at: at,
            at,
            excitation_at: Some(at),
            ..original
        };
        table.cells[32 + 1] = None;
        assert!(matches!(
            table.pair(Some(binding), (7, 2, Some(4)), 0, at, onset, onset),
            Pair::UnknownCell
        ));
        table.key.lookup_version = 1;
        assert!(matches!(
            table.pair(Some(binding), (7, 2, Some(4)), 0, 1000, original, original),
            Pair::IncompatibleModel
        ));
    }

    #[test]
    fn lookup_keeps_whole_cells_ties_unknowns_and_default_support() {
        let (mut table, binding, default) = fixture();
        assert!(table.bytes() < 2 * 1024 * 1024);
        for i in 0..31 {
            let a = table.offsets[i];
            let b = table.offsets[i + 1];
            for off in [(a + b) / 2, (a + b) / 2 + 1] {
                let expected = if off - a <= b - off { i } else { i + 1 };
                let candidate = Input {
                    class: Class::DelayedOnset,
                    at: 1000 + off,
                    ..default
                };
                let Pair::Paired {
                    candidate_index,
                    candidate: cell,
                    ..
                } = table.pair(Some(binding), (7, 2, Some(4)), 0, 1000, candidate, default)
                else {
                    panic!("supported cell pair")
                };
                assert_eq!(candidate_index, expected);
                assert_eq!(cell.action_at, 1000 + table.offsets[expected]);
            }
        }
        let candidate = Input {
            class: Class::DelayedOnset,
            at: 1000 + table.offsets[2],
            ..default
        };
        table.cells[32 + 2] = None;
        assert!(matches!(
            table.pair(Some(binding), (7, 2, Some(4)), 0, 1000, candidate, default),
            Pair::UnknownCell
        ));
        table.cells[32 + 2] = table.cells[32 + 3];
        assert!(matches!(
            table.pair(Some(binding), (7, 2, Some(4)), 0, 1000, candidate, default),
            Pair::UnknownCell
        ));
        let candidate = Input {
            at: 1000 + table.offsets[3],
            ..candidate
        };
        table.cells[0] = None;
        assert!(matches!(
            table.pair(Some(binding), (7, 2, Some(4)), 0, 1000, candidate, default),
            Pair::UnknownCell
        ));
    }

    #[test]
    fn body_clock_bus_model_and_group_gates_cannot_rebind_a_cell() {
        let (mut table, b, default) = fixture();
        let pair =
            |t: &Publication, b, owner, bus, now| t.pair(b, owner, bus, now, default, default);
        assert!(matches!(
            pair(&table, None, (7, 2, Some(4)), 0, 1000),
            Pair::NoBodyBinding
        ));
        assert!(matches!(
            pair(&table, Some(b), (7, 3, Some(4)), 0, 1000),
            Pair::WrongOwner
        ));
        assert!(matches!(
            pair(&table, Some(b), (7, 2, Some(5)), 0, 1000),
            Pair::WrongOwner
        ));
        assert!(matches!(
            pair(&table, Some(b), (7, 2, Some(4)), 1, 1000),
            Pair::WrongBus
        ));
        assert!(matches!(
            pair(
                &table,
                Some(Binding {
                    model_version: [3; 32],
                    ..b
                }),
                (7, 2, Some(4)),
                0,
                1000
            ),
            Pair::IncompatibleModel
        ));
        assert!(matches!(
            pair(
                &table,
                Some(Binding {
                    available: 1001,
                    ..b
                }),
                (7, 2, Some(4)),
                0,
                1000
            ),
            Pair::FutureBody
        ));
        assert!(matches!(
            pair(&table, Some(b), (7, 2, Some(4)), 0, 25000),
            Pair::StaleBody
        ));
        assert!(matches!(
            pair(
                &table,
                Some(Binding {
                    end: 999,
                    available: 999,
                    ..b
                }),
                (7, 2, Some(4)),
                0,
                999
            ),
            Pair::FutureTable
        ));
        assert!(matches!(
            pair(
                &table,
                Some(Binding {
                    end: 25000,
                    available: 25000,
                    ..b
                }),
                (7, 2, Some(4)),
                0,
                25000
            ),
            Pair::StaleTable
        ));
        let outside = Input {
            at: 193001,
            ..default
        };
        assert!(matches!(
            table.pair(Some(b), (7, 2, Some(4)), 0, 1000, outside, default),
            Pair::OutsideHorizon
        ));
        table.key.groups[0].as_mut().unwrap().generation += 1;
        assert!(matches!(
            pair(&table, Some(b), (7, 2, Some(4)), 0, 1000),
            Pair::UnknownCell
        ));
        table.key.groups[0] = None;
        assert!(matches!(
            pair(&table, Some(b), (7, 2, Some(4)), 0, 1000),
            Pair::NoGroup
        ));
    }
}
