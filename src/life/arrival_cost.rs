//! I11-2 arrival cost over one fixed set of eligible habitat groups.

use crate::temporal_cognition::{
    Handle, arrival::Payload, body::Record, body_model::Binding, observation::Snapshot,
    proposals::frontend::recurrence::GroupSnapshot,
};

pub(crate) struct ArrivalSet<'a> {
    groups: [Option<&'a Payload>; 7],
    count: usize,
    state: &'static str,
    exclusions: [Option<&'static str>; 7],
}

#[derive(Clone, Copy)]
pub(crate) struct ArrivalContext<'a> {
    pub snapshot: &'a Snapshot,
    pub self_group: Option<Handle>,
    pub self_group_state: &'static str,
    pub binding: Option<Binding>,
    pub habitat_routed: bool,
    pub body_record: Option<Record>,
    pub owner: (u64, u32, Option<u32>),
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct ArrivalProvenance {
    pub snapshot_frame_id: Option<u64>,
    pub snapshot_available_sample: u64,
    pub snapshot_source_epoch: u64,
    pub snapshot_bus: u8,
    pub snapshot_sample_rate: u32,
    pub parameters: Option<crate::config::TemporalPeriodConfig>,
    pub period_end_sample: Option<u64>,
    pub period_received_at: Option<u64>,
    pub groups: Vec<Option<GroupSnapshot>>,
    pub selected: [bool; 7],
    pub exclusions: [Option<&'static str>; 7],
    pub owner: (u64, u32, Option<u32>),
    pub binding: Option<Binding>,
    pub habitat_routed: bool,
    pub body_record: Option<Record>,
    pub assignment: Option<AssignmentProvenance>,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct AssignmentProvenance {
    pub model_version: [u8; 32],
    pub bus: u8,
    pub epoch: u64,
    pub end_sample: u64,
    pub assignments: [Option<crate::temporal_cognition::body_model::Assignment>; 8],
}

impl ArrivalProvenance {
    pub(crate) fn capture(context: &ArrivalContext<'_>, set: &ArrivalSet<'_>) -> Box<Self> {
        let snapshot = context.snapshot;
        let period = snapshot.period.as_ref();
        Box::new(Self {
            snapshot_frame_id: snapshot.frame_id,
            snapshot_available_sample: snapshot.available_sample,
            snapshot_source_epoch: snapshot.source_epoch,
            snapshot_bus: snapshot.bus,
            snapshot_sample_rate: snapshot.sample_rate,
            parameters: snapshot.period_parameters,
            period_end_sample: period.map(|period| period.end_sample),
            period_received_at: period.map(|period| period.received_at),
            groups: period.map_or_else(|| vec![None; 7], |period| period.groups.to_vec()),
            selected: std::array::from_fn(|index| set.groups[index].is_some()),
            exclusions: set.exclusions,
            owner: context.owner,
            binding: context.binding,
            habitat_routed: context.habitat_routed,
            body_record: context.body_record,
            assignment: snapshot
                .group_prototypes
                .as_ref()
                .map(|shared| AssignmentProvenance {
                    model_version: shared.model_version,
                    bus: shared.bus,
                    epoch: shared.epoch,
                    end_sample: shared.end_sample,
                    assignments: shared.assignments,
                }),
        })
    }
}

impl<'a> ArrivalSet<'a> {
    pub(crate) fn select(
        snapshot: &'a Snapshot,
        now: u64,
        candidates: &[Option<f64>; 23],
        width: f64,
        self_group: Option<Handle>,
        sample_rate: u32,
    ) -> Self {
        let mut result = Self {
            groups: [None; 7],
            count: 0,
            state: "no_eligible_group",
            exclusions: [None; 7],
        };
        if snapshot.bus != 0
            || sample_rate == 0
            || snapshot.sample_rate != sample_rate
            || !width.is_finite()
            || width < 0.0
        {
            result.state = "unsupported_snapshot";
            return result;
        }
        let Some(parameters) = snapshot.period_parameters else {
            result.state = "parameters_absent";
            return result;
        };
        let Some(period) = snapshot.period.as_ref() else {
            result.state = "period_absent";
            return result;
        };
        if period.end_sample > period.received_at || period.received_at > now {
            result.state = "period_future";
            return result;
        }
        for (index, group) in period.groups.iter().enumerate() {
            let Some(group) = group.as_ref() else {
                continue;
            };
            if !group.active {
                result.exclusions[index] = Some("inactive");
                continue;
            }
            let Some(forecast) = group.forecast else {
                result.exclusions[index] =
                    Some(group.arrival_unavailable.unwrap_or("forecast_unavailable"));
                continue;
            };
            let Some(arrival) = group.arrival.as_ref() else {
                result.exclusions[index] =
                    Some(group.arrival_unavailable.unwrap_or("arrival_unavailable"));
                continue;
            };
            let reason = if arrival.group != group.ledger.group
                || arrival.group.epoch != snapshot.source_epoch
                || arrival.group.bus != 0
                || arrival.issued_at != forecast.issued_at
                || arrival.horizon_end != forecast.horizon_end
                || forecast.model != parameters.model
            {
                Some("identity_mismatch")
            } else if self_group == Some(arrival.group) {
                Some("self_group")
            } else if !arrival.matches(forecast, sample_rate) {
                Some("payload_mismatch")
            } else if arrival.issued_at > now {
                Some("forecast_future")
            } else if now >= arrival.horizon_end {
                Some("horizon_expired")
            } else if !forecast.valid_for(arrival.group, forecast.model, now) {
                Some("forecast_invalid")
            } else if candidates.iter().flatten().any(|at| {
                !at.is_finite()
                    || *at < now as f64
                    || *at + width > arrival.horizon_end as f64
                    || arrival.window_probability(*at, width).is_none()
            }) {
                Some("candidate_window_invalid")
            } else {
                None
            };
            if let Some(reason) = reason {
                result.exclusions[index] = Some(reason);
                continue;
            }
            result.groups[index] = Some(arrival);
            result.count += 1;
        }
        if result.count > 0 {
            result.state = "known";
        }
        result
    }

    pub(crate) fn probability(&self, at: f64, width: f64) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        let mut sum = 0.0;
        for arrival in self.groups.iter().flatten() {
            sum += arrival.window_probability(at, width)?;
        }
        let mean = sum / self.count as f64;
        (mean.is_finite() && (0.0..=1.0).contains(&mean)).then_some(mean)
    }

    pub(crate) fn len(&self) -> usize {
        self.count
    }

    pub(crate) fn state(&self) -> &'static str {
        self.state
    }

    pub(crate) fn exclusions(&self) -> [Option<&'static str>; 7] {
        self.exclusions
    }
}

pub(crate) fn self_group(
    snapshot: &Snapshot,
    binding: Option<Binding>,
    habitat_routed: bool,
    body_record: Option<Record>,
    owner: (u64, u32, Option<u32>),
    now: u64,
) -> Result<Handle, &'static str> {
    if !habitat_routed {
        return Err("habitat_unrouted");
    }
    let binding = binding.ok_or("binding_absent")?;
    let record = body_record.ok_or("body_record_absent")?;
    if !record.active
        || record.bus != 0
        || record.source_id != owner.0
        || record.source_generation != owner.1
        || Some(record.body_generation) != owner.2
        || record.source_id != binding.source_id
        || record.source_generation != binding.source_generation
        || record.body_generation != binding.body_generation
        || record.bus != binding.bus
        || record.end != binding.end
        || record.available != binding.available
    {
        return Err("body_record_mismatch");
    }
    if record.start > record.end
        || record.source_start > record.end
        || record.end > record.available
        || record.available > now
    {
        return Err("body_record_future");
    }
    if record.mask & 0b11 != 0b11
        || !record.coverage[0].is_finite()
        || !record.coverage[1].is_finite()
        || record.coverage[0] < 0.9
        || record.coverage[1] < 0.9
    {
        return Err("body_record_no_spectral_mass");
    }
    let shared = snapshot
        .group_prototypes
        .as_ref()
        .ok_or("assignment_absent")?;
    let period = snapshot.period.as_ref().ok_or("period_absent")?;
    if period.end_sample > period.received_at || period.received_at > now {
        return Err("period_future");
    }
    if binding.source_id != owner.0
        || binding.source_generation != owner.1
        || Some(binding.body_generation) != owner.2
        || binding.bus != 0
        || snapshot.bus != 0
        || shared.bus != 0
        || shared.epoch != snapshot.source_epoch
        || shared.model_version != binding.model_version
        || binding.prototype >= shared.assignments.len()
    {
        return Err("identity_mismatch");
    }
    if binding.end > binding.available
        || shared.end_sample > period.received_at
        || binding.available > now
    {
        return Err("assignment_future");
    }
    let freshness_samples = u64::from(snapshot.sample_rate).div_ceil(2);
    if freshness_samples == 0
        || now.checked_sub(binding.end).ok_or("assignment_future")? >= freshness_samples
        || now
            .checked_sub(shared.end_sample)
            .ok_or("assignment_future")?
            >= freshness_samples
    {
        return Err("assignment_stale");
    }
    let assignment = shared.assignments[binding.prototype].ok_or("assignment_unmapped")?;
    let group = Handle {
        bus: shared.bus,
        epoch: assignment.key.0,
        generation: assignment.key.1,
    };
    if group.epoch != shared.epoch {
        return Err("identity_mismatch");
    }
    let group_snapshot = period
        .groups
        .iter()
        .flatten()
        .find(|g| g.active && g.ledger.group == group)
        .ok_or("group_absent")?;
    let arrival = group_snapshot.arrival.as_ref().ok_or("arrival_absent")?;
    if arrival.group != group
        || shared.end_sample > arrival.issued_at
        || binding.available > arrival.issued_at
    {
        return Err("assignment_after_arrival");
    }
    Ok(group)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::config::{ArrivalModel, TemporalPeriodConfig};
    use crate::temporal_cognition::{
        AccentSummary,
        arrival::Forecast,
        body_model::{Assignment, Shared},
        proposals::frontend::recurrence::{GroupSnapshot, Snapshot as PeriodSnapshot},
    };

    pub(crate) fn fixture() -> Snapshot {
        let group = Handle {
            bus: 0,
            epoch: 1,
            generation: 2,
        };
        let summary = AccentSummary {
            group,
            received_at: 0,
            retained_accents: 1,
            cumulative_count: 1,
            cumulative_weight: 1.0,
            capacity_evicted_through: None,
        };
        let forecast = Forecast {
            model: ArrivalModel::Periodic,
            version: 1,
            group,
            source_start: 0,
            source_end: 0,
            available: 0,
            issued_at: 0,
            horizon_end: 192_000,
            last_accent: 0,
            elapsed_seconds: [0.0; 2],
            reset_unknown: false,
            probability: None,
            evaluations: 0,
        };
        let mut groups = [None; 7];
        groups[0] = Some(GroupSnapshot {
            ledger: summary,
            active: true,
            acoustic_eligible: true,
            association_known: true,
            peaks: [None; 8],
            period_source: None,
            forecast: Some(forecast),
            arrival_unavailable: None,
            arrival: Some(Payload {
                group,
                issued_at: 0,
                horizon_end: 192_000,
                sample_rate: 48_000,
                model: ArrivalModel::Periodic,
                version: 1,
                source_start: 0,
                source_end: 0,
                available: 0,
                last_accent: 0,
                period_seconds: 0.6,
                next_at: 28_800.,
            }),
        });
        Snapshot {
            bus: 0,
            source_epoch: 1,
            sample_rate: 48_000,
            period_parameters: Some(TemporalPeriodConfig {
                model: ArrivalModel::Periodic,
                horizon_sec: 4.0,
            }),
            period: Some(PeriodSnapshot {
                end_sample: 0,
                received_at: 0,
                groups,
                residual: summary,
            }),
            ..Snapshot::default()
        }
    }

    #[test]
    fn one_group_set_is_fixed_across_candidates_and_zero_is_known() {
        let snapshot = fixture();
        let mut candidates = [None; 23];
        candidates[2] = Some(24_000.0);
        candidates[6] = Some(28_800.0);
        let selected = ArrivalSet::select(&snapshot, 0, &candidates, 2880.0, None, 48_000);
        assert_eq!(selected.len(), 1);
        assert_eq!(selected.probability(24_000.0, 2880.0), Some(0.0));
        assert_eq!(selected.probability(28_800.0, 2880.0), Some(1.0));
        candidates[22] = Some(191_000.0);
        let excluded = ArrivalSet::select(&snapshot, 0, &candidates, 2880.0, None, 48_000);
        assert_eq!(excluded.len(), 0);
        assert_eq!(excluded.state(), "no_eligible_group");
        assert_eq!(excluded.exclusions()[0], Some("candidate_window_invalid"));
        assert_eq!(
            ArrivalSet::select(&snapshot, 192_000, &candidates, 0.0, None, 48_000).len(),
            0
        );
        candidates[22] = None;
        let group = snapshot.period.unwrap().groups[0].unwrap().ledger.group;
        let excluded = ArrivalSet::select(&snapshot, 0, &candidates, 2880.0, Some(group), 48_000);
        assert_eq!(excluded.len(), 0);
        assert_eq!(excluded.exclusions()[0], Some("self_group"));
    }

    #[test]
    fn actual_rate_and_explicit_horizon_replace_old_fixed_conditions() {
        for rate in [44_100, 48_000, 96_000] {
            let mut snapshot = fixture();
            snapshot.sample_rate = rate;
            snapshot.period_parameters.as_mut().unwrap().horizon_sec = 1.25;
            let group = snapshot.period.as_mut().unwrap().groups[0]
                .as_mut()
                .unwrap();
            let end = (1.25 * f64::from(rate)) as u64;
            group.forecast.as_mut().unwrap().horizon_end = end;
            let payload = group.arrival.as_mut().unwrap();
            payload.horizon_end = end;
            payload.sample_rate = rate;
            payload.next_at = 0.6 * f64::from(rate);
            let mut candidates = [None; 23];
            candidates[2] = Some(0.5 * f64::from(rate));
            candidates[4] = Some(payload.next_at);
            let selected = ArrivalSet::select(
                &snapshot,
                0,
                &candidates,
                0.06 * f64::from(rate),
                None,
                rate,
            );
            assert_eq!(selected.len(), 1);
            assert_eq!(
                selected.probability(candidates[2].unwrap(), 0.06 * f64::from(rate)),
                Some(0.)
            );
            assert_eq!(
                selected.probability(candidates[4].unwrap(), 0.06 * f64::from(rate)),
                Some(1.)
            );
            assert_eq!(
                ArrivalSet::select(&snapshot, 0, &candidates, 1., None, rate + 1).state(),
                "unsupported_snapshot"
            );
            let period = snapshot.period.as_mut().unwrap();
            let mut foreign = period.groups[0].unwrap();
            foreign.ledger.group.generation += 1;
            period.groups[1] = Some(foreign);
            period.groups[0]
                .as_mut()
                .unwrap()
                .forecast
                .as_mut()
                .unwrap()
                .available = 1;
            let invalid = ArrivalSet::select(&snapshot, 0, &candidates, 1., None, rate);
            assert_eq!(invalid.len(), 0);
            assert_eq!(invalid.exclusions()[0], Some("payload_mismatch"));
            assert_eq!(invalid.exclusions()[1], Some("identity_mismatch"));
            assert_eq!(invalid.probability(0., 1.), None);
        }
    }

    #[test]
    fn producer_failure_reason_reaches_exclusions_and_remains_unknown() {
        let mut snapshot = fixture();
        let group = snapshot.period.as_mut().unwrap().groups[0]
            .as_mut()
            .unwrap();
        group.arrival = None;
        group.arrival_unavailable = Some("arrival_elapsed_uncertain");
        let mut candidates = [None; 23];
        candidates[2] = Some(24_000.0);
        let selected = ArrivalSet::select(&snapshot, 0, &candidates, 2880.0, None, 48_000);
        assert_eq!(selected.len(), 0);
        assert_eq!(selected.exclusions()[0], Some("arrival_elapsed_uncertain"));
        assert_eq!(selected.probability(24_000.0, 2880.0), None);
        let group = snapshot.period.as_mut().unwrap().groups[0]
            .as_mut()
            .unwrap();
        group.forecast = None;
        group.arrival_unavailable = Some("arrival_forecast_absent");
        let selected = ArrivalSet::select(&snapshot, 0, &candidates, 2880.0, None, 48_000);
        assert_eq!(selected.exclusions()[0], Some("arrival_forecast_absent"));
        assert_eq!(selected.probability(24_000.0, 2880.0), None);
    }

    #[test]
    fn self_group_requires_real_matching_binding_and_fresh_assignment() {
        let mut snapshot = fixture();
        let period = snapshot.period.as_mut().unwrap();
        period.end_sample = 100;
        period.received_at = 100;
        let group = period.groups[0].as_mut().unwrap();
        group.forecast.as_mut().unwrap().issued_at = 100;
        group.arrival.as_mut().unwrap().issued_at = 100;
        let mut assignments = [None; 8];
        assignments[0] = Some(Assignment {
            key: (1, 2),
            distance: 0.0,
            common_coordinates: 6,
        });
        snapshot.group_prototypes = Some(Shared {
            model_version: [7; 32],
            bus: 0,
            epoch: 1,
            end_sample: 90,
            descriptors: [None; 7],
            assignments,
        });
        let binding = Binding {
            source_id: 9,
            source_generation: 1,
            body_generation: 2,
            bus: 0,
            end: 90,
            available: 90,
            model_version: [7; 32],
            prototype: 0,
            distance: 0.0,
            common_coordinates: 6,
        };
        let record = Record {
            source_id: 9,
            source_generation: 1,
            body_generation: 2,
            bus: 0,
            start: 10,
            end: 90,
            source_start: 10,
            available: 90,
            mask: 0b11,
            coverage: [1.0; 6],
            active: true,
            ..Record::default()
        };
        assert_eq!(
            self_group(
                &snapshot,
                Some(binding),
                true,
                Some(record),
                (9, 1, Some(2)),
                100
            ),
            Ok(Handle {
                bus: 0,
                epoch: 1,
                generation: 2
            })
        );
        assert_eq!(
            self_group(
                &snapshot,
                Some(binding),
                true,
                Some(record),
                (9, 1, None),
                100
            ),
            Err("body_record_mismatch")
        );
        assert_eq!(
            self_group(
                &snapshot,
                Some(binding),
                true,
                Some(record),
                (9, 1, Some(2)),
                24_090
            ),
            Err("assignment_stale")
        );
        for rate in [44_101, 48_000, 96_000] {
            snapshot.sample_rate = rate;
            let expiry = 90 + u64::from(rate).div_ceil(2);
            assert!(
                self_group(
                    &snapshot,
                    Some(binding),
                    true,
                    Some(record),
                    (9, 1, Some(2)),
                    expiry - 1
                )
                .is_ok()
            );
            assert_eq!(
                self_group(
                    &snapshot,
                    Some(binding),
                    true,
                    Some(record),
                    (9, 1, Some(2)),
                    expiry
                ),
                Err("assignment_stale")
            );
        }
        snapshot.sample_rate = 48_000;
        snapshot.group_prototypes.as_mut().unwrap().assignments[0] = None;
        assert_eq!(
            self_group(
                &snapshot,
                Some(binding),
                true,
                Some(record),
                (9, 1, Some(2)),
                100
            ),
            Err("assignment_unmapped")
        );
        snapshot.period.as_mut().unwrap().received_at = 101;
        assert_eq!(
            self_group(
                &snapshot,
                Some(binding),
                true,
                Some(record),
                (9, 1, Some(2)),
                100
            ),
            Err("period_future")
        );
    }

    #[test]
    fn silent_body_record_cannot_exclude_its_assigned_habitat_group() {
        let mut snapshot = fixture();
        let period = snapshot.period.as_mut().unwrap();
        period.end_sample = 100;
        period.received_at = 100;
        let group = period.groups[0].as_mut().unwrap();
        group.forecast.as_mut().unwrap().issued_at = 100;
        group.arrival.as_mut().unwrap().issued_at = 100;
        let mut assignments = [None; 8];
        assignments[2] = Some(Assignment {
            key: (1, 2),
            distance: 0.0,
            common_coordinates: 3,
        });
        snapshot.group_prototypes = Some(Shared {
            model_version: [7; 32],
            bus: 0,
            epoch: 1,
            end_sample: 90,
            descriptors: [None; 7],
            assignments,
        });
        let binding = Binding {
            source_id: 9,
            source_generation: 1,
            body_generation: 2,
            bus: 0,
            end: 90,
            available: 90,
            model_version: [7; 32],
            prototype: 2,
            distance: 0.0,
            common_coordinates: 3,
        };
        let silent = Record {
            source_id: 9,
            source_generation: 1,
            body_generation: 2,
            bus: 0,
            start: 10,
            end: 90,
            source_start: 10,
            available: 90,
            raw_values: [0.0, 0.0, -19.9315685693, 0.0, 0.0, -0.0],
            mask: 44,
            active: true,
            ..Record::default()
        };
        let owner = (9, 1, Some(2));
        assert_eq!(
            self_group(&snapshot, Some(binding), false, Some(silent), owner, 100),
            Err("habitat_unrouted")
        );
        assert_eq!(
            self_group(&snapshot, Some(binding), true, Some(silent), owner, 100),
            Err("body_record_no_spectral_mass")
        );
        let mut spectral = Record {
            mask: 0b11,
            coverage: [1.0; 6],
            ..silent
        };
        assert_eq!(
            self_group(&snapshot, Some(binding), true, Some(spectral), owner, 100),
            Ok(Handle {
                bus: 0,
                epoch: 1,
                generation: 2
            })
        );
        spectral.coverage[0] = 0.0;
        assert_eq!(
            self_group(&snapshot, Some(binding), true, Some(spectral), owner, 100),
            Err("body_record_no_spectral_mass")
        );
        spectral.coverage[0] = 1.0;
        spectral.available = 101;
        assert_eq!(
            self_group(&snapshot, Some(binding), true, Some(spectral), owner, 100),
            Err("body_record_mismatch")
        );
        spectral.available = 90;
        spectral.source_generation = 2;
        assert_eq!(
            self_group(&snapshot, Some(binding), true, Some(spectral), owner, 100),
            Err("body_record_mismatch")
        );
        spectral.source_generation = 1;
        spectral.source_start = 91;
        assert_eq!(
            self_group(&snapshot, Some(binding), true, Some(spectral), owner, 100),
            Err("body_record_future")
        );
        spectral.source_start = 10;
        spectral.available = 101;
        let future_binding = Binding {
            available: 101,
            ..binding
        };
        assert_eq!(
            self_group(
                &snapshot,
                Some(future_binding),
                true,
                Some(spectral),
                owner,
                100
            ),
            Err("body_record_future")
        );
    }
}
