//! Bounded arrival-CDF comparison for one participation decision.

use crate::temporal_cognition::{
    action_profiles::consumer::{Binding, Key},
    arrival::{ArrivalPublication, CDF_LEN},
};
use serde::Serialize;
use std::sync::Arc;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ArrivalInput {
    pub publication: Arc<ArrivalPublication>,
    pub received_at: u64,
    pub current_epoch: u64,
    pub owner: (u64, u32, Option<u32>),
    pub current_profile_key: Option<Key>,
    pub bindings: [Option<Binding>; 8],
    pub bindings_overflow: bool,
}

impl ArrivalInput {
    pub(crate) fn new(
        publication: Arc<ArrivalPublication>,
        received_at: u64,
        current_epoch: u64,
        owner: (u64, u32, Option<u32>),
        current_profile_key: Option<Key>,
        bindings: &[Binding],
    ) -> Self {
        let mut selected = [None; 8];
        let mut count = 0;
        for binding in bindings
            .iter()
            .filter(|binding| binding.source_id == owner.0)
        {
            if count < selected.len() {
                selected[count] = Some(*binding);
            }
            count += 1;
        }
        Self {
            publication,
            received_at,
            current_epoch,
            owner,
            current_profile_key,
            bindings: selected,
            bindings_overflow: count > selected.len(),
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct BindingCheck {
    pub binding: Binding,
    pub status: &'static str,
    pub group_slot: Option<usize>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct GroupCheck {
    pub slot: usize,
    pub status: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ArrivalDecisionTrace {
    pub input: Option<ArrivalInput>,
    pub input_status: &'static str,
    pub self_status: &'static str,
    pub binding_checks: [Option<BindingCheck>; 8],
    pub groups: [Option<GroupCheck>; 7],
    pub eligible_count: usize,
    pub known_decision: bool,
}

pub(super) struct Selection<'a> {
    input: Option<&'a ArrivalInput>,
    used: [bool; 7],
    count: usize,
    trace: ArrivalDecisionTrace,
}

impl<'a> Selection<'a> {
    pub(super) fn new(
        input: Option<&'a ArrivalInput>,
        now: u64,
        max_upper: f64,
        expected_rate: u32,
        trace: bool,
    ) -> Self {
        let mut report = ArrivalDecisionTrace {
            input: trace.then(|| input.cloned()).flatten(),
            input_status: "no_publication",
            self_status: "no_binding",
            binding_checks: [None; 8],
            groups: [None; 7],
            eligible_count: 0,
            known_decision: false,
        };
        let Some(input) = input else {
            return Self {
                input,
                used: [false; 7],
                count: 0,
                trace: report,
            };
        };
        let issue = &input.publication.issue;
        report.input_status = if issue.bus != 0 {
            "wrong_bus"
        } else if issue.epoch != input.current_epoch {
            "wrong_epoch"
        } else if issue.version != 1
            || issue.sample_rate == 0
            || issue.sample_rate != expected_rate
            || u64::from(issue.sample_rate)
                .checked_mul(4)
                .and_then(|span| issue.issued_at.checked_add(span))
                != Some(issue.horizon_end)
            || input.received_at > now
            || issue.issued_at > input.received_at
        {
            "invalid_clock"
        } else if now >= issue.horizon_end {
            "expired"
        } else {
            "ready"
        };
        let mut self_slots = [false; 7];
        if !input.bindings_overflow {
            for (slot, binding) in input.bindings.iter().enumerate() {
                let Some(binding) = binding else { continue };
                let (status, group_slot) = check_binding(input, *binding, now);
                report.binding_checks[slot] = Some(BindingCheck {
                    binding: *binding,
                    status,
                    group_slot,
                });
                if let Some(group_slot) = group_slot {
                    self_slots[group_slot] = true;
                    report.self_status = "matched";
                } else if report.self_status == "no_binding" {
                    report.self_status = status;
                }
            }
        } else {
            report.self_status = "bindings_overflow";
        }
        let mut used = [false; 7];
        let mut count = 0;
        for (slot, group) in issue.groups.iter().enumerate() {
            let Some(group) = group else { continue };
            let status = if report.input_status != "ready" {
                report.input_status
            } else if group.group.bus != issue.bus || group.group.epoch != issue.epoch {
                "wrong_group_epoch"
            } else if issue.groups[..slot]
                .iter()
                .flatten()
                .any(|g| g.group == group.group)
            {
                "duplicate_group"
            } else if self_slots[slot] {
                "self"
            } else if !group.periodic {
                "nonperiodic"
            } else if let Some(unknown) = group.unknown {
                match unknown {
                    crate::temporal_cognition::arrival::CdfUnknown::NoForecast => "no_forecast",
                    crate::temporal_cognition::arrival::CdfUnknown::UnsupportedHorizon => {
                        "unsupported_horizon"
                    }
                    crate::temporal_cognition::arrival::CdfUnknown::UncertainElapsed => {
                        "uncertain_elapsed"
                    }
                    crate::temporal_cognition::arrival::CdfUnknown::ResetUnknown => "reset_unknown",
                    crate::temporal_cognition::arrival::CdfUnknown::Nonperiodic => "nonperiodic",
                    crate::temporal_cognition::arrival::CdfUnknown::InvalidClock => "invalid_clock",
                    crate::temporal_cognition::arrival::CdfUnknown::IntegralFailed => {
                        "integral_failed"
                    }
                }
            } else if group.cdf.is_none() {
                "no_cdf"
            } else if group.forecast.as_ref().is_none_or(|f| {
                f.group != group.group
                    || f.source_start > f.source_end
                    || f.source_end > f.available
                    || f.available > f.issued_at
                    || f.issued_at != issue.issued_at
                    || f.horizon_end != issue.horizon_end
                    || f.model != issue.model
                    || f.version != issue.version
                    || f.reset_unknown
                    || f.elapsed_seconds[0] != f.elapsed_seconds[1]
            }) {
                "invalid_source"
            } else if max_upper > issue.horizon_end as f64 {
                "outside_horizon"
            } else {
                used[slot] = true;
                count += 1;
                "used"
            };
            report.groups[slot] = Some(GroupCheck { slot, status });
        }
        report.eligible_count = count;
        Self {
            input: Some(input),
            used,
            count,
            trace: report,
        }
    }

    pub(super) fn probability(&self, at: f64, width: f64) -> Option<f64> {
        let input = self.input?;
        if self.count == 0 {
            return None;
        }
        let issue = &input.publication.issue;
        let step = f64::from(issue.sample_rate) / 50.0;
        let lower = (at - width).max(issue.issued_at as f64);
        let upper = at + width;
        let mut sum = 0.0;
        for (group, used) in issue.groups.iter().zip(self.used) {
            if !used {
                continue;
            }
            let cdf = group.as_ref()?.cdf.as_ref()?;
            sum += interpolate(cdf, upper, issue.issued_at, step)
                - interpolate(cdf, lower, issue.issued_at, step);
        }
        Some(sum / self.count as f64)
    }

    pub(super) fn known(&self) -> bool {
        self.count != 0
    }

    pub(super) fn into_trace(mut self) -> ArrivalDecisionTrace {
        let known = self.known();
        self.trace.known_decision = known;
        self.trace
    }
}

fn interpolate(cdf: &[f64; CDF_LEN], sample: f64, issued_at: u64, step: f64) -> f64 {
    let q = ((sample - issued_at as f64) / step).clamp(0.0, (CDF_LEN - 1) as f64);
    let i = q.floor() as usize;
    if i == CDF_LEN - 1 {
        return cdf[i];
    }
    cdf[i] + (cdf[i + 1] - cdf[i]) * (q - i as f64)
}

fn check_binding(input: &ArrivalInput, b: Binding, now: u64) -> (&'static str, Option<usize>) {
    let issue = &input.publication.issue;
    let Some(key) = input.current_profile_key else {
        return ("no_profile_key", None);
    };
    if input.publication.profile_key != Some(key) {
        return ("profile_key_mismatch", None);
    }
    if b.source_id != input.owner.0
        || b.source_generation != input.owner.1
        || Some(b.body_generation) != input.owner.2
    {
        return ("wrong_owner", None);
    }
    if b.bus != 0 || key.bus != 0 || issue.bus != 0 {
        return ("wrong_bus", None);
    }
    if key.lookup_version != 3
        || b.model_version != key.body_model_version
        || b.prototype >= key.groups.len()
    {
        return ("incompatible_model", None);
    }
    if key.epoch != input.current_epoch {
        return ("wrong_epoch", None);
    }
    if b.end > b.available || b.available > now {
        return ("future_body", None);
    }
    if now - b.end >= 24_000 {
        return ("stale_body", None);
    }
    if key.assignment_end > key.issued_at || key.issued_at > now || key.issued_at > issue.issued_at
    {
        return ("future_table", None);
    }
    if now - key.assignment_end >= 24_000 {
        return ("stale_table", None);
    }
    if key.routed_prototypes & (1 << b.prototype) == 0 {
        return ("unrouted_prototype", None);
    }
    let Some(handle) = key.groups[b.prototype] else {
        return ("no_group", None);
    };
    if handle.bus != 0 || handle.epoch != issue.epoch {
        return ("wrong_group_epoch", None);
    }
    let Some(slot) = issue
        .groups
        .iter()
        .position(|g| g.as_ref().is_some_and(|g| g.group == handle))
    else {
        return ("retired_group", None);
    };
    ("self", Some(slot))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::ArrivalModel,
        temporal_cognition::{
            TestHandle as Handle,
            arrival::{ArrivalIssue, Forecast, GroupArrival},
        },
    };

    fn handle(generation: u64) -> Handle {
        Handle {
            bus: 0,
            epoch: 7,
            generation,
        }
    }

    fn group(generation: u64, cdf: Option<[f64; CDF_LEN]>) -> GroupArrival {
        let group = handle(generation);
        GroupArrival {
            group,
            periodic: true,
            forecast: Some(Forecast {
                model: ArrivalModel::Periodic,
                version: 1,
                group,
                source_start: 800,
                source_end: 900,
                available: 900,
                issued_at: 1_000,
                horizon_end: 5_000,
                last_accent: 900,
                elapsed_seconds: [0.1; 2],
                reset_unknown: false,
                probability: None,
                observed_survival: None,
                evaluations: 0,
            }),
            cdf,
            unknown: None,
            evaluations: 0,
        }
    }

    fn fixture() -> (Arc<ArrivalPublication>, Key, Binding) {
        let key = Key {
            lookup_version: 3,
            routed_prototypes: 1,
            profile_sha256: [1; 32],
            body_model_version: [2; 32],
            bus: 0,
            epoch: 7,
            issued_at: 1_000,
            assignment_end: 950,
            evaluation_delay_samples: None,
            groups: [Some(handle(1)), None, None, None, None, None, None, None],
        };
        let mut groups = [None; 7];
        groups[0] = Some(group(1, Some(std::array::from_fn(|k| k as f64 / 200.0))));
        groups[1] = Some(group(2, Some([0.0; CDF_LEN])));
        let publication = Arc::new(ArrivalPublication {
            issue: ArrivalIssue {
                model: ArrivalModel::Periodic,
                version: 1,
                bus: 0,
                epoch: 7,
                sample_rate: 1_000,
                issued_at: 1_000,
                horizon_end: 5_000,
                groups,
            },
            profile_key: Some(key),
        });
        let binding = Binding {
            source_id: 10,
            source_generation: 2,
            body_generation: 3,
            bus: 0,
            end: 900,
            available: 900,
            model_version: [2; 32],
            prototype: 0,
            distance: 0.0,
            common_coordinates: 6,
        };
        (publication, key, binding)
    }

    #[test]
    fn fixed_set_excludes_self_and_keeps_known_zero() {
        let (publication, key, binding) = fixture();
        let input = ArrivalInput::new(
            publication,
            1_001,
            7,
            (10, 2, Some(3)),
            Some(key),
            &[binding],
        );
        let selection = Selection::new(Some(&input), 1_010, 5_000.0, 1_000, true);
        assert_eq!(selection.trace.groups[0].unwrap().status, "self");
        assert_eq!(selection.trace.groups[1].unwrap().status, "used");
        assert_eq!(selection.probability(1_100.0, 10.0), Some(0.0));
        assert_eq!(selection.trace.eligible_count, 1);

        let without_binding = ArrivalInput::new(
            Arc::clone(&input.publication),
            1_001,
            7,
            (10, 2, Some(3)),
            Some(key),
            &[],
        );
        let selection = Selection::new(Some(&without_binding), 1_010, 5_000.0, 1_000, true);
        assert_eq!(selection.trace.self_status, "no_binding");
        assert_eq!(selection.trace.eligible_count, 2);
        assert!((selection.probability(1_100.0, 10.0).unwrap() - 0.0025).abs() < 1e-15);
    }

    #[test]
    fn clocks_and_whole_grid_horizon_gate_every_group() {
        let (publication, key, _) = fixture();
        let input = ArrivalInput::new(
            Arc::clone(&publication),
            1_001,
            7,
            (10, 2, Some(3)),
            Some(key),
            &[],
        );
        assert_eq!(
            Selection::new(Some(&input), 1_010, 5_001.0, 1_000, true)
                .trace
                .eligible_count,
            0
        );
        assert_eq!(
            Selection::new(Some(&input), 1_010, 5_000.0, 1_000, true)
                .trace
                .eligible_count,
            2
        );
        let future = ArrivalInput::new(
            Arc::clone(&publication),
            1_020,
            7,
            (10, 2, Some(3)),
            Some(key),
            &[],
        );
        assert_eq!(
            Selection::new(Some(&future), 1_010, 1_100.0, 1_000, true)
                .trace
                .input_status,
            "invalid_clock"
        );
        assert_eq!(
            Selection::new(Some(&input), 5_000, 1_100.0, 1_000, true)
                .trace
                .input_status,
            "expired"
        );
        let wrong_epoch =
            ArrivalInput::new(publication, 1_001, 8, (10, 2, Some(3)), Some(key), &[]);
        assert_eq!(
            Selection::new(Some(&wrong_epoch), 1_010, 1_100.0, 1_000, true)
                .trace
                .input_status,
            "wrong_epoch"
        );
        assert_eq!(
            Selection::new(Some(&input), 1_010, 1_100.0, 48_000, true)
                .trace
                .input_status,
            "invalid_clock"
        );
    }

    #[test]
    fn malformed_forecast_cannot_lend_a_known_cdf() {
        let (mut publication, key, _) = fixture();
        Arc::get_mut(&mut publication).unwrap().issue.groups[0]
            .as_mut()
            .unwrap()
            .forecast
            .as_mut()
            .unwrap()
            .reset_unknown = true;
        let input = ArrivalInput::new(publication, 1_001, 7, (10, 2, Some(3)), Some(key), &[]);
        let selected = Selection::new(Some(&input), 1_010, 1_100.0, 1_000, true);
        assert_eq!(selected.trace.groups[0].unwrap().status, "invalid_source");
        assert_eq!(selected.trace.eligible_count, 1);
    }

    #[test]
    fn owner_generation_and_profile_key_guard_self_exclusion() {
        let (publication, key, binding) = fixture();
        let wrong_owner = ArrivalInput::new(
            Arc::clone(&publication),
            1_001,
            7,
            (10, 3, Some(3)),
            Some(key),
            &[binding],
        );
        let selected = Selection::new(Some(&wrong_owner), 1_010, 1_100.0, 1_000, true);
        assert_eq!(
            selected.trace.binding_checks[0].unwrap().status,
            "wrong_owner"
        );
        assert_eq!(selected.trace.eligible_count, 2);
        let wrong_key =
            ArrivalInput::new(publication, 1_001, 7, (10, 2, Some(3)), None, &[binding]);
        let selected = Selection::new(Some(&wrong_key), 1_010, 1_100.0, 1_000, true);
        assert_eq!(
            selected.trace.binding_checks[0].unwrap().status,
            "no_profile_key"
        );
        assert_eq!(selected.trace.eligible_count, 2);
    }

    #[test]
    fn all_valid_bindings_exclude_all_matching_handles() {
        let (mut publication, mut key, first) = fixture();
        key.groups[1] = Some(handle(2));
        key.routed_prototypes = 3;
        Arc::get_mut(&mut publication).unwrap().profile_key = Some(key);
        let second = Binding {
            prototype: 1,
            ..first
        };
        let input = ArrivalInput::new(
            publication,
            1_001,
            7,
            (10, 2, Some(3)),
            Some(key),
            &[first, second],
        );
        let selection = Selection::new(Some(&input), 1_010, 1_100.0, 1_000, true);
        assert_eq!(selection.trace.eligible_count, 0);
        assert_eq!(selection.trace.groups[0].unwrap().status, "self");
        assert_eq!(selection.trace.groups[1].unwrap().status, "self");
        assert_eq!(selection.probability(1_100.0, 10.0), None);
    }

    #[test]
    fn binding_and_assignment_deadlines_reject_at_exactly_24000_samples() {
        let (mut publication, mut key, mut binding) = fixture();
        let issue = &mut Arc::get_mut(&mut publication).unwrap().issue;
        issue.sample_rate = 48_000;
        issue.issued_at = 30_000;
        issue.horizon_end = 222_000;
        let (issued, horizon) = (issue.issued_at, issue.horizon_end);
        for group in issue.groups.iter_mut().flatten() {
            let forecast = group.forecast.as_mut().unwrap();
            forecast.source_start = 29_800;
            forecast.source_end = 29_900;
            forecast.available = 29_900;
            forecast.issued_at = issued;
            forecast.horizon_end = horizon;
            forecast.last_accent = 29_900;
        }
        key.issued_at = issue.issued_at;
        let near = 31_000;
        let boundary = near + 1;
        for body_deadline in [true, false] {
            binding.end = if body_deadline { near - 23_999 } else { 30_000 };
            binding.available = binding.end;
            key.assignment_end = if body_deadline { 30_000 } else { near - 23_999 };
            Arc::get_mut(&mut publication).unwrap().profile_key = Some(key);
            let input = ArrivalInput::new(
                Arc::clone(&publication),
                30_001,
                7,
                (10, 2, Some(3)),
                Some(key),
                &[binding],
            );
            let accepted = Selection::new(Some(&input), near, 33_000.0, 48_000, true);
            assert_eq!(accepted.trace.binding_checks[0].unwrap().status, "self");
            assert_eq!(accepted.trace.eligible_count, 1);
            assert_eq!(accepted.probability(32_000.0, 1_000.0), Some(0.0));

            let rejected = Selection::new(Some(&input), boundary, 33_000.0, 48_000, true);
            assert_eq!(
                rejected.trace.binding_checks[0].unwrap().status,
                if body_deadline {
                    "stale_body"
                } else {
                    "stale_table"
                }
            );
            assert_eq!(rejected.trace.eligible_count, 2);
            assert!(rejected.probability(32_000.0, 1_000.0).unwrap() > 0.0);
        }
    }

    #[test]
    fn changed_identity_fails_self_exclusion_without_suppressing_external_cdf() {
        let (publication, key, binding) = fixture();
        let check = |publication: Arc<ArrivalPublication>,
                     key: Key,
                     binding: Binding,
                     owner: (u64, u32, Option<u32>),
                     reason: &'static str| {
            let input = ArrivalInput::new(publication, 1_001, 7, owner, Some(key), &[binding]);
            let selection = Selection::new(Some(&input), 1_010, 1_100.0, 1_000, true);
            assert_eq!(selection.trace.binding_checks[0].unwrap().status, reason);
            assert_eq!(selection.trace.groups[0].unwrap().status, "used");
            assert_eq!(selection.trace.eligible_count, 2);
            assert!(selection.probability(1_100.0, 10.0).unwrap() > 0.0);
        };
        check(
            Arc::clone(&publication),
            key,
            binding,
            (10, 3, Some(3)),
            "wrong_owner",
        );
        check(
            Arc::clone(&publication),
            key,
            binding,
            (10, 2, Some(4)),
            "wrong_owner",
        );
        let mut changed_key = key;
        changed_key.profile_sha256[0] ^= 1;
        check(
            Arc::clone(&publication),
            changed_key,
            binding,
            (10, 2, Some(3)),
            "profile_key_mismatch",
        );
        let mut changed_key = key;
        changed_key.assignment_end -= 1;
        check(
            Arc::clone(&publication),
            changed_key,
            binding,
            (10, 2, Some(3)),
            "profile_key_mismatch",
        );
        check(
            Arc::clone(&publication),
            key,
            Binding {
                model_version: [3; 32],
                ..binding
            },
            (10, 2, Some(3)),
            "incompatible_model",
        );
        check(
            Arc::clone(&publication),
            key,
            Binding { bus: 1, ..binding },
            (10, 2, Some(3)),
            "wrong_bus",
        );

        let (mut matching_publication, mut unrouted, _) = fixture();
        unrouted.routed_prototypes = 0;
        Arc::get_mut(&mut matching_publication).unwrap().profile_key = Some(unrouted);
        check(
            matching_publication,
            unrouted,
            binding,
            (10, 2, Some(3)),
            "unrouted_prototype",
        );

        let (mut retired, _, _) = fixture();
        let group = Arc::get_mut(&mut retired).unwrap().issue.groups[0]
            .as_mut()
            .unwrap();
        group.group = handle(9);
        group.forecast.as_mut().unwrap().group = handle(9);
        check(retired, key, binding, (10, 2, Some(3)), "retired_group");
    }

    #[test]
    fn chosen_base_controls_skip_and_reservation_is_stable() {
        use crate::{
            config::{FootprintSource, TemporalOnsetComparisonConfig},
            life::temporal_participation::TemporalParticipation,
        };
        let (publication, key, _) = fixture();
        let input = ArrivalInput::new(publication, 1_001, 7, (10, 2, Some(3)), Some(key), &[]);
        let mut policy = TemporalParticipation::new(1_000, 2.0, 1.0, 1_000, 17, 1_000);
        policy.due_frame = 1_100.0;
        policy.set_overlap(0.8, [1.0, 0.5, 0.25]);
        policy.set_sound_duration(0.2, None);
        policy.set_onset_comparison(Some(TemporalOnsetComparisonConfig {
            footprint: FootprintSource::Proxy,
            arrival: true,
            arrival_weight: 4.0,
        }));
        policy.set_arrival_input(Some(input));
        policy.set_decision_trace(true);
        let forecast = crate::core::temporal_expectation::TemporalForecast::energy_fixture(
            1_000,
            1_000,
            |_| [0.0; 3],
        );
        let first = policy
            .candidate(1_010, 1_300, true, Some(&forecast))
            .unwrap();
        let decision = policy.drain_decisions().next().unwrap();
        assert!(decision.selected_cost > 1.0);
        assert!(decision.selected_base_cost.unwrap() < 1.0);
        assert!(!decision.skipped);
        assert_eq!(policy.take_arrival_known_decisions(), 1);
        policy.set_arrival_input(None);
        assert_eq!(
            policy.candidate(1_010, 1_300, true, Some(&forecast)),
            Some(first)
        );
        assert_eq!(policy.take_arrival_known_decisions(), 0);
    }

    #[test]
    fn local_cdf_intervention_changes_only_arrival_selection() {
        use crate::{
            config::{FootprintSource, TemporalOnsetComparisonConfig},
            life::temporal_participation::TemporalParticipation,
        };
        let (mut publication, key, _) = fixture();
        Arc::get_mut(&mut publication).unwrap().issue.groups[0]
            .as_mut()
            .unwrap()
            .cdf = Some(std::array::from_fn(|k| f64::from(k >= 10)));
        let input = ArrivalInput::new(publication, 1_001, 7, (10, 2, Some(3)), Some(key), &[]);
        let forecast = crate::core::temporal_expectation::TemporalForecast::energy_fixture(
            1_000,
            1_000,
            |_| [0.0; 3],
        );
        let policy = |weight: f64| {
            let mut policy = TemporalParticipation::new(1_000, 2.0, 1.0, 1_000, 17, 1_000);
            policy.due_frame = 1_100.0;
            policy.set_overlap(0.8, [1.0, 0.5, 0.25]);
            policy.set_sound_duration(0.2, None);
            policy.set_onset_comparison(Some(TemporalOnsetComparisonConfig {
                footprint: FootprintSource::Proxy,
                arrival: true,
                arrival_weight: weight,
            }));
            policy.set_arrival_input(Some(input.clone()));
            policy.set_decision_trace(true);
            policy
        };
        let mut off = policy(0.0);
        let mut on = policy(4.0);
        off.candidate(1_010, 1_300, true, Some(&forecast)).unwrap();
        on.candidate(1_010, 1_300, true, Some(&forecast)).unwrap();
        let a = off.drain_decisions().next().unwrap();
        let b = on.drain_decisions().next().unwrap();
        assert_eq!(a.selected_offset, 0);
        assert_ne!(a.selected_offset, b.selected_offset);
        for (a, b) in a
            .candidates
            .iter()
            .flatten()
            .zip(b.candidates.iter().flatten())
        {
            assert_eq!(a.base_cost, b.base_cost);
            assert_eq!(a.arrival_p, b.arrival_p);
        }
    }

    #[test]
    fn disabled_arrival_preserves_legacy_decision_shape() {
        use crate::{
            config::{FootprintSource, TemporalOnsetComparisonConfig},
            life::temporal_participation::TemporalParticipation,
        };
        let mut policy = TemporalParticipation::new(1_000, 2.0, 1.0, 1_000, 17, 1_000);
        policy.due_frame = 1_100.0;
        policy.set_onset_comparison(Some(TemporalOnsetComparisonConfig {
            footprint: FootprintSource::Proxy,
            arrival: false,
            arrival_weight: 1.0,
        }));
        policy.set_decision_trace(true);
        policy.candidate(1_010, 1_300, true, None).unwrap();
        let decision = policy.drain_decisions().next().unwrap();
        let value = serde_json::to_value(decision).unwrap();
        assert!(value.get("arrival").is_none());
        assert!(value.get("arrival_weight").is_none());
        assert!(value.get("selected_base_cost").is_none());
        for candidate in value["candidates"].as_array().unwrap() {
            if candidate.is_null() {
                continue;
            }
            assert!(candidate.get("base_cost").is_none());
            assert!(candidate.get("arrival_p").is_none());
            assert!(candidate.get("arrival_term").is_none());
        }
    }
}
