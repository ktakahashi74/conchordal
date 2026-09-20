//! Per-Tone terminal-time proxies preserve the event unit used by release learning.

use super::{Body, Record};
use crate::life::participation_trace::{Fit, Frozen, Origin};
use crate::life::self_prediction::ToneEnergy;
use crate::temporal_cognition::private_trace::Head;

#[derive(Debug, serde::Serialize)]
pub(crate) struct Context {
    pub version: u8,
    pub origin: Option<Origin>,
    pub body_generation: Option<u32>,
    pub intrinsic_period_sec: Option<f64>,
    pub horizon: [u64; 2],
    pub timing_model: &'static str,
    pub limitation: &'static str,
}

#[derive(Debug, serde::Serialize)]
pub(crate) struct EndPair {
    pub tone_id: u64,
    pub default_end_sample: Option<u64>,
    pub candidate_end_sample: Option<u64>,
    pub default_end_kind: &'static str,
    pub candidate_end_kind: Option<&'static str>,
    pub status: &'static str,
    pub fits: [Fit; 2],
}

pub(super) fn attach(
    record: &mut Record,
    retained: &[(u64, [bool; 2], ToneEnergy)],
    frozen: Option<&Frozen>,
) {
    let frozen = frozen.filter(|f| {
        f.head == Head::Release
            && f.origin.command_id.is_none()
            && f.origin.source_id == record.source_id
            && f.origin.source_generation == record.source_generation
            && f.origin.issued_at_sample == record.issued_at
            && f.origin.bus == 0
            && record.body_generation.is_some()
    });
    let horizon = [
        record.issued_at,
        record
            .issued_at
            .saturating_add(4 * u64::from(record.sample_rate)),
    ];
    record.release_trace_context = Some(Context {
        version: 1,
        origin: frozen.map(|f| f.origin),
        body_generation: record.body_generation,
        intrinsic_period_sec: frozen.and_then(|f| f.intrinsic_period_sec),
        horizon,
        timing_model: "per_tone_frozen_renderer_end_v1",
        limitation: "Conditional renderer-end lookup in the learned release head, including explicitly labelled natural envelope ends. No observed execution, validated natural-end transfer, Voice-wide aggregation or generation pressure. Missing endpoints and unsupported pairs remain unknown.",
    });
    let default = &record.candidates[0];
    let body = Body {
        retained,
        added: default.added,
        at: default.input.at,
        intervention: (default.intervention, default.input.withhold_until),
    };
    let defaults: Vec<_> = body
        .tones(0)
        .map(|(id, tone, intervention)| {
            (
                id.or(record.tone_id).expect("added recipe has a tone ID"),
                tone.renderer_end_after(record.issued_at, intervention),
                if intervention.is_some() {
                    "candidate_release"
                } else if tone.scheduled_release.is_some() {
                    "scheduled_release"
                } else {
                    "envelope_end"
                },
            )
        })
        .collect();
    for candidate in &mut record.candidates {
        let body = Body {
            retained,
            added: candidate.added,
            at: candidate.input.at,
            intervention: (candidate.intervention, candidate.input.withhold_until),
        };
        let mut ends: Vec<_> = body
            .tones(0)
            .map(|(id, tone, intervention)| {
                (
                    id.or(record.tone_id).expect("added recipe has a tone ID"),
                    tone.renderer_end_after(record.issued_at, intervention),
                    if intervention.is_some() {
                        "candidate_release"
                    } else if tone.scheduled_release.is_some() {
                        "scheduled_release"
                    } else {
                        "envelope_end"
                    },
                )
            })
            .collect();
        ends.sort_unstable_by_key(|v| v.0);
        candidate.release_trace = defaults
            .iter()
            .map(|&(tone_id, default_end, default_kind)| {
                let end = ends
                    .binary_search_by_key(&tone_id, |v| v.0)
                    .ok()
                    .map(|i| ends[i]);
                let candidate_end = end.and_then(|v| v.1);
                let mut row = EndPair {
                    tone_id,
                    default_end_sample: default_end,
                    candidate_end_sample: candidate_end,
                    default_end_kind: default_kind,
                    candidate_end_kind: end.map(|v| v.2),
                    status: "missing_event_pair",
                    fits: [Fit {
                        unassigned: 1.,
                        ..Fit::default()
                    }; 2],
                };
                if let Some((candidate, default)) = candidate_end.zip(default_end) {
                    row.status = "outside_horizon";
                    if candidate >= horizon[0]
                        && default >= horizon[0]
                        && candidate <= horizon[1]
                        && default <= horizon[1]
                    {
                        row.status = "trace_unavailable";
                        if let Some(frozen) = frozen {
                            row.fits = frozen.fit(record.sample_rate, candidate, default);
                            row.status = if row.fits.iter().any(|f| f.paired_support > 0.) {
                                "supported_terminal_proxy"
                            } else {
                                "unlearned_or_unsupported"
                            };
                        }
                    }
                }
                row
            })
            .collect();
    }
}

#[cfg(test)]
mod tests {
    use super::super::{ScheduledRequest, evaluate_scheduled, tests::tone};
    use super::*;
    use crate::life::action_candidates::Class;
    use crate::life::self_prediction::ScheduledRelease;

    #[test]
    fn multiple_tone_ends_remain_distinct_and_gap_removes_only_queued_events() {
        let mut a = tone(0);
        a.envelope.hold_end = 2000;
        a.envelope.release_end = 2020;
        a.envelope.release_ticks = 20;
        let mut b = a;
        b.envelope.hold_end = 2400;
        b.envelope.release_end = 2480;
        b.envelope.release_ticks = 80;
        let mut queued = a;
        queued.envelope.onset = 150;
        let retained = [
            (1, [true, false], a),
            (2, [true, true], b),
            (3, [true, false], queued),
            (4, [false, true], a),
        ];
        let request = ScheduledRequest {
            source_id: 7,
            source_generation: 2,
            body_generation: 4,
            issued_at: 100,
            sample_rate: 1000,
            hop: 20,
            period: Some(1000),
        };
        let mut record = evaluate_scheduled(request, &retained).unwrap();
        let frozen = Frozen::release_fixture();
        attach(&mut record, &retained, Some(&frozen));
        assert_eq!(
            record.release_trace_context.as_ref().unwrap().origin,
            Some(frozen.origin)
        );
        let default = &record.candidates[0].release_trace;
        assert_eq!(
            default
                .iter()
                .map(|e| (e.tone_id, e.default_end_sample))
                .collect::<Vec<_>>(),
            [(1, Some(2020)), (2, Some(2480)), (3, Some(2020))]
        );
        assert!(default.iter().all(|e| e.fits[0].difference == Some(0.)));
        let release = record
            .candidates
            .iter()
            .find(|c| c.input.class == Class::Release && c.input.at == 100)
            .unwrap();
        assert_eq!(
            release
                .release_trace
                .iter()
                .map(|e| e.candidate_end_sample)
                .collect::<Vec<_>>(),
            [Some(120), Some(180), Some(2020)]
        );
        assert!(release.release_trace[0].fits[0].difference.unwrap().abs() > 1e-6);
        assert_eq!(release.release_trace[2].fits[0].difference, Some(0.));
        let gap = record
            .candidates
            .iter()
            .find(|c| c.input.class == Class::Gap && c.input.at == 100)
            .unwrap();
        assert_eq!(gap.release_trace[2].candidate_end_sample, None);
        assert_eq!(gap.release_trace[2].status, "missing_event_pair");
        assert_eq!(gap.release_trace[2].fits[0].difference, None);
        assert_eq!(gap.release_trace[2].fits[0].unassigned, 1.);
        for candidate in &record.candidates {
            assert_eq!(candidate.release_trace.len(), 3);
            for e in &candidate.release_trace {
                if let Some(at) = e.candidate_end_sample {
                    if at <= 4100 {
                        assert_eq!(e.fits, frozen.fit(1000, at, e.default_end_sample.unwrap()));
                    }
                }
            }
        }
    }

    #[test]
    fn zero_tail_late_application_and_past_ends_match_renderer_termination() {
        let mut t = tone(0);
        t.envelope.hold_end = 500;
        t.envelope.release_end = 500;
        let immediate = Some(ScheduledRelease {
            apply_at_sample: 100,
            off_sample: 100,
        });
        assert_eq!(t.support_after(100, immediate), None);
        assert_eq!(t.renderer_end_after(100, immediate), Some(100));
        assert_eq!(t.renderer_end_after(101, immediate), None);
        t.envelope.release_ticks = 20;
        t.envelope.release_end = 520;
        t.scheduled_release = Some(ScheduledRelease {
            apply_at_sample: 250,
            off_sample: 180,
        });
        assert_eq!(t.renderer_end_after(100, None), Some(250));
        assert_eq!(t.renderer_end_after(251, None), None);
        let earlier = Some(ScheduledRelease {
            apply_at_sample: 150,
            off_sample: 150,
        });
        assert_eq!(t.renderer_end_after(100, earlier), Some(170));
        t.scheduled_release = Some(ScheduledRelease {
            apply_at_sample: 800,
            off_sample: 800,
        });
        assert_eq!(t.renderer_end_after(100, None), Some(520));
        let retained = [(1, [true, false], tone(0))];
        let request = ScheduledRequest {
            source_id: 7,
            source_generation: 2,
            body_generation: 4,
            issued_at: 100,
            sample_rate: 1000,
            hop: 20,
            period: Some(1000),
        };
        let mut record = evaluate_scheduled(request, &retained).unwrap();
        attach(&mut record, &retained, Some(&Frozen::release_fixture()));
        let release = record
            .candidates
            .iter()
            .find(|c| c.input.class == Class::Release)
            .unwrap();
        assert_eq!(release.release_trace[0].candidate_end_sample, Some(100));
        assert_eq!(release.release_trace[0].status, "outside_horizon");
        assert_eq!(release.release_trace[0].fits[0].difference, None);
    }

    #[test]
    fn wrong_owner_head_or_issue_cannot_supply_release_fit() {
        let mut t = tone(0);
        t.envelope.hold_end = 1000;
        t.envelope.release_end = 1000;
        let retained = [(1, [true, false], t)];
        let request = ScheduledRequest {
            source_id: 7,
            source_generation: 2,
            body_generation: 4,
            issued_at: 100,
            sample_rate: 1000,
            hop: 20,
            period: Some(1000),
        };
        for failure in 0..5 {
            let mut frozen = Frozen::release_fixture();
            match failure {
                0 => frozen.origin.source_id += 1,
                1 => frozen.origin.source_generation += 1,
                2 => frozen.origin.issued_at_sample += 1,
                3 => frozen.head = Head::Onset,
                _ => frozen.origin.command_id = Some(123),
            }
            let mut record = evaluate_scheduled(request, &retained).unwrap();
            attach(&mut record, &retained, Some(&frozen));
            assert!(
                record
                    .release_trace_context
                    .as_ref()
                    .unwrap()
                    .origin
                    .is_none()
            );
            assert!(record.candidates.iter().all(|c| {
                c.release_trace
                    .iter()
                    .all(|e| e.fits[0].difference.is_none())
            }));
        }
        let presentation = [(1, [false, true], t)];
        let mut record = evaluate_scheduled(request, &presentation).unwrap();
        attach(&mut record, &presentation, Some(&Frozen::release_fixture()));
        assert!(record.candidates.iter().all(|c| c.release_trace.is_empty()));
    }
}
