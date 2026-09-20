use super::*;
use crate::temporal_cognition::{descriptor, transport::Identity};

fn config(groups: usize) -> Config {
    Config {
        bus: 0,
        epoch: 1,
        sample_rate: 100,
        hop: 10,
        cadence_ms: 100,
        groups,
        knots: 128,
        snapshots: 8,
        episodes: 256,
    }
}

#[test]
fn completed_coarse_receipt_reaches_sealing_but_not_a_foreign_or_retired_group() {
    let mut scheduler = Scheduler::new(config(1)).unwrap();
    scheduler.bind(0, 4, 0).unwrap();
    let raw = raw(0, 4);
    scheduler.observe(0, &raw, &[(0, 10)], 10).unwrap();
    scheduler
        .submit(0, &span(0, 4).prefix(10).unwrap(), [1, 12, 1001], 10)
        .unwrap();
    let dispatch = scheduler.take(10).unwrap().unwrap();
    assert!(finish(&mut scheduler, &dispatch, 20));
    let group = raw.group;
    let selected = scheduler
        .coarse_for_commitment(group, 12, [0, 10], 60, 60)
        .unwrap()
        .unwrap();
    assert_eq!(selected.evidence().query_id, 1);
    assert_eq!(selected.evidence().support_id, 1001);
    assert!(
        scheduler
            .coarse_for_commitment(group, 13, [0, 10], 60, 60)
            .unwrap()
            .is_none()
    );
    assert!(
        scheduler
            .coarse_for_commitment(group, 12, [0, 10], 60, 59)
            .is_err()
    );
    assert!(
        scheduler
            .coarse_for_commitment(Handle { epoch: 2, ..group }, 12, [0, 10], 60, 60)
            .is_err()
    );
    assert!(
        scheduler
            .coarse_for_commitment(Handle { bus: 1, ..group }, 12, [0, 10], 60, 60)
            .is_err()
    );
    scheduler.retire(0, 60).unwrap();
    assert!(
        scheduler
            .coarse_for_commitment(group, 12, [0, 10], 60, 60)
            .unwrap()
            .is_none()
    );
    scheduler.bind(0, 5, 60).unwrap();
    assert!(
        scheduler
            .coarse_for_commitment(
                Handle {
                    generation: 5,
                    ..group
                },
                12,
                [0, 10],
                60,
                60
            )
            .unwrap()
            .is_none()
    );
}

fn raw(i: u64, generation: u64) -> RawDescriptor {
    RawDescriptor {
        group: Handle {
            bus: 0,
            epoch: 1,
            generation,
        },
        start: i * 10,
        end: (i + 1) * 10,
        source_start: i * 10,
        source_end: (i + 1) * 10,
        available_end: (i + 1) * 10,
        known_samples: 10,
        values: [Some(1.); 10],
    }
}

fn span(i: u64, generation: u64) -> descriptor::Span {
    let mut s = descriptor::Span::new(descriptor::Config {
        group: raw(i, generation).group,
        sample_rate: 100,
        first_hop_start: i * 10,
        hop: 10,
        cadence: 2,
        span_start: i as f64 / 10.,
        span_end: None,
        scales: [1.; 10],
        capacity: 128,
    })
    .unwrap();
    let r = raw(i, generation);
    s.push(&r, &[(r.start, r.end)], r.end).unwrap();
    s
}

fn finish(s: &mut Scheduler, job: &Dispatch, cut: u64) -> bool {
    let entries = [memory::CoarseEntry {
        identity: Identity {
            id: 10,
            generation: 2,
        },
        cost: Some(0.),
        similarity: Some(1.),
        approximate: false,
    }];
    let binding = [Binding {
        identity: entries[0].identity,
        slot: 0,
        handle: 1010,
    }];
    s.finish(
        job.ticket,
        Some(Completion {
            header: job.header,
            completed_at: cut,
            complete: true,
            superseded: false,
            coarse: &entries,
        }),
        &binding,
        cut,
    )
    .unwrap()
}

#[test]
fn acquired_union_not_wall_time_or_descriptor_count_controls_dispatch() {
    for cadence in [50, 100, 200] {
        let mut cfg = config(1);
        cfg.cadence_ms = cadence;
        let mut s = Scheduler::new(cfg).unwrap();
        s.bind(0, 4, 0).unwrap();
        let mut r = raw(0, 4);
        r.known_samples = 5;
        r.values = [None; 10];
        s.observe(0, &r, &[(2, 5), (0, 3), (0, 3)], 10).unwrap();
        assert!(!s.observe(0, &r, &[(0, 5)], 10).unwrap());
        assert_eq!(s.groups[0].samples, 5);
        s.submit(0, &span(0, 4).prefix(10).unwrap(), [1, 1, 1001], 10)
            .unwrap();
        let first = s.take(10).unwrap();
        assert_eq!(first.is_some(), cadence == 50);
        if let Some(job) = first {
            assert!(finish(&mut s, &job, 10));
        }
        let mut missing = raw(1, 4);
        missing.known_samples = 0;
        missing.values = [None; 10];
        s.observe(0, &missing, &[], 20).unwrap();
        assert!(s.take(100).unwrap().is_none());
        let mut silence = raw(10, 4);
        silence.values = [None; 10];
        s.observe(0, &silence, &[(100, 110)], 110).unwrap();
        let silence = raw(11, 4);
        s.observe(0, &silence, &[(110, 120)], 120).unwrap();
        s.submit(0, &span(11, 4).prefix(120).unwrap(), [2, 2, 1002], 120)
            .unwrap();
        let job = s.take(120).unwrap().unwrap();
        assert_eq!(s.groups[0].samples, 25);
        assert_eq!(s.groups[0].next_due, (25 / s.period + 1) * s.period);
        assert!(finish(&mut s, &job, 120));
        s.submit(0, &span(11, 4).prefix(120).unwrap(), [3, 2, 1002], 120)
            .unwrap();
        assert!(s.take(1000).unwrap().is_none());
    }
}

#[test]
fn latest_pending_keeps_running_query_and_oldest_group_priority() {
    let mut s = Scheduler::new(config(8)).unwrap();
    for slot in 0..8 {
        let generation = 4 + slot as u64;
        s.bind(slot, generation, 0).unwrap();
    }
    for slot in 0..8 {
        let generation = 4 + slot as u64;
        let r = raw(0, generation);
        s.observe(slot, &r, &[(0, 10)], 10).unwrap();
        s.submit(
            slot,
            &span(0, generation).prefix(10).unwrap(),
            [1, 1, 1001],
            10,
        )
        .unwrap();
    }
    let job = s.take(10).unwrap().unwrap();
    assert_eq!(job.group_slot, 0);
    let mut live = span(0, 4);
    for i in 1..4 {
        let r = raw(i, 4);
        s.observe(0, &r, &[(r.start, r.end)], r.end).unwrap();
        live.push(&r, &[(r.start, r.end)], r.end).unwrap();
        s.submit(0, &live.prefix(r.end).unwrap(), [i + 1, 1, 1001], r.end)
            .unwrap();
    }
    assert_eq!(job.header.query_id, 1);
    assert_eq!(job.descriptor.knots.len(), 1);
    assert_eq!(s.active.header.query_id, 1);
    assert_eq!(s.groups[0].pending.header.query_id, 4);
    assert_eq!(s.counts.superseded_pending, 2);
    assert!(s.take(40).unwrap().is_none());
    assert!(finish(&mut s, &job, 40));
    let mut order = Vec::new();
    for _ in 0..8 {
        let job = s.take(40).unwrap().unwrap();
        order.push(job.group_slot);
        assert!(s.take(40).unwrap().is_none());
        assert!(finish(&mut s, &job, 40));
    }
    assert_eq!(order, vec![1, 2, 3, 4, 5, 6, 7, 0]);
    let views = s.groups[0].cache.snapshots(&[(0, 1010)], 0.4).unwrap();
    assert_eq!(
        views.iter().map(|v| v.ids[2]).collect::<Vec<_>>(),
        vec![1, 4]
    );
    assert!(
        s.groups[0]
            .cache
            .snapshots(&[(0, 1010)], 0.39)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn retirement_restart_and_stale_receipts_do_not_create_a_second_worker() {
    let mut s = Scheduler::new(config(1)).unwrap();
    s.bind(0, 4, 0).unwrap();
    s.observe(0, &raw(0, 4), &[(0, 10)], 10).unwrap();
    s.submit(0, &span(0, 4).prefix(10).unwrap(), [1, 1, 1001], 10)
        .unwrap();
    let old = s.take(10).unwrap().unwrap();
    s.bind(0, 5, 10).unwrap();
    assert!(s.active_obsolete);
    assert!(s.bind(0, 4, 10).is_err());
    assert!(s.observe(0, &raw(1, 4), &[(10, 20)], 20).is_err());
    s.observe(0, &raw(1, 5), &[(10, 20)], 20).unwrap();
    s.submit(0, &span(1, 5).prefix(20).unwrap(), [1, 2, 1002], 20)
        .unwrap();
    assert!(s.take(20).unwrap().is_none());
    assert!(!s.finish(old.ticket + 1, None, &[], 20).unwrap());
    assert!(s.active_slot.is_some());
    assert!(!finish(&mut s, &old, 20));
    let next = s.take(20).unwrap().unwrap();
    assert!(!s.finish(old.ticket, None, &[], 20).unwrap());
    assert_eq!(s.active.header.generation, 5);
    s.restart(2, 200, 20, 0).unwrap();
    assert!(s.active_obsolete);
    s.bind(0, 1, 0).unwrap();
    assert!(s.take(0).unwrap().is_none());
    assert!(!s.finish(next.ticket, None, &[], 0).unwrap());
    assert!(s.active_slot.is_none());
    assert!(
        s.groups[0]
            .cache
            .snapshots(&[(0, 1010)], 0.)
            .unwrap()
            .is_empty()
    );
    assert_eq!(s.period, 20);
    assert_eq!(s.counts.invalidated_active, 2);
}

#[test]
fn conflicting_acquisition_wrong_rate_and_mutated_completion_are_atomic() {
    let mut s = Scheduler::new(config(1)).unwrap();
    s.bind(0, 4, 0).unwrap();
    let mut r = raw(0, 4);
    r.known_samples = 5;
    s.observe(0, &r, &[(0, 5)], 10).unwrap();
    assert!(s.observe(0, &r, &[(5, 10)], 10).is_err());
    assert_eq!(s.groups[0].samples, 5);
    let mut next = raw(1, 4);
    next.available_end = 30;
    assert!(s.observe(0, &next, &[(10, 20)], 20).is_err());
    assert_eq!(s.groups[0].last.unwrap()[1], 10);
    s.observe(0, &raw(1, 4), &[(10, 20)], 20).unwrap();
    let mut query = span(1, 4).prefix(20).unwrap();
    query.sample_rate = 200;
    assert!(s.submit(0, &query, [1, 1, 1001], 20).is_err());
    query.sample_rate = 100;
    s.submit(0, &query, [1, 1, 1001], 20).unwrap();
    let job = s.take(20).unwrap().unwrap();
    for header in [
        Header {
            support_id: 1002,
            ..job.header
        },
        Header {
            scales: [2.; 10],
            ..job.header
        },
        Header {
            audio_end: f64::NAN,
            ..job.header
        },
    ] {
        assert!(
            s.finish(
                job.ticket,
                Some(Completion {
                    header,
                    completed_at: 20,
                    complete: true,
                    superseded: false,
                    coarse: &[]
                }),
                &[],
                20
            )
            .is_err()
        );
        assert!(s.active_slot.is_some());
    }
    assert!(
        s.finish(
            job.ticket,
            Some(Completion {
                header: job.header,
                completed_at: 21,
                complete: true,
                superseded: false,
                coarse: &[]
            }),
            &[],
            20
        )
        .is_err()
    );
    assert!(finish(&mut s, &job, 20));
}

#[test]
fn older_ending_and_query_replays_do_not_replace_newer_pending_work() {
    let mut s = Scheduler::new(config(1)).unwrap();
    s.bind(0, 4, 0).unwrap();
    s.submit(0, &span(2, 4).prefix(30).unwrap(), [2, 2, 1002], 30)
        .unwrap();
    assert!(
        !s.submit(0, &span(0, 4).prefix(10).unwrap(), [3, 1, 1001], 30)
            .unwrap()
    );
    assert!(
        !s.submit(0, &span(2, 4).prefix(30).unwrap(), [2, 2, 1002], 30)
            .unwrap()
    );
    assert_eq!(s.groups[0].pending.header.query_id, 2);
    assert_eq!(s.groups[0].query_highwater, Some(3));
    assert_eq!(s.counts.older_query_loss, 1);
    assert_eq!(s.counts.stale_query, 1);
    assert_eq!(s.counts.submitted, 2);
}

#[test]
fn actual_matcher_coarse_output_reaches_cache_with_receipt_time_and_handles() {
    let mut old = span(0, 4);
    for i in 1..4 {
        let r = raw(i, 4);
        old.push(&r, &[(r.start, r.end)], r.end).unwrap();
    }
    let frozen = old.finish(40).unwrap();
    let episode = memory::Episode {
        identity: Identity {
            id: 10,
            generation: 2,
        },
        epoch: 1,
        available_end: 0.4,
        first_observed_end: 0.4,
        scales: frozen.scales,
        descriptor: frozen.matching().unwrap(),
    };
    let mut live = span(10, 4);
    let mut s = Scheduler::new(config(1)).unwrap();
    s.bind(0, 4, 0).unwrap();
    for i in 10..14 {
        let r = raw(i, 4);
        s.observe(0, &r, &[(r.start, r.end)], r.end).unwrap();
        if i > 10 {
            live.push(&r, &[(r.start, r.end)], r.end).unwrap();
        }
    }
    let q = live.prefix(140).unwrap();
    s.submit(0, &q, [1, 9, 19], 140).unwrap();
    let job = s.take(140).unwrap().unwrap();
    let report = memory::ordered(
        &memory::Query {
            descriptor: &job.descriptor,
            epoch: job.header.epoch,
            end: job.header.end,
            observed_end: 1.4,
            scales: job.header.scales,
        },
        &[episode],
    )
    .unwrap();
    assert_eq!(report.coarse.len(), 1);
    assert_eq!(report.coarse[0].similarity, Some(1.));
    assert!(!report.coarse[0].approximate);
    let bindings = [Binding {
        identity: report.coarse[0].identity,
        slot: 255,
        handle: 1010,
    }];
    assert!(
        s.finish(
            job.ticket,
            Some(Completion {
                header: job.header,
                completed_at: 145,
                complete: true,
                superseded: false,
                coarse: &report.coarse
            }),
            &bindings,
            150
        )
        .unwrap()
    );
    assert!(
        s.groups[0]
            .cache
            .snapshots(&[(255, 1010)], 1.49)
            .unwrap()
            .is_empty()
    );
    let view = s.groups[0]
        .cache
        .snapshots(&[(255, 1010)], 1.5)
        .unwrap()
        .remove(0);
    assert_eq!(view.ids, [1, 4, 1, 9, 19]);
    assert_eq!(view.end, 1.4);
    assert_eq!(view.audio_end, Some(1.4));
    assert_eq!(view.received_at, 1.5);
    assert_eq!(view.entries[0].handle, 1010);
    assert_eq!(view.entries[0].cost, Some(0.));
    assert_eq!(view.entries[0].similarity, Some(1.));
    assert!(!view.entries[0].approximate);
    assert!(
        s.groups[0].cache.snapshots(&[(255, 1011)], 1.5).unwrap()[0]
            .entries
            .is_empty()
    );
    println!(
        "QUERY_MATCHER {}",
        serde_json::json!({"query_end":view.end,"audio_end":view.audio_end,
        "dispatched_at_sample":job.dispatched_at,"received_at":view.received_at,"cost":view.entries[0].cost,"slot":255,"handle":view.entries[0].handle})
    );
}
