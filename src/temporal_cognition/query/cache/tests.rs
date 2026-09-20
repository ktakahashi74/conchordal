use super::*;

fn header(id: u64, end: f64) -> Header {
    Header {
        epoch: 1,
        generation: 4,
        query_id: id,
        occurrence_id: 9,
        support_id: 19,
        count: 1,
        start: 0.,
        end,
        audio_end: end,
        available: end,
        captured: end,
        error: 0.,
        scales: [1.; 10],
    }
}

fn entry(id: u64, cost: Option<f64>) -> memory::CoarseEntry {
    memory::CoarseEntry {
        identity: Identity { id, generation: 2 },
        cost,
        similarity: cost.map(|v| (-v).exp()),
        approximate: false,
    }
}

#[test]
fn snapshot_storage_masks_last_slot_and_actual_receipt() {
    assert_eq!(std::mem::size_of::<Header>(), 176);
    assert_eq!(std::mem::size_of::<Snapshot>(), 144);
    assert_eq!(std::mem::size_of::<Cell>(), 24);
    assert_eq!(std::mem::offset_of!(Snapshot, valid), 72);
    assert_eq!(std::mem::offset_of!(Snapshot, approximate), 96);
    assert_eq!(std::mem::offset_of!(Snapshot, cells), 120);
    let a = entry(10, Some(2f64.ln()));
    let mut b = entry(11, None);
    b.approximate = true;
    let bindings = [
        Binding {
            identity: a.identity,
            slot: 255,
            handle: 1010,
        },
        Binding {
            identity: b.identity,
            slot: 0,
            handle: 1111,
        },
    ];
    let mut snapshot = Snapshot::default();
    assert_eq!(
        snapshot
            .pack(&header(1, 1.), &[a, b], &bindings, 2.)
            .unwrap(),
        0
    );
    assert_eq!(snapshot.valid, [0, 0, 0, 1 << 63]);
    assert_eq!(snapshot.approximate, [1, 0, 0, 0]);
    let mut c = Cache::new(1, 4, 8, 256);
    c.insert(&snapshot).unwrap();
    assert!(c.snapshots(&[(255, 1010)], 1.5).unwrap().is_empty());
    let view = c
        .snapshots(&[(255, 1010), (0, 1111)], 2.)
        .unwrap()
        .remove(0);
    assert_eq!(view.entries[0].similarity, Some(0.5));
    assert_eq!(view.entries[1].cost, None);
    assert!(view.entries[1].approximate);
    assert!(!c.insert(&snapshot).unwrap());
    let before = c.blocks.clone();
    snapshot.times[2] = 2.1;
    assert!(c.insert(&snapshot).is_err());
    assert!(c.blocks.iter().zip(before.iter()).all(|(a, b)| a.same(b)));
    assert_eq!(c.clear(1, 5), 1);
    assert!(!c.insert(&snapshot).unwrap());
    assert_eq!(view.entries[0].cost, Some(2f64.ln()));
}

#[test]
fn bounded_cache_matches_full_history_endpoint_ranking_and_replay_guard() {
    let mut c = Cache::new(1, 4, 8, 256);
    let pointer = c.blocks.as_ptr();
    let mut history = Vec::new();
    let mut state = 47014u64;
    for id in 1..=1000 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let end = (1 + (state >> 32) % 32) as f64 / 8.;
        let mut snapshot = Snapshot::default();
        snapshot.pack(&header(id, end), &[], &[], 10.).unwrap();
        c.insert(&snapshot).unwrap();
        history.push((end, id));
        history.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let expected = &history[history.len().saturating_sub(8)..];
        let actual = c
            .snapshots(&[], 10.)
            .unwrap()
            .iter()
            .map(|v| (v.end, v.ids[2]))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert_eq!(pointer, c.blocks.as_ptr());
        assert!(!c.insert(&snapshot).unwrap());
    }
    assert_eq!(c.occupied.iter().filter(|v| **v).count(), 8);
    let mut old = Snapshot::default();
    old.pack(&header(1, 4.), &[], &[], 10.).unwrap();
    assert!(!c.insert(&old).unwrap());
}

#[test]
fn invalid_pack_keeps_bytes_and_unbound_unknown_stays_absent() {
    let mut snapshot = Snapshot::default();
    let a = entry(10, Some(0.));
    snapshot.pack(&header(1, 1.), &[a], &[], 1.).unwrap();
    let before = snapshot.clone();
    let bindings = [
        Binding {
            identity: a.identity,
            slot: 0,
            handle: 1,
        },
        Binding {
            identity: Identity {
                id: 11,
                generation: 2,
            },
            slot: 0,
            handle: 2,
        },
    ];
    assert!(snapshot.pack(&header(2, 1.), &[a], &bindings, 1.).is_err());
    assert!(snapshot.same(&before));
    assert!(snapshot.pack(&header(2, 1.), &[a, a], &[], 1.).is_err());
    assert!(snapshot.same(&before));
    let bad = memory::CoarseEntry {
        similarity: None,
        ..a
    };
    assert!(snapshot.pack(&header(2, 1.), &[bad], &[], 1.).is_err());
    assert!(snapshot.same(&before));
    let mut q = header(2, 1.);
    q.audio_end = f64::NAN;
    assert_eq!(snapshot.pack(&q, &[a], &[], 2.).unwrap(), 1);
    let mut c = Cache::new(1, 4, 8, 256);
    c.insert(&snapshot).unwrap();
    let view = c.snapshots(&[(0, 1)], 2.).unwrap().remove(0);
    assert_eq!(view.audio_end, None);
    assert!(view.entries.is_empty());
}

#[test]
fn expanded_cache_keeps_last_generation_and_reuses_allocated_storage() {
    let capacity = memory::MAX_EPISODES;
    let mut snapshot = Snapshot::new(capacity);
    let mut cache = Cache::new(1, 4, 1, capacity);
    let pointers = (
        snapshot.cells.as_ptr(),
        cache.blocks[0].cells.as_ptr(),
        cache.blocks[0].valid.as_ptr(),
        cache.blocks[0].approximate.as_ptr(),
    );
    for id in 1..=3 {
        let e = entry(id, Some(id as f64));
        let binding = Binding {
            identity: e.identity,
            slot: capacity - 1,
            handle: id,
        };
        snapshot
            .pack(&header(id, id as f64), &[e], &[binding], id as f64)
            .unwrap();
        cache.insert(&snapshot).unwrap();
        let view = cache.snapshots(&[(capacity - 1, id)], id as f64).unwrap();
        assert_eq!(view[0].entries[0].cost, Some(id as f64));
        assert!(
            cache
                .snapshots(&[(capacity - 1, id + 1)], id as f64)
                .unwrap()[0]
                .entries
                .is_empty()
        );
        assert_eq!(
            pointers,
            (
                snapshot.cells.as_ptr(),
                cache.blocks[0].cells.as_ptr(),
                cache.blocks[0].valid.as_ptr(),
                cache.blocks[0].approximate.as_ptr()
            )
        );
    }
    let old = snapshot.clone();
    let e = entry(4, Some(0.));
    assert!(
        snapshot
            .pack(
                &header(4, 4.),
                &[e],
                &[Binding {
                    identity: e.identity,
                    slot: capacity,
                    handle: 4,
                }],
                4.
            )
            .is_err()
    );
    assert!(snapshot.same(&old));
    assert!(cache.insert(&Snapshot::default()).is_err());
}
