use conchordal::core::timebase::Tick;
use conchordal::life::note_event::{NoteBoard, NoteEvent};

fn make_note(note_id: u64, onset: Tick, duration: Tick) -> NoteEvent {
    NoteEvent {
        source_id: 1,
        note_id,
        onset,
        duration,
        freq_hz: 440.0,
        amp: 0.5,
        tag: None,
        confidence: 1.0,
        body: None,
        articulation: None,
    }
}

#[test]
fn publish_and_query() {
    let mut board = NoteBoard::new(0, 0);
    board.publish(make_note(1, 10, 5));
    let hits: Vec<_> = board.query_range(0..20).collect();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].note_id, 1);
}

#[test]
fn overlap_is_detected() {
    let mut board = NoteBoard::new(0, 0);
    board.publish(make_note(1, 10, 10));
    let hits: Vec<_> = board.query_range(15..18).collect();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].note_id, 1);
}

#[test]
fn prune_removes_old_finished_notes() {
    let mut board = NoteBoard::new(10, 0);
    board.publish(make_note(1, 80, 5));
    board.publish(make_note(2, 90, 15));
    board.prune(100);
    let ids: Vec<_> = board.query_range(0..200).map(|i| i.note_id).collect();
    assert_eq!(ids, vec![2]);
}

#[test]
fn prune_removes_far_future_notes() {
    let mut board = NoteBoard::new(0, 10);
    board.publish(make_note(1, 105, 5));
    board.publish(make_note(2, 120, 5));
    board.prune(100);
    let ids: Vec<_> = board.query_range(0..200).map(|i| i.note_id).collect();
    assert_eq!(ids, vec![1]);
}

#[test]
fn snapshot_clones_within_range() {
    let mut board = NoteBoard::new(0, 0);
    board.publish(make_note(1, 5, 5));
    board.publish(make_note(2, 20, 5));
    let snap = board.snapshot(15, 20, 10);
    let ids: Vec<_> = snap.iter().map(|i| i.note_id).collect();
    assert_eq!(ids, vec![1, 2]);
}

#[test]
fn publish_out_of_order_keeps_sorted() {
    let mut board = NoteBoard::new(0, 0);
    board.publish(make_note(1, 20, 5));
    board.publish(make_note(2, 10, 5));
    board.publish(make_note(3, 15, 5));
    let onsets: Vec<_> = board.query_range(0..100).map(|i| i.onset).collect();
    assert_eq!(onsets, vec![10, 15, 20]);
}
