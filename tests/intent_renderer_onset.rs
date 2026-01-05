use conchordal::core::modulation::NeuralRhythms;
use conchordal::core::timebase::{Tick, Timebase};
use conchordal::life::intent::{Intent, IntentBoard, IntentKind};
use conchordal::life::schedule_renderer::ScheduleRenderer;

fn make_intent(intent_id: u64, onset: Tick, duration: Tick, freq: f32, amp: f32) -> Intent {
    Intent {
        source_id: 0,
        intent_id,
        kind: IntentKind::Normal,
        onset,
        duration,
        freq_hz: freq,
        amp,
        tag: None,
        confidence: 1.0,
        body: None,
        articulation: None,
    }
}

#[test]
fn onset_is_sample_accurate_across_hops() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let mut board = IntentBoard::new(tb.sec_to_tick(10.0), tb.sec_to_tick(10.0));
    let onset: Tick = tb.hop as Tick + 7;
    board.publish(make_intent(1, onset, 40, 440.0, 0.5));

    let mut renderer = ScheduleRenderer::new(tb);
    let rhythms = NeuralRhythms::default();
    let mut out = Vec::new();
    out.extend_from_slice(renderer.render(&board, 0, &rhythms));
    out.extend_from_slice(renderer.render(&board, tb.hop as Tick, &rhythms));

    let eps = 1e-6_f32;
    let first = out
        .iter()
        .position(|&s| s.abs() > eps)
        .expect("expected onset sample");
    assert_eq!(first as u64, onset);
    assert!(out[onset as usize - 1].abs() <= eps);
    assert!(out[onset as usize].abs() > eps);
}
