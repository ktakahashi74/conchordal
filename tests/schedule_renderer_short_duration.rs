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
fn short_intent_silence_after_end() {
    let tb = Timebase {
        fs: 1000.0,
        hop: 16,
    };
    let retention = tb.sec_to_tick(2.0);
    let future = tb.sec_to_tick(2.0);
    let mut board = IntentBoard::new(retention, future);
    let onset: Tick = 0;
    let duration: Tick = (tb.hop as Tick).min(4).max(1);
    board.publish(make_intent(1, onset, duration, 440.0, 0.5));

    let mut renderer = ScheduleRenderer::new(tb);
    let rhythms = NeuralRhythms::default();

    let out_start = renderer.render(&board, onset, &rhythms);
    let mut max_start = 0.0f32;
    for &s in out_start {
        max_start = max_start.max(s.abs());
    }
    assert!(max_start > 1e-6_f32);

    let now_tick = onset.saturating_add(tb.sec_to_tick(1.0));
    let out_end = renderer.render(&board, now_tick, &rhythms);
    let mut max_end = 0.0f32;
    for &s in out_end {
        max_end = max_end.max(s.abs());
    }
    assert!(max_end <= 1e-6_f32);
}
