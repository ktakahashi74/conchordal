use conchordal::core::modulation::NeuralRhythms;
use conchordal::core::timebase::Timebase;
use conchordal::life::intent::{Intent, IntentBoard};
use conchordal::life::schedule_renderer::ScheduleRenderer;

#[test]
fn cutoff_skips_future_intents() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let mut board = IntentBoard::new(tb.sec_to_tick(1.0), tb.sec_to_tick(1.0));
    board.publish(Intent {
        source_id: 0,
        intent_id: 1,
        onset: 0,
        duration: 20,
        freq_hz: 440.0,
        amp: 0.6,
        tag: None,
        confidence: 1.0,
        body: None,
    });
    let rhythms = NeuralRhythms::default();

    let mut renderer_on = ScheduleRenderer::new(tb);
    let out_on = renderer_on.render(&board, 0, &rhythms);
    assert!(out_on.iter().any(|s| s.abs() > 1e-6));

    let mut renderer_cut = ScheduleRenderer::new(tb);
    renderer_cut.set_cutoff_tick(Some(0));
    let out_cut = renderer_cut.render(&board, 0, &rhythms);
    assert!(out_cut.iter().all(|s| s.abs() <= 1e-6));
}

#[test]
fn shutdown_releases_active_voices() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let mut board = IntentBoard::new(tb.sec_to_tick(2.0), tb.sec_to_tick(2.0));
    board.publish(Intent {
        source_id: 0,
        intent_id: 1,
        onset: 0,
        duration: tb.sec_to_tick(1.0),
        freq_hz: 440.0,
        amp: 0.6,
        tag: None,
        confidence: 1.0,
        body: None,
    });
    let rhythms = NeuralRhythms::default();

    let mut renderer = ScheduleRenderer::new(tb);
    let _ = renderer.render(&board, 0, &rhythms);
    assert!(!renderer.is_idle());

    renderer.shutdown_at(0);
    let mut now: u64 = 0;
    for _ in 0..16 {
        let _ = renderer.render(&board, now, &rhythms);
        now = now.saturating_add(tb.hop as u64);
    }
    assert!(renderer.is_idle());
}
