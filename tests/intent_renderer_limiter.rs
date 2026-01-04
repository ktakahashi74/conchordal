use conchordal::core::modulation::NeuralRhythms;
use conchordal::core::timebase::Timebase;
use conchordal::life::intent::{Intent, IntentBoard};
use conchordal::life::schedule_renderer::ScheduleRenderer;

#[test]
fn limiter_clamps_peak_and_stays_finite() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 128,
    };
    let mut board = IntentBoard::new(tb.sec_to_tick(1.0), tb.sec_to_tick(1.0));
    for i in 0..10u64 {
        board.publish(Intent {
            source_id: 0,
            intent_id: i,
            onset: 0,
            duration: 64,
            freq_hz: 440.0 + (i as f32 * 2.0),
            amp: 1.0,
            tag: None,
            confidence: 1.0,
            body: None,
        });
    }

    let mut renderer = ScheduleRenderer::new(tb);
    let rhythms = NeuralRhythms::default();
    let out = renderer.render(&board, 0, &rhythms);
    let mut peak = 0.0f32;
    for &s in out {
        assert!(s.is_finite());
        peak = peak.max(s.abs());
    }
    assert!(peak <= 0.981, "peak too high: {peak}");
}
