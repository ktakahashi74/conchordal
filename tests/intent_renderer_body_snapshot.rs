use conchordal::core::modulation::NeuralRhythms;
use conchordal::core::timebase::Timebase;
use conchordal::life::intent::{BodySnapshot, Intent, IntentBoard};
use conchordal::life::schedule_renderer::ScheduleRenderer;

fn make_intent(kind: &str, brightness: f32) -> Intent {
    Intent {
        source_id: 0,
        intent_id: 1,
        onset: 5,
        duration: 20,
        freq_hz: 440.0,
        amp: 0.4,
        tag: None,
        confidence: 1.0,
        body: Some(BodySnapshot {
            kind: kind.to_string(),
            amp_scale: 1.0,
            brightness,
            noise_mix: 0.0,
        }),
    }
}

#[test]
fn body_snapshot_changes_timbre_without_shifting_onset() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let mut board_a = IntentBoard::new(tb.sec_to_tick(1.0), tb.sec_to_tick(1.0));
    let mut board_b = IntentBoard::new(tb.sec_to_tick(1.0), tb.sec_to_tick(1.0));
    board_a.publish(make_intent("sine", 0.0));
    board_b.publish(make_intent("harmonic", 0.8));

    let mut renderer_a = ScheduleRenderer::new(tb);
    let mut renderer_b = ScheduleRenderer::new(tb);
    let rhythms = NeuralRhythms::default();
    let out_a = renderer_a.render(&board_a, 0, &rhythms);
    let out_b = renderer_b.render(&board_b, 0, &rhythms);

    let eps = 1e-6_f32;
    let first_a = out_a
        .iter()
        .position(|&s| s.abs() > eps)
        .expect("expected onset sample");
    let first_b = out_b
        .iter()
        .position(|&s| s.abs() > eps)
        .expect("expected onset sample");
    assert_eq!(first_a, 5);
    assert_eq!(first_b, 5);

    assert!(
        out_a
            .iter()
            .zip(out_b.iter())
            .any(|(a, b)| (a - b).abs() > eps),
        "expected different timbre output"
    );
}
