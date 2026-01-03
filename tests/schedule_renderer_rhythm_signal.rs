use conchordal::core::modulation::NeuralRhythms;
use conchordal::core::timebase::{Tick, Timebase};
use conchordal::life::intent::{BodySnapshot, Intent, IntentBoard};
use conchordal::life::schedule_renderer::ScheduleRenderer;

fn make_intent(intent_id: u64, onset: Tick, duration: Tick, freq: f32, amp: f32) -> Intent {
    Intent {
        source_id: 0,
        intent_id,
        onset,
        duration,
        freq_hz: freq,
        amp,
        tag: None,
        confidence: 1.0,
        body: Some(BodySnapshot {
            kind: "harmonic".to_string(),
            amp_scale: 1.0,
            brightness: 0.5,
            noise_mix: 1.0,
        }),
    }
}

#[test]
fn rhythm_signal_changes_output() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let retention = tb.sec_to_tick(2.0);
    let future = tb.sec_to_tick(2.0);
    let mut board = IntentBoard::new(retention, future);
    let duration = tb.hop as Tick;
    board.publish(make_intent(1, 0, duration, 440.0, 0.5));

    let mut renderer0 = ScheduleRenderer::new(tb);
    let mut rhythms0 = NeuralRhythms::default();
    rhythms0.theta.beta = 0.0;
    let out0 = renderer0.render(&board, 0, &rhythms0);

    let mut renderer1 = ScheduleRenderer::new(tb);
    let mut rhythms1 = NeuralRhythms::default();
    rhythms1.theta.beta = 1.0;
    let out1 = renderer1.render(&board, 0, &rhythms1);

    let mut any_diff = false;
    for (a, b) in out0.iter().zip(out1.iter()) {
        if (a - b).abs() > 1e-6 {
            any_diff = true;
            break;
        }
    }
    assert!(any_diff);
}
