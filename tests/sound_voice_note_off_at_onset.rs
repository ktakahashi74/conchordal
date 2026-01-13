use conchordal::core::modulation::NeuralRhythms;
use conchordal::core::timebase::Timebase;
use conchordal::life::intent::Intent;
use conchordal::life::sound::Voice;

#[test]
fn note_off_at_onset_still_releases() {
    let tb = Timebase {
        fs: 48_000.0,
        hop: 64,
    };
    let intent = Intent {
        source_id: 1,
        intent_id: 42,
        onset: 0,
        duration: tb.sec_to_tick(0.5),
        freq_hz: 440.0,
        amp: 0.5,
        tag: None,
        confidence: 1.0,
        body: None,
        articulation: None,
    };

    let mut voice = Voice::from_intent(tb, intent).expect("voice");
    voice.note_off(0);
    voice.arm_onset_trigger(1.0);

    let dt = 1.0 / tb.fs;
    let mut rhythms = NeuralRhythms::default();
    let mut max_amp = 0.0f32;
    let sample_ticks = tb.sec_to_tick(0.02).max(1);
    for tick in 0..sample_ticks {
        let s = voice.render_tick(tick, tb.fs, dt, &rhythms);
        max_amp = max_amp.max(s.abs());
        rhythms.advance_in_place(dt);
    }
    assert!(max_amp > 1e-6);

    let done_tick = tb.sec_to_tick(0.2);
    assert!(voice.is_done(done_tick));
}
