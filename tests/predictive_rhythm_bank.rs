use conchordal::core::timebase::{Tick, Timebase};
use conchordal::life::intent::Intent;
use conchordal::life::predictive_rhythm::{RhythmBandSpec, build_pred_rhythm_bank_from_intents};

fn make_intent(onset: Tick, amp: f32, freq_hz: f32) -> Intent {
    Intent {
        source_id: 0,
        intent_id: onset,
        onset,
        duration: 1,
        freq_hz,
        amp,
        tag: None,
        confidence: 1.0,
        body: None,
    }
}

#[test]
fn predictive_rhythm_tracks_periodic_onsets() {
    let tb = Timebase {
        fs: 1000.0,
        hop: 100,
    };
    let now: Tick = 0;
    let period: Tick = 100;
    let mut intents = Vec::new();
    for i in 1..=6 {
        intents.push(make_intent(i * period, 1.0, 440.0));
    }
    intents.push(make_intent(300, 1.0, f32::NAN));
    let specs = vec![RhythmBandSpec {
        freq_hz: 10.0,
        tau_sec: 1.0,
        weight: 1.0,
    }];
    let bank = build_pred_rhythm_bank_from_intents(&tb, now, &intents, &specs, 1000);
    assert_eq!(bank.bands.len(), 1);
    let band = bank.bands[0];
    assert!(band.strength01 > 0.0);
    assert!(band.phase_rad.is_finite());

    let prior_on = bank.prior01_at_tick(&tb, period);
    let prior_half = bank.prior01_at_tick(&tb, period + period / 2);
    assert!(prior_on > prior_half);

    let bank_again = build_pred_rhythm_bank_from_intents(&tb, now, &intents, &specs, 1000);
    let strength_diff = (bank_again.bands[0].strength01 - bank.bands[0].strength01).abs();
    assert!(strength_diff < 1e-6);
    let phase_diff = (bank_again.bands[0].phase_rad - bank.bands[0].phase_rad).abs();
    assert!(phase_diff < 1e-6);
}

#[test]
fn predictive_rhythm_handles_empty_intents() {
    let tb = Timebase {
        fs: 1000.0,
        hop: 100,
    };
    let specs = vec![RhythmBandSpec {
        freq_hz: 5.0,
        tau_sec: 1.0,
        weight: 1.0,
    }];
    let bank = build_pred_rhythm_bank_from_intents(&tb, 0, &[], &specs, 1000);
    let prior = bank.prior01_at_tick(&tb, 100);
    assert!(prior >= 0.0 && prior <= 1.0);
    assert!(bank.bands[0].strength01.is_finite());
}
