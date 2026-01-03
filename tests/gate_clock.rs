use std::f32::consts::PI;

use conchordal::core::modulation::RhythmBand;
use conchordal::life::gate_clock::next_gate_tick;

fn band_with(phase: f32, freq_hz: f32) -> RhythmBand {
    RhythmBand {
        phase,
        freq_hz,
        mag: 0.0,
        alpha: 0.0,
        beta: 0.0,
    }
}

#[test]
fn next_gate_tick_always_future_when_some() {
    let fs = 48_000.0;
    let now_tick = 1000_u64;
    let target_phase = 0.0;
    let phases = [-PI, -PI + 1e-6, -0.1, 0.0, 0.1, PI - 1e-6, PI];
    let freqs = [0.1, 1.0, 4.0, 20.0];

    for &phase in &phases {
        for &freq_hz in &freqs {
            let theta = band_with(phase, freq_hz);
            let next_tick = next_gate_tick(now_tick, fs, theta, target_phase)
                .expect("expected Some for valid inputs");
            assert!(next_tick > now_tick);
        }
    }
}

#[test]
fn same_phase_goes_to_next_cycle() {
    let fs = 48_000.0;
    let now_tick = 1000_u64;
    let freq_hz = 4.0;
    let phase = 0.0;
    let theta = band_with(phase, freq_hz);
    let next = next_gate_tick(now_tick, fs, theta, phase).expect("expected Some");
    let expected = (fs / freq_hz).round() as u64;
    assert!(next > now_tick);
    let delta = next - now_tick;
    let diff = if delta > expected {
        delta - expected
    } else {
        expected - delta
    };
    assert!(diff <= 1, "delta={delta} expected={expected}");
}

#[test]
fn invalid_inputs_return_none() {
    let now_tick = 1000_u64;
    let target_phase = 0.0;
    let theta = band_with(0.0, 1.0);

    assert!(next_gate_tick(now_tick, 0.0, theta, target_phase).is_none());
    assert!(next_gate_tick(now_tick, -1.0, theta, target_phase).is_none());
    assert!(next_gate_tick(now_tick, f32::NAN, theta, target_phase).is_none());
    assert!(next_gate_tick(now_tick, f32::INFINITY, theta, target_phase).is_none());

    let theta_bad = band_with(0.0, 0.0);
    assert!(next_gate_tick(now_tick, 48_000.0, theta_bad, target_phase).is_none());
    let theta_bad = band_with(0.0, -1.0);
    assert!(next_gate_tick(now_tick, 48_000.0, theta_bad, target_phase).is_none());
    let theta_bad = band_with(0.0, f32::NAN);
    assert!(next_gate_tick(now_tick, 48_000.0, theta_bad, target_phase).is_none());
    let theta_bad = band_with(0.0, f32::INFINITY);
    assert!(next_gate_tick(now_tick, 48_000.0, theta_bad, target_phase).is_none());
}
