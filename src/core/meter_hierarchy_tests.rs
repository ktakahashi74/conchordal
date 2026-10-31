//! Registered causal controls; private state and both shaping stability endpoints.
use super::*;

const DT: f32 = 0.005;

fn periodic(hz: f32, stability: f32, skip: bool) -> MeterNetwork {
    let mut net = MeterNetwork::new();
    net.set_shaping(MeterShaping {
        stability,
        basin_hz: None,
    });
    let mut next = 0.0_f64;
    let mut ordinal = 0;
    let mut matched = 0;
    let mut failures = 0;
    let mut subdivision_error = None;
    for tick in 0..(20.0 / DT) as usize {
        let time = tick as f64 * DT as f64;
        let fired = time >= next;
        let drive = if fired {
            next += 1.0 / hz as f64;
            ordinal += 1;
            if skip && matches!(ordinal, 10 | 12) {
                0.0
            } else {
                1.0
            }
        } else {
            0.0
        };
        let previous = net.last;
        let seeded = net.seeded;
        let state = net.process(DT, drive);
        if skip && seeded {
            // Two corroborated double intervals must not transiently halve the learned tactus.
            let expected = if hz == 4.0 && stability == 1.0 {
                2.0
            } else {
                hz
            };
            assert!(
                (1.0 / state.beat.freq_hz - 1.0 / expected).abs() <= (0.2 / expected).min(0.06),
                "missed beats changed tactus to {}",
                state.beat.freq_hz
            );
        }
        if !seeded && net.seeded {
            assert_eq!(state.beat.confidence, 0.0);
            assert_eq!(net.plv_count, 0.0);
        }
        if drive > 0.0 && seeded {
            let ratio = if hz > F_BEAT_MAX || (hz == 4.0 && stability == 1.0) {
                2
            } else {
                1
            };
            if (ordinal - 1) % ratio == 0 {
                let error = wrap_pm_pi(previous.beat.phase).abs() / (TAU * previous.beat.freq_hz);
                if error <= (0.2 / previous.beat.freq_hz).min(0.06) {
                    matched += 1;
                } else {
                    failures += 1;
                }
            }
            if ratio == 2 {
                let error = wrap_pm_pi(previous.subdivision.phase).abs()
                    / (TAU * previous.subdivision.freq_hz);
                if previous.subdivision_ratio == 2 {
                    subdivision_error = Some((error, previous.subdivision.freq_hz));
                }
            }
        }
    }
    assert!(
        matched > failures,
        "phase alignment must persist: {matched} matched, {failures} failed"
    );
    if let Some((error, frequency)) = subdivision_error {
        assert!(
            error <= (0.2 / frequency).min(0.06),
            "subdivision {hz}Hz error {error}"
        );
    }
    net
}

#[test]
fn held_out_tempi_follow_registered_tactus_and_subdivision() {
    // Existing acoustic campaign rates: pulse_slow, alternating, ternary_marked.
    for stability in [0.0, 1.0] {
        for hz in [1.4, 4.0, 1.0 / 0.21] {
            let net = periodic(hz, stability, false);
            let divisor = if hz > 4.0 || (hz == 4.0 && stability == 1.0) {
                2.0
            } else {
                1.0
            };
            let expected = hz / divisor;
            assert!(
                (1.0 / net.last.beat.freq_hz - 1.0 / expected).abs() <= (0.2 / expected).min(0.06),
                "held out {hz}Hz stability {stability}: beat {}",
                net.last.beat.freq_hz
            );
            if divisor == 2.0 {
                assert_eq!(net.last.subdivision_ratio, 2);
                assert!(net.last.subdivision.confidence > 0.05);
            }
        }
    }
}

#[test]
fn missing_whole_beats_preserve_the_established_tactus() {
    for stability in [0.0, 1.0] {
        for hz in [1.4, 4.0] {
            let net = periodic(hz, stability, true);
            let expected = if hz == 4.0 && stability == 1.0 {
                2.0
            } else {
                hz
            };
            assert!(
                (1.0 / net.last.beat.freq_hz - 1.0 / expected).abs() <= (0.2 / expected).min(0.06)
            );
        }
    }
}

#[test]
fn held_out_tempo_step_revises_the_corroborated_interval() {
    for stability in [0.0, 1.0] {
        let mut net = periodic(1.4, stability, false);
        let mut next = 0.0_f64;
        for tick in 0..(20.0 / DT) as usize {
            let time = tick as f64 * DT as f64;
            let drive = if time >= next {
                next += 0.25;
                1.0
            } else {
                0.0
            };
            net.process(DT, drive);
        }
        let expected = if stability == 0.0 { 4.0 } else { 2.0 };
        assert!(
            (1.0 / net.last.beat.freq_hz - 1.0 / expected).abs() <= (0.2 / expected).min(0.06),
            "step selected {} instead of {expected}",
            net.last.beat.freq_hz
        );
    }
}

#[test]
fn first_interval_does_not_confirm_itself() {
    for stability in [0.0, 1.0] {
        let mut net = MeterNetwork::new();
        net.set_shaping(MeterShaping {
            stability,
            basin_hz: None,
        });
        let mut next = 0.0_f64;
        let mut events = 0;
        for tick in 0..(2.0 / DT) as usize {
            let time = tick as f64 * DT as f64;
            let drive = if time >= next {
                events += 1;
                next += 1.0 / 1.4;
                1.0
            } else {
                0.0
            };
            let before = net.seeded;
            let state = net.process(DT, drive);
            if events <= 2 {
                assert!(!net.seeded);
            }
            if !before && net.seeded {
                assert_eq!(state.beat.confidence, 0.0);
                assert_eq!(net.plv_count, 0.0);
            }
            if !before {
                assert_eq!(state.beat.confidence, 0.0);
                assert_eq!(state.subdivision.confidence, 0.0);
                assert_eq!(state.measure.confidence, 0.0);
                assert_eq!(state.subdivision_ratio, 0);
                assert_eq!(state.measure_ratio, 0);
                assert_eq!(net.plv_count, 0.0);
                assert_eq!((net.beat_re, net.beat_im), (0.0, 0.0));
                assert_eq!(net.sub_re, [0.0; 3]);
                assert_eq!(net.sub_im, [0.0; 3]);
                assert_eq!(net.meas_re, [0.0; 3]);
                assert_eq!(net.meas_im, [0.0; 3]);
                assert_eq!(net.meas_norm, 0.0);
            }
        }
        assert!(net.seeded);
        for tick in 0..(1.0 / DT) as usize {
            let time = 2.0 + tick as f64 * DT as f64;
            let drive = if time >= next {
                next += 1.0 / 1.4;
                1.0
            } else {
                0.0
            };
            net.process(DT, drive);
        }
        assert!(net.plv_count > 0.0);
        assert!(net.last.beat.confidence > 0.0);
    }
}

#[test]
fn basin_follows_in_band_evidence_and_retains_unknowns_without_it() {
    for stability in [0.0, 1.0] {
        for hz in [3.0, 2.25, 0.0] {
            let mut net = MeterNetwork::new();
            net.set_shaping(MeterShaping {
                stability,
                basin_hz: Some((2.8, 3.4)),
            });
            let mut next = 0.0_f64;
            for tick in 0..(20.0 / DT) as usize {
                let time = tick as f64 * DT as f64;
                let drive = if hz > 0.0 && time >= next {
                    next += 1.0 / hz;
                    1.0
                } else {
                    0.0
                };
                let state = net.process(DT, drive);
                assert!((2.8..=3.4).contains(&state.beat.freq_hz));
                if !net.seeded {
                    assert_eq!(state.beat.confidence, 0.0);
                    assert_eq!(state.subdivision.confidence, 0.0);
                    assert_eq!(state.measure.confidence, 0.0);
                    assert_eq!(state.subdivision_ratio, 0);
                    assert_eq!(state.measure_ratio, 0);
                }
            }
            if hz == 3.0 {
                assert!(
                    (1.0 / net.last.beat.freq_hz - 1.0 / 3.0).abs() <= (0.2 / 3.0_f32).min(0.06)
                );
            }
            if hz == 0.0 {
                assert_eq!(net.last.beat.confidence, 0.0);
                assert_eq!(net.last.measure_ratio, 0);
            }
            if hz == 2.25 {
                assert!(!net.seeded);
            }
        }
    }
}

#[cfg(feature = "profile-alloc")]
#[test]
fn meter_hierarchy_process_adds_no_allocations_at_either_stability_endpoint() {
    use crate::runtime_profile::{begin_allocations, finish_allocations};
    for stability in [0.0, 1.0] {
        let mut net = MeterNetwork::new();
        net.set_shaping(MeterShaping {
            stability,
            basin_hz: None,
        });
        let _ = net.process(DT, 0.0);
        begin_allocations();
        for tick in 0..8000 {
            std::hint::black_box(net.process(DT, if tick % 50 == 0 { 1.0 } else { 0.0 }));
        }
        let counts = finish_allocations().unwrap();
        assert_eq!(counts.count, 0);
        assert_eq!(counts.bytes, 0);
    }
}

#[test]
fn recurrence_candidates_share_the_complete_observation_denominator() {
    let mut recurrence = Recurrence::new();
    recurrence.history.extend([0.0, 0.5, 1.0, 1.5]);
    assert_eq!(recurrence.coverage(0.5, 0.06), 0.75);
    assert_eq!(recurrence.coverage(1.0, 0.06), 0.5);
}

#[test]
fn an_accepted_period_remains_available_without_a_new_confirming_triple() {
    let mut recurrence = Recurrence::new();
    for i in 0..6 {
        recurrence.observe(i * 5000, 5000, 10000.0, true, 1.0);
    }
    recurrence.accept(0.5, 1.0);
    assert_eq!(
        recurrence.observe(30000, 7500, 10000.0, true, 1.0),
        Some(0.5)
    );
}

#[test]
fn confirmation_retains_the_maximum_period_plus_its_window_tail() {
    let mut recurrence = Recurrence::new();
    assert_eq!(recurrence.observe(0, 1, 10000.0, true, 0.0), None);
    assert_eq!(recurrence.observe(1, 19999, 10000.0, true, 1.0), None);
    assert_eq!(
        recurrence.observe(20000, 20600, 10000.0, true, 1.0),
        Some(2.0)
    );
    assert_eq!(recurrence.history.len(), 3);
}

#[test]
fn alternating_and_eight_position_patterns_confirm_the_radiated_period() {
    for offsets in [
        vec![0.0, 0.13],
        vec![0.0, 0.25],
        vec![
            0.112250, 0.133000, 0.143708, 0.220938, 0.235833, 0.240271, 0.292188, 0.477875,
        ],
    ] {
        let expected = if offsets == [0.0, 0.25] { 0.25 } else { 0.5 };
        let mut recurrence = Recurrence::new();
        let mut now = 0.0;
        let mut selected = None;
        for cycle in 0..12 {
            for &offset in &offsets {
                let time = cycle as f64 * 0.5 + offset;
                selected = recurrence.observe(
                    (now * 1e6_f64).round() as u64,
                    ((time - now) * 1e6).round() as usize,
                    1e6,
                    true,
                    1.0,
                );
                now = time;
                if let Some(period) = selected {
                    recurrence.accept(period, 1.0);
                }
            }
        }
        let period = selected.expect("repeated positions must confirm a period");
        assert!((period - expected).abs() <= (0.2 * expected).min(0.06));
    }
}

#[test]
fn uniform_streams_keep_the_registered_native_frequency_and_confidence() {
    // Immutable T-7/T-12 controls: regular streams retain the adjacent-interval path.
    for (hz, stability, expected_hz, expected_confidence) in [
        (1.4, 0.0, 1.3986026, 0.7543916),
        (1.4, 1.0, 1.3986026, 0.99981546),
        (4.0, 0.0, 4.0, 0.9999992),
        (4.0, 1.0, 2.0000007, 0.05000431),
    ] {
        let net = periodic(hz, stability, false);
        assert!(net.recurrence.events().unwrap().uniform);
        assert_eq!(net.last.beat.freq_hz, expected_hz);
        assert_eq!(net.last.beat.confidence, expected_confidence);
    }
}

#[test]
fn bounded_beat_cycles_preserve_measure_motion_and_negative_corrections() {
    let mut net = MeterNetwork::new();
    for i in 0..100_000 {
        net.integrate_hopf(DT, 0.0, 1.0);
        assert!((0.0..12.0).contains(&net.beat_cycles));
        if i % 71 == 0 {
            net.correct_phase(DT, 0.5, 1.0, 1.0, net.beat_omega, Some(0.5));
            assert!((0.0..12.0).contains(&net.beat_cycles));
        }
        if i % 997 == 0 {
            net.seed_beat(2.0, DT, 0.3);
            assert!((0.0..12.0).contains(&net.beat_cycles));
        }
    }
    for ratio in MEASURE_RATIOS {
        net.beat_cycles = 12.0_f32.next_down();
        net.beat_omega = TAU * 2.0;
        let before = net.beat_cycles as f64;
        net.integrate_hopf(DT, 0.0, 1.0);
        let actual = (net.beat_cycles as f64 - before).rem_euclid(12.0) * std::f64::consts::TAU
            / ratio as f64;
        let expected = net.beat_omega as f64 * DT as f64 / ratio as f64;
        // One cycle-coordinate f32 ulp, propagated into measure radians.
        let tolerance =
            (12.0_f32 - 12.0_f32.next_down()) as f64 * std::f64::consts::TAU / ratio as f64;
        assert!((actual - expected).abs() <= tolerance);
    }
    net.beat_cycles = 0.0;
    net.previous_phase_error = None;
    net.correct_phase(DT, 1.0, 1.0, 1.0, net.beat_omega, None);
    assert!(net.beat_cycles > 11.0 && net.beat_cycles < 12.0);
    assert_eq!(wrap_beat_cycles(-f32::MIN_POSITIVE), 0.0);
}

#[test]
fn recurrence_translated_endpoint_patterns_preserve_coverage_and_period_bits() {
    for ticks in [
        vec![0, 4800, 9600],
        vec![0, 12000, 24000],
        vec![0, 4800, 9600, 14400],
    ] {
        let mut near = Recurrence::new();
        let mut far = Recurrence::new();
        for tick in ticks {
            assert_eq!(
                near.observe(tick, 0, 48000.0, true, 0.0),
                far.observe((1 << 40) + tick, 0, 48000.0, true, 0.0)
            );
            assert_eq!(near.history, far.history);
            if near.history.len() >= 2 {
                let period = near.history[1] - near.history[0];
                let coverage = near.coverage(period, ioi_window(period));
                assert_eq!(
                    coverage.to_bits(),
                    far.coverage(period, ioi_window(period)).to_bits()
                );
                let expected = (near.history.len() - 1) as f64 / near.history.len() as f64;
                assert!((coverage - expected).abs() < 8.0 * f64::EPSILON);
            }
        }
    }
}

#[test]
fn recurrence_roundoff_zero_weight_does_not_consume_a_later_full_match() {
    let mut coverages = Vec::new();
    for origin in [0, 1 << 40] {
        let mut recurrence = Recurrence::new();
        for tick in [0, 2880, 26880, 50880] {
            recurrence.observe(origin + tick, 0, 48000.0, true, 0.0);
        }
        let values = [0.5, 0.5 + 2.0_f64.powi(-53)]
            .map(|period| recurrence.coverage(period, ioi_window(period)));
        println!("roundoff-zero coverage origin={origin}: {values:?}");
        for value in values {
            assert!((value - 0.5).abs() <= 8.0 * f64::EPSILON, "{values:?}");
        }
        coverages.push(values.map(f64::to_bits));
    }
    assert_eq!(coverages[0], coverages[1]);
}

#[test]
fn recurrence_rebases_raw_ticks_through_fractional_onsets_silence_and_restart() {
    let mut near = Recurrence::new();
    let mut far = Recurrence::new();
    let mut saw_empty = false;
    for step in 0..1600 {
        let fired = !(600..1200).contains(&step) && step % 10 == 0;
        let fraction = [0.0, 0.375, 1.0][step % 3];
        let tick = step as u64 * 480;
        let selected = near.observe(tick, 480, 48000.0, fired, fraction);
        assert_eq!(
            selected,
            far.observe((1 << 40) + tick, 480, 48000.0, fired, fraction)
        );
        assert_eq!(near.history, far.history);
        assert_eq!(near.time.to_bits(), far.time.to_bits());
        assert_eq!(near.onset_time.to_bits(), far.onset_time.to_bits());
        assert_eq!(near.support().to_bits(), far.support().to_bits());
        assert_eq!(near.echo.len(), far.echo.len());
        for (a, b) in near.echo.iter().zip(&far.echo) {
            assert_eq!(
                (
                    a.target,
                    a.observation_deadline,
                    a.weight,
                    a.prediction_done,
                    a.observation_done
                ),
                (
                    b.target,
                    b.observation_deadline,
                    b.weight,
                    b.prediction_done,
                    b.observation_done
                )
            );
        }
        if step > 600 && near.history.is_empty() {
            saw_empty = true;
        }
        for state in [&mut near, &mut far] {
            if let Some(period) = selected {
                state.accept(period, 1.0);
            }
            if fired {
                state.issue();
            }
        }
    }
    assert!(saw_empty);
    assert!(near.origin_tick > 0);
    assert_eq!(far.origin_tick - near.origin_tick, 1 << 40);
}

#[test]
fn aged_recurrence_reproduces_old_tie_and_ticks_preserve_period() {
    fn legacy_choice(state: &Recurrence, steps: u64) -> f32 {
        let mut best = 0.0;
        let mut score = -1.0;
        let n = state.history.len();
        let now = *state.history.last().unwrap();
        for j in 1..n - 1 {
            for k in 0..j {
                let period = state.history[j] - state.history[k];
                if !(0.0625..=2.0).contains(&period)
                    || ((now - state.history[j]) - period).abs() > ioi_window(period)
                {
                    continue;
                }
                let coverage = state.coverage(period, ioi_window(period));
                let q = steps as f64 * f64::EPSILON;
                let roundoff = 4.0 * q / (1.0 - q) * state.time / ioi_window(period)
                    + (8 * n + 32) as f64 * f64::EPSILON;
                if coverage > score + roundoff
                    || ((coverage - score).abs() <= roundoff && (best == 0.0 || period < best))
                {
                    best = period;
                    score = coverage;
                }
            }
        }
        best as f32
    }
    let mut choices = Vec::new();
    for start in [0, 1u64 << 40] {
        let mut state = Recurrence::new();
        let mut selected = None;
        for cycle in 0..12 {
            for offset in [5388, 6384, 6898, 10605, 11320, 11533, 14025, 22938] {
                selected = state.observe(start + cycle * 24000 + offset, 0, 48000.0, true, 0.0);
                if let Some(p) = selected {
                    state.accept(p, 1.0);
                }
            }
        }
        choices.push(selected.unwrap());
        if start != 0 {
            // Inject the old multi-day accumulated time and step count, retaining
            // the same onset geometry. The former bound admits unrelated lags.
            let old_steps = start / 480;
            state.time += start as f64 / 48000.0;
            let old = legacy_choice(&state, old_steps);
            assert!(
                (old - selected.unwrap()).abs() > 0.05,
                "old={old} fixed={selected:?}"
            );
        }
    }
    assert_eq!(choices, vec![0.5, 0.5]);
}
