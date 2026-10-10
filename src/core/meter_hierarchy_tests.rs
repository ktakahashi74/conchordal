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
                    next += 1.0 / hz as f64;
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
