use super::*;
use crate::config::{T2Attribution, T2Combination, T2Unknown, T2Window};

fn config() -> TemporalT2Config {
    TemporalT2Config {
        component_weights: [0.0, 0.0, 1.0, 0.0],
        gain: 1.0,
        threshold: 1.0,
        weight_gain: 1.0,
        window: T2Window::HopMean,
        combination: T2Combination::PositiveComponentRise,
        attribution: T2Attribution::FirstHopMassFraction,
        unknown: T2Unknown::Suppress,
    }
}

fn stamp(step: u64) -> Stamp {
    Stamp {
        group: Handle {
            bus: 0,
            epoch: 0,
            generation: 2,
        },
        association: Some(2),
        grid_id: 1,
        start: 1000 + step * 100,
        end: 1100 + step * 100,
        source_start: 1000,
        source_end: 1100 + step * 100,
        available_end: 1100 + step * 100,
        known_samples: 100,
        observed: true,
    }
}

fn frame(step: u64, value: f64) -> Frame {
    let mut means = [[0.0; 4]; CHANNELS];
    means[12][2] = value;
    Frame {
        epoch_start: 0,
        start: 1000 + step * 100,
        end: 1100 + step * 100,
        means,
        processing_us: 0,
    }
}

#[test]
fn native_cell_attribution_keeps_shared_mass_and_residual_in_the_denominator() {
    let space = Log2Space::new(1000.0, 2000.0, 1);
    let mut adapter = Adapter::new(&space, config()).unwrap();
    let mut scans = std::array::from_fn(|_| vec![0.0; space.n_bins()]);
    scans[0][0] = 1.0;
    scans[1][0] = 2.0;
    scans[7][0] = 1.0;
    adapter.prepare(&space, &scans);
    assert_eq!(adapter.weights[0][12], 0.25);
    assert_eq!(adapter.weights[1][12], 0.5);
    assert!(adapter.weights[2..].iter().flatten().all(|&w| w == 0.0));
    assert!(adapter.weights[0][13..].iter().all(|&w| w == 0.0));
}

#[test]
#[should_panic(expected = "T2_group_mass_scan")]
fn group_mass_scan_length_is_checked_in_every_profile() {
    let space = Log2Space::new(1000.0, 2000.0, 1);
    let mut adapter = Adapter::new(&space, config()).unwrap();
    let mut scans = std::array::from_fn(|_| vec![0.0; space.n_bins()]);
    scans[7].pop();
    adapter.prepare(&space, &scans);
}

#[test]
fn direct_peak_needs_no_legacy_accent_and_retains_iir_prefix() {
    let mut stream = Stream::default();
    let mut weights = [0.0; CHANNELS];
    weights[12] = 1.0;
    let mut result = None;
    for (n, value) in [0.0, 0.0, 3.0, 3.0].into_iter().enumerate() {
        let f = frame(n as u64, value);
        result = Some(
            stream
                .push(config(), stamp(n as u64), Some(&f), weights, f.end)
                .unwrap(),
        );
    }
    let (d, diagnostic) = result.unwrap();
    assert!(diagnostic.known);
    assert_eq!(d.status, Status::Admitted);
    let event = d.accent.unwrap();
    assert_eq!(event.event_start, 1200);
    assert_eq!(event.event_end, 1300);
    assert_eq!(event.available_end, 1400);
    assert_eq!(event.source_start, 0);
    assert_eq!(event.source_end, 1400);
    assert_eq!(event.observed_prefix, 300);
    assert_eq!(event.weight, 1.0);
    assert_eq!(d.saliences, Some([0.0, 3.0, 0.0]));
}

#[test]
fn foreign_owner_gap_epoch_and_missing_pcm_cannot_splice_peak_support() {
    for variant in 0..6 {
        let mut stream = Stream::default();
        let mut weights = [0.0; CHANNELS];
        weights[12] = 1.0;
        let mut last = None;
        for (n, value) in [0.0, 0.0, 3.0, 3.0].into_iter().enumerate() {
            let mut s = stamp(n as u64);
            let mut f = frame(n as u64, value);
            if n == 2 {
                match variant {
                    0 => {
                        s.group.generation = 3;
                        s.association = Some(3);
                    }
                    1 => {
                        s.start += 100;
                        s.end += 100;
                        s.source_end += 100;
                        s.available_end += 100;
                        f.start += 100;
                        f.end += 100;
                    }
                    2 => f.epoch_start = 1000,
                    3 => {
                        s.observed = false;
                        s.known_samples = 0;
                    }
                    4 => s.association = None,
                    _ => (),
                }
            }
            last = Some(
                stream
                    .push(
                        config(),
                        s,
                        if variant == 5 && n == 2 {
                            None
                        } else {
                            Some(&f)
                        },
                        weights,
                        f.end,
                    )
                    .unwrap(),
            );
        }
        assert!(last.unwrap().0.accent.is_none(), "variant {variant}");
    }
}

#[test]
fn moving_attribution_is_not_a_rise_and_zero_differs_from_unknown() {
    let mut stream = Stream::default();
    let mut weights = [0.0; CHANNELS];
    weights[12] = 0.1;
    for n in 0..4 {
        weights[12] = [0.1, 0.1, 1.0, 1.0][n];
        let f = frame(n as u64, 20.0);
        let (d, diagnostic) = stream
            .push(config(), stamp(n as u64), Some(&f), weights, f.end)
            .unwrap();
        assert!(diagnostic.known);
        assert!(d.accent.is_none());
        if n == 3 {
            assert_eq!(d.status, Status::BelowThreshold);
        }
    }
    let f = frame(4, 0.0);
    let (d, diagnostic) = stream
        .push(config(), stamp(4), Some(&f), [0.0; CHANNELS], f.end)
        .unwrap();
    assert!(!diagnostic.known);
    assert_eq!(diagnostic.unknown, Some("t2_native_group_mass_absent"));
    assert_eq!(d.status, Status::Unsupported);
}
