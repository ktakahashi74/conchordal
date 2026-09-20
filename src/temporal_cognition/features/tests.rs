use super::*;
use serde_json::Value;

fn config() -> Config {
    Config {
        means: [0.; 2],
        deviations: [1.; 2],
        threshold: 1.,
        rms_floor: 1e-6,
    }
}

fn space() -> Log2Space {
    Log2Space::new(64., 1024., 2)
}

fn stamp(start: u64) -> Stamp {
    Stamp {
        group: Handle {
            bus: 0,
            epoch: 2,
            generation: 3,
        },
        association: Some(8),
        grid_id: 1,
        start,
        end: start + 10,
        source_start: start,
        source_end: start + 10,
        available_end: start + 10,
        known_samples: 10,
        observed: true,
    }
}

fn close(actual: Option<f64>, expected: &Value) {
    match (actual, expected.as_f64()) {
        (Some(a), Some(b)) => assert!((a - b).abs() <= 1e-12 * b.abs().max(1.), "{a} != {b}"),
        (None, None) => assert!(expected.is_null()),
        pair => panic!("feature mask mismatch: {pair:?}"),
    }
}

#[test]
fn registered_python_raw_and_accent_references_match() {
    let fixtures: Value = serde_json::from_str(include_str!(
        "../../../tests/fixtures/temporal_cognition/features.json"
    ))
    .unwrap();
    let grid = space();
    let mut endpoints = 0;
    let mut admitted = 0;
    for sequence in fixtures["sequences"].as_array().unwrap() {
        let cfg = &sequence["config"];
        let mut stream = Stream::new(
            grid.n_bins(),
            Config {
                means: std::array::from_fn(|i| cfg["means"][i].as_f64().unwrap()),
                deviations: std::array::from_fn(|i| cfg["deviations"][i].as_f64().unwrap()),
                threshold: cfg["threshold"].as_f64().unwrap(),
                rms_floor: cfg["rms_floor"].as_f64().unwrap(),
            },
        )
        .unwrap();
        let pointer = stream.previous_scan.as_ptr();
        for step in sequence["steps"].as_array().unwrap() {
            let h = &step["input"];
            let mut s = stamp(h["sample_start"].as_u64().unwrap());
            s.group.generation = h["generation"].as_u64().unwrap();
            s.association = h["association_known"]
                .as_bool()
                .unwrap()
                .then(|| h["association_handle"].as_u64().unwrap());
            s.grid_id = h["grid_id"].as_u64().unwrap();
            s.observed = h["observed"].as_bool().unwrap();
            s.known_samples = h["known_sample_intervals"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v[1].as_u64().unwrap() - v[0].as_u64().unwrap())
                .sum();
            let scan: Option<Vec<f64>> = h["spectrum"]
                .as_array()
                .map(|a| a.iter().map(|v| v.as_f64().unwrap()).collect());
            let out = stream
                .push(
                    &grid,
                    Input {
                        stamp: s,
                        energy: h["energy"].as_f64(),
                        bus_energy: h["bus_energy"].as_f64(),
                        energy_scan: scan.as_deref(),
                    },
                    s.end,
                )
                .unwrap()
                .unwrap();
            let raw = &step["raw"];
            for (i, actual) in out.raw.values.iter().enumerate() {
                close(*actual, &raw["values"][i]);
            }
            assert_eq!(out.raw.group, s.group);
            assert_eq!(
                (out.raw.start, out.raw.end, out.raw.known_samples),
                (s.start, s.end, s.known_samples)
            );
            for (value, key) in [
                (out.raw.source_start, "raw_support_start"),
                (out.raw.source_end, "raw_support_end"),
                (out.raw.available_end, "available_end"),
            ] {
                close(Some(value as f64), &raw[key]);
            }
            let expected = &step["detection"];
            if let Some(detection) = out.detector {
                close(
                    Some(detection.acquisition_coverage),
                    &expected["acquisition_coverage"],
                );
                assert_eq!(
                    detection.detector_coverage,
                    expected["detector_coverage"].as_f64().unwrap() == 1.
                );
                let status = match detection.status {
                    Status::Unsupported => "unsupported",
                    Status::BelowThreshold => "below_threshold",
                    Status::NotLocalPeak => "not_local_peak",
                    Status::Admitted => "admitted",
                };
                assert_eq!(
                    status,
                    expected["status"].as_str().unwrap(),
                    "{} end={}",
                    sequence["name"],
                    s.end
                );
                if let Some(saliences) = detection.saliences {
                    for (i, salience) in saliences.iter().enumerate() {
                        close(Some(*salience), &expected["saliences"][i]);
                    }
                } else {
                    assert!(expected["saliences"].is_null());
                }
                if let Some(a) = detection.accent {
                    let e = &expected["accent"];
                    close(Some(a.weight), &e["weight"]);
                    assert_eq!(
                        [a.event_start, a.event_end],
                        [
                            e["event_interval"][0].as_f64().unwrap() as u64,
                            e["event_interval"][1].as_f64().unwrap() as u64
                        ]
                    );
                    assert_eq!(a.group, s.group);
                    assert_eq!(
                        (a.source_start, a.source_end, a.available_end),
                        (s.end - 40, s.end, s.end)
                    );
                    assert!(a.at_cut(s.end));
                    assert!(!a.at_cut(s.end - 1));
                    admitted += 1;
                } else {
                    assert!(expected["accent"].is_null());
                }
            } else {
                assert!(expected.is_null());
            }
            assert_eq!(pointer, stream.previous_scan.as_ptr());
            endpoints += 1;
        }
    }
    assert_eq!(endpoints, 1152);
    assert!(admitted > 100);
    println!("RAW_FEATURE_ORACLE endpoints={endpoints} admitted={admitted}");
}

#[test]
fn original_extended_support_and_availability_survive_peak_admission() {
    let grid = space();
    let mut stream = Stream::new(grid.n_bins(), config()).unwrap();
    for (i, energy) in [1., 1., 16., 16.].into_iter().enumerate() {
        let mut s = stamp(100 + i as u64 * 10);
        s.source_start -= 80;
        s.source_end += 5;
        s.available_end += 15;
        let scan = [energy / 9.; 9];
        assert!(
            stream
                .push(
                    &grid,
                    Input {
                        stamp: s,
                        energy: Some(energy),
                        bus_energy: Some(energy),
                        energy_scan: Some(&scan)
                    },
                    s.available_end - 1
                )
                .unwrap()
                .is_none()
        );
        let out = stream
            .push(
                &grid,
                Input {
                    stamp: s,
                    energy: Some(energy),
                    bus_energy: Some(energy),
                    energy_scan: Some(&scan),
                },
                s.available_end,
            )
            .unwrap()
            .unwrap();
        assert_eq!(
            out.raw.source_start,
            if i == 0 { 20 } else { s.source_start - 10 }
        );
        if i == 3 {
            let a = out.detector.unwrap().accent.unwrap();
            assert_eq!((a.event_start, a.event_end), (120, 130));
            assert_eq!(
                a.raw_intervals,
                [(100, 110), (110, 120), (120, 130), (130, 140)]
            );
            assert_eq!(
                (a.source_start, a.source_end, a.available_end),
                (20, 145, 155)
            );
            assert!(!a.at_cut(130));
            assert!(!a.at_cut(145));
            assert!(a.at_cut(155));
        }
    }
}

#[test]
fn repeats_errors_and_reset_do_not_fabricate_a_predecessor() {
    let grid = space();
    let mut stream = Stream::new(grid.n_bins(), config()).unwrap();
    let pointer = stream.previous_scan.as_ptr();
    let scan = [1.; 9];
    let s = stamp(0);
    let input = || Input {
        stamp: s,
        energy: Some(9.),
        bus_energy: Some(9.),
        energy_scan: Some(&scan),
    };
    stream.push(&grid, input(), 10).unwrap().unwrap();
    assert!(stream.push(&grid, input(), 10).unwrap().is_none());
    let bad = Input {
        energy: Some(8.),
        ..input()
    };
    assert!(stream.push(&grid, bad, 10).is_err());
    assert_eq!(stream.len, 1);
    let bad = Input {
        stamp: stamp(10),
        energy: Some(f64::NAN),
        ..input()
    };
    assert!(stream.push(&grid, bad, 20).is_err());
    assert_eq!(stream.len, 1);
    stream.clear();
    let out = stream.push(&grid, input(), 10).unwrap().unwrap();
    assert!(out.raw.values[3..6].iter().all(Option::is_none));
    assert!(out.detector.is_none());
    assert_eq!(stream.previous_scan.as_ptr(), pointer);
}

#[test]
fn cached_saliences_cannot_cross_an_earlier_evidence_cut() {
    let grid = space();
    let mut stream = Stream::new(grid.n_bins(), config()).unwrap();
    for (i, energy) in [1., 1., 16., 16.].into_iter().enumerate() {
        let mut s = stamp(i as u64 * 10);
        if i < 2 {
            s.available_end = 1000;
        }
        let scan = [energy / 9.; 9];
        let out = stream
            .push(
                &grid,
                Input {
                    stamp: s,
                    energy: Some(energy),
                    bus_energy: Some(energy),
                    energy_scan: Some(&scan),
                },
                if i < 3 { 1000 } else { 40 },
            )
            .unwrap()
            .unwrap();
        if i == 3 {
            let detector = out.detector.unwrap();
            assert_eq!(detector.status, Status::Unsupported);
            assert!(!detector.detector_coverage);
            assert!(detector.accent.is_none());
        }
    }
}

#[test]
fn generation_grid_and_association_changes_mask_differences() {
    let grid = space();
    for change in 0..4 {
        let mut stream = Stream::new(grid.n_bins(), config()).unwrap();
        let scan = [1.; 9];
        stream
            .push(
                &grid,
                Input {
                    stamp: stamp(0),
                    energy: Some(1.),
                    bus_energy: Some(1.),
                    energy_scan: Some(&scan),
                },
                10,
            )
            .unwrap();
        let mut s = stamp(10);
        match change {
            0 => s.group.generation += 1,
            1 => s.group.epoch += 1,
            2 => s.grid_id += 1,
            _ => s.association = Some(9),
        }
        let out = stream
            .push(
                &grid,
                Input {
                    stamp: s,
                    energy: Some(16.),
                    bus_energy: Some(16.),
                    energy_scan: Some(&scan),
                },
                20,
            )
            .unwrap()
            .unwrap();
        assert_eq!(out.raw.values[2], Some(2.));
        assert!(out.raw.values[3..6].iter().all(Option::is_none));
        assert_eq!(out.raw.source_start, 10);
        assert_eq!(stream.observed_prefix, if change < 2 { 10 } else { 20 });
    }
}

#[test]
fn scale_validation_and_scan_alignment_are_hard_boundaries() {
    let grid = space();
    for cfg in [
        Config {
            threshold: 0.,
            ..config()
        },
        Config {
            deviations: [f64::NAN, 1.],
            ..config()
        },
        Config {
            rms_floor: f64::MIN_POSITIVE,
            ..config()
        },
    ] {
        assert!(Stream::new(grid.n_bins(), cfg).is_err());
    }
    assert!(Stream::new(0, config()).is_err());
    let mut stream = Stream::new(grid.n_bins(), config()).unwrap();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| stream.push(
            &grid,
            Input {
                stamp: stamp(0),
                energy: Some(0.),
                bus_energy: Some(0.),
                energy_scan: Some(&[0.; 8])
            },
            10
        )))
        .is_err()
    );
    let mut wrong = Stream::new(8, config()).unwrap();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| wrong.push(
            &grid,
            Input {
                stamp: stamp(0),
                energy: None,
                bus_energy: None,
                energy_scan: None
            },
            10
        )))
        .is_err()
    );
}
