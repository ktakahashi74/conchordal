use super::*;

#[test]
fn observed_and_projected_features_share_formulas_and_track_each_dependency() {
    let space = Log2Space::new(64., 1024., 2);
    let spectrum = [0., 0., 0., 0., 2., 0., 2., 0., 0.];
    let old = [0., 0., 0., 0., 1., 0., 1., 0., 0.];
    let expected = [8.5, 0.5, 1., 0.5, 0., 1. / 9., 0.5, 0., 1., 0.];
    // Vary each input independently; held background cannot erase a candidate origin.
    for mask in 0..32 {
        let scalar = |value, bit| Feature::from(Some(value), mask & (1 << bit) != 0);
        let projection = evaluate(
            &space,
            Frame {
                energy: scalar(4., 0),
                bus_energy: scalar(8., 1),
                spectrum: Some(Spectrum {
                    energy_scan: &spectrum,
                    projected: mask & 4 != 0,
                }),
            },
            Some(Frame {
                energy: scalar(2., 3),
                bus_energy: Feature::Unsupported,
                spectrum: Some(Spectrum {
                    energy_scan: &old,
                    projected: mask & 16 != 0,
                }),
            }),
            1e-6,
        )
        .unwrap();
        let dependencies = [4, 4, 1, 9, 9, 29, 3, 4, 4, 4];
        for i in 0..10 {
            let value = projection.values[i];
            assert!(
                (value.value().unwrap() - expected[i]).abs() < 1e-14,
                "coordinate {i}"
            );
            assert_eq!(
                value.projected(),
                mask & dependencies[i] != 0,
                "coordinate {i}, mask {mask}"
            );
        }
        assert_eq!(projection.accent_flux, 1.);
        let encoded = serde_json::to_value(projection.values[2]).unwrap();
        assert_eq!(
            encoded["origin"],
            if mask & 1 != 0 {
                "projected"
            } else {
                "observed"
            }
        );
    }
}

#[test]
fn missing_inputs_are_not_replaced_by_zero_spectrum_or_observed_provenance() {
    let space = Log2Space::new(64., 1024., 2);
    let absent = Frame {
        energy: Feature::Unsupported,
        bus_energy: Feature::Unsupported,
        spectrum: None,
    };
    let empty = evaluate(&space, absent, None, 1e-6).unwrap();
    assert_eq!(empty.values, [Feature::Unsupported; 10]);
    let partial = Frame {
        energy: Feature::Projected(4.),
        ..absent
    };
    let values = evaluate(&space, partial, Some(absent), 1e-6)
        .unwrap()
        .values;
    assert_eq!(values[2], Feature::Projected(1.));
    assert_eq!(values.iter().filter(|v| v.value().is_some()).count(), 1);
    let zero_scan = [0.; 9];
    let zero = Frame {
        energy: Feature::Projected(0.),
        bus_energy: Feature::Observed(0.),
        spectrum: Some(Spectrum {
            energy_scan: &zero_scan,
            projected: true,
        }),
    };
    let values = evaluate(&space, zero, Some(zero), 1e-6).unwrap().values;
    assert_eq!(values[2], Feature::Projected(1e-6_f64.log2()));
    for i in [3, 4, 5, 6] {
        assert_eq!(values[i], Feature::Projected(0.));
    }
    for i in [0, 1, 7, 8, 9] {
        assert_eq!(values[i], Feature::Unsupported);
    }
    // Positive assigned energy with missing spectral mass cannot certify zero flux.
    let shapeless = Frame {
        energy: Feature::Projected(1.),
        bus_energy: Feature::Unsupported,
        ..zero
    };
    assert_eq!(
        evaluate(&space, shapeless, Some(zero), 1e-6)
            .unwrap()
            .values[5],
        Feature::Unsupported
    );
    assert_eq!(
        evaluate(&space, zero, Some(shapeless), 1e-6)
            .unwrap()
            .values[5],
        Feature::Unsupported
    );
    let energy_only = evaluate(
        &space,
        Frame {
            energy: Feature::Observed(1.),
            ..absent
        },
        Some(partial),
        1e-6,
    )
    .unwrap()
    .values;
    assert_eq!(energy_only[2], Feature::Observed(0.));
    assert_eq!(energy_only[3], Feature::Projected(0.));
    assert_eq!(energy_only[4], Feature::Projected(1.));
    assert_eq!(energy_only[5], Feature::Unsupported);
}

#[test]
fn projection_rejects_invalid_numeric_inputs_and_both_scan_boundaries() {
    let space = Log2Space::new(64., 1024., 2);
    let frame = Frame {
        energy: Feature::Observed(1.),
        bus_energy: Feature::Observed(2.),
        spectrum: None,
    };
    for invalid in [f64::NAN, f64::INFINITY, -1.] {
        for projected in [false, true] {
            let bad = Frame {
                energy: Feature::from(Some(invalid), projected),
                ..frame
            };
            assert!(evaluate(&space, bad, None, 1e-6).is_err());
            assert!(evaluate(&space, frame, Some(bad), 1e-6).is_err());
            let scan = [invalid; 9];
            let bad = Frame {
                spectrum: Some(Spectrum {
                    energy_scan: &scan,
                    projected,
                }),
                ..frame
            };
            assert!(evaluate(&space, bad, None, 1e-6).is_err());
            assert!(evaluate(&space, frame, Some(bad), 1e-6).is_err());
        }
    }
    assert!(
        evaluate(
            &space,
            Frame {
                energy: Feature::Projected(3.),
                ..frame
            },
            None,
            1e-6
        )
        .is_err()
    );
    for floor in [0., -1., f64::NAN, f64::INFINITY] {
        assert!(evaluate(&space, frame, None, floor).is_err());
    }
    let wrong = Frame {
        spectrum: Some(Spectrum {
            energy_scan: &[0.; 8],
            projected: true,
        }),
        ..frame
    };
    assert!(std::panic::catch_unwind(|| evaluate(&space, wrong, None, 1e-6)).is_err());
    assert!(std::panic::catch_unwind(|| evaluate(&space, frame, Some(wrong), 1e-6)).is_err());
}

#[test]
fn alternate_accent_floor_never_changes_the_ten_feature_coordinates() {
    let space = Log2Space::new(64., 1024., 2);
    let spectrum = [1e-10; 9];
    let old_scan = [1e-12; 9];
    let current = Frame {
        energy: Feature::Projected(9e-10),
        bus_energy: Feature::Unsupported,
        spectrum: Some(Spectrum {
            energy_scan: &spectrum,
            projected: true,
        }),
    };
    let old = Frame {
        energy: Feature::Observed(9e-12),
        bus_energy: Feature::Unsupported,
        spectrum: Some(Spectrum {
            energy_scan: &old_scan,
            projected: false,
        }),
    };
    let normal = evaluate(&space, current, Some(old), 1e-6).unwrap();
    let insensitive = evaluate(&space, current, Some(old), 1e-3).unwrap();
    assert_eq!(normal.values, insensitive.values);
    assert!(normal.accent_flux > 0.);
    assert_eq!(insensitive.accent_flux, 0.);
}
