//! Per-frame acoustic candidates, before ridge continuity or group association.

use crate::core::log2space::Log2Space;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct TrajectoryFrame {
    pub peak_bins: [Option<usize>; 7],
    pub energy: [f64; 8],
    pub log_envelope: [f64; 8],
    pub spectral_shape_supported: bool,
}

pub(super) fn owner(peak_bins: &[Option<usize>; 7], bin: usize) -> usize {
    peak_bins
        .iter()
        .enumerate()
        .filter_map(|(slot, peak)| peak.map(|p| (p.abs_diff(bin), p, slot)))
        .filter(|(distance, _, _)| *distance <= 2)
        .min()
        .map_or(7, |(_, _, slot)| slot)
}

pub(super) fn partition(
    space: &Log2Space,
    power_scan: &[f32],
    mono_energy: f64,
) -> Option<TrajectoryFrame> {
    space.assert_scan_len_named(power_scan, "trajectory_power_scan");
    if !mono_energy.is_finite()
        || mono_energy < 0.0
        || power_scan.iter().any(|x| !x.is_finite() || *x < 0.0)
    {
        return None;
    }
    let mut peaks = [None; 7];
    let maximum = power_scan.iter().copied().fold(0.0_f32, f32::max);
    for (bin, &power) in power_scan.iter().enumerate() {
        if power == 0.0
            || f64::from(power) < 0.01 * f64::from(maximum)
            || (bin > 0 && power <= power_scan[bin - 1])
            || (bin + 1 < power_scan.len() && power < power_scan[bin + 1])
        {
            continue;
        }
        let left = bin.checked_sub(2).map(|i| power_scan[i]);
        let right = power_scan.get(bin + 2).copied();
        let neighbor = match (left, right) {
            (Some(a), Some(b)) => a.max(b),
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => continue,
        };
        if f64::from(power) - f64::from(neighbor) < 0.1 * f64::from(power) {
            continue;
        }
        // Ascending bin traversal gives stable ties in the bounded energy ranking.
        if let Some(slot) = peaks
            .iter()
            .position(|previous| previous.is_none_or(|i| power > power_scan[i]))
        {
            for index in (slot + 1..peaks.len()).rev() {
                peaks[index] = peaks[index - 1];
            }
            peaks[slot] = Some(bin);
        }
    }
    let mut masses = [0.0; 8];
    for (bin, &power) in power_scan.iter().enumerate() {
        masses[owner(&peaks, bin)] += f64::from(power);
    }
    let total: f64 = masses.iter().sum();
    let mut energy = [0.0; 8];
    if total > 0.0 {
        for (value, mass) in energy.iter_mut().zip(masses) {
            *value = mono_energy * (mass / total);
        }
    } else {
        energy[7] = mono_energy;
    }
    Some(TrajectoryFrame {
        peak_bins: peaks,
        energy,
        log_envelope: energy.map(|value| 0.5 * value.max(1e-12).log2()),
        spectral_shape_supported: total > 0.0 || mono_energy == 0.0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "release kernel timing probe; excludes queues, NSGT and other workers"]
    fn trajectory_kernel_cost_probe() {
        use std::hint::black_box;
        use std::time::Instant;

        for bins_per_oct in [128, 512, 2048] {
            let space = Log2Space::new(1.0, 2.0, bins_per_oct);
            for kind in ["flat", "peaked"] {
                let scan: Vec<f32> = (0..space.n_bins())
                    .map(|i| {
                        if kind == "flat" {
                            1.0
                        } else {
                            (i * 29 % 67) as f32 / 8.0
                        }
                    })
                    .collect();
                for _ in 0..100 {
                    black_box(partition(&space, black_box(&scan), black_box(0.25)).unwrap());
                }
                let mut durations = Vec::with_capacity(6000);
                for _ in 0..6000 {
                    let start = Instant::now();
                    black_box(partition(&space, black_box(&scan), black_box(0.25)).unwrap());
                    durations.push(start.elapsed().as_secs_f64() * 1e6);
                }
                durations.sort_by(f64::total_cmp);
                println!(
                    "trajectory_cost {}",
                    serde_json::json!({
                        "bins": space.n_bins(), "kind": kind, "calls": durations.len(),
                        "median_us": durations[3000], "p99_us": durations[5939],
                        "max_us": durations[5999], "output_bytes": std::mem::size_of::<TrajectoryFrame>(),
                        "full_O04": false
                    })
                );
            }
        }
    }

    #[test]
    fn rational_reference_covers_thresholds_grids_and_conservation() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/temporal_cognition/trajectories.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let scan: Vec<f32> = case["power_scan"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_f64().unwrap() as f32)
                .collect();
            let space = if scan.len() == 1 {
                Log2Space::new(1.0, 1.1, 1)
            } else {
                Log2Space::new(1.0, 2.0, (scan.len() - 1) as u32)
            };
            let energy = case["mono_energy"].as_f64().unwrap();
            let actual = partition(&space, &scan, energy).unwrap();
            for (index, expected) in case["peak_bins"].as_array().unwrap().iter().enumerate() {
                assert_eq!(
                    actual.peak_bins[index],
                    expected.as_u64().map(|i| i as usize)
                );
            }
            for (index, expected) in case["energy"].as_array().unwrap().iter().enumerate() {
                assert!((actual.energy[index] - expected.as_f64().unwrap()).abs() < 1e-13);
            }
            assert!((actual.energy.iter().sum::<f64>() - energy).abs() < 1e-13);
            assert_eq!(
                actual.spectral_shape_supported,
                case["spectral_shape_supported"].as_bool().unwrap()
            );
        }
    }

    #[test]
    fn plateau_edges_and_nearest_bin_ties_have_exact_conserved_ownership() {
        let space = Log2Space::new(1.0, 2.0, 8);
        let frame = partition(&space, &[4., 0., 0., 1., 1., 0., 0., 0., 2.], 8.0).unwrap();
        assert_eq!(
            frame.peak_bins,
            [Some(0), Some(8), Some(3), None, None, None, None]
        );
        assert_eq!(frame.energy, [4., 2., 2., 0., 0., 0., 0., 0.]);
        let tied = partition(&space, &[0., 1., 0., 0.5, 0., 2., 0., 0., 0.], 7.0).unwrap();
        assert_eq!(
            tied.peak_bins,
            [Some(5), Some(1), None, None, None, None, None]
        );
        assert_eq!(tied.energy, [4., 3., 0., 0., 0., 0., 0., 0.]);
        assert_eq!(tied.log_envelope[0], 1.0);
    }

    #[test]
    fn ranking_is_bounded_and_plateau_without_prominence_stays_residual() {
        let space = Log2Space::new(1.0, 2.0, 32);
        let mut scan = vec![0.0; space.n_bins()];
        for index in (0..33).step_by(4) {
            scan[index] = 1.0;
        }
        let frame = partition(&space, &scan, 9.0).unwrap();
        assert_eq!(
            frame.peak_bins,
            [
                Some(0),
                Some(4),
                Some(8),
                Some(12),
                Some(16),
                Some(20),
                Some(24)
            ]
        );
        assert_eq!(frame.energy, [1., 1., 1., 1., 1., 1., 1., 2.]);
        let flat = partition(&space, &vec![1.; space.n_bins()], 4.0).unwrap();
        assert_eq!(flat.peak_bins, [None; 7]);
        assert_eq!(flat.energy, [0., 0., 0., 0., 0., 0., 0., 4.]);
    }

    #[test]
    fn silence_spectral_failure_and_invalid_acquisition_are_distinct() {
        let space = Log2Space::new(1.0, 2.0, 8);
        let zero = vec![0.; space.n_bins()];
        let silent = partition(&space, &zero, 0.0).unwrap();
        assert!(silent.spectral_shape_supported);
        assert_eq!(silent.energy, [0.; 8]);
        assert_eq!(silent.log_envelope, [0.5 * 1e-12_f64.log2(); 8]);
        let unresolved = partition(&space, &zero, 0.25).unwrap();
        assert!(!unresolved.spectral_shape_supported);
        assert_eq!(unresolved.energy[7], 0.25);
        assert!(partition(&space, &zero, f64::NAN).is_none());
        assert!(partition(&space, &zero, -1.0).is_none());
        assert!(partition(&space, &vec![f32::INFINITY; space.n_bins()], 0.0).is_none());
    }

    #[test]
    #[should_panic(expected = "scan length mismatch: trajectory_power_scan")]
    fn input_scan_must_match_the_log2_grid() {
        partition(&Log2Space::new(1.0, 2.0, 8), &[1.], 1.0);
    }
}
