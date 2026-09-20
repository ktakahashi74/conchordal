use super::*;

pub(super) fn config(bus: u8) -> Config {
    Config {
        group: group::Config {
            bus,
            epoch: 2,
            sample_rate: 48000,
            hop: 512,
            means: [0.; 3],
            deviations: [0.05, 4., 1.],
            distance_limit: 1.,
            residual_raw: (-1.0_f64).exp(),
        },
        ridge: ridge::Config {
            means: [0.; 3],
            deviations: [0.05, 4., 1.],
            distance_limit: 1.,
            retirement_sec: 0.25,
        },
        epoch_start: 0,
        group_retirement_samples: 96000,
        inactive_energy_max: 1e-8,
        correlation_window_samples: 12000,
        min_pairs: 8,
        min_coverage: 0.9,
        persistence_hops: 3,
        features: features::Config {
            means: [0.; 2],
            deviations: [1.; 2],
            threshold: 1.,
            rms_floor: 1e-6,
        },
    }
}

pub(super) fn observed(end: u64, scan: &[f32], energy: f64) -> Option<Observation<'_>> {
    Some(Observation {
        power_scan: scan,
        mono_energy: energy,
        source_start: end - 512,
        source_end: end,
        available_end: end,
    })
}

pub(super) fn space() -> Log2Space {
    Log2Space::new(100., 6400., 32)
}

#[test]
fn frontend_descriptors_compress_and_match_a_later_independent_trace() {
    use crate::temporal_cognition::{descriptor, memory, transport::Identity};

    let mut descriptors = Vec::new();
    let mut consumed = Vec::new();
    for offset in [0, 512 * 1024] {
        let grid = space();
        let mut spectrum = vec![0.; grid.n_bins()];
        spectrum[20] = 1.;
        let mut cfg = config(0);
        cfg.epoch_start = offset;
        let mut frontend = Frontend::new(grid, cfg).unwrap();
        let mut span: Option<descriptor::Span> = None;
        let mut owner = None;
        let mut count = 0;
        for step in 1..=320 {
            let end = offset + step * 512;
            let energy = 0.1 + 0.02 * (step as f64 * 0.1).sin();
            let out = frontend
                .advance(end, observed(end, &spectrum, energy))
                .unwrap();
            for update in out.features[..7].iter().flatten() {
                let raw = &update.raw;
                let handle = *owner.get_or_insert(raw.group);
                assert_eq!(raw.group, handle);
                if span.is_none() {
                    span = Some(
                        descriptor::Span::new(descriptor::Config {
                            group: handle,
                            sample_rate: 48000,
                            first_hop_start: raw.start,
                            hop: 512,
                            cadence: 2,
                            span_start: raw.start as f64 / 48000.,
                            span_end: None,
                            scales: [1.; 10],
                            capacity: 128,
                        })
                        .unwrap(),
                    );
                }
                assert_eq!(raw.known_samples, 512);
                assert!(
                    span.as_mut()
                        .unwrap()
                        .push(raw, &[(raw.start, raw.end)], end)
                        .unwrap()
                );
                count += 1;
            }
        }
        let frozen = span.unwrap().finish(offset + 320 * 512).unwrap();
        assert!(frozen.reconstruction_error > 0.);
        let descriptor = frozen.matching().unwrap();
        assert_eq!(descriptor.knots.len(), 128);
        consumed.push(count);
        descriptors.push((descriptor, frozen.available_at as f64 / 48000.));
    }
    let (query_descriptor, query_end) = descriptors.pop().unwrap();
    let (stored, stored_end) = descriptors.pop().unwrap();
    let episode = memory::Episode {
        identity: Identity {
            id: 91,
            generation: 1,
        },
        epoch: 2,
        available_end: stored_end,
        first_observed_end: stored.knots.last().unwrap().end,
        scales: [1.; 10],
        descriptor: stored,
    };
    let report = memory::ordered(
        &memory::Query {
            descriptor: &query_descriptor,
            epoch: 2,
            end: query_end,
            observed_end: query_end,
            scales: [1.; 10],
        },
        &[episode],
    )
    .unwrap();
    let best = report
        .matches
        .iter()
        .filter(|m| m.relation.supported)
        .min_by(|a, b| a.cost.unwrap().total_cmp(&b.cost.unwrap()))
        .unwrap();
    assert!(best.cost.unwrap() < 1e-9);
    assert_eq!(best.applied, [0.; 2]);
    assert_eq!(best.relation.identity.id, 91);
    println!(
        "DESCRIPTOR_FRONTEND {}",
        serde_json::json!({
            "consumed_raw_hops": consumed, "knots_per_trace": 128,
            "best_cost": best.cost, "applied": best.applied,
            "dp_cells": report.dp_cells,
            "input": "two independent frontends with supplied power scans; not waveform recovery"
        })
    );
}

#[test]
fn spectra_preserve_bin_energy_and_source_handles_across_same_hop_births() {
    let grid = space();
    let mut spectrum = vec![0.; grid.n_bins()];
    spectrum[20] = 1.;
    spectrum[80] = 0.7;
    let mut f = Frontend::new(grid, config(0)).unwrap();
    let pointers = f.energy_scans.each_ref().map(|s| s.as_ptr());
    let mut births = 0;
    for step in 1..=40 {
        let mono = 0.1 + (step % 7) as f64 * 0.001;
        let out = f
            .advance(step * 512, observed(step * 512, &spectrum, mono))
            .unwrap();
        let energy = out.energy.unwrap();
        assert!((energy.group_energy.iter().sum::<f64>() - mono).abs() < 1e-12);
        let snapshot = f.energy_snapshot().unwrap();
        assert_eq!(snapshot.end_sample, step * 512);
        assert_eq!(
            snapshot.group_handles[..7],
            out.groups.assignment.group_handles
        );
        assert_eq!(snapshot.group_handles[7].unwrap().generation, 1);
        assert!(snapshot.eligible[7]);
        assert!(out.feature_gaps.iter().all(Option::is_none));
        for (i, update) in out.features.iter().enumerate() {
            if let Some(update) = update {
                assert_eq!(Some(update.raw.group), snapshot.group_handles[i]);
                assert_eq!(update.raw.end, snapshot.end_sample);
                if !snapshot.eligible[i] {
                    assert!(update.raw.values.iter().all(Option::is_none));
                }
            } else {
                assert!(snapshot.group_handles[i].is_none());
            }
        }
        for (bin, &power) in spectrum.iter().enumerate() {
            let expected =
                mono * f64::from(power) / spectrum.iter().map(|&v| f64::from(v)).sum::<f64>();
            let actual: f64 = snapshot.scans.iter().map(|s| s[bin]).sum();
            assert!((actual - expected).abs() < 1e-12);
        }
        for admission in out.groups.admissions.iter().flatten() {
            if admission.rejection.is_none() {
                births += 1;
                for child in admission.children.iter().flatten() {
                    assert!(!snapshot.group_handles.contains(&Some(*child)));
                }
            }
        }
        assert_eq!(pointers, f.energy_scans.each_ref().map(|s| s.as_ptr()));
        let residual = out.groups.assignment.rows[7].unwrap().trajectory;
        assert_eq!(residual.generation, u64::MAX);
        assert!(
            out.ridges
                .current
                .iter()
                .flatten()
                .all(|r| r.handle != residual)
        );
    }
    assert!(births >= 2);
}

#[test]
fn peak_energy_reordering_preserves_identity_and_assignment_owner_slots() {
    let grid = space();
    let mut scan = vec![0.; grid.n_bins()];
    let mut f = Frontend::new(grid, config(0)).unwrap();
    let mut identities = [None; 2];
    for step in 1..=30 {
        scan[20] = if step % 2 == 0 { 1.001 } else { 1. };
        scan[80] = if step % 2 == 0 { 1. } else { 1.001 };
        let out = f
            .advance(step * 512, observed(step * 512, &scan, 0.1))
            .unwrap();
        let partition = out.partition.unwrap();
        for i in 0..2 {
            let bin = partition.peak_bins[i].unwrap();
            let identity_index = usize::from(bin == 80);
            let r = out.ridges.current[i].unwrap();
            assert_eq!(
                *identities[identity_index].get_or_insert(r.handle),
                r.handle
            );
            assert_eq!(out.groups.assignment.rows[i].unwrap().trajectory, r.handle);
        }
    }
    assert_ne!(identities[0], identities[1]);
}

#[test]
fn silence_is_observed_but_does_not_propose_zero_energy_groups() {
    let grid = space();
    let silence = vec![0.; grid.n_bins()];
    let mut f = Frontend::new(grid, config(0)).unwrap();
    for step in 1..=240 {
        let out = f
            .advance(step * 512, observed(step * 512, &silence, 0.))
            .unwrap();
        assert!(out.proposals.candidates.iter().all(Option::is_none));
        assert!(out.groups.admissions.iter().all(Option::is_none));
        assert!(out.energy.unwrap().group_energy.iter().all(|e| *e == 0.));
        assert!(
            f.energy_snapshot()
                .unwrap()
                .scans
                .iter()
                .flatten()
                .all(|e| *e == 0.)
        );
        assert_eq!(out.groups.assignment.rows[7].unwrap().weights[7], 1.);
    }
}

#[test]
fn missing_and_unknown_spectral_shape_do_not_retire_groups_as_silence() {
    let grid = space();
    let mut scan = vec![0.; grid.n_bins()];
    scan[20] = 1.;
    let zero = vec![0.; grid.n_bins()];
    let mut f = Frontend::new(grid, config(0)).unwrap();
    let mut end = 0;
    for _ in 0..10 {
        end += 512;
        f.advance(end, observed(end, &scan, 0.1)).unwrap();
    }
    let before = f.lifecycle.eligible_handles();
    assert!(before.iter().any(Option::is_some));
    for _ in 0..200 {
        end += 512;
        let out = f.advance(end, None).unwrap();
        assert!(out.groups.retired.iter().all(Option::is_none));
        assert!(out.ridges.retired.iter().all(Option::is_none));
        assert!(out.energy.is_none());
        assert!(f.energy_snapshot().is_none());
    }
    assert_eq!(f.lifecycle.eligible_handles(), before);
    for _ in 0..200 {
        end += 512;
        let out = f.advance(end, observed(end, &zero, 0.1)).unwrap();
        assert!(out.groups.retired.iter().all(Option::is_none));
        let energy = out.energy.unwrap();
        assert!(!energy.spectral_shape_supported);
        assert_eq!(energy.group_energy[7], 0.1);
        assert!(f.energy_snapshot().is_none());
    }
    let mut retired = 0;
    for _ in 0..200 {
        end += 512;
        let out = f.advance(end, observed(end, &zero, 0.)).unwrap();
        retired += out.groups.retired.iter().flatten().count();
    }
    assert!(retired > 0);
    assert!(f.lifecycle.eligible_handles().iter().all(Option::is_none));
}

#[test]
fn invalid_epoch_inputs_fail_closed_and_new_epoch_does_not_inherit_identity() {
    let grid = space();
    let mut scan = vec![0.; grid.n_bins()];
    scan[20] = 1.;
    let mut f = Frontend::new(grid.clone(), config(0)).unwrap();
    for step in 1..=3 {
        f.advance(step * 512, observed(step * 512, &scan, 0.1))
            .unwrap();
    }
    assert!(f.energy_snapshot().is_some());
    scan[5] = f32::NAN;
    assert!(f.advance(2048, observed(2048, &scan, 0.1)).is_err());
    assert!(f.failure.is_some());
    assert!(f.energy_snapshot().is_none());
    scan[5] = 0.;
    assert!(f.advance(2560, observed(2560, &scan, 0.1)).is_err());
    let mut cfg = config(1);
    cfg.group.epoch = 3;
    cfg.epoch_start = 2560;
    let mut new_epoch = Frontend::new(grid, cfg).unwrap();
    let out = new_epoch.advance(3072, observed(3072, &scan, 0.1)).unwrap();
    assert!(
        out.groups
            .assignment
            .rows
            .iter()
            .flatten()
            .all(|r| r.trajectory.bus == 1 && r.trajectory.epoch == 3)
    );
    assert!(
        out.groups
            .assignment
            .group_handles
            .iter()
            .all(Option::is_none)
    );
    assert!(new_epoch.advance(3072, None).is_err());
    let mut bad = config(0);
    bad.group.deviations[0] = -1.;
    assert!(Frontend::new(space(), bad).is_err());
}

#[test]
fn actual_nsgt_hops_feed_two_independent_buses_with_warmup_and_conserved_energy() {
    use crate::core::nsgt_kernel::{NsgtKernelLog2, NsgtLog2Config, PowerMode};
    use crate::core::nsgt_rt::RtNsgtKernelLog2;
    let grid = Log2Space::new(100., 6400., 24);
    let kernel = NsgtKernelLog2::new(
        NsgtLog2Config {
            fs: 48000.,
            overlap: 0.75,
            nfft_override: Some(2048),
            ..Default::default()
        },
        grid.clone(),
        None,
        PowerMode::Coherent,
    );
    let mut analysis = [
        RtNsgtKernelLog2::new(kernel.clone()),
        RtNsgtKernelLog2::new(kernel),
    ];
    let mut frontends = [
        Frontend::new(grid.clone(), config(0)).unwrap(),
        Frontend::new(grid, config(1)).unwrap(),
    ];
    let mut audio = [vec![0.; 512], vec![0.; 512]];
    let mut admitted = [0usize; 2];
    let mut complete = [0usize; 2];
    let mut max_error: f64 = 0.;
    let mut raw_count = [0usize; 2];
    let mut accent_count = [0usize; 2];
    for step in 1..=240 {
        for i in 0..512 {
            let t = ((step - 1) * 512 + i) as f64 / 48000.;
            let amp = 0.05 * (1.0 + 0.2 * (std::f64::consts::TAU * 2.0 * t).sin());
            audio[0][i] =
                (amp * ((std::f64::consts::TAU * 500.0 * t).sin()
                    + 0.5 * (std::f64::consts::TAU * 750.0 * t).sin())) as f32;
        }
        for bus in 0..2 {
            let energy = audio[bus]
                .iter()
                .map(|&x| f64::from(x).powi(2))
                .sum::<f64>()
                / 512.;
            let window = analysis[bus].nfft();
            assert_eq!(analysis[bus].hop(), 512);
            let power = analysis[bus].process_hop(&audio[bus]);
            let input = (step * 512 >= window).then(|| Observation {
                power_scan: power,
                mono_energy: energy,
                source_start: (step * 512 - window) as u64,
                source_end: step as u64 * 512,
                available_end: step as u64 * 512,
            });
            let out = frontends[bus].advance(step as u64 * 512, input).unwrap();
            for update in out.features.iter().flatten() {
                let raw = &update.raw;
                assert_eq!(raw.group.bus as usize, bus);
                assert_eq!(raw.end, step as u64 * 512);
                if input.is_some() {
                    let expected_start = (step * 512 - window) as u64;
                    assert_eq!(
                        raw.source_start,
                        if raw.values[3..6].iter().any(Option::is_some) {
                            expected_start - 512
                        } else {
                            expected_start
                        }
                    );
                    assert_eq!(raw.source_end, raw.end);
                    assert_eq!(raw.available_end, raw.end);
                    raw_count[bus] += 1;
                }
                if let Some(a) = update.detector.and_then(|d| d.accent) {
                    assert_eq!(a.event_end + 512, raw.end);
                    assert_eq!(a.source_start, raw.end - window as u64 - 3 * 512);
                    assert!(!a.at_cut(a.event_end));
                    assert!(a.at_cut(raw.end));
                    accent_count[bus] += 1;
                }
            }
            admitted[bus] += out
                .groups
                .admissions
                .iter()
                .flatten()
                .filter(|a| a.rejection.is_none())
                .count();
            if let Some(result) = out.energy {
                complete[bus] += 1;
                let error = (result.group_energy.iter().sum::<f64>() - energy).abs();
                assert!(error < 1e-12);
                max_error = max_error.max(error);
                assert!(
                    out.groups
                        .assignment
                        .rows
                        .iter()
                        .flatten()
                        .all(|r| r.trajectory.bus as usize == bus)
                );
            }
        }
    }
    assert_eq!(complete, [237, 237]);
    assert!(admitted[0] > 0);
    assert_eq!(admitted[1], 0);
    assert!(raw_count[0] > raw_count[1]);
    assert_eq!(raw_count[1], 237);
    assert_eq!(accent_count[1], 0);
    println!(
        "acoustic_nsgt_trace {}",
        serde_json::json!({"hops_per_bus":240,"fully_supported_per_bus":complete,"admitted_per_bus":admitted,"raw_descriptors_per_bus":raw_count,"accents_per_bus":accent_count,"max_energy_error":max_error,"fs":48000,"hop":512,"nfft":2048,"bins":frontends[0].space.n_bins(),"fixture":"bus0 synthetic AM500+750Hz, bus1 known silence; supplied numeric scales; not source recovery acceptance"})
    );
}

#[test]
#[should_panic(expected = "trajectory_power_scan")]
fn scan_alignment_remains_a_hard_boundary() {
    let grid = space();
    let bad = vec![0.; grid.n_bins() - 1];
    Frontend::new(grid, config(0))
        .unwrap()
        .advance(512, observed(512, &bad, 0.))
        .ok();
}

#[test]
fn gaps_are_single_missing_intervals_and_break_feature_differences() {
    let grid = space();
    let scan = vec![0.; grid.n_bins()];
    let mut f = Frontend::new(grid, config(0)).unwrap();
    f.advance(512, observed(512, &scan, 0.)).unwrap();
    let skipped = f.advance(2048, observed(2048, &scan, 0.)).unwrap();
    let gap = skipped.feature_gaps[7].unwrap();
    assert_eq!((gap.start, gap.end, gap.known_samples), (512, 1536, 0));
    assert!(gap.values.iter().all(Option::is_none));
    let raw = &skipped.features[7].as_ref().unwrap().raw;
    assert_eq!((raw.start, raw.end, raw.known_samples), (1536, 2048, 512));
    assert!(raw.values[3..6].iter().all(Option::is_none));
    let missing = f.advance(4096, None).unwrap();
    assert!(missing.feature_gaps.iter().all(Option::is_none));
    let raw = &missing.features[7].as_ref().unwrap().raw;
    assert_eq!((raw.start, raw.end, raw.known_samples), (2048, 4096, 0));
    assert!(raw.values.iter().all(Option::is_none));
    let next = f.advance(4608, observed(4608, &scan, 0.)).unwrap();
    assert!(
        next.features[7].as_ref().unwrap().raw.values[3..6]
            .iter()
            .all(Option::is_none)
    );
}

#[test]
fn missing_shape_keeps_scalar_residual_without_resolved_silence_features() {
    let grid = space();
    let mut scan = vec![0.; grid.n_bins()];
    scan[20] = 1.;
    let mut f = Frontend::new(grid, config(0)).unwrap();
    for step in 1..=20 {
        f.advance(step * 512, observed(step * 512, &scan, 0.1))
            .unwrap();
    }
    scan.fill(0.);
    let out = f.advance(21 * 512, observed(21 * 512, &scan, 0.1)).unwrap();
    assert!(f.energy_snapshot().is_none());
    assert!(out.features[..7].iter().any(Option::is_some));
    for update in out.features[..7].iter().flatten() {
        assert!(update.raw.values.iter().all(Option::is_none));
        assert!(update.detector.is_none_or(|d| !d.detector_coverage));
    }
    let residual = &out.features[7].as_ref().unwrap().raw;
    assert_eq!(residual.values[2], Some(0.1_f64.sqrt().log2()));
    assert_eq!(residual.values[6], Some(1.));
    assert!(residual.values[0].is_none() && residual.values[5].is_none());
}

#[test]
fn invalid_source_provenance_fails_epoch_without_feature_publication() {
    for variant in 0..3 {
        let grid = space();
        let scan = vec![0.; grid.n_bins()];
        let mut f = Frontend::new(grid, config(0)).unwrap();
        let mut o = observed(512, &scan, 0.).unwrap();
        match variant {
            0 => o.source_start = 1,
            1 => o.source_end = 511,
            _ => o.available_end = 511,
        }
        assert!(f.advance(512, Some(o)).is_err());
        assert!(f.energy_snapshot().is_none());
        assert!(f.advance(1024, observed(1024, &scan, 0.)).is_err());
    }
}

#[test]
fn unknown_resolved_shape_does_not_add_intervening_group_observation_support() {
    use std::collections::BTreeMap;
    let grid = space();
    let mut scan = vec![0.; grid.n_bins()];
    let mut f = Frontend::new(grid, config(0)).unwrap();
    let mut totals = BTreeMap::new();
    let mut endpoints = BTreeMap::new();
    let mut pre_gap_handles = Vec::new();
    let mut checked = 0;
    for step in 1..=80 {
        scan[20] = if step == 21 { 0. } else { 1. };
        let energy = if step <= 24 || step % 4 < 2 {
            0.01
        } else {
            0.16
        };
        let out = f
            .advance(step * 512, observed(step * 512, &scan, energy))
            .unwrap();
        for update in out.features[..7].iter().flatten() {
            let raw = &update.raw;
            let count = totals.entry(raw.group).or_insert(0u64);
            if raw.values[2].is_some() {
                *count += raw.known_samples;
            }
            endpoints.insert((raw.group, raw.end), *count);
            if step == 20 {
                pre_gap_handles.push(raw.group);
            }
            if let Some(a) = update.detector.and_then(|d| d.accent)
                && a.event_end > 21 * 512
                && pre_gap_handles.contains(&a.group)
            {
                assert_eq!(a.observed_prefix, endpoints[&(a.group, a.event_end)]);
                checked += 1;
            }
        }
    }
    assert!(
        checked > 0,
        "fixture must observe a retained group after unknown spectral shape"
    );
}

#[test]
fn split_children_start_without_parent_features_and_superseded_handles_are_masked() {
    let grid = space();
    let mut scan = vec![0.; grid.n_bins()];
    let mut f = Frontend::new(grid, config(0)).unwrap();
    let mut superseded = Vec::new();
    let mut children = Vec::new();
    let mut splits = 0;
    let mut masked = 0;
    for step in 1..=200 {
        let phase = (step as f64 * 0.4).sin();
        scan[20] = (1. + 0.3 * phase) as f32;
        scan[80] = if step <= 50 {
            scan[20]
        } else {
            (1. - 0.3 * phase) as f32
        };
        let mono = if step <= 50 {
            0.1 * (1. + 0.3 * phase)
        } else {
            0.1
        };
        let out = f
            .advance(step * 512, observed(step * 512, &scan, mono))
            .unwrap();
        for update in out.features.iter().flatten() {
            if children.contains(&update.raw.group) {
                assert!(update.raw.values[3..6].iter().all(Option::is_none));
            }
            if superseded.contains(&update.raw.group) {
                assert!(update.raw.values.iter().all(Option::is_none));
                masked += 1;
            }
        }
        children.clear();
        for admission in out
            .groups
            .admissions
            .iter()
            .flatten()
            .filter(|a| a.rejection.is_none())
        {
            if admission.children.iter().flatten().count() == 2 {
                splits += 1;
            }
            children.extend(admission.children.iter().flatten().copied());
        }
        superseded.extend(out.groups.superseded.iter().flatten().map(|g| g.handle));
    }
    assert!(splits > 0 && masked > 0, "splits={splits} masked={masked}");
}

#[test]
#[ignore = "release composed acoustic frontend cost; excludes NSGT, beam and full O04"]
fn acoustic_frontend_cost_probe() {
    use std::{hint::black_box, time::Instant};
    for bins_per_oct in [32, 128] {
        let grid = Log2Space::new(100., 6400., bins_per_oct);
        let bins = grid.n_bins();
        let peaks: [usize; 7] = std::array::from_fn(|i| (i + 1) * bins / 8);
        let mut power = vec![0.001; bins];
        let mut f = Frontend::new(grid, config(0)).unwrap();
        let mut times = Vec::with_capacity(6000);
        let mut ridge_evaluations = 0;
        let mut group_evaluations = 0;
        let mut accepted = 0;
        let mut rejected = 0;
        let mut retired = 0;
        for step in 1..=6100 {
            for (i, &bin) in peaks.iter().enumerate() {
                power[bin] = (1.0 + 0.1 * (step as f64 * 0.03 + i as f64 * 0.1).sin()) as f32;
            }
            let mono = 0.03 * (1.0 + 0.1 * (step as f64 * 0.03).sin());
            let start = Instant::now();
            let out = black_box(
                f.advance(
                    step * 512,
                    observed(step * 512, black_box(&power), black_box(mono)),
                )
                .unwrap(),
            );
            let elapsed = start.elapsed().as_secs_f64() * 1e6;
            assert_eq!(out.groups.assignment.rows.iter().flatten().count(), 8);
            assert!(
                out.ridges.distance_evaluations <= 98
                    && out.groups.assignment.distance_evaluations <= 896
            );
            if step > 100 {
                times.push(elapsed);
                ridge_evaluations += out.ridges.distance_evaluations;
                group_evaluations += out.groups.assignment.distance_evaluations;
                accepted += out
                    .groups
                    .admissions
                    .iter()
                    .flatten()
                    .filter(|a| a.rejection.is_none())
                    .count();
                rejected += out
                    .groups
                    .admissions
                    .iter()
                    .flatten()
                    .filter(|a| a.rejection.is_some())
                    .count();
                retired += out.groups.retired.iter().flatten().count();
            }
        }
        times.sort_by(f64::total_cmp);
        println!(
            "acoustic_frontend_cost {}",
            serde_json::json!({"bins":bins,"calls":times.len(),"current_trajectories":8,"median_us":times[3000],"p99_us":times[5939],"max_us":times[5999],"ridge_distance_evaluations":ridge_evaluations,"group_distance_evaluations":group_evaluations,"admitted":accepted,"capacity_or_parent_rejections":rejected,"retired":retired,"frontend_header_bytes":std::mem::size_of::<Frontend>(),"output_bytes":std::mem::size_of::<Output>(),"feature_previous_scan_payload_bytes":8*bins*std::mem::size_of::<f64>(),"energy_scan_buffer_bytes":f.energy_scans.iter().map(|v|v.capacity()*std::mem::size_of::<f64>()).sum::<usize>(),"full_O04":false,"scope":"single bus composed frontend, 7 peaks plus nonzero residual; spectra and mono input preparation outside timing; NSGT/beam/context/queues/device excluded; not saturated full-worker inventory"})
        );
    }
}
