use super::super::tests::{config, observed, space};
use super::*;
use std::collections::{BTreeMap, BTreeSet};

fn settings(capacity: usize) -> Settings {
    Settings {
        forecast: None,
        capacity,
        window_samples: 48000 * 32,
        rebuild_admissions: 1024,
        peak_separation: 1. / 24.,
        grouping: groupings::Controls {
            tolerance: 0.1,
            integers_234_only: false,
            strict_integer: false,
            one_skip_words: false,
        },
    }
}

fn pulse(step: u64) -> f64 {
    if (8..10).contains(&(step % 24)) {
        0.16
    } else {
        0.01
    }
}

#[test]
fn ordinary_acoustic_recurrence_reports_public_within_timing_without_private_actions() {
    let grid = space();
    let mut scan = vec![0.; grid.n_bins()];
    scan[20] = 1.;
    let mut model = Recurrence::new(grid, config(0), settings(128)).unwrap();
    let mut positive = 0;
    let mut maximum_records = 0;
    let mut bytes = None;
    let mut density_frames = 0;
    let mut grouping_frames = 0;
    for step in 1..=720 {
        let end = step * 512;
        model
            .advance(end, end, observed(end, &scan, pulse(step)))
            .unwrap();
        let snapshot = model.diagnostics().unwrap();
        let timing = snapshot.timing.unwrap();
        assert_eq!(timing.window[1], end);
        assert!(timing.record_bytes <= 64);
        assert_eq!(*bytes.get_or_insert(timing.owned_bytes), timing.owned_bytes);
        for g in snapshot.groups.iter().flatten() {
            density_frames += usize::from(g.active);
            grouping_frames += usize::from(g.grouping.is_some());
        }
        for group in timing.groups.iter().flatten() {
            assert!(group.group.generation > 1);
            maximum_records = maximum_records.max(group.within.records);
            if group.within.supported {
                positive += 1;
                assert!(group.within.coverage >= 0.9);
                assert_eq!(group.within.reference, group.group);
                assert_eq!(group.within.family, auditory_timing::Family::Periodic);
                assert!((group.within.bins.iter().sum::<f64>() - 1.).abs() < 1e-10);
                assert_eq!(group.within.overflow, 0.);
                let values = group.timing_features[2];
                assert!(group.within.mode_count.is_some());
                assert_eq!(values[8], Some(1.));
                assert_eq!(values[9], Some(1.));
                assert_eq!(values[10], group.within.residual_dispersion);
                assert_eq!(values[12], Some(0.));
                assert_eq!(values[13], Some(group.within.coverage));
                assert!(values.iter().flatten().all(|x| x.is_finite()));
            }
        }
    }
    assert!(
        positive > 0 && maximum_records >= 4,
        "positive={positive} records={maximum_records}"
    );
    assert!(
        density_frames > 0 && grouping_frames > 0,
        "active={density_frames} grouping={grouping_frames}"
    );
    let before = serde_json::to_value(model.diagnostics().unwrap().timing).unwrap();
    model.finish(720 * 512 + 1000).unwrap();
    assert_eq!(
        before,
        serde_json::to_value(model.diagnostics().unwrap().timing).unwrap()
    );
    println!(
        "public timing ordinary recurrence: supported_frames={positive} maximum_records={maximum_records} owned_bytes={}",
        bytes.unwrap()
    );
}

#[test]
fn energy_accents_reach_one_original_owner_and_newborns_start_without_credit() {
    let grid = space();
    let mut scan = vec![0.; grid.n_bins()];
    scan[20] = 1.;
    let mut model = Recurrence::new(grid, config(0), settings(8)).unwrap();
    let buffers = model.slots.each_ref().map(|s| s.estimator.storage_layout());
    let mut events = BTreeSet::new();
    let mut counts = BTreeMap::<Handle, u64>::new();
    let mut weights = BTreeMap::<Handle, f64>::new();
    let (mut births, mut periodic, mut grouped, mut cap_loss, mut inserted) = (0, 0, 0, 0, 0);
    for step in 1..=360 {
        let end = step * 512;
        let output = model
            .advance(end, end, observed(end, &scan, pulse(step)))
            .unwrap();
        for update in output.features.iter().flatten() {
            if let Some(a) = update.detector.as_ref().and_then(|d| d.accent) {
                assert!(events.insert((a.group, a.event_start, a.event_end)));
                *counts.entry(a.group).or_default() += 1;
                *weights.entry(a.group).or_default() += a.weight;
            }
        }
        let frame = model.snapshot().unwrap();
        assert_eq!((frame.end_sample, frame.received_at), (end, end));
        for (i, g) in frame
            .evidence_groups
            .iter()
            .enumerate()
            .filter_map(|(i, g)| g.as_ref().map(|g| (i, g)))
        {
            assert_eq!(
                Some(g.ledger.group),
                output.groups.assignment.group_handles[i]
            );
            assert_eq!(
                g.ledger.cumulative_count,
                *counts.get(&g.ledger.group).unwrap_or(&0)
            );
            assert!(
                (g.ledger.cumulative_weight - *weights.get(&g.ledger.group).unwrap_or(&0.)).abs()
                    < 1e-12
            );
            assert_eq!(g.ledger.received_at, end);
            assert_eq!(g.period.group, g.ledger.group);
            assert!(g.ledger.retained_accents <= 8);
            inserted += g.period.work.inserted_pairs;
            periodic += usize::from(g.period.peaks.iter().any(Option::is_some));
            grouped += usize::from(g.grouping.is_some_and(|v| v.proposal_count() > 0));
            cap_loss += usize::from(g.ledger.capacity_evicted_through.is_some());
            if let Some(v) = g.grouping {
                assert_eq!(v.group, g.ledger.group);
                assert!(v.refreshed_at <= end);
            }
            if g.association_known {
                assert!(g.acoustic_eligible);
            }
        }
        assert_eq!(frame.residual.group.generation, 1);
        assert_eq!(
            frame.residual.cumulative_count,
            *counts.get(&frame.residual.group).unwrap_or(&0)
        );
        for (i, born) in frame
            .newborn
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.map(|b| (i, b)))
        {
            births += 1;
            assert_eq!(born.cumulative_count, 0);
            assert_eq!(born.cumulative_weight, 0.);
            assert_eq!(born.retained_accents, 0);
            assert_eq!(frame.next_owners[i], Some(born.group));
            assert!(
                !frame
                    .evidence_groups
                    .iter()
                    .flatten()
                    .any(|g| g.ledger.group == born.group)
            );
            assert!(
                model.slots[i]
                    .estimator
                    .view()
                    .peaks
                    .iter()
                    .all(Option::is_none)
            );
            assert!(model.slots[i].grouping.snapshot().is_none());
        }
        assert_eq!(
            buffers,
            model.slots.each_ref().map(|s| s.estimator.storage_layout())
        );
    }
    assert!(
        births > 0 && periodic > 0 && grouped > 0 && cap_loss > 0 && inserted > 0,
        "births={births} periodic={periodic} grouped={grouped} cap_loss={cap_loss} inserted={inserted}"
    );
    println!(
        "RECURRENCE_OWNER_TRACE {}",
        serde_json::json!({"hops":360,"unique_accents":events.len(),"births":births,"periodic_frames":periodic,"grouped_frames":grouped,"capacity_limited_frames":cap_loss,"inserted_pairs_reported":inserted,"capacity":8})
    );
}

#[test]
fn missing_time_expires_banks_without_retirement_and_known_silence_retires_owners() {
    let grid = space();
    let mut scan = vec![0.; grid.n_bins()];
    scan[20] = 1.;
    let zero = vec![0.; grid.n_bins()];
    let mut model = Recurrence::new(grid, config(0), settings(128)).unwrap();
    for step in 1..=200 {
        model
            .advance(
                step * 512,
                step * 512,
                observed(step * 512, &scan, pulse(step)),
            )
            .unwrap();
    }
    let before = model.slots.each_ref().map(|s| s.owner);
    let totals = model
        .slots
        .each_ref()
        .map(|s| s.estimator.ledger_summary().cumulative_count);
    assert!(totals.iter().any(|&n| n > 0));
    let mut end = 200 * 512 + 48000 * 40;
    let out = model.advance(end, end, None).unwrap();
    assert!(out.groups.retired.iter().all(Option::is_none));
    assert_eq!(model.snapshot().unwrap().next_owners, before);
    for (i, g) in model
        .snapshot()
        .unwrap()
        .evidence_groups
        .iter()
        .enumerate()
        .filter_map(|(i, g)| g.map(|g| (i, g)))
    {
        assert_eq!(g.ledger.retained_accents, 0);
        assert_eq!(g.ledger.cumulative_count, totals[i]);
        assert!(!g.association_known);
        assert!(g.period.peaks.iter().all(Option::is_none));
        assert_eq!(g.grouping.unwrap().proposal_count(), 0);
    }
    let mut retired = BTreeSet::new();
    for _ in 0..200 {
        end += 512;
        let out = model.advance(end, end, observed(end, &zero, 0.)).unwrap();
        for retired_group in out.groups.retired.iter().flatten() {
            assert!(retired.insert(retired_group.group.handle));
            assert!(
                model
                    .snapshot()
                    .unwrap()
                    .evidence_groups
                    .iter()
                    .flatten()
                    .any(|g| g.ledger.group == retired_group.group.handle)
            );
        }
    }
    assert_eq!(retired, before.into_iter().flatten().collect());
    assert!(
        model
            .snapshot()
            .unwrap()
            .next_owners
            .iter()
            .all(Option::is_none)
    );
    let mut newborn = 0;
    for step in 1..=40 {
        end += 512;
        model
            .advance(end, end, observed(end, &scan, pulse(step)))
            .unwrap();
        for g in model.snapshot().unwrap().newborn.iter().flatten() {
            assert!(!retired.contains(&g.group));
            assert_eq!(g.cumulative_count, 0);
            newborn += 1;
        }
    }
    assert!(newborn > 0);
}

#[test]
fn split_parents_keep_their_final_credit_and_children_reuse_empty_caches() {
    let grid = space();
    let mut scan = vec![0.; grid.n_bins()];
    let mut cfg = config(0);
    cfg.features.threshold = 0.05;
    cfg.features.deviations = [0.1; 2];
    let mut model = Recurrence::new(grid, cfg, settings(16)).unwrap();
    let buffers = model.slots.each_ref().map(|s| s.estimator.storage_layout());
    let mut counts = BTreeMap::<Handle, u64>::new();
    let mut superseded = BTreeMap::<Handle, u64>::new();
    let (mut splits, mut observed_old, mut replacements) = (0, 0, 0);
    for step in 1..=240 {
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
        let end = step * 512;
        let out = model.advance(end, end, observed(end, &scan, mono)).unwrap();
        for update in out.features.iter().flatten() {
            if let Some(a) = update.detector.as_ref().and_then(|d| d.accent) {
                *counts.entry(a.group).or_default() += 1;
            }
        }
        let frame = model.snapshot().unwrap();
        for g in frame.evidence_groups.iter().flatten() {
            assert_eq!(
                g.ledger.cumulative_count,
                *counts.get(&g.ledger.group).unwrap_or(&0)
            );
            if let Some(&frozen) = superseded.get(&g.ledger.group) {
                assert!(!g.acoustic_eligible);
                assert!(!g.association_known);
                assert_eq!(g.ledger.cumulative_count, frozen);
                observed_old += 1;
            }
        }
        for old in out.groups.superseded.iter().flatten() {
            superseded.insert(old.handle, *counts.get(&old.handle).unwrap_or(&0));
        }
        splits += out
            .groups
            .admissions
            .iter()
            .flatten()
            .filter(|a| a.children.iter().flatten().count() == 2)
            .count();
        for (i, born) in frame
            .newborn
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.map(|b| (i, b)))
        {
            assert_eq!(born.cumulative_count, 0);
            if frame.evidence_groups[i].is_some() {
                replacements += 1;
            }
        }
        assert_eq!(
            buffers,
            model.slots.each_ref().map(|s| s.estimator.storage_layout())
        );
    }
    assert!(
        splits > 0 && observed_old > 0,
        "splits={splits} observed_old={observed_old} replacements={replacements}"
    );
    assert!(superseded.values().any(|&n| n > 0));
}

#[test]
fn bus_epoch_and_actual_availability_remain_separate_from_acoustic_event_time() {
    let grid = space();
    let mut scan = vec![0.; grid.n_bins()];
    scan[20] = 1.;
    let silence = vec![0.; grid.n_bins()];
    let mut left = Recurrence::new(grid.clone(), config(0), settings(16)).unwrap();
    let mut right = Recurrence::new(grid.clone(), config(1), settings(16)).unwrap();
    for step in 1..=120 {
        let end = step * 512;
        let received = end + 1536;
        let input = Observation {
            available_end: received,
            ..observed(end, &scan, pulse(step)).unwrap()
        };
        left.advance(end, received, Some(input)).unwrap();
        right
            .advance(end, received, observed(end, &silence, 0.))
            .unwrap();
        for g in left.snapshot().unwrap().evidence_groups.iter().flatten() {
            assert_eq!(g.ledger.group.bus, 0);
            assert_eq!(g.ledger.received_at, received);
        }
        assert!(
            right
                .snapshot()
                .unwrap()
                .next_owners
                .iter()
                .all(Option::is_none)
        );
        assert_eq!(right.snapshot().unwrap().residual.cumulative_count, 0);
    }
    let end = 121 * 512;
    let future = Observation {
        available_end: end + 4096,
        ..observed(end, &scan, 0.1).unwrap()
    };
    assert!(left.advance(end, end + 1536, Some(future)).is_err());
    assert!(left.snapshot().is_none());
    assert!(left.advance(end + 512, end + 2048, None).is_err());
    let mut cfg = config(0);
    cfg.group.epoch = 3;
    cfg.epoch_start = end + 2048;
    let mut restart = Recurrence::new(grid, cfg, settings(16)).unwrap();
    restart
        .advance(cfg.epoch_start + 512, cfg.epoch_start + 512, None)
        .unwrap();
    let fresh = restart.snapshot().unwrap();
    assert_eq!(fresh.residual.group.epoch, 3);
    assert_eq!(fresh.residual.cumulative_count, 0);
    assert!(fresh.next_owners.iter().all(Option::is_none));
}

#[test]
fn actual_nsgt_pulse_silence_and_steady_controls_reach_separate_inventories() {
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
    let mut analysis: [_; 6] = std::array::from_fn(|_| RtNsgtKernelLog2::new(kernel.clone()));
    let mut models = Vec::new();
    for (index, bus) in [0, 1, 0, 0, 1, 0].into_iter().enumerate() {
        let mut cfg = config(bus);
        if index < 3 {
            cfg.features.threshold = 0.05;
            cfg.features.deviations = [0.2; 2];
        }
        let mut settings = settings(128);
        settings.forecast = Some(crate::config::TemporalPeriodConfig {
            model: if index < 3 {
                crate::config::ArrivalModel::Hazard
            } else {
                crate::config::ArrivalModel::Periodic
            },
            coefficients: [0.; 18],
            means: [0.; 8],
            deviations: [1.; 8],
            horizon_sec: 0.1,
        });
        models.push(Recurrence::new(grid.clone(), cfg, settings).unwrap());
    }
    let mut audio: [_; 6] = std::array::from_fn(|_| vec![0.; 512]);
    let mut accents = [0usize; 6];
    let mut periodic = [0usize; 6];
    let mut grouped = [0usize; 6];
    let mut target_period = [0usize; 6];
    let mut forecasts = [0usize; 6];
    for step in 1..=480 {
        for i in 0..512 {
            let t = ((step - 1) * 512 + i) as f64 / 48000.;
            let amp = if (8..10).contains(&(step % 24)) {
                0.04
            } else {
                0.01
            };
            audio[0][i] = (amp * (std::f64::consts::TAU * 500. * t).sin()) as f32;
            audio[2][i] = (0.01 * (std::f64::consts::TAU * 500. * t).sin()) as f32;
        }
        let (diagnostic, baseline) = audio.split_at_mut(3);
        for (destination, source) in baseline.iter_mut().zip(diagnostic) {
            destination.copy_from_slice(source);
        }
        for bus in 0..6 {
            let end = step as u64 * 512;
            let energy = audio[bus]
                .iter()
                .map(|&x| f64::from(x).powi(2))
                .sum::<f64>()
                / 512.;
            let window = analysis[bus].nfft() as u64;
            let power = analysis[bus].process_hop(&audio[bus]);
            let input = (end >= window).then(|| Observation {
                power_scan: power,
                mono_energy: energy,
                source_start: end - window,
                source_end: end,
                available_end: end,
            });
            let out = models[bus].advance(end, end, input).unwrap();
            let issues = models[bus].arrival_issues(end);
            assert!(
                models[bus]
                    .arrival_issues(end + 1)
                    .iter()
                    .all(Option::is_none)
            );
            for (i, issue) in issues.iter().enumerate() {
                if let Some(issue) = issue {
                    assert_eq!(models[bus].slots[i].owner, Some(issue.group));
                    assert_eq!(issue.issued_at, end);
                    let projected = issue.project(end).unwrap().finish().unwrap();
                    let ordinary = models[bus].diagnostics().unwrap().groups[i]
                        .unwrap()
                        .forecast
                        .unwrap();
                    assert_eq!(projected.reset_unknown, ordinary.reset_unknown);
                    let point = ordinary
                        .probability
                        .filter(|p| !ordinary.reset_unknown && p[0] == p[1]);
                    assert_eq!(projected.probability.value(), point.map(|p| p[0]));
                }
            }
            accents[bus] += out
                .features
                .iter()
                .flatten()
                .filter(|u| u.detector.as_ref().is_some_and(|d| d.accent.is_some()))
                .count();
            for g in models[bus]
                .snapshot()
                .unwrap()
                .evidence_groups
                .iter()
                .flatten()
            {
                assert_eq!(g.ledger.group.bus as usize, [0, 1, 0, 0, 1, 0][bus]);
                if let Some(f) = g.forecast {
                    assert_eq!(f.group, g.ledger.group);
                    assert!(f.valid_for(
                        g.ledger.group,
                        models[bus].settings.forecast.unwrap().model,
                        end
                    ));
                    if f.probability.is_some() {
                        forecasts[bus] += 1;
                    }
                }
                periodic[bus] += usize::from(g.period.peaks.iter().any(Option::is_some));
                grouped[bus] += usize::from(g.grouping.is_some_and(|v| v.proposal_count() > 0));
                target_period[bus] += usize::from(
                    g.period
                        .peaks
                        .iter()
                        .flatten()
                        .any(|p| (p.period_seconds / 0.256).log2().abs() < 0.04),
                );
            }
        }
    }
    println!(
        "RECURRENCE_NSGT_TRACE {}",
        serde_json::json!({"hops_per_instance":480,"sample_rate":48000,"hop":512,"nfft":2048,"instances":["diagnostic_pulse_bus0","diagnostic_silence_bus1","diagnostic_steady_bus0","baseline_pulse_bus0","baseline_silence_bus1","baseline_steady_bus0"],"thresholds":[0.05,0.05,0.05,1.,1.,1.],"salience_deviations":[0.2,0.2,0.2,1.,1.,1.],"accents":accents,"arrival_forecasts":forecasts,"periodic_frames":periodic,"grouped_frames":grouped,"frames_with_0_256s_candidate":target_period,"fixture":"500Hz carrier with24-hop amplitude pulses versus silence and separate steady-amplitude control; fixed supplied numeric scales; candidate presence alone is not full O11 recovery or perceptual acceptance"})
    );
    assert!(forecasts[0] > 0 && forecasts[3] > 0);
    assert_eq!((forecasts[1], forecasts[4]), (0, 0));
    assert!(accents[0] > 0 && periodic[0] > 0 && grouped[0] > 0 && target_period[0] > 0);
    assert_eq!(
        (accents[1], periodic[1], grouped[1], target_period[1]),
        (0, 0, 0, 0)
    );
    assert!(accents[3] > 0 && periodic[3] > 0 && grouped[3] > 0 && target_period[3] > 0);
    assert_eq!(
        (accents[4], periodic[4], grouped[4], target_period[4]),
        (0, 0, 0, 0)
    );
    assert_eq!((periodic[5], grouped[5], target_period[5]), (0, 0, 0));
}

#[test]
#[ignore = "release composed recurrence ownership cost; excludes full O04"]
fn recurrence_ownership_cost_probe() {
    use std::{hint::black_box, time::Instant};
    let grid = space();
    let mut scan = vec![0.; grid.n_bins()];
    for bin in [12, 36, 60, 84, 108, 132, 156] {
        scan[bin] = 1.;
    }
    let mut model = Recurrence::new(grid, config(0), settings(128)).unwrap();
    let buffers = model.slots.each_ref().map(|s| s.estimator.storage_layout());
    let mut times = Vec::with_capacity(6000);
    let mut max_owners = 0;
    let mut resets = 0;
    let mut grouped = 0;
    for step in 1..=6600 {
        let end = step * 512;
        let energy = pulse(step);
        let start = Instant::now();
        let out = model
            .advance(end, end, observed(end, black_box(&scan), energy))
            .unwrap();
        black_box(&out);
        black_box(model.snapshot());
        if step > 600 {
            times.push(start.elapsed().as_secs_f64() * 1e6);
        }
        let frame = model.snapshot().unwrap();
        max_owners = max_owners.max(frame.next_owners.iter().flatten().count());
        resets += frame.newborn.iter().flatten().count();
        grouped += frame
            .evidence_groups
            .iter()
            .flatten()
            .filter(|g| g.grouping.is_some_and(|v| v.proposal_count() > 0))
            .count();
    }
    times.sort_by(f64::total_cmp);
    assert_eq!(
        buffers,
        model.slots.each_ref().map(|s| s.estimator.storage_layout())
    );
    println!(
        "recurrence_ownership_cost {}",
        serde_json::json!({"calls":6000,"warmup_hops":600,"median_us":times[3000],"p99_us":times[5940],"max_us":times[5999],"maximum_owners":max_owners,"cache_resets":resets,"grouped_frames":grouped,"owner_bytes":std::mem::size_of::<Recurrence>(),"frame_bytes":std::mem::size_of::<Frame>(),"slot_pool_payload_bytes":7*std::mem::size_of::<Slot>(),"frame_pool_payload_bytes":std::mem::size_of::<Option<Frame>>(),"full_O04":false,"scope":"single bus acoustic frontend plus residual ledger,period/grouping pools and snapshots; seven input peaks do not assert seven active grouping banks; NSGT/beam/context/two-bus/device excluded"})
    );
}

#[test]
fn capacity_replacement_preserves_old_frame_before_rebinding_preallocated_slot() {
    let grid = space();
    let bins = grid.n_bins();
    let peaks: [usize; 7] = std::array::from_fn(|i| (i + 1) * bins / 8);
    let mut power = vec![0.001; bins];
    let mut cfg = config(0);
    cfg.features.threshold = 0.05;
    cfg.features.deviations = [0.1; 2];
    let mut model = Recurrence::new(grid, cfg, settings(16)).unwrap();
    let buffers = model.slots.each_ref().map(|s| s.estimator.storage_layout());
    let (mut capacity_retirements, mut replacements) = (0, 0);
    let mut counts = BTreeMap::<Handle, u64>::new();
    for step in 1..=1200 {
        for (i, &bin) in peaks.iter().enumerate() {
            power[bin] = (1. + 0.1 * (step as f64 * 0.03 + i as f64 * 0.1).sin()) as f32;
        }
        let energy = 0.03 * (1. + 0.1 * (step as f64 * 0.03).sin());
        let end = step * 512;
        let out = model
            .advance(end, end, observed(end, &power, energy))
            .unwrap();
        for update in out.features.iter().flatten() {
            if let Some(a) = update.detector.as_ref().and_then(|d| d.accent) {
                *counts.entry(a.group).or_default() += 1;
            }
        }
        let frame = model.snapshot().unwrap();
        capacity_retirements += out
            .groups
            .retired
            .iter()
            .flatten()
            .filter(|r| r.reason == super::super::super::lifecycle::Retirement::Capacity)
            .count();
        for (i, born) in frame
            .newborn
            .iter()
            .enumerate()
            .filter_map(|(i, b)| b.map(|b| (i, b)))
        {
            if let Some(old) = frame.evidence_groups[i] {
                replacements += 1;
                assert_ne!(old.ledger.group, born.group);
                assert_eq!(
                    old.ledger.cumulative_count,
                    *counts.get(&old.ledger.group).unwrap_or(&0)
                );
                assert!(
                    out.groups
                        .retired
                        .iter()
                        .flatten()
                        .any(|r| r.group.handle == old.ledger.group)
                );
                assert_eq!(born.cumulative_count, 0);
                assert_eq!(
                    model.slots[i].estimator.ledger_summary().cumulative_count,
                    0
                );
            }
        }
        assert_eq!(
            buffers,
            model.slots.each_ref().map(|s| s.estimator.storage_layout())
        );
    }
    println!(
        "RECURRENCE_CAPACITY_TRACE {}",
        serde_json::json!({"hops":1200,"capacity_retirements":capacity_retirements,"same_hop_replacements":replacements})
    );
    assert!(capacity_retirements > 0 && replacements > 0);
}

#[test]
fn mr2_frontend_exchange_preserves_candidates_on_omission_tempo_and_aperiodic_input() {
    use crate::config::{ArrivalModel, TemporalPeriodConfig};
    let mut models: Vec<_> = [ArrivalModel::Hazard, ArrivalModel::Periodic]
        .into_iter()
        .map(|model| {
            Recurrence::configured(
                Frontend::new(space(), config(0)).unwrap(),
                TemporalPeriodConfig {
                    model,
                    coefficients: [0.; 18],
                    means: [0.; 8],
                    deviations: [1.; 8],
                    horizon_sec: 0.1,
                },
            )
            .unwrap()
        })
        .collect();
    let mut scan = vec![0.; space().n_bins()];
    scan[20] = 1.;
    let mut comparisons = 0;
    let mut alternatives = 0;
    let mut differences = 0;
    for step in 1..=360 {
        let energy = if step < 120 {
            if (72..96).contains(&step) {
                0.01
            } else {
                pulse(step)
            }
        } else if step < 240 {
            if (4..6).contains(&(step % 18)) {
                0.16
            } else {
                0.01
            }
        } else if [249, 250, 281, 282, 300, 301, 347, 348].contains(&step) {
            0.16
        } else {
            0.01
        };
        let end = step * 512;
        for model in &mut models {
            model
                .advance(end, end, observed(end, &scan, energy))
                .unwrap();
        }
        let a = models[0].diagnostics().unwrap();
        let b = models[1].diagnostics().unwrap();
        for (a, b) in a.groups.iter().flatten().zip(b.groups.iter().flatten()) {
            assert_eq!(a.ledger.group, b.ledger.group);
            assert_eq!(a.ledger.cumulative_count, b.ledger.cumulative_count);
            assert_eq!(a.peaks, b.peaks);
            assert_eq!(a.period_source, b.period_source);
            assert_eq!(
                serde_json::to_value(a.grouping).unwrap(),
                serde_json::to_value(b.grouping).unwrap()
            );
            alternatives += usize::from(a.peaks.iter().flatten().count() > 1);
            if let (Some(h), Some(p)) = (a.forecast, b.forecast) {
                assert_eq!(
                    (
                        h.group,
                        h.source_start,
                        h.source_end,
                        h.available,
                        h.issued_at,
                        h.horizon_end,
                        h.last_accent,
                        h.elapsed_seconds
                    ),
                    (
                        p.group,
                        p.source_start,
                        p.source_end,
                        p.available,
                        p.issued_at,
                        p.horizon_end,
                        p.last_accent,
                        p.elapsed_seconds
                    )
                );
                comparisons += 1;
                differences += usize::from(h.probability != p.probability);
            }
        }
    }
    assert!(comparisons > 100 && alternatives > 100 && differences > 100);
    for model in &mut models {
        let previous = model.diagnostics().unwrap();
        model.finish(400 * 512).unwrap();
        let finished = model.diagnostics().unwrap();
        assert_eq!(finished.end_sample, previous.end_sample);
        for g in finished.groups.iter().flatten().filter(|g| g.active) {
            if let Some(f) = g.forecast {
                assert!(f.reset_unknown);
                assert_eq!(f.elapsed_seconds[0], 0.);
                assert!(f.source_end <= previous.end_sample);
            }
        }
    }
    println!(
        "I6_MR2_EXCHANGE {}",
        serde_json::json!({"hops":360,"forecast_pairs":comparisons,"groups_with_alternatives":alternatives,"different_forecasts":differences,"input":"same acoustic power and energy; omitted pulses, changed tempo, aperiodic tail","calibrated":false})
    );
}
