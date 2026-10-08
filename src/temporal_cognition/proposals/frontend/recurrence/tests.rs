use super::super::tests::{config, observed, space};
use super::*;
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn actual_pcm_t2_delivers_without_legacy_accent_to_both_consumers() {
    use crate::config::{T2Attribution, T2Combination, T2Unknown, T2Window, TemporalT2Config};
    use crate::core::nsgt_kernel::{NsgtKernelLog2, NsgtLog2Config, PowerMode};
    use crate::core::nsgt_rt::RtNsgtKernelLog2;
    use crate::temporal_cognition::t2;
    let grid = space();
    let mut c = config(0);
    c.features.means = [1e100; 2];
    let frontend = Frontend::new(grid.clone(), c).unwrap();
    let mut baseline = Recurrence::configured(
        Frontend::new(grid.clone(), c).unwrap(),
        crate::config::TemporalPeriodConfig {
            model: crate::config::ArrivalModel::Periodic,
            horizon_sec: 4.0,
        },
        true,
    )
    .unwrap();
    let frontend = frontend
        .with_t2(TemporalT2Config {
            component_weights: [0.0, 0.0, 1.0, 0.0],
            gain: 1.0,
            threshold: 0.0,
            weight_gain: 1.0,
            window: T2Window::HopMean,
            combination: T2Combination::PositiveComponentRise,
            attribution: T2Attribution::FirstHopMassFraction,
            unknown: T2Unknown::Suppress,
        })
        .unwrap();
    let mut model = Recurrence::configured(
        frontend,
        crate::config::TemporalPeriodConfig {
            model: crate::config::ArrivalModel::Periodic,
            horizon_sec: 4.0,
        },
        true,
    )
    .unwrap();
    let mut nsgt = RtNsgtKernelLog2::new(NsgtKernelLog2::new(
        NsgtLog2Config {
            fs: 48000.0,
            overlap: 0.75,
            nfft_override: Some(2048),
            ..Default::default()
        },
        grid,
        None,
        PowerMode::Coherent,
    ));
    let mut producer = t2::Producer::new(48000).unwrap();
    let mut events = 0;
    let mut forecasts = 0;
    for step in 0_u64..600 {
        let start = step * 512;
        let end = start + 512;
        let pcm: Vec<f32> = (start..end)
            .map(|n| {
                let time = n as f64 / 48000.0;
                (0.001
                    * (1.0 + 0.5 * (2.0 * std::f64::consts::PI * 5.0 * time).sin())
                    * (2.0 * std::f64::consts::PI * 440.0 * time).sin()) as f32
            })
            .collect();
        let mono = pcm.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / 512.0;
        let power = nsgt.process_hop(&pcm);
        let observation = (end >= 2048).then_some(Observation {
            power_scan: power,
            mono_energy: mono,
            source_start: end.saturating_sub(2048),
            source_end: end,
            available_end: end,
        });
        model.observe_t2(Some(producer.process(start, &pcm).unwrap()));
        let old = baseline.advance(end, end, observation).unwrap();
        assert!(
            old.features
                .iter()
                .flatten()
                .all(|u| u.detector.is_none_or(|d| d.accent.is_none()))
        );
        let output = model.advance(end, end, observation).unwrap();
        for (i, update) in output.features[..7].iter().enumerate() {
            if let Some(accent) = update.and_then(|u| u.detector.and_then(|d| d.accent)) {
                events += 1;
                assert_eq!(
                    Some(accent.group),
                    output.groups.assignment.group_handles[i]
                );
                assert!(accent.event_end < accent.available_end);
                assert_eq!(accent.source_start, 0);
                // Validate actual consumer state, including weight, prefix and all support clocks.
                assert_eq!(
                    model.frame.as_ref().as_ref().unwrap().evidence_groups[i]
                        .unwrap()
                        .delivered,
                    [Some(accent); 2]
                );
                if model.slots[i].owner != Some(accent.group) {
                    assert_eq!(
                        model.slots[i].estimator.ledger_summary().cumulative_count,
                        0
                    );
                }
            }
        }
        if let Some(snapshot) = model.diagnostics() {
            forecasts += snapshot
                .groups
                .iter()
                .flatten()
                .filter(|g| g.arrival.is_some())
                .count();
        }
    }
    assert!(events > 0, "actual T2 produced no supported direct event");
    assert!(
        forecasts > 0,
        "actual T2 produced no Periodic arrival payload"
    );
    println!("T2_ACTUAL_PCM_DELIVERY events={events} arrival_payloads={forecasts} legacy_events=0");
}

fn settings(capacity: usize) -> Settings {
    Settings {
        forecast: None,
        arrival_payload: false,
        capacity,
        window_samples: 48000 * 32,
        rebuild_admissions: 1024,
        peak_separation: 1. / 24.,
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
fn energy_accents_reach_one_original_owner_and_newborns_start_without_credit() {
    let grid = space();
    let mut scan = vec![0.; grid.n_bins()];
    scan[20] = 1.;
    let mut model = Recurrence::new(grid, config(0), settings(8)).unwrap();
    let buffers = model.slots.each_ref().map(|s| s.estimator.storage_layout());
    let mut events = BTreeSet::new();
    let mut counts = BTreeMap::<Handle, u64>::new();
    let mut weights = BTreeMap::<Handle, f64>::new();
    let (mut births, mut periodic, mut cap_loss, mut inserted) = (0, 0, 0, 0);
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
            cap_loss += usize::from(g.ledger.capacity_evicted_through.is_some());

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
        }
        assert_eq!(
            buffers,
            model.slots.each_ref().map(|s| s.estimator.storage_layout())
        );
    }
    assert!(
        births > 0 && periodic > 0 && cap_loss > 0 && inserted > 0,
        "births={births} periodic={periodic} cap_loss={cap_loss} inserted={inserted}"
    );
    println!(
        "RECURRENCE_OWNER_TRACE {}",
        serde_json::json!({"hops":360,"unique_accents":events.len(),"births":births,"periodic_frames":periodic,"capacity_limited_frames":cap_loss,"inserted_pairs_reported":inserted,"capacity":8})
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
            model: crate::config::ArrivalModel::Periodic,
            horizon_sec: 0.1,
        });
        models.push(Recurrence::new(grid.clone(), cfg, settings).unwrap());
    }
    let mut audio: [_; 6] = std::array::from_fn(|_| vec![0.; 512]);
    let mut accents = [0usize; 6];
    let mut periodic = [0usize; 6];
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
        serde_json::json!({"hops_per_instance":480,"sample_rate":48000,"hop":512,"nfft":2048,"instances":["diagnostic_pulse_bus0","diagnostic_silence_bus1","diagnostic_steady_bus0","baseline_pulse_bus0","baseline_silence_bus1","baseline_steady_bus0"],"thresholds":[0.05,0.05,0.05,1.,1.,1.],"salience_deviations":[0.2,0.2,0.2,1.,1.,1.],"accents":accents,"arrival_forecasts":forecasts,"periodic_frames":periodic,"frames_with_0_256s_candidate":target_period,"fixture":"500Hz carrier with24-hop amplitude pulses versus silence and separate steady-amplitude control; fixed supplied numeric scales; candidate presence alone is not full O11 recovery or perceptual acceptance"})
    );
    assert!(forecasts[0] > 0 && forecasts[3] > 0);
    assert_eq!((forecasts[1], forecasts[4]), (0, 0));
    assert!(accents[0] > 0 && periodic[0] > 0 && target_period[0] > 0);
    assert_eq!((accents[1], periodic[1], target_period[1]), (0, 0, 0));
    assert!(accents[3] > 0 && periodic[3] > 0 && target_period[3] > 0);
    assert_eq!((accents[4], periodic[4], target_period[4]), (0, 0, 0));
    assert_eq!((periodic[5], target_period[5]), (0, 0));
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
