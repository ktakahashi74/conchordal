use super::*;

fn config() -> TemporalGestureConfig {
    let mut coefficients = [[[0.; 11]; 4]; 4];
    for row in &mut coefficients {
        for cell in row {
            cell[0] = (1_f64 / 3.).exp_m1().ln();
        }
    }
    TemporalGestureConfig {
        rms_reference: 0.1,
        means: [0.; 5],
        deviations: [1.; 5],
        coefficients,
    }
}

fn stamp(start: u64) -> Evidence {
    Evidence {
        start,
        end: start + 1,
        source_start: start.saturating_sub(100),
        source_end: start + 1,
        available: start + 1,
    }
}

fn run(attack: u64, release: Option<u64>, gap: Option<u64>, end: u64) -> Run {
    Run {
        attack: Some(stamp(attack)),
        release: release.map(stamp),
        gap: gap.map(stamp),
        observed_end: end,
        censored: false,
    }
}

fn group(generation: u64, run: Run, alpha: f64, known: u64) -> Group {
    Group {
        handle: ridge::Handle {
            bus: 0,
            epoch: 0,
            generation,
        },
        born: 0,
        end: 8000,
        paths: vec![Path {
            state: State::Release,
            entered: 1000,
            run,
            mass: 1.,
        }],
        scratch: Vec::with_capacity(64),
        unknown: 0.,
        support: VecDeque::from([Support {
            start: 8000 - known,
            end: 8000,
            alpha,
        }]),
    }
}

#[test]
fn competing_rates_match_survival_and_finer_subdivision() {
    let dt = 0.01;
    let row = transition(State::Attack, [0., 1. / 3., 1. / 3., 1. / 3.], dt, true);
    assert!((row[0] - (-dt).exp()).abs() < 1e-14);
    assert!((row.iter().sum::<f64>() - 1.).abs() < 1e-14);
    for &p in &row[1..] {
        assert!((p - (1. - (-dt).exp()) / 3.).abs() < 1e-14);
    }
    let mut fine = [1., 0., 0., 0.];
    for _ in 0..100 {
        let mut next = [0.; 4];
        for i in 0..4 {
            let r = transition(STATES[i], [1. / 3.; 4], dt / 100., true);
            for j in 0..4 {
                next[j] += fine[i] * r[j];
            }
        }
        fine = next;
    }
    assert!(row.iter().zip(fine).all(|(a, b)| (a - b).abs() < 2e-5));
    assert_eq!(transition(State::Attack, [1.; 4], dt, false)[3], 0.);
    assert!(transition(State::Gap, [1.; 4], dt, false)[3] > 0.);
}

#[test]
fn handoff_uses_observed_gap_after_the_release_tail() {
    for duration in [100, 1000, 5000] {
        for separation in [250, 750, 1250] {
            let gap = 1000 + duration;
            let early = run(0, Some(1000), Some(gap), gap);
            let late = run(gap + separation, None, None, gap + separation + 100);
            assert_eq!(handoff(early, late, 1000), Some(separation < 1000));
        }
    }
    assert_eq!(
        handoff(
            run(0, Some(1000), None, 2000),
            run(1500, None, None, 2000),
            1000
        ),
        Some(true)
    );
    assert_eq!(
        handoff(run(0, None, None, 2000), run(1500, None, None, 2000), 1000),
        Some(false)
    );
    assert_eq!(
        handoff(
            run(0, Some(1000), None, 1200),
            run(1500, None, None, 2000),
            1000
        ),
        None
    );
    let mut missing = run(0, Some(1000), Some(2000), 2000);
    missing.censored = true;
    assert_eq!(handoff(missing, run(2250, None, None, 2500), 1000), None);
    assert_eq!(
        handoff(
            run(0, Some(1000), None, 2000),
            run(0, None, None, 2000),
            1000
        ),
        Some(false)
    );
}

#[test]
fn normalized_families_keep_masks_residual_and_coalesced_mass() {
    let mut g = Gesture::new(0, 0, 1000, 100, config()).unwrap();
    g.groups[0] = Some(group(2, run(0, Some(6000), None, 8000), 1. / 3., 8000));
    g.groups[1] = Some(group(3, run(7000, None, None, 8000), 1. / 3., 8000));
    g.groups[2] = Some(group(1, run(100, None, None, 8000), 1. / 3., 8000));
    g.refresh(8000);
    assert_eq!(g.snapshot.family_count, 4);
    assert!((g.snapshot.unresolved).abs() < 1e-14);
    let union = g
        .snapshot
        .candidates
        .iter()
        .flatten()
        .find(|c| c.members[1].is_some())
        .unwrap();
    assert!((union.mass - 0.25).abs() < 1e-14);
    assert!((g.snapshot.groups[0].unwrap().outgoing_union_mass - 0.25).abs() < 1e-14);
    // Equal alpha integrals but only half the second member's lifetime is supported.
    g.groups[1] = Some(group(3, run(7000, None, None, 8000), 2. / 3., 4000));
    g.refresh(8000);
    assert!((g.snapshot.unresolved - 0.5).abs() < 1e-14);
    assert!(
        !g.snapshot
            .candidates
            .iter()
            .flatten()
            .any(|c| c.members[1].is_some())
    );
    // A nonqualifying pair contributes to the two existing singles before pruning.
    g.groups[0] = Some(group(2, run(0, None, None, 8000), 1. / 3., 8000));
    g.groups[1] = Some(group(3, run(7000, None, None, 8000), 1. / 3., 8000));
    g.refresh(8000);
    assert_eq!(g.snapshot.candidates.iter().flatten().count(), 3);
    assert!((g.snapshot.unresolved).abs() < 1e-14);
    assert!(
        g.snapshot
            .candidates
            .iter()
            .flatten()
            .any(|c| (c.mass - 0.375).abs() < 1e-14)
    );
    assert!(std::mem::size_of::<Candidate>() <= 96);
}

#[test]
fn missing_input_censors_endpoints_without_creating_gap() {
    let mut g = group(2, run(0, Some(7000), None, 8000), 1., 8000);
    g.advance(None, 9000, None, config(), 1000).unwrap();
    assert!(
        g.paths
            .iter()
            .all(|p| p.run.censored && p.state != State::Gap && p.run.gap.is_none())
    );
    assert!((g.unknown + g.paths.iter().map(|p| p.mass).sum::<f64>() - 1.).abs() < 1e-12);
    let mut empty = Gesture::new(0, 0, 1000, 100, config()).unwrap();
    empty.finish(8000).unwrap();
    assert_eq!(empty.snapshot.unresolved, 1.);
    assert!(empty.snapshot.candidates.iter().all(Option::is_none));
}

#[test]
fn acoustic_frontend_drives_paths_and_generation_local_support() {
    use crate::{
        config::{TemporalAcousticConfig, TemporalRidgeConfig},
        core::log2space::Log2Space,
    };
    let space = Log2Space::new(100., 6400., 32);
    let mut scan = vec![0.; space.n_bins()];
    scan[20] = 1.;
    let mut f = frontend::Frontend::configured(
        space,
        0,
        (0, 0),
        48000,
        512,
        TemporalRidgeConfig {
            means: [0.; 3],
            deviations: [0.05, 4., 1.],
        },
        TemporalAcousticConfig {
            group_means: [0.; 3],
            group_deviations: [0.05, 4., 1.],
            accent_means: [0.; 2],
            accent_deviations: [1.; 2],
            group_retirement_sec: 2.,
            inactive_energy_max: 1e-8,
            correlation_window_sec: 0.25,
            min_pairs: 8,
            min_coverage: 0.9,
            persistence_hops: 3,
        },
    )
    .unwrap();
    let mut g = Gesture::new(0, 0, 48000, 512, config()).unwrap();
    for step in 1..=100 {
        let end = step * 512;
        let out = f
            .advance(
                end,
                Some(frontend::Observation {
                    power_scan: &scan,
                    mono_energy: 0.01,
                    source_start: end - 512,
                    source_end: end,
                    available_end: end,
                }),
            )
            .unwrap();
        g.advance(&f.snapshot(&out), &out.ridges, end).unwrap();
    }
    g.finish(51200).unwrap();
    assert!(
        g.snapshot
            .groups
            .iter()
            .flatten()
            .any(|s| s.group.generation > 1)
    );
    for s in g.snapshot.groups.iter().flatten() {
        assert!((s.states.iter().sum::<f64>() + s.unknown - 1.).abs() < 1e-12);
        assert!(s.coverage <= 1. && s.support_fraction <= 1.);
        assert!(s.window_start >= s.lifetime_start);
        assert_eq!(s.states[3], 0., "non-low-energy audio introduced gap");
    }
    assert!(g.snapshot.candidates.iter().flatten().count() > 0);
}

#[test]
fn maximum_pair_projection_preserves_pruned_mass_and_clips_lifetimes() {
    let mut g = Gesture::new(0, 0, 1000, 100, config()).unwrap();
    for i in 0..8 {
        let mut owned = group(i as u64 + 1, Run::default(), 1. / 8., 8000);
        owned.paths = (0..15)
            .map(|j| Path {
                state: State::Release,
                entered: 6000,
                run: run(100 + (i * 100 + j) as u64, Some(6000), None, 8000),
                mass: 1. / 15.,
            })
            .collect();
        g.groups[i] = Some(owned);
    }
    g.refresh(8000);
    assert_eq!(g.snapshot.family_count, 29);
    assert_eq!(g.snapshot.candidates.iter().flatten().count(), 16);
    assert!(g.snapshot.discarded_mass > 0.);
    let total = g
        .snapshot
        .candidates
        .iter()
        .flatten()
        .map(|c| c.mass)
        .sum::<f64>()
        + g.snapshot.unresolved;
    assert!((total - 1.).abs() < 1e-12);
    assert!(
        g.scratch.capacity() <= 10880,
        "projection grew beyond its reserved scratch"
    );
    g.finish(16001).unwrap();
    assert_eq!(g.snapshot.unresolved, 1.);
    assert!(g.snapshot.groups.iter().flatten().all(|x| x.coverage == 0.));
    // A fresh generation has its own clipped lifetime, not the retired slot's eight seconds.
    g.groups = std::array::from_fn(|_| None);
    let mut fresh = group(99, run(15901, None, None, 16001), 1., 100);
    fresh.born = 15901;
    fresh.end = 16001;
    fresh.support = VecDeque::from([Support {
        start: 15901,
        end: 16001,
        alpha: 1.,
    }]);
    g.groups[0] = Some(fresh);
    g.refresh(16001);
    let summary = g.snapshot.groups[0].unwrap();
    assert_eq!(summary.window_start, 15901);
    assert_eq!(summary.coverage, 1.);
    assert_eq!(summary.support_fraction, 1.);
}

#[test]
fn observed_envelope_changes_drive_configured_rates_with_original_support() {
    let mut release_mass = Vec::new();
    for rise in [0., 1.] {
        let mut cfg = config();
        cfg.coefficients[0][2][1] = 10.;
        let mut g = group(2, run(0, None, None, 8000), 1., 8000);
        g.paths[0].state = State::Attack;
        let raw = RawDescriptor {
            group: g.handle,
            start: 8000,
            end: 8100,
            known_samples: 100,
            values: [
                None,
                None,
                Some(-2.),
                Some(rise),
                Some(0.),
                Some(0.),
                None,
                None,
                None,
                None,
            ],
            source_start: 7800,
            source_end: 8100,
            available_end: 8100,
        };
        g.advance(Some(&raw), 8100, None, cfg, 1000).unwrap();
        let p = g.paths.iter().find(|p| p.state == State::Release).unwrap();
        release_mass.push(p.mass);
        assert_eq!(p.run.release.unwrap().source_start, 7800);
        assert_eq!(p.run.release.unwrap().available, 8100);
    }
    assert!(release_mass[1] > release_mass[0] * 3.);
}

#[test]
fn vanishing_admitted_scores_transfer_all_mass_to_unknown() {
    let mut cfg = config();
    for row in &mut cfg.coefficients {
        for cell in row {
            cell[0] = -1000.;
        }
    }
    cfg.coefficients[0][3][0] = 10000.;
    let mut g = group(2, run(0, None, None, 8000), 1., 8000);
    g.paths[0].state = State::Attack;
    g.advance(None, 9000, None, cfg, 1000).unwrap();
    assert!(g.paths.is_empty());
    assert_eq!(g.unknown, 1.);
}

#[test]
fn ordinary_updates_use_registered_known_and_unknown_proposals_without_double_leak() {
    for state in STATES {
        for (observed, low) in [(false, false), (true, false), (true, true)] {
            for duration in [1_u64, 100, 32000] {
                let mut g = group(2, run(1000, None, None, 8000), 1., 100);
                g.paths[0].state = state;
                g.paths[0].mass = 0.65;
                g.unknown = 0.35;
                let dt = duration as f64 / 1000.;
                let mut raw = RawDescriptor {
                    group: g.handle,
                    start: 8000,
                    end: 8000 + duration,
                    known_samples: duration,
                    values: [None; 10],
                    source_start: 8000,
                    source_end: 8000 + duration,
                    available_end: 8000 + duration,
                };
                raw.values[2] = Some(if low { -20. } else { -1. });
                g.advance(observed.then_some(&raw), raw.end, None, config(), 1000)
                    .unwrap();
                let stay = (-dt).exp();
                let exit = -(-dt).exp_m1() / 3.;
                let mut expected: [f64; 4] = std::array::from_fn(|i| {
                    if i == state as usize {
                        stay
                    } else if i == State::Gap as usize && !low {
                        0.
                    } else {
                        exit
                    }
                });
                let sum: f64 = expected.iter().sum();
                let keep = (-dt / 10.).exp();
                for (i, value) in expected.iter_mut().enumerate() {
                    *value = 0.65 * keep * *value / sum;
                    if observed && (i != 3 || low) {
                        *value += 0.35 * keep / if low { 4. } else { 3. };
                    }
                }
                let mut actual = [0.; 4];
                for p in &g.paths {
                    actual[p.state as usize] += p.mass;
                }
                for (actual, expected) in actual.into_iter().zip(expected) {
                    assert!((actual - expected).abs() < 2e-14);
                }
                let unknown = (if observed { 1. } else { 0.65 }) * -(-dt / 10.).exp_m1()
                    + if observed { 0. } else { 0.35 };
                assert!((g.unknown - unknown).abs() < 2e-14);
                assert!((actual.iter().sum::<f64>() + g.unknown - 1.).abs() < 2e-14);
            }
        }
    }
}

#[test]
fn articulation_proposals_keep_tiny_leaks_and_large_duration_logs() {
    let tiny = articulation_proposals(Some(State::Attack), [0.; 4], 1e-17, false, false).unwrap();
    assert!((tiny.unknown / 1e-18 - 1.).abs() < 1e-14);
    let long = articulation_proposals(Some(State::Attack), [0.; 4], 10000., false, false).unwrap();
    assert_eq!(long.log_keep, -1000.);
    assert_eq!(long.unknown, 1.);
    for dt in [-1., f64::NAN, f64::INFINITY] {
        assert!(articulation_proposals(None, [0.; 4], dt, true, false).is_err());
    }
}
