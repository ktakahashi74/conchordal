use super::*;
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-11, "{a} != {b}");
}
fn particle(b: f64) -> Particle {
    Particle {
        v: 1.0,
        s: 3.0,
        m: 2.0,
        w: 2.0,
        log_b: b.ln(),
    }
}

#[test]
fn kalman_equations_and_posterior_to_prior_kl() {
    let mut p = particle(1.0);
    let kl = kalman(&mut p, 6.0);
    // delta=4, prior variance=3, alpha=1/2, m=4, w=3/2.
    close(p.m, 4.0);
    close(p.w, 1.5);
    close(kl, 0.5 * (2_f64.ln() + 5.5 / 3.0 - 1.0));
    let reverse = 0.5 * (0.5_f64.ln() + 7.0 / 1.5 - 1.0);
    assert!((kl - reverse).abs() > 0.1);
}
#[test]
fn beta_precision_transition_mean_and_gamma_shape_correction() {
    let mut rng = Rng::new(99);
    let n = 100_000;
    let mean = (0..n).map(|_| rng.beta_transition() / ETA).sum::<f64>() / n as f64;
    assert!((mean - 1.0).abs() < 0.001, "{mean}");
    for shape in [0.5, 1.0, 49.5] {
        let mean = (0..n).map(|_| rng.gamma(shape)).sum::<f64>() / n as f64;
        assert!((mean / shape - 1.0).abs() < 0.015, "{shape}: {mean}");
    }
    close(
        (0.5 * ETA / (1.0 - ETA)) / (0.5 * ETA / (1.0 - ETA) + 0.5),
        ETA,
    );
}
#[test]
fn log_weights_previous_times_likelihood_and_cutoff() {
    let mut ps = [particle(0.25), particle(0.75)];
    for (p, likelihood) in ps.iter_mut().zip([2_f64, 1.0]) {
        p.log_b += likelihood.ln();
    }
    assert_eq!(normalize(&mut ps), Some(0));
    close(ps[0].log_b.exp(), 0.4);
    close(ps[1].log_b.exp(), 0.6);
    let mut ps = [particle(1.0); 4];
    for (p, w) in ps.iter_mut().zip([10.0, -689.0, -690.0, f64::NEG_INFINITY]) {
        p.log_b = w;
    }
    assert_eq!(normalize(&mut ps), Some(1));
    assert!(ps[1].log_b.is_finite());
    assert_eq!(ps[2].log_b, f64::NEG_INFINITY);
    assert_eq!(ps[3].log_b, f64::NEG_INFINITY);
    assert_eq!(normalize(&mut [particle(0.0)]), None);
}
#[test]
fn realized_information_gain_uniform_nonuniform_and_zero_terms() {
    let ps = [particle(0.25), particle(0.75)];
    let n = information_gain(&ps, &[0.5_f64.ln(); 2]).unwrap();
    let entropy = -0.25 * 0.25_f64.ln() - 0.75 * 0.75_f64.ln();
    close(n, 2_f64.ln() - entropy);
    close(
        information_gain(&ps, &[0.2_f64.ln(), 0.8_f64.ln()]).unwrap(),
        0.25 * (1.25_f64).ln() + 0.75 * (0.9375_f64).ln(),
    );
    close(
        information_gain(&[particle(0.0), particle(1.0)], &[f64::NEG_INFINITY, 0.0]).unwrap(),
        0.0,
    );
    assert_eq!(information_gain(&ps, &[f64::NEG_INFINITY, 0.0]), None);
}
#[test]
fn effective_size_strict_trigger_and_systematic_tuple_resampling() {
    close(ess(&[particle(0.25); 4]), 4.0);
    close(ess(&[particle(0.0), particle(1.0)]), 1.0);
    assert!(!needs_resampling(2.0, 4));
    assert!(needs_resampling(1.99, 4));
    let mut f = Filter::new(12, 0, 4, (V0, S0));
    f.particles = [particle(0.0), particle(0.0), particle(1.0), particle(0.0)].to_vec();
    f.particles[2].v = 0.7;
    f.particles[2].s = 0.9;
    f.particles[2].m = -8.0;
    f.particles[2].w = 0.3;
    let rngs = f.transitions.clone();
    f.resample();
    assert!(f.particles.iter().all(|p| *p
        == Particle {
            log_b: -4_f64.ln(),
            ..Particle {
                v: 0.7,
                s: 0.9,
                m: -8.0,
                w: 0.3,
                log_b: 0.0
            }
        }));
    assert_eq!(f.transitions, rngs);
}
#[test]
fn common_valid_set_renormalizes_and_empty_is_missing() {
    let row = |energy, a, n, sx| CellHop {
        energy,
        metrics: Some(Metrics {
            a,
            n,
            sx,
            v: 1.0,
            s: 1.0,
        }),
        ..CellHop::missing("test")
    };
    let rows = [
        row(1.0, 1.0, 2.0, 3.0),
        row(100.0, 99.0, f64::NAN, 0.0),
        row(3.0, 5.0, 6.0, 7.0),
    ];
    let (m, count) = aggregate(&rows);
    assert_eq!(count, 2);
    assert_eq!(m, Some([4.0, 5.0, 6.0]));
    assert_eq!(aggregate(&[]), (None, 0));
    assert_eq!(aggregate(&[CellHop::missing("first")]), (None, 0));
}
#[test]
fn cell_silence_freezes_rng_and_state_and_resumes_exactly() {
    let space = Log2Space::new(900.0, 1100.0, 96);
    let n = space.n_bins();
    let mut a = Terrain::new(space.clone(), 0);
    let mut b = Terrain::new(space, 0);
    a.observe(0, &vec![0.01; n], &vec![0.5; n], Some(0.1), false);
    b.observe(0, &vec![0.01; n], &vec![0.5; n], Some(0.1), false);
    for i in 1..10 {
        let hop = a.observe(i, &vec![0.0; n], &vec![0.5; n], Some(0.1), false);
        assert!(hop.values.iter().all(Option::is_none));
    }
    let resumed = a.observe(10, &vec![0.02; n], &vec![0.5; n], Some(0.1), false);
    let contiguous = b.observe(10, &vec![0.02; n], &vec![0.5; n], Some(0.1), false);
    assert_eq!(
        serde_json::to_value(resumed).unwrap(),
        serde_json::to_value(contiguous).unwrap()
    );
    for (a, b) in a.filters.iter().zip(&b.filters) {
        assert_eq!(a.particles, b.particles);
        assert_eq!(a.transitions, b.transitions);
    }
    // Refilling, startup, and M=0 precede first-observation initialization.
    for (maximum, warming) in [(Some(0.0), false), (None, false), (Some(1.0), true)] {
        let mut t = Terrain::new(Log2Space::new(900.0, 1100.0, 96), 0);
        assert_eq!(
            t.observe(0, &vec![0.01; n], &vec![0.5; n], maximum, warming)
                .valid_cells,
            0
        );
        assert!(t.filters.iter().all(|f| !f.initialized));
    }
}
#[test]
fn first_observation_and_reinitialization_values() {
    for bad in 0..7 {
        let mut f = Filter::new(4, 0, 4, (0.001, 0.5));
        assert_eq!(f.observe(-2.0).reason.as_deref(), Some("first"));
        match bad {
            0 => f.particles[0].v = 0.0,
            1 => f.particles[0].s = -1.0,
            2 => f.particles[0].w = 0.0,
            3 => f.particles[0].m = f64::NAN,
            4 => f.particles[0].m = f64::INFINITY,
            5 => f.particles[0].v = f64::INFINITY,
            _ => {
                f.particles[0].v = f64::MAX;
                f.particles[0].w = f64::MAX;
            }
        }
        assert_eq!(f.observe(3.0).reason.as_deref(), Some("numerical"));
        assert!(f.particles.iter().all(|p| *p
            == Particle {
                v: 0.001,
                s: 0.5,
                m: 3.0,
                w: 1.0,
                log_b: -4_f64.ln()
            }));
        assert!(f.observe(3.1).metrics.is_some());
    }
}
#[test]
fn deterministic_rng_open_uniform_and_separate_streams() {
    let mut a = Rng::new(11);
    let mut b = Rng::new(11);
    assert_eq!(Rng::new(0).next(), 0xe220a8397b1dcdaf);
    for _ in 0..10000 {
        let u = a.uniform();
        assert!(u > 0.0 && u < 1.0);
        assert_eq!(u, b.uniform());
        assert_eq!(a.gamma(0.5), b.gamma(0.5));
    }
    let mut a = Filter::new(12, 0, 64, (V0, S0));
    let mut b = Filter::new(12, 0, 64, (V0, S0));
    let mut c = Filter::new(12, 1, 64, (V0, S0));
    for i in 0..50 {
        a.observe(i as f64 * 0.01);
        b.observe(i as f64 * 0.01);
        c.observe(i as f64 * 0.01);
    }
    assert_eq!(a.particles, b.particles);
    assert_ne!(a.particles, c.particles);
}
#[test]
fn reference_mismatch_uses_all_cells_and_evaluation_before_resampling() {
    let mut t = Terrain::new(Log2Space::new(200.0, 4000.0, 96), 0);
    let n = t.space.n_bins();
    let mut p = vec![0.0; n];
    p[t.cells[0].bins.clone()].fill(1.0);
    let hop = t.observe(0, &p, &vec![0.5; n], Some(1.0), false);
    close(hop.values[3].unwrap(), (t.cells.len() as f64).ln());
    assert_eq!(hop.values[0], None);
    let mut f = Filter::new(99, 0, 64, (V0, S0));
    f.observe(0.0);
    let mut saw_resampling = false;
    for i in 0..500 {
        let row = f.observe(if i % 5 == 0 { 2.0 } else { 0.0 });
        if row.resampled {
            saw_resampling = true;
            assert!(row.ess.unwrap() < 32.0);
            close(ess(&f.particles), 64.0);
        }
        if let Some(m) = row.metrics {
            assert!(
                m.a >= 0.5 * (std::f64::consts::TAU * std::f64::consts::E * s_floor()).ln() - 1e-12
            );
        }
    }
    assert!(saw_resampling);
}
#[test]
fn every_efe_scan_boundary_rejects_wrong_lengths() {
    for which in 0..3 {
        let error = std::panic::catch_unwind(|| {
            let space = Log2Space::new(900.0, 1100.0, 96);
            let n = space.n_bins();
            match which {
                0 => {
                    Terrain::new(space, 0).observe(
                        0,
                        &vec![1.0; n - 1],
                        &vec![0.5; n],
                        Some(1.0),
                        false,
                    );
                }
                1 => {
                    Terrain::new(space, 0).observe(
                        0,
                        &vec![1.0; n],
                        &vec![0.5; n - 1],
                        Some(1.0),
                        false,
                    );
                }
                _ => {
                    partition_erb_cells(&space, &vec![0.0; n - 1], (900.0, 1100.0));
                }
            }
        })
        .unwrap_err();
        let text = error.downcast_ref::<String>().unwrap();
        assert!(text.contains(["efe_power_scan", "efe_field_level_scan", "erb_scan"][which]));
    }
}

#[test]
fn nonfinite_posterior_mean_reinitializes_and_gap_keeps_block_counts() {
    let mut filter = Filter::new(12, 0, 4, (V0, S0));
    filter.observe(-f64::MAX);
    assert_eq!(
        filter.observe(f64::MAX).reason.as_deref(),
        Some("numerical")
    );
    assert!(
        filter
            .particles
            .iter()
            .all(|p| p.m == f64::MAX && p.w == 1.0 && p.v == V0 && p.s == S0)
    );
    let mut config = crate::config::AppConfig::default();
    config.info_terrain.efe = true;
    config.analysis.nfft = 2048;
    let mut stream = crate::runtime::info_terrain_efe_eval_stream(&config, 48_000);
    let space = stream.last().space.clone();
    let history =
        crate::core::spectral_history::SpectralHistory::new(space.clone(), 48_000.0, 2048);
    let (mut observer, output) =
        crate::core::info_terrain::Observer::new(space, 48_000, 512, "habitat", false, true);
    observer.enable_efe(&config, 2048);
    for id in 0..8 {
        let frame = stream.process(&[0.1; 512]);
        observer.observe_analysis(id, &frame, &history, false, stream.landscape_params());
    }
    assert_eq!(observer.efe.as_ref().unwrap().hops, 8);
    observer.reset_gap();
    assert_eq!(observer.efe.as_ref().unwrap().hops, 8);
    assert!(
        observer
            .efe
            .as_ref()
            .unwrap()
            .terrain
            .filters
            .iter()
            .all(|f| !f.initialized)
    );
    let frame = stream.process(&[0.1; 512]);
    observer.observe_analysis(10, &frame, &history, true, stream.landscape_params());
    assert_eq!(
        observer
            .efe
            .as_ref()
            .unwrap()
            .trace
            .as_ref()
            .unwrap()
            .last()
            .unwrap()
            .cells[0]
            .reason
            .as_deref(),
        Some("refill")
    );
    drop(observer);
    let block = output.rx.recv().unwrap();
    let json = serde_json::to_value(block).unwrap();
    assert_eq!(json["input_gap"], true);
    assert_eq!(json["efe"]["missing_hops"][0], 9);
}

#[test]
fn whole_runtime_space_is_partitioned_and_particle_count_keeps_rng_domains() {
    let t = Terrain::new(Log2Space::new(55.0, 8000.0, 96), 0);
    assert_eq!(t.cells[0].bins.start, 0);
    assert_eq!(t.cells.last().unwrap().bins.end, t.space.n_bins());
    assert!(
        t.cells
            .windows(2)
            .all(|c| c[0].bins.end == c[1].bins.start && c[0].hi_erb == c[1].lo_erb)
    );
    let a = Filter::new(99, 0, 64, (V0, S0));
    let b = Filter::new(99, 0, 128, (V0, S0));
    assert_eq!(a.resampling, b.resampling);
    assert_eq!(a.transitions, b.transitions[..64]);
}
