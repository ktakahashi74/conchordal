use super::*;
use crate::core::harmonicity_kernel::HarmonicityParams;
use std::sync::OnceLock;

fn geometry() -> Arc<Geometry> {
    static GEOMETRY: OnceLock<Arc<Geometry>> = OnceLock::new();
    GEOMETRY
        .get_or_init(|| {
            let space = Log2Space::new(55., 1760., 24);
            Geometry::new(
                &space,
                &HarmonicityKernel::new(&space, HarmonicityParams::default()),
            )
        })
        .clone()
}

#[test]
fn support_action_joint_dense_simd_matches_fixed_width_scalar() {
    let g = geometry();
    let mut snapshot = Snapshot::new(g.clone());
    let e: Vec<_> = (0..g.space.n_bins())
        .map(|i| (i % 11) as f32 / 16.)
        .collect();
    snapshot.observe(&e, 1e-12, 512);
    let row: Vec<_> = (0..e.len()).map(|i| (i % 17) as f64 / 16.).collect();
    let template: Vec<_> = (-13_isize..=15)
        .map(|i| (i, (16 - i.abs()) as f32 / 32.))
        .collect();
    for start in 0..5 {
        let expected = snapshot.joint_density_summary(
            row[start..row.len() - start]
                .iter()
                .copied()
                .enumerate()
                .map(|(i, q)| (start + i, q)),
        );
        crate::core::for_each_simd_level!(|name, simd| {
            let actual = snapshot.joint_density_lanes(simd, start, &row[start..row.len() - start]);
            assert_eq!(
                actual.map(f64::to_bits),
                expected.map(f64::to_bits),
                "{name}"
            );
        });
    }
    for (removed, added) in [
        ((2, 0.3), (12, 0.7)),
        ((45, 0.8), (45, 0.2)),
        ((0, 0.), (110, 0.4)),
        ((90, 0.5), (0, 0.)),
        ((20, 0.3), (80, 0.9)),
    ] {
        let initial = snapshot.joint_density_summary(row.iter().copied().enumerate());
        let bounds = |(bin, scale): (usize, f32)| {
            if scale == 0. {
                return (0, 0);
            }
            let start = bin as isize - 13;
            (
                start.clamp(0, row.len() as isize) as usize,
                (start + template.len() as isize).clamp(0, row.len() as isize) as usize,
            )
        };
        let (rlo, rhi) = bounds(removed);
        let (alo, ahi) = bounds(added);
        let mut edges = [rlo, rhi, alo, ahi];
        edges.sort_unstable();
        let mut sums = [[0.; 8]; 3];
        let mut expected_row = row.clone();
        for pair in edges.windows(2) {
            let [lo, hi] = [pair[0], pair[1]];
            let remove = lo >= rlo && hi <= rhi;
            let add = lo >= alo && hi <= ahi;
            if !remove && !add {
                continue;
            }
            for i in lo..hi {
                let r = if remove {
                    f64::from(
                        template[(i as isize - removed.0 as isize + 13) as usize].1 * removed.1,
                    )
                } else {
                    0.
                };
                let a = if add {
                    f64::from(template[(i as isize - added.0 as isize + 13) as usize].1 * added.1)
                } else {
                    0.
                };
                let old = row[i];
                let new = (old - r).max(0.) + a;
                let delta = new - old;
                let e = f64::from(snapshot.density[i]);
                let k = (i - lo) % 8;
                sums[0][k] += delta * snapshot.g_c[i];
                sums[1][k] += snapshot.k[i] * ((new - e).max(0.) - (old - e).max(0.));
                sums[2][k] += delta;
                expected_row[i] = new;
            }
        }
        let expected: [f64; 3] = std::array::from_fn(|i| {
            initial[i] + sums[i][1..].iter().fold(sums[i][0], |sum, &x| sum + x)
        });
        crate::core::for_each_simd_level!(|name, simd| {
            let mut actual_row = row.clone();
            let actual = snapshot.update_joint_dense(
                simd,
                &mut actual_row,
                initial,
                &template,
                -13,
                removed,
                added,
                true,
            );
            assert_eq!(actual_row, expected_row, "{name}");
            assert_eq!(
                actual.map(f64::to_bits),
                expected.map(f64::to_bits),
                "{name}"
            );
        });
        let dense = snapshot.joint_density_summary(expected_row.iter().copied().enumerate());
        for (a, b) in dense.into_iter().zip(expected) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}

#[test]
fn support_action_joint_novelty_matches_dense_overlap_not_independent_addition() {
    let g = geometry();
    let mut snapshot = Snapshot::new(g.clone());
    let mut e = vec![0.; g.space.n_bins()];
    e[45] = 1.;
    snapshot.observe(&e, 1e-12, 512);
    let mut background = vec![0.; e.len()];
    background[45] = 0.6;
    background[70] = 0.4;
    let template = [(0, 0.8), (25, 0.3)];
    let summary = snapshot.density_summary(background.iter().enumerate().map(|(i, &q)| (i, q)));
    let base = snapshot.add_template_features(&background, summary, 45, &template, 0.);
    let added = snapshot.add_template_features(&background, summary, 45, &template, 1.);
    let mut dense = background.clone();
    for &(offset, q) in &template {
        dense[(45 + offset) as usize] += q;
    }
    let expected = snapshot.candidate(&dense.iter().copied().enumerate().collect::<Vec<_>>());
    for (a, b) in added.into_iter().zip(expected) {
        assert!((a - b).abs() < 1e-12);
    }
    let independent = snapshot.template_features(45, &template);
    assert!((added[2] - base[2] - independent[2]).abs() > 1e-6);
    let joint_summary = [summary[0], summary[2], summary[3]];
    let mut joint = background.iter().map(|&x| f64::from(x)).collect::<Vec<_>>();
    let original = joint.clone();
    let merged = snapshot.update_joint_density(
        &mut joint,
        joint_summary,
        &template,
        (0, 0.),
        (45, 1.),
        false,
    );
    assert_eq!(joint, original);
    let features = [
        merged[0],
        4. * merged[1] / (snapshot.sum + merged[2]).powi(2),
    ];
    // The independent reference rounds two merged bin masses to f32.
    for (a, b) in features.into_iter().zip([expected[0], expected[2]]) {
        assert!((a - b).abs() < 4. * f64::from(f32::EPSILON));
    }
    let committed = snapshot.update_joint_density(
        &mut joint,
        joint_summary,
        &template,
        (0, 0.),
        (45, 1.),
        true,
    );
    assert_eq!(committed, merged);
    let removed =
        snapshot.update_joint_density(&mut joint, committed, &template, (45, 1.), (0, 0.), true);
    for (a, b) in removed.into_iter().zip(joint_summary) {
        assert!((a - b).abs() < 1e-12);
    }
    for (a, b) in joint.into_iter().zip(original) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn action_partial_pairs_merge_overlap_before_novelty() {
    let g = geometry();
    let mut snapshot = Snapshot::new(g.clone());
    let mut density = vec![0.; g.space.n_bins()];
    density[45] = 0.7;
    snapshot.observe(&density, 1e-12, 512);
    let template = [(-3, 0.3), (0, 0.8), (3, 0.2)];
    for parts in [
        [(45, 0.4), (48, 0.9)],
        [(0, 0.), (2, 0.5)],
        [(45, 1.), (45, 0.)],
    ] {
        let mut merged = vec![0.; density.len()];
        for (bin, scale) in parts {
            for (offset, q) in template {
                let i = bin as isize + offset;
                if i >= 0 && i < merged.len() as isize {
                    merged[i as usize] += q * scale;
                }
            }
        }
        let q: Vec<_> = merged
            .into_iter()
            .enumerate()
            .filter(|(_, q)| *q > 0.)
            .collect();
        let expected = snapshot.candidate(&q);
        let actual = snapshot.paired_template_features(&template, parts);
        for (a, b) in actual.into_iter().zip(expected) {
            assert!((a - b).abs() < 1e-14);
        }
    }
}

#[test]
fn gradients_converge_to_finite_differences_and_n_reads_partials() {
    let g = geometry();
    let n = g.space.n_bins();
    let mut e = vec![0.; n];
    e[45] = 1.;
    e[80] = 0.7;
    let mut reference = Snapshot::new(g.clone());
    reference.observe(&e, 1e-12, 512);
    let indices = [0, 22, 45, 63, 80, n - 1];
    let mut errors = Vec::new();
    for step in [0.04, 0.02, 0.01] {
        let mut error = 0.;
        for i in indices {
            let mut added = e.clone();
            added[i] += step;
            let actual_step = f64::from(added[i]) - f64::from(e[i]);
            let mut next = Snapshot::new(g.clone());
            next.observe(&added, 1e-12, 1024);
            error += ((next.focus - reference.focus) / actual_step - reference.g_c[i]).abs();
            error += ((next.conflict - reference.conflict) / actual_step - reference.g_d[i]).abs();
        }
        errors.push(error);
    }
    assert!(
        errors[1] < errors[0] * 0.6 && errors[2] < errors[1] * 0.6,
        "{errors:?}"
    );
    let q = [(22, 0.8), (45, 0.6), (80, 1.1)];
    let values = reference.candidate(&q);
    let mut novelty = 0.;
    for &(i, v) in &q {
        let expected_k: f64 = e
            .iter()
            .enumerate()
            .map(|(j, &power)| {
                let dot: f64 = g.columns[i]
                    .iter()
                    .map(|&(a, x)| {
                        x * g.columns[j]
                            .iter()
                            .find(|(b, _)| *b == a)
                            .map_or(0., |(_, y)| *y)
                    })
                    .sum();
                f64::from(power) * dot / (g.column_norm[i] * g.column_norm[j])
            })
            .sum();
        assert!((reference.k[i] - expected_k).abs() < 1e-12);
        novelty += expected_k * (f64::from(v) - f64::from(e[i])).max(0.);
    }
    let total = e
        .iter()
        .chain(q.iter().map(|(_, v)| v))
        .map(|&v| f64::from(v))
        .sum::<f64>();
    assert!((values[2] - 4. * novelty / total.powi(2)).abs() < 1e-12);
    assert_eq!(reference.candidate(&[(45, 0.8)])[2], 0.);
    let mut silent = Snapshot::new(g);
    silent.observe(&vec![0.; n], 1e-6, 0);
    assert_eq!((silent.focus, silent.conflict, silent.mass), (0., 0., 0.));
    assert_eq!(silent.candidate(&q), [0.; 3]);
}

#[test]
fn support_full_gradient_at_silence_below_at_and_above_floor() {
    let g = geometry();
    let n = g.space.n_bins();
    let floor = g.du[45];
    for level in [0., 0.25, 1., 2.] {
        let mut e = vec![0.; n];
        e[45] = level;
        let mut before = Snapshot::new(g.clone());
        before.observe(&e, floor, 0);
        let mut errors = Vec::new();
        for step in [0.04, 0.02, 0.01] {
            let mut error = 0.;
            for i in [0, 45, 80, n - 1] {
                let mut added = e.clone();
                added[i] += step;
                let delta = f64::from(added[i]) - f64::from(e[i]);
                let mut after = Snapshot::new(g.clone());
                after.observe(&added, floor, 1);
                error += ((after.focus - before.focus) / delta - before.g_c[i]).abs();
                error += ((after.conflict - before.conflict) / delta - before.g_d[i]).abs();
            }
            errors.push(error);
        }
        assert!(
            errors[1] < errors[0] * 0.6 && errors[2] < errors[1] * 0.6,
            "level={level} errors={errors:?}"
        );
    }
}

#[test]
fn partial_templates_clip_without_wrapping_and_read_fractional_positions() {
    let g = geometry();
    let n = g.space.n_bins();
    let mut s = Snapshot::new(g.clone());
    let mut e = vec![0.; n];
    e[32] = 1.;
    s.observe(&e, 1e-12, 512);
    let t = [(-3, 0.2), (0, 0.4), (5, 0.6)];
    for i in [0, 20, n - 1] {
        let absolute: Vec<_> = t
            .iter()
            .filter_map(|&(d, v)| {
                let j = i as isize + d;
                (j >= 0 && j < n as isize).then_some((j as usize, v))
            })
            .collect();
        assert_eq!(s.template_features(i, &t), s.candidate(&absolute));
    }
    let at = (g.space.centers_log2[10] + g.space.centers_log2[11]) * 0.5;
    let p = (at - g.space.centers_log2[0]) / g.space.step();
    let lo = p.floor() as usize;
    let f = f64::from(p - lo as f32);
    let value = s.template_value(at, &[(0, 0.5)], Weights([1., 0., 0.]));
    assert!((value - 0.5 * (s.g_c[lo] + f * (s.g_c[lo + 1] - s.g_c[lo]))).abs() < 1e-12);
}

#[test]
fn support_weights_validation_and_zero() {
    let mut w = Weights::default();
    assert!(!w.active());
    for axis in 0..3 {
        assert!(!w.set(axis, f64::NAN));
        assert!(!w.set(axis, f64::INFINITY));
        assert!(w.set(axis, f64::MAX));
        assert_eq!(w.0[axis], f64::MAX);
        assert!(w.set(axis, 0.));
    }
    assert!(!w.set(2, -1.));
    assert!(w.set(0, -1.));
    assert!(w.set(1, -1.));
}

#[test]
fn support_quadratic_forms_match_direct_convolved_templates() {
    let g = geometry();
    let h = HarmonicityKernel::new(&g.space, HarmonicityParams::default());
    let mut e = vec![0.; g.space.n_bins()];
    for (i, value) in e.iter_mut().enumerate() {
        *value = if i % 19 == 0 {
            (i + 1) as f32 / 31.
        } else {
            0.
        };
    }
    let mut frame = Snapshot::new(g.clone());
    frame.observe(&e, 1e-12, 512);
    let gaussian: Vec<f64> = h
        .support_smoothing()
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    let mut harmonic = vec![0.; gaussian.len() + 4 * g.space.bins_per_oct as usize + 2];
    for k in 1..=16 {
        let pos = (k as f32).log2() * g.space.bins_per_oct as f32;
        let lo = pos.floor() as usize;
        let fraction = f64::from(pos - lo as f32);
        let weight = f64::from((k as f32).powf(-0.4));
        for (j, &v) in gaussian.iter().enumerate() {
            harmonic[lo + j] += weight * v * (1. - fraction);
            harmonic[lo + j + 1] += weight * v * fraction;
        }
    }
    let direct = |pattern: &[f64]| {
        let norm = pattern.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut result = vec![0.; frame.p.len() + pattern.len()];
        for (i, &p) in frame.p.iter().enumerate() {
            for (j, &v) in pattern.iter().enumerate() {
                result[i + j] += p * v / norm;
            }
        }
        result.iter().map(|v| v * v).sum::<f64>()
    };
    // Projection rounding is tested on the dimensionless consumer features, not FFT bits.
    assert!((frame.focus - direct(&gaussian)).abs() < 1e-7);
    assert!((frame.conflict - (1. - direct(&harmonic))).abs() < 1e-7);
}

#[test]
fn support_reuses_actual_projection_and_does_not_change_hr() {
    use crate::core::stream::analysis::AnalysisStream;
    let (nsgt, params) = crate::runtime::body_metabolism_test_core();
    let mut baseline = AnalysisStream::new(params.clone(), nsgt).with_nsgt_power(false);
    let mut active = baseline.clone();
    let geometry = Geometry::new(&baseline.last().space, &params.harmonicity_kernel);
    active.enable_support(geometry, false);
    let mut pcm = vec![0.; active.hop_samples()];
    let hop_len = pcm.len();
    for hop in 0..8 {
        for (i, value) in pcm.iter_mut().enumerate() {
            *value = 0.02
                * (std::f32::consts::TAU * 440. * (hop * hop_len + i) as f32 / params.fs).sin();
        }
        let before = baseline.process(&pcm);
        let after = active.process(&pcm);
        assert_eq!(before.harmonicity, after.harmonicity);
        assert_eq!(before.roughness, after.roughness);
        assert_eq!(before.subjective_intensity, after.subjective_intensity);
        let support = after.support.as_ref().unwrap();
        assert_eq!(support.end_sample, ((hop + 1) * pcm.len()) as u64);
        let mut smeared = vec![0.; pcm.len().max(after.space.n_bins())];
        smeared.truncate(after.space.n_bins());
        let mut r = vec![0.; params.harmonicity_kernel.root_buffer_len(&after.space)];
        let mut h = vec![0.; after.space.n_bins()];
        params.harmonicity_kernel.potential_h_into(
            &after.subjective_intensity,
            &after.space,
            &mut smeared,
            &mut r,
            &mut h,
        );
        let sum = r.iter().map(|&v| f64::from(v)).sum::<f64>();
        if sum > 0. {
            for (p, r) in support.p.iter().zip(r) {
                assert_eq!(*p, f64::from(r) / sum);
            }
        }
    }
    active.gap_to(96000);
    assert!(active.last().support.is_none());
    active.process(&pcm);
    assert_eq!(
        active.last().support.as_ref().unwrap().end_sample,
        96000 + pcm.len() as u64
    );
    active.apply_update(crate::core::landscape::LandscapeUpdate {
        support_preference: [Some(0.); 3],
        ..Default::default()
    });
    assert!(active.process(&pcm).support.is_none());
    active.apply_update(crate::core::landscape::LandscapeUpdate {
        support_preference: [Some(1.), None, None],
        ..Default::default()
    });
    assert!(active.process(&pcm).support.is_some());
}

#[cfg(feature = "profile-alloc")]
#[test]
fn support_processing_reuses_scratch_with_one_publication_in_flight() {
    use crate::core::stream::analysis::AnalysisStream;
    let (nsgt, params) = crate::runtime::body_metabolism_test_core();
    let mut stream = AnalysisStream::new(params.clone(), nsgt).with_nsgt_power(false);
    let g = Geometry::new(&stream.last().space, &params.harmonicity_kernel);
    stream.enable_support(g, false);
    let pcm = vec![0.002; stream.hop_samples()];
    stream.process(&pcm);
    let mut previous = stream.last().support.clone();
    crate::runtime_profile::begin_allocations();
    for _ in 0..16 {
        let frame = stream.process(&pcm);
        let next = frame.support.clone();
        std::hint::black_box(&previous);
        previous = next;
    }
    let counts = crate::runtime_profile::finish_allocations().unwrap();
    assert_eq!(counts.count, 0, "{counts:?}");
    assert_eq!(counts.bytes, 0, "{counts:?}");
}

#[test]
fn support_display_survives_zero_updates_and_gaps_without_enabling_loo() {
    use crate::core::stream::analysis::AnalysisStream;
    let (nsgt, params) = crate::runtime::body_metabolism_test_core();
    let mut stream = AnalysisStream::new(params.clone(), nsgt).with_nsgt_power(false);
    let geometry = Geometry::new(&stream.last().space, &params.harmonicity_kernel);
    stream.enable_support(geometry, false);
    stream.publish_support_for_display();
    let pcm = vec![0.002; stream.hop_samples()];
    let zero = crate::core::landscape::LandscapeUpdate {
        support_preference: [Some(0.); 3],
        harmonic_relations_weight: Some(0.),
        harmonic_continuation_weight: Some(0.),
        ..Default::default()
    };
    stream.apply_update(zero);
    assert!(stream.process(&pcm).support.is_some());
    assert!(stream.voice_observer().process(&pcm).support.is_none());
    stream.gap_to(96_000);
    stream.apply_update(zero);
    assert_eq!(
        stream.process(&pcm).support.as_ref().unwrap().end_sample,
        96_000 + pcm.len() as u64
    );
    stream.apply_update(crate::core::landscape::LandscapeUpdate {
        support_preference: [Some(1.), None, None],
        ..Default::default()
    });
    assert!(stream.voice_observer().process(&pcm).support.is_some());
}
