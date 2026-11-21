use super::synth::{noise_draw, random_unit};
use super::*;
use crate::core::float::unit_gaussian;
use crate::scenario::control::Routing;

fn spec() -> FloraSpec {
    FloraSpec {
        amp: 0.4,
        fade_sec: 0.0,
        residual: true,
        ..FloraSpec::default()
    }
}

fn terrain() -> Landscape {
    Landscape::new(Log2Space::new(440.0, 880.0, 96))
}

fn flora(landscape: &Landscape) -> Flora {
    let mut flora = Flora::new(
        7,
        spec(),
        (440.0, 880.0),
        FloraSeeding::Uniform,
        42,
        landscape,
        2000.0,
        100,
    );
    // Legacy biomass-only fixtures have a known silent external environment.
    flora.external_observed_frame = Some(0);
    flora
}

fn uniform(f: &mut Flora) {
    f.biomass_mass_scan.fill(0.0);
    f.biomass_mass_scan[f.range.clone()].fill(1.0 / f.range.len() as f32);
}

fn normalize(f: &mut Flora) {
    let total: f32 = f.biomass_mass_scan.iter().sum();
    for p in &mut f.biomass_mass_scan {
        *p /= total;
    }
}

fn hill(f: &mut Flora, bin: usize, height: f32) {
    for (offset, weight) in [(-2, 0.1), (-1, 0.5), (0, 1.0), (1, 0.5), (2, 0.1)] {
        let i = (bin as isize + offset) as usize;
        f.biomass_mass_scan[i] = height * weight;
    }
}

fn allocate(f: &mut Flora) {
    f.metric_allocate(false, 0);
}

fn hop(f: &mut Flora, l: &Landscape) {
    let mut p = vec![0.0; f.hop];
    let mut h = vec![0.0; f.hop];
    f.advance(l);
    f.render_add(&mut p, &mut h);
    assert!(p.iter().chain(&h).all(|v| v.is_finite()));
}

fn mass_accounting(f: &Flora) {
    let mut expected = 0.0;
    for cell in &f.cells {
        assert!((cell.tonal_mass + cell.residual_mass - cell.excess).abs() < 1e-7);
        if cell.peak.as_ref().is_none_or(|p| p.claimed.is_none()) {
            assert_eq!(cell.tonal_mass, 0.0);
            assert_eq!(cell.residual_mass, cell.excess);
        }
        expected += cell.tonal_mass;
        if f.spec.residual {
            expected += (cell.residual_mass - DEAD_MASS).max(0.0);
        }
    }
    let actual: f32 = f
        .lanes
        .iter()
        .filter(|lane| lane.mode == Mode::Current)
        .map(|lane| lane.target.powi(2))
        .sum();
    assert!(
        (actual - expected.min(1.0)).abs() < 2e-7,
        "{actual} != {expected}"
    );
    assert!(actual <= 1.0 + 1e-6);
}

#[test]
fn harmonic_relations_flora_zero_keeps_rng_and_observed_hold_prevents_birth() {
    use crate::temporal_cognition::harmonic_relations::{Acoustic, Bin, Value, consumer};
    use std::sync::Arc;
    let (mut l, _) = consumer::test_landscape(1., true);
    l.harmonic.continuation_weight = 1.;
    let baseline = l.harmonic.support.as_ref().unwrap().clone();
    let decision = Arc::make_mut(l.harmonic.relations.as_mut().unwrap());
    let atom = &mut decision.prediction.atoms[0];
    atom.acoustic = Acoustic {
        mass: Value::exact(baseline.mass),
        focus: Value::exact(baseline.focus),
        competition: Value::exact(baseline.conflict),
        ..atom.acoustic
    };
    atom.bins = baseline
        .p
        .iter()
        .enumerate()
        .filter_map(|(index, &p)| {
            (p > 0.).then_some(Bin {
                index,
                mass: p as f32,
            })
        })
        .collect();
    atom.shape_error = baseline.numerical_errors().shape;
    let make = |landscape: &Landscape| {
        let mut f = Flora::new(
            7,
            FloraSpec::default(),
            (100., 1000.),
            FloraSeeding::Uniform,
            42,
            landscape,
            48000.,
            512,
        );
        f.external_observed_frame = Some(0);
        f.temporal.as_mut().unwrap().now = landscape.harmonic.now_sample;
        f.advance(landscape);
        f.biomass_mass_scan.fill(0.);
        hill(&mut f, 50, 0.7);
        hill(&mut f, 90, 0.3);
        f
    };
    let mut held = make(&l);
    let held_rng = held.rng;
    allocate(&mut held);
    assert_eq!(
        held.births, 0,
        "observed no-addition successor must allow retaining an empty lane"
    );
    assert_eq!(
        held.rng, held_rng,
        "hold must precede lane initialization RNG"
    );
    l.harmonic.relations_weight = 0.;
    l.harmonic.continuation_weight = 0.;
    let mut explicit_zero = make(&l);
    let mut legacy = explicit_zero.clone();
    legacy.harmonic.consumer = None;
    allocate(&mut explicit_zero);
    allocate(&mut legacy);
    assert!(legacy.births > 0);
    assert_eq!(explicit_zero.births, legacy.births);
    assert_eq!(explicit_zero.rng, legacy.rng);
    assert_eq!(
        explicit_zero
            .lanes
            .iter()
            .map(|x| (x.position.to_bits(), x.target.to_bits()))
            .collect::<Vec<_>>(),
        legacy
            .lanes
            .iter()
            .map(|x| (x.position.to_bits(), x.target.to_bits()))
            .collect::<Vec<_>>()
    );
}

#[test]
fn harmonic_relations_flora_return_to_zero_clears_continuation_and_decision() {
    use crate::temporal_cognition::harmonic_relations::consumer;
    let (mut l, _) = consumer::test_landscape(0., true);
    l.harmonic.continuation_weight = 0.5;
    let mut f = Flora::new(
        7,
        FloraSpec::default(),
        (100., 1000.),
        FloraSeeding::Uniform,
        42,
        &l,
        48000.,
        512,
    );
    f.advance(&l);
    f.harmonic
        .consumer
        .as_ref()
        .unwrap()
        .set_sample(l.harmonic.now_sample);
    assert!(f.harmonic.consumer.as_ref().unwrap().active());
    l.harmonic.continuation_weight = 0.;
    f.advance(&l);
    assert!(!f.harmonic.consumer.as_ref().unwrap().active());
}

#[test]
fn support_changes_lane_placement_but_not_growth_input() {
    use crate::core::harmonic_support::{Geometry, Snapshot};
    use crate::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
    let mut l = terrain();
    l.harmonic.support_planned = true;
    let g = Geometry::new(
        &l.space,
        &HarmonicityKernel::new(&l.space, HarmonicityParams::default()),
    );
    let mut e = vec![0.; l.space.n_bins()];
    e[20] = 1.;
    let mut s = Snapshot::new(g);
    s.observe(&e, 1e-12, 512);
    l.harmonic.support = Some(std::sync::Arc::new(s));
    for tone in [FloraTone::Sine, FloraTone::Resonant] {
        let run = |l: &Landscape| {
            let mut f = flora(l);
            f.spec.tone = tone;
            f.refresh_support_template();
            assert!(!f.support.template.is_empty());
            f.tonal_len = 1;
            f.advance(l);
            f.biomass_mass_scan.fill(0.);
            hill(&mut f, 20, 1.);
            hill(&mut f, 75, 1.);
            allocate(&mut f);
            let position = f
                .lanes
                .iter()
                .find(|lane| lane.tonal && lane.mode == Mode::Current)
                .map(|lane| lane.position);
            (position, f.growth_level_scan.clone())
        };
        l.harmonic.support_weights.set(0, 1000.);
        let positive = run(&l);
        l.harmonic.support_weights.set(0, -1000.);
        let negative = run(&l);
        assert_ne!(positive.0, negative.0);
        assert_eq!(positive.1, negative.1);
        assert_eq!(positive.1, l.consonance_field_level);
    }
}

fn joint_fixture() -> Flora {
    let context = crate::life::support_action::tests::fixture();
    let mut l = Landscape::new(context.snapshot.geometry.space.clone());
    l.harmonic.support_planned = true;
    let mut specification = spec();
    specification.residual = false;
    specification.pitch_apply_mode = PitchApplyMode::Rearticulate;
    let mut f = Flora::new(
        7,
        specification,
        (55., 1760.),
        FloraSeeding::Uniform,
        42,
        &l,
        2000.,
        100,
    );
    f.support.action = Some((context.snapshot, context.weights));
    f.support.template = std::sync::Arc::from([(0, 1.)]);
    f.amp.value = f.spec.amp;
    f.gain.value = 1.;
    f.temporal.as_mut().unwrap().now = 512;
    f.biomass_mass_scan.fill(0.);
    hill(&mut f, l.space.nearest_index(context.pitch.exp2()), 1.);
    f.decompose();
    f
}

#[test]
fn support_action_crossing_window_matches_actual_allocation_samples() {
    use crate::life::phonation_engine::CandidatePoint;
    let mut f = joint_fixture();
    f.support.action.as_mut().unwrap().1.0[0] = -1.;
    f.support.records = Some(Vec::new());
    let m = f.temporal.as_mut().unwrap();
    m.active = true;
    m.crossings.extend([
        CandidatePoint { tick: 532, gate: 1 },
        CandidatePoint { tick: 592, gate: 2 },
    ]);
    let mut output = vec![0.; f.hop];
    let mut habitat = vec![0.; f.hop];
    f.render_add(&mut output, &mut habitat);
    let records = &f.support.records;
    let records = records.as_ref().unwrap();
    assert!(records.iter().any(|r| r.action == "birth"));
    assert!(
        records
            .iter()
            .all(|r| (r.at, r.end, r.opportunity) == (532, Some(592), "crossing"))
    );
    assert_eq!(f.support.window.unwrap().start, 592);
    assert_eq!(f.support.window.unwrap().end, None);
    assert!(f.support_width().is_none());
    assert!(habitat[..20].iter().all(|&x| x == 0.));
    assert!(habitat[20..80].iter().any(|&x| x != 0.));
}

#[test]
fn support_action_fallback_and_crossing_rebirth_compare_the_selected_tail() {
    use crate::life::phonation_engine::CandidatePoint;
    for crossing in [false, true] {
        let mut f = joint_fixture();
        f.support.template = std::sync::Arc::from([(0, 4.)]);
        f.support.loudness_exp = 0.3;
        let snapshot = &f.support.action.as_ref().unwrap().0;
        let bin = (1..f.space.n_bins() - 1)
            .max_by(|&a, &b| {
                let score = |bin| {
                    let features = snapshot.template_features(bin, &f.support.template);
                    features[2] * features[0].abs()
                };
                score(a).total_cmp(&score(b))
            })
            .unwrap();
        let c = snapshot.template_features(bin, &f.support.template)[0];
        assert_ne!(c, 0.);
        f.support.action.as_mut().unwrap().1.0[0] = -c.signum();
        f.biomass_mass_scan.fill(0.);
        hill(&mut f, bin, 1.);
        f.decompose();
        let cell = f.cells.iter().position(|c| c.peak.is_some()).unwrap();
        let peak = f.cells[cell].peak.as_ref().unwrap();
        let position = peak.position;
        let mut old = Lane::new(false, position);
        old.mode = Mode::Current;
        old.cell = Some(cell);
        old.generation = 1;
        old.r = 0.15;
        old.q = old.r * old.r;
        old.target = peak.mass.sqrt();
        old.age_samples = 200;
        f.lanes[0] = old;
        f.births = 1;
        f.support.records = Some(Vec::new());
        let start = if crossing { 532 } else { 512 };
        let end = if crossing { 592 } else { 612 };
        if crossing {
            let m = f.temporal.as_mut().unwrap();
            m.active = true;
            m.crossings.extend([
                CandidatePoint {
                    tick: start,
                    gate: 1,
                },
                CandidatePoint { tick: end, gate: 2 },
            ]);
        }
        f.metric_allocate(crossing, (start - 512) as usize);
        let records = &f.support.records;
        let records = records.as_ref().unwrap();
        let release = records
            .iter()
            .find(|r| r.action == "release")
            .expect("retirement candidate wins");
        let birth = records
            .iter()
            .find(|r| {
                matches!(r.action, "birth" | "no_spawn")
                    && r.position.to_bits() == position.to_bits()
            })
            .expect("same-position birth is compared, not prohibited");
        assert_eq!((release.at, release.end), (start, Some(end)));
        assert_eq!((birth.at, birth.end), (start, Some(end)));
        assert_eq!(release.reference_end, birth.reference_end);
        assert!(birth.retiring > 0 && f.lanes[0].r > 0.);
        assert_ne!(birth.baseline, [0., 0.]);
        assert_eq!(
            birth.delta,
            [
                birth.candidate[0] - birth.baseline[0],
                birth.candidate[1] - birth.baseline[1]
            ]
        );
        assert_eq!(release.release, Some(release.candidate));
        let witness = format!(
            "HV16_R2_WITNESS {}\n",
            serde_json::to_string(&serde_json::json!({
                "release": release, "birth": birth,
                "retiring_amplitude": f.lanes[0].r,
            }))
            .unwrap()
        );
        std::io::Write::write_all(&mut std::io::stderr().lock(), witness.as_bytes()).unwrap();
        let mut born = Lane::new(false, position);
        born.mode = Mode::Current;
        born.target = f.cells[cell].peak.as_ref().unwrap().mass.sqrt();
        let mut alone = f.clone();
        alone.lanes.iter_mut().for_each(|l| {
            l.mode = Mode::Free;
            l.r = 0.;
            l.q = 0.;
            l.target = 0.;
        });
        alone.support.scratch.cells = 0;
        let standalone = alone
            .support_action_mean(None, Some(&born), birth.cells)
            .unwrap();
        assert!(
            (birth.delta[1] - standalone[1]).abs() > 1e-12,
            "novelty must include the overlapping tail: {birth:?}, standalone={standalone:?}"
        );
        let mut tail = f.lanes[0].clone();
        tail.slew(f.fs);
        assert!(tail.r > 0.);
    }
}

#[test]
fn support_action_joint_cache_matches_fresh_rebuild_after_selected_changes() {
    let mut f = joint_fixture();
    f.support.window = Some(SupportWindow {
        start: 512,
        end: Some(612),
        hop_start: 512,
        kind: "fallback",
    });
    let position = f.space.freq_of_index(70).log2();
    f.support.template = std::sync::Arc::from([(-25, 2.), (0, 1.)]);
    f.support.loudness_exp = 0.3;
    for i in 0..3 {
        let lane = &mut f.lanes[i];
        *lane = Lane::new(false, position);
        lane.mode = Mode::Current;
        lane.r = 0.15;
        lane.q = lane.r * lane.r;
        lane.target = 0.3;
    }
    for turn in 0..32 {
        f.support.window.as_mut().unwrap().end = Some(if turn == 0 {
            560
        } else if turn % 2 == 0 {
            600
        } else {
            612
        });
        f.lanes[turn % 3].target = 0.2 + turn as f32 / 256.;
        let mut candidate = f.lanes[0].clone();
        if turn % 2 == 0 {
            candidate.retirement_decay = (-0.5 * GROWTH_RATE * f.pace * METABOLISM / f.fs).exp();
            candidate.retire();
        }
        for cells in [2, 4, 8, 16] {
            let cached = f
                .support_action_mean(Some(0), Some(&candidate), cells)
                .unwrap();
            let mut shifted = candidate.clone();
            shifted.target_position += 0.1 * f.space.step();
            let quantized = f
                .support_action_mean(Some(0), Some(&shifted), cells)
                .unwrap();
            assert_eq!(cached, quantized);
            let mut reference = f.clone();
            reference
                .support
                .scratch
                .reserve(f.hop, f.space.n_bins(), f.lanes.len());
            let fresh = reference
                .support_action_mean(Some(0), Some(&candidate), cells)
                .unwrap();
            for (a, b) in cached.into_iter().zip(fresh) {
                assert!((a - b).abs() < 1e-12, "cache drift: {a} versus {b}");
            }
        }
    }
}

#[test]
fn support_action_stationary_joint_rows_keep_midpoint_weights() {
    let mut f = joint_fixture();
    f.support.window = Some(SupportWindow {
        start: 512,
        end: Some(612),
        hop_start: 512,
        kind: "fallback",
    });
    for lane in f.lanes.iter_mut().take(3) {
        *lane = Lane::new(false, 440_f32.log2());
        lane.mode = Mode::Current;
        lane.r = 0.3;
        lane.target = 0.3;
        lane.slew_target = 0.3;
    }
    let candidate = f.lanes[0].clone();
    let once = f.support_action_mean(Some(0), Some(&candidate), 1).unwrap();
    for cells in [2, 25, 50, 100] {
        let repeated = f
            .support_action_mean(Some(0), Some(&candidate), cells)
            .unwrap();
        for (a, b) in once.into_iter().zip(repeated) {
            assert!((a - b).abs() < 1e-12);
        }
    }
}

#[test]
fn support_action_slew_deadline_projection_matches_renderer_sample_steps() {
    for noisy in [false, true] {
        for retiring in [false, true] {
            for scale in [0.25, 1.] {
                let mut original = Lane::new(noisy, 440.0_f32.log2());
                original.mode = Mode::Current;
                original.r = 0.13;
                original.q = original.r * original.r;
                original.target = 0.4;
                original.target_position += 0.1;
                for _ in 0..17 {
                    original.slew(48000.);
                }
                original.retirement_decay = 0.9998;
                if retiring {
                    original.retire();
                }
                let mut actual = original.clone();
                actual.target *= scale;
                for t in 0_usize..512 {
                    actual.slew(48000.);
                    if actual.mode != Mode::Free {
                        actual.age_samples += 1;
                    }
                    let offset = 23 + t;
                    if offset.is_multiple_of(64) || actual.age_samples == 1 {
                        actual.sounding_log2 = actual.position
                            + (actual.target_position - actual.position) * (offset + 1) as f32
                                / 1024.;
                    }
                    let (r, position) =
                        support::project(&original, t, 23, 1024, 48000., scale, None);
                    assert_eq!(r.to_bits(), actual.r.to_bits(), "sample {t}");
                    assert_eq!(
                        position.to_bits(),
                        actual.sounding_log2.to_bits(),
                        "sample {t}"
                    );
                }
            }
        }
    }
}

#[test]
fn support_action_residual_birth_is_unknown_before_and_after_sounding() {
    let mut f = joint_fixture();
    f.support.window = Some(SupportWindow {
        start: 512,
        end: Some(612),
        hop_start: 512,
        kind: "fallback",
    });
    assert!(f.support_width().is_some());
    f.spec.residual = true;
    assert!(f.lanes.iter().all(|lane| lane.r == 0.));
    assert!(f.support_action_mean(None, None, 2).is_none());
    let residual = f.lanes.iter_mut().find(|lane| !lane.tonal).unwrap();
    residual.mode = Mode::Current;
    residual.r = 0.1;
    residual.target = 0.1;
    assert!(f.support_action_mean(None, None, 2).is_none());
}

#[cfg(feature = "profile-alloc")]
#[test]
fn support_action_joint_preview_reuses_allocation_scratch() {
    let mut f = joint_fixture();
    f.support.window = Some(SupportWindow {
        start: 512,
        end: Some(612),
        hop_start: 512,
        kind: "fallback",
    });
    let mut born = Lane::new(false, 440_f32.log2());
    born.mode = Mode::Current;
    born.target = 0.3;
    f.support_action_mean(None, Some(&born), 4).unwrap();
    crate::runtime_profile::begin_allocations();
    for _ in 0..64 {
        let window = f.support.window.as_mut().unwrap();
        window.start += f.hop as u64;
        window.hop_start = window.start;
        window.end = Some(window.start + f.hop as u64);
        std::hint::black_box(f.support_action_mean(None, Some(&born), 4).unwrap());
    }
    let allocations = crate::runtime_profile::finish_allocations().unwrap();
    assert_eq!((allocations.count, allocations.bytes), (0, 0));
}

#[test]
fn support_action_rearticulate_preview_is_the_selected_lane_target() {
    let context = crate::life::support_action::tests::fixture();
    let mut l = Landscape::new(context.snapshot.geometry.space.clone());
    l.harmonic.support_weights = context.weights;
    l.harmonic.support_planned = true;
    let mut f = Flora::new(
        7,
        spec(),
        (55., 1760.),
        FloraSeeding::Uniform,
        42,
        &l,
        2000.,
        100,
    );
    f.spec.pitch_apply_mode = PitchApplyMode::Rearticulate;
    f.support.window = Some(SupportWindow {
        start: 512,
        end: Some(512 + f.hop as u64),
        hop_start: 512,
        kind: "fallback",
    });
    f.support
        .scratch
        .reserve(f.hop, f.space.n_bins(), f.lanes.len());
    f.spec.residual = false;
    f.support.action = Some((
        context.snapshot.clone(),
        crate::core::harmonic_support::Weights([-1., 0., 0.]),
    ));
    f.support.template = std::sync::Arc::from([(0, 1.)]);
    f.amp.value = f.spec.amp;
    f.gain.value = 1.;
    f.harmonic.act_now = true;
    f.tonal_len = 1;
    let bin = l.space.nearest_index(context.pitch.exp2());
    let cell = f.cells.iter().position(|c| c.bins.contains(&bin)).unwrap();
    let peak = Peak {
        bin,
        position: context.pitch + 0.02,
        sigma: 0.,
        window: bin..bin + 1,
        mass: 1.,
        enter: false,
        stay: true,
        claimed: None,
        suppressed: false,
        visited: false,
    };
    let mut lane = Lane::new(false, context.pitch);
    lane.mode = Mode::Current;
    lane.r = 0.5;
    lane.target = 0.5;
    lane.cell = Some(cell);
    let expected = lane.selected_target(&peak, f.spec.pitch_apply_mode, false);
    assert_eq!(expected.1, lane.position);
    assert_ne!(expected.1, peak.position);
    let mut preview = lane.clone();
    (preview.target, preview.target_position) = expected;
    assert!(f.support_action_mean(Some(0), Some(&preview), 2).is_some());
    f.lanes[0] = lane;
    f.cells[cell].peak = Some(peak);
    f.update_tones();
    assert_eq!(f.lanes[0].mode, Mode::Current);
    assert_eq!((f.lanes[0].target, f.lanes[0].target_position), expected);
    let born = Lane::new(false, context.pitch + 0.02);
    assert_eq!(
        born.selected_target(
            f.cells[cell].peak.as_ref().unwrap(),
            f.spec.pitch_apply_mode,
            false
        )
        .1,
        context.pitch + 0.02
    );
}

#[test]
fn support_action_flora_birth_vs_no_spawn_and_retirement_keep_signed_midpoint_rank() {
    let context = crate::life::support_action::tests::fixture();
    let mut l = Landscape::new(context.snapshot.geometry.space.clone());
    l.harmonic.support_planned = true;
    let mut f = flora(&l);
    f.support.window = Some(SupportWindow {
        start: 512,
        end: Some(512 + f.hop as u64),
        hop_start: 512,
        kind: "fallback",
    });
    f.support
        .scratch
        .reserve(f.hop, f.space.n_bins(), f.lanes.len());
    f.spec.residual = false;
    f.support.action = Some((context.snapshot.clone(), context.weights));
    f.support.template = std::sync::Arc::from([(0, 1.)]);
    f.amp.value = f.spec.amp;
    f.gain.value = 1.;
    let mut born = Lane::new(false, context.pitch);
    born.mode = Mode::Current;
    born.target = 0.5;
    let mut held = born.clone();
    held.r = 0.5;
    held.q = 0.25;
    let mut released = held.clone();
    released.retirement_decay = (-0.5 * GROWTH_RATE * f.pace * METABOLISM / f.fs).exp();
    released.retire();
    for subdivisions in [-1, 0, 1] {
        let birth = f
            .support_action_mean(
                None,
                Some(&born),
                f.support_cells(None, Some(&born), subdivisions).unwrap(),
            )
            .unwrap();
        let hold = f
            .support_action_mean(
                None,
                Some(&held),
                f.support_cells(None, Some(&held), subdivisions).unwrap(),
            )
            .unwrap();
        let release = f
            .support_action_mean(
                None,
                Some(&released),
                f.support_cells(None, Some(&released), subdivisions)
                    .unwrap(),
            )
            .unwrap();
        assert!(birth[0] < 0., "no-spawn has zero added C/N");
        assert!(
            release[0] < 0. && release[0] > hold[0],
            "retirement retains the actual slew tail"
        );
    }
    f.support.action.as_mut().unwrap().1.0[0] = -1.;
    let birth = f.support_action_mean(None, Some(&born), 2).unwrap();
    assert!(
        f.support
            .action
            .as_ref()
            .unwrap()
            .1
            .value([birth[0], 0., birth[1]])
            > 0.
    );
    f.support.action = None;
    assert!(f.support_action_mean(None, Some(&held), 2).is_none());
}

#[test]
fn dynamics_large_hop_high_rate_preserves_nonnegativity_and_extent() {
    let mut l = Landscape::new(Log2Space::new(110.0, 1760.0, 96));
    l.consonance_field_level.fill(1.0);
    l.harmonicity01.fill(1.0);
    let mut f = Flora::new(
        1,
        spec(),
        (440.0, 880.0),
        FloraSeeding::Uniform,
        1,
        &l,
        2000.0,
        800,
    );
    f.external_observed_frame = Some(0);
    assert_eq!(f.substeps, 64);
    for _ in 0..100 {
        f.advance(&l);
        for (i, &v) in f.biomass_mass_scan.iter().enumerate() {
            assert!(v.is_finite() && v >= 0.0);
            assert!((0.0..=1.0).contains(&f.resource_state_scan[i]));
            if !f.range.contains(&i) {
                assert_eq!(v, 0.0);
                assert_eq!(f.resource_state_scan[i], 0.0);
            }
        }
    }
}

#[test]
fn resource_recovers_without_biomass_and_falls_under_consumption() {
    let l = terrain();
    let mut f = flora(&l);
    f.biomass_mass_scan.fill(0.0);
    f.resource_state_scan[f.range.clone()].fill(0.2);
    for _ in 0..120 {
        f.advance_resource(1.0);
    }
    assert!(
        f.resource_state_scan[f.range.clone()]
            .iter()
            .all(|&u| (u - 1.0).abs() < 1e-5)
    );
    f.biomass_mass_scan[f.range.clone()].fill(100.0 / f.space.bins_per_oct as f32);
    f.advance_resource(1.0);
    assert!(
        f.resource_state_scan[f.range.clone()]
            .iter()
            .all(|&u| u < 0.01)
    );
    for _ in 0..120 {
        f.advance_resource(1.0);
        assert!(
            f.resource_state_scan
                .iter()
                .all(|&u| (0.0..=1.0).contains(&u))
        );
    }
    let before = f.resource_state_scan.clone();
    f.observe_rhythm(
        TemporalObservation::default(),
        0,
        crate::core::meter::MeterShaping {
            basin_hz: Some((0.5, 1.5)),
            ..Default::default()
        },
    );
    assert_eq!(f.resource_state_scan, before);
}

#[test]
fn consumption_kernel_is_truncated_row_normalized_and_extent_local() {
    let l = Landscape::new(Log2Space::new(110.0, 1760.0, 96));
    let mut f = flora(&l);
    for (row, (lo, weights)) in f.consumption_rows.iter().enumerate() {
        let i = f.range.start + row;
        let weights = &f.consumption_weights[weights.clone()];
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        for (j, &weight) in weights.iter().enumerate() {
            assert!(f.range.contains(&(lo + j)));
            let distance = f.erb_scan[lo + j] - f.erb_scan[i];
            assert!(distance.abs() <= 1.5 + 1e-6);
            assert!(weight > 0.0);
            let centre_weight = weights[i - lo];
            assert!((weight / centre_weight - unit_gaussian(distance, 0.5)).abs() < 1e-6);
        }
    }
    f.biomass_mass_scan[f.range.clone()].fill(REFERENCE_DENSITY / f.space.bins_per_oct as f32);
    f.advance_resource(1.0);
    let a = 1.0 / RENEWAL_SEC + 1.0 / CONSUMPTION_SEC;
    let equilibrium = (1.0 / RENEWAL_SEC) / a;
    let expected = equilibrium + (1.0 - equilibrium) * (-a).exp();
    assert!(
        f.resource_state_scan[f.range.clone()]
            .iter()
            .all(|&u| (u - expected).abs() < 1e-6)
    );
}

#[test]
fn flat_terrain_biomass_stays_below_twenty_over_120_seconds() {
    let mut l = terrain();
    l.consonance_field_level.fill(0.5);
    l.harmonicity01.fill(0.0);
    for total in [0.05, 5.0] {
        let mut f = flora(&l);
        f.biomass_mass_scan[f.range.clone()].fill(total / f.range.len() as f32);
        for _ in 0..2400 {
            f.advance(&l);
            let biomass: f32 = f.biomass_mass_scan.iter().sum();
            assert!(
                biomass.is_finite() && biomass < 20.0,
                "initial {total}, biomass {biomass}"
            );
        }
    }
}

#[test]
fn dynamics_reflecting_spread_conserves_mass_except_metabolism_at_both_ends() {
    let l = terrain();
    let mut results = Vec::new();
    for right in [false, true] {
        let mut f = flora(&l);
        f.hop = 1;
        f.fs = 200.0;
        f.biomass_mass_scan.fill(0.0);
        let edge = if right {
            f.range.end - 1
        } else {
            f.range.start
        };
        f.biomass_mass_scan[edge] = 1.0;
        let dt = f.hop as f32 / f.fs;
        let expected = 1.0 - dt * (f.diffusion + GROWTH_RATE * METABOLISM);
        for step in 0..200 {
            // Zero resource suppresses seeds for this single-substep hop.
            f.resource_state_scan.fill(0.0);
            f.advance(&l);
            if step == 0 {
                assert!((f.biomass_mass_scan[edge] - expected).abs() < 2e-6);
            }
            let expected_total = (1.0 - dt * GROWTH_RATE * METABOLISM).powi(step + 1);
            assert!((f.biomass_mass_scan.iter().sum::<f32>() - expected_total).abs() < 2e-6);
        }
        results.push(f.biomass_mass_scan[edge]);
    }
    assert!((results[0] - results[1]).abs() < 2e-6);
}

#[test]
fn substep_count_uses_smallest_rate_and_spread_bound() {
    let l = terrain();
    let mut f = Flora::new(
        7,
        spec(),
        (440.0, 880.0),
        FloraSeeding::Uniform,
        42,
        &l,
        48000.0,
        512,
    );
    // Default rate: ceil(8 * (512 / 48000) / 0.05) = 2; spread needs only 1.
    assert_eq!(f.substeps, 2);
    for (rate, expected) in [(4.0, 1), (8.0, 2)] {
        // The scene basin is fixed before placement, not a live clock patch.
        f = Flora::new(
            7,
            spec(),
            (440.0, 880.0),
            FloraSeeding::Uniform,
            42,
            &l,
            48000.0,
            512,
        );
        f.observe_rhythm(
            TemporalObservation::default(),
            0,
            crate::core::meter::MeterShaping {
                basin_hz: Some((rate / 4.0, rate / 4.0)),
                ..Default::default()
            },
        );
        f.advance(&l);
        assert_eq!(f.substeps, expected);
        let hop_dt = f.hop as f32 / f.fs;
        let valid = |n| {
            rate * hop_dt / n as f32 <= 0.05 && f.diffusion * f.pace * hop_dt / n as f32 <= 0.25
        };
        assert!(valid(f.substeps));
        assert!(f.substeps == 1 || !valid(f.substeps - 1));
    }
    f.on_space_change(&Log2Space::new(440.0, 880.0, 1200));
    assert_eq!(f.substeps, 3);
}

#[test]
fn biomass_growth_uses_harmonic_seed_rain_and_not_amplitude() {
    let mut l = terrain();
    let mut a = flora(&l);
    a.biomass_mass_scan.fill(0.0);
    let mut b = a.clone();
    a.spec.amp = 0.0;
    b.spec.amp = 1.0;
    l.harmonicity01[33] = 1.0;
    a.advance(&l);
    b.advance(&l);
    assert_eq!(a.biomass_mass_scan, b.biomass_mass_scan);
    assert_eq!(a.resource_state_scan, b.resource_state_scan);
    assert!(a.biomass_mass_scan[33] > 100.0 * a.biomass_mass_scan[73]);
}

#[test]
fn target_scaling_bounds_tonal_and_residual_amplitudes_together() {
    let l = terrain();
    let mut f = flora(&l);
    f.biomass_mass_scan.fill(0.001);
    hill(&mut f, 33, 100.0);
    // An increasing cell has residual mass without another tonal candidate.
    for (offset, i) in f.cells[3].bins.clone().enumerate() {
        f.biomass_mass_scan[i] = 10.0 + offset as f32;
    }
    let mut p = vec![0.0; f.hop];
    let mut h = p.clone();
    f.render_add(&mut p, &mut h);
    assert!(f.cells.iter().map(|c| c.excess).sum::<f32>() > 1.0);
    assert!(f.lanes[..f.tonal_len].iter().any(|l| l.target > 0.0));
    assert!(
        f.lanes[f.tonal_len..f.residual_end]
            .iter()
            .any(|l| l.target > 0.0)
    );
    let target_power: f32 = f.lanes.iter().map(|l| l.target.powi(2)).sum();
    assert!(
        (target_power - 1.0).abs() <= 1e-6,
        "target power {target_power}"
    );
    mass_accounting(&f);
}

#[test]
fn seeding_is_reproducible_restricted_and_has_uniform_fallback() {
    let mut l = terrain();
    l.consonance_density_mass[31..35].fill(2.0);
    let make = |l: &Landscape| {
        Flora::new(
            7,
            spec(),
            (440.0, 880.0),
            FloraSeeding::Consonance,
            42,
            l,
            2000.0,
            100,
        )
    };
    let a = make(&l);
    let b = make(&l);
    assert_eq!(a.biomass_mass_scan, b.biomass_mass_scan);
    let total: f32 = a.biomass_mass_scan.iter().sum();
    assert!((total - INITIAL_BIOMASS).abs() < 1e-6, "{total}");
    assert_eq!(total, a.biomass_mass_scan[31..35].iter().sum::<f32>());
    assert!(
        a.resource_state_scan[a.range.clone()]
            .iter()
            .all(|&u| u == 1.0)
    );
    l.consonance_density_mass.fill(0.0);
    let f = make(&l);
    assert!(
        f.biomass_mass_scan[f.range.clone()]
            .iter()
            .all(|&p| p > 0.0)
    );
    assert_ne!(f.biomass_mass_scan[0], f.biomass_mass_scan[1]);
    let total: f32 = f.biomass_mass_scan.iter().sum();
    assert!((total - INITIAL_BIOMASS).abs() < 1e-6, "{total}");
    // Tiny weights next to zero bins must not overflow to Inf or NaN.
    l.consonance_density_mass[31..33].fill(1e-40);
    let tiny = make(&l);
    assert!(tiny.biomass_mass_scan.iter().all(|v| v.is_finite()));
    let total: f32 = tiny.biomass_mass_scan.iter().sum();
    assert!((total - INITIAL_BIOMASS).abs() < 1e-6, "{total}");
}

#[test]
fn habituation_changes_neither_base_seeding_nor_base_driven_growth() {
    use crate::core::consonance_kernel::ConsonanceRepresentationParams;
    use crate::core::habituation::{HabituationField, HabituationParams};

    let mut base = terrain();
    let repr = ConsonanceRepresentationParams::default();
    let n = base.space.n_bins();
    for i in 0..n {
        base.consonance_field_score[i] = 0.5 + i as f32 / n as f32;
        base.consonance_field_level[i] = repr.level(base.consonance_field_score[i]);
        base.consonance_density_mass[i] = 0.1 + i as f32 / n as f32;
        base.harmonicity01[i] = (i % 11) as f32 / 10.0;
    }
    base.apply_habituation(&vec![0.0; n], repr.theta, &repr);
    let mut habituated = base.clone();
    let mut hab = HabituationField::new(&HabituationParams::default(), repr.theta, n);
    assert!(hab.is_enabled());
    let projection: Vec<_> = (0..n)
        .map(|i| if i < n / 2 { 10.0 } else { 0.01 })
        .collect();
    for _ in 0..200 {
        hab.advance_from_parts(&base.consonance_field_level, &projection, 0.05);
    }
    habituated.apply_habituation(hab.state(), repr.theta, &repr);
    assert_ne!(
        habituated.consonance_field_level,
        habituated.consonance_field_level_eff
    );
    assert_ne!(
        habituated.consonance_density_mass,
        habituated.consonance_density_mass_eff
    );
    assert_eq!(habituated.harmonicity01, base.harmonicity01);
    let make = |l: &Landscape| {
        Flora::new(
            7,
            spec(),
            (440.0, 880.0),
            FloraSeeding::Consonance,
            42,
            l,
            2000.0,
            100,
        )
    };
    let mut expected = make(&base);
    let mut actual = make(&habituated);
    assert_eq!(actual.biomass_mass_scan, expected.biomass_mass_scan);
    let mut reduced = make(&base);
    let mut reduced_base = habituated.clone();
    reduced_base
        .consonance_field_level
        .copy_from_slice(&habituated.consonance_field_level_eff);
    for _ in 0..100 {
        expected.advance(&base);
        actual.advance(&habituated);
        reduced.advance(&reduced_base);
    }
    assert_eq!(actual.biomass_mass_scan, expected.biomass_mass_scan);
    assert_eq!(actual.resource_state_scan, expected.resource_state_scan);
    assert_ne!(actual.biomass_mass_scan, reduced.biomass_mass_scan);
}

#[test]
fn advance_matches_the_closed_form_update_for_uniform_biomass() {
    // Uniform biomass removes the spread and makes consumption the local density.
    let mut l = terrain();
    l.consonance_field_level.fill(0.7);
    l.harmonicity01.fill(0.4);
    // At a 50 ms hop, rate 8 requires 6, 8 or 12 substeps respectively.
    for (pace, substeps) in [(0.75, 6), (1.0, 8), (1.5, 12)] {
        let mut f = flora(&l);
        f.pace = pace;
        let bins_per_oct = f.space.bins_per_oct as f32;
        let (v0, u0) = (2e-3f32, 0.8f32);
        f.biomass_mass_scan.fill(0.0);
        f.biomass_mass_scan[f.range.clone()].fill(v0);
        f.resource_state_scan[f.range.clone()].fill(u0);
        let dt = f.hop as f32 / f.fs * pace / substeps as f32;
        let (mut v, mut u) = (v0, u0);
        for _ in 0..substeps {
            let allee = ALLEE_DENSITY / bins_per_oct;
            v += dt
                * (GROWTH_RATE * v * (0.7 * u * v / (v + allee) - METABOLISM)
                    + u * (BACKGROUND_SEED_DENSITY + SEED_RAIN_DENSITY * 0.4) / bins_per_oct);
            let a = 1.0 / RENEWAL_SEC + v * bins_per_oct / REFERENCE_DENSITY / CONSUMPTION_SEC;
            let fixed = (1.0 / RENEWAL_SEC) / a;
            u = fixed + (u - fixed) * (-a * dt).exp();
        }
        f.advance(&l);
        assert_eq!(f.substeps, substeps, "pace {pace}");
        for i in f.range.clone() {
            assert!(
                (f.biomass_mass_scan[i] - v).abs() <= 1e-6 * v,
                "pace {pace}: biomass {} != {v}",
                f.biomass_mass_scan[i]
            );
            assert!(
                (f.resource_state_scan[i] - u).abs() <= 1e-5,
                "pace {pace}: resource {} != {u}",
                f.resource_state_scan[i]
            );
        }
        assert!(v != v0 && u != u0);
    }
}

#[test]
fn partition_tiles_merges_remnants_and_assigns_boundary_upward() {
    let l = Landscape::new(Log2Space::new(110.0, 1760.0, 96));
    let lo = hz_to_erb(220.0);
    for width in [0.3, 1.3, 1.6, 4.3, 4.7] {
        let f = Flora::new(
            1,
            spec(),
            (erb_to_hz(lo), erb_to_hz(lo + width)),
            FloraSeeding::Uniform,
            1,
            &l,
            4000.0,
            100,
        );
        assert!(!f.cells.is_empty());
        assert_eq!(f.cells[0].bins.start, f.range.start);
        assert_eq!(f.cells.last().unwrap().bins.end, f.range.end);
        for (i, cell) in f.cells.iter().enumerate() {
            assert!(!cell.bins.is_empty());
            for bin in cell.bins.clone() {
                assert!(f.erb_scan[bin] >= cell.lo_erb - 1e-5);
                assert!(f.erb_scan[bin] < cell.hi_erb || i + 1 == f.cells.len());
            }
            if i > 0 {
                assert_eq!(f.cells[i - 1].hi_erb, cell.lo_erb);
            }
        }
        if width < 0.5 {
            assert_eq!(f.cells.len(), 1);
        }
        if width == 4.3 {
            assert_eq!(f.cells.len(), 4);
        }
        if width == 4.7 {
            assert_eq!(f.cells.len(), 5);
        }
    }
    // Choose an exactly representable ERB boundary through the real partition path.
    let space = Log2Space::new(220.0, 880.0, 96);
    let (bin, lower, boundary) = (30..80)
        .find_map(|bin| {
            let boundary = hz_to_erb(space.centers_hz[bin]);
            let approximate = erb_to_hz(boundary - 1.0).to_bits();
            (approximate - 256..=approximate + 256)
                .map(f32::from_bits)
                .find(|&f| f >= space.fmin && hz_to_erb(f) + 1.0 == boundary)
                .map(|lower| (bin, lower, boundary))
        })
        .unwrap();
    let f = Flora::new(
        1,
        spec(),
        (lower, 880.0),
        FloraSeeding::Uniform,
        1,
        &Landscape::new(space),
        2000.0,
        100,
    );
    assert_eq!(f.cells[0].hi_erb, boundary);
    assert_eq!(f.erb_scan[bin], boundary);
    assert!(!f.cells[0].bins.contains(&bin));
    assert!(f.cells[1].bins.contains(&bin));
    let full = flora(&terrain());
    assert!(
        full.cells
            .last()
            .unwrap()
            .bins
            .contains(&(full.range.end - 1))
    );
}

#[test]
fn partition_empty_first_merges_up_and_empty_later_merges_down() {
    let l = Landscape::new(Log2Space::new(110.0, 3520.0, 1));
    let f = Flora::new(
        1,
        spec(),
        (111.0, 3519.0),
        FloraSeeding::Uniform,
        1,
        &l,
        8000.0,
        100,
    );
    assert!(f.cells[0].lo_erb < f.erb_scan[f.range.start]);
    assert_eq!(f.cells[0].lo_erb, hz_to_erb(111.0));
    assert!(f.cells.iter().any(|c| c.hi_erb - c.lo_erb > 1.5));
    let covered: Vec<_> = f.cells.iter().flat_map(|c| c.bins.clone()).collect();
    assert_eq!(covered, f.range.clone().collect::<Vec<_>>());
}

#[test]
fn decomposition_slope_is_not_a_peak_and_plateau_uses_left_centre_cell() {
    let l = terrain();
    let mut f = flora(&l);
    for i in f.range.clone() {
        f.biomass_mass_scan[i] = i as f32 + 1.0;
    }
    f.decompose();
    assert!(
        f.cells[..f.cells.len() - 1]
            .iter()
            .all(|c| c.peak.is_none())
    );
    let boundary = f.cells[1].bins.start;
    f.biomass_mass_scan.fill(0.001);
    f.biomass_mass_scan[boundary - 2..boundary + 2].fill(1.0);
    f.decompose();
    assert_eq!(f.cells[0].peak.as_ref().unwrap().bin, boundary - 1);
    assert!(f.cells[1].peak.is_none());
    assert_eq!(f.cells[0].peak.as_ref().unwrap().sigma, 1.0);
}

#[test]
fn decomposition_equal_maxima_choose_lower_and_window_stops_at_minimum() {
    let l = terrain();
    let mut f = flora(&l);
    f.biomass_mass_scan.fill(0.001);
    let c = f.cells[1].bins.clone();
    let a = c.start + 3;
    let b = a + 4;
    f.biomass_mass_scan[a - 1..b + 2].copy_from_slice(&[0.7, 1.0, 0.8, 0.7, 0.8, 1.0, 0.7]);
    f.decompose();
    let peak = f.cells[1].peak.as_ref().unwrap();
    assert_eq!(peak.bin, a);
    assert!(peak.window.end <= a + 3);
    assert!(!peak.window.contains(&b));
    f.biomass_mass_scan[a] = 1.2;
    f.decompose();
    let peak = f.cells[1].peak.as_ref().unwrap();
    assert!((0.5..=4.0).contains(&peak.sigma));
    assert!((peak.position - f.space.centers_log2[a]).abs() <= 0.5 * f.space.step());
}

#[test]
fn decomposition_flat_floor_zeros_and_end_peaks_are_finite() {
    let l = terrain();
    let mut f = flora(&l);
    uniform(&mut f);
    let positions: Vec<_> = f.lanes[f.tonal_len..f.residual_end]
        .iter()
        .map(|l| l.target_position)
        .collect();
    allocate(&mut f);
    assert!(f.cells.iter().all(|c| c.excess == 0.0 && c.peak.is_none()));
    assert!(f.lanes.iter().all(|l| l.target == 0.0));
    assert_eq!(
        positions,
        f.lanes[f.tonal_len..f.residual_end]
            .iter()
            .map(|l| l.target_position)
            .collect::<Vec<_>>()
    );
    mass_accounting(&f);
    let mut p = vec![0.0; f.hop];
    let mut h = p.clone();
    f.render_add(&mut p, &mut h);
    assert!(p.iter().all(|&x| x == 0.0));
    for bin in [f.range.start, f.range.end - 1] {
        f.biomass_mass_scan.fill(0.0);
        f.biomass_mass_scan[bin] = 1.0;
        allocate(&mut f);
        assert!(
            f.cells
                .iter()
                .any(|c| c.peak.as_ref().is_some_and(|p| p.bin == bin))
        );
        mass_accounting(&f);
        f.render_add(&mut p, &mut h);
        assert!(p.iter().all(|x| x.is_finite()));
    }
}

#[test]
fn mass_accounting_no_candidate_failed_candidacy_waiting_and_residual_off() {
    let l = terrain();
    let mut f = flora(&l);
    for i in f.range.clone() {
        f.biomass_mass_scan[i] = i as f32 + 1.0;
    }
    normalize(&mut f);
    allocate(&mut f);
    mass_accounting(&f);
    f = flora(&l);
    f.biomass_mass_scan.fill(0.00001);
    hill(&mut f, 33, 1.0);
    hill(&mut f, 73, 0.005);
    normalize(&mut f);
    allocate(&mut f);
    assert_eq!(f.cells[3].tonal_mass, 0.0);
    mass_accounting(&f);
    f = flora(&l);
    f.biomass_mass_scan.fill(0.00001);
    hill(&mut f, 33, 1.0);
    normalize(&mut f);
    for lane in &mut f.lanes[..f.tonal_len] {
        lane.mode = Mode::Retiring;
        lane.r = 1.0;
        lane.q = 1.0;
    }
    allocate(&mut f);
    assert!(f.cells.iter().all(|c| c.tonal_mass == 0.0));
    mass_accounting(&f);
    f.spec.residual = false;
    allocate(&mut f);
    mass_accounting(&f);
    assert!(
        f.lanes[f.tonal_len..f.residual_end]
            .iter()
            .all(|l| l.target == 0.0)
    );
}

#[test]
fn tracking_drift_boundary_crossings_phase_and_hysteresis() {
    let l = terrain();
    let mut f = flora(&l);
    let boundary = f.cells[2].bins.start;
    let mut held = None;
    let mut visited_cells = Vec::new();
    for shift in [-0.6, -0.2, 0.2, 0.6, 0.2, -0.2, -0.6] {
        for i in f.range.clone() {
            f.biomass_mass_scan[i] =
                0.000001 + unit_gaussian(i as f32 - (boundary as f32 + shift), 1.2);
        }
        normalize(&mut f);
        allocate(&mut f);
        let lane = f.lanes[..f.tonal_len]
            .iter()
            .position(|l| l.mode == Mode::Current)
            .unwrap();
        if let Some(old) = held {
            assert_eq!(lane, old);
        } else {
            held = Some(lane);
        }
        visited_cells.push(f.lanes[lane].cell.unwrap());
        let phase = f.lanes[lane].z;
        f.metric_allocate(false, 0);
        assert_eq!(phase, f.lanes[lane].z, "tracking reset the phase");
        assert!(
            !f.lanes[..f.tonal_len]
                .iter()
                .any(|l| l.mode == Mode::Retiring)
        );
        hop(&mut f, &l);
    }
    assert!(visited_cells.windows(2).any(|w| w[0] != w[1]));
    f = flora(&l);
    f.biomass_mass_scan.fill(0.000001);
    hill(&mut f, 33, 1.0);
    hill(&mut f, 73, 0.03);
    normalize(&mut f);
    allocate(&mut f);
    let lane = f.cells[3].peak.as_ref().unwrap().claimed.unwrap();
    for height in [0.012, 0.03, 0.012, 0.03] {
        f.biomass_mass_scan.fill(0.000001);
        hill(&mut f, 33, 1.0);
        hill(&mut f, 73, height);
        normalize(&mut f);
        allocate(&mut f);
        assert_eq!(f.cells[3].peak.as_ref().unwrap().claimed, Some(lane));
    }
}

#[test]
fn tracking_jump_retires_old_lane_and_births_another() {
    let l = terrain();
    let mut f = flora(&l);
    f.biomass_mass_scan.fill(0.000001);
    hill(&mut f, 33, 1.0);
    normalize(&mut f);
    hop(&mut f, &l);
    let old = f.lanes[..f.tonal_len]
        .iter()
        .position(|l| l.mode == Mode::Current)
        .unwrap();
    f.biomass_mass_scan.fill(0.000001);
    hill(&mut f, 43, 1.0);
    normalize(&mut f);
    allocate(&mut f);
    assert_eq!(f.lanes[old].mode, Mode::Retiring);
    assert_eq!(f.lanes[old].target, 0.0);
    assert!(
        f.lanes[..f.tonal_len]
            .iter()
            .enumerate()
            .any(|(i, l)| i != old && l.mode == Mode::Current)
    );
}

#[test]
fn separation_suppresses_smaller_and_preserves_mass() {
    let l = terrain();
    let mut f = flora(&l);
    let boundary = f.cells[2].bins.start;
    f.biomass_mass_scan.fill(0.000001);
    f.biomass_mass_scan[boundary - 2] = 0.6;
    f.biomass_mass_scan[boundary + 1] = 1.0;
    normalize(&mut f);
    f.decompose();
    // Start with both held to exercise separation, rather than birth rejection.
    for (i, c) in [1, 2].into_iter().enumerate() {
        let peak = f.cells[c].peak.as_ref().unwrap();
        let lane = &mut f.lanes[i];
        lane.mode = Mode::Current;
        lane.position = peak.position;
        lane.target_position = peak.position;
        lane.target = peak.mass.sqrt();
    }
    f.metric_allocate(false, 0);
    assert_eq!(f.lanes[0].mode, Mode::Retiring);
    assert_eq!(f.cells[1].tonal_mass, 0.0);
    assert!(f.cells[1].peak.as_ref().unwrap().suppressed);
    mass_accounting(&f);
}

#[test]
fn separation_three_cell_chain_is_deterministic_for_input_order() {
    let l = terrain();
    let mut template = flora(&l);
    template.biomass_mass_scan.fill(0.001);
    hill(&mut template, 33, 1.0);
    normalize(&mut template);
    template.decompose();
    let peak = template.cells[1].peak.clone().unwrap();
    let base = hz_to_erb(600.0);
    for (i, mass) in [0.4, 0.6, 0.5].into_iter().enumerate() {
        let mut p = peak.clone();
        p.bin = i * 20;
        p.position = erb_to_hz(base + 0.3 * i as f32).log2();
        p.mass = mass;
        template.cells[i].peak = Some(p);
    }
    template.cells.truncate(3);
    let mut sets = Vec::new();
    for order in [[0, 1, 2], [2, 0, 1], [1, 2, 0]] {
        let mut f = template.clone();
        f.cells = order.iter().map(|&i| template.cells[i].clone()).collect();
        for i in 0..3 {
            f.lanes[i].mode = Mode::Current;
            f.lanes[i].position = template.cells[i].peak.as_ref().unwrap().position;
            f.lanes[i].target = template.cells[i].peak.as_ref().unwrap().mass.sqrt();
        }
        f.update_tones();
        let set: Vec<_> = f.lanes[..f.tonal_len]
            .iter()
            .filter(|l| l.mode == Mode::Current)
            .map(|l| f.cells[l.cell.unwrap()].peak.as_ref().unwrap().bin)
            .collect();
        assert_eq!(set.len(), 1);
        sets.push(set);
    }
    assert!(sets.windows(2).all(|w| w[0] == w[1]));
}

#[test]
fn births_wait_until_next_boundary_after_long_hop_and_follow_mass_order() {
    let l = terrain();
    let mut f = Flora::new(
        7,
        spec(),
        (440.0, 880.0),
        FloraSeeding::Uniform,
        42,
        &l,
        2000.0,
        400,
    );
    f.biomass_mass_scan.fill(0.000001);
    hill(&mut f, 33, 1.0);
    hill(&mut f, 73, 0.5);
    normalize(&mut f);
    for lane in &mut f.lanes[..f.tonal_len] {
        lane.mode = Mode::Retiring;
        lane.r = 1.0;
        lane.q = 1.0;
        lane.position = 100.0_f32.log2();
        lane.target_position = lane.position;
    }
    f.spec.residual = false;
    let mut p = vec![0.0; f.hop];
    let mut h = p.clone();
    f.render_add(&mut p, &mut h);
    assert!(f.cells.iter().all(|c| c.tonal_mass == 0.0));
    assert!(
        f.lanes[..f.tonal_len]
            .iter()
            .all(|l| l.mode == Mode::Retiring)
    );
    let tail_seconds = (1.0 + 1.0 / DEAD_MASS.sqrt()).ln() / (0.5 * GROWTH_RATE * METABOLISM);
    let max_hops = (tail_seconds * f.fs / f.hop as f32).ceil() as usize + 1;
    for _ in 1..max_hops {
        if f.lanes[..f.tonal_len].iter().all(|l| l.mode == Mode::Free) {
            break;
        }
        f.render_add(&mut p, &mut h);
        assert_eq!(f.births, 0);
    }
    assert!(f.lanes[..f.tonal_len].iter().all(|l| l.mode == Mode::Free));
    f.render_add(&mut p, &mut h);
    assert_eq!(f.cells[1].peak.as_ref().unwrap().claimed, Some(0));
    assert_eq!(f.cells[3].peak.as_ref().unwrap().claimed, Some(1));
}

#[test]
fn slew_bounds_and_exact_deadlines_hold_even_at_amp_zero() {
    for noisy in [false, true] {
        let mut lane = Lane::new(noisy, 440.0_f32.log2());
        lane.mode = Mode::Current;
        lane.target = 1.0;
        let step = 1.0 / 160.0;
        for _ in 0..160 {
            let old = if noisy { lane.q } else { lane.r };
            lane.slew(2000.0);
            let new = if noisy { lane.q } else { lane.r };
            assert!((new - old).abs() <= step + 1e-7);
        }
        assert_eq!(lane.r, 1.0);
        lane.retire();
        for _ in 0..160 {
            lane.slew(2000.0);
        }
        assert_eq!(lane.r, 0.0);
        assert_eq!(lane.mode, Mode::Free);
    }
    let l = terrain();
    let mut f = flora(&l);
    f.hop = 400;
    f.habitat_pcm.resize(f.hop, 0.0);
    f.noise.resize(2 * f.noisy_len * f.hop, 0.0);
    f.biomass_mass_scan.fill(0.001);
    hill(&mut f, 33, 1.0);
    normalize(&mut f);
    f.apply_patch(&FloraPatch {
        amp: Some(0.0),
        ..Default::default()
    });
    let mut p = vec![0.0; f.hop];
    let mut h = p.clone();
    f.render_add(&mut p, &mut h);
    assert_eq!(f.amp.value, 0.0);
    assert!(p[160..].iter().all(|&y| y == 0.0));
    assert!(f.lanes.iter().all(|l| l.r == l.target));
}

#[test]
fn sounding_lanes_spread_target_changes_and_live_amp_follows_fade() {
    let fs = 2000.0;
    let mut lane = Lane::new(false, 440.0_f32.log2());
    lane.mode = Mode::Current;
    lane.target = 0.5;
    // A birth keeps the full-scale onset: half of full scale in half of SLEW_SEC.
    for _ in 0..80 {
        lane.slew(fs);
        lane.age_samples += 1;
    }
    assert_eq!(lane.r, 0.5);
    lane.age_samples = (SLEW_SEC * fs) as usize;
    lane.target = 0.25;
    let change = (CHANGE_SEC * fs).ceil() as usize;
    for _ in 1..change {
        lane.slew(fs);
        lane.age_samples += 1;
    }
    assert!(lane.r > 0.25 && lane.r < 0.2503, "{}", lane.r);
    lane.slew(fs);
    assert_eq!(lane.r, 0.25);

    let l = terrain();
    let mut f = flora(&l);
    f.spec.fade_sec = 1.0;
    f.apply_patch(&FloraPatch {
        amp: Some(0.25),
        ..Default::default()
    });
    assert_eq!(f.amp.left, f.fs.ceil() as usize);
}

#[test]
fn atoms_sine_is_unit_and_envelope_step_matches_logged_noise() {
    let mut sine = Lane::new(false, 0.0);
    sine.w = [0.8, 0.6];
    for _ in 0..100 {
        sine.atom([0.0; 2]);
        assert_eq!(sine.e, [1.0, 0.0]);
    }
    let mut lane = Lane::new(true, 0.0);
    lane.a = 0.7;
    lane.b = (1.0 - lane.a * lane.a).sqrt();
    lane.e = [0.4, -0.6];
    let n = noise_draw(&mut 123);
    let expected = [
        lane.a * lane.e[0] + lane.b * n[0],
        lane.a * lane.e[1] + lane.b * n[1],
    ];
    lane.atom(n);
    assert_eq!(lane.e, expected);
}

#[test]
fn noise_order_and_resonant_bandwidth_are_canonical() {
    let l = terrain();
    let mut s = spec();
    s.tone = FloraTone::Resonant;
    let mut f = Flora::new(
        7,
        s,
        (440.0, 880.0),
        FloraSeeding::Uniform,
        42,
        &l,
        2000.0,
        100,
    );
    uniform(&mut f);
    let mut rng = f.rng;
    let expected: Vec<_> = (0..f.hop * f.noisy_len)
        .flat_map(|_| noise_draw(&mut rng))
        .collect();
    let mut p = vec![0.0; f.hop];
    let mut h = p.clone();
    f.render_add(&mut p, &mut h);
    assert_eq!(f.noise, expected);
    assert_eq!(f.rng, rng);
    for width in [0.1, 0.5, 1.0, 4.0, 12.0] {
        f = Flora::new(
            7,
            s,
            (440.0, 880.0),
            FloraSeeding::Uniform,
            42,
            &l,
            2000.0,
            100,
        );
        for bin in f.range.clone() {
            f.biomass_mass_scan[bin] = unit_gaussian(bin as f32 - 33.0, width);
        }
        normalize(&mut f);
        f.render_add(&mut p, &mut h);
        let i = f.cells[1].peak.as_ref().unwrap().claimed.unwrap();
        let lane = &f.lanes[i];
        assert!((lane.sigma - width.clamp(0.5, 4.0)).abs() < 0.002);
        let hz = lane.target_position.exp2();
        let expected = (hz * ((lane.sigma / f.space.bins_per_oct as f32).exp2() - 1.0))
            .clamp(0.5, 0.5 * erb_bw_hz(hz));
        let actual = -lane.a.ln() * f.fs / PI;
        assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
        let residual = &f.lanes[f.tonal_len];
        assert!(
            (-residual.a.ln() * f.fs / PI - erb_bw_hz(residual.target_position.exp2())).abs()
                < 1e-4
        );
    }
    for (lo, hi, bins_per_oct, lower_clamp) in [(10.0, 40.0, 96, true), (100.0, 1600.0, 1, false)] {
        let l = Landscape::new(Log2Space::new(lo, hi, bins_per_oct));
        let mut f = Flora::new(7, s, (lo, hi), FloraSeeding::Uniform, 42, &l, 4000.0, 100);
        f.biomass_mass_scan.fill(0.0);
        let peak = f.range.start + f.range.len() / 2;
        f.biomass_mass_scan[peak] = 1.0;
        f.render_add(&mut p, &mut h);
        if !lower_clamp {
            // A singleton cell has no excess above its floor. The default
            // reservation path retires this zero-mass candidate immediately.
            assert!(f.lanes.iter().all(|l| !l.tonal || l.mode != Mode::Current));
            assert!(
                f.lanes
                    .iter()
                    .any(|l| l.forced_reason == Some("empty_window"))
            );
            continue;
        }
        let lane = f
            .lanes
            .iter()
            .find(|l| l.tonal && l.mode == Mode::Current)
            .unwrap();
        let expected = 0.5;
        assert!((-lane.a.ln() * f.fs / PI - expected).abs() < 1e-4);
    }
    // A broad peak on a coarser grid retains positive excess within its ERB cell.
    // Exercise both the hop-boundary fallback and a birth inside the hop.
    for crossing in [false, true] {
        let l = Landscape::new(Log2Space::new(220.0, 880.0, 12));
        let mut f = Flora::new(
            7,
            s,
            (220.0, 880.0),
            FloraSeeding::Uniform,
            42,
            &l,
            4000.0,
            100,
        );
        for bin in f.range.clone() {
            f.biomass_mass_scan[bin] = unit_gaussian(bin as f32 - 12.0, 4.0);
        }
        normalize(&mut f);
        if crossing {
            f.observe_rhythm(
                metric::TemporalObservation {
                    observed_end: Some(0),
                    known: true,
                    state: crate::core::meter::MeterState {
                        beat: crate::core::meter::MeterBand {
                            freq_hz: 2.0,
                            confidence: 0.9,
                            amplitude: 1.0,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                    ..Default::default()
                },
                0,
                Default::default(),
            );
            let m = f.temporal.as_mut().unwrap();
            assert!(m.active);
            m.crossings.clear();
            m.crossings
                .push(crate::life::phonation_engine::CandidatePoint { tick: 37, gate: 0 });
        } else {
            assert!(!f.temporal.as_ref().unwrap().active);
        }
        f.render_add(&mut p, &mut h);
        assert_eq!(f.births, 1);
        let lane = f
            .lanes
            .iter()
            .find(|l| l.tonal && l.mode == Mode::Current)
            .unwrap();
        assert!(lane.noisy && lane.target > 0.0);
        assert_eq!(lane.age_samples, if crossing { 63 } else { 100 });
        let cell = &f.cells[lane.cell.unwrap()];
        assert!(cell.excess > 0.0 && cell.peak.as_ref().unwrap().mass > 0.0);
        let hz = lane.target_position.exp2();
        let unclamped = hz * ((lane.sigma / f.space.bins_per_oct as f32).exp2() - 1.0);
        let expected = 0.5 * erb_bw_hz(hz);
        assert!(unclamped > expected);
        let actual = -lane.a.ln() * f.fs / PI;
        assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
    }
}

#[test]
fn atoms_seeded_ensemble_power_and_autocorrelation() {
    const LANES: usize = 1024;
    for bandwidth in [0.5, erb_bw_hz(1000.0)] {
        let fs = if bandwidth < 1.0 { 512.0 } else { 2048.0 };
        let samples = (10.0 * fs) as usize;
        let a = (-PI * bandwidth / fs).exp();
        let b = (1.0 - a * a).sqrt();
        let lags = [1, (fs / (10.0 * bandwidth)).round() as usize];
        let mut history = vec![[0.0; 2]; samples];
        let mut powers = 0.0_f64;
        let mut numerator = [0.0_f64; 2];
        let mut denominator = [0.0_f64; 2];
        for id in 0..LANES {
            let mut rng = 0xCAFE_1234 ^ (id as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
            let mut lane = Lane::new(true, 0.0);
            lane.a = a;
            lane.b = b;
            lane.e = noise_draw(&mut rng);
            let (sin, cos) = (TAU * random_unit(&mut rng)).sin_cos();
            lane.z = [cos, sin];
            let (sin, cos) = (TAU * 37.0 / fs).sin_cos();
            lane.w = [cos, sin];
            for (t, e) in history.iter_mut().enumerate() {
                if t % 64 == 0 {
                    let norm = lane.z[0].hypot(lane.z[1]);
                    lane.z[0] /= norm;
                    lane.z[1] /= norm;
                }
                let y = lane.atom(noise_draw(&mut rng));
                powers += f64::from(y * y);
                *e = lane.e;
            }
            for (j, &k) in lags.iter().enumerate() {
                for t in 0..samples - k {
                    let x = history[t];
                    let y = history[t + k];
                    numerator[j] += f64::from(x[0] * y[0] + x[1] * y[1]);
                    denominator[j] += f64::from(x[0] * x[0] + x[1] * x[1]);
                }
            }
        }
        let power = powers / (LANES * samples) as f64;
        assert!(
            (power / 0.5 - 1.0).abs() <= 0.05,
            "B={bandwidth}, power={power}"
        );
        for (j, &k) in lags.iter().enumerate() {
            let rho = numerator[j] / denominator[j];
            assert!(
                (rho - f64::from(a).powi(k as i32)).abs() <= 0.06,
                "B={bandwidth}, lag={k}, rho={rho}, expected={}",
                a.powi(k as i32)
            );
        }
    }
}

#[test]
fn simd_and_scalar_share_noise_and_match_bits_for_ten_seconds() {
    let l = terrain();
    for tone in [FloraTone::Sine, FloraTone::Resonant] {
        let mut f = flora(&l);
        f.spec.tone = tone;
        if tone == FloraTone::Resonant {
            f = Flora::new(
                7,
                f.spec,
                f.extent_hz,
                FloraSeeding::Uniform,
                42,
                &l,
                f.fs,
                f.hop,
            );
        }
        let mut scalar = f.clone();
        let mut p = vec![0.0; f.hop];
        let mut h = p.clone();
        let mut reference = p.clone();
        for hop in 0..200 {
            // Drift, jump beyond the tracking radius, disappear, then reuse freed lanes.
            let bin = match hop {
                0..50 => Some(33 + hop / 10),
                50..100 => Some(43 + (hop - 50) / 10),
                100..180 => None,
                _ => Some(33 + (hop - 180) / 10),
            };
            f.biomass_mass_scan.fill(0.00001);
            if let Some(bin) = bin {
                hill(&mut f, bin, 1.0);
                hill(&mut f, 73, 0.7);
            }
            normalize(&mut f);
            scalar
                .biomass_mass_scan
                .copy_from_slice(&f.biomass_mass_scan);
            let position = f.lanes[0].position;
            let births = f.births;
            let retirements = f.retirements;
            p.fill(0.0);
            h.fill(0.0);
            reference.fill(0.0);
            f.render::<true>(&mut p, &mut h);
            h.fill(0.0);
            scalar.render::<false>(&mut reference, &mut h);
            assert_eq!(f.noise, scalar.noise);
            assert_eq!(f.rng, scalar.rng);
            assert_eq!(f.births, scalar.births);
            assert_eq!(f.retirements, scalar.retirements);
            match hop {
                10 => {
                    assert_eq!(f.lanes[0].mode, Mode::Current);
                    assert!(f.lanes[0].position > position);
                    assert_eq!(f.births, births);
                    assert_eq!(f.retirements, retirements);
                }
                50 => {
                    assert_ne!(f.lanes[0].mode, Mode::Current);
                    assert!(f.retirements > retirements);
                    assert!(f.births > births);
                }
                179 => assert!(f.lanes[..f.tonal_len].iter().all(|l| l.mode == Mode::Free)),
                180 => {
                    assert_eq!(f.lanes[0].mode, Mode::Current);
                    assert!(f.births > births);
                }
                _ => {}
            }
            assert!(
                p.iter()
                    .map(|x| x.to_bits())
                    .eq(reference.iter().map(|y| y.to_bits())),
                "hop {hop}"
            );
        }
    }
}

#[test]
fn reconfiguration_interpolates_biomass_and_resource_without_normalizing() {
    let l = Landscape::new(Log2Space::new(256.0, 2048.0, 1));
    let mut f = Flora::new(
        7,
        spec(),
        (256.0, 2048.0),
        FloraSeeding::Uniform,
        42,
        &l,
        8000.0,
        100,
    );
    // Old log2 bins: 8, 9, 10, 11. New in-range bins: 8.5, 9.5, 10.5.
    f.biomass_mass_scan.copy_from_slice(&[1.0, 5.0, 3.0, 7.0]);
    f.resource_state_scan
        .copy_from_slice(&[0.0, 0.5, 1.0, 0.25]);
    f.on_space_change(&Log2Space::new(2.0_f32.powf(7.5), 2.0_f32.powf(11.5), 1));
    assert_eq!(f.range, 1..4);
    let interpolated = [0.0, 3.0, 4.0, 5.0, 0.0];
    let resource = [0.0, 0.25, 0.75, 0.625, 0.0];
    for i in 0..f.space.n_bins() {
        assert!((f.biomass_mass_scan[i] - interpolated[i]).abs() < 1e-6);
        assert!((f.resource_state_scan[i] - resource[i]).abs() < 1e-7);
    }
}

#[test]
fn reconfiguration_preserves_biomass_density_across_bin_resolutions() {
    let l = Landscape::new(Log2Space::new(256.0, 512.0, 96));
    let mut f = Flora::new(
        7,
        spec(),
        (256.0, 512.0),
        FloraSeeding::Uniform,
        42,
        &l,
        2000.0,
        100,
    );
    f.biomass_mass_scan.fill(3.0 / 96.0);
    f.resource_state_scan.fill(0.4);
    let initial: f32 = f.biomass_mass_scan.iter().sum();
    for bins in [192, 72, 120, 96] {
        let mut resampled = f.clone();
        resampled.on_space_change(&Log2Space::new(256.0, 512.0, bins));
        let total: f32 = resampled.biomass_mass_scan.iter().sum();
        // Endpoint inclusion can change the sum by one bin on the coarser grid.
        let endpoint_error = 3.0 / 96.0_f32.min(bins as f32);
        assert!(
            (total - initial).abs() <= endpoint_error + 1e-5,
            "{bins} bins/octave: initial {initial}, resampled {total}"
        );
        for i in resampled.range.clone() {
            assert!((resampled.biomass_mass_scan[i] * bins as f32 - 3.0).abs() < 1e-5);
            assert!((resampled.resource_state_scan[i] - 0.4).abs() < 1e-6);
        }
    }
}

#[test]
fn reconfiguration_carries_orphans_across_two_changes() {
    let l = terrain();
    let mut f = flora(&l);
    f.biomass_mass_scan.fill(0.000001);
    hill(&mut f, 33, 1.0);
    normalize(&mut f);
    hop(&mut f, &l);
    f.lanes[f.tonal_len].r = 0.5;
    f.lanes[f.tonal_len].q = 0.25;
    let old_position = f.lanes[f.tonal_len].position;
    let old_tonal: Vec<_> = f.lanes[..f.tonal_len]
        .iter()
        .filter(|l| l.mode != Mode::Free)
        .map(|l| l.position)
        .collect();
    f.on_space_change(&Log2Space::new(400.0, 1000.0, 120));
    assert!(
        f.lanes[f.residual_end..]
            .iter()
            .any(|l| l.position == old_position && l.mode == Mode::Retiring)
    );
    for pos in old_tonal {
        assert!(f.lanes.iter().any(|l| l.tonal && l.position == pos));
    }
    f.on_space_change(&Log2Space::new(420.0, 960.0, 72));
    assert!(
        f.lanes[f.residual_end..]
            .iter()
            .any(|l| l.position == old_position && l.mode == Mode::Retiring)
    );
    let mut p = vec![0.0; f.hop];
    let mut h = p.clone();
    let tail_seconds = (1.0 + 0.5 / DEAD_MASS.sqrt()).ln() / (0.5 * GROWTH_RATE * METABOLISM);
    let max_hops = (tail_seconds * f.fs / f.hop as f32).ceil() as usize + 1;
    for _ in 0..max_hops {
        f.render_add(&mut p, &mut h);
    }
    assert!(
        f.lanes[f.residual_end..]
            .iter()
            .filter(|l| !l.tonal)
            .all(|l| l.mode == Mode::Free)
    );
}

#[test]
fn reconfiguration_dormancy_requires_an_actual_bin_and_resumes_uniformly() {
    let l = terrain();
    let mut f = Flora::new(
        1,
        spec(),
        (450.0, 460.0),
        FloraSeeding::Uniform,
        42,
        &l,
        2000.0,
        100,
    );
    f.on_space_change(&Log2Space::new(440.0, 880.0, 1));
    assert!(f.range.is_empty());
    assert!(f.cells.is_empty());
    let mut p = vec![0.0; f.hop];
    let mut h = p.clone();
    f.render_add(&mut p, &mut h);
    assert!(p.iter().all(|&x| x == 0.0));
    f.on_space_change(&l.space);
    assert!(!f.range.is_empty());
    assert!(
        f.resource_state_scan[f.range.clone()]
            .iter()
            .all(|&u| u == 1.0)
    );
    assert!((f.biomass_mass_scan.iter().sum::<f32>() - INITIAL_BIOMASS).abs() < 1e-6);
    assert!(
        f.biomass_mass_scan[f.range.clone()]
            .windows(2)
            .all(|w| w[0] == w[1])
    );
    f.on_space_change(&Log2Space::new(1000.0, 2000.0, 96));
    assert!(f.range.is_empty());
}

#[test]
fn release_uses_current_gain_and_scenario_end_keeps_existing_release() {
    let mut linear = Ramp::new(0.0);
    linear.to(1.0, 2.0, 48000.0);
    for t in 1..=96000 {
        let value = linear.tick();
        if t % 24000 == 0 {
            assert_eq!(value, t as f32 / 96000.0);
        }
    }

    let l = terrain();
    let mut s = spec();
    s.fade_sec = 1.0;
    let mut f = Flora::new(
        7,
        s,
        (440.0, 880.0),
        FloraSeeding::Uniform,
        42,
        &l,
        2000.0,
        100,
    );
    hop(&mut f, &l);
    let current = f.gain.value;
    assert!(current > 0.0 && current < 1.0);
    f.release(0.1);
    assert_eq!(f.gain.value, current);
    let left = f.gain.left;
    f.release_at_scenario_end();
    assert_eq!(f.gain.left, left);
    for _ in 0..3 {
        hop(&mut f, &l);
    }
    assert!(f.is_finished());
    let mut zero = flora(&l);
    zero.release(0.0);
    let mut pcm = vec![0.0; zero.hop];
    let mut habitat = pcm.clone();
    zero.render_add(&mut pcm, &mut habitat);
    assert!(pcm.iter().chain(&habitat).all(|&x| x == 0.0));
    assert!(zero.is_finished());
}

#[test]
fn routing_adds_to_only_the_selected_buffers() {
    let l = terrain();
    for presentation in [false, true] {
        let mut f = flora(&l);
        f.spec.routing = Routing {
            to_presentation: presentation,
            to_habitat: !presentation,
        };
        f.biomass_mass_scan.fill(0.001);
        hill(&mut f, 33, 1.0);
        normalize(&mut f);
        let mut p = vec![0.25; f.hop];
        let mut h = p.clone();
        f.render_add(&mut p, &mut h);
        let (sounding, silent) = if presentation { (&p, &h) } else { (&h, &p) };
        assert!(sounding.iter().any(|&x| x != 0.25));
        assert!(silent.iter().all(|&x| x == 0.25));
    }
}

#[test]
fn f2_every_internal_scan_boundary_names_the_bad_scan() {
    let l = terrain();
    for name in [
        "biomass_mass_scan",
        "resource_state_scan",
        "scratch_scan",
        "erb_scan",
    ] {
        for operation in 0..7 {
            let mut f = flora(&l);
            match name {
                "biomass_mass_scan" => {
                    f.biomass_mass_scan.pop();
                }
                "resource_state_scan" => {
                    f.resource_state_scan.pop();
                }
                "scratch_scan" => {
                    f.scratch_scan.pop();
                }
                _ => {
                    f.erb_scan.pop();
                }
            }
            let result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match operation {
                    0 => f.advance(&l),
                    1 => f.decompose(),
                    2 => f.metric_allocate(false, 0),
                    3 => f.render_add(&mut [0.0; 100], &mut [0.0; 100]),
                    4 => {
                        f.report(0.0, &l);
                    }
                    5 => f.advance_resource(0.01),
                    _ => f.on_space_change(&l.space),
                }));
            let error = result.expect_err(name);
            let message = error
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| error.downcast_ref::<&str>().copied())
                .unwrap();
            assert!(message.contains(name), "{message}");
        }
    }
}

#[test]
#[should_panic(expected = "consonance_field_level")]
fn f2_advance_checks_landscape_input() {
    let mut l = terrain();
    let mut f = flora(&l);
    l.consonance_field_level.pop();
    f.advance(&l);
}

#[test]
#[should_panic(expected = "harmonicity01")]
fn f2_advance_checks_harmonicity_input() {
    let mut l = terrain();
    let mut f = flora(&l);
    l.harmonicity01.pop();
    f.advance(&l);
}

#[test]
#[should_panic(expected = "consonance_density_mass")]
fn f2_placement_checks_landscape_input() {
    let mut l = terrain();
    l.consonance_density_mass.pop();
    let _ = flora(&l);
}

#[test]
#[cfg(feature = "profile-alloc")]
fn support_flora_partial_scan_reuses_cache_without_allocations() {
    use crate::core::harmonic_support::{Geometry, Snapshot};
    use crate::core::harmonicity_kernel::{HarmonicityKernel, HarmonicityParams};
    for tone in [FloraTone::Sine, FloraTone::Resonant] {
        let mut l = terrain();
        l.harmonic.support_planned = true;
        l.harmonic.support_weights.0 = [1.; 3];
        let g = Geometry::new(
            &l.space,
            &HarmonicityKernel::new(&l.space, HarmonicityParams::default()),
        );
        let mut s = Snapshot::new(g);
        let mut e = vec![0.; l.space.n_bins()];
        e[30] = 1.;
        s.observe(&e, 1e-12, 512);
        l.harmonic.support = Some(std::sync::Arc::new(s));
        let mut f = flora(&l);
        f.spec.tone = tone;
        f.refresh_support_template();
        let ptr = f.support.template.as_ptr();
        f.advance(&l);
        crate::runtime_profile::begin_allocations();
        for _ in 0..16 {
            f.advance(&l);
        }
        let counts = crate::runtime_profile::finish_allocations().unwrap();
        assert_eq!((counts.count, counts.bytes), (0, 0));
        assert_eq!(ptr, f.support.template.as_ptr());
        assert!(f.support.active);
    }
}

#[test]
#[cfg(feature = "profile-alloc")]
fn advance_and_render_allocate_nothing_during_continual_turnover() {
    use crate::runtime_profile::{begin_allocations, finish_allocations};
    for (tone, respond) in [
        (FloraTone::Sine, 0.0),
        (FloraTone::Resonant, 0.0),
        (FloraTone::Sine, 2.0),
    ] {
        let mut l = terrain();
        l.harmonicity01.fill(0.0);
        l.onset_habituation_state_scan = vec![0.0; l.space.n_bins()].into();
        l.onset_habituation_known = true;
        let mut f = flora(&l);
        f.spec.tone = tone;
        f.respond = (respond > 0.0).then(|| Respond::new(respond));
        f.on_space_change(&l.space);
        let mut p = vec![0.0; f.hop];
        let mut h = p.clone();
        let boundary = f.cells[2].bins.start;
        let mut last_cell = None;
        let mut crossings = 0;
        begin_allocations();
        for t in 0..300 {
            l.consonance_field_level.fill(0.5);
            f.biomass_mass_scan.fill(0.000001);
            if t % 30 < 10 {
                hill(&mut f, 33, 1.0);
            } else if t % 30 < 20 {
                hill(&mut f, 73, 1.0);
            } else {
                let shift = if t % 2 == 0 { -0.6 } else { 0.6 };
                for i in f.range.clone() {
                    f.biomass_mass_scan[i] +=
                        unit_gaussian(i as f32 - (boundary as f32 + shift), 1.2);
                }
            }
            normalize(&mut f);
            let history = std::sync::Arc::get_mut(&mut l.onset_habituation_state_scan).unwrap();
            for (i, x) in history.iter_mut().enumerate() {
                *x = if (t + i) % 7 == 0 {
                    *x + 0.01
                } else {
                    0.9 * *x
                };
            }
            l.onset_habituation_end = t as u64 * f.hop as u64;
            f.observe_onsets(&l, l.onset_habituation_end);
            f.advance(&l);
            p.fill(0.0);
            h.fill(0.0);
            f.render_add(&mut p, &mut h);
            if t % 30 >= 20 {
                let cell = f
                    .lanes
                    .iter()
                    .find(|l| l.tonal && l.mode == Mode::Current)
                    .and_then(|l| l.cell);
                if last_cell.is_some() && cell != last_cell {
                    crossings += 1;
                }
                last_cell = cell;
            } else {
                last_cell = None;
            }
        }
        let counts = finish_allocations().unwrap();
        assert!(f.births > 0 && f.retirements > 0 && crossings > 0);
        assert_eq!(f.respond.is_some(), respond > 0.0);
        assert!(f.respond.as_ref().is_none_or(|r| r.extra_sec > 0.0));
        assert_eq!(counts.count, 0, "{counts:?}");
        assert_eq!(counts.bytes, 0, "{counts:?}");
        println!(
            "F12_FLORA_ALLOCATIONS tone={tone:?} respond={respond} counts={counts:?} births={} retirements={} crossings={crossings}",
            f.births, f.retirements
        );
    }
}

impl Flora {
    #[cfg(test)]
    pub(crate) fn rng_state_for_test(&self) -> u64 {
        self.rng
    }

    #[cfg(test)]
    pub(crate) fn aud_m1_start_window(landscape: &Landscape) -> Vec<Self> {
        let mut states = Vec::new();
        for density in [0.02, 0.144, 1.0] {
            for resource in [0.5, 1.0] {
                let mut flora = Self::new(
                    1,
                    FloraSpec::default(),
                    (landscape.space.fmin, landscape.space.fmax),
                    FloraSeeding::Uniform,
                    1,
                    landscape,
                    48_000.0,
                    512,
                );
                flora
                    .biomass_mass_scan
                    .fill(density / landscape.space.bins_per_oct as f32);
                flora.resource_state_scan.fill(resource);
                states.push(flora);
            }
        }
        states
    }

    #[cfg(test)]
    pub(crate) fn aud_m1_window_signs(states: &[Self]) -> Vec<Vec<i8>> {
        assert_eq!(states.len(), 6);
        states
            .iter()
            .enumerate()
            .map(|(i, flora)| {
                let before = [0.02, 0.144, 1.0][i / 2] / flora.space.bins_per_oct as f32;
                flora
                    .biomass_mass_scan
                    .iter()
                    .map(|&v| {
                        if v > before {
                            1
                        } else if v < before {
                            -1
                        } else {
                            0
                        }
                    })
                    .collect()
            })
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn aud_m1_cell_signs(landscape: &Landscape) -> Vec<Vec<i8>> {
        let mut states = Self::aud_m1_start_window(landscape);
        for flora in &mut states {
            flora.advance(landscape);
        }
        Self::aud_m1_window_signs(&states)
    }
}
