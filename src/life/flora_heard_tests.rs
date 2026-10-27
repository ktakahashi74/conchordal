use super::*;
use crate::core::stream::{
    analysis::AnalysisStream,
    loo::{FloraObservation, Observer, SourceKind},
};

#[test]
fn heard_resource_unknown_hold_geometry_and_fixed_reference() {
    let mut terrain = Landscape::new(Log2Space::new(440.0, 880.0, 96));
    let mut flora = Flora::new(
        7,
        FloraSpec::default(),
        (440.0, 880.0),
        FloraSeeding::Uniform,
        21,
        &terrain,
        48000.0,
        512,
    );
    let initial = flora.resource_state_scan.clone();
    flora.advance(&terrain);
    assert_eq!(initial, flora.resource_state_scan);
    terrain.flora_observations.push(FloraObservation {
        id: 7,
        generation: 1,
        perception: None,
        level_scan: Vec::new(),
        power_scan: Some(vec![0.00045; terrain.space.n_bins()]),
    });
    flora.observe_external(&terrain, Some(8));
    assert!(flora.external_observed_frame.is_none());
    terrain.flora_observations[0].generation = 0;
    flora.observe_external(&terrain, Some(9));
    let expected = terrain.space.bins_per_oct as f32 / REFERENCE_DENSITY;
    for &ce in &flora.external_consumption_scan {
        assert!((ce - expected).abs() < 16.0 * f32::EPSILON * expected);
    }
    let ce = flora.external_consumption_scan.clone();
    flora.apply_patch(&FloraPatch {
        amp: Some(0.0),
        ..Default::default()
    });
    flora.observe_external(&terrain, Some(10));
    assert_eq!(ce, flora.external_consumption_scan);
    flora.advance(&terrain);
    assert!(
        flora
            .resource_state_scan
            .iter()
            .all(|&u| (0.0..1.0).contains(&u))
    );
    terrain.flora_observations.clear();
    flora.observe_external(&terrain, None);
    assert_eq!(flora.external_observed_frame, Some(10));
    assert!(!flora.external_current);
    let before = flora.resource_state_scan.clone();
    flora.advance(&terrain);
    assert_eq!(ce, flora.external_consumption_scan);
    assert!(
        flora
            .resource_state_scan
            .iter()
            .zip(before)
            .all(|(&a, b)| a < b)
    );
    flora.on_space_change(&Log2Space::new(440.0, 880.0, 48));
    assert!(flora.external_observed_frame.is_none());
    let before = flora.resource_state_scan.clone();
    flora.advance_resource(0.1);
    assert_eq!(before, flora.resource_state_scan);
}

#[test]
fn heard_flora_pcm_oracle_two_sources_routing_mute_fade_and_gap() {
    use crate::core::timebase::Timebase;
    use crate::life::schedule_renderer::{ScheduleRenderer, SourcePcm};
    use crate::scenario::control::Routing;
    let (nsgt, params) = crate::runtime::body_metabolism_test_core();
    let fs = nsgt.fs();
    let hop = nsgt.hop();
    let mut terrain = Landscape::new(nsgt.space().clone());
    terrain.consonance_field_level.fill(0.7);
    terrain.harmonicity01.fill(0.4);
    for (tone, residual) in [
        (FloraTone::Sine, false),
        (FloraTone::Resonant, false),
        (FloraTone::Resonant, true),
    ] {
        for (presentation, habitat) in [(true, true), (false, true), (true, false)] {
            let spec = FloraSpec {
                tone,
                residual,
                fade_sec: 0.04,
                routing: Routing {
                    to_presentation: presentation,
                    to_habitat: habitat,
                },
                ..Default::default()
            };
            let mut flora = [
                Flora::new(
                    7,
                    spec,
                    (330.0, 880.0),
                    FloraSeeding::Uniform,
                    21,
                    &terrain,
                    fs,
                    hop,
                ),
                Flora::new(
                    8,
                    spec,
                    (330.0, 880.0),
                    FloraSeeding::Uniform,
                    22,
                    &terrain,
                    fs,
                    hop,
                ),
            ];
            let mut mixed = AnalysisStream::new(params.clone(), nsgt.clone());
            let mut observers = [
                Observer::flora(7, 0, &mut mixed, true),
                Observer::flora(8, 0, &mut mixed, true),
            ];
            let mut references = [
                Observer::flora(7, 0, &mut mixed, true),
                Observer::flora(8, 0, &mut mixed, true),
            ];
            let mut renderer = ScheduleRenderer::new(Timebase { fs, hop });
            renderer.loo_pcm = [7, 8]
                .into_iter()
                .map(|id| SourcePcm {
                    kind: SourceKind::Flora,
                    analyze_hr: true,
                    tonal_context: None,
                    id,
                    generation: 0,
                    habitat: vec![0.0; hop],
                })
                .collect();
            let zero = vec![0.0; hop];
            let mut external = vec![0.0; hop];
            let mut oracle = vec![0.0; hop];
            let mut max_l1 = 0.0f32;
            for frame in 0..160 {
                if frame == 48 {
                    flora[0].apply_patch(&FloraPatch {
                        amp: Some(0.0),
                        ..Default::default()
                    });
                }
                if frame == 64 {
                    flora[0].apply_patch(&FloraPatch {
                        amp: Some(0.03),
                        ..Default::default()
                    });
                }
                if frame == 112 {
                    flora[0].release(0.08);
                }
                if frame == 80 {
                    for o in observers.iter_mut().chain(&mut references) {
                        o.reset();
                    }
                    mixed.reset();
                }
                for f in &mut flora {
                    f.advance(&terrain);
                }
                let full = renderer
                    .render_with_source_pcm(
                        &[],
                        (frame * hop) as u64,
                        &Default::default(),
                        &mut [],
                        &mut flora,
                        |_, _, _, _| {},
                    )
                    .habitat
                    .to_vec();
                for (i, x) in external.iter_mut().enumerate() {
                    *x = 0.03 * (TAU * 440.0 * (frame * hop + i) as f32 / fs + 0.7).sin();
                }
                let full: Vec<_> = full.iter().zip(&external).map(|(a, b)| a + b).collect();
                for j in 0..2 {
                    assert_eq!(renderer.loo_pcm[j].habitat, flora[j].habitat_pcm);
                    for i in 0..hop {
                        oracle[i] = external[i] + flora[1 - j].habitat_pcm[i];
                        let removed = full[i] - flora[j].habitat_pcm[i];
                        assert!(
                            (removed - oracle[i]).abs()
                                <= 32.0
                                    * f32::EPSILON
                                    * 1.0f32.max(full[i].abs()).max(oracle[i].abs())
                        );
                    }
                    observers[j].process(&full, &renderer.loo_pcm[j].habitat);
                    references[j].process(&oracle, &zero);
                    let got = observers[j].resource_snapshot();
                    let expected = references[j].resource_snapshot();
                    assert_eq!(got.power_scan.is_some(), expected.power_scan.is_some());
                    if let (Some(got), Some(expected)) = (got.power_scan, expected.power_scan) {
                        let mass: f32 = expected.iter().sum();
                        let error: f32 =
                            got.iter().zip(&expected).map(|(a, b)| (a - b).abs()).sum();
                        max_l1 = max_l1.max(error / mass.max(1e-12));
                        assert!(error / mass.max(1e-12) <= 1e-3, "L-1 finite PCM oracle");
                    }
                }
                mixed.process(&full);
            }
            println!(
                "F10_PCM tone={tone:?} residual={residual} presentation={presentation} habitat={habitat} normalized_l1={max_l1}"
            );
        }
    }
}

#[test]
#[should_panic(expected = "external_power_scan")]
fn heard_power_boundary_checks_geometry() {
    let mut terrain = Landscape::new(Log2Space::new(440.0, 880.0, 96));
    let mut flora = Flora::new(
        7,
        FloraSpec::default(),
        (440.0, 880.0),
        FloraSeeding::Uniform,
        21,
        &terrain,
        48000.0,
        512,
    );
    terrain.flora_observations.push(FloraObservation {
        id: 7,
        generation: 0,
        perception: None,
        level_scan: Vec::new(),
        power_scan: Some(vec![0.0; 1]),
    });
    flora.observe_external(&terrain, Some(1));
}

#[test]
fn growth_and_seed_follow_recorded_pcm_reference() {
    use crate::core::timebase::Timebase;
    use crate::life::schedule_renderer::{ScheduleRenderer, SourcePcm};
    use crate::scenario::control::Routing;

    let (nsgt, params) = crate::runtime::body_metabolism_test_core();
    let fs = nsgt.fs();
    let hop = nsgt.hop();
    for count in [1, 2] {
        for (residual, habitat, leave_self_out) in [
            (false, true, true),
            (true, true, true),
            (true, false, true),
            (false, true, false),
            (true, true, false),
            (true, false, false),
        ] {
            let mut mixed = AnalysisStream::new(params.clone(), nsgt.clone());
            let mut references: Vec<_> = (0..count)
                .map(|_| AnalysisStream::new(params.clone(), nsgt.clone()))
                .collect();
            let mut external = vec![0.0; hop];
            // Independent streams receive the same real pre-birth history.
            for frame in 0..40 {
                for (i, x) in external.iter_mut().enumerate() {
                    *x = 0.03 * (TAU * 220.0 * (frame * hop + i) as f32 / fs).sin();
                }
                mixed.process(&external);
                for reference in &mut references {
                    reference.process(&external);
                }
            }
            let mut terrain = mixed.last().clone();
            terrain.recompute_consonance(&params);
            let mut flora: Vec<_> = (0..count)
                .map(|i| {
                    let mut f = Flora::new(
                        7 + i as u64,
                        FloraSpec {
                            residual,
                            leave_self_out,
                            tone: FloraTone::Resonant,
                            fade_sec: 0.08,
                            routing: Routing {
                                to_presentation: true,
                                to_habitat: habitat,
                            },
                            ..Default::default()
                        },
                        (150.0, 1000.0),
                        FloraSeeding::Consonance,
                        21 + i as u64,
                        &terrain,
                        fs,
                        hop,
                    );
                    // Ensure the assay exercises sounding ownership and retirement.
                    let peak = terrain.space.nearest_index(330.0 + 220.0 * i as f32);
                    f.biomass_mass_scan[peak] = 0.5;
                    f
                })
                .collect();
            let mut observers: Vec<_> = (0..count)
                .map(|i| Observer::flora(7 + i as u64, 0, &mut mixed, flora[i].needs_hr_observer()))
                .collect();
            let mut renderer = ScheduleRenderer::new(Timebase { fs, hop });
            renderer.loo_pcm = (0..count)
                .map(|i| SourcePcm {
                    kind: SourceKind::Flora,
                    analyze_hr: flora[i].needs_hr_observer(),
                    tonal_context: None,
                    id: 7 + i as u64,
                    generation: 0,
                    habitat: vec![0.0; hop],
                })
                .collect();
            let mut removed = vec![0.0; hop];
            let mut own_energy = 0.0;
            let mut tail_energy = 0.0;
            let mut max_hr_error = 0.0f32;
            for frame in 0..160 {
                if frame == 112 {
                    flora[0].release(0.08);
                }
                for (i, f) in flora.iter_mut().enumerate() {
                    f.observe_external(&terrain, Some(frame as u64));
                    f.advance(&terrain);
                    let mut expected = references[i].last().clone();
                    expected.recompute_consonance(&params);
                    let report = f.report(frame as f64 * hop as f64 / fs as f64, &terrain);
                    for (actual, expected) in [
                        (report.growth_level_scan, &expected.consonance_field_level),
                        (report.harmonicity_state_scan, &expected.harmonicity01),
                    ] {
                        for (&a, &b) in actual.iter().zip(expected) {
                            assert!(
                                (a - b).abs() <= 1e-3,
                                "consumed field differs from PCM reference"
                            );
                        }
                    }
                }
                if frame == 80 {
                    mixed.reset();
                    for observer in &mut observers {
                        observer.reset();
                    }
                    for reference in &mut references {
                        reference.reset();
                    }
                }
                let mut full = renderer
                    .render_with_source_pcm(
                        &[],
                        (frame * hop) as u64,
                        &Default::default(),
                        &mut [],
                        &mut flora,
                        |_, _, _, _| {},
                    )
                    .habitat
                    .to_vec();
                for (i, x) in full.iter_mut().enumerate() {
                    *x += 0.03 * (TAU * 220.0 * ((frame + 40) * hop + i) as f32 / fs).sin();
                }
                terrain = mixed.process(&full).clone();
                for i in 0..count {
                    let own = &renderer.loo_pcm[i].habitat;
                    assert_eq!(own, &flora[i].habitat_pcm);
                    let energy: f32 = own.iter().map(|x| x * x).sum();
                    own_energy += energy;
                    if frame >= 112 && i == 0 {
                        tail_energy += energy;
                    }
                    for ((out, &mix), &self_pcm) in removed.iter_mut().zip(&full).zip(own) {
                        *out = if leave_self_out { mix - self_pcm } else { mix };
                    }
                    observers[i].process(&full, own);
                    let mut expected = references[i].process(&removed).clone();
                    expected.recompute_consonance(&params);
                    if habitat && leave_self_out {
                        for (actual, reference) in [
                            (&observers[i].last().harmonicity, &expected.harmonicity),
                            (&observers[i].last().roughness, &expected.roughness),
                        ] {
                            for (&a, &b) in actual.iter().zip(reference) {
                                max_hr_error = max_hr_error.max((a - b).abs());
                                assert!((a - b).abs() <= 1e-3, "B-1 raw H/R tolerance");
                            }
                        }
                    }
                    let mut observation = observers[i].resource_snapshot();
                    observation.revalue(&terrain.space, &params);
                    assert_eq!(observation.perception.is_some(), habitat && leave_self_out);
                    if (40..80).contains(&frame) {
                        assert!(observation.power_scan.is_some());
                    }
                    if let Some(perception) = &observation.perception {
                        for (actual, reference) in [
                            (&perception.h_scan, &expected.harmonicity01),
                            (&perception.r_scan, &expected.roughness01),
                            (&observation.level_scan, &expected.consonance_field_level),
                        ] {
                            for (&a, &b) in actual.iter().zip(reference) {
                                max_hr_error = max_hr_error.max((a - b).abs());
                                assert!((a - b).abs() <= 1e-3, "B-1 raw H/R/C tolerance");
                            }
                        }
                    }
                    terrain.flora_observations.push(observation);
                }
                terrain.recompute_consonance(&params);
            }
            if habitat {
                assert!(own_energy > 0.0 && tail_energy > 0.0);
            } else {
                assert_eq!(own_energy, 0.0);
            }
            println!(
                "F19_PCM count={count} residual={residual} habitat={habitat} leave_self_out={leave_self_out} max_hr_error={max_hr_error} own_energy={own_energy} tail_energy={tail_energy}"
            );
        }
    }
}

#[test]
fn growth_observation_revalues_holds_and_discards_old_geometry() {
    let (nsgt, mut params) = crate::runtime::body_metabolism_test_core();
    let mut mixed = AnalysisStream::new(params.clone(), nsgt);
    let hop = mixed.hop_samples();
    let mut observer = Observer::flora(7, 0, &mut mixed, true);
    let pcm: Vec<_> = (0..hop).map(|i| 0.03 * (i as f32 * 0.17).sin()).collect();
    let zero = vec![0.0; hop];
    for _ in 0..40 {
        observer.process(&pcm, &zero);
    }
    let mut terrain = mixed.last().clone();
    terrain
        .flora_observations
        .push(observer.resource_snapshot());
    params.set_roughness_aversion(2.1);
    params.consonance_representation.beta = 3.0;
    params.consonance_representation.theta = 0.2;
    terrain.recompute_consonance(&params);
    let mut flora = Flora::new(
        7,
        FloraSpec::default(),
        (220.0, 880.0),
        FloraSeeding::Consonance,
        21,
        &terrain,
        params.fs,
        hop,
    );
    flora.rendered_samples = hop as u64;
    flora.advance(&terrain);
    let observation = &terrain.flora_observations[0];
    let perception = observation.perception.as_ref().unwrap();
    for (i, &level) in flora.growth_level_scan.iter().enumerate() {
        assert_eq!(
            level,
            params.consonance_representation.level(
                params
                    .consonance_field_kernel()
                    .score(perception.h_scan[i], perception.r_scan[i])
            )
        );
    }
    let held_growth = flora.growth_level_scan.clone();
    let held_seeds = flora.harmonicity_state_scan.clone();
    terrain.flora_observations.clear();
    terrain.consonance_field_level.fill(1.0);
    terrain.harmonicity01.fill(1.0);
    flora.advance(&terrain);
    assert_eq!(flora.growth_level_scan, held_growth);
    assert_eq!(flora.harmonicity_state_scan, held_seeds);
    let new_space = Log2Space::new(110.0, 1000.0, 48);
    flora.on_space_change(&new_space);
    let mut new_terrain = Landscape::new(new_space);
    new_terrain.consonance_field_level.fill(1.0);
    new_terrain.harmonicity01.fill(1.0);
    flora.advance(&new_terrain);
    assert!(
        flora
            .growth_level_scan
            .iter()
            .chain(&flora.harmonicity_state_scan)
            .all(|&x| x == 0.0)
    );
}
