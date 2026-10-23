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
    terrain.flora_power.push(FloraObservation {
        id: 7,
        generation: 1,
        power_scan: Some(vec![0.00045; terrain.space.n_bins()]),
    });
    flora.observe_external(&terrain, Some(8));
    assert!(flora.external_observed_frame.is_none());
    terrain.flora_power[0].generation = 0;
    flora.observe_external(&terrain, Some(9));
    let expected = terrain.space.bins_per_oct as f32 / REFERENCE_DENSITY;
    for &ce in &flora.external_consumption_scan {
        assert!((ce - expected).abs() < 16.0 * f32::EPSILON * expected);
    }
    let ce = flora.external_consumption_scan.clone();
    flora.apply_patch(&FloraPatch { amp: Some(0.0) });
    flora.observe_external(&terrain, Some(10));
    assert_eq!(ce, flora.external_consumption_scan);
    flora.advance(&terrain);
    assert!(
        flora
            .resource_state_scan
            .iter()
            .all(|&u| (0.0..1.0).contains(&u))
    );
    terrain.flora_power.clear();
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
                Observer::flora(7, 0, &mut mixed),
                Observer::flora(8, 0, &mut mixed),
            ];
            let mut references = [
                Observer::flora(7, 0, &mut mixed),
                Observer::flora(8, 0, &mut mixed),
            ];
            let mut renderer = ScheduleRenderer::new(Timebase { fs, hop });
            renderer.loo_pcm = [7, 8]
                .into_iter()
                .map(|id| SourcePcm {
                    kind: SourceKind::Flora,
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
                    flora[0].apply_patch(&FloraPatch { amp: Some(0.0) });
                }
                if frame == 64 {
                    flora[0].apply_patch(&FloraPatch { amp: Some(0.03) });
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
    terrain.flora_power.push(FloraObservation {
        id: 7,
        generation: 0,
        power_scan: Some(vec![0.0; 1]),
    });
    flora.observe_external(&terrain, Some(1));
}
