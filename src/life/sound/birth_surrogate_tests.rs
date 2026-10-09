use super::*;
use crate::life::voice::sound_body::{SoundBody, SoundBodyBuildInput, SoundBodyFactory};
use rand::{SeedableRng, rngs::SmallRng};
use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Write};

#[test]
fn deterministic_recipe_preview_matches_each_builtin_body() {
    crate::life::modal::register_modal();
    let space = Log2Space::new(55.0, 8000.0, 96);
    let mut landscape = LandscapeFrame::new(space);
    for i in 0..landscape.space.n_bins() {
        landscape.consonance_density_mass[i] = (i % 17) as f32 / 17.0;
        landscape.consonance_field_level[i] = (i % 13) as f32 / 13.0;
    }
    for method in [
        crate::scenario::control::BodyMethod::Sine,
        crate::scenario::control::BodyMethod::Harmonic,
        crate::scenario::control::BodyMethod::Modal,
    ] {
        for pattern in [
            None,
            Some(ModePattern::harmonic_modes().with_count(6)),
            Some(ModePattern::odd_modes().with_count(6)),
            Some(ModePattern::power_modes(1.2).with_count(6)),
            Some(ModePattern::stiff_string_modes(0.02).with_count(6)),
            Some(ModePattern::custom_modes(vec![1.0, 2.7, 5.3, 8.8])),
            Some(ModePattern::landscape_density_modes().with_count(8)),
            Some(ModePattern::landscape_peaks_modes().with_count(8)),
        ] {
            let mut control = VoiceControl::default();
            control.body.method = method;
            control.body.timbre.brightness = 1.0;
            control.body.timbre.spread = 0.0;
            control.body.timbre.unison = 1;
            control.body.modes = pattern;
            for hz in [220.0, 440.0, 1800.0] {
                let mut rng = SmallRng::seed_from_u64(17);
                let actual = crate::life::voice::sound_body::build_sound_body_from_control(
                    &control,
                    hz,
                    48000.0,
                    Some(&landscape),
                    &mut rng,
                )
                .snapshot();
                let mut work = BirthSurrogate::new(
                    Timebase {
                        fs: 48000.0,
                        hop: 512,
                    },
                    0.23,
                    1e-4,
                );
                let slot = work
                    .candidates(&control, &landscape, 0, (hz, hz), |_| 1.0)
                    .unwrap();
                assert_eq!(legacy_renderer_kind(&control), Some(actual.kind));
                assert!(BirthSurrogate::supports_snapshot(&actual));
                assert!(
                    (work.candidate_mass(slot, landscape.space.nearest_index(hz)) - 1.0).abs()
                        < 1e-6
                );
                let expected = actual.ratios.as_deref();
                let mut preview = Vec::with_capacity(MAX_RATIOS);
                if let Some(pattern) = &control.body.modes {
                    assert!(pattern.eval_without_jitter_into(
                        hz,
                        &landscape,
                        &mut preview,
                        &mut work.pattern_weights,
                        &mut work.pattern_candidates
                    ));
                    if actual.kind != BodyKind::Sine {
                        assert_eq!(expected, Some(preview.as_slice()));
                    }
                }
            }
        }
    }
}

#[test]
fn new_factories_do_not_claim_legacy_renderer_conformance() {
    struct NewRenderer;
    impl SoundBodyFactory for NewRenderer {
        fn build(
            &self,
            _: &SoundBodyBuildInput<'_>,
            _: &mut SmallRng,
        ) -> crate::life::voice::sound_body::AnySoundBody {
            unreachable!()
        }
    }
    assert_eq!(NewRenderer.legacy_renderer_kind(), None);
}

#[test]
fn cache_reuses_equal_recipes_but_reads_current_terrain_and_space() {
    let mut control = VoiceControl::default();
    control.body.method = crate::scenario::control::BodyMethod::Harmonic;
    control.body.timbre.spread = 0.0;
    control.body.timbre.unison = 1;
    let mut landscape = LandscapeFrame::new(Log2Space::new(100.0, 2000.0, 24));
    let mut work = BirthSurrogate::new(
        Timebase {
            fs: 48000.0,
            hop: 512,
        },
        0.23,
        1e-4,
    );
    for _ in 0..10 {
        work.candidates(&control, &landscape, 7, (200.0, 400.0), |_| 0.5)
            .unwrap();
    }
    assert_eq!(work.preparations, 1);
    for n in 0..9 {
        control.body.timbre.brightness = n as f32 / 10.0;
        work.candidates(&control, &landscape, 7, (200.0, 400.0), |_| 0.5)
            .unwrap();
    }
    control.body.timbre.brightness = 0.6;
    work.candidates(&control, &landscape, 7, (200.0, 400.0), |_| 0.5)
        .unwrap();
    assert_eq!(work.preparations, 9);
    work.candidates(&control, &landscape, 7, (200.0, 400.0), |_| 0.25)
        .unwrap();
    assert_eq!(work.preparations, 10);
    landscape = LandscapeFrame::new(Log2Space::new(100.0, 2000.0, 12));
    work.candidates(&control, &landscape, 7, (200.0, 400.0), |_| 0.25)
        .unwrap();
    assert_eq!(work.preparations, 11);
    let (_, expected_du) = crate::core::roughness_kernel::erb_grid(&landscape.space);
    assert_eq!(work.du_scan, expected_du);
}

/// Recover only the four environment inputs with the adopted v1 generator.
/// Candidate body render references remain the immutable saved acquisition.
#[test]
#[ignore = "authorized input recovery; requires B4_REFERENCE_JSONL and B4_ARTIFACT_DIR"]
fn recover_saved_environments() {
    use crate::core::stream::analysis::AnalysisStream;
    use crate::life::phonation_engine::OnsetKick;
    use crate::life::schedule_renderer::modal_phase_seed;
    use crate::life::sound::{RenderModulatorSpec, Tone, ToneAdsr};
    let input = std::env::var("B4_REFERENCE_JSONL").unwrap();
    let dir = std::env::var("B4_ARTIFACT_DIR").unwrap();
    let mut saved = BTreeMap::new();
    for line in BufReader::new(std::fs::File::open(input).unwrap()).lines() {
        let row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        if row["schema"] == "direct-body-v2-environment" {
            saved.insert(row["environment"].as_str().unwrap().to_owned(), row);
        }
    }
    assert_eq!(saved.len(), 4);
    let (params, kernel, hop) =
        crate::runtime::birth_surrogate_assay_core(&crate::config::AppConfig::default(), 48000);
    assert_eq!(hop, 512);
    let path = std::path::Path::new(&dir).join("recovered-environments.jsonl");
    let mut output = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap();
    for env in ["silence", "sine_440", "sine_660", "harmonic_330"] {
        let mut analysis = AnalysisStream::new(params.clone(), kernel.clone());
        let mut tone = if env == "silence" {
            None
        } else {
            let (kind, hz) = match env {
                "sine_440" => (BodyKind::Sine, 440.0),
                "sine_660" => (BodyKind::Sine, 660.0),
                "harmonic_330" => (BodyKind::Harmonic, 330.0),
                _ => unreachable!(),
            };
            let mut tone = Tone::from_parts(
                Timebase {
                    fs: 48000.0,
                    hop: 512,
                },
                0,
                48000,
                hz,
                1.0,
                Some(BodySnapshot {
                    kind,
                    amp_scale: 1.0,
                    brightness: 0.8,
                    inharmonic: 0.0,
                    spread: 0.0,
                    unison: 1,
                    motion: 0.0,
                    ratios: None,
                }),
                Some(RenderModulatorSpec::SeqGate { duration_sec: 1.0 }),
                Some(ToneAdsr {
                    attack_sec: 0.005,
                    decay_sec: 0.0,
                    sustain_level: 1.0,
                    release_sec: 0.5,
                }),
            )
            .unwrap();
            tone.seed_modal_phases(modal_phase_seed(900, 0, 0));
            let kick = OnsetKick { strength: 1.0 };
            tone.schedule_planned_kick(kick);
            tone.arm_onset_trigger(kick.strength);
            Some(tone)
        };
        let mut rhythms = crate::core::modulation::NeuralRhythms::default();
        let mut pcm = vec![0.0; 512];
        for frame in 0..72 {
            pcm.fill(0.0);
            if let Some(tone) = tone.as_mut() {
                let now = (frame * 512) as u64;
                tone.kick_planned_if_due(now);
                tone.render_block(now, 48000.0, 1.0 / 48000.0, &mut rhythms, &mut pcm);
            }
            analysis.process(&pcm);
        }
        let mut landscape = analysis.last().clone();
        landscape.recompute_consonance(&params);
        let (_, du) = crate::core::roughness_kernel::erb_grid(&landscape.space);
        let original = &saved[env];
        assert_eq!(original["fs"], 48000.0);
        assert_eq!(original["hop"], 512);
        assert_eq!(original["frames"], 72);
        let mut differences = serde_json::Map::new();
        for (name, scan) in [
            ("du", &du),
            ("c_score_eff_scan", &landscape.consonance_field_score_eff),
        ] {
            let prior: Vec<f32> = serde_json::from_value(original[name].clone()).unwrap();
            landscape.space.assert_scan_len_named(&prior, name);
            let differing_bins = prior
                .iter()
                .zip(scan)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let max_abs = prior
                .iter()
                .zip(scan)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            differences.insert(
                name.into(),
                serde_json::json!({"differing_bins":differing_bins,"max_abs":max_abs}),
            );
        }
        let space = serde_json::json!({"fmin":landscape.space.fmin,"fmax":landscape.space.fmax,
            "bins_per_oct":landscape.space.bins_per_oct});
        differences.insert(
            "space_matches".into(),
            serde_json::json!(original["space"] == space),
        );
        let row = serde_json::json!({"schema":"b4-recovered-environment", "environment":env,
            "fs":48000.0,"hop":512,"frames":72,"space":space,"du":du,
            "density_mass_eff_scan":landscape.consonance_density_mass_eff,"saved_differences":differences,
            "source":"fa093b8 direct_body_fitness_tests::environment_landscape; runtime default core"});
        writeln!(output, "{row}").unwrap();
        println!(
            "{}",
            serde_json::json!({"environment":env,"saved_differences":row["saved_differences"]})
        );
    }
}

/// Run the production density and inner product over all saved candidate recipes.
#[test]
#[ignore = "requires B4_REFERENCE_JSONL and B4_ARTIFACT_DIR with recovered environments"]
fn saved_v1_density_and_distribution_conformance() {
    crate::life::modal::register_modal();
    let path = std::env::var("B4_REFERENCE_JSONL").expect("adopted semantic-v1 JSONL path");
    let file = std::fs::File::open(path).unwrap();
    let dir = std::env::var("B4_ARTIFACT_DIR").unwrap();
    let mut terrains = BTreeMap::new();
    for line in BufReader::new(
        std::fs::File::open(std::path::Path::new(&dir).join("recovered-environments.jsonl"))
            .unwrap(),
    )
    .lines()
    {
        let row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        let scan: Vec<f32> = serde_json::from_value(row["density_mass_eff_scan"].clone()).unwrap();
        terrains.insert(row["environment"].as_str().unwrap().to_owned(), scan);
    }
    assert_eq!(terrains.len(), 4);
    let mut output =
        std::fs::File::create(std::path::Path::new(&dir).join("implemented-distributions.jsonl"))
            .unwrap();
    let mut implemented_weights = Vec::<f64>::with_capacity(7);
    let mut max_saved_weight_abs = 0.0f64;
    let mut max_preview_weight_abs = 0.0f64;
    let mut space = None;
    let mut work = BirthSurrogate::new(
        Timebase {
            fs: 48000.0,
            hop: 512,
        },
        0.23,
        1e-4,
    );
    let mut support = false;
    let mut candidates = 0;
    let mut groups = 0;
    let mut maxima = BTreeMap::<String, (bool, f64)>::new();
    let mut max_density_l1 = 0.0f32;
    for line in BufReader::new(file).lines() {
        let row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        match row["schema"].as_str().unwrap() {
            "direct-body-v2-environment" => {
                let s = &row["space"];
                space = Some(Log2Space::new(
                    s["fmin"].as_f64().unwrap() as f32,
                    s["fmax"].as_f64().unwrap() as f32,
                    s["bins_per_oct"].as_u64().unwrap() as u32,
                ));
                assert_eq!(row["fs"], 48000.0);
                assert_eq!(row["hop"], 512);
                assert!(work.set_space(space.as_ref().unwrap()));
                let du: Vec<f32> = serde_json::from_value(row["du"].clone()).unwrap();
                assert_eq!(work.du_scan, du);
            }
            "direct-body-v2-candidate" => {
                let space = space.as_ref().unwrap();
                let body: BodySnapshot =
                    serde_json::from_value(row["recipe"]["body"].clone()).unwrap();
                support = BirthSurrogate::supports_snapshot(&body);
                assert!(work.prepare(&body, body.ratios.as_deref()));
                let mass = work
                    .density(
                        body.kind,
                        row["candidate_hz"].as_f64().unwrap() as f32,
                        space,
                    )
                    .unwrap();
                let expected: Vec<f32> =
                    serde_json::from_value(row["new"]["normalized_bin_mass"].clone()).unwrap();
                space.assert_scan_len_named(&expected, "saved_birth_normalized_bin_mass_scan");
                let l1 = work
                    .density_scan
                    .iter()
                    .zip(&work.du_scan)
                    .zip(expected)
                    .map(|((&d, &w), e)| (d * w / mass - e).abs())
                    .sum::<f32>();
                max_density_l1 = max_density_l1.max(l1);
                assert!(
                    l1 <= 1e-5,
                    "{} {} density L1={l1}",
                    row["case"],
                    row["candidate_hz"]
                );
                candidates += 1;
                let terrain = &terrains[row["environment"].as_str().unwrap()];
                space.assert_scan_len_named(terrain, "recovered_birth_density_mass_scan");
                let direct_weight = work.density_weight(mass, terrain, space);
                let weight = if support {
                    let mut landscape = LandscapeFrame::new(space.clone());
                    landscape.consonance_density_mass_eff.clone_from(terrain);
                    let mut control = VoiceControl::default();
                    control.body.method = match body.kind {
                        BodyKind::Sine => crate::scenario::control::BodyMethod::Sine,
                        BodyKind::Harmonic => crate::scenario::control::BodyMethod::Harmonic,
                        BodyKind::Modal => crate::scenario::control::BodyMethod::Modal,
                    };
                    control.body.timbre.brightness = body.brightness;
                    control.body.timbre.inharmonic = body.inharmonic;
                    control.body.timbre.spread = body.spread;
                    control.body.timbre.unison = body.unison;
                    control.body.timbre.motion = body.motion;
                    control.body.modes = body
                        .ratios
                        .as_ref()
                        .map(|r| ModePattern::custom_modes(r.to_vec()));
                    let hz = row["candidate_hz"].as_f64().unwrap() as f32;
                    let slot = work
                        .candidates(&control, &landscape, 0, (hz, hz), |i| terrain[i])
                        .unwrap();
                    let preview_weight = work.candidate_mass(slot, space.nearest_index(hz));
                    max_preview_weight_abs = max_preview_weight_abs
                        .max(f64::from((preview_weight - direct_weight).abs()));
                    preview_weight
                } else {
                    direct_weight
                };
                implemented_weights.push(f64::from(weight));
            }
            "direct-body-v2-selection" => {
                assert_eq!(implemented_weights.len(), 7);
                let saved_weights: Vec<f64> =
                    serde_json::from_value(row["new_density_weights"].clone()).unwrap();
                for (&actual, prior) in implemented_weights.iter().zip(saved_weights) {
                    max_saved_weight_abs = max_saved_weight_abs.max((actual - prior).abs());
                }
                let new = &implemented_weights;
                let old: Vec<f64> =
                    serde_json::from_value(row["old_probabilities"].clone()).unwrap();
                let total = new.iter().sum::<f64>();
                let old_total = old.iter().sum::<f64>();
                let tv = 0.5
                    * new
                        .iter()
                        .zip(old)
                        .map(|(&w, p)| {
                            let q = if total > 0.0 {
                                w / total
                            } else {
                                1.0 / new.len() as f64
                            };
                            (q - p / old_total).abs()
                        })
                        .sum::<f64>();
                let case = row["case"].as_str().unwrap().to_owned();
                let entry = maxima.entry(case).or_insert((support, 0.0));
                assert_eq!(entry.0, support);
                entry.1 = entry.1.max(tv);
                writeln!(output, "{}", serde_json::json!({"environment":row["environment"],
                    "case":row["case"],"base_hz":row["base_hz"],"supported":support,
                    "implemented_weights":new,"rendered_probabilities":row["old_probabilities"],"tv":tv})).unwrap();
                implemented_weights.clear();
                groups += 1;
            }
            "direct-body-v2-semantic-summary" => {}
            schema => panic!("unexpected producer row: {schema}"),
        }
    }
    assert_eq!((candidates, groups, maxima.len()), (728, 104, 13));
    assert_eq!(
        maxima.values().filter(|(supported, _)| *supported).count(),
        9
    );
    for (family, (supported, maximum_tv)) in &maxima {
        if *supported {
            assert!(*maximum_tv <= 0.1, "{family} TV={maximum_tv}");
        } else {
            assert!(*maximum_tv > 0.1, "{family} exclusion lacks a >0.1 example");
        }
    }
    let summary = serde_json::json!({"schema":"b4-implemented-conformance",
        "candidates": candidates, "groups": groups, "max_density_l1": max_density_l1,
        "families": maxima,"max_saved_weight_abs":max_saved_weight_abs,
        "max_preview_weight_abs":max_preview_weight_abs,
        "scope":"implemented v1 density and birth inner product; saved candidate rendering; recovered environment inputs"});
    std::fs::write(
        std::path::Path::new(&dir).join("conformance-summary.json"),
        format!("{summary}\n"),
    )
    .unwrap();
    println!("{summary}");
}

#[test]
fn respawn_point_scores_keep_preallocated_capacity_and_exact_frequency_mapping() {
    let mut work = BirthSurrogate::new(
        Timebase {
            fs: 48000.0,
            hop: 512,
        },
        0.23,
        1e-4,
    );
    let mut control = VoiceControl::default();
    control.body.method = crate::scenario::control::BodyMethod::Sine;
    let mut landscape = LandscapeFrame::new(Log2Space::new(55.0, 8000.0, 96));
    let hz = 440.37;
    let bin = landscape.space.nearest_index(hz);
    landscape.consonance_field_score_eff[bin] = 1.0;
    let before: Vec<usize> = work.cache.iter().map(|c| c.masses.capacity()).collect();
    let scratch_before = [
        work.du_scan.capacity(),
        work.power_scan.capacity(),
        work.density_scan.capacity(),
        work.terrain_scan.capacity(),
        work.lanes.capacity(),
        work.ratios.capacity(),
    ];
    for frame in 0..10 {
        let slot = work
            .respawn_scores(
                &control,
                crate::scenario::RespawnPolicy::Hereditary { sigma_oct: 0.1 },
                &landscape,
                frame,
                &[hz; 16],
            )
            .unwrap();
        assert_eq!(work.candidate_mass(slot, 0), 1.0);
    }
    assert_eq!(
        before,
        work.cache
            .iter()
            .map(|c| c.masses.capacity())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        scratch_before,
        [
            work.du_scan.capacity(),
            work.power_scan.capacity(),
            work.density_scan.capacity(),
            work.terrain_scan.capacity(),
            work.lanes.capacity(),
            work.ratios.capacity()
        ]
    );
}

#[test]
#[should_panic(expected = "respawn_consonance_field_score_eff_scan")]
fn respawn_score_boundary_rejects_misaligned_scan() {
    let mut work = BirthSurrogate::new(
        Timebase {
            fs: 48000.0,
            hop: 512,
        },
        0.23,
        1e-4,
    );
    let mut control = VoiceControl::default();
    control.body.method = crate::scenario::control::BodyMethod::Sine;
    let mut landscape = LandscapeFrame::new(Log2Space::new(55.0, 8000.0, 96));
    landscape.consonance_field_score_eff.pop();
    work.respawn_scores(
        &control,
        crate::scenario::RespawnPolicy::Hereditary { sigma_oct: 0.1 },
        &landscape,
        0,
        &[440.0],
    );
}
