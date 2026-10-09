use super::super::tests::{spawn_spec_with_freq, test_pop};
use super::*;
use crate::core::consonance_kernel::ConsonanceRepresentationParams;
use crate::core::log2space::Log2Space;
use crate::core::mode_pattern::ModePattern;
use crate::core::timebase::Timebase;
use crate::life::sound::{BodyKind, BodySnapshot, RenderModulatorSpec, Tone, ToneAdsr};
use crate::scenario::control::BodyMethod;
use std::io::{BufRead, BufReader, Write};

fn population(control: crate::scenario::control::VoiceControl) -> RuntimePopulationState {
    let mut spec = spawn_spec_with_freq(440.0);
    spec.control = control;
    RuntimePopulationState {
        template: spec,
        strategy: None,
        respawn_policy: RespawnPolicy::Random,
        respawn_settle_strategy: None,
        respawn_capacity: 10,
        respawn_min_c_level: None,
        respawn_background_death_rate_per_sec: 0.0,
        crowding_target_same: false,
        crowding_target_other: false,
        released: false,
        next_member_idx: 0,
        spawn_count_hint: 1,
    }
}

fn control_for(body: &BodySnapshot) -> crate::scenario::control::VoiceControl {
    let mut control = crate::scenario::control::VoiceControl::default();
    control.body.method = match body.kind {
        BodyKind::Sine => BodyMethod::Sine,
        BodyKind::Harmonic => BodyMethod::Harmonic,
        BodyKind::Modal => BodyMethod::Modal,
    };
    let t = &mut control.body.timbre;
    t.brightness = body.brightness;
    t.inharmonic = body.inharmonic;
    t.motion = body.motion;
    t.spread = body.spread;
    t.unison = body.unison;
    control.body.modes = body
        .ratios
        .as_ref()
        .map(|r| ModePattern::custom_modes(r.to_vec()));
    control
}

#[test]
fn respawn_body_level_uses_configured_sigmoid_and_upper_partials() {
    let mut pop = test_pop();
    let repr = ConsonanceRepresentationParams {
        beta: 3.0,
        theta: 0.1,
    };
    pop.enable_birth_surrogate(0.23, 1e-4, repr);
    let mut control = crate::scenario::control::VoiceControl::default();
    control.body.method = BodyMethod::Harmonic;
    control.body.timbre.motion = 0.0;
    control.body.timbre.spread = 0.0;
    control.body.modes = Some(ModePattern::custom_modes(vec![2.0]));
    control.pitch.freq = 440.0;
    let state = population(control);
    let mut landscape = LandscapeFrame::new(Log2Space::new(100.0, 1600.0, 96));
    landscape.consonance_field_score_eff.fill(-1.0);
    landscape.consonance_field_level_eff.fill(repr.level(-1.0));
    let bin = landscape.space.nearest_index(880.0);
    landscape.consonance_field_score_eff[bin] = 1.0;
    assert_eq!(
        pop.respawn_level(&state, &landscape, 440.0),
        repr.level(1.0)
    );
    assert!(landscape.evaluate_pitch_level(440.0) < 0.22);
    assert!(pop.respawn_level(&state, &landscape, 440.0) > 0.22);
    let mut state = state;
    state.respawn_min_c_level = Some(0.22);
    let parents = BTreeMap::new();
    assert!(
        pop.pick_respawn_candidate(
            1,
            &state,
            &parents,
            &landscape,
            &mut SmallRng::seed_from_u64(17),
            0
        )
        .is_some()
    );
    landscape.consonance_field_score_eff.fill(-1.0);
    assert!(
        pop.pick_respawn_candidate(
            1,
            &state,
            &parents,
            &landscape,
            &mut SmallRng::seed_from_u64(17),
            0
        )
        .is_none()
    );
}

#[test]
fn excluded_final_patterns_keep_legacy_selection_and_rng() {
    let mut pop = test_pop();
    pop.enable_birth_surrogate(0.23, 1e-4, Default::default());
    let legacy = test_pop();
    let mut control = crate::scenario::control::VoiceControl::default();
    control.body.method = BodyMethod::Harmonic;
    control.body.timbre.spread = 0.0;
    let mut state = population(control);
    let landscape = super::super::tests::peak_bias_landscape();
    let parents = BTreeMap::from([(
        1,
        vec![ParentCandidate {
            id: 17,
            freq_hz: 440.0,
            energy: 1.0,
            generation: 3,
        }],
    )]);
    for (policy, pattern) in [
        (
            RespawnPolicy::Random,
            ModePattern::landscape_density_modes(),
        ),
        (
            RespawnPolicy::Hereditary { sigma_oct: 0.1 },
            ModePattern::landscape_density_modes(),
        ),
        (
            RespawnPolicy::Hereditary { sigma_oct: 0.1 },
            ModePattern::landscape_peaks_modes(),
        ),
    ] {
        state.respawn_policy = policy;
        state.template.control.body.modes = Some(pattern);
        for seed in 0..32 {
            let mut a = SmallRng::seed_from_u64(seed);
            let mut b = a.clone();
            assert_eq!(
                pop.pick_respawn_candidate(1, &state, &parents, &landscape, &mut a, 0),
                legacy.pick_respawn_candidate(1, &state, &parents, &landscape, &mut b, 0)
            );
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }
}

fn normalized(scores: &[f32], policy: RespawnPolicy) -> Vec<f64> {
    if matches!(policy, RespawnPolicy::Random) {
        let weights: Vec<f64> = scores.iter().map(|s| f64::from(s.max(0.0))).collect();
        let total: f64 = weights.iter().sum();
        if total > 0.0 {
            return weights.iter().map(|w| w / total).collect();
        }
    }
    let repr = ConsonanceRepresentationParams::default();
    let idx = scores
        .iter()
        .enumerate()
        .max_by(|a, b| repr.level(*a.1).total_cmp(&repr.level(*b.1)))
        .unwrap()
        .0;
    (0..scores.len()).map(|i| f64::from(i == idx)).collect()
}

#[test]
#[ignore = "registered saved fixture final-selection assay; requires N2_ARTIFACT_DIR"]
fn n2_saved_final_distribution() {
    crate::life::modal::register_modal();
    let dir = std::path::PathBuf::from(std::env::var("N2_ARTIFACT_DIR").unwrap());
    let input = std::env::var("N2_REFERENCE_JSONL").unwrap();
    let mut out = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(dir.join("saved-final-distributions.jsonl"))
        .unwrap();
    let mut pop = Community::new(Timebase {
        fs: 48000.0,
        hop: 512,
    });
    pop.enable_birth_surrogate(0.23, 1e-4, Default::default());
    let mut landscape = None;
    let mut old = Vec::new();
    let mut new = Vec::new();
    let mut enabled = true;
    for line in BufReader::new(std::fs::File::open(input).unwrap()).lines() {
        let row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        match row["schema"].as_str().unwrap() {
            "direct-body-v2-environment" => {
                let space = &row["space"];
                let mut frame = LandscapeFrame::new(Log2Space::new(
                    space["fmin"].as_f64().unwrap() as f32,
                    space["fmax"].as_f64().unwrap() as f32,
                    space["bins_per_oct"].as_u64().unwrap() as u32,
                ));
                frame.consonance_field_score_eff =
                    serde_json::from_value(row["c_score_eff_scan"].clone()).unwrap();
                landscape = Some(frame);
            }
            "direct-body-v2-candidate" => {
                let body: BodySnapshot =
                    serde_json::from_value(row["recipe"]["body"].clone()).unwrap();
                let control = control_for(&body);
                let hz = row["candidate_hz"].as_f64().unwrap() as f32;
                let frame = landscape.as_ref().unwrap();
                let mut workspace = pop.birth_surrogate.borrow_mut();
                let work = workspace.as_mut().unwrap();
                let score = if let Some(slot) = work.candidates(&control, frame, 0, (hz, hz), |i| {
                    frame.consonance_field_score_eff[i]
                }) {
                    work.candidate_mass(slot, frame.space.nearest_index(hz))
                } else {
                    enabled = false;
                    0.0
                };
                new.push(score);
                old.push(row["old"]["score"].as_f64().unwrap() as f32);
            }
            "direct-body-v2-selection" => {
                assert_eq!(old.len(), 7);
                let a: Vec<f32> = (0..16).map(|i| old[i % 7]).collect();
                let b: Vec<f32> = (0..16).map(|i| new[i % 7]).collect();
                for (policy, name) in [
                    (RespawnPolicy::Random, "Random"),
                    (RespawnPolicy::Hereditary { sigma_oct: 0.1 }, "Hereditary"),
                ] {
                    let p = normalized(&a, policy);
                    let q = normalized(&b, policy);
                    let tv: f64 = p.iter().zip(&q).map(|(a, b)| (a - b).abs()).sum::<f64>() / 2.0;
                    writeln!(
                        out,
                        "{}",
                        serde_json::json!({"case":row["case"],"environment":row["environment"],
                        "base_hz":row["base_hz"],"policy":name,"body_capability":enabled,
                        "candidate_count":16,"old_scores":a,"body_scores":b,"old_probabilities":p,
                        "body_probabilities":q,"tv":tv})
                    )
                    .unwrap();
                }
                old.clear();
                new.clear();
                enabled = true;
            }
            "direct-body-v2-semantic-summary" => {}
            _ => panic!("unknown schema: {}", row["schema"]),
        }
    }
}

// Independent rendered-and-analyzed reference, restricted to the registered assay.
fn rendered_score(
    row: &serde_json::Value,
    body: BodySnapshot,
    hz: f32,
    landscape: &LandscapeFrame,
    params: &crate::core::landscape::LandscapeParams,
    kernel: &crate::core::nsgt_rt::RtNsgtKernelLog2,
) -> (f32, Vec<f32>) {
    let adsr = &row["recipe"]["adsr"];
    let adsr = ToneAdsr {
        attack_sec: adsr["attack_sec"].as_f64().unwrap() as f32,
        decay_sec: adsr["decay_sec"].as_f64().unwrap() as f32,
        sustain_level: adsr["sustain_level"].as_f64().unwrap() as f32,
        release_sec: adsr["release_sec"].as_f64().unwrap() as f32,
    };
    let modulator = if row["case"] == "drone_sway" {
        RenderModulatorSpec::DroneSway {
            phase: 0.3,
            sway_rate: 0.8,
        }
    } else {
        RenderModulatorSpec::SeqGate { duration_sec: 1.0 }
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
        Some(body),
        Some(modulator),
        Some(adsr),
    )
    .unwrap();
    tone.set_smoothing_tau_sec(0.0);
    tone.seed_modal_phases(crate::life::schedule_renderer::modal_phase_seed(700, 0, 0));
    let kick = crate::life::phonation_engine::OnsetKick { strength: 1.0 };
    tone.schedule_planned_kick(kick);
    tone.arm_onset_trigger(1.0);
    let mut nsgt = kernel.clone();
    let mut frontend =
        crate::core::landscape_spectral::SpectralFrontEnd::new(landscape.space.clone(), params);
    let mut rhythms = crate::core::modulation::NeuralRhythms::default();
    let mut pcm = vec![0.0; 512];
    let mut mean = vec![0.0; landscape.space.n_bins()];
    for frame in 0..72 {
        pcm.fill(0.0);
        let now = frame * 512;
        tone.kick_planned_if_due(now);
        tone.render_block(now, 48000.0, 1.0 / 48000.0, &mut rhythms, &mut pcm);
        let power = nsgt.process_hop(&pcm);
        let observed = frontend.process_nsgt_power(power, 512.0 / 48000.0, params);
        for (m, d) in mean.iter_mut().zip(observed.subjective_intensity) {
            *m += d;
        }
    }
    for m in &mut mean {
        *m *= 1.0 / 72.0;
    }
    let (_, du) = crate::core::roughness_kernel::erb_grid(&landscape.space);
    let mass = mean
        .iter()
        .zip(&du)
        .map(|(d, w)| f64::from(*d) * f64::from(*w))
        .sum::<f64>() as f32;
    let score = mean
        .iter()
        .zip(&du)
        .zip(&landscape.consonance_field_score_eff)
        .map(|((d, w), s)| f64::from(*d) * f64::from(*w) * f64::from(*s))
        .sum::<f64>() as f32
        / mass;
    let normalized = mean.iter().zip(&du).map(|(d, w)| d * w / mass).collect();
    (score, normalized)
}

#[test]
#[ignore = "registered bounded rendered PeakBiased acquisition; requires N2_ARTIFACT_DIR"]
fn n2_peak_render_reference() {
    crate::life::modal::register_modal();
    let dir = std::path::PathBuf::from(std::env::var("N2_ARTIFACT_DIR").unwrap());
    let mut recipes = BTreeMap::new();
    let mut landscape = None;
    for line in
        BufReader::new(std::fs::File::open(std::env::var("N2_REFERENCE_JSONL").unwrap()).unwrap())
            .lines()
    {
        let row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        if row["environment"] != "harmonic_330" {
            continue;
        }
        if row["schema"] == "direct-body-v2-environment" {
            let s = &row["space"];
            let mut frame = LandscapeFrame::new(Log2Space::new(
                s["fmin"].as_f64().unwrap() as f32,
                s["fmax"].as_f64().unwrap() as f32,
                s["bins_per_oct"].as_u64().unwrap() as u32,
            ));
            frame.consonance_field_score_eff =
                serde_json::from_value(row["c_score_eff_scan"].clone()).unwrap();
            for i in 0..frame.space.n_bins() {
                frame.consonance_field_level[i] = ConsonanceRepresentationParams::default()
                    .level(frame.consonance_field_score_eff[i]);
            }
            for line in BufReader::new(
                std::fs::File::open(std::env::var("N2_ENVIRONMENT_JSONL").unwrap()).unwrap(),
            )
            .lines()
            {
                let r: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
                if r["environment"] == "harmonic_330" {
                    frame.consonance_density_mass =
                        serde_json::from_value(r["density_mass_eff_scan"].clone()).unwrap();
                }
            }
            landscape = Some(frame);
        } else if row["schema"] == "direct-body-v2-candidate"
            && row["base_hz"] == 440.0
            && row["cents"] == 0.0
        {
            recipes.insert(row["case"].as_str().unwrap().to_owned(), row);
        }
    }
    let landscape = landscape.unwrap();
    let (params, kernel, _) =
        crate::runtime::birth_surrogate_assay_core(&crate::config::AppConfig::default(), 48000);
    let mut pop = Community::new(Timebase {
        fs: 48000.0,
        hop: 512,
    });
    pop.enable_birth_surrogate(0.23, 1e-4, Default::default());
    let pilot = std::env::var_os("N2_PILOT").is_some();
    let file = if pilot {
        "peak-pilot.jsonl"
    } else {
        "peak-reference.jsonl"
    };
    let mut out = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(dir.join(file))
        .unwrap();
    let mut cases = if pilot {
        vec!["sine"]
    } else {
        vec![
            "sine",
            "harmonic_dark",
            "harmonic_bright",
            "harmonic_inharmonic",
            "modal_bright",
            "adsr_slow",
            "drone_sway",
            "landscape_density",
            "landscape_peaks",
        ]
    };
    if std::env::var_os("N2_SKIP_SINE").is_some() {
        cases.retain(|c| *c != "sine");
    }
    let config = RespawnPeakBiasConfig::default();
    let lo = 440.0 * 2.0f32.powf(-120.0 / 1200.0);
    let hi = 440.0 * 2.0f32.powf(120.0 / 1200.0);
    let bins = peak_bias_candidate_bins(&landscape, lo, hi, 16);
    for case in cases {
        let row = &recipes[case];
        let body: BodySnapshot = serde_json::from_value(row["recipe"]["body"].clone()).unwrap();
        // Evaluate the proposed physical score even for domains rejected by the policy gate.
        let mut state = population(control_for(&body));
        state.respawn_policy = RespawnPolicy::None;
        if case == "landscape_density" {
            state.template.control.body.modes =
                Some(ModePattern::landscape_density_modes().with_count(8));
        }
        if case == "landscape_peaks" {
            state.template.control.body.modes =
                Some(ModePattern::landscape_peaks_modes().with_count(8));
        }
        let mut observations = BTreeMap::<u32, (f32, f32)>::new();
        if pilot {
            let (score, mass) = rendered_score(row, body, 440.0, &landscape, &params, &kernel);
            let expected: Vec<f32> =
                serde_json::from_value(row["old"]["normalized_bin_mass"].clone()).unwrap();
            let l1: f32 = mass.iter().zip(expected).map(|(a, b)| (a - b).abs()).sum();
            writeln!(out,"{}",serde_json::json!({"schema":"pilot","case":case,"saved_density_l1":l1,"rendered_score":score,"saved_score":row["old"]["score"]})).unwrap();
            out.flush().unwrap();
        }
        for &bin in &bins {
            let center = landscape.space.centers_hz[bin].clamp(lo, hi);
            let mut frequencies = vec![center];
            let min = lo
                .log2()
                .max(center.log2() - config.local_search_radius_st / 12.0);
            let max = hi
                .log2()
                .min(center.log2() + config.local_search_radius_st / 12.0);
            let mut cur = min;
            while cur <= max + 1e-6 {
                frequencies.push(2.0f32.powf(cur).clamp(lo, hi));
                cur += config.local_search_step_st / 12.0;
            }
            for hz in frequencies {
                if observations.contains_key(&hz.to_bits()) {
                    continue;
                }
                let mut rng = SmallRng::seed_from_u64(17);
                let body = crate::life::voice::sound_body::build_sound_body_from_control(
                    &state.template.control,
                    hz,
                    48000.0,
                    Some(&landscape),
                    &mut rng,
                )
                .snapshot();
                let (old, _) = rendered_score(row, body, hz, &landscape, &params, &kernel);
                let mut score = [0.0];
                assert!(pop.respawn_body_scores(&state, &landscape, &[hz], &mut score));
                observations.insert(hz.to_bits(), (old, score[0]));
                writeln!(out,"{}",serde_json::json!({"schema":"candidate","case":case,"hz":hz,"hz_bits":hz.to_bits(),"old_score":old,"body_score":score[0]})).unwrap();
            }
        }
        let mut old_weights = Vec::new();
        let mut new_weights = Vec::new();
        let mut old_winners = Vec::new();
        let mut new_winners = Vec::new();
        for &bin in &bins {
            let center = landscape.space.centers_hz[bin].clamp(lo, hi);
            let prior =
                peak_bias_gaussian_weight(12.0 * (center / 440.0).log2(), config.proposal_sigma_st)
                    * if peak_bias_same_band(440.0, center, config.same_band_window_cents) {
                        config.same_band_discount
                    } else {
                        1.0
                    }
                    * if peak_bias_parent_octave(440.0, center, config.octave_window_cents) {
                        config.octave_discount
                    } else {
                        1.0
                    };
            let scores = observations[&center.to_bits()];
            old_weights.push(f64::from(
                scores.0.max(0.0).powf(config.scene_score_exponent) * prior,
            ));
            new_weights.push(f64::from(
                scores.1.max(0.0).powf(config.scene_score_exponent) * prior,
            ));
            old_winners.push(
                peak_bias_local_search_with(center, lo, hi, config, |hz| {
                    observations[&hz.to_bits()].0
                })
                .to_bits(),
            );
            new_winners.push(
                peak_bias_local_search_with(center, lo, hi, config, |hz| {
                    observations[&hz.to_bits()].1
                })
                .to_bits(),
            );
        }
        let mut distributions = [BTreeMap::<u32, f64>::new(), BTreeMap::new()];
        for (dist, (weights, winners)) in distributions
            .iter_mut()
            .zip([(&old_weights, &old_winners), (&new_weights, &new_winners)])
        {
            let total: f64 = weights.iter().sum();
            assert!(total > 0.0);
            for (w, hz) in weights.iter().zip(winners) {
                *dist.entry(*hz).or_default() += w / total;
            }
        }
        let mut support = std::collections::BTreeSet::new();
        support.extend(distributions[0].keys());
        support.extend(distributions[1].keys());
        let tv: f64 = support
            .iter()
            .map(|k| {
                (distributions[0].get(k).unwrap_or(&0.0) - distributions[1].get(k).unwrap_or(&0.0))
                    .abs()
            })
            .sum::<f64>()
            / 2.0;
        writeln!(out,"{}",serde_json::json!({"schema":"distribution","case":case,"policy":"PeakBiased","centers":bins.len(),"rendered_candidates":observations.len(),"old_distribution":distributions[0],"body_distribution":distributions[1],"tv":tv})).unwrap();
        out.flush().unwrap();
        println!("{case}: PeakBiased TV={tv}");
    }
}

#[test]
fn peak_body_capability_matches_measured_recipe_domains() {
    let mut pop = test_pop();
    pop.enable_birth_surrogate(0.23, 1e-4, Default::default());
    let landscape = super::super::tests::peak_bias_landscape();
    let mut state = population(crate::scenario::control::VoiceControl::default());
    state.respawn_policy = RespawnPolicy::PeakBiased {
        config: Default::default(),
    };
    state.template.control.body.method = BodyMethod::Harmonic;
    state.template.control.body.timbre.spread = 0.0;
    state.template.control.body.timbre.inharmonic = 0.7;
    let mut score = [0.0];
    assert!(pop.respawn_body_scores(&state, &landscape, &[440.0], &mut score));
    for pattern in [
        ModePattern::landscape_density_modes(),
        ModePattern::landscape_peaks_modes(),
    ] {
        state.template.control.body.modes = Some(pattern);
        assert!(pop.respawn_body_scores(&state, &landscape, &[440.0], &mut score));
    }
    state.template.control.body.modes = Some(ModePattern::custom_modes(vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
    ]));
    assert!(!pop.respawn_body_scores(&state, &landscape, &[440.0], &mut score));
    for method in [BodyMethod::Sine, BodyMethod::Modal] {
        state.template.control.body.method = method;
        assert!(!pop.respawn_body_scores(&state, &landscape, &[440.0], &mut score));
    }
}

#[test]
fn constant_body_scene_keeps_random_and_hereditary_rng_and_parent_rules() {
    let mut pop = test_pop();
    pop.enable_birth_surrogate(0.23, 1e-4, Default::default());
    let legacy = test_pop();
    let mut control = crate::scenario::control::VoiceControl::default();
    control.body.method = BodyMethod::Harmonic;
    control.body.timbre.spread = 0.0;
    let mut state = population(control);
    state.strategy = Some(SpawnStrategy::Linear {
        start_freq: 220.0,
        end_freq: 880.0,
    });
    state.spawn_count_hint = 16;
    let parents = BTreeMap::from([(
        1,
        vec![ParentCandidate {
            id: 17,
            freq_hz: 440.0,
            energy: 1.0,
            generation: 3,
        }],
    )]);
    let mut landscape = LandscapeFrame::new(Log2Space::new(55.0, 8000.0, 96));
    for score in [0.0, 0.5] {
        landscape.consonance_field_score_eff.fill(score);
        landscape
            .consonance_field_level_eff
            .fill(ConsonanceRepresentationParams::default().level(score));
        for policy in [
            RespawnPolicy::Random,
            RespawnPolicy::Hereditary { sigma_oct: 0.1 },
        ] {
            state.respawn_policy = policy;
            for seed in 0..32 {
                let mut a = SmallRng::seed_from_u64(seed);
                let mut b = a.clone();
                let on = pop.pick_respawn_candidate(1, &state, &parents, &landscape, &mut a, 0);
                let off = legacy.pick_respawn_candidate(1, &state, &parents, &landscape, &mut b, 0);
                assert_eq!(on, off);
                assert_eq!(a.next_u64(), b.next_u64());
                if matches!(policy, RespawnPolicy::Hereditary { .. }) {
                    assert_eq!(on.unwrap().1, Some(17));
                    assert_eq!(on.unwrap().2, Some(3));
                }
            }
        }
    }
}

#[test]
#[ignore = "registered release respawn cost diagnostic; requires N2_ARTIFACT_DIR"]
fn n2_respawn_cost() {
    let dir = std::path::PathBuf::from(std::env::var("N2_ARTIFACT_DIR").unwrap());
    let mut out = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(dir.join("respawn-cost.jsonl"))
        .unwrap();
    let mut pop = Community::new(Timebase {
        fs: 48000.0,
        hop: 512,
    });
    pop.enable_birth_surrogate(0.23, 1e-4, Default::default());
    let legacy = Community::new(Timebase {
        fs: 48000.0,
        hop: 512,
    });
    let source = super::super::tests::peak_bias_landscape();
    let mut landscape = LandscapeFrame::new(Log2Space::new(55.0, 8000.0, 96));
    for (idx, &hz) in landscape.space.centers_hz.iter().enumerate() {
        landscape.consonance_field_score_eff[idx] = if (220.0..=880.0).contains(&hz) {
            source.evaluate_pitch_score(hz)
        } else {
            0.0
        };
    }
    let mut control = crate::scenario::control::VoiceControl::default();
    control.body.method = BodyMethod::Harmonic;
    control.body.timbre.spread = 0.0;
    control.body.timbre.inharmonic = 0.7;
    let mut state = population(control);
    state.strategy = Some(SpawnStrategy::Linear {
        start_freq: 220.0,
        end_freq: 880.0,
    });
    state.spawn_count_hint = 16;
    let parents = BTreeMap::from([(
        1,
        vec![ParentCandidate {
            id: 17,
            freq_hz: 440.0,
            energy: 1.0,
            generation: 3,
        }],
    )]);
    for (policy, name) in [
        (RespawnPolicy::Random, "Random"),
        (RespawnPolicy::Hereditary { sigma_oct: 0.1 }, "Hereditary"),
        (
            RespawnPolicy::PeakBiased {
                config: Default::default(),
            },
            "PeakBiased",
        ),
    ] {
        state.respawn_policy = policy;
        let frequencies: [f32; 16] = std::array::from_fn(|i| 220.0 + 660.0 * i as f32 / 15.0);
        let mut scores = [0.0; 16];
        assert!(pop.respawn_body_scores(&state, &landscape, &frequencies, &mut scores));
        for count in [1, 10] {
            for sample in 0..5 {
                pop.current_frame += 1;
                for (enabled, model) in [(false, &legacy), (true, &pop)] {
                    let mut rng = SmallRng::seed_from_u64(7304 + sample);
                    let started = std::time::Instant::now();
                    for member in 0..count {
                        assert!(
                            std::hint::black_box(model.pick_respawn_candidate(
                                1, &state, &parents, &landscape, &mut rng, member
                            ))
                            .is_some()
                        );
                    }
                    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
                    writeln!(out,"{}",serde_json::json!({"policy":name,"respawns":count,"sample":sample,"enabled":enabled,"ms":elapsed})).unwrap();
                }
            }
        }
    }
}

#[test]
fn finite_body_scores_with_overflowing_weight_sum_still_choose_a_candidate() {
    let mut pop = test_pop();
    pop.enable_birth_surrogate(0.23, 1e30, Default::default());
    let mut control = crate::scenario::control::VoiceControl::default();
    control.body.method = BodyMethod::Harmonic;
    control.body.timbre.spread = 0.0;
    control.body.modes = Some(ModePattern::custom_modes(vec![1.0]));
    let mut state = population(control);
    state.strategy = Some(SpawnStrategy::Linear {
        start_freq: 220.0,
        end_freq: 880.0,
    });
    state.spawn_count_hint = 16;
    let mut landscape = LandscapeFrame::new(Log2Space::new(55.0, 8000.0, 96));
    landscape.consonance_field_score_eff.fill(1e38);
    let mut scores = [0.0; 16];
    let frequencies: [f32; 16] = std::array::from_fn(|i| 220.0 + 660.0 * i as f32 / 15.0);
    assert!(pop.respawn_body_scores(&state, &landscape, &frequencies, &mut scores));
    assert!(scores.iter().all(|s| s.is_finite() && *s > 0.0));
    assert!(!scores.iter().sum::<f32>().is_finite());
    assert!(
        pop.pick_respawn_candidate(
            1,
            &state,
            &BTreeMap::new(),
            &landscape,
            &mut SmallRng::seed_from_u64(17),
            0
        )
        .is_some()
    );
}
