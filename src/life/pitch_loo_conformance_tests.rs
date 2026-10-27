//! B-1 PCM self-removal reference, ported to the current runtime and three LOO methods.
//! PCM consumer checks replace the obsolete T-31 score snapshots.
use super::*;
use crate::config::AppConfig;
use crate::core::consonance_kernel::ConsonanceRepresentationParams;
use crate::core::landscape::Landscape;
use crate::core::mode_pattern::ModePattern;
use crate::core::modulation::NeuralRhythms;
use crate::core::roughness_kernel::erb_grid;
use crate::core::stream::analysis::AnalysisStream;
use crate::core::stream::synchronous::SynchronousAnalysis;
use crate::core::timebase::Timebase;
use crate::life::body_fitness;
use crate::life::onset_footprint::{Identity, Recipe};
use crate::life::voice::ArticulationSignal;

use crate::life::phonation_engine::{OnsetKick, ToneCmd};
use crate::life::schedule_renderer::ScheduleRenderer;
use crate::life::sound::{BodyKind, BodySnapshot, RenderModulatorSpec, ToneAdsr};
use crate::life::voice::{PhonationBatch, ToneSpec, VoiceMetadata};
use crate::scenario::control::{BodyMethod, VoiceControl};
use crate::scenario::{ArticulationCoreConfig, VoiceSpec};
use rand::{SeedableRng, rngs::SmallRng};
use serde_json::json;
use std::cell::Cell;
use std::collections::BTreeMap;
use std::sync::Arc;

const FS: f32 = 48_000.0;
const ENV_FRAMES: usize = 64;
const BODY_FRAMES: usize = 72;
const CENTS: [f32; 7] = [-120.0, -60.0, -24.0, 0.0, 24.0, 60.0, 120.0];

fn batch(source_id: u64, recipe: &Recipe) -> PhonationBatch {
    PhonationBatch {
        source_id,
        cmds: vec![ToneCmd::On {
            tone_id: 1,
            kick: OnsetKick { strength: 1.0 },
        }],
        tones: vec![ToneSpec {
            phrase: None,
            participation_onset: false,
            tone_id: 1,
            onset: 0,
            hold_ticks: Some(recipe.hold),
            freq_hz: recipe.freq_hz,
            amp: 0.35,
            smoothing_tau_sec: recipe.smoothing_tau_sec,
            body: recipe.body.clone(),
            render_modulator: recipe.modulator.clone(),
            adsr: recipe.adsr,
        }],
        ..Default::default()
    }
}

fn recipe(family: &str, hz: f32, environment: &Landscape) -> Recipe {
    let kind = match family {
        "sine" => BodyKind::Sine,
        "modal_dark" | "modal_bright" | "modal_spread" | "modal_default" | "landscape_peaks" => {
            BodyKind::Modal
        }
        _ => BodyKind::Harmonic,
    };
    let ratios = if family == "harmonic_default" || family == "modal_default" {
        None
    } else {
        match kind {
            BodyKind::Sine => None,
            BodyKind::Modal => Some(Arc::<[f32]>::from(vec![1.0, 2.7, 5.3, 8.8])),
            BodyKind::Harmonic => Some(Arc::<[f32]>::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
        }
    };
    let mut body = BodySnapshot {
        kind,
        amp_scale: 1.0,
        brightness: 0.8,
        inharmonic: 0.0,
        spread: 0.0,
        unison: 1,
        motion: 0.0,
        ratios,
    };
    match family {
        "harmonic_dark" | "modal_dark" => body.brightness = 0.1,
        "harmonic_bright" | "modal_bright" => body.brightness = 1.0,
        "harmonic_inharmonic" => {
            body.ratios = None;
            body.inharmonic = 0.7;
        }
        "harmonic_spread" | "modal_spread" => {
            body.spread = 0.8;
            body.unison = 5;
        }
        "motion" => body.motion = 0.9,
        _ => {}
    }
    let adsr = if family == "adsr_slow" {
        ToneAdsr {
            attack_sec: 0.4,
            decay_sec: 0.1,
            sustain_level: 0.3,
            release_sec: 0.5,
        }
    } else {
        ToneAdsr {
            attack_sec: 0.005,
            decay_sec: 0.0,
            sustain_level: 1.0,
            release_sec: 0.5,
        }
    };
    if family == "landscape_density" || family == "landscape_peaks" {
        let mut control = VoiceControl::default();
        control.body.method = if kind == BodyKind::Harmonic {
            BodyMethod::Harmonic
        } else {
            BodyMethod::Modal
        };
        control.body.modes = Some(
            if family == "landscape_peaks" {
                ModePattern::landscape_peaks_modes()
            } else {
                ModePattern::landscape_density_modes()
            }
            .with_count(8),
        );
        control.pitch.freq = hz;
        let voice = VoiceSpec {
            control,
            articulation: ArticulationCoreConfig::default(),
        }
        .spawn_with_landscape(700, 0, VoiceMetadata::default(), FS, Some(environment), 17);
        return voice.representative_body_recipe(FS, 48_000);
    }
    Recipe {
        renderer_phase3: false,
        body,
        freq_hz: hz,
        hold: 48_000,
        adsr: Some(adsr),
        smoothing_tau_sec: 0.0,
        fs: FS,
        modulator: if family == "drone_sway" {
            RenderModulatorSpec::DroneSway {
                phase: 0.3,
                sway_rate: 0.8,
            }
        } else {
            RenderModulatorSpec::SeqGate { duration_sec: 1.0 }
        },
    }
}

fn metrics(
    scores: &[f32],
    reference: &[f32],
    repr: &ConsonanceRepresentationParams,
) -> serde_json::Value {
    assert!(scores.iter().chain(reference).all(|x| x.is_finite()));
    let ds = scores
        .iter()
        .zip(reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let dl = scores
        .iter()
        .zip(reference)
        .map(|(a, b)| (repr.level(*a) - repr.level(*b)).abs())
        .fold(0.0_f32, f32::max);
    let spread = |values: &[f32]| {
        values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
            - values.iter().copied().fold(f32::INFINITY, f32::min)
    };
    let reference_tied = spread(reference) <= 1e-6;
    let shortcut_tied = spread(scores) <= 1e-6;
    let mut reversals = 0;
    for i in 0..scores.len() {
        for j in i + 1..scores.len() {
            if (reference[i] - reference[j]).abs() >= 0.1
                && (reference[i] - reference[j]).signum() != (scores[i] - scores[j]).signum()
            {
                reversals += 1;
            }
        }
    }
    json!({"reference_tied":reference_tied,"shortcut_tied":shortcut_tied,
           "reference_spread":spread(reference),"shortcut_spread":spread(scores),"max_score_error":ds,"max_level_error":dl,"rank_reversals_at_ref_gap_0p1":reversals,
           "registered_movement_gate_pass":ds<=0.025 && dl<=0.0125 && reversals==0})
}

fn viability_voice(hz: f32, kind: PitchCoreKind) -> crate::life::voice::Voice {
    let mut control = VoiceControl::default();
    control.body.method = BodyMethod::Sine;
    control.pitch.freq = hz;
    control.pitch.core_kind = kind;
    control.pitch.leave_self_out = false;
    let lifecycle = crate::scenario::lifecycle::LifecycleConfig::Sustain {
        endurance_sec: Some(10.0),
        recovery_sec: Some(1.0),
        attack_cost_fraction: Some(0.0),
        attack_recharge_fraction: Some(0.0),
        continuous_recharge_score_low: Some(-1.0),
        continuous_recharge_score_high: Some(1.0),
        selection_environment: true,
        dissonance_penalty: 0.0,
        envelope: crate::scenario::EnvelopeConfig::default(),
    };
    let mut voice = VoiceSpec {
        control,
        articulation: ArticulationCoreConfig::Entrain {
            lifecycle,
            rhythm_freq: None,
            rhythm_coupling: crate::scenario::RhythmCouplingMode::TemporalOnly,
            rhythm_reward: None,
            breath_gain_init: Some(1.0),
        },
    }
    .spawn_with_landscape(2, 0, VoiceMetadata::default(), FS, None, 17);
    if let crate::life::voice::AnyArticulationCore::Entrain(core) = &mut voice.articulation.core {
        core.energy = 0.5;
    }
    voice
}

#[test]
fn environment_viability_consumes_pcm_observation_without_pitch_self_exclusion() {
    let mut landscape = Landscape::new(Log2Space::new(110.0, 880.0, 96));
    landscape.subjective_intensity[landscape.space.nearest_index(220.0)] = 1.0;
    landscape.loudness_exp = 0.23;
    let score = recompute_loo_score_scan(&landscape, &landscape.subjective_intensity, &mut None);
    landscape.consonance_field_score_eff = score;
    for kind in [PitchCoreKind::HillClimb, PitchCoreKind::PeakSampler] {
        let mut voice = viability_voice(220.0, kind);
        // An isolated caller without a snapshot still reads the observed field.
        assert_eq!(
            voice.pitch_ctl.environment_score(&landscape, 220.0),
            landscape.evaluate_pitch_score(220.0)
        );
        voice.last_signal = ArticulationSignal {
            amplitude: 1.0,
            is_active: true,
            ..Default::default()
        };
        landscape.loo = vec![crate::core::stream::loo::Observation {
            id: voice.id(),
            generation: voice.metadata.generation,
            h_scan: vec![0.; landscape.space.n_bins()],
            r_scan: vec![0.; landscape.space.n_bins()],
            score_scan: vec![0.; landscape.space.n_bins()].into(),
        }];
        let mut community = crate::life::community::Community::new(Timebase { fs: FS, hop: 512 });
        community.add_voice(voice);
        community.advance(512, 0, 512.0 / FS, &landscape);
        let voice = &mut community.voices[0];
        assert!(!voice.effective_control.pitch.leave_self_out);
        let input = voice.pitch_ctl.loo_input_mut();
        assert!(input.exact_score.is_some());
        assert!(voice.pitch_ctl.environment_score(&landscape, 220.0).abs() <= 1e-6);
    }
}

#[test]
fn rendered_loo_conformance() {
    crate::life::modal::register_modal();
    let mut config = AppConfig::default();
    config.psychoacoustics.habituation.enabled = false;
    let (params, kernel, hop) = crate::runtime::birth_surrogate_assay_core(&config, FS as u32);
    let template = AnalysisStream::new(params.clone(), kernel.clone());
    let space = template.last().space.clone();
    let (_, du) = erb_grid(&space);
    let repr = params.consonance_representation;
    let timebase = Timebase { fs: FS, hop };
    let families = [
        "sine",
        "harmonic_dark",
        "harmonic_bright",
        "harmonic_inharmonic",
        "modal_dark",
        "modal_bright",
        "harmonic_spread",
        "modal_spread",
        "motion",
        "adsr_slow",
        "drone_sway",
        "landscape_density",
        "landscape_peaks",
        "harmonic_default",
        "modal_default",
    ];
    assert_eq!(hop, 512);
    assert!(!params.habituation.enabled);
    let pilot = std::env::var_os("T32_PILOT").is_some();
    let motion_only = std::env::var_os("T33_MOTION_ONLY").is_some();
    let ties_only = std::env::var_os("T33_TIES_ONLY").is_some();
    assert!(u8::from(pilot) + u8::from(motion_only) + u8::from(ties_only) <= 1);
    let mut total = 0;
    let mut viability_failures = Vec::new();
    let mut family_pass: BTreeMap<String, [bool; 3]> = BTreeMap::new();
    let mut density_cache = BTreeMap::new();
    for base_hz in [440.0_f32, 1800.0] {
        for other_hz in [None, Some(base_hz), Some(466.0_f32)] {
            if ties_only && other_hz.is_some() {
                continue;
            }
            let others: Vec<_> = other_hz
                .map(|hz| batch(3, &recipe("sine", hz, template.last())))
                .into_iter()
                .collect();
            let mut other_renderer = ScheduleRenderer::new(timebase);
            let mut other_stream = template.clone();
            let mut other_pcm = Vec::with_capacity(ENV_FRAMES);
            for frame in 0..ENV_FRAMES {
                let pcm = other_renderer
                    .render(
                        if frame == 0 { &others } else { &[] },
                        (frame * hop) as u64,
                        &NeuralRhythms::default(),
                    )
                    .habitat
                    .to_vec();
                let mut landscape = other_stream.process(&pcm).clone();
                landscape.recompute_consonance(&params);
                other_pcm.push(pcm);
            }
            let mut reference = other_stream.last().clone();
            reference.recompute_consonance(&params);
            for family in families {
                if motion_only && family != "motion" {
                    continue;
                }
                if pilot && (family != "sine" || base_hz != 440.0 || other_hz != Some(466.0)) {
                    continue;
                }
                if base_hz != 440.0 && (family == "harmonic_default" || family == "modal_default") {
                    continue;
                }
                let own_recipe = recipe(family, base_hz, &reference);
                let own = [batch(2, &own_recipe)];
                let mut all = own.to_vec();
                all.extend(others.iter().cloned());
                let mut mixed_renderer = ScheduleRenderer::new(timebase);
                let mut own_renderer = ScheduleRenderer::new(timebase);
                let mut candidate = crate::core::stream::loo::Observer::new(2, 0, &template);
                let mut mixed_stream = template.clone();
                let mut removed_stream = template.clone();
                let mut oracle_stream = template.clone();
                let mut max_pcm_error = 0.0_f32;
                let mut max_l1 = 0.0_f32;
                let mut max_field_error = 0.0_f32;
                let mut audit: Option<[Vec<f32>; 4]> =
                    crate::core::auditory_shadow::selected_test_active()
                        .then(|| std::array::from_fn(|_| Vec::new()));
                for frame in 0..ENV_FRAMES {
                    let now = (frame * hop) as u64;
                    let full = mixed_renderer
                        .render(
                            if frame == 0 { &all } else { &[] },
                            now,
                            &NeuralRhythms::default(),
                        )
                        .habitat
                        .to_vec();
                    let own_pcm = own_renderer
                        .render(
                            if frame == 0 { &own } else { &[] },
                            now,
                            &NeuralRhythms::default(),
                        )
                        .habitat
                        .to_vec();
                    let removed: Vec<_> = full
                        .iter()
                        .zip(&own_pcm)
                        .zip(&other_pcm[frame])
                        .map(|((&a, &b), &c)| {
                            let value = a - b;
                            max_pcm_error = max_pcm_error.max((value - c).abs());
                            let tolerance = 32.0
                                * f32::EPSILON
                                * 1.0_f32.max(a.abs()).max(b.abs()).max(c.abs());
                            assert!(
                                (value - c).abs() <= tolerance,
                                "PCM {family} {base_hz} {other_hz:?} {frame}"
                            );
                            value
                        })
                        .collect();
                    if let Some(history) = &mut audit {
                        for (target, pcm) in history.iter_mut().zip([
                            &full[..],
                            &own_pcm[..],
                            &removed[..],
                            &other_pcm[frame][..],
                        ]) {
                            target.extend_from_slice(pcm);
                        }
                    }
                    candidate.process(&full, &own_pcm);
                    let mut mix = mixed_stream.process(&full).clone();
                    mix.recompute_consonance(&params);
                    let mut actual = removed_stream.process(&removed).clone();
                    actual.recompute_consonance(&params);
                    let mut oracle = oracle_stream.process(&other_pcm[frame]).clone();
                    oracle.recompute_consonance(&params);
                    let mass = oracle
                        .subjective_intensity
                        .iter()
                        .zip(&du)
                        .map(|(q, w)| q * w)
                        .sum::<f32>();
                    let l1 = actual
                        .subjective_intensity
                        .iter()
                        .zip(&oracle.subjective_intensity)
                        .zip(&du)
                        .map(|((a, b), w)| (a - b).abs() * w)
                        .sum::<f32>();
                    max_l1 = max_l1.max(l1 / mass.max(1e-12));
                    let mut frame_field_error = 0.0_f32;
                    for (a, b) in [
                        (&actual.harmonicity, &oracle.harmonicity),
                        (&actual.roughness, &oracle.roughness),
                        (&actual.roughness_shape_raw, &oracle.roughness_shape_raw),
                        (
                            &actual.consonance_field_score,
                            &oracle.consonance_field_score,
                        ),
                    ] {
                        let error = a
                            .iter()
                            .zip(b)
                            .map(|(a, b)| (a - b).abs())
                            .fold(0.0_f32, f32::max);
                        max_field_error = max_field_error.max(error);
                        frame_field_error = frame_field_error.max(error);
                    }
                    if (l1 / mass.max(1e-12) > 1e-3 || frame_field_error > 1e-3)
                        && let Some(history) = &audit
                    {
                        crate::core::auditory_shadow::save_v3_witness(
                            json!({"gate":"B-1 PCM oracle","family":family,"base_hz":base_hz,
                            "other_hz":other_hz,"frame":frame,"fs":FS,"hop":hop,"density_l1":l1/mass.max(1e-12),
                            "field_error":frame_field_error,"tolerance":1e-3,
                            "actual_shadow":removed_stream.aud_m1_shadow_diagnostic(),"oracle_shadow":oracle_stream.aud_m1_shadow_diagnostic()}),
                            history,
                        );
                    }
                    if other_hz.is_none() {
                        assert!(removed.iter().all(|x| *x == 0.0));
                        assert!(actual.subjective_intensity.iter().all(|x| *x == 0.0));
                        assert!(actual.harmonicity.iter().all(|x| *x == 0.0));
                        assert!(actual.roughness.iter().all(|x| *x == 0.0));
                    }
                }
                let mut mix = mixed_stream.last().clone();
                mix.recompute_consonance(&params);
                let mut removed = removed_stream.last().clone();
                removed.recompute_consonance(&params);
                assert!(max_l1 <= 1e-3 && max_field_error <= 1e-3, "PCM oracle gate");
                let current = base_hz.log2();
                let ratios = own_recipe.body.ratios.as_deref();
                let own_power = projected_power(&own_recipe, &mix);
                let mut other_power = vec![0_f64; own_power.len()];
                if let Some(hz) = other_hz {
                    let projected = projected_power(&recipe("sine", hz, &reference), &mix);
                    for (other, value) in other_power.iter_mut().zip(projected) {
                        *other = value as f64;
                    }
                }
                let mut input = LooInput {
                    tonal_context: None,
                    exact_score: Some(candidate.snapshot().score_scan),
                    own_power_scan: own_power,
                    other_power_scan: other_power,
                    computed_score: None,
                };
                let candidate_scan = loo_score_scan(&mix, &mut input);
                let expected_score =
                    sample_ownership_score(&mix, current, candidate_scan.as_deref());
                let old_score = legacy_sample_consonance_score_with_loo(
                    current,
                    current,
                    &mix,
                    true,
                    ReferenceLooMode::ApproxHarmonics,
                    1,
                    None,
                    ratios,
                );
                let reference_score = removed.evaluate_pitch_score_log2(current);
                for kind in [PitchCoreKind::HillClimb, PitchCoreKind::PeakSampler] {
                    let mut voice = viability_voice(base_hz, kind);
                    *voice.pitch_ctl.loo_input_mut() = LooInput {
                        computed_score: None,
                        ..input.clone()
                    };
                    let score = voice.pitch_ctl.environment_score(&mix, base_hz);
                    assert_eq!(score, expected_score);
                    if (score - reference_score).abs() - (old_score - reference_score).abs() > 1e-6
                    {
                        if let Some(history) = &audit {
                            crate::core::auditory_shadow::save_v3_witness(
                                json!({"gate":"B-1 environment","family":family,"base_hz":base_hz,"other_hz":other_hz,
                                "fs":FS,"hop":hop,"score":score,"reference":reference_score,"legacy":old_score,
                                "actual_shadow":removed_stream.aud_m1_shadow_diagnostic(),"oracle_shadow":oracle_stream.aud_m1_shadow_diagnostic()}),
                                history,
                            );
                        }
                        viability_failures.push(json!({"family":family,"base_hz":base_hz,"other_hz":other_hz,"kind":format!("{kind:?}"),"path":"environment"}));
                    }
                    let mut expected = voice.articulation.clone();
                    expected.process(
                        mix.evaluate_pitch_level(base_hz),
                        expected_score,
                        &NeuralRhythms::default(),
                        0.1,
                        0.0,
                    );
                    voice.tick_articulation_lifecycle(0.1, &NeuralRhythms::default(), &mix, 0.0);
                    match (&voice.articulation.core, &expected.core) {
                        (
                            crate::life::voice::AnyArticulationCore::Entrain(actual),
                            crate::life::voice::AnyArticulationCore::Entrain(expected),
                        ) => assert_eq!(actual.energy, expected.energy),
                        _ => unreachable!(),
                    }
                }
                let mut legacy_errors = [0.0_f32; 4];
                let old_exact_scan = exact_loo_consonance_score_scan(
                    &mix,
                    current,
                    true,
                    ReferenceLooMode::ExactScan,
                    &mut None,
                );
                for (method_index, method) in ["approx", "exact", "t31"].into_iter().enumerate() {
                    let score = |pitch| match method {
                        "approx" => legacy_sample_consonance_score_with_loo(
                            pitch,
                            current,
                            &mix,
                            true,
                            ReferenceLooMode::ApproxHarmonics,
                            1,
                            None,
                            ratios,
                        ),
                        "exact" => sample_consonance_score_with_loo(
                            pitch,
                            current,
                            &mix,
                            true,
                            ReferenceLooMode::ExactScan,
                            1,
                            old_exact_scan.as_deref(),
                            ratios,
                        ),
                        "t31" => sample_consonance_score_with_loo(
                            pitch,
                            current,
                            &mix,
                            true,
                            ReferenceLooMode::ApproxHarmonics,
                            1,
                            candidate_scan.as_deref(),
                            ratios,
                        ),
                        _ => unreachable!(),
                    };
                    let shortcut_scan: Vec<_> =
                        space.centers_log2.iter().map(|&x| score(x)).collect();
                    let pitches: Vec<_> = CENTS.iter().map(|&c| current + c / 1200.0).collect();
                    let mut point_shortcut = Vec::new();
                    let mut point_reference = Vec::new();
                    let mut body_shortcut = Vec::new();
                    let mut body_reference = Vec::new();
                    let mut representative = SynchronousAnalysis::new(kernel.clone(), &params);
                    for &pitch in &pitches {
                        point_shortcut.push(score(pitch));
                        point_reference.push(removed.evaluate_pitch_score_log2(pitch));
                        let key = (
                            Identity::new(2, 0, &own_recipe).recipe_hash,
                            pitch.exp2().to_bits(),
                        );
                        let density = density_cache.entry(key).or_insert_with(|| {
                            representative_density(
                                &own_recipe,
                                pitch.exp2(),
                                &mut representative,
                                &params,
                            )
                        });
                        body_shortcut.push(
                            body_fitness::evaluate(&space, density, &du, &shortcut_scan, &repr)
                                .unwrap()
                                .score,
                        );
                        body_reference.push(
                            body_fitness::evaluate(
                                &space,
                                density,
                                &du,
                                &removed.consonance_field_score_eff,
                                &repr,
                            )
                            .unwrap()
                            .score,
                        );
                    }
                    let mut final_decisions = Vec::new();
                    let mut native_decisions = Vec::new();
                    let native_shortcut_bounds = Cell::new((f32::INFINITY, f32::NEG_INFINITY));
                    let native_reference_bounds = Cell::new((f32::INFINITY, f32::NEG_INFINITY));
                    for temperature in [0.0_f32, 0.05, 0.2] {
                        let mut hill = PitchHillClimbPitchCore::new(24.0, current, 0.0);
                        hill.set_temperature(temperature);
                        let count = if temperature == 0.0 { 1 } else { 64 };
                        for ordinal in 0..count {
                            let seed = 0xF17E0001_u64 + ordinal;
                            let mut rng = SmallRng::seed_from_u64(seed);
                            let point = final_scores(&hill, &pitches, &point_shortcut, &mut rng);
                            let mut rng = SmallRng::seed_from_u64(seed);
                            let point_ref =
                                final_scores(&hill, &pitches, &point_reference, &mut rng);
                            let mut rng = SmallRng::seed_from_u64(seed);
                            let body = final_scores(&hill, &pitches, &body_shortcut, &mut rng);
                            let mut rng = SmallRng::seed_from_u64(seed);
                            let body_ref = final_scores(&hill, &pitches, &body_reference, &mut rng);
                            final_decisions.push(json!({"temperature":temperature,"seed":seed,
                            "point_shortcut_target_bits":point.target_pitch_log2.to_bits(),"point_pcm_target_bits":point_ref.target_pitch_log2.to_bits(),
                            "body_shortcut_target_bits":body.target_pitch_log2.to_bits(),"body_pcm_target_bits":body_ref.target_pitch_log2.to_bits()}));
                            let mut rng = SmallRng::seed_from_u64(seed);
                            let native = hill.propose_with_scorer(
                                current,
                                current,
                                &mix,
                                &[],
                                &mut rng,
                                |x| {
                                    let value = score(x);
                                    assert!(value.is_finite());
                                    let (lo, hi) = native_shortcut_bounds.get();
                                    native_shortcut_bounds.set((lo.min(value), hi.max(value)));
                                    value
                                },
                            );
                            let mut rng = SmallRng::seed_from_u64(seed);
                            let native_ref = hill.propose_with_scorer(
                                current,
                                current,
                                &removed,
                                &[],
                                &mut rng,
                                |x| {
                                    let value = removed.evaluate_pitch_score_log2(x);
                                    assert!(value.is_finite());
                                    let (lo, hi) = native_reference_bounds.get();
                                    native_reference_bounds.set((lo.min(value), hi.max(value)));
                                    value
                                },
                            );
                            native_decisions.push(json!({"temperature":temperature,"seed":seed,
                            "shortcut_target_bits":native.target_pitch_log2.to_bits(),"pcm_target_bits":native_ref.target_pitch_log2.to_bits()}));
                        }
                    }
                    let (shortcut_min, shortcut_max) = native_shortcut_bounds.get();
                    let (reference_min, reference_max) = native_reference_bounds.get();
                    let native_reference_tied = reference_max - reference_min <= 1e-6;
                    let native_shortcut_tied = shortcut_max - shortcut_min <= 1e-6;
                    let main_level = mix.evaluate_pitch_level(base_hz);
                    let point_shortcut_level = repr.level(point_shortcut[3]);
                    let point_reference_level = repr.level(point_reference[3]);
                    let body_shortcut_level = repr.level(body_shortcut[3]);
                    let body_reference_level = repr.level(body_reference[3]);
                    let errors = [
                        (point_shortcut[3] - point_reference[3]).abs(),
                        (point_shortcut_level - point_reference_level).abs(),
                        (body_shortcut[3] - body_reference[3]).abs(),
                        (body_shortcut_level - body_reference_level).abs(),
                    ];
                    if method == "approx" {
                        legacy_errors = errors;
                    }
                    if method == "t31"
                        && !errors
                            .iter()
                            .zip(legacy_errors)
                            .all(|(new, old)| new - old <= 1e-6)
                    {
                        if let Some(history) = &audit {
                            crate::core::auditory_shadow::save_v3_witness(
                                json!({"gate":"B-1 point/body environment","family":family,"base_hz":base_hz,"other_hz":other_hz,
                                "fs":FS,"hop":hop,"new_errors":errors,"legacy_errors":legacy_errors,
                                "actual_shadow":removed_stream.aud_m1_shadow_diagnostic(),"oracle_shadow":oracle_stream.aud_m1_shadow_diagnostic()}),
                                history,
                            );
                        }
                        viability_failures.push(json!({"family":family,"base_hz":base_hz,"other_hz":other_hz,"path":"point_body","new_errors":errors,"legacy_errors":legacy_errors}));
                    }
                    println!(
                        "T32_CASE {}",
                        json!({"method":method,"family":family,"base_hz":base_hz,"other_hz":other_hz,"recipe_body":format!("{:?}",own_recipe.body),
                    "ownership_bins":if family == "motion" {
                        Some((0..space.n_bins()).filter(|&i| mix.subjective_intensity[i] > 0.0 || removed.subjective_intensity[i] > 0.0)
                            .map(|i| json!([i,space.centers_hz[i],input.own_power_scan[i],(input.own_power_scan[i] as f64 + input.other_power_scan[i]),mix.subjective_intensity[i],removed.subjective_intensity[i]]))
                            .collect::<Vec<_>>())
                    } else { None },
                    "source_id":2,"environment_frames":ENV_FRAMES,"representative_frames":BODY_FRAMES,"cents":CENTS,
                    "oracle":{"pcm_max_abs":max_pcm_error,"density_max_normalized_l1":max_l1,"field_max_abs":max_field_error,
                        "registered_f1_pass":max_l1<=1e-3 && max_field_error<=1e-3},
                    "point_shortcut_scores":point_shortcut,"point_pcm_scores":point_reference,
                    "body_shortcut_scores":body_shortcut,"body_pcm_scores":body_reference,
                    "point_metrics":metrics(&point_shortcut,&point_reference,&repr),"body_metrics":metrics(&body_shortcut,&body_reference,&repr),
                    "final_decisions":final_decisions,"native_point_decisions":native_decisions,
                    "native_ties":{"reference_tied":native_reference_tied,"shortcut_tied":native_shortcut_tied,
                        "reference_min":reference_min,"reference_max":reference_max,
                        "shortcut_min":shortcut_min,"shortcut_max":shortcut_max},
                    "metabolism":{"main_mix_level":main_level,"point_shortcut_score":point_shortcut[3],"point_pcm_score":point_reference[3],
                        "body_shortcut_score":body_shortcut[3],"body_pcm_score":body_reference[3],
                        "point_shortcut_level":point_shortcut_level,"point_pcm_level":point_reference_level,
                        "body_shortcut_level":body_shortcut_level,"body_pcm_level":body_reference_level,
                        "point_shortcut_minus_point_pcm_level":point_shortcut_level-point_reference_level,
                        "body_shortcut_minus_body_pcm_level":body_shortcut_level-body_reference_level,
                        "main_mix_minus_body_pcm_level":main_level-body_reference_level}})
                    );
                    let point_metrics = metrics(&point_shortcut, &point_reference, &repr);
                    let body_metrics = metrics(&body_shortcut, &body_reference, &repr);
                    let point_selection =
                        if point_metrics["reference_tied"] == true && native_reference_tied {
                            point_metrics["shortcut_tied"] == true && native_shortcut_tied
                        } else {
                            final_decisions.iter().all(|d| {
                                d["point_shortcut_target_bits"] == d["point_pcm_target_bits"]
                            }) && native_decisions
                                .iter()
                                .all(|d| d["shortcut_target_bits"] == d["pcm_target_bits"])
                        };
                    let body_selection = if body_metrics["reference_tied"] == true {
                        body_metrics["shortcut_tied"] == true
                    } else {
                        final_decisions
                            .iter()
                            .all(|d| d["body_shortcut_target_bits"] == d["body_pcm_target_bits"])
                    };
                    let method_pass = point_metrics["registered_movement_gate_pass"] == true
                        && body_metrics["registered_movement_gate_pass"] == true
                        && point_selection
                        && body_selection;
                    family_pass.entry(family.to_string()).or_insert([true; 3])[method_index] &=
                        method_pass;
                    if method == "t31"
                        && !method_pass
                        && let Some(history) = &audit
                    {
                        crate::core::auditory_shadow::save_v3_witness(
                            json!({"gate":"B-1 consumer","family":family,"base_hz":base_hz,"other_hz":other_hz,
                            "fs":FS,"hop":hop,"point":point_metrics,"body":body_metrics,"point_selection":point_selection,"body_selection":body_selection,
                            "actual_shadow":removed_stream.aud_m1_shadow_diagnostic(),"oracle_shadow":oracle_stream.aud_m1_shadow_diagnostic()}),
                            history,
                        );
                    }
                }
                total += 1;
            }
        }
    }
    assert_eq!(
        total,
        if pilot {
            1
        } else if motion_only {
            6
        } else if ties_only {
            28
        } else {
            84
        }
    );
    println!(
        "T32_COMPLETE {}",
        json!({"cases":total,"fs":FS,"hop":hop,"families":families,"family_pass":family_pass,"habituation_enabled":false,"distinct_body_densities":density_cache.len(),"viability_failures":viability_failures})
    );
    assert!(
        viability_failures.is_empty(),
        "viability counterexamples: {viability_failures:?}"
    );
    if !pilot {
        let unsupported: Vec<_> = family_pass
            .iter()
            .filter(|(_, methods)| !methods[2])
            .map(|(family, _)| family.as_str())
            .collect();
        assert!(
            unsupported.is_empty(),
            "T-31 candidate does not conform for: {unsupported:?}"
        );
    }
}

#[test]
#[ignore = "AUD-M1 selected front-end B-1 gate"]
fn aud_m1_v3_b1() {
    crate::core::auditory_shadow::with_selected_test_frontend(rendered_loo_conformance);
}

fn final_scores(
    hill: &PitchHillClimbPitchCore,
    pitches: &[f32],
    scores: &[f32],
    rng: &mut SmallRng,
) -> TargetProposal {
    let mut scored = pitches
        .iter()
        .copied()
        .zip(scores.iter().copied())
        .collect();
    hill.select_scored(pitches[3], scores[3], &mut scored, rng, None)
}

fn projected_power(recipe: &Recipe, environment: &Landscape) -> Vec<f32> {
    let mut control = VoiceControl::default();
    control.pitch.freq = recipe.freq_hz;
    control.body.amp = 0.35 * recipe.body.amp_scale;
    control.body.method = match recipe.body.kind {
        BodyKind::Sine => BodyMethod::Sine,
        BodyKind::Harmonic => BodyMethod::Harmonic,
        BodyKind::Modal => BodyMethod::Modal,
    };
    control.body.timbre.brightness = recipe.body.brightness;
    control.body.timbre.inharmonic = recipe.body.inharmonic;
    control.body.timbre.spread = recipe.body.spread;
    control.body.timbre.unison = recipe.body.unison;
    control.body.timbre.motion = recipe.body.motion;
    control.body.modes = recipe
        .body
        .ratios
        .as_ref()
        .map(|ratios| ModePattern::custom_modes(ratios.to_vec()));
    let mut voice = VoiceSpec {
        control,
        articulation: ArticulationCoreConfig::default(),
    }
    .spawn_with_landscape(2, 0, VoiceMetadata::default(), FS, Some(environment), 17);
    // B-1 fixes a representative onset, not a live articulation trajectory.
    // Keep the same unit active signal for both sources and call the production projection.
    voice.last_signal = ArticulationSignal {
        amplitude: 1.0,
        is_active: true,
        ..Default::default()
    };
    voice.project_loo_power(&environment.space).to_vec()
}

fn representative_density(
    recipe: &Recipe,
    hz: f32,
    analysis: &mut SynchronousAnalysis,
    params: &crate::core::landscape::LandscapeParams,
) -> Vec<f32> {
    use crate::life::schedule_renderer::modal_phase_seed;
    use crate::life::sound::Tone;
    let hop = 512;
    let mut tone = Tone::from_parts(
        Timebase { fs: FS, hop },
        0,
        recipe.hold,
        hz,
        1.0,
        Some(recipe.body.clone()),
        Some(recipe.modulator.clone()),
        recipe.adsr,
    )
    .expect("B-1 representative Tone");
    tone.set_smoothing_tau_sec(recipe.smoothing_tau_sec);
    tone.seed_modal_phases(modal_phase_seed(2, 0, 0));
    let kick = OnsetKick { strength: 1.0 };
    tone.schedule_planned_kick(kick);
    tone.arm_onset_trigger(kick.strength);
    analysis.reset_density();
    let mut rhythms = NeuralRhythms::default();
    let mut pcm = vec![0.0; hop];
    let mut density = vec![0.0; analysis.landscape.space.n_bins()];
    for frame in 0..BODY_FRAMES {
        let now = (frame * hop) as u64;
        tone.kick_planned_if_due(now);
        tone.render_block(now, FS, 1.0 / FS, &mut rhythms, &mut pcm);
        let spectral = analysis.density(&pcm, params);
        for (sum, &value) in density.iter_mut().zip(spectral.subjective_intensity) {
            *sum += value;
        }
    }
    for value in &mut density {
        *value *= 1.0 / BODY_FRAMES as f32;
    }
    density
}

#[test]
fn flora_all_consumers_conform() {
    flora_ownership_cases(true, false);
    flora_ownership_cases(false, false);
}

#[test]
#[ignore = "AUD-M1 selected front-end Flora gate"]
fn aud_m1_v3_flora() {
    crate::core::auditory_shadow::with_selected_test_frontend(flora_all_consumers_conform);
}

#[test]
fn flora_active_timing_consumers_conform() {
    flora_ownership_cases(true, true);
}

fn flora_ownership_cases(supported: bool, active_timing: bool) {
    use crate::life::community::Community;
    use crate::life::flora::Flora;
    use crate::scenario::control::Routing;
    use crate::scenario::{FloraPatch, FloraSeeding, FloraSpec, FloraTone};

    let mut config = AppConfig::default();
    config.psychoacoustics.habituation.enabled = false;
    let (params, kernel, hop) = crate::runtime::birth_surrogate_assay_core(&config, FS as u32);
    let template = AnalysisStream::new(params.clone(), kernel.clone());
    let timebase = Timebase { fs: FS, hop };
    let mut failed_cases = Vec::new();
    let mut records = Vec::new();
    let mut density_cache = BTreeMap::new();
    for (tone, residual, habitat) in [
        (FloraTone::Sine, false, true),
        (FloraTone::Resonant, true, true),
        (FloraTone::Sine, true, true),
        (FloraTone::Sine, true, false),
    ] {
        if (!residual || !habitat) != supported || (active_timing && (residual || !habitat)) {
            continue;
        }
        let mut terrain = template.last().clone();
        terrain.consonance_field_level.fill(0.5);
        terrain.harmonicity01.fill(0.0);
        let peak = terrain
            .space
            .nearest_index(if residual { 466.0 } else { 440.0 });
        terrain.consonance_density_mass.fill(0.0);
        for (i, value) in terrain.consonance_density_mass.iter_mut().enumerate() {
            *value = crate::core::float::unit_gaussian(
                i as f32 - peak as f32,
                if residual { 8.0 } else { 0.2 },
            );
        }
        let flora_spec = FloraSpec {
            pace: 1.0,
            leave_self_out: true,
            amp: 0.35,
            fade_sec: 0.,
            tone,
            residual,
            routing: Routing {
                to_habitat: habitat,
                to_presentation: !habitat,
            },
            ..Default::default()
        };
        let mut flora = [Flora::new(
            7,
            flora_spec,
            (330., 660.),
            FloraSeeding::Consonance,
            42,
            &terrain,
            FS,
            hop,
        )];
        let mut flora_only = flora[0].clone();
        let mut own_recipe = recipe("sine", 440., &terrain);
        own_recipe.hold = (hop * 300) as u64;
        own_recipe.modulator = RenderModulatorSpec::SeqGate { duration_sec: 10.0 };
        let own_batch = [batch(2, &own_recipe)];
        let mut mixed_renderer = ScheduleRenderer::new(timebase);
        let mut own_renderer = ScheduleRenderer::new(timebase);
        let mut candidate = crate::core::stream::loo::Observer::new(2, 0, &template);
        let mut mixed_stream = template.clone();
        let mut removed_stream = template.clone();
        let mut reference_stream = template.clone();
        let mut sources = [crate::life::schedule_renderer::SourcePcm {
            kind: crate::core::stream::loo::SourceKind::Voice,
            analyze_hr: true,
            tonal_context: None,
            id: 2,
            generation: 0,
            habitat: vec![0.; hop],
        }];
        let mut solo_p = vec![0.; hop];
        let mut solo_h = vec![0.; hop];
        let mut audit: Option<[Vec<f32>; 4]> = crate::core::auditory_shadow::selected_test_active()
            .then(|| std::array::from_fn(|_| Vec::new()));
        let mut crossing_births = 0;
        for frame in 0..256 {
            if active_timing {
                let now = (frame * hop) as u64;
                let observation = crate::life::flora::TemporalObservation {
                    known: true,
                    observed_end: Some(now),
                    state: crate::core::meter::MeterState {
                        beat: crate::core::meter::MeterBand {
                            phase: (now as f32 / FS * 2.0).fract(),
                            freq_hz: 2.0,
                            confidence: 1.0,
                            amplitude: 1.0,
                        },
                        ..Default::default()
                    },
                    ..Default::default()
                };
                flora[0].observe_rhythm(observation, now, Default::default());
                flora_only.observe_rhythm(observation, now, Default::default());
            }
            if frame == 64 || frame == 128 {
                let patch = FloraPatch {
                    amp: Some(if frame == 64 { 0. } else { flora_spec.amp }),
                    ..Default::default()
                };
                flora[0].apply_patch(&patch);
                flora_only.apply_patch(&patch);
            }
            if frame == 192 {
                flora[0].release(0.2);
                flora_only.release(0.2);
            }
            flora[0].advance(&terrain);
            flora_only.advance(&terrain);
            let commands = if frame == 0 { &own_batch[..] } else { &[] };
            let now = (frame * hop) as u64;
            let full = mixed_renderer
                .render_with_source_pcm(
                    commands,
                    now,
                    &NeuralRhythms::default(),
                    &mut sources,
                    &mut flora,
                    &mut [],
                    |_, _, _, _| {},
                )
                .habitat
                .to_vec();
            let own = own_renderer
                .render(commands, now, &NeuralRhythms::default())
                .habitat;
            assert_eq!(sources[0].habitat, own);
            solo_p.fill(0.);
            solo_h.fill(0.);
            flora_only.render_add(&mut solo_p, &mut solo_h);
            if active_timing {
                let report =
                    serde_json::to_value(flora[0].report(now as f64 / FS as f64, &terrain))
                        .unwrap();
                assert_eq!(report["temporal"]["active"], true);
                for event in report["temporal"]["events"].as_array().unwrap() {
                    if event["action"] == "born" {
                        assert_eq!(event["scheduled_tick"], event["tick"]);
                        crossing_births += 1;
                    }
                }
            }
            let removed: Vec<_> = full
                .iter()
                .zip(own)
                .zip(&solo_h)
                .map(|((&a, &b), &c)| {
                    let got = a - b;
                    assert!(
                        (got - c).abs()
                            <= 32. * f32::EPSILON * 1_f32.max(a.abs()).max(b.abs()).max(c.abs())
                    );
                    got
                })
                .collect();
            if let Some(history) = &mut audit {
                for (target, pcm) in
                    history
                        .iter_mut()
                        .zip([&full[..], own, &removed[..], &solo_h[..]])
                {
                    target.extend_from_slice(pcm);
                }
            }
            candidate.process(&full, &sources[0].habitat);
            let mut mix = mixed_stream.process(&full).clone();
            mix.recompute_consonance(&params);
            let mut actual = removed_stream.process(&removed).clone();
            actual.recompute_consonance(&params);
            let mut reference = reference_stream.process(&solo_h).clone();
            reference.recompute_consonance(&params);
            for (bin, (a, b)) in actual
                .consonance_field_score
                .iter()
                .zip(&reference.consonance_field_score)
                .enumerate()
            {
                if ((a - b).abs() > 1e-3 || !(a - b).is_finite())
                    && let Some(history) = &audit
                {
                    crate::core::auditory_shadow::save_v3_witness(
                        json!({"gate":"Flora PCM oracle","tone":format!("{tone:?}"),
                        "residual":residual,"habitat":habitat,"supported":supported,"frame":frame,"bin":bin,
                        "fs":FS,"hop":hop,"error":(a-b).abs(),"tolerance":1e-3,"hz":mix.space.centers_hz[bin],
                        "actual_c":actual.consonance_field_score,"oracle_c":reference.consonance_field_score,
                        "actual_h":actual.harmonicity,"oracle_h":reference.harmonicity,
                        "actual_r":actual.roughness01,"oracle_r":reference.roughness01,
                        "actual_shadow":removed_stream.aud_m1_shadow_diagnostic(),"oracle_shadow":reference_stream.aud_m1_shadow_diagnostic()}),
                        history,
                    );
                }
                assert!((a - b).abs() <= 1e-3, "existing PCM oracle gate");
            }
            if ![63, 127, 191, 199, 255].contains(&frame) {
                continue;
            }
            assert_eq!(
                solo_h.iter().any(|x| *x != 0.),
                habitat && [63, 191, 199].contains(&frame)
            );
            for kind in [PitchCoreKind::HillClimb, PitchCoreKind::PeakSampler] {
                let mut control = VoiceControl::default();
                control.body.amp = 0.35;
                control.pitch.freq = 440.;
                control.pitch.core_kind = kind;
                control.pitch.leave_self_out = true;
                let mut voice = VoiceSpec {
                    control,
                    articulation: ArticulationCoreConfig::default(),
                }
                .spawn_with_landscape(
                    2,
                    0,
                    VoiceMetadata::default(),
                    FS,
                    Some(&mix),
                    17,
                );
                voice.selection_environment = true;
                voice.last_signal = ArticulationSignal {
                    amplitude: 1.,
                    is_active: true,
                    ..Default::default()
                };
                let own_power = voice.project_loo_power(&mix.space).to_vec();
                let mut observation = candidate.snapshot();
                observation.id = voice.id();
                observation.generation = voice.metadata.generation;
                mix.loo = vec![observation];
                let mut pop = Community::new(timebase);
                pop.voices.push(voice);
                pop.flora.push(flora[0].clone());
                pop.advance(64, frame as u64, 1e-4, &mix);
                let input = pop.voices[0].pitch_ctl.loo_input_mut().clone();
                assert!(input.exact_score.is_some());
                {
                    let method = "pcm";
                    let mut input_copy = input.clone();
                    let scores = loo_score_scan(&mix, &mut input_copy);
                    let density = candidate.last().subjective_intensity.clone();
                    let pitches: Vec<_> =
                        CENTS.iter().map(|c| 440_f32.log2() + c / 1200.).collect();
                    let got: Vec<_> = pitches
                        .iter()
                        .map(|p| sample_ownership_score(&mix, *p, scores.as_deref()))
                        .collect();
                    let expected: Vec<_> = pitches
                        .iter()
                        .map(|p| actual.evaluate_pitch_score_log2(*p))
                        .collect();
                    let mut legacy_input = LooInput {
                        tonal_context: None,
                        exact_score: None,
                        own_power_scan: own_power.clone(),
                        other_power_scan: vec![0.; own_power.len()],
                        computed_score: None,
                    };
                    let legacy_scan = approx_loo_score_scan(&mix, &mut legacy_input, &mut None);
                    let legacy: Vec<_> = pitches
                        .iter()
                        .map(|p| sample_ownership_score(&mix, *p, legacy_scan.as_deref()))
                        .collect();
                    let repr = &params.consonance_representation;
                    let point_metrics = metrics(&got, &expected, repr);
                    let (_, du) = erb_grid(&mix.space);
                    let scan: Vec<_> = mix
                        .space
                        .centers_log2
                        .iter()
                        .map(|p| sample_ownership_score(&mix, *p, scores.as_deref()))
                        .collect();
                    let old_scan: Vec<_> = mix
                        .space
                        .centers_log2
                        .iter()
                        .map(|p| sample_ownership_score(&mix, *p, legacy_scan.as_deref()))
                        .collect();
                    let mut representative = SynchronousAnalysis::new(kernel.clone(), &params);
                    let mut body = Vec::new();
                    let mut body_ref = Vec::new();
                    let mut body_old = Vec::new();
                    for &pitch in &pitches {
                        let density = density_cache.entry(pitch.to_bits()).or_insert_with(|| {
                            representative_density(
                                &own_recipe,
                                pitch.exp2(),
                                &mut representative,
                                &params,
                            )
                        });
                        for (target, field) in [
                            (&mut body, &scan),
                            (&mut body_ref, &actual.consonance_field_score_eff),
                            (&mut body_old, &old_scan),
                        ] {
                            target.push(
                                body_fitness::evaluate(&mix.space, density, &du, field, repr)
                                    .unwrap()
                                    .score,
                            );
                        }
                    }
                    let body_metrics = metrics(&body, &body_ref, repr);
                    let full_metrics = metrics(&scan, &actual.consonance_field_score_eff, repr);
                    let mut decisions = Vec::new();
                    let mut selection_pass = true;
                    let mut perceptual =
                        AdaptationContext::from_config(&Default::default(), mix.space.n_bins());
                    perceptual.set_enabled(false);
                    let features = FeaturesNow {
                        density: vec![0.; mix.space.n_bins()],
                        mass: 0.,
                    };
                    for temperature in [0.0_f32, 0.05, 0.2] {
                        let mut hill = PitchHillClimbPitchCore::new(24.0, pitches[3], 0.0);
                        hill.set_temperature(temperature);
                        for ordinal in 0..if temperature == 0. { 1 } else { 64 } {
                            let seed = 0xF17E0001_u64 + ordinal;
                            for (label, values, reference_values, m) in [
                                ("point", &got, &expected, &point_metrics),
                                ("body", &body, &body_ref, &body_metrics),
                            ] {
                                let a = final_scores(
                                    &hill,
                                    &pitches,
                                    values,
                                    &mut SmallRng::seed_from_u64(seed),
                                );
                                let b = final_scores(
                                    &hill,
                                    &pitches,
                                    reference_values,
                                    &mut SmallRng::seed_from_u64(seed),
                                );
                                let pass = if m["reference_tied"] == true {
                                    m["shortcut_tied"] == true
                                } else {
                                    a.target_pitch_log2.to_bits() == b.target_pitch_log2.to_bits()
                                };
                                selection_pass &= pass;
                                if !pass {
                                    decisions.push(json!({"kind":label,"temperature":temperature,"seed":seed,"got":a.target_pitch_log2,"reference":b.target_pitch_log2}));
                                }
                            }
                            let mut native = pop.voices[0].pitch_core_for_test().clone();
                            *native.loo_input_mut() = input.clone();
                            native.set_temperature(temperature);
                            let mut reference_core = native.clone();
                            *reference_core.loo_input_mut() = LooInput::default();
                            let a = native.propose_target(
                                pitches[3],
                                pitches[3],
                                440.,
                                1.,
                                &mix,
                                &perceptual,
                                &features,
                                &[],
                                &mut SmallRng::seed_from_u64(seed),
                            );
                            let b = reference_core.propose_target(
                                pitches[3],
                                pitches[3],
                                440.,
                                1.,
                                &actual,
                                &perceptual,
                                &features,
                                &[],
                                &mut SmallRng::seed_from_u64(seed),
                            );
                            let pass = if full_metrics["reference_tied"] == true {
                                full_metrics["shortcut_tied"] == true
                            } else {
                                a.target_pitch_log2.to_bits() == b.target_pitch_log2.to_bits()
                            };
                            selection_pass &= pass;
                            if !pass {
                                decisions.push(json!({"kind":"native","temperature":temperature,"seed":seed,"got":a.target_pitch_log2,"reference":b.target_pitch_log2}));
                            }
                        }
                    }
                    let mut environment_deltas = Vec::new();
                    let mut environment_errors = Vec::new();
                    for (new, old, reference) in [
                        (got[3], legacy[3], expected[3]),
                        (body[3], body_old[3], body_ref[3]),
                    ] {
                        environment_errors.push((new - reference).abs());
                        environment_errors.push((repr.level(new) - repr.level(reference)).abs());
                        environment_deltas.push((new - reference).abs() - (old - reference).abs());
                        environment_deltas.push(
                            (repr.level(new) - repr.level(reference)).abs()
                                - (repr.level(old) - repr.level(reference)).abs(),
                        );
                    }
                    let environment_pass = environment_deltas.iter().all(|d| *d <= 1e-6);
                    // Density loss is diagnostic; acceptance follows the registered consumers.
                    let reference_mass: f32 = actual
                        .subjective_intensity
                        .iter()
                        .zip(&du)
                        .map(|(x, w)| x * w)
                        .sum();
                    let over_removed: f32 = actual
                        .subjective_intensity
                        .iter()
                        .zip(&density)
                        .zip(&du)
                        .map(|((&reference, &retained), &w)| (reference - retained).max(0.) * w)
                        .sum();
                    let pass = point_metrics["registered_movement_gate_pass"] == true
                        && body_metrics["registered_movement_gate_pass"] == true
                        && selection_pass
                        && environment_pass;
                    let record = json!({"active_timing":active_timing,"conformance_scope":"supported","method":method,"tone":format!("{tone:?}"),"residual":residual,"habitat":habitat,"frame":frame,"kind":format!("{kind:?}"),
                    "point_scores":got,"point_reference":expected,"body_scores":body,"body_reference":body_ref,"point":point_metrics,"body":body_metrics,"selection_pass":selection_pass,"environment_pass":environment_pass,"native_ties":full_metrics,
                    "environment_errors":environment_errors,"environment_error_order":["point_score","point_level","body_score","body_level"],"environment_error_deltas":environment_deltas,"decision_failures":decisions,"decision_failure_count":decisions.len(),"decision_failure_examples":decisions.iter().take(4).collect::<Vec<_>>(),"owned_bins":input.own_power_scan.iter().enumerate().filter(|(_,x)| **x>0.).map(|(i,x)| json!([i,x,mix.subjective_intensity[i],actual.subjective_intensity[i]])).collect::<Vec<_>>(),"density_over_removed":over_removed/reference_mass.max(1e-12),"pass":pass});
                    if !pass && let Some(history) = &audit {
                        crate::core::auditory_shadow::save_v3_witness(
                            json!({"gate":"Flora consumer","record":record,"fs":FS,"hop":hop,
                            "actual_shadow":removed_stream.aud_m1_shadow_diagnostic(),"oracle_shadow":reference_stream.aud_m1_shadow_diagnostic()}),
                            history,
                        );
                    }
                    println!("T35_CASE {record}");
                    records.push(record);
                    if !pass {
                        failed_cases.push((method, tone, residual, habitat, frame, kind));
                    }
                    let mut environment = pop.voices[0].pitch_core_for_test().clone();
                    *environment.loo_input_mut() = input.clone();
                    assert_eq!(environment.environment_score(&mix, 440.), got[3]);
                }
            }
        }
        if active_timing {
            assert!(
                crossing_births > 0,
                "active timing never allocated at a crossing"
            );
        }
    }
    let directory = std::env::var_os("FLORA_LOO_REPORT_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::env::temp_dir().join(format!("conchordal-flora-loo-{}", std::process::id()))
        });
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join(if active_timing {
        "active-timing.json"
    } else if supported {
        "supported.json"
    } else {
        "residual-supported.json"
    });
    std::fs::write(&path, serde_json::to_vec_pretty(&records).unwrap()).unwrap();
    println!("Flora LOO record: {}", path.display());
    assert!(
        failed_cases.is_empty(),
        "Flora consumer counterexamples: {failed_cases:?}"
    );
}
