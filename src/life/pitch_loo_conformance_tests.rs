//! B-1 PCM self-removal reference, ported to the current runtime and three LOO methods.
//! Run the ignored acquisition with --nocapture; T32_CASE lines are complete witnesses.
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
fn environment_viability_prepares_ownership_without_pitch_self_exclusion() {
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
        let mut community = crate::life::community::Community::new(Timebase { fs: FS, hop: 512 });
        community.add_voice(voice);
        community.advance(512, 0, 512.0 / FS, &landscape);
        let voice = &mut community.voices[0];
        assert!(!voice.effective_control.pitch.leave_self_out);
        let input = voice.pitch_ctl.loo_input_mut();
        assert_eq!(input.own_power_scan.len(), landscape.space.n_bins());
        assert!(input.own_power_scan.iter().any(|&x| x > 0.0));
        assert!(
            input.computed_score.is_some(),
            "lifecycle must consume the snapshot"
        );
        assert!(voice.pitch_ctl.environment_score(&landscape, 220.0).abs() <= 1e-6);
    }
}

#[test]
#[ignore = "84 rendered scenes; run explicitly with --nocapture"]
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
                let mut mixed_stream = template.clone();
                let mut removed_stream = template.clone();
                let mut oracle_stream = template.clone();
                let mut max_pcm_error = 0.0_f32;
                let mut max_l1 = 0.0_f32;
                let mut max_field_error = 0.0_f32;
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
                    for (a, b) in [
                        (&actual.harmonicity, &oracle.harmonicity),
                        (&actual.roughness, &oracle.roughness),
                        (&actual.roughness_shape_raw, &oracle.roughness_shape_raw),
                        (
                            &actual.consonance_field_score,
                            &oracle.consonance_field_score,
                        ),
                    ] {
                        max_field_error = max_field_error.max(
                            a.iter()
                                .zip(b)
                                .map(|(a, b)| (a - b).abs())
                                .fold(0.0_f32, f32::max),
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
                let mut total_power = own_power.clone();
                if let Some(hz) = other_hz {
                    let other_power = projected_power(&recipe("sine", hz, &reference), &mix);
                    for (total, other) in total_power.iter_mut().zip(other_power) {
                        *total += other;
                    }
                }
                let mut input = LooInput {
                    own_power_scan: own_power,
                    total_power_scan: total_power.into(),
                    computed_score: None,
                };
                let candidate_scan = loo_score_scan(&mix, &mut input, &mut None);
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
                    assert!(
                        (score - reference_score).abs() - (old_score - reference_score).abs()
                            <= 1e-6,
                        "environment viability worsened: {family}, {base_hz}, {other_hz:?}, {kind:?}"
                    );
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
                    if method == "t31" {
                        assert!(
                            errors
                                .iter()
                                .zip(legacy_errors)
                                .all(|(new, old)| new - old <= 1e-6),
                            "viability current-position errors: {family}, {base_hz}, {other_hz:?}"
                        );
                    }
                    println!(
                        "T32_CASE {}",
                        json!({"method":method,"family":family,"base_hz":base_hz,"other_hz":other_hz,"recipe_body":format!("{:?}",own_recipe.body),
                    "ownership_bins":if family == "motion" {
                        Some((0..space.n_bins()).filter(|&i| mix.subjective_intensity[i] > 0.0 || removed.subjective_intensity[i] > 0.0)
                            .map(|i| json!([i,space.centers_hz[i],input.own_power_scan[i],input.total_power_scan[i],mix.subjective_intensity[i],removed.subjective_intensity[i]]))
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
                    family_pass.entry(family.to_string()).or_insert([true; 3])[method_index] &=
                        point_metrics["registered_movement_gate_pass"] == true
                            && body_metrics["registered_movement_gate_pass"] == true
                            && point_selection
                            && body_selection;
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
        json!({"cases":total,"fs":FS,"hop":hop,"families":families,"family_pass":family_pass,"habituation_enabled":false,"distinct_body_densities":density_cache.len()})
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
    hill.select_scored(pitches[3], scores[3], &mut scored, rng)
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
