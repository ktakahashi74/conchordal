//! Explicit PCM diagnostics; ordinary tests exercise history and cache contracts.
use super::*;
use crate::life::phonation_engine::ToneCmd;
use crate::life::sound::{BodySnapshot, RenderModulatorSpec, ToneAdsr};
use crate::life::voice::{PhonationBatch, ToneSpec, VoiceMetadata};
use crate::scenario::control::{BodyMethod, VoiceControl};
use crate::scenario::{ArticulationCoreConfig, VoiceSpec};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Write};

#[derive(Default, serde::Serialize)]
struct Costs {
    prepare: f64,
    score: f64,
    observe: f64,
    capture_delta: f64,
}
thread_local! {
    static COSTS: std::cell::RefCell<Costs> = Default::default();
    static NOW: std::cell::Cell<Tick> = const { std::cell::Cell::new(0) };
    static MOVES: std::cell::RefCell<Option<std::io::BufWriter<std::fs::File>>> = const { std::cell::RefCell::new(None) };
}
pub(crate) fn begin_hop(now: Tick) {
    NOW.set(now);
}
pub(crate) fn cost(key: &str, seconds: f64) {
    COSTS.with_borrow_mut(|c| match key {
        "prepare" => c.prepare += seconds,
        "score" => c.score += seconds,
        "observe" => c.observe += seconds,
        "capture" => c.capture_delta += seconds,
        _ => unreachable!(),
    });
}
pub(crate) fn take_costs() -> Value {
    COSTS.with_borrow_mut(|c| serde_json::to_value(std::mem::take(c)).unwrap())
}
pub(crate) fn slot_snapshot(runtime: &BodyMetabolism) -> Value {
    let slots: Vec<_> = runtime
        .sources
        .iter()
        .zip(&runtime.pcm)
        .map(|(source, pcm)| {
            assert_eq!((source.id, source.generation), (pcm.id, pcm.generation));
            if source.last_evaluated.is_none() {
                assert_eq!(source.evaluations, 0);
                assert!(source.held.is_none());
            }
            let held = source.held.map(|fitness| {
                assert!(fitness.score.is_finite());
                assert!((0.0..=1.0).contains(&fitness.level));
                assert!(fitness.in_band_mass.is_finite() && fitness.in_band_mass > 0.0);
                [fitness.score.to_bits(), fitness.level.to_bits()]
            });
            json!({"id":source.id,"generation":source.generation,
                "birth":source.birth_sample,"last":source.last_evaluated,
                "visit":source.visit_number,"evaluations":source.evaluations,"held":held})
        })
        .collect();
    json!(slots)
}
pub(crate) fn log_move(owner: (u64, u32), target: f32) {
    let Ok(path) = std::env::var("B7_MOVES") else {
        return;
    };
    MOVES.with_borrow_mut(|file| {
        let file = file
            .get_or_insert_with(|| std::io::BufWriter::new(std::fs::File::create(path).unwrap()));
        writeln!(
            file,
            "{}",
            json!({"id":owner.0,"generation":owner.1,
            "issue":NOW.get(),"completed":NOW.get(),"target_bits":target.to_bits(),"candidates":0})
        )
        .unwrap();
        file.flush().unwrap();
    });
}
fn voice(id: u64, generation: u32, method: BodyMethod) -> Voice {
    let mut control = VoiceControl::default();
    control.body.method = method;
    control.pitch.freq = 440.;
    VoiceSpec {
        control,
        articulation: ArticulationCoreConfig::default(),
    }
    .spawn(
        id,
        0,
        VoiceMetadata {
            generation,
            ..Default::default()
        },
        48_000.,
        17,
    )
}

fn batch(id: u64, recipe: &Recipe) -> PhonationBatch {
    PhonationBatch {
        source_id: id,
        cmds: vec![ToneCmd::On {
            tone_id: 1,
            kick: OnsetKick { strength: 1. },
        }],
        tones: vec![ToneSpec {
            opportunity: None,
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

fn recipe(row: &Value, fs: f32) -> Recipe {
    let family = row["family"].as_str().unwrap();
    let mut recipe = voice(2, 0, BodyMethod::Sine).representative_body_recipe(fs, 48_000);
    recipe.body = serde_json::from_value::<BodySnapshot>(row["body"].clone()).unwrap();
    recipe.freq_hz = row["base_hz"].as_f64().unwrap() as f32;
    if family != "landscape_density" && family != "landscape_peaks" {
        recipe.adsr = Some(if family == "adsr_slow" {
            ToneAdsr {
                attack_sec: 0.4,
                decay_sec: 0.1,
                sustain_level: 0.3,
                release_sec: 0.5,
            }
        } else {
            ToneAdsr {
                attack_sec: 0.005,
                decay_sec: 0.,
                sustain_level: 1.,
                release_sec: 0.5,
            }
        });
        recipe.smoothing_tau_sec = 0.;
        recipe.modulator = if family == "drone_sway" {
            RenderModulatorSpec::DroneSway {
                phase: 0.3,
                sway_rate: 0.8,
            }
        } else {
            RenderModulatorSpec::SeqGate { duration_sec: 1. }
        };
    }
    recipe
}

fn settings(frames: usize) -> BodyMetabolismConfig {
    BodyMetabolismConfig {
        enabled: true,
        updates_per_hop: 2,
        observation_frames: frames,
        representative_hold_sec: 1.,
    }
}

#[test]
#[ignore = "B-7 registered 84-scene PCM acquisition"]
fn b7_static_current_scores() {
    let inputs = std::fs::File::open(std::env::var("B7_INPUTS").unwrap()).unwrap();
    let mut output = std::io::BufWriter::new(
        std::fs::File::create(std::env::var("B7_OUTPUT").unwrap()).unwrap(),
    );
    let (nsgt, params) = crate::runtime::body_metabolism_test_core();
    let gaps = [1_i32, 2, 4, 8, 18];
    let mut count = 0;
    for line in BufReader::new(inputs).lines() {
        let row: Value = serde_json::from_str(&line.unwrap()).unwrap();
        let own = recipe(&row, params.fs);
        let other = recipe(
            &json!({"family":"sine", "base_hz":row["other_hz"].as_f64().unwrap_or(440.),
            "body":{"kind":"Sine","amp_scale":1.,"brightness":0.8,"inharmonic":0.,
                "spread":0.,"unison":1,"motion":0.,"ratios":null}}),
            params.fs,
        );
        let mut runtime = BodyMetabolism::new(
            nsgt.clone(),
            &params,
            settings(row["representative_frames"].as_u64().unwrap() as usize),
        );
        runtime.prepare(&[voice(2, 0, BodyMethod::Sine)], 0, &params);
        runtime.sources[0].prepare_body(&own);
        let mut renderer = super::super::schedule_renderer::ScheduleRenderer::new(runtime.time);
        let mut batches = vec![batch(2, &own)];
        if !row["other_hz"].is_null() {
            batches.push(batch(3, &other));
        }
        let mut thinned: Vec<_> = gaps
            .iter()
            .map(|_| SynchronousAnalysis::new(nsgt.clone(), &params))
            .collect();
        let mut last = [0_i32; 5];
        let frames = row["environment_frames"].as_u64().unwrap() as usize;
        for frame in 0..frames {
            let now = frame as Tick * runtime.time.hop as Tick;
            let mixed = renderer.render_with_source_pcm(
                if frame == 0 { &batches } else { &[] },
                now,
                &NeuralRhythms::default(),
                &mut runtime.pcm,
                |_, _, _, _| {},
            );
            runtime.observe(now, mixed.habitat, &params);
            for (i, gap) in gaps.iter().enumerate() {
                let end = frame as i32 + 1;
                if end % gap == 0 || frame + 1 == frames {
                    thinned[i].process_gap(&runtime.subtraction, end - last[i], &params);
                    last[i] = end;
                } else {
                    thinned[i].skip(&runtime.subtraction);
                }
            }
        }
        let source = &mut runtime.sources[0];
        source.current_sample = runtime.next_sample;
        let current = own.freq_hz.log2();
        let reference = source
            .score_valid(current, &params, &runtime.shared.landscape)
            .unwrap();
        let scores: Vec<_> = thinned
            .iter()
            .map(|analysis| {
                evaluate(
                    &analysis.landscape.space,
                    &source.density_scan,
                    &analysis.du,
                    &analysis.landscape.consonance_field_score_eff,
                    &params.consonance_representation,
                )
                .unwrap()
                .score
            })
            .collect();
        let current_index = row["cents"]
            .as_array()
            .unwrap()
            .iter()
            .position(|c| c.as_f64() == Some(0.))
            .unwrap();
        let saved = row["body_pcm_scores"][current_index].as_f64().unwrap();
        writeln!(
            output,
            "{}",
            json!({"scene":count,"family":row["family"],"reference":reference.score,
            "saved":saved,"gaps":gaps,"scores":scores})
        )
        .unwrap();
        for score in scores {
            assert!((score - reference.score).abs() <= 0.025);
        }
        assert!((f64::from(reference.score) - saved).abs() <= 0.025);
        count += 1;
    }
    assert_eq!(count, 84);
}

#[test]
fn newborn_unknown_hold_and_exact_density_invalidation() {
    let (nsgt, params) = crate::runtime::body_metabolism_test_core();
    let mut runtime = BodyMetabolism::new(nsgt, &params, settings(8));
    let voices = [voice(2, 0, BodyMethod::Sine)];
    runtime.prepare(&voices, 0, &params);
    let shared = runtime.shared.landscape.clone();
    assert!(
        runtime.sources[0]
            .score(440_f32.log2(), &params, &shared)
            .is_err()
    );
    runtime.observe(0, &vec![0.; runtime.time.hop], &params);
    let source = &mut runtime.sources[0];
    source.current_sample = runtime.next_sample;
    let first = source.score(440_f32.log2(), &params, &shared).unwrap();
    let builds = source.density_builds;
    let held = source.score(441_f32.log2(), &params, &shared).unwrap();
    assert_eq!(first.score.to_bits(), held.score.to_bits());
    assert_eq!(source.density_builds, builds);
    source.epoch += 1;
    source.score(441_f32.log2(), &params, &shared).unwrap();
    assert_eq!(source.density_builds, builds + 1);
    let mut changed = params.clone();
    changed.loudness_exp += 0.01;
    source.epoch += 1;
    source.score(441_f32.log2(), &changed, &shared).unwrap();
    assert_eq!(source.density_builds, builds + 2);
}

#[test]
fn numbered_slots_match_the_registered_cursor_for_eighteen_voices() {
    let (nsgt, params) = crate::runtime::body_metabolism_test_core();
    let mut runtime = BodyMetabolism::new(nsgt, &params, settings(1));
    let voices: Vec<_> = (1..=18).map(|id| voice(id, 0, BodyMethod::Sine)).collect();
    runtime.prepare(&voices, 0, &params);
    for frame in 0..27 {
        let now = frame * runtime.time.hop as Tick;
        runtime.select_next(now, now + runtime.time.hop as Tick);
        for offset in 0..2 {
            assert_eq!(
                runtime.sources[offset].id,
                (2 * frame + offset as u64) % 18 + 1
            );
        }
        for (source, pcm) in runtime.sources.iter().zip(&runtime.pcm) {
            assert_eq!((source.id, source.generation), (pcm.id, pcm.generation));
        }
    }
}

#[test]
fn numbered_slots_match_the_former_queue_across_membership_changes() {
    let (nsgt, params) = crate::runtime::body_metabolism_test_core();
    let mut runtime = BodyMetabolism::new(nsgt, &params, settings(1));
    let mut voices: Vec<_> = [6, 4, 2, 5, 1, 3]
        .into_iter()
        .map(|id| voice(id, 0, BodyMethod::Sine))
        .collect();
    // The independent oracle reproduces the saved queue's prepare and visit operations.
    let mut queue = std::collections::VecDeque::new();
    let zero = vec![0.; runtime.time.hop];
    let mut latest = 0;
    let shared = runtime.shared.landscape.clone();
    for frame in 0..30 {
        if frame == 4 {
            voices.retain(|voice| ![2, 5].contains(&voice.id()));
            voices.push(voice(8, 0, BodyMethod::Sine));
            voices.push(voice(7, 0, BodyMethod::Sine));
        }
        if frame == 7 {
            voices.push(voice(2, 1, BodyMethod::Sine));
        }
        if frame == 9 {
            voices.reverse();
        }
        if frame == 12 {
            voices.retain(|voice| voice.id() != 7);
            voices.push(voice(7, 1, BodyMethod::Sine));
        }
        queue.retain(|&(id, generation)| {
            voices
                .iter()
                .any(|voice| (voice.id(), voice.metadata.generation) == (id, generation))
        });
        for voice in &voices {
            let identity = (voice.id(), voice.metadata.generation);
            if !queue.contains(&identity) {
                queue.push_back(identity);
            }
        }
        let budget = runtime.settings.updates_per_hop.min(queue.len());
        let expected: Vec<_> = queue.iter().copied().take(budget).collect();
        let now = frame * runtime.time.hop as Tick;
        runtime.prepare(&voices, now, &params);
        if frame == 4 {
            for id in [8, 7] {
                let source = runtime.source(id, 0).unwrap().0;
                assert!(source.last_evaluated.is_none());
                assert!(source.score(440_f32.log2(), &params, &shared).is_err());
                assert!(!expected.contains(&(id, 0)), "birth joins the back");
            }
        }
        runtime.observe(now, &zero, &params);
        assert_eq!(
            runtime.sources[..budget]
                .iter()
                .map(|s| (s.id, s.generation))
                .collect::<Vec<_>>(),
            expected
        );
        for _ in 0..budget {
            let identity = queue.pop_front().unwrap();
            queue.push_back(identity);
        }
        let mut numbered: Vec<_> = runtime
            .sources
            .iter()
            .map(|s| (s.visit_number, s.id, s.generation))
            .collect();
        numbered.sort_unstable();
        assert_eq!(
            numbered.iter().map(|s| (s.1, s.2)).collect::<Vec<_>>(),
            queue.iter().copied().collect::<Vec<_>>()
        );
        assert!(runtime.latest_visit > latest);
        latest = runtime.latest_visit;
    }
}

#[test]
fn numbered_slots_keep_cursor_order_when_the_population_is_not_divisible() {
    let (nsgt, params) = crate::runtime::body_metabolism_test_core();
    let mut runtime = BodyMetabolism::new(nsgt, &params, settings(1));
    let voices: Vec<_> = (1..=3).map(|id| voice(id, 0, BodyMethod::Sine)).collect();
    runtime.prepare(&voices, 0, &params);
    for (frame, expected) in [[1, 2], [3, 1], [2, 3], [1, 2]].into_iter().enumerate() {
        let now = frame as Tick * runtime.time.hop as Tick;
        runtime.select_next(now, now + runtime.time.hop as Tick);
        assert_eq!(
            runtime.sources[..2]
                .iter()
                .map(|source| source.id)
                .collect::<Vec<_>>(),
            expected
        );
    }
}

#[test]
fn bounded_number_selection_matches_sort_without_depending_on_slot_order() {
    let (nsgt, params) = crate::runtime::body_metabolism_test_core();
    for budget in [1, 2, 9, 20] {
        let mut config = settings(1);
        config.updates_per_hop = budget;
        let mut runtime = BodyMetabolism::new(nsgt.clone(), &params, config);
        let voices: Vec<_> = (1..=18)
            .rev()
            .map(|id| voice(id, 0, BodyMethod::Sine))
            .collect();
        runtime.prepare(&voices, 0, &params);
        for frame in 0..20 {
            let mut expected: Vec<_> = runtime
                .sources
                .iter()
                .map(|s| (s.visit_number, s.id, s.generation))
                .collect();
            expected.sort_unstable();
            expected.truncate(budget.min(18));
            let now = frame * runtime.time.hop as Tick;
            runtime.select_next(now, now + runtime.time.hop as Tick);
            assert_eq!(
                runtime.sources[..expected.len()]
                    .iter()
                    .map(|s| (s.id, s.generation))
                    .collect::<Vec<_>>(),
                expected.iter().map(|s| (s.1, s.2)).collect::<Vec<_>>()
            );
        }
    }
}

#[cfg(feature = "profile-alloc")]
#[test]
fn prepared_body_slots_capture_skip_and_density_rebuild_without_allocating() {
    let (nsgt, params) = crate::runtime::body_metabolism_test_core();
    crate::life::modal::register_modal();
    let mut runtime = BodyMetabolism::new(nsgt, &params, settings(8));
    let voices = [
        voice(3, 0, BodyMethod::Modal),
        voice(1, 0, BodyMethod::Sine),
        voice(2, 0, BodyMethod::Harmonic),
    ];
    runtime.prepare(&voices, 0, &params);
    let shared = runtime.shared.landscape.clone();
    let mut renderer = crate::life::schedule_renderer::ScheduleRenderer::new(runtime.time);
    let batches: Vec<_> = voices
        .iter()
        .map(|voice| {
            batch(
                voice.id(),
                &voice.representative_body_recipe(params.fs, 48000),
            )
        })
        .collect();
    let frame = renderer.render_with_source_pcm(
        &batches,
        0,
        &NeuralRhythms::default(),
        &mut runtime.pcm,
        |_, _, _, _| {},
    );
    runtime.observe(0, frame.habitat, &params);
    // Warm diagnostic TLS before measuring the production body path.
    let _ = take_costs();
    for frame in 1..25 {
        let now = frame * runtime.time.hop as Tick;
        crate::runtime_profile::begin_allocations();
        runtime.prepare(&voices, now, &params);
        for source in &mut runtime.sources {
            // Force a new current density and modulator recipe, with existing Tone storage.
            let mut recipe = voices
                .iter()
                .find(|voice| voice.id() == source.id)
                .unwrap()
                .representative_body_recipe(params.fs, 48000);
            recipe.modulator = RenderModulatorSpec::DroneSway {
                phase: 0.,
                sway_rate: 0.3 + frame as f32 * 0.01,
            };
            source.prepare_body(&recipe);
            let _ = source.score((440. + frame as f32).log2(), &params, &shared);
        }
        let rendered = renderer.render_with_source_pcm(
            &[],
            now,
            &NeuralRhythms::default(),
            &mut runtime.pcm,
            |_, _, _, _| {},
        );
        runtime.observe(now, rendered.habitat, &params);
        let counts = crate::runtime_profile::finish_allocations().unwrap();
        assert_eq!(counts.count, 0, "prepared body hop {frame}");
        assert_eq!(counts.bytes, 0, "prepared body hop {frame}");
    }
    assert!(
        runtime
            .sources
            .iter()
            .all(|source| source.density_builds > 1)
    );
}
