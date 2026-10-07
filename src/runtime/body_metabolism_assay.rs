//! B-7 deterministic, bounded offline rendering on actual Voice and PCM paths.

use super::*;
use serde_json::json;
use std::io::Write;

#[derive(PartialEq)]
struct FlagSnapshot {
    sample: Tick,
    voices: serde_json::Value,
    slots: serde_json::Value,
    pcm: [Vec<u32>; 2],
}

fn render_birth_metabolism_flags(
    birth: bool,
    metabolism: Option<bool>,
    prototype: bool,
) -> Vec<FlagSnapshot> {
    let path = "tests/scripts/body_birth_metabolism_flags.rhai";
    let mut config = AppConfig {
        birth_surrogate: birth,
        body_metabolism: metabolism.map(|enabled| crate::config::BodyMetabolismConfig {
            enabled,
            updates_per_hop: 2,
            observation_frames: 8,
            representative_hold_sec: 1.0,
        }),
        render_prototype: prototype.then(crate::config::RenderPrototypeConfig::default),
        ..Default::default()
    };
    config.analysis.nfft = 2048;
    config.dcc.coupling_strength = 0.0;
    let scenario = compile_scenario_from_script(
        std::path::Path::new(path),
        &render_compile_args(path, Some(42)),
        &config,
    )
    .unwrap();
    assert_eq!(scenario.seed, 42);
    let snapshots = Arc::new(std::sync::Mutex::new(Vec::new()));
    let captured = Arc::clone(&snapshots);
    let probe: OfflineBodyProbe = Box::new(move |state, now, _, pcm| {
        if prototype {
            assert!(
                state.pop.body_metabolism.is_none(),
                "replacement renderer enabled legacy PCM capture and body evaluation"
            );
        }
        let voices: Vec<_> = state
            .pop
            .voices
            .iter()
            .map(|voice| {
                let energy = match &voice.articulation.core {
                    crate::life::voice::AnyArticulationCore::Entrain(core) => {
                        assert!((0.0..=1.0).contains(&core.energy));
                        Some(core.energy.to_bits())
                    }
                    _ => None,
                };
                let frequency = voice.body.base_freq_hz();
                assert!(frequency.is_finite() && frequency > 0.0);
                json!({"id":voice.id(),"generation":voice.metadata.generation,
                "alive":voice.is_alive(),"energy":energy,"freq":frequency.to_bits(),
                "target":voice.pitch_ctl.target_pitch_log2().to_bits()})
            })
            .collect();
        let slots = state.pop.body_metabolism.as_ref().map_or_else(
            || json!([]),
            crate::life::body_metabolism::assay::slot_snapshot,
        );
        let pcm = pcm.map(|bus| {
            bus.iter()
                .map(|sample| {
                    assert!(sample.is_finite());
                    sample.to_bits()
                })
                .collect()
        });
        captured.lock().unwrap().push(FlagSnapshot {
            sample: now,
            voices: json!(voices),
            slots,
            pcm,
        });
    });
    let (wav_tx, _wav_rx) = crossbeam_channel::unbounded();
    let wiring = wire_runtime(
        &config,
        48_000,
        "body flag regression".into(),
        scenario,
        Arc::new(AtomicBool::new(false)),
        WiringOptions {
            offline_body_probe: Some(probe),
            ui_channel_capacity: 1,
            listener_forced: false,
            wait_user_exit: false,
            start_playing: true,
            audio_prod: None,
            wav_tx: prototype.then_some(wav_tx),
            reporter: None,
            deterministic_analysis: true,
            deterministic_footprints: true,
            guard_meter: None,
            underrun_frames: None,
            reserve_runtime_ids_through: 0,
            profile: None,
            audio_counters: None,
        },
    )
    .unwrap();
    join_thread("worker", wiring.worker_handle).unwrap();
    join_thread("analysis", wiring.analysis_handle).unwrap();
    if let Some(handle) = wiring.listener_analysis_handle {
        join_thread("listener", handle).unwrap();
    }
    Arc::try_unwrap(snapshots)
        .ok()
        .unwrap()
        .into_inner()
        .unwrap()
}

#[test]
fn body_birth_and_metabolism_flags_are_deterministic_and_keep_unknown_births() {
    let mut runs = Vec::new();
    for birth in [false, true] {
        for metabolism in [false, true] {
            let enabled = metabolism.then_some(true);
            let first = render_birth_metabolism_flags(birth, enabled, false);
            let second = render_birth_metabolism_flags(birth, enabled, false);
            assert!(
                first == second,
                "non-deterministic flags: birth={birth}, metabolism={metabolism}"
            );
            assert!(!first.is_empty());
            assert!(
                first.iter().any(|row| row
                    .voices
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|voice| voice["id"].as_u64().unwrap() > 7)),
                "missing respawn"
            );
            if metabolism {
                assert!(
                    first.iter().any(|row| row
                        .slots
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|slot| slot["last"].is_null() && slot["id"].as_u64().unwrap() >= 2)),
                    "missing pending newborn"
                );
                assert!(
                    first.iter().any(|row| row
                        .slots
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|slot| !slot["held"].is_null())),
                    "missing body fitness"
                );
            } else {
                assert!(
                    first
                        .iter()
                        .all(|row| row.slots.as_array().unwrap().is_empty())
                );
                let disabled = render_birth_metabolism_flags(birth, Some(false), false);
                assert!(
                    first == disabled,
                    "explicitly disabled metabolism changed birth={birth}"
                );
            }
            println!(
                "body flags birth={birth} metabolism={metabolism}: {} deterministic hops",
                first.len()
            );
            runs.push(first);
        }
    }
    let first_field_birth = |run: &Vec<FlagSnapshot>| {
        let row = run
            .iter()
            .find(|row| {
                row.voices
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|voice| voice["id"] == 5)
            })
            .unwrap();
        (
            row.sample,
            row.voices
                .as_array()
                .unwrap()
                .iter()
                .filter(|voice| (2..=5).contains(&voice["id"].as_u64().unwrap()))
                .cloned()
                .collect::<Vec<_>>(),
        )
    };
    assert_eq!(
        first_field_birth(&runs[2]),
        first_field_birth(&runs[3]),
        "newborn energy or Field birth changed before its first body evaluation"
    );
    assert_ne!(
        first_field_birth(&runs[0]).1,
        first_field_birth(&runs[2]).1,
        "fixture did not exercise the body-aware Field placement"
    );
    assert!(
        runs[0]
            .iter()
            .zip(&runs[1])
            .any(|(point, body)| point.voices != body.voices),
        "fixture did not exercise body metabolism's energy input"
    );
}

#[test]
fn phase3_uses_point_metabolism_without_legacy_pcm_capture_even_with_birth_enabled() {
    let control = render_birth_metabolism_flags(false, None, true);
    assert!(!control.is_empty());
    assert!(control.iter().any(|row| {
        row.pcm
            .iter()
            .flatten()
            .any(|sample| f32::from_bits(*sample) != 0.0)
    }));
    assert!(control.iter().any(|row| {
        row.voices
            .as_array()
            .unwrap()
            .iter()
            .any(|voice| voice["id"].as_u64().unwrap() > 7)
    }));
    for birth in [false, true] {
        for metabolism in [None, Some(false), Some(true)] {
            let actual = render_birth_metabolism_flags(birth, metabolism, true);
            assert!(
                actual
                    .iter()
                    .all(|row| row.slots.as_array().unwrap().is_empty())
            );
            assert!(
                actual == control,
                "legacy body flag changed Phase 3 energy, births, tails or actual bus PCM: \
                 birth={birth}, metabolism={metabolism:?}"
            );
            println!(
                "Phase 3 birth={birth} metabolism={metabolism:?}: {} point-metabolism hops",
                actual.len()
            );
        }
    }
}

#[test]
#[ignore = "B-7 explicit dynamic diagnostic, no real-time acceptance"]
fn b7_dynamic_etude_render() {
    let path = std::env::var("B7_SCENARIO").unwrap();
    let directory = std::path::PathBuf::from(std::env::var("B7_DIRECTORY").unwrap());
    let seed = std::env::var("B7_SEED")
        .map(|value| value.parse::<u64>().unwrap())
        .unwrap_or(42);
    let config = AppConfig {
        body_metabolism: Some(crate::config::BodyMetabolismConfig {
            enabled: true,
            updates_per_hop: 2,
            observation_frames: 8,
            representative_hold_sec: 1.,
        }),
        ..Default::default()
    };
    crate::life::modal::register_modal();
    let scenario = compile_scenario_from_script(
        std::path::Path::new(&path),
        &render_compile_args(&path, Some(seed)),
        &config,
    )
    .unwrap();
    assert_eq!(scenario.seed, seed);
    let mut states =
        std::io::BufWriter::new(std::fs::File::create(directory.join("states.jsonl")).unwrap());
    let mut wav = hound::WavWriter::create(
        directory.join("habitat.wav"),
        hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        },
    )
    .unwrap();
    let probe: OfflineBodyProbe = Box::new(move |state, now, _, pcm| {
        for &sample in pcm[0] {
            assert!(sample.is_finite());
            wav.write_sample(sample).unwrap();
        }
        let voices: Vec<_> = state
            .pop
            .voices
            .iter()
            .map(|voice| {
                let energy = match &voice.articulation.core {
                    crate::life::voice::AnyArticulationCore::Entrain(core) => Some(core.energy),
                    _ => None,
                };
                json!({"id":voice.id(),"generation":voice.metadata.generation,"energy":energy,
                "alive":voice.is_alive(),"freq_hz":voice.body.base_freq_hz(),
                "target_bits":voice.pitch_ctl.target_pitch_log2().to_bits()})
            })
            .collect();
        writeln!(
            states,
            "{}",
            json!({"sample":now,"voices":voices,
            "metabolic_counts":crate::life::body_metabolism::assay::take_costs(),
            "body_evaluations":state.pop.body_metabolism.as_ref().unwrap().evaluation_count()})
        )
        .unwrap();
        states.flush().unwrap();
    });
    let mut reporter =
        JsonlReporter::create(directory.join("native.jsonl").to_str().unwrap()).unwrap();
    reporter.write_meta(scenario.seed).unwrap();
    let wiring = wire_runtime(
        &config,
        48_000,
        "B-7 etude".into(),
        scenario,
        Arc::new(AtomicBool::new(false)),
        WiringOptions {
            offline_body_probe: Some(probe),
            ui_channel_capacity: 1,
            listener_forced: false,
            wait_user_exit: false,
            start_playing: true,
            audio_prod: None,
            wav_tx: None,
            reporter: Some(reporter),
            deterministic_analysis: true,
            deterministic_footprints: true,
            guard_meter: None,
            underrun_frames: None,
            reserve_runtime_ids_through: 0,
            profile: None,
            audio_counters: None,
        },
    )
    .unwrap();
    join_thread("worker", wiring.worker_handle).unwrap();
    join_thread("analysis", wiring.analysis_handle).unwrap();
    if let Some(handle) = wiring.listener_analysis_handle {
        join_thread("listener", handle).unwrap();
    }
    if let Some(rx) = wiring.report_error_rx {
        if let Ok(error) = rx.try_recv() {
            panic!("{error}");
        }
    }
}
