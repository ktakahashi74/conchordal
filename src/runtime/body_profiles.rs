//! Explicit offline acquisition from the actual development Voice and renderer state.

use super::*;
use crate::life::action_candidates::{BodyState, Class};
use crate::life::phonation_engine::{OnsetKick, ToneCmd, ToneUpdate};
use crate::life::voice::{SoundBody, ToneSpec};
use serde_json::{Value, json};
use std::fs::{self, File};
use std::io::{BufWriter, Write};

const HORIZON: u64 = 192_000;
const PERIOD: u64 = 9_600;

struct Frozen {
    actual: ScheduleRenderer,
    replay: ScheduleRenderer,
    prefix: [Vec<f32>; 2],
    onset: ToneSpec,
    routing: crate::scenario::control::Routing,
    continuous_drive: f32,
    rhythms: NeuralRhythms,
}

fn write_pcm(path: &Path, samples: &[f32]) {
    let mut file = BufWriter::new(File::create(path).unwrap());
    for sample in samples {
        assert!(sample.is_finite());
        file.write_all(&sample.to_le_bytes()).unwrap();
    }
    file.flush().unwrap();
}

#[test]
#[ignore = "explicit offline medoid action acquisition; requires model directory and fresh output"]
fn acquire_medoid_action_profiles() {
    let input = std::path::PathBuf::from(std::env::var("CONCHORDAL_I10_MEDOID_DIR").unwrap());
    let output =
        std::path::PathBuf::from(std::env::var("CONCHORDAL_I10_MEDOID_ACTION_DIR").unwrap());
    let model: Value =
        serde_json::from_slice(&fs::read(input.join("model-8.json")).unwrap()).unwrap();
    assert_eq!(
        model["model_version"],
        "b3eb8cba69007e16aedb013d9ee8ba617b3ecb4d3420afb492768bfeba02fd28"
    );
    let config =
        AppConfig::load_or_default(input.join("extraction.toml").to_str().unwrap()).unwrap();
    config.validate().unwrap();
    assert_eq!(
        (
            config.audio.sample_rate,
            config.analysis.hop_size,
            config.analysis.nfft
        ),
        (48000, 512, 2048)
    );
    crate::life::modal::register_modal();
    fs::create_dir(&output).expect("fresh acquisition directory required");
    let mut cases = Vec::new();
    for medoid in model["medoids"].as_array().unwrap() {
        let source = model["records"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["record_id"] == medoid["record_id"])
            .unwrap();
        let id = medoid["record_id"].as_str().unwrap();
        let issue = source["descriptor"]["end"].as_u64().unwrap();
        assert_eq!(issue % 512, 0);
        let directory = output.join(id);
        fs::create_dir(&directory).unwrap();
        let scenario_path = input.join(source["scenario"].as_str().unwrap());
        let scenario = compile_scenario_from_script(
            &scenario_path,
            &render_compile_args(scenario_path.to_str().unwrap(), None),
            &config,
        )
        .unwrap();
        let mut reporter =
            JsonlReporter::create(directory.join("ordinary.jsonl").to_str().unwrap()).unwrap();
        reporter.write_meta(scenario.seed).unwrap();
        let time = crate::core::timebase::Timebase {
            fs: 48000.,
            hop: 512,
        };
        let (tx, rx) = bounded(1);
        let mut replay = ScheduleRenderer::new(time);
        let mut prefix = [Vec::new(), Vec::new()];
        let mut recipe: Option<ToneSpec> = None;
        let probe: OfflineBodyProbe = Box::new(move |state, now, count, actual| {
            if now >= issue {
                return;
            }
            assert_eq!(state.pop.voices.len(), 1);
            let voice = &state.pop.voices[0];
            assert_eq!((voice.id, voice.metadata.generation), (1, 0));
            let batches = &state.phonation_batches_buf[..count];
            for spec in batches.iter().flat_map(|b| &b.tones) {
                assert!(recipe.is_none(), "registered prefix has exactly one onset");
                recipe = Some(spec.clone());
            }
            let expected = replay.render(batches, now, &state.current_landscape.rhythm);
            assert_eq!(expected.habitat, actual[0]);
            assert_eq!(expected.presentation, actual[1]);
            for bus in 0..2 {
                prefix[bus].extend_from_slice(actual[bus]);
            }
            if now + 512 == issue {
                let mut onset = recipe.clone().unwrap();
                onset.opportunity = None;
                onset.tone_id = 1;
                onset.onset = issue;
                onset.freq_hz = voice.body.base_freq_hz();
                onset.amp = voice.compute_target_amp();
                onset.body = voice.body_snapshot();
                onset.render_modulator = voice
                    .articulation
                    .render_modulator_spec(voice.phonation_engine.mode);
                let mut rhythms = state.current_landscape.rhythm;
                for _ in 0..512 {
                    rhythms.advance_in_place(1. / 48000.);
                }
                tx.send(Frozen {
                    actual: state.schedule_renderer.fork_source(1),
                    replay: replay.fork_source(1),
                    prefix: std::mem::take(&mut prefix),
                    onset,
                    routing: voice.effective_control.body.routing,
                    continuous_drive: voice.effective_control.body.continuous_drive,
                    rhythms,
                })
                .unwrap();
            }
        });
        let wiring = wire_runtime(
            &config,
            48000,
            id.to_string(),
            scenario,
            Arc::new(AtomicBool::new(false)),
            WiringOptions {
                offline_body_probe: Some(probe),
                ui_channel_capacity: 1,
                listener_forced: true,
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
        join_thread("listener", wiring.listener_analysis_handle.unwrap()).unwrap();
        assert!(wiring.report_error_rx.unwrap().try_recv().is_err());
        let frozen = rx.recv().unwrap();
        let ordinary: Vec<Value> = fs::read_to_string(directory.join("ordinary.jsonl"))
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let descriptor = ordinary
            .iter()
            .find(|r| {
                r["type"] == "body_descriptor"
                    && r["bus"] == source["descriptor"]["bus"]
                    && r["end"] == issue
            })
            .unwrap();
        for field in [
            "raw_values",
            "mask",
            "start",
            "source_start",
            "available",
            "coverage",
        ] {
            assert_eq!(
                descriptor[field], source["descriptor"][field],
                "{id}/{field}"
            );
        }
        assert_eq!(frozen.prefix[0].len(), issue as usize);
        assert_eq!(frozen.prefix[1].len(), issue as usize);
        for bus in 0..2 {
            write_pcm(
                &directory.join(format!("prefix-bus{bus}.f32le")),
                &frozen.prefix[bus],
            );
        }
        let offsets: Vec<u64> = (0..32).map(|k| (k * HORIZON + 15) / 31).collect();
        let mut branches = Vec::new();
        for (name, class) in [
            ("onset_now", Class::OnsetNow),
            ("delayed_onset", Class::DelayedOnset),
            ("wait", Class::Wait),
            ("skip", Class::Skip),
            ("continue", Class::Continue),
            ("release", Class::Release),
            ("gap", Class::Gap),
        ] {
            for &offset in &offsets {
                let input = class.input(
                    issue,
                    issue + offset,
                    Some(PERIOD),
                    48000,
                    BodyState {
                        permits_action: Some(true),
                        active_at_candidate: Some(true),
                        pending_opportunity: true,
                        due_unconsumed: true,
                    },
                );
                let Some(action) = input else {
                    continue;
                };
                let mut batch = PhonationBatch {
                    source_id: 1,
                    source_generation: 0,
                    routing: frozen.routing,
                    ..Default::default()
                };
                if let Some(at) = action.excitation_at {
                    let mut spec = frozen.onset.clone();
                    spec.onset = at;
                    batch.cmds.push(ToneCmd::On {
                        tone_id: spec.tone_id,
                        kick: OnsetKick { strength: 1. },
                    });
                    batch.cmds.push(ToneCmd::Update {
                        tone_id: spec.tone_id,
                        at_tick: Some(at),
                        update: ToneUpdate {
                            target_freq_hz: None,
                            target_amp: None,
                            continuous_drive: Some(frozen.continuous_drive),
                        },
                    });
                    batch.tones.push(spec);
                }
                if let Some(at) = action.release_at {
                    batch.cmds.push(ToneCmd::Off {
                        tone_id: 0,
                        off_tick: at,
                    });
                }
                let mut actual = frozen.actual.fork_source(1);
                let mut direct = frozen.replay.fork_source(1);
                let mut rhythms = frozen.rhythms;
                let mut pcm = [
                    Vec::with_capacity(HORIZON as usize),
                    Vec::with_capacity(HORIZON as usize),
                ];
                for delta in (0..HORIZON).step_by(512) {
                    let batches = if delta == 0 {
                        std::slice::from_ref(&batch)
                    } else {
                        &[]
                    };
                    let a = actual.render(batches, issue + delta, &rhythms);
                    let b = direct.render(batches, issue + delta, &rhythms);
                    assert_eq!(a.habitat, b.habitat, "{id}/{name}/{offset}");
                    assert_eq!(a.presentation, b.presentation, "{id}/{name}/{offset}");
                    pcm[0].extend_from_slice(a.habitat);
                    pcm[1].extend_from_slice(a.presentation);
                    for _ in 0..512 {
                        rhythms.advance_in_place(1. / 48000.);
                    }
                }
                let paths: [String; 2] =
                    std::array::from_fn(|bus| format!("{name}-{offset}-bus{bus}.f32le"));
                for bus in 0..2 {
                    write_pcm(&directory.join(&paths[bus]), &pcm[bus]);
                }
                branches.push(json!({"action": action, "offset": offset, "pcm": paths,
                    "energy_per_hop": pcm.map(|samples| samples.chunks_exact(512).map(|chunk|
                        chunk.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / 512.).collect::<Vec<_>>())}));
            }
        }
        let metadata = json!({"record_id": id, "model_version": model["model_version"],
            "source_record": source, "issue_sample": issue, "source_id": 1, "source_generation": 0,
            "sample_rate": 48000, "hop_samples": 512, "horizon_samples": HORIZON,
            "offset_samples": offsets, "assay_period_samples": PERIOD,
            "new_onset": format!("{:?}", frozen.onset), "continuous_drive": frozen.continuous_drive,
            "routing": [frozen.routing.to_habitat, frozen.routing.to_presentation],
            "rhythms_at_issue": format!("{:?}", frozen.rhythms),
            "branches": branches,
            "scope": "offline conditional targets; fixed future rhythm and no later commands; pending/due/permission/period are assay conditions, not observed clock facts"});
        fs::write(
            directory.join("profiles.json"),
            serde_json::to_vec_pretty(&metadata).unwrap(),
        )
        .unwrap();
        println!(
            "{id}: {} branches; exact ordinary prefix and descriptor",
            metadata["branches"].as_array().unwrap().len()
        );
        cases.push(id.to_string());
    }
    fs::write(output.join("manifest.json"), serde_json::to_vec_pretty(&json!({
        "schema": "i10-medoid-action-targets-v1", "model_version": model["model_version"],
        "cases": cases, "claim": "actual renderer forks and independent command replays; not live acoustic predictions or transfer calibration"
    })).unwrap()).unwrap();
}

mod policy_default;
mod representative_gap;
