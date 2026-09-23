//! I11-1 §5.9: the representative footprint a decision used, against the tone that sounded.

use super::*;
use crate::life::action_candidates::energy::project_window;
use crate::life::self_prediction::{ScheduledRelease, ToneEnergy};
use crate::life::sound::Tone;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Mutex;

/// A granted onset frozen as issued: actual kick, seed, rhythms and planned release.
struct Issued {
    source_id: u64,
    tone_id: u64,
    onset: u64,
    issued_at: u64,
    kick: f32,
    energy: ToneEnergy,
}

#[test]
#[ignore = "registered offline re-projection for I11-1 §5.9; requires registered inputs and fresh output"]
fn acquire_representative_gap() {
    let input = std::path::PathBuf::from(std::env::var("CONCHORDAL_I11_GAP_INPUTS").unwrap());
    let output = std::path::PathBuf::from(std::env::var("CONCHORDAL_I11_GAP_OUTPUT").unwrap());
    let registration: Value =
        serde_json::from_slice(&fs::read(input.join("registration.json")).unwrap()).unwrap();
    assert_eq!(registration["schema"], "i11-representative-gap-v1");
    crate::life::modal::register_modal();
    fs::create_dir(&output).expect("fresh acquisition directory required");
    for case in registration["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let directory = output.join(id);
        fs::create_dir(&directory).unwrap();
        let [script, config_path] = ["script", "config"].map(|key| {
            let path = input.join(case[key].as_str().unwrap());
            assert_eq!(
                format!("{:x}", Sha256::digest(fs::read(&path).unwrap())),
                case[format!("{key}_sha256")].as_str().unwrap(),
                "{id}: {key} differs from its registration"
            );
            path
        });
        let config = AppConfig::load_or_default(config_path.to_str().unwrap()).unwrap();
        config.validate().unwrap();
        assert!(config.temporal_onset_comparison.is_some());
        let scenario = compile_scenario_from_script(
            &script,
            &render_compile_args(script.to_str().unwrap(), None),
            &config,
        )
        .unwrap();
        let report_path = directory.join("report.jsonl");
        let mut reporter = JsonlReporter::create(report_path.to_str().unwrap()).unwrap();
        reporter.write_meta(scenario.seed).unwrap();
        let time = crate::core::timebase::Timebase {
            fs: config.audio.sample_rate as f32,
            hop: config.analysis.hop_size,
        };
        let issued = Arc::new(Mutex::new(Vec::new()));
        let sink = Arc::clone(&issued);
        let probe: OfflineBodyProbe = Box::new(move |state, now, count, _| {
            let rhythms = state.current_landscape.rhythm;
            for batch in &state.phonation_batches_buf[..count] {
                for cmd in &batch.cmds {
                    let ToneCmd::On { tone_id, kick } = *cmd else {
                        continue;
                    };
                    let recipe = batch
                        .tones
                        .iter()
                        .find(|tone| tone.tone_id == tone_id)
                        .expect("a granted onset carries its recipe");
                    // The same freeze as the I10 `issued` tone (energy_projection.rs).
                    let Some(mut tone) = Tone::from_parts(
                        time,
                        recipe.onset,
                        recipe.hold_ticks.unwrap_or(60 * 48000),
                        recipe.freq_hz,
                        recipe.amp,
                        Some(recipe.body.clone()),
                        Some(recipe.render_modulator.clone()),
                        recipe.adsr,
                    ) else {
                        continue;
                    };
                    tone.seed_modal_phases(crate::life::schedule_renderer::modal_phase_seed(
                        batch.source_id,
                        recipe.onset,
                        tone_id,
                    ));
                    tone.set_smoothing_tau_sec(recipe.smoothing_tau_sec);
                    tone.schedule_planned_kick(kick);
                    tone.arm_onset_trigger(kick.strength.max(0.));
                    let (_, amplitude, envelope) = tone.prediction_parameters(None);
                    let hop = time.hop as u64;
                    let energy = ToneEnergy {
                        sine: tone.prediction_sine(now),
                        bank: tone.prediction_bank(now),
                        amplitude,
                        envelope,
                        control: Some(tone.prediction_control(now, &rhythms)),
                        scheduled_release: recipe
                            .opportunity
                            .and_then(|receipt| receipt.planned_release_at)
                            .map(|off| ScheduledRelease {
                                apply_at_sample: now + (off - now) / hop * hop,
                                off_sample: off,
                            }),
                    };
                    sink.lock().unwrap().push(Issued {
                        source_id: batch.source_id,
                        tone_id,
                        onset: recipe.onset,
                        issued_at: now,
                        kick: kick.strength,
                        energy,
                    });
                }
            }
        });
        let wiring = wire_runtime(
            &config,
            config.audio.sample_rate,
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

        // The decision that chose each onset supplies the representative span `D_rep`.
        let mut chosen = HashMap::new();
        for line in fs::read_to_string(&report_path).unwrap().lines() {
            let record: Value = serde_json::from_str(line).unwrap();
            if record["type"] != "participation_decision" || record["skipped"] == true {
                continue;
            }
            let at = record["selected_at"].as_f64().unwrap().round() as u64;
            chosen.insert(
                (record["voice_id"].as_u64().unwrap(), at),
                (
                    record["now"].as_u64().unwrap(),
                    record["footprint_d_samples"].as_f64().unwrap(),
                ),
            );
        }
        let mut out = BufWriter::new(File::create(directory.join("gap.jsonl")).unwrap());
        for issue in issued.lock().unwrap().iter() {
            let key = json!({"source_id": issue.source_id, "tone_id": issue.tone_id,
                "onset": issue.onset, "issued_at": issue.issued_at, "kick": issue.kick});
            let row = match chosen.get(&(issue.source_id, issue.onset)) {
                None => json!({"key": key, "status": "no_decision"}),
                Some(&(decided_at, d_rep)) => {
                    let window = project_window(
                        &[],
                        Some(([true, false], issue.energy)),
                        issue.onset,
                        (None, None),
                        0,
                        [issue.onset, issue.onset + d_rep.ceil() as u64],
                        true,
                    );
                    json!({"key": key, "status": if window.is_some() { "projected" } else { "no_window" },
                        "decided_at": decided_at, "d_rep": d_rep,
                        "coherent_energies": window.map(|w| w.coherent_energies)})
                }
            };
            serde_json::to_writer(&mut out, &row).unwrap();
            writeln!(out).unwrap();
        }
        out.flush().unwrap();
    }
}
