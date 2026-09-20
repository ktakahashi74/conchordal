//! Offline acquisition of real policy defaults, with serialized-input replay.

use super::*;
use sha2::{Digest, Sha256};

mod energy_projection;
mod local_continuation;
mod onset_branches;

#[derive(serde::Serialize, serde::Deserialize)]
struct Hop {
    now: u64,
    rhythms: NeuralRhythms,
    batches: Vec<PhonationBatch>,
    body: Option<Value>,
}

#[test]
#[ignore = "explicit offline policy-default acquisition; requires registered inputs and fresh output"]
fn acquire_policy_defaults() {
    let input = std::path::PathBuf::from(std::env::var("CONCHORDAL_I10_DEFAULT_INPUTS").unwrap());
    let output = std::path::PathBuf::from(std::env::var("CONCHORDAL_I10_DEFAULT_OUTPUT").unwrap());
    let registration: Value =
        serde_json::from_slice(&fs::read(input.join("registration.json")).unwrap()).unwrap();
    let issue = registration["issue_sample"].as_u64().unwrap();
    let horizon = registration["horizon_samples"].as_u64().unwrap();
    assert_eq!(registration["schema"], "i10-policy-default-acquisition-v1");
    assert_eq!((issue, horizon), (24576, HORIZON));
    assert_eq!(registration["sample_rate"], 48000);
    assert_eq!(registration["hop_samples"], 512);
    let config_path = input.join("config.toml");
    assert_eq!(
        format!("{:x}", Sha256::digest(fs::read(&config_path).unwrap())),
        registration["config_sha256"].as_str().unwrap()
    );
    let config = AppConfig::load_or_default(config_path.to_str().unwrap()).unwrap();
    config.validate().unwrap();
    assert_eq!(
        (config.audio.sample_rate, config.analysis.hop_size),
        (48000, 512)
    );
    crate::life::modal::register_modal();
    fs::create_dir(&output).expect("fresh acquisition directory required");
    let mut manifest = Vec::new();
    for case in registration["cases"].as_array().unwrap() {
        assert_eq!(case["split"], "development");
        let id = case["id"].as_str().unwrap();
        let directory = output.join(id);
        fs::create_dir(&directory).unwrap();
        let script = input.join(case["script"].as_str().unwrap());
        assert_eq!(
            format!("{:x}", Sha256::digest(fs::read(&script).unwrap())),
            case["script_sha256"].as_str().unwrap()
        );
        let scenario = compile_scenario_from_script(
            &script,
            &render_compile_args(script.to_str().unwrap(), None),
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
        let mut fork = None;
        let mut local = None;
        let acquire_local = registration.get("local_continuation").is_some();
        let mut pcm = [Vec::new(), Vec::new()];
        let mut hops = Vec::new();
        let probe: OfflineBodyProbe = Box::new(move |state, now, count, actual| {
            if now >= issue + horizon {
                return;
            }
            assert!(
                state.pop.voices.len() <= 1,
                "private target must contain only one source"
            );
            let batches = state.phonation_batches_buf[..count].to_vec();
            assert!(
                batches
                    .iter()
                    .all(|b| b.source_id == 1 && b.source_generation == 0)
            );
            let body = state.pop.voices.first().map(|voice| {
                assert_eq!((voice.id, voice.metadata.generation), (1, 0));
                json!({"source_id": voice.id, "source_generation": voice.metadata.generation,
                    "body_generation": state.schedule_renderer.body_capture.as_ref()
                        .and_then(|c| c.token(voice.id, voice.metadata.generation)).map(|(_, generation)| generation),
                    "snapshot": voice.body_snapshot(), "pitch_hz": voice.body.base_freq_hz(),
                    "target_amp": voice.compute_target_amp(),
                    "continuous_drive": voice.effective_control.body.continuous_drive,
                    "routing": voice.effective_control.body.routing})
            });
            hops.push(Hop {
                now,
                rhythms: state.current_landscape.rhythm,
                batches,
                body,
            });
            for bus in 0..2 {
                pcm[bus].extend_from_slice(actual[bus]);
            }
            if now + 512 == issue {
                fork = Some(state.schedule_renderer.fork_source(1));
                if acquire_local {
                    local = Some(local_continuation::Snapshot::capture(state, issue));
                }
            }
            if now + 512 == issue + horizon {
                tx.send((
                    fork.take().expect("issue renderer captured"),
                    std::mem::take(&mut pcm),
                    std::mem::take(&mut hops),
                    local.take(),
                ))
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
        let (mut frozen, actual, original, local) = rx.recv().unwrap();
        let trace_path = directory.join("policy-inputs.jsonl");
        let mut trace = BufWriter::new(File::create(&trace_path).unwrap());
        for hop in &original {
            // Match metadata's exact f32-to-f64 JSON numbers, not short f32 spellings.
            serde_json::to_writer(&mut trace, &serde_json::to_value(hop).unwrap()).unwrap();
            writeln!(trace).unwrap();
        }
        trace.flush().unwrap();
        drop(trace);
        drop(original);
        // Read the saved representation, not an in-memory clone of the input commands.
        let restored: Vec<Hop> = fs::read_to_string(&trace_path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let mut replay = ScheduleRenderer::new(time);
        let mut no_commands = frozen.fork_source(1);
        let mut replay_pcm = [Vec::new(), Vec::new()];
        let mut default_pcm = [Vec::new(), Vec::new()];
        let mut no_commands_pcm = [Vec::new(), Vec::new()];
        let mut future_commands = [0usize; 3];
        let mut first_onset = None;
        for (index, hop) in restored.iter().enumerate() {
            assert_eq!(hop.now, index as u64 * 512);
            if hop.now == issue
                && let Some(snapshot) = &local
            {
                local_continuation::acquire(
                    snapshot,
                    &frozen,
                    &replay,
                    &directory,
                    &registration["local_continuation"],
                );
            }
            let rendered = replay.render(&hop.batches, hop.now, &hop.rhythms);
            for (bus, samples) in [rendered.habitat, rendered.presentation]
                .into_iter()
                .enumerate()
            {
                assert_eq!(
                    samples,
                    &actual[bus][index * 512..(index + 1) * 512],
                    "serialized prefix/default replay: {id}/{bus}/{}",
                    hop.now
                );
                replay_pcm[bus].extend_from_slice(samples);
            }
            if hop.now < issue {
                continue;
            }
            for batch in &hop.batches {
                for command in &batch.cmds {
                    match command {
                        ToneCmd::On { tone_id, .. } => {
                            future_commands[0] += 1;
                            let spec = batch
                                .tones
                                .iter()
                                .find(|tone| tone.tone_id == *tone_id)
                                .unwrap();
                            if first_onset.is_none() {
                                first_onset = Some(json!({"issued_at": hop.now, "tone": spec,
                                    "source_generation": batch.source_generation, "routing": batch.routing,
                                    "policy": batch.body_policy, "opportunity": batch.body_opportunity,
                                    "intrinsic_period_sec": batch.intrinsic_period_sec}));
                            }
                        }
                        ToneCmd::Off { .. } => future_commands[1] += 1,
                        ToneCmd::Update { .. } => future_commands[2] += 1,
                    }
                }
            }
            let frame = frozen.render(&hop.batches, hop.now, &hop.rhythms);
            for (bus, samples) in [frame.habitat, frame.presentation].into_iter().enumerate() {
                assert_eq!(
                    samples,
                    &actual[bus][index * 512..(index + 1) * 512],
                    "frozen default replay: {id}/{bus}/{}",
                    hop.now
                );
                default_pcm[bus].extend_from_slice(samples);
            }
            let frame = no_commands.render(&[], hop.now, &hop.rhythms);
            no_commands_pcm[0].extend_from_slice(frame.habitat);
            no_commands_pcm[1].extend_from_slice(frame.presentation);
        }
        assert_eq!(restored.len() * 512, (issue + horizon) as usize);
        let issue_input = serde_json::to_value(&restored[issue as usize / 512]).unwrap();
        let mut buses = Vec::new();
        for bus in 0..2 {
            let future = &actual[bus][issue as usize..];
            write_pcm(
                &directory.join(format!("prefix-bus{bus}.f32le")),
                &actual[bus][..issue as usize],
            );
            write_pcm(
                &directory.join(format!("actual-default-bus{bus}.f32le")),
                future,
            );
            write_pcm(
                &directory.join(format!("serialized-replay-bus{bus}.f32le")),
                &replay_pcm[bus][issue as usize..],
            );
            write_pcm(
                &directory.join(format!("frozen-default-bus{bus}.f32le")),
                &default_pcm[bus],
            );
            write_pcm(
                &directory.join(format!("no-commands-bus{bus}.f32le")),
                &no_commands_pcm[bus],
            );
            let energy = [future, no_commands_pcm[bus].as_slice()].map(|samples| {
                samples
                    .chunks_exact(512)
                    .map(|chunk| chunk.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / 512.)
                    .collect::<Vec<_>>()
            });
            buses.push(json!({"bus": bus, "energy_per_hop": energy,
                "different_samples_from_no_commands": future.iter().zip(&no_commands_pcm[bus]).filter(|(a,b)| a.to_bits()!=b.to_bits()).count()}));
        }
        let metadata = json!({"schema": "i10-real-policy-default-v1", "case": case,
            "issue_sample": issue, "horizon_samples": horizon, "sample_rate": 48000, "hop_samples": 512,
            "issue_input": issue_input, "future_commands": future_commands, "first_future_onset": first_onset,
            "replay": "serialized full prefix and future, plus exact issue-state fork; both buses sample-exact",
            "future_input_policy": "later actual policy commands and rhythms are target data, never issue-time predictor input",
            "control": "no additional commands after issue; same actual rhythm sequence, not the default",
            "buses": buses});
        fs::write(
            directory.join("default.json"),
            serde_json::to_vec_pretty(&metadata).unwrap(),
        )
        .unwrap();
        println!(
            "{id}: {} hops, future On/Off/Update={future_commands:?}; default replay exact",
            restored.len()
        );
        manifest.push(json!({"id": id, "future_commands": future_commands}));
    }
    fs::write(output.join("manifest.json"), serde_json::to_vec_pretty(&json!({
        "schema": "i10-real-policy-default-corpus-v1", "registration": registration,
        "cases": manifest, "scope": "actual default target acquisition; not seven-class counterfactuals or transfer validation"
    })).unwrap()).unwrap();
}
