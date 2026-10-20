use std::{
    fs,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

#[test]
fn passive_observation_preserves_offline_audio_and_reports_both_streams() {
    let directory =
        std::env::temp_dir().join(format!("conchordal-info-terrain-{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    let scenario = directory.join("scenario.rhai");
    fs::write(
        &scenario,
        r#"
seed(20261020);
let sun = place(harmonic().brain("drone").amp(0.05).sustain().anchor(), at(110.0));
wait(0.5);
let group = place(sine().amp(0.03).sustain().seek_consonance(), consonance(220.0, 880.0).count(3));
wait(2.0);
release(group);
release(sun);
wait(0.7);
"#,
    )
    .unwrap();
    for report in [false, true] {
        let mut reference = None;
        let mut a2_reference = std::collections::BTreeMap::new();
        for (enabled, trace, efe) in [
            (false, false, false),
            (false, true, false),
            (true, false, false),
            (true, true, false),
            (false, false, true),
            (true, false, true),
            (true, true, true),
        ] {
            let config = directory.join("config.toml");
            fs::write(&config, format!("[audio]\nsample_rate = 48000\n[analysis]\nnfft = 2048\nhop_size = 512\n[dcc]\ncoupling_strength = 0.0\n[info_terrain]\nenabled = {enabled}\ntrace = {trace}\nefe = {efe}\n")).unwrap();
            let wav = directory.join("render.wav");
            let jsonl = directory.join("report.jsonl");
            let stderr = directory.join("stderr.txt");
            let mut command = Command::new(env!("CARGO_BIN_EXE_conchordal-render"));
            command
                .env("RUST_LOG", "warn")
                .arg(&scenario)
                .arg("--config")
                .arg(&config)
                .arg("-o")
                .arg(&wav)
                .stdout(Stdio::null())
                .stderr(fs::File::create(&stderr).unwrap());
            if report {
                command.arg("--report").arg(&jsonl);
            }
            let mut child = command.spawn().unwrap();
            let deadline = Instant::now() + Duration::from_secs(90);
            let status = loop {
                if let Some(status) = child.try_wait().unwrap() {
                    break status;
                }
                if Instant::now() >= deadline {
                    child.kill().unwrap();
                    child.wait().unwrap();
                    panic!("render deadlocked: report={report}, enabled={enabled}, trace={trace}");
                }
                thread::sleep(Duration::from_millis(20));
            };
            assert!(status.success(), "{}", fs::read_to_string(&stderr).unwrap());
            let audio = fs::read(&wav).unwrap();
            if let Some(reference) = &reference {
                assert_eq!(
                    &audio, reference,
                    "report={report}, enabled={enabled}, trace={trace}"
                );
            } else {
                reference = Some(audio);
            }
            if !report {
                continue;
            }
            let records: Vec<serde_json::Value> = fs::read_to_string(&jsonl)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            let blocks: Vec<_> = records
                .iter()
                .filter(|r| r["type"] == "info_terrain_block")
                .collect();
            let summaries: Vec<_> = records
                .iter()
                .enumerate()
                .filter(|(_, r)| r["type"] == "info_terrain_summary")
                .collect();
            if !enabled {
                assert!(blocks.is_empty() && summaries.is_empty());
                continue;
            }
            let mut a2_blocks: Vec<serde_json::Value> =
                blocks.iter().map(|b| (*b).clone()).collect();
            for block in &mut a2_blocks {
                assert_eq!(block.get("efe").is_some(), efe);
                block.as_object_mut().unwrap().remove("efe");
            }
            a2_blocks.sort_by(|a, b| {
                a["stream"]
                    .as_str()
                    .cmp(&b["stream"].as_str())
                    .then_with(|| a["block_id"].as_u64().cmp(&b["block_id"].as_u64()))
            });
            if efe {
                assert!(
                    a2_blocks == a2_reference[&trace],
                    "A-3 changed A-2 reports: trace={trace}"
                );
            } else {
                a2_reference.insert(trace, a2_blocks);
            }
            assert_eq!(summaries.len(), 1);
            assert_eq!(summaries[0].1["dropped_blocks"], 0);
            assert_eq!(summaries[0].0, records.len() - 1);
            for stream in ["habitat", "presentation"] {
                let stream_blocks: Vec<_> =
                    blocks.iter().filter(|b| b["stream"] == stream).collect();
                assert!(stream_blocks.len() >= 3, "missing blocks for {stream}");
                let mut next_hop = 0;
                for (id, block) in stream_blocks.iter().enumerate() {
                    assert_eq!(block["block_id"], id as u64);
                    assert_eq!(block["hop_start"], next_hop);
                    let first = block["start_sample"].as_u64().unwrap();
                    assert_eq!(first / 48000, id as u64);
                    next_hop = block["hop_end_exclusive"].as_u64().unwrap();
                    assert_eq!(block["input_gap"], false);
                    assert_eq!(block.get("trace").is_some(), trace);
                    if trace {
                        assert_eq!(
                            block["trace"].as_array().unwrap().len() as u64,
                            block["hops"].as_u64().unwrap()
                        );
                    }
                    for version in ["original", "extended"] {
                        assert_eq!(
                            block[version]["surprise_mean_scan"]
                                .as_array()
                                .unwrap()
                                .len(),
                            block[version]["precision_mean_scan"]
                                .as_array()
                                .unwrap()
                                .len()
                        );
                    }
                }
                assert_eq!(stream_blocks.last().unwrap()["incomplete"], true);
                assert!(
                    stream_blocks
                        .iter()
                        .any(|b| b["original"]["soundgen"]["defined_hops"].as_u64().unwrap() > 0)
                );
                assert!(
                    stream_blocks
                        .iter()
                        .any(|b| b["extended"]["soundgen"]["defined_hops"].as_u64().unwrap() > 0)
                );
            }
        }
    }
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn info_terrain_config_is_default_off_and_rejects_unknown_fields() {
    let defaults = conchordal::config::AppConfig::default();
    assert!(
        !defaults.info_terrain.enabled
            && !defaults.info_terrain.trace
            && !defaults.info_terrain.efe
    );
    let parsed: conchordal::config::AppConfig =
        toml::from_str("[info_terrain]\nenabled = true\ntrace = true\nefe = true\n").unwrap();
    assert!(parsed.info_terrain.enabled && parsed.info_terrain.trace && parsed.info_terrain.efe);
    let mut too_short = parsed;
    too_short.analysis.nfft = 65536;
    too_short.analysis.hop_size = 32768;
    assert!(
        too_short
            .validate()
            .unwrap_err()
            .to_string()
            .contains("info_terrain")
    );
    too_short.info_terrain.enabled = false;
    too_short.validate().unwrap();
    assert!(
        toml::from_str::<conchordal::config::AppConfig>("[info_terrain]\ntrcae = true\n").is_err()
    );
}
