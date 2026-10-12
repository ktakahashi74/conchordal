use std::fs;
use std::process::Command;

#[test]
fn intrinsic_rate_renders_bodies_reproducibly_and_stops_on_release() {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "conchordal-intrinsic-rate-{}-{stamp}",
        std::process::id()
    ));
    fs::create_dir(&dir).unwrap();
    for prototype in [false, true] {
        let config = dir.join(format!("config-{prototype}.toml"));
        fs::write(
            &config,
            format!("render_prototype = {prototype}\n[analysis]\nnfft = 2048\nhop_size = 512\n"),
        )
        .unwrap();
        for body in ["sine", "harmonic", "modal"] {
            {
                let end = "release";
                let name = format!("{body}-{prototype}-{end}");
                let scene = dir.join(format!("{name}.rhai"));
                fs::write(&scene,format!(
                    "let p=place({body}().amp(0.045).brain(\"drone\").sustain().entrained().entrainment(0.05).intrinsic_rate(4).pitch_apply_mode(\"gate_snap\"),at(220.0)); wait(2.0); {end}(p); wait(2.0);"
                )).unwrap();
                let mut acquired = Vec::new();
                for repeat in 0..2 {
                    let wav = dir.join(format!("{name}-{repeat}.wav"));
                    let report = dir.join(format!("{name}-{repeat}.jsonl"));
                    let output = Command::new(env!("CARGO_BIN_EXE_conchordal-render"))
                        .arg(&scene)
                        .arg("--config")
                        .arg(&config)
                        .args(["--seed", "21", "-o"])
                        .arg(&wav)
                        .arg("--report")
                        .arg(&report)
                        .env("RUST_LOG", "warn")
                        .output()
                        .unwrap();
                    assert!(
                        output.status.success(),
                        "{name}: {}",
                        String::from_utf8_lossy(&output.stderr)
                    );
                    let pcm: Vec<i16> = hound::WavReader::open(&wav)
                        .unwrap()
                        .samples::<i16>()
                        .map(Result::unwrap)
                        .collect();
                    assert!(pcm.iter().any(|x| *x != 0), "{name}: no radiated sound");
                    let semantic: Vec<serde_json::Value> = fs::read_to_string(&report)
                        .unwrap()
                        .lines()
                        .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
                        .filter(|row| row["type"] != "hop_timing")
                        .collect();
                    let onsets: Vec<_> = semantic
                        .iter()
                        .filter(|row| row["type"] == "onset")
                        .collect();
                    assert!(onsets.len() > 1, "{name}: no participation onsets");
                    // The scheduled action is applied at the next 512-frame hop boundary.
                    let cutoff = (2 * 48_000usize).div_ceil(512) * 512;
                    assert!(
                        onsets
                            .iter()
                            .all(|row| row["onset_frame"].as_u64().unwrap() < cutoff as u64),
                        "{name}: onset after {end}"
                    );
                    acquired.push((fs::read(wav).unwrap(), semantic));
                }
                assert_eq!(acquired[0], acquired[1], "{name}: seeded output changed");
            }
        }
    }
    fs::remove_dir_all(dir).unwrap();
}
