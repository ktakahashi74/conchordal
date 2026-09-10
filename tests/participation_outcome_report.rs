use std::fs;
use std::process::Command;

use serde_json::Value;

#[test]
fn participation_outcomes_join_actions_to_habitat_audio_without_changing_the_sound() {
    let root = std::path::PathBuf::from(format!(
        "target/participation-outcome-report-{}",
        std::process::id()
    ));
    fs::create_dir_all(&root).unwrap();
    let config = root.join("config.toml");
    fs::write(&config, "[dcc]\ncoupling_strength = 0.0\n").unwrap();
    let report = root.join("report.jsonl");
    for enabled in [false, true] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_conchordal-render"));
        command
            .args(["tests/scripts/participation_outcome.rhai", "--config"])
            .arg(&config)
            .arg("--output")
            .arg(root.join(format!("report-{enabled}.wav")));
        if enabled {
            command.arg("--report").arg(&report);
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    assert_eq!(
        fs::read(root.join("report-false.wav")).unwrap(),
        fs::read(root.join("report-true.wav")).unwrap()
    );
    let records: Vec<Value> = fs::read_to_string(&report)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let outcomes: Vec<_> = records
        .iter()
        .filter(|r| r["type"] == "participation_outcome")
        .collect();
    assert!(
        outcomes.len() >= 5,
        "expected active participation: {}",
        outcomes.len()
    );
    let mut keys = std::collections::BTreeSet::new();
    for row in &outcomes {
        assert_eq!(row["status"], "observed");
        let frame = |key: &str| row[key].as_u64().unwrap();
        let fs = row["sample_rate"].as_f64().unwrap();
        assert!(frame("forecast_observed_frame") <= frame("issued_frame"));
        assert!(frame("issued_frame") <= frame("onset_frame"));
        assert!(frame("onset_frame") <= frame("target_start_frame"));
        assert_eq!(row["target_start_frame"], row["observed_start_frame"]);
        assert_eq!(row["target_end_frame"], row["observed_end_frame"]);
        assert!(keys.insert((frame("voice_id"), frame("onset_frame"))));
        assert!(records.iter().any(|r| r["type"] == "onset"
            && r["voice_id"] == row["voice_id"]
            && r["onset_frame"] == row["onset_frame"]
            && (r["time_sec"].as_f64().unwrap() * fs - frame("onset_frame") as f64).abs() < 1.0));
        if frame("target_end_frame") as f64 / fs < 1.9 {
            assert_eq!(
                row["observed_habitat_band_energy"],
                serde_json::json!([0.0, 0.0, 0.0])
            );
        }
    }
    assert!(outcomes.iter().any(|r| {
        r["observed_habitat_band_energy"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .sum::<f64>()
            > 0.001
    }));
    let contexts: Vec<_> = records
        .iter()
        .filter(|r| r["type"] == "participation_context")
        .collect();
    assert!(contexts.len() >= 3);
    let mut previous = std::collections::BTreeMap::<u64, [[f64; 3]; 2]>::new();
    for row in contexts {
        assert_eq!(row["status"], "observed");
        assert_eq!(row["energy_prediction_model"], "local_history_mix");
        let frame = |key: &str| row[key].as_u64().unwrap();
        assert!(frame("forecast_observed_frame") <= frame("onset_frame"));
        let history = &row["decision_external_history"];
        assert_eq!(history["post_order"], 12);
        assert_eq!(history["ages_sec"].as_array().unwrap().len(), 8);
        assert_eq!(
            history["known_band_rms_by_age"].as_array().unwrap().len(),
            8
        );
        for age in 0..8 {
            let coverage = history["known_coverage_by_age"][age].as_f64().unwrap();
            assert!((0.0..=1.0).contains(&coverage));
            for band in 0..3 {
                let rms = history["known_band_rms_by_age"][age][band]
                    .as_f64()
                    .unwrap();
                assert!(rms.is_finite() && rms >= 0.0);
                if frame("forecast_observed_frame") < 48_000 {
                    assert_eq!(rms, 0.0, "presentation-only sound entered habitat history");
                }
            }
        }
        let mut observed = [[0.0; 3]; 2];
        for (i, bands) in observed.iter_mut().enumerate() {
            let start = row["target_start_frames"][i].as_u64().unwrap();
            let end = row["target_end_frames"][i].as_u64().unwrap();
            assert!(frame("onset_frame") <= start && start < end);
            assert!(end <= frame("observed_through_frame"));
            for (band, energy) in bands.iter_mut().enumerate() {
                *energy = row["observed_external_band_energy"][i][band]
                    .as_f64()
                    .unwrap();
                assert!(energy.is_finite() && *energy >= 0.0);
            }
        }
        assert!(records.iter().any(|r| r["type"] == "onset"
            && r["voice_id"] == row["voice_id"]
            && r["onset_frame"] == row["onset_frame"]));
        let id = frame("voice_id");
        let expected = previous.get(&id).map_or(observed, |prior| {
            std::array::from_fn(|i| {
                std::array::from_fn(|b| prior[i][b] + 0.1 * (observed[i][b] - prior[i][b]))
            })
        });
        for (i, bands) in expected.iter().enumerate() {
            for (b, energy) in bands.iter().enumerate() {
                let memory = row["memory_external_band_energy"][i][b].as_f64().unwrap();
                assert!((memory - energy).abs() < 1e-6 * energy.abs().max(1e-5));
            }
        }
        previous.insert(id, expected);
    }
    let errors: Vec<_> = records
        .iter()
        .filter(|r| r["type"] == "local_prediction_error")
        .collect();
    assert!(!errors.is_empty());
    let matches: Vec<_> = records
        .iter()
        .filter(|r| r["type"] == "local_prediction_match")
        .collect();
    assert!(!matches.is_empty());
    for row in &matches {
        let observed = row["forecast_observed_frame"].as_u64().unwrap();
        let requested = row["requested_frame"].as_u64().unwrap();
        let end = row["target_end_frame"].as_u64().unwrap();
        assert!(observed <= requested && requested < end);
        assert_eq!(row["issued_features"].as_array().unwrap().len(), 57);
        for band in 0..3 {
            let w = row["history_weight"][band].as_f64().unwrap() as f32;
            let a = row["recurrence"][band].as_f64().unwrap() as f32;
            let b = row["history"][band].as_f64().unwrap() as f32;
            assert_eq!(
                row["mixed"][band].as_f64().unwrap() as f32,
                (1.0 - w) * a + w * b
            );
        }
    }
    let mut last_by_voice = std::collections::BTreeMap::new();
    let mut counts = [0u64; 7];
    let mut issued = 0;
    for row in &errors {
        let id = row["voice_id"].as_u64().unwrap();
        let from = row["observed_from_frame"].as_u64().unwrap();
        let through = row["observed_through_frame"].as_u64().unwrap();
        let window = row["window_frames"].as_u64().unwrap();
        assert!(from <= through && (through - from).is_multiple_of(window));
        issued += row["issued"].as_u64().unwrap();
        if let Some(last) = last_by_voice.insert(id, through) {
            assert!(from >= last);
        }
        for (i, lead) in [0, 5, 10, 25, 50, 100, 200].into_iter().enumerate() {
            assert_eq!(
                row["horizon_frames"][i].as_u64().unwrap(),
                lead * 2 * window
            );
            let n = row["completed"][i].as_u64().unwrap();
            let matched: Vec<_> = matches
                .iter()
                .filter(|m| {
                    m["voice_id"].as_u64() == Some(id)
                        && m["target_start_frame"].as_u64().unwrap() >= from
                        && m["target_end_frame"].as_u64().unwrap() <= through
                        && m["target_step"].as_u64().unwrap() - m["issued_step"].as_u64().unwrap()
                            == lead * 2
                })
                .collect();
            assert_eq!(n, matched.len() as u64);
            for model in ["recurrence", "history", "mixed"] {
                for band in 0..3 {
                    let expected: f64 = matched
                        .iter()
                        .map(|m| {
                            ((m[model][band].as_f64().unwrap() as f32) as f64
                                - (m["observed"][band].as_f64().unwrap() as f32) as f64)
                                .powi(2)
                        })
                        .sum();
                    let actual = row[format!("{model}_squared_error")][i][band]
                        .as_f64()
                        .unwrap();
                    assert!((actual - expected).abs() < 1e-12 * expected.max(1e-12));
                }
            }
            assert!(n <= (through - from) / window);
            counts[i] += n;
            for key in [
                "recurrence_squared_error",
                "history_squared_error",
                "mixed_squared_error",
            ] {
                for band in row[key][i].as_array().unwrap() {
                    let value = band.as_f64().unwrap();
                    assert!(value.is_finite() && value >= 0.0);
                    if n == 0 || through < 91_200 {
                        assert_eq!(value, 0.0);
                    }
                }
            }
        }
    }
    assert!(counts[0] > counts[6] && counts[6] > 0);
    assert_eq!(counts.iter().sum::<u64>(), matches.len() as u64);
    // Every 200 ms target finishes before retirement; most 4 s targets cannot.
    assert_eq!(counts[0], issued);
    assert_eq!(counts[2], issued);
    assert!(counts[6] < issued);
    let reader = hound::WavReader::open(root.join("report-true.wav")).unwrap();
    assert!(
        reader
            .into_samples::<i16>()
            .take(48000)
            .any(|x| x.unwrap().abs() > 100)
    );
    fs::remove_dir_all(root).unwrap();
}
