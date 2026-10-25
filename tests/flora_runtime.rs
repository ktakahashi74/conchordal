use std::fs;
use std::sync::{Arc, atomic::AtomicBool};
use std::time::{SystemTime, UNIX_EPOCH};

use conchordal::{app::run_render, config::AppConfig};

#[path = "support/test_output_directory.rs"]
mod test_output_directory;
use test_output_directory::TestOutputDirectory;

#[test]
fn flora_pace_uses_the_last_scene_basin_before_or_after_placement() {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "conchordal-flora-basin-{}-{id}",
        std::process::id()
    ));
    let _guard = TestOutputDirectory::new(dir.clone());
    for (first, last, expected_pace) in [(1, 3, 1.5), (3, 1, 0.5)] {
        for after_placement in [false, true] {
            let declaration = format!("temporal_basin({last}, {last});");
            let script = format!(
                "temporal_basin({first}, {first}); {} \
                     place(flora().fade(0), random(220, 880)); {} wait(0.04);",
                if after_placement { "" } else { &declaration },
                if after_placement { &declaration } else { "" },
            );
            let script_path = dir.join("input.rhai");
            fs::write(&script_path, script).unwrap();
            let report = dir.join("report.jsonl");
            let mut config = AppConfig::default();
            config.analysis.nfft = 2048;
            run_render(
                script_path.to_str().unwrap(),
                dir.join("audio.wav").to_string_lossy().into_owned(),
                config,
                Arc::new(AtomicBool::new(false)),
                Some(21),
                Some(report.to_str().unwrap()),
                0,
            )
            .unwrap();
            let rows: Vec<serde_json::Value> = fs::read_to_string(report)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .filter(|row: &serde_json::Value| row["type"] == "flora")
                .collect();
            assert!(!rows.is_empty());
            for row in rows {
                assert_eq!(
                    row["pace"].as_f64().unwrap(),
                    expected_pace,
                    "last={last}, after placement={after_placement}"
                );
            }
        }
    }
}

#[test]
fn flora_finishes_with_its_fade_in_both_renderers() {
    let id = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "conchordal-flora-runtime-{}-{id}",
        std::process::id()
    ));
    let _guard = TestOutputDirectory::new(dir.clone());
    for prototype in [false, true] {
        for (name, script, end) in [
            (
                "lone",
                "place(flora().amp(0.4).fade(0.1), random(220, 880)); wait(0.2);",
                0.3,
            ),
            (
                "residual_opt_in",
                "place(flora().amp(0.4).fade(0.1).residual(true), random(220, 880)); wait(0.2);",
                0.3,
            ),
            (
                "fade_in",
                "place(flora().amp(0.4).fade(1), random(220, 880)); wait(0.05);",
                1.05,
            ),
            (
                "explicit",
                "let m = place(flora().amp(0.4).fade(0.4), random(220, 880)); wait(0.1); release(m);",
                0.5,
            ),
            (
                "zero",
                "place(flora().amp(0.4).fade(0), random(220, 880)); wait(0.1);",
                0.1,
            ),
            (
                "early_flora",
                "place(sine().brain(\"drone\").sustain().amp(0.1), at(330)); let m = place(flora().amp(0.4).fade(0), random(220, 880)); wait(0.1); release(m); wait(0.2);",
                0.3,
            ),
        ] {
            let script_path = dir.join("input.rhai");
            fs::write(&script_path, script).unwrap();
            let wav = dir.join(format!("{name}-{prototype}.wav"));
            let report = dir.join(format!("{name}-{prototype}.jsonl"));
            let mut config = AppConfig::default();
            config.analysis.nfft = 2048;
            config.render_prototype = prototype;
            let hop_sec = config.analysis.hop_size as f64 / config.audio.sample_rate as f64;
            run_render(
                script_path.to_str().unwrap(),
                wav.to_string_lossy().into_owned(),
                config,
                Arc::new(AtomicBool::new(false)),
                Some(21),
                Some(report.to_str().unwrap()),
                0,
            )
            .unwrap();
            let mut reader = hound::WavReader::open(&wav).unwrap();
            let seconds = reader.duration() as f64 / reader.spec().sample_rate as f64;
            // Placement/Finish dispatch and release completion round to hop boundaries.
            assert!(seconds >= end, "{name}/{prototype}: {seconds} < {end}");
            if name != "early_flora" {
                assert!(
                    seconds <= end + 3.0 * hop_sec,
                    "{name}/{prototype}: {seconds}"
                );
            }
            let pcm: Vec<_> = reader.samples::<i16>().map(Result::unwrap).collect();
            assert!(pcm.iter().any(|x| *x != 0), "{name}/{prototype}: silent");
            assert!(
                pcm[pcm.len() - 512..].iter().all(|x| *x == 0),
                "{name}/{prototype}: tail cut"
            );
            let rows: Vec<serde_json::Value> = fs::read_to_string(report)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
            let flora: Vec<_> = rows.iter().filter(|r| r["type"] == "flora").collect();
            assert!(!flora.is_empty());
            assert_eq!(flora[0]["age_sec"].as_f64().unwrap(), hop_sec);
            assert!(flora[0]["tonal_energy"].as_f64().unwrap() > 0.);
            if name == "residual_opt_in" {
                assert!(
                    flora
                        .iter()
                        .any(|r| r["residual_energy"].as_f64().unwrap() > 0.)
                );
            } else {
                assert!(
                    flora
                        .iter()
                        .all(|r| r["residual_energy"].as_f64().unwrap() == 0.)
                );
            }
            for pair in flora.windows(2) {
                let before = pair[0];
                let after = pair[1];
                let now = after["time_sec"].as_f64().unwrap();
                assert!(now > before["time_sec"].as_f64().unwrap());
                if now.fract() < hop_sec + 1e-6 {
                    continue;
                }
                // Extra snapshots must document a lifecycle change between seconds.
                let states = |row: &serde_json::Value| {
                    row["lanes"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|lane| {
                            (
                                lane["slot"].as_u64().unwrap(),
                                lane["state"].as_str().unwrap().to_owned(),
                            )
                        })
                        .collect::<Vec<_>>()
                };
                assert!(
                    before["births"] != after["births"]
                        || before["retirements"] != after["retirements"]
                        || states(before) != states(after)
                );
            }
            assert_eq!(
                rows.iter()
                    .filter(|r| r["type"] == "rhythm_summary" && r["population_id"].is_null())
                    .count(),
                1
            );
        }
    }
}

#[test]
fn flora_alone_starts_shared_habituation_before_late_placement_and_reports_finish() {
    let dir = std::env::temp_dir().join(format!(
        "conchordal-flora-span-start-{}",
        std::process::id()
    ));
    let _guard = TestOutputDirectory::new(dir.clone());
    for weight in [0, 1] {
        let script = dir.join("input.rhai");
        fs::write(&script, format!("wait(1.1); place(flora().fade(0).rest_on_habituation({weight}), random(220,880)); wait(0.2);")).unwrap();
        let report = dir.join("report.jsonl");
        let mut config = AppConfig::default();
        config.analysis.nfft = 2048;
        run_render(
            script.to_str().unwrap(),
            dir.join("audio.wav").to_string_lossy().into_owned(),
            config,
            Arc::new(AtomicBool::new(false)),
            Some(21),
            Some(report.to_str().unwrap()),
            0,
        )
        .unwrap();
        let rows: Vec<serde_json::Value> = fs::read_to_string(report)
            .unwrap()
            .lines()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect();
        let early = rows.iter().any(|r| {
            r["type"] == "onset_habituation_scan"
                && r["known"] == true
                && r["time_sec"].as_f64().unwrap() < 1.1
        });
        assert_eq!(early, weight > 0);
        let spans: Vec<_> = rows
            .iter()
            .filter(|r| r["type"] == "span_gap" && r["flora_id"] == 1)
            .collect();
        if weight == 0 {
            assert!(spans.is_empty());
        } else {
            assert!(spans.iter().any(|r| r["state"] == "span"));
            assert_eq!(spans.last().unwrap()["state"], "finished");
            assert!(
                spans
                    .iter()
                    .all(|r| r.get("voice_id").is_none() && r["generation"] == 0)
            );
        }
    }
}
