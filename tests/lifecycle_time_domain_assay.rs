use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

fn unique_report_path() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "conchordal-lifecycle-it-{}-{nanos}.jsonl",
        std::process::id()
    ))
}

#[test]
fn endurance_report_matches_energy_depletion_and_observable_lifetime() {
    let report_path = unique_report_path();
    let scenario = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("samples/research/lifecycle_time_domain_assay.rhai");
    let output = Command::new(env!("CARGO_BIN_EXE_conchordal"))
        .arg(scenario)
        .args(["--nogui", "--play=false", "--report"])
        .arg(&report_path)
        .output()
        .expect("run lifecycle assay");
    assert!(
        output.status.success(),
        "lifecycle assay failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let report = fs::read_to_string(&report_path).expect("read lifecycle report");
    let mut deaths: Vec<(f64, f64, f64)> = report
        .lines()
        .filter_map(|line| {
            let record: Value = serde_json::from_str(line).expect("valid report JSON");
            if record["type"] != "death" {
                return None;
            }
            Some((
                record["configured_endurance_sec"]
                    .as_f64()
                    .expect("configured endurance"),
                record["energy_depletion_sec"]
                    .as_f64()
                    .expect("energy depletion"),
                record["lifetime_sec"]
                    .as_f64()
                    .expect("observable lifetime"),
            ))
        })
        .collect();
    deaths.sort_by(|a, b| a.0.total_cmp(&b.0));

    assert_eq!(deaths.len(), 3);
    for ((configured, depleted, lifetime), expected) in deaths.into_iter().zip([2.0, 4.0, 8.0]) {
        assert!((configured - expected).abs() < 1e-6);
        assert!(
            (depleted - expected).abs() <= 0.02,
            "configured={configured} depleted={depleted}"
        );
        assert!(lifetime >= depleted);
        assert!(
            lifetime - depleted <= 0.02,
            "depleted={depleted} lifetime={lifetime}"
        );
    }

    let _ = fs::remove_file(report_path);
}
