use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

// Deterministic habituation-field assay. Asserts the robust closure signatures:
// causal control (off vs on), erosion occurs, erosion stays bounded (no global
// runaway), and reproducibility. Single-basin recovery/return is NOT asserted
// here: the `tracked_bin` telemetry is the per-hop argmax of the raw score and
// moves between bins, so it cannot cleanly measure one basin recovering. The
// full closure verdict (recovery + return-to-vacated-basin) is the manual
// research campaign recorded in the design docs.

const SCENARIO: &str = "samples/research/habituation_field_assay.rhai";

fn temp_path(tag: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    p.push(format!("conchordal_hab_{nanos}_{tag}"));
    p
}

fn write_config(enabled: bool) -> PathBuf {
    let path = temp_path("cfg.toml");
    let mut f = std::fs::File::create(&path).expect("create config");
    write!(
        f,
        "[psychoacoustics.habituation]\nenabled = {enabled}\nsatiation_sec = 5.0\nrecovery_sec = 8.0\nref_drive = 0.25\n"
    )
    .expect("write config");
    path
}

fn run(config: &PathBuf, report: &PathBuf) {
    let status = Command::new(env!("CARGO_BIN_EXE_conchordal"))
        .arg(SCENARIO)
        .arg("--config")
        .arg(config)
        .args(["--nogui", "--play=false", "--report"])
        .arg(report)
        .status()
        .expect("run conchordal");
    assert!(status.success(), "conchordal exited nonzero");
}

// (mean_h, max_h) per habituation record, in order.
fn hab_series(report: &PathBuf) -> Vec<(f32, f32)> {
    let text = std::fs::read_to_string(report).expect("read report");
    let mut out = Vec::new();
    for line in text.lines() {
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if v["type"] == "habituation" {
            out.push((
                v["mean_h"].as_f64().expect("mean_h f64") as f32,
                v["max_h"].as_f64().expect("max_h f64") as f32,
            ));
        }
    }
    out
}

#[test]
fn habituation_off_is_the_causal_control() {
    let cfg = write_config(false);
    let rep = temp_path("off.jsonl");
    run(&cfg, &rep);
    let s = hab_series(&rep);
    assert!(!s.is_empty(), "no habituation records emitted");
    for (mean_h, max_h) in &s {
        assert!(
            *mean_h < 1e-6 && *max_h < 1e-6,
            "disabled path must keep h==0, got mean={mean_h} max={max_h}"
        );
    }
    let _ = std::fs::remove_file(&rep);
    let _ = std::fs::remove_file(&cfg);
}

#[test]
fn habituation_on_erodes_and_stays_bounded() {
    let cfg = write_config(true);
    let rep = temp_path("on.jsonl");
    run(&cfg, &rep);
    let s = hab_series(&rep);
    assert!(s.len() > 1000, "expected a long series, got {}", s.len());
    let peak_max_h = s.iter().map(|(_, x)| *x).fold(0.0f32, f32::max);
    assert!(
        peak_max_h > 0.4,
        "habituation-on should erode (peak max_h > 0.4), got {peak_max_h}"
    );
    let peak_mean_h = s.iter().map(|(m, _)| *m).fold(0.0f32, f32::max);
    assert!(
        peak_mean_h < 0.4,
        "erosion must stay localized/bounded (mean_h < 0.4), got {peak_mean_h}"
    );
    let _ = std::fs::remove_file(&rep);
    let _ = std::fs::remove_file(&cfg);
}

#[test]
fn habituation_is_deterministic() {
    let cfg = write_config(true);
    let r1 = temp_path("det1.jsonl");
    let r2 = temp_path("det2.jsonl");
    run(&cfg, &r1);
    run(&cfg, &r2);
    let s1 = hab_series(&r1);
    let s2 = hab_series(&r2);
    assert_eq!(s1.len(), s2.len(), "record counts differ across runs");
    for (i, (a, b)) in s1.iter().zip(s2.iter()).enumerate() {
        assert_eq!(
            a, b,
            "habituation series diverged at record {i}: {a:?} vs {b:?}"
        );
    }
    let _ = std::fs::remove_file(&r1);
    let _ = std::fs::remove_file(&r2);
    let _ = std::fs::remove_file(&cfg);
}
