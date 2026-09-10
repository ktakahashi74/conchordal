use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

// Test bounded, reproducible erosion and controlled recovery at fixed Log2Space
// coordinates. Re-exposure is scripted; autonomous return remains a separate
// causal research assay and is not inferred from the moving `tracked_bin`.

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

fn run(config: &PathBuf, report: &PathBuf, scenario: &str) {
    let status = Command::new(env!("CARGO_BIN_EXE_conchordal"))
        .arg(scenario)
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
    run(&cfg, &rep, SCENARIO);
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
    run(&cfg, &rep, SCENARIO);
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
    run(&cfg, &r1, SCENARIO);
    run(&cfg, &r2, SCENARIO);
    let s1 = hab_series(&r1);
    let s2 = hab_series(&r2);
    // hab_series drops unparseable lines, so an empty series would make the
    // comparison below vacuously true. Guard before comparing.
    assert!(
        !s1.is_empty(),
        "habituation series is empty — the report format likely drifted"
    );
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

#[test]
fn a_fixed_region_recovers_after_withdrawal_and_responds_to_reexposure() {
    let mut traces = Vec::new();
    for condition in ["off", "normal", "slow"] {
        let cfg = write_config(condition != "off");
        if condition == "slow" {
            std::fs::write(
                &cfg,
                "[psychoacoustics.habituation]\nenabled = true\nrecovery_sec = 80.0\n",
            )
            .unwrap();
        }
        let report = temp_path("fixed-region.jsonl");
        run(
            &cfg,
            &report,
            "samples/research/habituation_recovery_probe.rhai",
        );
        let rows: Vec<serde_json::Value> = std::fs::read_to_string(&report)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("valid JSONL"))
            .filter(|row| row["type"] == "habituation_scan")
            .collect();
        assert!(rows.len() >= 36, "missing one-second observations");
        let reference = &rows[0];
        let fmin = reference["fmin_hz"].as_f64().unwrap();
        let bins_per_octave = reference["bins_per_octave"].as_f64().unwrap();
        let bin = ((220.0 / fmin).log2() * bins_per_octave).round() as usize;
        for row in &rows {
            assert_eq!(row["fmin_hz"], reference["fmin_hz"]);
            assert_eq!(row["bins_per_octave"], reference["bins_per_octave"]);
            assert_eq!(row["n_bins"], reference["n_bins"]);
            for key in ["state_scan", "raw_score_scan", "eff_score_scan"] {
                assert_eq!(
                    row[key].as_array().unwrap().len(),
                    row["n_bins"].as_u64().unwrap() as usize
                );
            }
            if condition == "off" {
                assert!(
                    row["state_scan"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .all(|h| h.as_f64() == Some(0.0))
                );
                assert_eq!(row["raw_score_scan"], row["eff_score_scan"]);
            }
        }
        let state_at = |sec: f64| {
            rows.iter()
                .find(|row| {
                    let time = row["time_sec"].as_f64().unwrap();
                    time >= sec && time < sec + 0.03
                })
                .expect("reference time observed")["state_scan"][bin]
                .as_f64()
                .unwrap()
        };
        traces.push([state_at(9.0), state_at(25.0), state_at(35.0)]);
        std::fs::remove_file(report).unwrap();
        std::fs::remove_file(cfg).unwrap();
    }
    let [exposed, recovered, reexposed] = traces[1];
    assert!(exposed > 0.15, "weak initial exposure: {traces:?}");
    assert!(
        recovered < exposed * 0.15,
        "fixed region did not recover: {traces:?}"
    );
    assert!(
        reexposed > exposed * 0.7,
        "fixed region did not respond again: {traces:?}"
    );
    assert!(
        traces[2][1] > traces[2][0] * 0.5,
        "slow recovery did not retain state: {traces:?}"
    );
    assert!(
        traces[2][1] > recovered * 3.0,
        "recovery-rate intervention had no effect: {traces:?}"
    );
}
