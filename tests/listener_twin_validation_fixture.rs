use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

#[derive(Clone, Copy)]
struct Window {
    start_sec: f64,
    end_sec: f64,
}

#[derive(Default)]
struct ListenerStats {
    n: usize,
    stability_sum: f64,
    resolvability_sum: f64,
    tension_sum: f64,
    attention_sum: f64,
    max_tension: f64,
}

#[derive(Default)]
struct PressureStats {
    n: usize,
    max_tension_pressure: f64,
    max_exploration_bonus: f64,
}

impl PressureStats {
    fn push(&mut self, record: &Value) {
        self.n += 1;
        self.max_tension_pressure = self
            .max_tension_pressure
            .max(record["tension_pressure"].as_f64().unwrap_or(0.0));
        self.max_exploration_bonus = self
            .max_exploration_bonus
            .max(record["exploration_bonus"].as_f64().unwrap_or(0.0));
    }
}

impl ListenerStats {
    fn push(&mut self, record: &Value) {
        self.n += 1;
        self.stability_sum += record["stability_level"].as_f64().unwrap_or(0.0);
        self.resolvability_sum += record["resolvability_level"].as_f64().unwrap_or(0.0);
        self.tension_sum += record["tension_level"].as_f64().unwrap_or(0.0);
        self.attention_sum += record["attention_level"].as_f64().unwrap_or(0.0);
        self.max_tension = self
            .max_tension
            .max(record["tension_level"].as_f64().unwrap_or(0.0));
    }

    fn avg_stability(&self) -> f64 {
        self.stability_sum / self.n.max(1) as f64
    }

    fn avg_resolvability(&self) -> f64 {
        self.resolvability_sum / self.n.max(1) as f64
    }

    fn avg_tension(&self) -> f64 {
        self.tension_sum / self.n.max(1) as f64
    }

    fn avg_attention(&self) -> f64 {
        self.attention_sum / self.n.max(1) as f64
    }
}

fn report_path() -> PathBuf {
    unique_target_path("listener_twin_validation", "jsonl")
}

fn config_path() -> PathBuf {
    unique_target_path("listener_twin_validation_config", "toml")
}

fn unique_target_path(name: &str, ext: &str) -> PathBuf {
    let now_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time after epoch")
        .as_nanos();
    PathBuf::from(format!(
        "target/{name}_{}_{}.{}",
        std::process::id(),
        now_ns,
        ext
    ))
}

fn collect_stats(report: &str, window: Window) -> ListenerStats {
    let mut stats = ListenerStats::default();
    for line in report.lines() {
        let record: Value = serde_json::from_str(line).expect("valid report JSON");
        if record["type"] != "listener_state" {
            continue;
        }
        let time_sec = record["time_sec"].as_f64().unwrap_or(-1.0);
        if time_sec >= window.start_sec && time_sec < window.end_sec {
            stats.push(&record);
        }
    }
    stats
}

fn collect_pressure_stats(report: &str) -> PressureStats {
    let mut stats = PressureStats::default();
    for line in report.lines() {
        let record: Value = serde_json::from_str(line).expect("valid report JSON");
        if record["type"] == "dcc_pressure" {
            stats.push(&record);
        }
    }
    stats
}

#[test]
fn listener_twin_tension_resolution_fixture_reports_expected_shape() {
    let exe = env!("CARGO_BIN_EXE_conchordal");
    let report_path = report_path();
    let config_path = config_path();
    fs::write(
        &config_path,
        "[dcc]\ncoupling_strength = 0.0\nmax_exploration_bonus = 0.1\n",
    )
    .expect("write test config");
    let output = Command::new(exe)
        .args([
            "--nogui",
            "--play=false",
            "--config",
            config_path.to_str().expect("utf8 config path"),
            "--report",
            report_path.to_str().expect("utf8 report path"),
            "tests/scripts/listener_twin_tension_resolution.rhai",
        ])
        .output()
        .expect("run tension validation fixture");

    if !output.status.success() {
        panic!(
            "fixture failed: status={} stderr={}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let report = fs::read_to_string(&report_path).expect("read report");
    let _ = fs::remove_file(&report_path);
    let _ = fs::remove_file(&config_path);

    let hidden = collect_stats(
        &report,
        Window {
            start_sec: 0.40,
            end_sec: 1.20,
        },
    );
    let stable = collect_stats(
        &report,
        Window {
            start_sec: 2.10,
            end_sec: 3.30,
        },
    );
    let tension = collect_stats(
        &report,
        Window {
            start_sec: 4.05,
            end_sec: 4.95,
        },
    );
    let resolved = collect_stats(
        &report,
        Window {
            start_sec: 5.85,
            end_sec: 6.25,
        },
    );
    let settled = collect_stats(
        &report,
        Window {
            start_sec: 7.00,
            end_sec: 8.30,
        },
    );
    let pressure = collect_pressure_stats(&report);

    assert!(hidden.n > 20, "hidden window has no listener samples");
    assert!(stable.n > 20, "stable window has no listener samples");
    assert!(tension.n > 20, "tension window has no listener samples");
    assert!(resolved.n > 10, "resolved window has no listener samples");
    assert!(settled.n > 20, "settled window has no listener samples");
    assert!(
        pressure.n > 20,
        "dcc_pressure should be reported beside listener_state"
    );
    assert!(
        pressure.max_tension_pressure < 1.0e-6 && pressure.max_exploration_bonus < 1.0e-6,
        "default DCC coupling should be report-only: pressure={} bonus={}",
        pressure.max_tension_pressure,
        pressure.max_exploration_bonus
    );

    assert!(
        (hidden.avg_stability() - 0.5).abs() < 1.0e-6,
        "hidden habitat must not change listener stability: {}",
        hidden.avg_stability()
    );
    assert!(
        hidden.avg_resolvability() < 1.0e-6 && hidden.avg_tension() < 1.0e-6,
        "hidden habitat leaked into listener tension: res={} ten={}",
        hidden.avg_resolvability(),
        hidden.avg_tension()
    );
    assert!(
        hidden.avg_attention() < 1.0e-6,
        "hidden habitat leaked into listener salience: {}",
        hidden.avg_attention()
    );

    assert!(
        stable.avg_stability() > 0.75,
        "stable presentation should have high stability: {}",
        stable.avg_stability()
    );
    assert!(
        stable.avg_tension() < 0.005,
        "stable presentation should have low tension: {}",
        stable.avg_tension()
    );

    assert!(
        tension.avg_resolvability() > 0.04,
        "tension cluster should expose resolution affordance: {}",
        tension.avg_resolvability()
    );
    assert!(
        tension.max_tension > 0.03,
        "tension cluster should raise listener tension: max={}",
        tension.max_tension
    );
    assert!(
        resolved.avg_stability() > 0.75,
        "resolved presentation should return to high stability: {}",
        resolved.avg_stability()
    );
    assert!(
        resolved.avg_tension() < tension.avg_tension() * 0.5,
        "resolution should lower tension: resolved={} tension={}",
        resolved.avg_tension(),
        tension.avg_tension()
    );
    assert!(
        settled.avg_tension() < tension.avg_tension() * 0.5,
        "settled presentation should stay low tension: settled={} tension={}",
        settled.avg_tension(),
        tension.avg_tension()
    );
}
