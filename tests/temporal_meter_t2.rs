//! Public configuration and offline causal determinism.
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "support/test_output_directory.rs"]
mod test_output_directory;
use test_output_directory::TestOutputDirectory;

fn directory() -> (PathBuf, TestOutputDirectory) {
    let path = std::env::temp_dir().join(format!(
        "conchordal-l2-it-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let guard = TestOutputDirectory::new(path.clone());
    (path, guard)
}
fn render(
    directory: &Path,
    script: &Path,
    config: &str,
    name: &str,
) -> (Vec<u8>, Vec<serde_json::Value>) {
    let cfg = directory.join(format!("{name}.toml"));
    fs::write(&cfg, config).unwrap();
    let wav = directory.join(format!("{name}.wav"));
    let report = directory.join(format!("{name}.jsonl"));
    let output = Command::new(env!("CARGO_BIN_EXE_conchordal-render"))
        .env("RUST_LOG", "warn")
        .arg(script)
        .arg("--config")
        .arg(cfg)
        .arg("--output")
        .arg(&wav)
        .arg("--report")
        .arg(&report)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let mut records = Vec::new();
    for line in BufReader::new(fs::File::open(report).unwrap()).lines() {
        let mut row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
        if row["type"] == "hop_timing" {
            continue;
        }
        if row["type"] == "temporal_meter" {
            row["input"]
                .as_object_mut()
                .unwrap()
                .remove("processing_us");
        }
        records.push(row);
    }
    (fs::read(wav).unwrap(), records)
}
#[test]
fn continuous_config_combinations_preserve_group_dependencies() {
    use conchordal::config::AppConfig;
    let (dir, _guard) = directory();
    let template = dir.join("config.toml");
    let generated = AppConfig::load_or_default(template.to_str().unwrap()).unwrap();
    assert!(!generated.temporal_meter_t2);
    assert!(
        fs::read_to_string(template)
            .unwrap()
            .contains("# temporal_meter_t2 = false")
    );
    let legacy = concat!(
        include_str!("fixtures/temporal_meter_t2-events.toml"),
        "\n[temporal_t2]\ncomponent_weights=[0.0,0.0,1.0,0.0]\ngain=1.0\nthreshold=0.0\nweight_gain=1.0\n",
        "window='hop_mean'\ncombination='positive_component_rise'\nattribution='first_hop_mass_fraction'\nunknown='suppress'\n"
    );
    for text in [
        String::new(),
        "temporal_meter_t2=false\n".into(),
        "temporal_meter_t2=true\n".into(),
        legacy.into(),
        format!("temporal_meter_t2=true\n{legacy}"),
    ] {
        let cfg: AppConfig = toml::from_str(&text).unwrap();
        cfg.validate().unwrap();
    }
    let missing: AppConfig = toml::from_str("[temporal_onset_comparison]\narrival=true\n").unwrap();
    assert!(missing.validate().is_err());
    let arrival: AppConfig =
        toml::from_str(include_str!("fixtures/temporal_meter_t2-events.toml")).unwrap();
    arrival.validate().unwrap();
    assert!(!arrival.temporal_meter_t2);
    assert!(toml::from_str::<AppConfig>("temporal_meter_t22=true\n").is_err());
}
#[test]
fn continuous_offline_render_is_deterministic_before_synthesis() {
    let (dir, _guard) = directory();
    let script = dir.join("input.rhai");
    fs::write(&script, include_str!("fixtures/temporal_meter_t2.rhai")).unwrap();
    let a = render(&dir, &script, "temporal_meter_t2=true\n", "a");
    let b = render(&dir, &script, "temporal_meter_t2=true\n", "b");
    assert_eq!(a, b);
    let mut seen = 0;
    for row in &a.1 {
        if row["type"] != "temporal_meter" {
            continue;
        }
        seen += 1;
        if row["input"]["terminal"].is_null() {
            assert_eq!(row["input"]["available"], row["use_cut"]);
            assert_eq!(row["input"]["received"], row["use_cut"]);
            assert_eq!(row["input"]["configuration_epoch"], 0);
        }
    }
    assert!(seen > 0);
}
#[test]
fn continuous_omission_and_false_preserve_audio_and_non_timing_report() {
    let (dir, _guard) = directory();
    let script = dir.join("input.rhai");
    fs::write(&script, include_str!("fixtures/temporal_meter_t2.rhai")).unwrap();
    let a = render(&dir, &script, "", "omitted");
    let b = render(&dir, &script, "temporal_meter_t2=false\n", "disabled");
    assert_eq!(a, b);
    assert!(a.1.iter().all(|r| r["type"] != "temporal_meter"));
}
