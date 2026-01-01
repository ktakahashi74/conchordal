use std::path::Path;

use walkdir::WalkDir;

use conchordal::app::{compile_scenario_from_script, validate_scenario};
use conchordal::cli::Args;
use conchordal::config::AppConfig;
use conchordal::life::scenario::Action;

fn args_for_path(path: &Path) -> Args {
    Args {
        play: false,
        wav: None,
        scenario_path: path.to_string_lossy().to_string(),
        config: "config.toml".to_string(),
        wait_user_exit: None,
        wait_user_start: None,
        nogui: false,
        compile_only: false,
    }
}

#[test]
fn compile_only_samples_tests() {
    let dir = Path::new("samples/tests");
    let mut count = 0;
    for entry in WalkDir::new(dir).into_iter().filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("rhai") {
            continue;
        }
        count += 1;
        let args = args_for_path(path);
        let config = AppConfig::default();
        let scenario = compile_scenario_from_script(path, &args, &config)
            .unwrap_or_else(|e| panic!("compile-only failed for {}: {e}", path.display()));
        validate_scenario(&scenario)
            .unwrap_or_else(|e| panic!("validate failed for {}: {e}", path.display()));
    }
    assert!(count > 0, "no .rhai scripts found in samples/tests");
}

#[test]
fn stable_order_same_time_actions_preserved() {
    let path = Path::new("samples/tests/stable_order_same_time_actions.rhai");
    let args = args_for_path(path);
    let config = AppConfig::default();
    let scenario = compile_scenario_from_script(path, &args, &config)
        .unwrap_or_else(|e| panic!("compile-only failed for {}: {e}", path.display()));
    validate_scenario(&scenario)
        .unwrap_or_else(|e| panic!("validate failed for {}: {e}", path.display()));

    let mut set_freqs = Vec::new();
    for scene in &scenario.scenes {
        for event in &scene.events {
            for action in &event.actions {
                if let Action::SetFreq { freq_hz, .. } = action {
                    set_freqs.push(*freq_hz);
                }
            }
        }
    }

    let expected = vec![200.0, 300.0];
    assert!(
        set_freqs.starts_with(&expected),
        "set_freq order mismatch: {:?}",
        set_freqs
    );
}
