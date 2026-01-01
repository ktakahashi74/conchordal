use std::path::Path;

use walkdir::WalkDir;

use conchordal::app::compile_scenario_from_script;
use conchordal::cli::Args;
use conchordal::config::AppConfig;

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
    let dir = Path::new("samples");
    let mut count = 0;
    for entry in WalkDir::new(dir).into_iter().filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("rhai") {
            continue;
        }
        count += 1;
        let args = args_for_path(path);
        let config = AppConfig::default();
        compile_scenario_from_script(path, &args, &config)
            .unwrap_or_else(|e| panic!("compile-only failed for {}: {e}", path.display()));
    }
    assert!(count > 0, "no .rhai scripts found in samples");
}
