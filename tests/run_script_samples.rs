use std::path::Path;
use std::process::Command;

use walkdir::WalkDir;

#[test]
fn run_headless_scripts() {
    let exe = env!("CARGO_BIN_EXE_conchordal");
    let dir = Path::new("tests/scripts");
    let mut count = 0;

    for entry in WalkDir::new(dir).into_iter().filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("rhai") {
            continue;
        }
        count += 1;
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("script");
        let marker = format!("TEST_OK: {stem}");

        let output = Command::new(exe)
            .env("RUST_LOG", "debug")
            .args(["--nogui", "--play=false"])
            .arg(path)
            .output()
            .unwrap_or_else(|e| panic!("failed to run {}: {e}", path.display()));

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!(
                "script failed ({}): status={} stderr={}",
                path.display(),
                output.status,
                stderr
            );
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let combined = format!("{stdout}\n{stderr}");
        assert!(
            combined.contains(&marker),
            "marker not found for {}: expected {:?}",
            path.display(),
            marker
        );
    }

    assert!(count > 0, "no .rhai scripts found in tests/scripts");
}
