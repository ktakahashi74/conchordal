use std::process::Command;

#[test]
fn run_agent_intent_smoke() {
    let exe = env!("CARGO_BIN_EXE_conchordal");
    let script = "tests/scripts/agent_intent_smoke.rhai";

    let output = Command::new(exe)
        .env("RUST_LOG", "debug")
        .args(["--nogui", "--play=false", "--intent-only=true"])
        .arg(script)
        .output()
        .unwrap_or_else(|e| panic!("failed to run {script}: {e}"));

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "script failed ({}): status={} stderr={}",
            script, output.status, stderr
        );
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let combined = format!("{stdout}\n{stderr}");
    assert!(
        combined.contains("TEST_OK: agent_intent_smoke"),
        "marker not found for {script}"
    );
}
