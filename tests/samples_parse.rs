use std::path::PathBuf;

use conchordal::life::scripting::ScriptHost;

#[test]
fn samples_parse_successfully() {
    let mut stack = vec![PathBuf::from("samples")];
    let mut found = false;
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("samples dir exists") {
            let path = entry.expect("dir entry").path();
            let name = path
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string();
            if name.starts_with('#') || name.starts_with('.') || name.ends_with('~') {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            if path.extension().and_then(|s| s.to_str()) != Some("rhai") {
                continue;
            }
            found = true;
            ScriptHost::load_script(path.to_str().expect("path str"))
                .unwrap_or_else(|e| panic!("script {name} should parse: {e}"));
        }
    }
    assert!(found, "no sample scripts found under samples/");
}
