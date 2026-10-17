use std::path::Path;

use walkdir::WalkDir;

use conchordal::scripting::ScriptHost;

#[test]
fn compile_only_samples_tests() {
    let dir = Path::new("samples");
    let mut count = 0;
    for entry in WalkDir::new(dir).into_iter().filter_map(Result::ok) {
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str())
            && name.starts_with(".#")
        {
            continue;
        }
        if path.extension().and_then(|s| s.to_str()) != Some("rhai") {
            continue;
        }
        count += 1;
        let (_, warnings) = ScriptHost::load_script_with_warnings(&path.to_string_lossy(), None)
            .unwrap_or_else(|e| panic!("compile-only failed for {}: {e}", path.display()));
        assert!(warnings.is_empty(), "{}: {warnings:?}", path.display());
    }
    assert!(count > 0, "no .rhai scripts found in samples");
}
