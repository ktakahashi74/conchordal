//! Renders without Flora must stay bit-identical to references captured before Flora
//! existed. Capture with `FLORA_INVARIANCE_CAPTURE=1 cargo test --test flora_invariance`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const FIXTURE: &str = "tests/fixtures/flora_invariance_hashes.txt";

const BASE_CONFIG: &str = r#"
[audio]
sample_rate = 48000
[analysis]
nfft = 2048
hop_size = 512
"#;

const HABITUATION_CONFIG: &str = r#"
[audio]
sample_rate = 48000
[analysis]
nfft = 2048
hop_size = 512
[psychoacoustics.habituation]
enabled = true
"#;

const CASES: &[(&str, &str, &str)] = &[
    (
        "sine",
        BASE_CONFIG,
        r#"
seed(11);
let v = sine().amp(0.05).sustain().adsr(0.05, 0.1, 0.8, 0.3);
let g = place(v, consonance(220.0, 880.0).count(3));
wait(2.0);
release(g);
wait(0.5);
"#,
    ),
    (
        "harmonic",
        BASE_CONFIG,
        r#"
seed(12);
let sun = place(harmonic().brain("drone").amp(0.05).sustain().anchor(), at(110.0));
wait(0.5);
let body = place(
    harmonic().amp(0.03).sustain(),
    consonance(110.0).peak().range(1.0, 4.0).count(4)
);
wait(2.0);
release(body);
release(sun);
wait(0.6);
"#,
    ),
    (
        "modal",
        BASE_CONFIG,
        r#"
seed(13);
let m = modal().amp(0.03).flow().cycles(3).adsr(0.03, 0.3, 0.5, 0.7);
let g = place(m, consonance(220.0, 880.0).spacing(0.8).count(4));
wait(2.5);
release(g);
wait(0.8);
"#,
    ),
    (
        "habituation",
        HABITUATION_CONFIG,
        r#"
seed(14);
let v = harmonic().amp(0.04).sustain().seek_consonance();
let g = place(v, consonance(196.0, 784.0).count(3));
wait(3.0);
release(g);
wait(0.5);
"#,
    ),
];

fn unique_temp_path(ext: &str) -> PathBuf {
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "conchordal-flora-inv-{}-{nanos}-{sequence}.{ext}",
        std::process::id()
    ))
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        hash ^= u64::from(b);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn render_hash(config: &str, script: &str) -> u64 {
    let config_path = unique_temp_path("toml");
    let scenario = unique_temp_path("rhai");
    let wav = unique_temp_path("wav");
    fs::write(&config_path, config).expect("write config");
    fs::write(&scenario, script).expect("write scenario");
    let output = Command::new(env!("CARGO_BIN_EXE_conchordal-render"))
        .env("RUST_LOG", "warn")
        .arg(&scenario)
        .arg("-o")
        .arg(&wav)
        .arg("--config")
        .arg(&config_path)
        .output()
        .expect("run conchordal-render");
    assert!(
        output.status.success(),
        "render failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = fs::read(&wav).expect("read wav");
    for path in [&config_path, &scenario, &wav] {
        let _ = fs::remove_file(path);
    }
    fnv1a64(&bytes)
}

fn read_fixture(path: &Path) -> Vec<(String, u64)> {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .map(|line| {
            let (name, hash) = line.split_once(' ').expect("fixture line: name hash");
            let hash = u64::from_str_radix(hash.trim(), 16).expect("hex hash");
            (name.to_string(), hash)
        })
        .collect()
}

#[test]
fn renders_without_flora_match_pre_flora_references() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    if std::env::var_os("FLORA_INVARIANCE_CAPTURE").is_some() {
        let mut text =
            String::from("# name fnv1a64(wav bytes), captured from the pre-Flora build\n");
        for (name, config, script) in CASES {
            let first = render_hash(config, script);
            let second = render_hash(config, script);
            assert_eq!(first, second, "{name}: render is not deterministic");
            text.push_str(&format!("{name} {first:016x}\n"));
        }
        fs::write(&fixture, text).expect("write fixture");
        return;
    }
    let expected = read_fixture(&fixture);
    assert_eq!(expected.len(), CASES.len(), "fixture case count");
    for ((name, config, script), (fixture_name, hash)) in CASES.iter().zip(&expected) {
        assert_eq!(name, fixture_name, "fixture order");
        assert_eq!(
            render_hash(config, script),
            *hash,
            "{name}: render changed without Flora"
        );
    }
}
