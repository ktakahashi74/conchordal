//! Fixed no-Flora render snapshots for the selected auditory front end.
//! Authorized front-end migrations recapture each registered target-feature group
//! with `FLORA_INVARIANCE_CAPTURE=1`; every case must match in two acquisitions.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

const FIXTURE: &str = "tests/fixtures/flora_invariance_hashes.txt";

const BASE_CONFIG: &str = r#"
[audio]
sample_rate = 48000
output_gain_db = 0.0
[analysis]
nfft = 2048
hop_size = 512
[psychoacoustics.habituation]
enabled = false
"#;

const HABITUATION_CONFIG: &str = r#"
[audio]
sample_rate = 48000
output_gain_db = 0.0
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

fn render_hash(name: &str, config: &str, script: &str) -> u64 {
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
        "{name}: render failed: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let bytes = fs::read(&wav).expect("read wav");
    let mut reader = hound::WavReader::new(bytes.as_slice()).expect("decode wav");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 48000, "{name}: sample rate");
    assert_eq!(spec.channels, 1, "{name}: mono output");
    assert_eq!(spec.sample_format, hound::SampleFormat::Int, "{name}: PCM");
    assert_eq!(spec.bits_per_sample, 16, "{name}: PCM width");
    // Successfully decoded integer PCM contains only finite output samples.
    let samples: Vec<i16> = reader
        .samples::<i16>()
        .collect::<Result<_, _>>()
        .expect("read finite PCM samples");
    assert!(samples.len() >= 512, "{name}: at least one output hop");
    assert!(samples.iter().any(|&s| s != 0), "{name}: audible output");
    assert!(
        samples[samples.len() - 512..].iter().all(|&s| s == 0),
        "{name}: final hop must be silent after release"
    );
    for path in [&config_path, &scenario, &wav] {
        let _ = fs::remove_file(path);
    }
    fnv1a64(&bytes)
}

fn read_fixture(path: &Path) -> Vec<(String, String, u64)> {
    fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .map(|line| {
            let mut fields = line.split_whitespace();
            let key = fields.next().expect("fixture target features");
            let name = fields.next().expect("fixture case name");
            let hash =
                u64::from_str_radix(fields.next().expect("fixture hash"), 16).expect("hex hash");
            assert!(fields.next().is_none(), "fixture line: key case hash");
            (key.to_string(), name.to_string(), hash)
        })
        .collect()
}

#[test]
fn renders_without_flora_match_auditory_references() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE);
    let feature_key = format!(
        "fma={},avx2={},avx512f={}",
        u8::from(cfg!(target_feature = "fma")),
        u8::from(cfg!(target_feature = "avx2")),
        u8::from(cfg!(target_feature = "avx512f")),
    );
    // DSP avoids mul_add, so native AVX/FMA and portable x86_64 builds currently
    // agree; groups stay keyed so a divergence shows per group. Compare only a
    // registered Linux x86_64 feature group. Unregistered groups and other
    // platforms retain the finite-output and silent-tail checks above.
    let compare_hashes = cfg!(all(target_os = "linux", target_arch = "x86_64"));
    if std::env::var("FLORA_INVARIANCE_CAPTURE").as_deref() == Ok("1") {
        assert!(compare_hashes, "capture requires Linux x86_64");
        let toolchain = Command::new("rustc")
            .arg("-V")
            .output()
            .expect("read capture toolchain");
        assert!(toolchain.status.success(), "rustc -V failed");
        let group_prefix = format!("{feature_key} ");
        let capture_prefix = format!("# Capture {feature_key}:");
        // Recapturing one group must preserve references for the other groups.
        let mut text = fs::read_to_string(&fixture)
            .expect("read fixture before capture")
            .lines()
            .filter(|line| !line.starts_with(&group_prefix) && !line.starts_with(&capture_prefix))
            .collect::<Vec<_>>()
            .join("\n");
        text.push_str(&format!(
            "\n{capture_prefix} {}; RUSTFLAGS={}\n",
            String::from_utf8_lossy(&toolchain.stdout).trim(),
            std::env::var("RUSTFLAGS").unwrap_or_else(|_| "<unset; Cargo config applies>".into()),
        ));
        for (name, config, script) in CASES {
            let first = render_hash(name, config, script);
            let second = render_hash(name, config, script);
            assert_eq!(first, second, "{name}: render is not deterministic");
            println!("{feature_key} {name}: capture 1={first:016x}, capture 2={second:016x}");
            text.push_str(&format!("{feature_key} {name} {first:016x}\n"));
        }
        fs::write(&fixture, text).expect("write fixture");
        return;
    }
    if compare_hashes {
        let expected: Vec<_> = read_fixture(&fixture)
            .into_iter()
            .filter(|(key, _, _)| key == &feature_key)
            .collect();
        if !expected.is_empty() {
            assert_eq!(
                expected.len(),
                CASES.len(),
                "{feature_key}: fixture case count"
            );
            for ((name, config, script), (_, fixture_name, hash)) in CASES.iter().zip(&expected) {
                assert_eq!(name, fixture_name, "{feature_key}: fixture order");
                assert_eq!(
                    render_hash(name, config, script),
                    *hash,
                    "{feature_key} {name}: render changed without Flora"
                );
            }
            return;
        }
        println!("Flora invariance: no baseline for {feature_key}; checking renders only");
    }
    for (name, config, script) in CASES {
        render_hash(name, config, script);
    }
}
