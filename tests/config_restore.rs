use std::fs;
use std::path::PathBuf;

use conchordal::config::{
    AnalysisConfig, AppConfig, AudioConfig, LimiterSetting, PlaybackConfig, PsychoAcousticsConfig,
};
use conchordal::core::nsgt_kernel::KernelAlign;

fn unique_path(name: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "conchordal_config_restore_{}_{}",
        name,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    path
}

fn assert_close(a: f32, b: f32, label: &str) {
    let diff = (a - b).abs();
    assert!(diff <= 1e-6, "{label} mismatch: {a} vs {b}");
}

fn assert_config_eq(actual: &AppConfig, expected: &AppConfig) {
    assert_close(
        actual.audio.latency_ms,
        expected.audio.latency_ms,
        "audio.latency_ms",
    );
    assert_eq!(actual.audio.sample_rate, expected.audio.sample_rate);
    assert_eq!(actual.audio.limiter, expected.audio.limiter);
    assert_eq!(actual.analysis.nfft, expected.analysis.nfft);
    assert_eq!(actual.analysis.hop_size, expected.analysis.hop_size);
    assert_close(
        actual.analysis.tau_ms,
        expected.analysis.tau_ms,
        "analysis.tau_ms",
    );
    assert_eq!(actual.analysis.kernel_align, expected.analysis.kernel_align);
    assert_close(
        actual.psychoacoustics.loudness_exp,
        expected.psychoacoustics.loudness_exp,
        "psychoacoustics.loudness_exp",
    );
    assert_close(
        actual.psychoacoustics.roughness_k,
        expected.psychoacoustics.roughness_k,
        "psychoacoustics.roughness_k",
    );
    assert_close(
        actual.psychoacoustics.harmonicity_weight,
        expected.psychoacoustics.harmonicity_weight,
        "psychoacoustics.harmonicity_weight",
    );
    assert_close(
        actual.psychoacoustics.roughness_weight_floor,
        expected.psychoacoustics.roughness_weight_floor,
        "psychoacoustics.roughness_weight_floor",
    );
    assert_close(
        actual.psychoacoustics.roughness_weight,
        expected.psychoacoustics.roughness_weight,
        "psychoacoustics.roughness_weight",
    );
    assert_close(
        actual.psychoacoustics.c_state_beta,
        expected.psychoacoustics.c_state_beta,
        "psychoacoustics.c_state_beta",
    );
    assert_close(
        actual.psychoacoustics.c_state_theta,
        expected.psychoacoustics.c_state_theta,
        "psychoacoustics.c_state_theta",
    );
    assert_eq!(
        actual.psychoacoustics.use_incoherent_power,
        expected.psychoacoustics.use_incoherent_power
    );
    assert_eq!(
        actual.playback.wait_user_exit,
        expected.playback.wait_user_exit
    );
    assert_eq!(
        actual.playback.wait_user_start,
        expected.playback.wait_user_start
    );
}

#[test]
fn config_roundtrip_default_toml() {
    let default_cfg = AppConfig::default();
    let text = toml::to_string_pretty(&default_cfg).expect("serialize default");
    let parsed: AppConfig = toml::from_str(&text).expect("parse default");
    assert_config_eq(&parsed, &default_cfg);
}

#[test]
fn config_load_custom_values() {
    let path = unique_path("custom.toml");
    let path_str = path.to_string_lossy().to_string();
    let custom = AppConfig {
        audio: AudioConfig {
            latency_ms: 75.0,
            sample_rate: 44_100,
            limiter: LimiterSetting::SoftClip,
        },
        analysis: AnalysisConfig {
            nfft: 8192,
            hop_size: 256,
            tau_ms: 60.0,
            kernel_align: KernelAlign::Center,
        },
        psychoacoustics: PsychoAcousticsConfig {
            loudness_exp: 0.3,
            roughness_k: 0.2,
            harmonicity_weight: 1.2,
            roughness_weight_floor: 0.4,
            roughness_weight: 0.8,
            c_state_beta: 3.25,
            c_state_theta: -0.15,
            use_incoherent_power: false,
        },
        playback: PlaybackConfig {
            wait_user_exit: false,
            wait_user_start: true,
        },
    };
    let text = toml::to_string_pretty(&custom).expect("serialize custom");
    fs::write(&path, text).expect("write custom config");

    let loaded = AppConfig::load_or_default(&path_str);
    assert_config_eq(&loaded, &custom);

    let _ = fs::remove_file(&path);
}

#[test]
fn config_missing_file_fallback() {
    let path = unique_path("missing.toml");
    let path_str = path.to_string_lossy().to_string();
    let _ = fs::remove_file(&path);

    let loaded = AppConfig::load_or_default(&path_str);
    let defaults = AppConfig::default();
    assert!(path.exists(), "missing config should be created");
    assert_config_eq(&loaded, &defaults);

    let _ = fs::remove_file(&path);
}

#[test]
fn config_load_legacy_harmonicity_deficit_weight() {
    let text = r#"
[psychoacoustics]
harmonicity_deficit_weight = 1.75
"#;
    let parsed: AppConfig = toml::from_str(text).expect("parse legacy harmonicity key");
    assert_close(
        parsed.psychoacoustics.harmonicity_weight,
        1.75,
        "psychoacoustics.harmonicity_weight",
    );
}
