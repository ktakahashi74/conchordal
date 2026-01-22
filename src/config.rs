use crate::core::nsgt_kernel::KernelAlign;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfig {
    #[serde(default = "AudioConfig::default_latency_ms")]
    pub latency_ms: f32,
    #[serde(default = "AudioConfig::default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default)]
    pub output_guard: OutputGuardSetting,
}

impl AudioConfig {
    fn default_latency_ms() -> f32 {
        50.0
    }
    fn default_sample_rate() -> u32 {
        48_000
    }
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            latency_ms: Self::default_latency_ms(),
            sample_rate: Self::default_sample_rate(),
            output_guard: OutputGuardSetting::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum OutputGuardSetting {
    None,
    SoftClip,
    PeakLimiter,
}

impl Default for OutputGuardSetting {
    fn default() -> Self {
        Self::PeakLimiter
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    #[serde(default = "AnalysisConfig::default_nfft")]
    pub nfft: usize,
    #[serde(default = "AnalysisConfig::default_hop_size")]
    pub hop_size: usize,
    #[serde(default = "AnalysisConfig::default_tau_ms")]
    pub tau_ms: f32,
    #[serde(default)]
    pub kernel_align: KernelAlign,
}

impl AnalysisConfig {
    fn default_nfft() -> usize {
        16_384
    }
    fn default_hop_size() -> usize {
        512
    }
    fn default_tau_ms() -> f32 {
        10.0
    }
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            nfft: Self::default_nfft(),
            hop_size: Self::default_hop_size(),
            tau_ms: Self::default_tau_ms(),
            kernel_align: KernelAlign::Right,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PsychoAcousticsConfig {
    #[serde(default = "PsychoAcousticsConfig::default_loudness_exp")]
    pub loudness_exp: f32,
    #[serde(default = "PsychoAcousticsConfig::default_roughness_k")]
    pub roughness_k: f32,
    #[serde(default = "PsychoAcousticsConfig::default_roughness_weight")]
    pub roughness_weight: f32,
    #[serde(default = "PsychoAcousticsConfig::default_use_incoherent_power")]
    pub use_incoherent_power: bool,
}

impl PsychoAcousticsConfig {
    fn default_loudness_exp() -> f32 {
        0.23
    }
    fn default_roughness_k() -> f32 {
        0.428571
    }
    fn default_roughness_weight() -> f32 {
        1.0
    }
    fn default_use_incoherent_power() -> bool {
        false
    }
}

impl Default for PsychoAcousticsConfig {
    fn default() -> Self {
        Self {
            loudness_exp: Self::default_loudness_exp(),
            roughness_k: Self::default_roughness_k(),
            roughness_weight: Self::default_roughness_weight(),
            use_incoherent_power: Self::default_use_incoherent_power(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AppConfig {
    #[serde(default)]
    pub audio: AudioConfig,
    #[serde(default)]
    pub analysis: AnalysisConfig,
    #[serde(default)]
    pub psychoacoustics: PsychoAcousticsConfig,
    #[serde(default)]
    pub playback: PlaybackConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaybackConfig {
    #[serde(default = "PlaybackConfig::default_wait_user_exit")]
    pub wait_user_exit: bool,
    #[serde(default = "PlaybackConfig::default_wait_user_start")]
    pub wait_user_start: bool,
}

impl PlaybackConfig {
    fn default_wait_user_exit() -> bool {
        true
    }
    fn default_wait_user_start() -> bool {
        false
    }
}

impl Default for PlaybackConfig {
    fn default() -> Self {
        Self {
            wait_user_exit: Self::default_wait_user_exit(),
            wait_user_start: Self::default_wait_user_start(),
        }
    }
}

impl AppConfig {
    fn round_f32(x: f32) -> f32 {
        (x * 1_000_000.0).round() / 1_000_000.0
    }

    fn format_f32_compact(x: f32) -> String {
        let mut s = format!("{:.6}", x);
        while s.contains('.') && s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
        if s.is_empty() { "0".to_string() } else { s }
    }

    fn rounded(mut self) -> Self {
        self.audio.latency_ms = Self::round_f32(self.audio.latency_ms);
        self.analysis.tau_ms = Self::round_f32(self.analysis.tau_ms);
        self.psychoacoustics.loudness_exp = Self::round_f32(self.psychoacoustics.loudness_exp);
        self.psychoacoustics.roughness_k = Self::round_f32(self.psychoacoustics.roughness_k);
        self.psychoacoustics.roughness_weight =
            Self::round_f32(self.psychoacoustics.roughness_weight);
        self
    }

    pub fn load_or_default(path: &str) -> Self {
        let path_obj = Path::new(path);
        if path_obj.exists() {
            match fs::read_to_string(path_obj) {
                Ok(contents) => match toml::from_str(&contents) {
                    Ok(cfg) => return cfg,
                    Err(err) => {
                        eprintln!("Failed to parse config {path}: {err}. Using defaults.");
                    }
                },
                Err(err) => {
                    eprintln!("Failed to read config {path}: {err}. Using defaults.");
                }
            }
            return Self::default();
        }

        // File does not exist: write defaults and return them.
        let default_cfg = Self::default().rounded();
        if let Ok(text) = toml::to_string_pretty(&default_cfg) {
            let mut commented = String::new();
            for line in text.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    commented.push('\n');
                } else if trimmed.starts_with('[') && trimmed.ends_with(']') {
                    commented.push_str(line);
                    commented.push('\n');
                } else {
                    let mut out_line = line.to_string();
                    if let Some((lhs, rhs)) = line.split_once('=') {
                        let rhs_trim = rhs.trim();
                        let has_decimal = rhs_trim.contains('.');
                        if (has_decimal || rhs_trim.contains('e') || rhs_trim.contains('E'))
                            && !rhs_trim.contains('"')
                            && rhs_trim != "true"
                            && rhs_trim != "false"
                        {
                            if let Ok(val) = rhs_trim.parse::<f32>() {
                                let mut formatted = Self::format_f32_compact(val);
                                if has_decimal && !formatted.contains('.') {
                                    formatted.push_str(".0");
                                }
                                out_line = format!("{} = {}", lhs.trim(), formatted);
                            }
                        }
                    }
                    commented.push_str("# ");
                    commented.push_str(&out_line);
                    commented.push('\n');
                }
            }
            if let Err(err) = fs::write(path_obj, commented) {
                eprintln!("Failed to write default config to {path}: {err}");
            }
        } else {
            eprintln!("Failed to serialize default config; continuing with defaults");
        }
        default_cfg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn unique_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "conchordal_config_test_{}_{}",
            name,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        p
    }

    #[test]
    fn load_or_default_writes_defaults_cleanly() {
        let path = unique_path("defaults.toml");
        let path_str = path.to_string_lossy().to_string();
        // Ensure clean slate
        let _ = fs::remove_file(&path);

        let cfg = AppConfig::load_or_default(&path_str);
        assert!(path.exists(), "config file should be created");
        assert_eq!(cfg.audio.latency_ms, 50.0);
        assert_eq!(cfg.audio.sample_rate, 48_000);
        assert_eq!(cfg.audio.output_guard, OutputGuardSetting::PeakLimiter);
        assert_eq!(cfg.psychoacoustics.loudness_exp, 0.23);
        assert!((cfg.psychoacoustics.roughness_k - 0.428571).abs() < 1e-6);
        assert_eq!(cfg.psychoacoustics.roughness_weight, 1.0);
        assert!(!cfg.psychoacoustics.use_incoherent_power);

        let contents = fs::read_to_string(&path).expect("read written config");
        assert!(
            contents.contains("# loudness_exp = 0.23"),
            "should write commented loudness_exp"
        );
        assert!(
            contents.contains("# roughness_k = 0.428571"),
            "should write commented roughness_k"
        );
        assert!(
            contents.contains("# roughness_weight = 1.0"),
            "should write commented roughness_weight"
        );
        assert!(
            contents.contains("# use_incoherent_power = false"),
            "should write commented use_incoherent_power"
        );

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn load_or_default_reads_existing() {
        let path = unique_path("custom.toml");
        let path_str = path.to_string_lossy().to_string();
        let custom = AppConfig {
            audio: AudioConfig {
                latency_ms: 75.0,
                sample_rate: 44_100,
                output_guard: OutputGuardSetting::SoftClip,
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
                roughness_weight: 0.8,
                use_incoherent_power: false,
            },
            playback: PlaybackConfig {
                wait_user_exit: false,
                wait_user_start: true,
            },
        };
        let text = toml::to_string_pretty(&custom).unwrap();
        fs::write(&path, text).unwrap();

        let cfg = AppConfig::load_or_default(&path_str);
        assert_eq!(cfg.audio.latency_ms, 75.0);
        assert_eq!(cfg.audio.sample_rate, 44_100);
        assert_eq!(cfg.audio.output_guard, OutputGuardSetting::SoftClip);
        assert_eq!(cfg.analysis.nfft, 8192);
        assert_eq!(cfg.analysis.hop_size, 256);
        assert_eq!(cfg.analysis.tau_ms, 60.0);
        assert_eq!(cfg.psychoacoustics.loudness_exp, 0.3);
        assert_eq!(cfg.psychoacoustics.roughness_k, 0.2);
        assert_eq!(cfg.psychoacoustics.roughness_weight, 0.8);
        assert!(!cfg.psychoacoustics.use_incoherent_power);
        assert!(!cfg.playback.wait_user_exit);
        assert!(cfg.playback.wait_user_start);

        let _ = fs::remove_file(&path);
    }
}
