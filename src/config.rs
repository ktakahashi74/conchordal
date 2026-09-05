use crate::core::nsgt_kernel::KernelAlign;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioConfig {
    #[serde(default = "AudioConfig::default_latency_ms")]
    pub latency_ms: f32,
    #[serde(default = "AudioConfig::default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default)]
    pub limiter: LimiterSetting,
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
            limiter: LimiterSetting::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "kebab-case")]
pub enum LimiterSetting {
    None,
    SoftClip,
    #[default]
    PeakLimiter,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisConfig {
    #[serde(default = "AnalysisConfig::default_nfft")]
    pub nfft: usize,
    #[serde(default = "AnalysisConfig::default_hop_size")]
    pub hop_size: usize,
    #[serde(default = "AnalysisConfig::default_tau_ms")]
    pub tau_ms: f32,
    #[serde(default = "AnalysisConfig::default_kernel_align")]
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
    fn default_kernel_align() -> KernelAlign {
        // Match the Default impl: runtime stamps observations at frame end,
        // which assumes right-aligned kernels.
        KernelAlign::Right
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
#[serde(deny_unknown_fields)]
pub struct ConsonanceKernelConfig {
    #[serde(default = "ConsonanceKernelConfig::default_a")]
    pub a: f32,
    #[serde(default = "ConsonanceKernelConfig::default_b")]
    pub b: f32,
    #[serde(default = "ConsonanceKernelConfig::default_c")]
    pub c: f32,
    #[serde(default = "ConsonanceKernelConfig::default_d")]
    pub d: f32,
}

impl ConsonanceKernelConfig {
    fn default_a() -> f32 {
        1.0
    }
    fn default_b() -> f32 {
        -1.35
    }
    fn default_c() -> f32 {
        1.0
    }
    fn default_d() -> f32 {
        0.0
    }
}

impl Default for ConsonanceKernelConfig {
    fn default() -> Self {
        Self {
            a: Self::default_a(),
            b: Self::default_b(),
            c: Self::default_c(),
            d: Self::default_d(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsonanceLevelConfig {
    #[serde(default = "ConsonanceLevelConfig::default_beta")]
    pub beta: f32,
    #[serde(default = "ConsonanceLevelConfig::default_theta")]
    pub theta: f32,
}

impl ConsonanceLevelConfig {
    fn default_beta() -> f32 {
        2.0
    }
    fn default_theta() -> f32 {
        0.0
    }
}

impl Default for ConsonanceLevelConfig {
    fn default() -> Self {
        Self {
            beta: Self::default_beta(),
            theta: Self::default_theta(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ConsonanceFieldConfig {
    #[serde(default)]
    pub kernel: ConsonanceKernelConfig,
    #[serde(default)]
    pub level: ConsonanceLevelConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ConsonanceDensityConfig {
    /// rho in density kernel H * (1 - rho * R). Used to absorb roughness scale arbitrariness.
    #[serde(default = "ConsonanceDensityConfig::default_roughness_gain")]
    pub roughness_gain: f32,
}

impl ConsonanceDensityConfig {
    fn default_roughness_gain() -> f32 {
        1.0
    }
}

impl Default for ConsonanceDensityConfig {
    fn default() -> Self {
        Self {
            roughness_gain: Self::default_roughness_gain(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HabituationConfig {
    #[serde(default = "HabituationConfig::default_enabled")]
    pub enabled: bool,
    #[serde(default = "HabituationConfig::default_satiation_sec")]
    pub satiation_sec: f32,
    #[serde(default = "HabituationConfig::default_recovery_sec")]
    pub recovery_sec: f32,
    #[serde(default = "HabituationConfig::default_ref_drive")]
    pub ref_drive: f32,
}

impl HabituationConfig {
    fn default_enabled() -> bool {
        false
    }
    fn default_satiation_sec() -> f32 {
        5.0
    }
    fn default_recovery_sec() -> f32 {
        8.0
    }
    fn default_ref_drive() -> f32 {
        0.25
    }
}

impl Default for HabituationConfig {
    fn default() -> Self {
        Self {
            enabled: Self::default_enabled(),
            satiation_sec: Self::default_satiation_sec(),
            recovery_sec: Self::default_recovery_sec(),
            ref_drive: Self::default_ref_drive(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ConsonanceConfig {
    #[serde(default)]
    pub field: ConsonanceFieldConfig,
    #[serde(default)]
    pub density: ConsonanceDensityConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PsychoAcousticsConfig {
    #[serde(default = "PsychoAcousticsConfig::default_loudness_exp")]
    pub loudness_exp: f32,
    #[serde(default = "PsychoAcousticsConfig::default_roughness_k")]
    pub roughness_k: f32,
    #[serde(default)]
    pub consonance: ConsonanceConfig,
    #[serde(default)]
    pub habituation: HabituationConfig,
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
    fn default_use_incoherent_power() -> bool {
        false
    }
}

impl Default for PsychoAcousticsConfig {
    fn default() -> Self {
        Self {
            loudness_exp: Self::default_loudness_exp(),
            roughness_k: Self::default_roughness_k(),
            consonance: ConsonanceConfig::default(),
            habituation: HabituationConfig::default(),
            use_incoherent_power: Self::default_use_incoherent_power(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct AppConfig {
    #[serde(default)]
    pub audio: AudioConfig,
    #[serde(default)]
    pub analysis: AnalysisConfig,
    #[serde(default)]
    pub psychoacoustics: PsychoAcousticsConfig,
    #[serde(default)]
    pub dcc: DccConfig,
    #[serde(default)]
    pub playback: PlaybackConfig,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DccConfig {
    #[serde(default = "DccConfig::default_coupling_strength")]
    pub coupling_strength: f32,
    #[serde(default = "DccConfig::default_max_temperature_bonus")]
    pub max_temperature_bonus: f32,
}

impl DccConfig {
    fn default_coupling_strength() -> f32 {
        0.0
    }

    fn default_max_temperature_bonus() -> f32 {
        0.10
    }
}

impl Default for DccConfig {
    fn default() -> Self {
        Self {
            coupling_strength: Self::default_coupling_strength(),
            max_temperature_bonus: Self::default_max_temperature_bonus(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    /// Validate the dimensions shared by the runtime and the NSGT kernel.
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.audio.sample_rate > 0,
            "audio.sample_rate must be positive"
        );
        ensure!(
            self.audio.latency_ms.is_finite() && self.audio.latency_ms > 0.0,
            "audio.latency_ms must be finite and positive"
        );
        let buffer_frames =
            self.audio.sample_rate as f64 * self.audio.latency_ms as f64 / 1000.0 * 2.0;
        ensure!(
            buffer_frames < (isize::MAX as usize / std::mem::size_of::<f32>()) as f64,
            "audio.latency_ms exceeds the addressable audio buffer size"
        );
        ensure!(
            (1..=1 << 18).contains(&self.analysis.nfft),
            "analysis.nfft must be between 1 and 262144 (the NSGT limit)"
        );
        ensure!(
            self.analysis.hop_size > 0 && self.analysis.hop_size <= self.analysis.nfft,
            "analysis.hop_size must be positive and no greater than analysis.nfft"
        );
        let overlap = 1.0 - self.analysis.hop_size as f32 / self.analysis.nfft as f32;
        ensure!(
            (0.0..0.99).contains(&overlap),
            "analysis.hop_size is too small for analysis.nfft: NSGT overlap must be below 0.99"
        );
        ensure!(
            !std::time::Duration::from_secs_f32(
                self.analysis.hop_size as f32 / self.audio.sample_rate as f32
            )
            .is_zero(),
            "analysis.hop_size / audio.sample_rate must produce a nonzero hop duration"
        );
        Ok(())
    }

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

    pub fn round_f32_inplace(&mut self) {
        self.audio.latency_ms = Self::round_f32(self.audio.latency_ms);
        self.analysis.tau_ms = Self::round_f32(self.analysis.tau_ms);
        self.psychoacoustics.loudness_exp = Self::round_f32(self.psychoacoustics.loudness_exp);
        self.psychoacoustics.roughness_k = Self::round_f32(self.psychoacoustics.roughness_k);

        let kernel = &mut self.psychoacoustics.consonance.field.kernel;
        kernel.a = Self::round_f32(kernel.a);
        kernel.b = Self::round_f32(kernel.b);
        kernel.c = Self::round_f32(kernel.c);
        kernel.d = Self::round_f32(kernel.d);

        let level = &mut self.psychoacoustics.consonance.field.level;
        level.beta = Self::round_f32(level.beta);
        level.theta = Self::round_f32(level.theta);

        let density = &mut self.psychoacoustics.consonance.density;
        density.roughness_gain = Self::round_f32(density.roughness_gain);
        density.roughness_gain = if density.roughness_gain.is_finite() {
            density.roughness_gain.max(0.0)
        } else {
            1.0
        };

        self.dcc.coupling_strength = Self::round_f32(self.dcc.coupling_strength);
        self.dcc.coupling_strength = if self.dcc.coupling_strength.is_finite() {
            self.dcc.coupling_strength.clamp(0.0, 1.0)
        } else {
            0.0
        };
        self.dcc.max_temperature_bonus = Self::round_f32(self.dcc.max_temperature_bonus);
        self.dcc.max_temperature_bonus = if self.dcc.max_temperature_bonus.is_finite() {
            self.dcc.max_temperature_bonus.max(0.0)
        } else {
            DccConfig::default_max_temperature_bonus()
        };
    }

    fn rounded(mut self) -> Self {
        self.round_f32_inplace();
        self
    }

    /// Loads `path`, or writes and returns defaults when the file is absent.
    /// An existing but unreadable or malformed file (including unknown keys)
    /// is a hard error: a mistyped key must not silently fall back to a default.
    pub fn load_or_default(path: &str) -> Result<Self> {
        let path_obj = Path::new(path);
        if path_obj.exists() {
            let contents = fs::read_to_string(path_obj)
                .with_context(|| format!("failed to read config file '{path}'"))?;
            let mut cfg: Self = toml::from_str(&contents)
                .with_context(|| format!("failed to parse config file '{path}'"))?;
            cfg.round_f32_inplace();
            cfg.validate()
                .with_context(|| format!("invalid config file '{path}'"))?;
            return Ok(cfg);
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
                            && let Ok(val) = rhs_trim.parse::<f32>()
                        {
                            let mut formatted = Self::format_f32_compact(val);
                            if has_decimal && !formatted.contains('.') {
                                formatted.push_str(".0");
                            }
                            out_line = format!("{} = {}", lhs.trim(), formatted);
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
        Ok(default_cfg)
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

        let cfg = AppConfig::load_or_default(&path_str).expect("write and load defaults");
        assert!(path.exists(), "config file should be created");
        assert_eq!(cfg.audio.latency_ms, 50.0);
        assert_eq!(cfg.audio.sample_rate, 48_000);
        assert_eq!(cfg.audio.limiter, LimiterSetting::PeakLimiter);
        assert_eq!(cfg.psychoacoustics.loudness_exp, 0.23);
        assert!((cfg.psychoacoustics.roughness_k - 0.428571).abs() < 1e-6);
        assert_eq!(cfg.psychoacoustics.consonance.field.kernel.a, 1.0);
        assert!((cfg.psychoacoustics.consonance.field.kernel.b + 1.35).abs() < 1e-6);
        assert_eq!(cfg.psychoacoustics.consonance.field.kernel.c, 1.0);
        assert_eq!(cfg.psychoacoustics.consonance.field.kernel.d, 0.0);
        assert_eq!(cfg.psychoacoustics.consonance.field.level.beta, 2.0);
        assert_eq!(cfg.psychoacoustics.consonance.field.level.theta, 0.0);
        assert_eq!(cfg.psychoacoustics.consonance.density.roughness_gain, 1.0);
        assert!(!cfg.psychoacoustics.use_incoherent_power);
        assert_eq!(cfg.dcc.coupling_strength, 0.0);
        assert_eq!(cfg.dcc.max_temperature_bonus, 0.10);

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
            contents.contains("# a = 1.0"),
            "should write commented consonance.field.kernel.a"
        );
        assert!(
            contents.contains("# beta = 2.0"),
            "should write commented consonance.field.level.beta"
        );
        assert!(
            contents.contains("# use_incoherent_power = false"),
            "should write commented use_incoherent_power"
        );
        assert!(
            contents.contains("# coupling_strength = 0.0"),
            "should write commented dcc.coupling_strength"
        );

        // Written defaults must survive a reload under deny_unknown_fields.
        AppConfig::load_or_default(&path_str).expect("written defaults must reload");

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
                consonance: ConsonanceConfig {
                    field: ConsonanceFieldConfig {
                        kernel: ConsonanceKernelConfig {
                            a: 0.9,
                            b: -0.7,
                            c: 0.4,
                            d: 0.2,
                        },
                        level: ConsonanceLevelConfig {
                            beta: 3.25,
                            theta: -0.15,
                        },
                    },
                    density: ConsonanceDensityConfig {
                        roughness_gain: 0.5,
                    },
                },
                habituation: HabituationConfig::default(),
                use_incoherent_power: false,
            },
            playback: PlaybackConfig {
                wait_user_exit: false,
                wait_user_start: true,
            },
            dcc: DccConfig {
                coupling_strength: 0.25,
                max_temperature_bonus: 0.12,
            },
        };
        let text = toml::to_string_pretty(&custom).unwrap();
        fs::write(&path, text).unwrap();

        let cfg = AppConfig::load_or_default(&path_str).expect("load existing config");
        assert_eq!(cfg.audio.latency_ms, 75.0);
        assert_eq!(cfg.audio.sample_rate, 44_100);
        assert_eq!(cfg.audio.limiter, LimiterSetting::SoftClip);
        assert_eq!(cfg.analysis.nfft, 8192);
        assert_eq!(cfg.analysis.hop_size, 256);
        assert_eq!(cfg.analysis.tau_ms, 60.0);
        assert_eq!(cfg.psychoacoustics.loudness_exp, 0.3);
        assert_eq!(cfg.psychoacoustics.roughness_k, 0.2);
        assert_eq!(cfg.psychoacoustics.consonance.field.kernel.a, 0.9);
        assert_eq!(cfg.psychoacoustics.consonance.field.kernel.b, -0.7);
        assert_eq!(cfg.psychoacoustics.consonance.field.kernel.c, 0.4);
        assert_eq!(cfg.psychoacoustics.consonance.field.kernel.d, 0.2);
        assert_eq!(cfg.psychoacoustics.consonance.field.level.beta, 3.25);
        assert_eq!(cfg.psychoacoustics.consonance.field.level.theta, -0.15);
        assert_eq!(cfg.psychoacoustics.consonance.density.roughness_gain, 0.5);
        assert!(!cfg.psychoacoustics.use_incoherent_power);
        assert_eq!(cfg.dcc.coupling_strength, 0.25);
        assert_eq!(cfg.dcc.max_temperature_bonus, 0.12);
        assert!(!cfg.playback.wait_user_exit);
        assert!(cfg.playback.wait_user_start);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn rejects_unknown_top_level_key() {
        let path = unique_path("unknown_top.toml");
        let path_str = path.to_string_lossy().to_string();
        fs::write(&path, "[features]\nmulti_agent = true\n").unwrap();

        let err = AppConfig::load_or_default(&path_str).expect_err("unknown section must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains(&path_str), "error must name the file: {msg}");
        assert!(
            msg.contains("unknown field `features`"),
            "error must name the offending key: {msg}"
        );

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn rejects_unknown_nested_key() {
        let path = unique_path("unknown_nested.toml");
        let path_str = path.to_string_lossy().to_string();
        fs::write(&path, "[audio]\nlatency_msec = 30.0\n").unwrap();

        let err = AppConfig::load_or_default(&path_str).expect_err("mistyped key must fail");
        let msg = format!("{err:#}");
        assert!(msg.contains(&path_str), "error must name the file: {msg}");
        assert!(
            msg.contains("unknown field `latency_msec`"),
            "error must name the offending key: {msg}"
        );

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn missing_keys_fall_back_to_defaults() {
        let path = unique_path("partial.toml");
        let path_str = path.to_string_lossy().to_string();
        fs::write(&path, "[audio]\nsample_rate = 44100\n").unwrap();

        let cfg = AppConfig::load_or_default(&path_str).expect("partial config must load");
        assert_eq!(cfg.audio.sample_rate, 44_100);
        assert_eq!(cfg.audio.latency_ms, AudioConfig::default_latency_ms());
        assert_eq!(cfg.audio.limiter, LimiterSetting::default());
        assert_eq!(cfg.analysis.nfft, AnalysisConfig::default_nfft());
        assert_eq!(
            cfg.psychoacoustics.loudness_exp,
            PsychoAcousticsConfig::default_loudness_exp()
        );
        assert!(cfg.playback.wait_user_exit);

        let _ = fs::remove_file(&path);
    }

    #[test]
    fn rejects_invalid_runtime_dimensions_when_loading() {
        for (name, contents, key) in [
            (
                "zero_rate",
                "[audio]\nsample_rate = 0\n",
                "audio.sample_rate",
            ),
            (
                "zero_latency",
                "[audio]\nlatency_ms = 0.0\n",
                "audio.latency_ms",
            ),
            (
                "negative_latency",
                "[audio]\nlatency_ms = -1.0\n",
                "audio.latency_ms",
            ),
            (
                "nan_latency",
                "[audio]\nlatency_ms = nan\n",
                "audio.latency_ms",
            ),
            (
                "infinite_latency",
                "[audio]\nlatency_ms = inf\n",
                "audio.latency_ms",
            ),
            (
                "huge_latency",
                "[audio]\nlatency_ms = 1e30\n",
                "audio.latency_ms",
            ),
            ("zero_fft", "[analysis]\nnfft = 0\n", "analysis.nfft"),
            ("capped_fft", "[analysis]\nnfft = 262145\n", "analysis.nfft"),
            (
                "zero_hop",
                "[analysis]\nhop_size = 0\n",
                "analysis.hop_size",
            ),
            (
                "oversized_hop",
                "[analysis]\nnfft = 256\nhop_size = 257\n",
                "analysis.hop_size",
            ),
            (
                "excess_overlap",
                "[analysis]\nnfft = 16384\nhop_size = 163\n",
                "analysis.hop_size",
            ),
            (
                "zero_duration",
                "[audio]\nsample_rate = 4294967295\n[analysis]\nnfft = 1\nhop_size = 1\n",
                "analysis.hop_size",
            ),
        ] {
            let path = unique_path(name);
            fs::write(&path, contents).unwrap();
            let err = AppConfig::load_or_default(path.to_str().unwrap())
                .expect_err("invalid dimensions must fail before runtime startup");
            let message = format!("{err:#}");
            assert!(message.contains(key), "{name}: {message}");
            assert!(message.contains(path.to_str().unwrap()), "{message}");
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn accepts_supported_fft_and_overlap_boundaries() {
        let mut cfg = AppConfig::default();
        cfg.validate().unwrap();
        for (nfft, hop_size) in [(16384, 164), (256, 256), (262144, 2622), (100, 2)] {
            cfg.analysis.nfft = nfft;
            cfg.analysis.hop_size = hop_size;
            cfg.validate().unwrap();
        }
    }

    #[test]
    fn repo_config_toml_has_no_unknown_keys() {
        // config.toml is gitignored, so a fresh checkout may not have one.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config.toml");
        if !path.exists() {
            return;
        }
        let contents = fs::read_to_string(&path).expect("read repo config.toml");
        toml::from_str::<AppConfig>(&contents).expect("repo config.toml must have no unknown keys");
    }

    #[test]
    fn parse_nested_consonance_density_keys() {
        let text = r#"
[psychoacoustics.consonance.density]
roughness_gain = 0.5
"#;
        let parsed: AppConfig = toml::from_str(text).expect("parse consonance density keys");
        assert_eq!(
            parsed.psychoacoustics.consonance.density.roughness_gain,
            0.5
        );
    }

    #[test]
    fn habituation_defaults_off_and_parses() {
        let cfg = AppConfig::default();
        assert!(!cfg.psychoacoustics.habituation.enabled);
        assert_eq!(cfg.psychoacoustics.habituation.satiation_sec, 5.0);
        assert_eq!(cfg.psychoacoustics.habituation.recovery_sec, 8.0);
        assert_eq!(cfg.psychoacoustics.habituation.ref_drive, 0.25);

        let text = "\
[psychoacoustics.habituation]
enabled = true
satiation_sec = 3.0
recovery_sec = 6.0
ref_drive = 0.4
";
        let parsed: AppConfig = toml::from_str(text).expect("parse habituation keys");
        assert!(parsed.psychoacoustics.habituation.enabled);
        assert_eq!(parsed.psychoacoustics.habituation.satiation_sec, 3.0);
    }

    #[test]
    fn round_f32_inplace_clamps_negative_density_roughness_gain_to_zero() {
        let text = r#"
[psychoacoustics.consonance.density]
roughness_gain = -1.0
"#;
        let mut parsed: AppConfig =
            toml::from_str(text).expect("parse consonance density negative roughness gain");
        parsed.round_f32_inplace();
        assert_eq!(
            parsed.psychoacoustics.consonance.density.roughness_gain,
            0.0
        );
    }

    #[test]
    fn round_f32_inplace_maps_nan_density_roughness_gain_to_one() {
        let mut cfg = AppConfig::default();
        cfg.psychoacoustics.consonance.density.roughness_gain = f32::NAN;
        cfg.round_f32_inplace();
        assert_eq!(cfg.psychoacoustics.consonance.density.roughness_gain, 1.0);
    }

    #[test]
    fn round_f32_inplace_sanitizes_dcc_coupling() {
        let mut cfg = AppConfig::default();
        cfg.dcc.coupling_strength = 2.0;
        cfg.dcc.max_temperature_bonus = f32::NAN;
        cfg.round_f32_inplace();
        assert_eq!(cfg.dcc.coupling_strength, 1.0);
        assert_eq!(
            cfg.dcc.max_temperature_bonus,
            DccConfig::default_max_temperature_bonus()
        );
    }
}
