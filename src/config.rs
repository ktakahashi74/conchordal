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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_ridge: Option<TemporalRidgeConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_acoustic: Option<TemporalAcousticConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_memory: Option<TemporalMemoryConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_gesture: Option<TemporalGestureConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_period: Option<TemporalPeriodConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_body: Option<TemporalBodyConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_body_prototypes: Option<TemporalBodyPrototypesConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_action_profiles: Option<TemporalActionProfilesConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_private_trace: Option<TemporalPrivateTraceConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal_onset_comparison: Option<TemporalOnsetComparisonConfig>,
}

/// Which footprint the participation onset comparison consumes (I11-1 §4.10).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum FootprintSource {
    #[default]
    Body,
    Proxy,
}

/// Absent keeps the legacy 64-point proxy path; present selects the bounded comparison.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalOnsetComparisonConfig {
    #[serde(default)]
    pub footprint: FootprintSource,
    #[serde(default)]
    pub arrival: bool,
    #[serde(default = "TemporalOnsetComparisonConfig::default_arrival_weight")]
    pub arrival_weight: f64,
}

impl TemporalOnsetComparisonConfig {
    fn default_arrival_weight() -> f64 {
        1.0
    }
}

impl Default for TemporalOnsetComparisonConfig {
    fn default() -> Self {
        Self {
            footprint: FootprintSource::default(),
            arrival: false,
            arrival_weight: Self::default_arrival_weight(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalPrivateTraceConfig {
    pub tau_sec: f64,
    pub kappa: f64,
    pub strength_max: f64,
}

/// Frozen development scales for private body diagnostics; no adopted calibration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalBodyConfig {
    pub means: [f64; 6],
    pub deviations: [f64; 6],
    pub accent_means: [f64; 2],
    pub accent_deviations: [f64; 2],
}

/// Frozen descriptive medoids; these do not authorize counterfactual action projections.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalBodyPrototypesConfig {
    pub model_version: String,
    pub sample_rate: u32,
    pub nfft: usize,
    pub hop_size: usize,
    pub means: [f64; 6],
    pub deviations: [f64; 6],
    pub accent_means: [f64; 2],
    pub accent_deviations: [f64; 2],
    pub medoids: Vec<TemporalBodyMedoid>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalBodyMedoid {
    pub record_id: String,
    pub raw_values: [f64; 6],
    pub mask: u8,
}

/// Frozen conditional descriptor-transfer data; no action or calibration is enabled.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalActionProfilesConfig {
    pub file: String,
    pub sha256: String,
    /// Omitted: exact action time. Set: delay, then round up on the issue-relative hop grid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluation_delay_ms: Option<u32>,
}

/// Explicit research scales for passive ridge diagnostics, not an adopted fit.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalRidgeConfig {
    pub means: [f64; 3],
    pub deviations: [f64; 3],
}

/// Frozen inputs for uncalibrated acoustic diagnostics; no implicit fit scales.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalAcousticConfig {
    pub group_means: [f64; 3],
    pub group_deviations: [f64; 3],
    pub accent_means: [f64; 2],
    pub accent_deviations: [f64; 2],
    pub group_retirement_sec: f64,
    pub inactive_energy_max: f64,
    pub correlation_window_sec: f64,
    pub min_pairs: usize,
    pub min_coverage: f64,
    pub persistence_hops: u8,
}

/// Explicit assay windows and scales for passive, uncalibrated memory diagnostics.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalMemoryConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retention: Option<TemporalRetentionConfig>,
    /// Explicit search capacity; absent selects the frozen 16-candidate baseline.
    pub candidates: Option<usize>,
    pub scales: [f64; 10],
    /// Assay span, and with it the sealed episode length: a span is sealed into one episode
    /// once this many observed hops accumulate. Short spans seal often and evict the bank
    /// before a retained reference can be reused.
    pub span_hops: u64,
    /// Episode bank capacity. It bounds how far back a reference can still be retained.
    pub episodes: usize,
    pub query_cadence_ms: u64,
    pub deadline_ms: u64,
}

/// Explicit diagnostic retention parameters; no implicit stage-1 fit.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalRetentionConfig {
    pub tau_sec: f64,
    pub kappa: f64,
    pub strength_max: f64,
    pub r_max: f64,
    pub no_memory_bias: f64,
    /// Explicit match rule: score = -(distance / match_temperature) - edit_penalty * edits,
    /// where distance sums the coordinate RMS residuals over `scales`, the motion RMS over
    /// `motion_scale` and the interval RMS over `interval_scale`. No fitted coefficients.
    pub match_temperature: f64,
    pub edit_penalty: f64,
    pub motion_scale: f64,
    pub interval_scale: f64,
}

/// Frozen research inputs for the single-context articulation/gesture diagnostic.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalGestureConfig {
    pub rms_reference: f64,
    pub means: [f64; 5],
    pub deviations: [f64; 5],
    pub coefficients: [[[f64; 11]; 4]; 4],
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArrivalModel {
    Hazard,
    Periodic,
}

/// Explicit uncalibrated arrival parameters; coefficient layout is versioned with the model.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalPeriodConfig {
    pub model: ArrivalModel,
    pub coefficients: [f64; 18],
    pub means: [f64; 8],
    pub deviations: [f64; 8],
    pub horizon_sec: f64,
}

pub(crate) mod fixed_array {
    use serde::Serialize;

    pub fn serialize<T: Serialize, S: serde::Serializer, const N: usize>(
        values: &[T; N],
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        values.as_slice().serialize(serializer)
    }
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
        crate::temporal_cognition::action_profiles::validate_config(self)?;
        if let Some(ridge) = self.temporal_ridge {
            ensure!(
                ridge.means.iter().all(|v| v.is_finite())
                    && ridge.deviations.iter().all(|v| v.is_finite() && *v >= 0.0),
                "temporal_ridge requires finite means and finite nonnegative deviations"
            );
        }
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
        if let Some(body) = self.temporal_body {
            crate::temporal_cognition::body::validate(body).map_err(anyhow::Error::msg)?;
            if let Some(acoustic) = self.temporal_acoustic {
                ensure!(
                    body.accent_means == acoustic.accent_means
                        && body.accent_deviations == acoustic.accent_deviations,
                    "temporal_body and temporal_acoustic must share frozen accent scales"
                );
            }
        }
        crate::temporal_cognition::body_model::validate(self)?;
        if let Some(acoustic) = self.temporal_acoustic {
            let ridge = self
                .temporal_ridge
                .context("temporal_acoustic requires temporal_ridge scales")?;
            crate::temporal_cognition::proposals::frontend::Frontend::configured(
                crate::core::log2space::Log2Space::new(100.0, 200.0, 1),
                0,
                (0, 0),
                self.audio.sample_rate,
                self.analysis.hop_size as u64,
                ridge,
                acoustic,
            )
            .map_err(anyhow::Error::msg)?;
        }
        if let Some(memory) = self.temporal_memory {
            ensure!(
                self.temporal_acoustic.is_some(),
                "temporal_memory requires temporal_acoustic"
            );
            crate::temporal_cognition::recall::Recall::new(
                0,
                0,
                self.audio.sample_rate,
                self.analysis.hop_size as u64,
                memory,
            )
            .map_err(anyhow::Error::msg)?;
        }
        if let Some(gesture) = self.temporal_gesture {
            ensure!(
                self.temporal_acoustic.is_some(),
                "temporal_gesture requires temporal_acoustic"
            );
            crate::temporal_cognition::gesture::Gesture::new(
                0,
                0,
                self.audio.sample_rate,
                self.analysis.hop_size as u64,
                gesture,
            )
            .map_err(anyhow::Error::msg)?;
        }
        if let Some(trace) = self.temporal_private_trace {
            anyhow::ensure!(
                self.temporal_memory.is_some_and(|m| m.retention.is_some()),
                "temporal_private_trace requires temporal_memory.retention"
            );
            anyhow::ensure!(
                [trace.tau_sec, trace.kappa, trace.strength_max]
                    .iter()
                    .all(|v| v.is_finite() && *v > 0.),
                "temporal_private_trace requires finite positive parameters"
            );
        }
        if let Some(period) = self.temporal_period {
            ensure!(
                self.temporal_acoustic.is_some(),
                "temporal_period requires temporal_acoustic"
            );
            crate::temporal_cognition::arrival::Engine::new(period).map_err(anyhow::Error::msg)?;
        }
        if let Some(onset) = self.temporal_onset_comparison {
            ensure!(
                onset.arrival_weight.is_finite() && (0.0..=4.0).contains(&onset.arrival_weight),
                "temporal_onset_comparison.arrival_weight must lie within 0..=4"
            );
        }
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

    #[test]
    fn ridge_diagnostics_require_explicit_finite_scales() {
        assert!(AppConfig::default().temporal_ridge.is_none());
        let config = "[temporal_ridge]\nmeans = [0.0, 0.0, 0.0]\ndeviations = [0.05, 4.0, 1.0]\n";
        let valid: AppConfig = toml::from_str(config).unwrap();
        valid.validate().unwrap();
        for text in [
            "[temporal_ridge]\nmeans = [0.0, 0.0, 0.0]\n",
            "[temporal_ridge]\ndeviations = [0.05, 4.0, 1.0]\n",
            "[temporal_ridge]\nmeans = [0.0, 0.0, 0.0]\ndeviatons = [0.05, 4.0, 1.0]\n",
        ] {
            assert!(toml::from_str::<AppConfig>(text).is_err());
        }
        for value in [f64::NAN, f64::INFINITY, -1.0] {
            let mut invalid = valid.clone();
            invalid.temporal_ridge.as_mut().unwrap().deviations[0] = value;
            assert!(invalid.validate().is_err());
        }
        let restored: AppConfig = toml::from_str(&toml::to_string(&valid).unwrap()).unwrap();
        assert_eq!(
            restored.temporal_ridge.unwrap().deviations,
            [0.05, 4.0, 1.0]
        );
    }

    #[test]
    fn onset_comparison_defaults_to_body_and_bounds_arrival_weight() {
        assert!(AppConfig::default().temporal_onset_comparison.is_none());
        let bare: AppConfig = toml::from_str("[temporal_onset_comparison]\n").unwrap();
        bare.validate().unwrap();
        let bare = bare.temporal_onset_comparison.unwrap();
        assert_eq!(bare.footprint, FootprintSource::Body);
        assert!(!bare.arrival);
        assert_eq!(bare.arrival_weight, 1.0);

        let proxy: AppConfig =
            toml::from_str("[temporal_onset_comparison]\nfootprint = \"proxy\"\n").unwrap();
        assert_eq!(
            proxy.temporal_onset_comparison.unwrap().footprint,
            FootprintSource::Proxy
        );
        assert!(
            toml::from_str::<AppConfig>("[temporal_onset_comparison]\nfootprnt = \"proxy\"\n")
                .is_err()
        );

        for weight in [5.0, -0.5, f64::NAN] {
            let mut invalid: AppConfig = toml::from_str("[temporal_onset_comparison]\n").unwrap();
            invalid
                .temporal_onset_comparison
                .as_mut()
                .unwrap()
                .arrival_weight = weight;
            assert!(
                invalid.validate().is_err(),
                "weight {weight} must be rejected"
            );
        }
    }

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
            temporal_ridge: None,
            temporal_acoustic: None,
            temporal_memory: None,
            temporal_gesture: None,
            temporal_period: None,
            temporal_body: None,
            temporal_body_prototypes: None,
            temporal_action_profiles: None,
            temporal_private_trace: None,
            temporal_onset_comparison: None,
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
    fn temporal_diagnostics_require_explicit_dependencies_and_valid_scales() {
        let acoustic = TemporalAcousticConfig {
            group_means: [0.; 3],
            group_deviations: [0.05, 4., 1.],
            accent_means: [0.; 2],
            accent_deviations: [1.; 2],
            group_retirement_sec: 2.,
            inactive_energy_max: 1e-8,
            correlation_window_sec: 0.25,
            min_pairs: 8,
            min_coverage: 0.9,
            persistence_hops: 3,
        };
        let memory = TemporalMemoryConfig {
            retention: None,
            candidates: None,
            scales: [1.; 10],
            span_hops: 128,
            episodes: 16,
            query_cadence_ms: 100,
            deadline_ms: 200,
        };
        let mut cfg = AppConfig {
            temporal_memory: Some(memory),
            ..Default::default()
        };
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("temporal_acoustic")
        );
        cfg.temporal_acoustic = Some(acoustic);
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("temporal_ridge")
        );
        cfg.temporal_ridge = Some(TemporalRidgeConfig {
            means: [0.; 3],
            deviations: [0.05, 4., 1.],
        });
        cfg.validate().unwrap();
        for change in 0..9 {
            let mut bad = cfg.clone();
            let m = bad.temporal_memory.as_mut().unwrap();
            match change {
                0 => m.scales[0] = f64::NAN,
                1 => m.scales[0] = 0.,
                2 => m.span_hops = 0,
                3 => m.episodes = 16385,
                4 => m.query_cadence_ms = 75,
                5 => m.deadline_ms = u64::MAX,
                7 => m.candidates = Some(0),
                8 => m.candidates = Some(1025),
                _ => {
                    bad.temporal_acoustic
                        .as_mut()
                        .unwrap()
                        .correlation_window_sec = -1.
                }
            }
            assert!(bad.validate().is_err(), "accepted invalid case {change}");
        }
        let text = toml::to_string(&cfg).unwrap();
        let parsed: AppConfig = toml::from_str(&text).unwrap();
        parsed.validate().unwrap();
        assert_eq!(parsed.temporal_memory.unwrap().scales, memory.scales);
        assert!(toml::from_str::<AppConfig>(&text.replace("span_hops", "span_hpos")).is_err());
    }

    #[test]
    fn period_configuration_requires_explicit_valid_model_inputs() {
        let period = TemporalPeriodConfig {
            model: ArrivalModel::Hazard,
            coefficients: [0.; 18],
            means: [0.; 8],
            deviations: [1.; 8],
            horizon_sec: 0.1,
        };
        let cfg = AppConfig {
            temporal_period: Some(period),
            ..Default::default()
        };
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("temporal_acoustic")
        );
        for case in 0..5 {
            let mut bad = period;
            match case {
                0 => bad.coefficients[17] = f64::INFINITY,
                1 => bad.means[0] = f64::NAN,
                2 => bad.deviations[0] = -1.,
                3 => bad.horizon_sec = 0.,
                _ => bad.horizon_sec = 33.,
            }
            assert!(crate::temporal_cognition::arrival::Engine::new(bad).is_err());
        }
        let text = toml::to_string(&period).unwrap();
        let parsed: TemporalPeriodConfig = toml::from_str(&text).unwrap();
        assert_eq!(parsed.model, period.model);
        assert_eq!(parsed.coefficients, period.coefficients);
        assert!(
            toml::from_str::<TemporalPeriodConfig>(&text.replace("hazard", "other_model")).is_err()
        );
        assert!(
            toml::from_str::<TemporalPeriodConfig>(&text.replace("horizon_sec", "horizon_secs"))
                .is_err()
        );
    }

    #[test]
    fn gesture_configuration_requires_frozen_finite_inputs() {
        let gesture = TemporalGestureConfig {
            rms_reference: 0.1,
            means: [0.; 5],
            deviations: [1.; 5],
            coefficients: [[[0.; 11]; 4]; 4],
        };
        let cfg = AppConfig {
            temporal_gesture: Some(gesture),
            ..Default::default()
        };
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("temporal_acoustic")
        );
        for case in 0..4 {
            let mut bad = gesture;
            match case {
                0 => bad.rms_reference = 0.,
                1 => bad.coefficients[0][1][3] = f64::INFINITY,
                2 => bad.means[0] = f64::NAN,
                _ => bad.deviations[0] = -1.,
            }
            assert!(
                crate::temporal_cognition::gesture::Gesture::new(0, 0, 48000, 512, bad).is_err()
            );
        }
        let text = toml::to_string(&gesture).unwrap();
        let parsed: TemporalGestureConfig = toml::from_str(&text).unwrap();
        assert_eq!(parsed.coefficients, gesture.coefficients);
        assert!(
            toml::from_str::<TemporalGestureConfig>(&text.replace("rms_reference", "rms_refernce"))
                .is_err()
        );
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
