//! Display-only wire contract. Version 1 carries no engine state or audio output API.
use std::sync::Arc;
pub const RHYTHM_HISTORY_SEC: f64 = 30.;
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct WaveFrame {
    pub fs: f32,
    pub samples: Arc<[f32]>,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct SpecFrame {
    pub spec_hz: Vec<f32>,
    pub amps: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct DorsalFrame {
    pub e_low: f32,
    pub e_mid: f32,
    pub e_high: f32,
    pub flux: f32,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ListenerRhythmHop {
    pub time_sec: f64,
    pub attention: DorsalFrame,
    pub meter: MeterState,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ListenerForecast {
    pub version: u64,
    pub start_sample: u64,
    pub end_sample: u64,
    pub reference_sample: u64,
    pub reference_bin: Option<usize>,
    pub horizon_sec: f32,
    pub p_known: Option<[f32; 5]>,
    pub q: Option<[f32; 5]>,
    pub marginal: Option<[f32; 5]>,
    pub entropy_nats: Option<f32>,
    pub coverage_score: f32,
    pub arrival_score: Option<f32>,
    pub examples: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ListenerOutcome {
    pub forecast: ListenerForecast,
    pub outcome: usize,
    pub observed_sample: u64,
    pub surprise_nats: Option<f32>,
    pub zero_likelihood: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ListenerCensored {
    pub forecast: ListenerForecast,
    pub observed_sample: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ListenerPrediction {
    pub forecast: Option<ListenerForecast>,
    pub completed: Option<ListenerOutcome>,
    pub censored: Option<ListenerCensored>,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ListenerFrame {
    pub prediction: ListenerPrediction,
    pub time_sec: f64,
    pub generated_frame_id: u64,
    pub analysis_frame_id: u64,
    pub analysis_lag_frames: u64,
    pub attention_level: f32,
    pub attention: DorsalFrame,
    pub meter: MeterState,
    pub stability_level: f32,
    pub nearby_improvement_level: f32,
    pub static_pull_level: f32,
    pub has_fast_state: bool,
    pub has_state: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PlaybackState {
    #[default]
    NotStarted,
    Playing,
    Ending,
    Finished,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct PredictionFrame {
    pub next_gate_tick_est: Option<u64>,
    pub theta_hz: Option<f32>,
    pub delta_hz: Option<f32>,
    pub pred_n_theta_per_delta: Option<u32>,
    pub pred_tau_tick: Option<u64>,
    pub pred_horizon_tick: Option<u64>,
    pub pred_c_field_level_next_gate: Option<Arc<[f32]>>,
    pub pred_gain_raw_mean: Option<f32>,
    pub pred_gain_raw_min: Option<f32>,
    pub pred_gain_raw_max: Option<f32>,
    pub pred_gain_mixed_mean: Option<f32>,
    pub pred_gain_mixed_min: Option<f32>,
    pub pred_gain_mixed_max: Option<f32>,
    pub pred_sync_mean: Option<f32>,
    pub gate_boundary_in_hop: Option<bool>,
    pub pred_available_in_hop: Option<bool>,
    pub phonation_onsets_in_hop: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct MeterBand {
    /// Wrapped oscillator phase in [-pi, pi].
    pub phase: f32,
    /// Current frequency in Hz.
    pub freq_hz: f32,
    /// Limit-cycle amplitude (presence). Not salience.
    pub amplitude: f32,
    /// Entrainment confidence in [0, 1] from phase and pattern recurrence. This is the
    /// salience: a clean quiet beat reads high, a loud wandering beat reads low.
    pub confidence: f32,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct MeterState {
    /// Tactus (delta-rate beat).
    pub beat: MeterBand,
    /// Tatum (theta-rate), mode-locked to the beat at the emergent ratio.
    pub subdivision: MeterBand,
    /// Measure: a slow subharmonic of the beat, induced by periodic accent.
    pub measure: MeterBand,
    /// Onset / flux drive salience (the force term), kept distinct from
    /// confidence so reports and UI never blur drive with entrainment.
    pub attention_level: f32,
    /// Emergent integer subdivision ratio (2, 3, or 4); 0 when none is detected.
    pub subdivision_ratio: u8,
    /// Emergent beats-per-measure (2, 3, or 4); 0 when no accent grouping holds.
    pub measure_ratio: u8,
}
impl Default for WaveFrame {
    fn default() -> Self {
        Self {
            fs: 0.0,
            samples: Arc::from(Vec::<f32>::new()),
        }
    }
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct SimulationMeta {
    pub script: Option<ScriptStatus>,
    pub time_sec: f64,
    pub end: Option<f64>,
    pub population_count: usize,
    pub slow_hops: u64,
    pub voice_count: usize,
    pub event_queue_len: usize,
    pub estimator_freezes: u64,
    pub estimator_last_frozen_issue: u64,
    pub audio_xruns: u64,
    pub audio_rebuilds: u64,
    pub audio_recovering: bool,
    pub peak_level: f32,
    pub scenario_name: String,
    pub scene_name: Option<String>,
    pub playback_state: PlaybackState,
    pub channel_peak: [f32; 2],
    pub window_peak: [f32; 2],
    pub kuramoto_order_r: Option<f32>,
    pub kuramoto_active_count: usize,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ScriptStatus {
    pub evaluated_through: f64,
    pub done: bool,
    pub stopped: bool,
    pub error: Option<String>,
    pub error_position: PositionLabel,
    pub warnings: Vec<ScriptWarning>,
}
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct ScriptWarning {
    pub position: PositionLabel,
    pub message: String,
    pub count: u64,
}
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Diagnostic {
    Heading(String),
    Label(String),
    Separator,
}

#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Space {
    pub centers_hz: Vec<f32>,
    pub centers_log2: Vec<f32>,
    pub step_log2: f32,
}
impl Space {
    pub fn step(&self) -> f32 {
        self.step_log2
    }
    pub fn n_bins(&self) -> usize {
        self.centers_hz.len()
    }
}
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Geometry {
    pub root_start_log2: f32,
    pub space: Space,
}
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Snapshot {
    pub geometry: Geometry,
    pub p: Vec<f64>,
    pub mass: f64,
    pub focus: f64,
    pub conflict: f64,
    pub end_sample: u64,
}
#[derive(Clone, Copy, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct ConsonanceKernel {
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
}
impl ConsonanceKernel {
    pub fn score(&self, h: f32, r: f32) -> f32 {
        let finite = |x: f32| if x.is_finite() { x } else { 0. };
        let h = finite(h).clamp(0., 1.);
        let r = finite(r).clamp(0., 1.);
        finite(finite(self.a) * h + finite(self.b) * r + finite(self.c) * h * r + finite(self.d))
    }
}
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Landscape {
    pub space: Space,
    pub support: Option<Arc<Snapshot>>,
    pub subjective_intensity: Vec<f32>,
    pub nsgt_power: Vec<f32>,
    pub consonance_field_score: Vec<f32>,
    pub consonance_field_level: Vec<f32>,
    pub harmonicity01: Vec<f32>,
    pub roughness01: Vec<f32>,
    pub consonance_kernel: ConsonanceKernel,
}
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct VoiceStateInfo {
    pub id: u64,
    pub freq_hz: f32,
    pub target_freq: f32,
    pub integration_window: f32,
    pub consonance: f32,
    pub support_template: Option<Arc<[(isize, f32)]>>,
    pub support_scans: Option<[Vec<f64>; 3]>,
}
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct UiFrame {
    pub wave: WaveFrame,
    pub spec: SpecFrame,
    pub listener: ListenerFrame,
    pub listener_rhythm: Vec<ListenerRhythmHop>,
    pub landscape: Landscape,
    pub meta: SimulationMeta,
    pub prediction: PredictionFrame,
    pub voices: Vec<VoiceStateInfo>,
    pub diagnostics: Vec<Diagnostic>,
}

#[derive(serde::Serialize, serde::Deserialize)]
pub struct Envelope {
    pub version: u32,
    pub frame: UiFrame,
}
impl UiFrame {
    pub fn encode(self) -> Result<Vec<u8>, serde_json::Error> {
        serde_json::to_vec(&Envelope {
            version: 1,
            frame: self,
        })
    }
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        let packet: Envelope = serde_json::from_slice(bytes).map_err(|e| e.to_string())?;
        if packet.version != 1 {
            return Err(format!(
                "Unsupported UI snapshot version {}",
                packet.version
            ));
        }
        Ok(packet.frame)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_unknown_version_and_truncated_packet() {
        let frame = UiFrame::default();
        let bytes = serde_json::to_vec(&Envelope { version: 2, frame }).unwrap();
        assert!(UiFrame::decode(&bytes).unwrap_err().contains("version 2"));
        assert!(UiFrame::decode(&bytes[..bytes.len() / 2]).is_err());
    }
}

/// Already formatted by the producer; preserve Rhai's diagnostic spelling.
#[derive(Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct PositionLabel(pub String);
impl std::fmt::Debug for PositionLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
