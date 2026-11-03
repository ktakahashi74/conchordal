use std::sync::Arc;

use crate::core::landscape::Landscape;
use crate::core::meter::MeterState;
use crate::core::timebase::Tick;

#[derive(Clone, Debug)]
pub struct WaveFrame {
    pub fs: f32,
    pub samples: Arc<[f32]>,
}

#[derive(Clone, Debug, Default)]
pub struct SpecFrame {
    pub spec_hz: Vec<f32>,
    pub amps: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DorsalFrame {
    pub e_low: f32,
    pub e_mid: f32,
    pub e_high: f32,
    pub flux: f32,
}

pub(crate) const RHYTHM_HISTORY_SEC: f64 = 30.0;

/// Presentation observations at hop resolution, before UI rate limiting.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ListenerRhythmHop {
    pub time_sec: f64,
    pub attention: DorsalFrame,
    pub meter: MeterState,
}

/// E0 categories: unchanged, audible entry, lower peak, higher peak, silence.
/// The point forecast q backs off to the supported marginal; p_known defines entropy.
/// Model scores are not calibrated human probabilities.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
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

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub struct ListenerOutcome {
    pub forecast: ListenerForecast,
    pub outcome: usize,
    pub observed_sample: u64,
    pub surprise_nats: Option<f32>,
    pub zero_likelihood: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub struct ListenerCensored {
    pub forecast: ListenerForecast,
    pub observed_sample: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize)]
pub struct ListenerPrediction {
    pub forecast: Option<ListenerForecast>,
    pub completed: Option<ListenerOutcome>,
    pub censored: Option<ListenerCensored>,
}

#[derive(Clone, Copy, Debug, Default)]
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

#[derive(Clone, Debug, Default)]
pub struct VoiceStateInfo {
    pub(crate) support: Option<std::sync::Arc<crate::core::harmonic_support::Snapshot>>,
    pub(crate) support_single_density: f32,
    pub(crate) support_template: Option<std::sync::Arc<[(isize, f32)]>>,
    pub id: u64,
    pub freq_hz: f32,
    pub target_freq: f32,
    pub integration_window: f32,
    pub consonance: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PlaybackState {
    #[default]
    NotStarted,
    Playing,
    Ending,
    Finished,
}

#[derive(Clone, Debug, Default)]
pub struct SimulationMeta {
    pub script: Option<crate::scripting::ScriptStatus>,
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

#[derive(Clone, Debug, Default)]
pub struct PredictionFrame {
    pub next_gate_tick_est: Option<Tick>,
    pub theta_hz: Option<f32>,
    pub delta_hz: Option<f32>,
    pub pred_n_theta_per_delta: Option<u32>,
    pub pred_tau_tick: Option<Tick>,
    pub pred_horizon_tick: Option<Tick>,
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

#[derive(Clone, Debug, Default)]
pub struct UiFrame {
    pub body: Option<Arc<crate::temporal_cognition::body::Snapshot>>,
    pub temporal: Arc<[crate::temporal_cognition::observation::Snapshot; 2]>,
    pub wave: WaveFrame,
    pub spec: SpecFrame,
    pub listener: ListenerFrame,
    pub(crate) listener_rhythm: Vec<ListenerRhythmHop>,
    pub landscape: Landscape,
    pub meta: SimulationMeta,
    pub prediction: PredictionFrame,
    pub voices: Vec<VoiceStateInfo>,
}

impl Default for WaveFrame {
    fn default() -> Self {
        Self {
            fs: 0.0,
            samples: Arc::from(Vec::<f32>::new()),
        }
    }
}
