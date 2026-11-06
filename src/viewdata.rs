use std::sync::Arc;

use crate::core::landscape::Landscape;

pub use conchordal_web_ui::frame::WaveFrame;

pub use conchordal_web_ui::frame::SpecFrame;

pub use conchordal_web_ui::frame::DorsalFrame;

pub(crate) use conchordal_web_ui::frame::RHYTHM_HISTORY_SEC;

/// Presentation observations at hop resolution, before UI rate limiting.
pub use conchordal_web_ui::frame::ListenerRhythmHop;

/// E0 categories: unchanged, audible entry, lower peak, higher peak, silence.
/// The point forecast q backs off to the supported marginal; p_known defines entropy.
/// Model scores are not calibrated human probabilities.
pub use conchordal_web_ui::frame::ListenerForecast;

pub use conchordal_web_ui::frame::ListenerOutcome;

pub use conchordal_web_ui::frame::ListenerCensored;

pub use conchordal_web_ui::frame::ListenerPrediction;

pub use conchordal_web_ui::frame::ListenerFrame;

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

pub use conchordal_web_ui::frame::PlaybackState;

#[derive(Clone, Debug, Default)]
pub struct SimulationMeta {
    pub(crate) script: Option<crate::scripting::ScriptStatus>,
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

pub use conchordal_web_ui::frame::PredictionFrame;

#[derive(Clone, Debug, Default)]
pub struct UiFrame {
    pub(crate) body: Option<Arc<crate::temporal_cognition::body::Snapshot>>,
    pub(crate) temporal: Arc<[crate::temporal_cognition::observation::Snapshot; 2]>,
    pub wave: WaveFrame,
    pub spec: SpecFrame,
    pub listener: ListenerFrame,
    pub(crate) listener_rhythm: Vec<ListenerRhythmHop>,
    pub landscape: Landscape,
    pub meta: SimulationMeta,
    pub prediction: PredictionFrame,
    pub voices: Vec<VoiceStateInfo>,
}

#[path = "viewdata_web.rs"]
pub(crate) mod web;
