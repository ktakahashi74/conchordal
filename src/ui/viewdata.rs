use std::sync::Arc;

use crate::core::landscape::LandscapeFrame;
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

#[derive(Clone, Debug, Default)]
pub struct AgentStateInfo {
    pub id: u64,
    pub freq_hz: f32,
    pub target_freq: f32,
    pub integration_window: f32,
    pub breath_gain: f32,
    pub consonance: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum PlaybackState {
    #[default]
    NotStarted,
    Playing,
    Finished,
}

#[derive(Clone, Debug, Default)]
pub struct SimulationMeta {
    pub time_sec: f32,
    pub duration_sec: f32,
    pub agent_count: usize,
    pub event_queue_len: usize,
    pub peak_level: f32,
    pub scenario_name: String,
    pub scene_name: Option<String>,
    pub playback_state: PlaybackState,
    pub channel_peak: [f32; 2],
    pub window_peak: [f32; 2],
}

#[derive(Clone, Debug, Default)]
pub struct UiFrame {
    pub wave: WaveFrame,
    pub spec: SpecFrame,
    pub dorsal: DorsalFrame,
    pub landscape: LandscapeFrame,
    pub time_sec: f32,
    pub meta: SimulationMeta,
    pub next_gate_tick_est: Option<Tick>,
    pub theta_hz: Option<f32>,
    pub delta_hz: Option<f32>,
    pub pred_n_theta_per_delta: Option<u32>,
    pub pred_tau_tick: Option<Tick>,
    pub pred_horizon_tick: Option<Tick>,
    pub pred_c_state01_next_gate: Option<Arc<[f32]>>,
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
    pub agents: Vec<AgentStateInfo>,
}

impl Default for WaveFrame {
    fn default() -> Self {
        Self {
            fs: 0.0,
            samples: Arc::from(Vec::<f32>::new()),
        }
    }
}
