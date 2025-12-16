use crate::core::landscape::LandscapeFrame;

#[derive(Clone, Debug, Default)]
pub struct WaveFrame {
    pub fs: f32,
    pub samples: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct SpecFrame {
    pub spec_hz: Vec<f32>,
    pub amps: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct AgentStateInfo {
    pub id: u64,
    pub freq_hz: f32,
    pub target_freq: f32,
    pub integration_window: f32,
    pub breath_gain: f32,
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
    pub landscape: LandscapeFrame,
    pub time_sec: f32,
    pub meta: SimulationMeta,
    pub agents: Vec<AgentStateInfo>,
}
