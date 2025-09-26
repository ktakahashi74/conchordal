use crate::core::landscape::LandscapeFrame;

#[derive(Clone, Debug, Default)]
pub struct WaveChunk {
    pub fs: f64,
    pub samples: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct UiFrame {
    pub wave: WaveChunk,
    pub spec_hz: Vec<f32>,
    pub amps: Vec<f32>,
    pub landscape: LandscapeFrame,
}

impl UiFrame {
    pub fn empty() -> Self {
        Self::default()
    }
}
