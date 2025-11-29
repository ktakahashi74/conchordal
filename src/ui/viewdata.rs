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
pub struct UiFrame {
    pub wave: WaveFrame,
    pub spec: SpecFrame,
    pub landscape: LandscapeFrame,
}
