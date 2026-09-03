#[derive(Debug, Clone, Copy)]
pub struct ControlRamp {
    pub start: f32,
    pub step: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct ToneControlBlock {
    pub pitch_hz: ControlRamp,
    pub amp: ControlRamp,
}
