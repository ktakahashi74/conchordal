#[derive(Debug, Clone, Copy)]
pub struct ControlRamp {
    pub start: f32,
    pub step: f32,
}

impl ControlRamp {
    pub fn for_len(start: f32, end: f32, len: usize) -> Self {
        let len = len.max(1);
        let step = (end - start) / len as f32;
        Self { start, step }
    }

    pub fn value_at(&self, idx: usize) -> f32 {
        self.start + self.step * idx as f32
    }
}

#[derive(Debug, Clone, Copy)]
pub struct VoiceControlBlock {
    pub pitch_hz: ControlRamp,
    pub amp: ControlRamp,
}
