use super::events::AudioEvent;

pub trait Exciter {
    fn trigger(&mut self, ev: AudioEvent);
    fn next_drive(&mut self) -> f32;
}

#[derive(Debug, Clone)]
pub struct ImpulseExciter {
    pending: f32,
}

impl ImpulseExciter {
    pub fn new() -> Self {
        Self { pending: 0.0 }
    }
}

impl Default for ImpulseExciter {
    fn default() -> Self {
        Self::new()
    }
}

impl Exciter for ImpulseExciter {
    fn trigger(&mut self, ev: AudioEvent) {
        match ev {
            AudioEvent::Impulse { energy } => {
                if energy.is_finite() {
                    self.pending += energy;
                }
            }
        }
    }

    fn next_drive(&mut self) -> f32 {
        let out = self.pending;
        self.pending = 0.0;
        out
    }
}
