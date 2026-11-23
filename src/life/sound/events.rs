use std::sync::Arc;

#[cfg(test)]
fn is_zero(value: &f32) -> bool {
    *value == 0.
}

pub type VoiceId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(test, derive(serde::Serialize, serde::Deserialize))]
pub enum BodyKind {
    Sine,
    Harmonic,
    Modal,
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(test, derive(serde::Serialize, serde::Deserialize))]
pub struct BodySnapshot {
    pub kind: BodyKind,
    pub amp_scale: f32,
    pub brightness: f32,
    pub inharmonic: f32,
    pub spread: f32,
    pub unison: usize,
    pub motion: f32,
    #[cfg_attr(test, serde(default, skip_serializing_if = "is_zero"))]
    pub attack_noise: f32,
    pub ratios: Option<Arc<[f32]>>,
}

impl BodySnapshot {
    pub(crate) fn needs_onset_motion_clock(&self) -> bool {
        self.kind == BodyKind::Harmonic && self.motion > 0.
    }
}
