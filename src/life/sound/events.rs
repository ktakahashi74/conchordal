use std::sync::Arc;

pub type VoiceId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(test, derive(serde::Serialize, serde::Deserialize))]
pub enum BodyKind {
    Sine,
    Harmonic,
    Modal,
}

impl BodyKind {
    pub(crate) fn supports_body_models(self, renderer_phase3: bool) -> bool {
        // Native Modal has no output ADSR and keeps free radiation after Off.
        !renderer_phase3 || matches!(self, Self::Sine | Self::Harmonic)
    }
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
    pub ratios: Option<Arc<[f32]>>,
}
