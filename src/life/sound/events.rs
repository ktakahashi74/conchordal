use std::sync::Arc;

pub type IndividualId = u64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyKind {
    Sine,
    Harmonic,
    Modal,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BodySnapshot {
    pub kind: BodyKind,
    pub amp_scale: f32,
    pub brightness: f32,
    pub inharmonic: f32,
    pub spread: f32,
    pub voices: usize,
    pub motion: f32,
    pub ratios: Option<Arc<[f32]>>,
}
