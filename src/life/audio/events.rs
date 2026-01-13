pub type IndividualId = u64;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BodyKind {
    Sine,
    Harmonic,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodySpec {
    pub kind: BodyKind,
    pub amp_scale: f32,
    pub brightness: f32,
    pub noise_mix: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum AudioEvent {
    Impulse { energy: f32 },
}

#[derive(Debug, Clone)]
pub enum LifeEvent {
    Spawned { id: IndividualId },
}

#[derive(Debug, Clone)]
pub enum AudioCommand {
    Trigger { id: IndividualId, ev: AudioEvent },
}

#[derive(Debug, Clone, Copy)]
pub struct VoiceTarget {
    pub id: IndividualId,
    pub pitch_hz: f32,
    pub amp: f32,
    pub body: BodySpec,
}
