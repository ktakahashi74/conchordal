pub type IndividualId = u64;

#[derive(Clone, Debug, PartialEq)]
pub struct BodySnapshot {
    pub kind: String,
    pub amp_scale: f32,
    pub brightness: f32,
    pub noise_mix: f32,
}

#[derive(Debug, Clone)]
pub enum AudioCommand {
    EnsureVoice {
        id: IndividualId,
        body: BodySnapshot,
        pitch_hz: f32,
        amp: f32,
    },
    Impulse {
        id: IndividualId,
        energy: f32,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct VoiceTarget {
    pub id: IndividualId,
    pub pitch_hz: f32,
    pub amp: f32,
}
