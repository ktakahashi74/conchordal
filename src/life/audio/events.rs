use crate::life::intent::BodySnapshot;

pub type IndividualId = u64;

#[derive(Debug, Clone, Copy)]
pub enum AudioEvent {
    Impulse { energy: f32 },
}

#[derive(Debug, Clone)]
pub enum LifeEvent {
    Spawned {
        id: IndividualId,
        body: BodySnapshot,
    },
    BodyChanged {
        id: IndividualId,
        body: BodySnapshot,
    },
}

#[derive(Debug, Clone)]
pub enum AudioCommand {
    Trigger {
        id: IndividualId,
        ev: AudioEvent,
        body: Option<BodySnapshot>,
    },
    SetBody {
        id: IndividualId,
        body: BodySnapshot,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct VoiceTarget {
    pub id: IndividualId,
    pub pitch_hz: f32,
    pub amp: f32,
}
