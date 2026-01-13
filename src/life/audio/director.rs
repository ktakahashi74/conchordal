use crate::life::audio::events::{AudioCommand, AudioEvent, LifeEvent};

#[derive(Debug, Clone)]
pub struct StimulusDirector {
    pub birth_energy: f32,
}

impl StimulusDirector {
    pub fn new(birth_energy: f32) -> Self {
        Self { birth_energy }
    }

    pub fn emit(&mut self, ev: LifeEvent, out: &mut Vec<AudioCommand>) {
        match ev {
            LifeEvent::Spawned { id, body } => {
                out.push(AudioCommand::Trigger {
                    id,
                    ev: AudioEvent::Impulse {
                        energy: self.birth_energy,
                    },
                    body: Some(body),
                });
            }
            LifeEvent::BodyChanged { id, body } => {
                out.push(AudioCommand::SetBody { id, body });
            }
        }
    }
}
