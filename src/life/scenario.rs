use serde::{Deserialize, Serialize};

use crate::life::individual::{AudioAgent, PureToneAgent};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    pub episodes: Vec<Episode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvelopeConfig {
    pub attack_sec: f32,
    pub decay_sec: f32,
    pub sustain_level: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub start_time: f32,
    pub events: Vec<Event>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub time: f32,
    pub actions: Vec<Action>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AgentConfig {
    #[serde(rename = "pure_tone")]
    PureTone {
        id: u64,
        freq: f32,
        amp: f32,
        phase: Option<f32>,
        envelope: Option<EnvelopeConfig>,
    }, // future variants
}

impl AgentConfig {
    pub fn id(&self) -> u64 {
        match self {
            AgentConfig::PureTone { id, .. } => *id,
        }
    }

    pub fn spawn(&self) -> Box<dyn AudioAgent> {
        match self {
            AgentConfig::PureTone {
                id,
                freq,
                amp,
                phase,
                envelope,
            } => {
                let mut agent = PureToneAgent::new(*id, *freq, *amp, 0, envelope.clone());
                if let Some(p) = phase {
                    agent.set_phase(*p);
                }
                Box::new(agent)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::life::individual::PureToneAgent;

    #[test]
    fn spawn_carries_envelope_and_params() {
        let env = EnvelopeConfig {
            attack_sec: 0.05,
            decay_sec: 0.2,
            sustain_level: 0.4,
        };
        let cfg = AgentConfig::PureTone {
            id: 7,
            freq: 220.0,
            amp: 0.3,
            phase: Some(0.25),
            envelope: Some(env.clone()),
        };

        let agent = cfg.spawn();
        let pt = agent.as_any().downcast_ref::<PureToneAgent>().unwrap();
        assert_eq!(pt.id, 7);
        assert_eq!(pt.freq_hz, 220.0);
        assert_eq!(pt.amp, 0.3);
        assert_eq!(pt.start_frame, 0);
        assert!((pt.envelope.attack_sec - env.attack_sec).abs() < 1e-6);
        assert!((pt.envelope.decay_sec - env.decay_sec).abs() < 1e-6);
        assert!((pt.envelope.sustain_level - env.sustain_level).abs() < 1e-6);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Action {
    AddAgent {
        agent: AgentConfig,
    },
    SpawnAgents {
        method: SpawnMethod,
        count: usize,
        base_id: u64,
        amp: f32,
        envelope: Option<EnvelopeConfig>,
    },
    RemoveAgent {
        id: u64,
    },
    SetFreq {
        id: u64,
        freq_hz: f32,
    },
    SetAmp {
        id: u64,
        amp: f32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpawnMethod {
    HighConsonance,
    SpectralGap,
    RandomUniform,
}
