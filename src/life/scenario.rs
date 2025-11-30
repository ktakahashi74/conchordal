use serde::{Deserialize, Serialize};

use crate::life::individual::{AudioAgent, PureToneAgent};
use crate::life::lifecycle::LifecycleConfig;

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
        lifecycle: LifecycleConfig,
    }, // future variants
}

impl AgentConfig {
    pub fn id(&self) -> u64 {
        match self {
            AgentConfig::PureTone { id, .. } => *id,
        }
    }

    pub fn spawn(&self, start_frame: u64) -> Box<dyn AudioAgent> {
        match self {
            AgentConfig::PureTone {
                id,
                freq,
                amp,
                phase,
                lifecycle,
            } => {
                let mut agent = PureToneAgent::new(
                    *id,
                    *freq,
                    *amp,
                    start_frame,
                    lifecycle.clone().create_lifecycle(),
                );
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
        let cfg = AgentConfig::PureTone {
            id: 7,
            freq: 220.0,
            amp: 0.3,
            phase: Some(0.25),
            lifecycle: LifecycleConfig::Decay {
                initial_energy: 1.0,
                half_life_sec: 0.5,
            },
        };

        let agent = cfg.spawn(5);
        let pt = agent.as_any().downcast_ref::<PureToneAgent>().unwrap();
        assert_eq!(pt.id, 7);
        assert_eq!(pt.freq_hz, 220.0);
        assert_eq!(pt.amp, 0.3);
        assert_eq!(pt.start_frame, 5);
        // LifecycleConfig::Sustain stores envelope; not directly inspectable here but spawn should succeed.
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
        lifecycle: LifecycleConfig,
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
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum SpawnMethod {
    /// H = C - R を最大化する周波数を探索（決定論的）
    Harmonicity { min_freq: f32, max_freq: f32 },
    /// H = C - R を最小化する周波数を探索（決定論的）
    LowHarmonicity { min_freq: f32, max_freq: f32 },
    /// H の値を確率密度としてサンプリング（確率的・群生）
    HarmonicDensity {
        min_freq: f32,
        max_freq: f32,
        temperature: Option<f32>,
    },
    /// H ≈ 0 となる領域を探索
    ZeroCrossing { min_freq: f32, max_freq: f32 },
    /// エネルギーの空白域（スペクトルの谷）を探索
    SpectralGap { min_freq: f32, max_freq: f32 },
    /// 単純なランダム（オクターブ等間隔分布）
    RandomLogUniform { min_freq: f32, max_freq: f32 },
}
