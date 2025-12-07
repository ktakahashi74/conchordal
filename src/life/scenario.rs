use serde::{Deserialize, Serialize};

use crate::life::individual::AgentMetadata;
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
    #[serde(default)]
    pub name: Option<String>,
    pub start_time: f32,
    pub events: Vec<Event>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepeatConfig {
    pub count: usize,           // 繰り返し回数
    pub interval: f32,          // 実行間隔（秒）
    pub id_offset: Option<u64>, // 回ごとに base_id をずらす量（SpawnAgents用）
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    pub time: f32,
    pub repeat: Option<RepeatConfig>,
    pub actions: Vec<Action>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AgentConfig {
    #[serde(rename = "pure_tone")]
    PureTone {
        freq: f32,
        amp: f32,
        phase: Option<f32>,
        lifecycle: LifecycleConfig,
        tag: Option<String>,
    }, // future variants
}

impl AgentConfig {
    pub fn id(&self) -> Option<u64> {
        match self {
            AgentConfig::PureTone { .. } => None,
        }
    }

    pub fn tag(&self) -> Option<&String> {
        match self {
            AgentConfig::PureTone { tag, .. } => tag.as_ref(),
        }
    }

    pub fn spawn(
        &self,
        assigned_id: u64,
        start_frame: u64,
        mut metadata: AgentMetadata,
    ) -> Box<dyn AudioAgent> {
        metadata.id = assigned_id;
        if metadata.tag.is_none() {
            metadata.tag = self.tag().cloned();
        }
        match self {
            AgentConfig::PureTone {
                freq,
                amp,
                phase,
                lifecycle,
                ..
            } => {
                let lifecycle = lifecycle.clone().create_lifecycle();
                let mut agent = PureToneAgent::new(
                    assigned_id,
                    *freq,
                    *amp,
                    start_frame,
                    lifecycle,
                    metadata,
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
            freq: 220.0,
            amp: 0.3,
            phase: Some(0.25),
            lifecycle: LifecycleConfig::Decay {
                initial_energy: 1.0,
                half_life_sec: 0.5,
                attack_sec: crate::life::lifecycle::default_decay_attack(),
            },
            tag: Some("test".into()),
        };

        let agent = cfg.spawn(
            7,
            5,
            AgentMetadata {
                id: 7,
                tag: Some("test".into()),
                group_idx: 0,
                member_idx: 0,
            },
        );
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
        amp: f32,
        lifecycle: LifecycleConfig,
        tag: Option<String>,
    },
    RemoveAgent {
        target: String,
    },
    SetFreq {
        target: String,
        freq_hz: f32,
    },
    SetAmp {
        target: String,
        amp: f32,
    },
    Finish,
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
