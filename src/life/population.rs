use super::individual::{AudioAgent, PureToneAgent};
use super::scenario::{Action, AgentConfig};
use tracing::warn;

pub struct PopulationParams {
    pub initial_tones_hz: Vec<f32>,
    pub amplitude: f32,
}

pub struct Population {
    pub agents: Vec<Box<dyn AudioAgent>>,
}

impl Population {
    pub fn new(p: PopulationParams) -> Self {
        let agents = p
            .initial_tones_hz
            .into_iter()
            .enumerate()
            .map(|(idx, f)| {
                let cfg = AgentConfig::PureTone {
                    id: idx as u64,
                    freq: f,
                    amp: p.amplitude,
                    phase: None,
                    envelope: None,
                };
                cfg.spawn()
            })
            .collect();
        Self { agents }
    }

    fn find_agent_mut(&mut self, id: u64) -> Option<&mut Box<dyn AudioAgent>> {
        self.agents.iter_mut().find(|a| a.id() == id)
    }

    pub fn add_agent(&mut self, agent: Box<dyn AudioAgent>) {
        let id = agent.id();
        self.agents.retain(|a| a.id() != id);
        self.agents.push(agent);
    }

    pub fn apply_action(&mut self, action: Action) {
        match action {
            Action::AddAgent { agent } => {
                let spawned = agent.spawn();
                self.add_agent(spawned);
            }
            Action::SpawnAgents { .. } => {
                todo!("SpawnAgents logic not implemented yet");
            }
            Action::RemoveAgent { id } => self.remove_agent(id),
            Action::SetFreq { id, freq_hz } => {
                if let Some(a) = self.find_agent_mut(id) {
                    if let Some(pt) = a.as_any_mut().downcast_mut::<PureToneAgent>() {
                        pt.set_freq(freq_hz);
                    } else {
                        warn!("SetFreq: agent {id} is not a PureToneAgent");
                    }
                } else {
                    warn!("SetFreq: agent {id} not found");
                }
            }
            Action::SetAmp { id, amp } => {
                if let Some(a) = self.find_agent_mut(id) {
                    if let Some(pt) = a.as_any_mut().downcast_mut::<PureToneAgent>() {
                        pt.set_amp(amp);
                    } else {
                        warn!("SetAmp: agent {id} is not a PureToneAgent");
                    }
                } else {
                    warn!("SetAmp: agent {id} not found");
                }
            }
        }
    }

    pub fn remove_agent(&mut self, id: u64) {
        self.agents.retain(|a| a.id() != id);
    }

    /// Mix audio samples for the next hop.
    pub fn process_audio(&mut self, samples_len: usize, fs: f32) -> Vec<f32> {
        let mut buf = vec![0.0f32; samples_len];
        for agent in self.agents.iter_mut() {
            if agent.is_alive() {
                agent.render_wave(&mut buf, fs);
            }
        }
        buf
    }

    /// Render spectral bodies for landscape processing.
    pub fn render_landscape_body(
        &self,
        current_frame: u64,
        n_bins: usize,
        fs: f32,
        n_fft: usize,
    ) -> Vec<f32> {
        let mut body = vec![0.0f32; n_bins];
        for agent in self.agents.iter() {
            if agent.is_alive() {
                agent.render_body(&mut body, n_fft, fs, current_frame);
            }
        }
        body
    }
}
