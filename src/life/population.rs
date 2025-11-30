use super::individual::{AudioAgent, PureToneAgent};
use super::scenario::{Action, AgentConfig, SpawnMethod};
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
                    lifecycle: crate::life::lifecycle::LifecycleConfig::Decay {
                        initial_energy: 1.0,
                        half_life_sec: 0.5,
                    },
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
            Action::SpawnAgents {
                method,
                count,
                base_id,
                amp,
                lifecycle,
            } => {
                use rand::Rng;
                let mut rng = rand::thread_rng();

                fn sample_log_uniform(
                    rng: &mut rand::rngs::ThreadRng,
                    min_freq: f32,
                    max_freq: f32,
                ) -> f32 {
                    let min_l = min_freq.log2();
                    let max_l = max_freq.log2();
                    let r = rng.gen_range(min_l..max_l);
                    2.0f32.powf(r)
                }

                for i in 0..count {
                    let freq = match &method {
                        SpawnMethod::RandomLogUniform { min_freq, max_freq } => {
                            sample_log_uniform(&mut rng, *min_freq, *max_freq)
                        }
                        SpawnMethod::Harmonicity { min_freq, max_freq } => {
                            // Placeholder: choose geometric mean of range.
                            (min_freq * max_freq).sqrt()
                        }
                        SpawnMethod::LowHarmonicity { min_freq, max_freq } => {
                            // Placeholder: alternate picking range edges.
                            if i % 2 == 0 { *min_freq } else { *max_freq }
                        }
                        SpawnMethod::HarmonicDensity {
                            min_freq, max_freq, ..
                        } => {
                            // Placeholder: log-uniform sampling until density-based spawn is available.
                            sample_log_uniform(&mut rng, *min_freq, *max_freq)
                        }
                        SpawnMethod::ZeroCrossing { min_freq, max_freq } => {
                            // Placeholder: choose midpoint in log-space.
                            (min_freq * max_freq).sqrt()
                        }
                        SpawnMethod::SpectralGap { min_freq, max_freq } => {
                            // Placeholder: log-uniform sampling until gap detection is implemented.
                            sample_log_uniform(&mut rng, *min_freq, *max_freq)
                        }
                    };

                    let cfg = AgentConfig::PureTone {
                        id: base_id + i as u64,
                        freq,
                        amp,
                        phase: None,
                        lifecycle: lifecycle.clone(),
                    };
                    let spawned = cfg.spawn();
                    self.add_agent(spawned);
                }
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
    pub fn process_audio(
        &mut self,
        samples_len: usize,
        fs: f32,
        current_frame: u64,
        dt_sec: f32,
    ) -> Vec<f32> {
        let mut buf = vec![0.0f32; samples_len];
        for agent in self.agents.iter_mut() {
            if agent.is_alive() {
                agent.render_wave(&mut buf, fs, current_frame, dt_sec);
            }
        }
        buf
    }

    /// Render spectral bodies for landscape processing.
    pub fn process_frame(
        &mut self,
        current_frame: u64,
        n_bins: usize,
        fs: f32,
        nfft: usize,
        dt_sec: f32,
    ) -> Vec<f32> {
        let mut amps = vec![0.0f32; n_bins];
        for agent in self.agents.iter_mut() {
            if agent.is_alive() {
                agent.render_spectrum(&mut amps, fs, nfft, current_frame, dt_sec);
            }
        }
        self.agents.retain(|a| a.is_alive());
        amps
    }
}
