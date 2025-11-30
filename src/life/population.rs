use super::individual::{AudioAgent, PureToneAgent};
use super::scenario::{Action, AgentConfig, SpawnMethod};
use crate::core::landscape::LandscapeFrame;
use rand::{Rng, distr::Distribution, distr::weighted::WeightedIndex};
use tracing::warn;

pub struct PopulationParams {
    pub initial_tones_hz: Vec<f32>,
    pub amplitude: f32,
}

pub struct Population {
    pub agents: Vec<Box<dyn AudioAgent>>,
    current_frame: u64,
    pub abort_requested: bool,
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
                cfg.spawn(0)
            })
            .collect();
        Self {
            agents,
            current_frame: 0,
            abort_requested: false,
        }
    }

    fn find_agent_mut(&mut self, id: u64) -> Option<&mut Box<dyn AudioAgent>> {
        self.agents.iter_mut().find(|a| a.id() == id)
    }

    pub fn add_agent(&mut self, agent: Box<dyn AudioAgent>) {
        let id = agent.id();
        self.agents.retain(|a| a.id() != id);
        self.agents.push(agent);
    }

    pub fn set_current_frame(&mut self, frame: u64) {
        self.current_frame = frame;
    }

    fn decide_frequency(&self, method: &SpawnMethod, landscape: &LandscapeFrame) -> f32 {
        use rand::Rng;

        let space = &landscape.space;
        let n_bins = space.n_bins();
        if n_bins == 0 {
            return 440.0;
        }

        let (min_freq, max_freq) = match method {
            SpawnMethod::Harmonicity { min_freq, max_freq }
            | SpawnMethod::LowHarmonicity { min_freq, max_freq }
            | SpawnMethod::HarmonicDensity {
                min_freq, max_freq, ..
            }
            | SpawnMethod::ZeroCrossing { min_freq, max_freq }
            | SpawnMethod::SpectralGap { min_freq, max_freq }
            | SpawnMethod::RandomLogUniform { min_freq, max_freq } => (*min_freq, *max_freq),
        };

        let mut idx_min = space.index_of_freq(min_freq).unwrap_or(0);
        let mut idx_max = space
            .index_of_freq(max_freq)
            .unwrap_or_else(|| n_bins.saturating_sub(1));
        if idx_min > idx_max {
            std::mem::swap(&mut idx_min, &mut idx_max);
        }
        idx_max = idx_max.min(n_bins.saturating_sub(1));
        if idx_min >= n_bins || idx_min > idx_max {
            return space.freq_of_index(n_bins / 2);
        }

        let pick_idx = match method {
            SpawnMethod::Harmonicity { .. } => {
                let mut best = idx_min;
                let mut best_val = f32::MIN;
                for i in idx_min..=idx_max {
                    if let Some(&v) = landscape.c_last.get(i) {
                        if v > best_val {
                            best_val = v;
                            best = i;
                        }
                    }
                }
                best
            }
            SpawnMethod::LowHarmonicity { .. } => {
                let mut best = idx_min;
                let mut best_val = f32::MAX;
                for i in idx_min..=idx_max {
                    if let Some(&v) = landscape.c_last.get(i) {
                        if v < best_val {
                            best_val = v;
                            best = i;
                        }
                    }
                }
                best
            }
            SpawnMethod::ZeroCrossing { .. } => {
                let mut best = idx_min;
                let mut best_val = f32::MAX;
                for i in idx_min..=idx_max {
                    if let Some(&v) = landscape.c_last.get(i) {
                        let d = v.abs();
                        if d < best_val {
                            best_val = d;
                            best = i;
                        }
                    }
                }
                best
            }
            SpawnMethod::SpectralGap { .. } => {
                let mut best = idx_min;
                let mut best_val = f32::MAX;
                for i in idx_min..=idx_max {
                    if let Some(&v) = landscape.amps_last.get(i) {
                        if v < best_val {
                            best_val = v;
                            best = i;
                        }
                    }
                }
                best
            }
            SpawnMethod::HarmonicDensity { temperature, .. } => {
                let mut weights: Vec<f32> = (idx_min..=idx_max)
                    .map(|i| landscape.c_last.get(i).copied().unwrap_or(0.0).max(0.0))
                    .collect();
                if let Some(temp) = temperature {
                    if *temp > 0.0 {
                        for w in &mut weights {
                            *w = w.powf(1.0 / temp);
                        }
                    }
                }
                if let Ok(dist) = WeightedIndex::new(&weights) {
                    let mut rng = rand::thread_rng();
                    idx_min + dist.sample(&mut rng)
                } else {
                    // fallback to random log-uniform
                    let mut rng = rand::thread_rng();
                    let min_l = min_freq.log2();
                    let max_l = max_freq.log2();
                    let r = rng.gen_range(min_l..max_l);
                    let f = 2.0f32.powf(r);
                    return f;
                }
            }
            SpawnMethod::RandomLogUniform { .. } => {
                let mut rng = rand::thread_rng();
                let min_l = min_freq.log2();
                let max_l = max_freq.log2();
                let r = rng.gen_range(min_l..max_l);
                let f = 2.0f32.powf(r);
                return f;
            }
        };

        space.freq_of_index(pick_idx.min(n_bins - 1))
    }

    pub fn apply_action(&mut self, action: Action, landscape: &LandscapeFrame) {
        match action {
            Action::AddAgent { agent } => {
                let spawned = agent.spawn(self.current_frame);
                self.add_agent(spawned);
            }
            Action::Finish => {
                self.abort_requested = true;
            }
            Action::SpawnAgents {
                method,
                count,
                base_id,
                amp,
                lifecycle,
            } => {
                for i in 0..count {
                    let freq = self.decide_frequency(&method, landscape);
                    let cfg = AgentConfig::PureTone {
                        id: base_id + i as u64,
                        freq,
                        amp,
                        phase: None,
                        lifecycle: lifecycle.clone(),
                    };
                    let spawned = cfg.spawn(self.current_frame);
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
        self.current_frame = current_frame;
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
        self.current_frame = current_frame;
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
