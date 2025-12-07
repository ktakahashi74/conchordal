use super::individual::{AgentMetadata, AudioAgent, PureToneAgent};
use super::scenario::{Action, AgentConfig, SpawnMethod};
use crate::core::landscape::LandscapeFrame;
use rand::{Rng, distr::Distribution, distr::weighted::WeightedIndex};
use std::collections::HashMap;
use tracing::{debug, warn};

pub struct PopulationParams {
    pub initial_tones_hz: Vec<f32>,
    pub amplitude: f32,
}

pub struct Population {
    pub agents: Vec<Box<dyn AudioAgent>>,
    current_frame: u64,
    pub abort_requested: bool,
    next_auto_id: u64,
    tag_counters: HashMap<String, usize>,
}

impl Population {
    pub fn new(p: PopulationParams) -> Self {
        let agents = p
            .initial_tones_hz
            .into_iter()
            .enumerate()
            .map(|(idx, f)| {
                let cfg = AgentConfig::PureTone {
                    freq: f,
                    amp: p.amplitude,
                    phase: None,
                    lifecycle: crate::life::lifecycle::LifecycleConfig::Decay {
                        initial_energy: 1.0,
                        half_life_sec: 0.5,
                    },
                    tag: None,
                };
                let metadata = AgentMetadata {
                    id: idx as u64,
                    tag: None,
                    group_idx: 0,
                    member_idx: idx,
                };
                cfg.spawn(idx as u64, 0, metadata)
            })
            .collect();
        Self {
            agents,
            current_frame: 0,
            abort_requested: false,
            next_auto_id: 1_000_000,
            tag_counters: HashMap::new(),
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

    fn next_group_idx(&mut self, tag: Option<&str>) -> usize {
        if let Some(t) = tag {
            let entry = self.tag_counters.entry(t.to_string()).or_insert(0);
            let idx = *entry;
            *entry += 1;
            idx
        } else {
            0
        }
    }

    fn resolve_targets(&mut self, target_str: &str) -> Vec<u64> {
        let mut tag_end = target_str.find('[').unwrap_or(target_str.len());
        if tag_end == 0 {
            return Vec::new();
        }
        if tag_end > target_str.len() {
            tag_end = target_str.len();
        }
        let tag = &target_str[..tag_end];
        let mut rest = &target_str[tag_end..];
        let mut group_idx: Option<usize> = None;
        let mut member_idx: Option<usize> = None;

        if rest.starts_with('[') {
            if let Some(end) = rest.find(']') {
                let grp_str = &rest[1..end];
                if let Ok(g) = grp_str.parse::<usize>() {
                    group_idx = Some(g);
                }
                rest = &rest[(end + 1)..];
                if rest.starts_with('[') {
                    if let Some(end2) = rest.find(']') {
                        let mem_str = &rest[1..end2];
                        if let Ok(m) = mem_str.parse::<usize>() {
                            member_idx = Some(m);
                        }
                    }
                }
            }
        }

        self.agents
            .iter()
            .filter_map(|a| {
                let meta = a.metadata();
                if let Some(t) = &meta.tag {
                    if t != tag {
                        return None;
                    }
                } else {
                    return None;
                }
                if let Some(g) = group_idx {
                    if meta.group_idx != g {
                        return None;
                    }
                }
                if let Some(m) = member_idx {
                    if meta.member_idx != m {
                        return None;
                    }
                }
                Some(meta.id)
            })
            .collect()
    }

    fn decide_frequency<R: Rng + ?Sized>(
        &self,
        method: &SpawnMethod,
        landscape: &LandscapeFrame,
        rng: &mut R,
    ) -> f32 {
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

        let jitter_bin = |idx: usize, rng: &mut R| -> f32 {
            let idx = idx.min(n_bins - 1);
            let center = space.freq_of_index(idx);
            let step = space.step();
            let half = step * 0.5;
            let center_log2 = center.log2();
            let sample_log2 = rng.random_range((center_log2 - half)..(center_log2 + half));
            2.0f32.powf(sample_log2).clamp(space.fmin, space.fmax)
        };

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
                    idx_min + dist.sample(rng)
                } else {
                    // fallback to random log-uniform
                    let min_l = min_freq.log2();
                    let max_l = max_freq.log2();
                    let r = rng.random_range(min_l..max_l);
                    let f = 2.0f32.powf(r);
                    return f;
                }
            }
            SpawnMethod::RandomLogUniform { .. } => {
                let min_l = min_freq.log2();
                let max_l = max_freq.log2();
                let r = rng.random_range(min_l..max_l);
                let f = 2.0f32.powf(r);
                return f;
            }
        };

        jitter_bin(pick_idx, rng)
    }

    pub fn apply_action(&mut self, action: Action, landscape: &LandscapeFrame) {
        match action {
            Action::AddAgent { agent } => {
                let id = {
                    let id = self.next_auto_id;
                    self.next_auto_id += 1;
                    id
                };
                let tag = agent.tag().cloned();
                let group_idx = self.next_group_idx(tag.as_deref());
                let metadata = AgentMetadata {
                    id,
                    tag,
                    group_idx,
                    member_idx: 0,
                };
                let spawned = agent.spawn(id, self.current_frame, metadata);
                self.add_agent(spawned);
            }
            Action::Finish => {
                self.abort_requested = true;
            }
            Action::SpawnAgents {
                method,
                count,
                amp,
                lifecycle,
                tag,
            } => {
                let mut rng = rand::thread_rng();
                let group_idx = self.next_group_idx(tag.as_deref());
                for i in 0..count {
                    let freq = self.decide_frequency(&method, landscape, &mut rng);
                    let phase = rng.random_range(0.0..std::f32::consts::TAU);
                    let id = {
                        let id = self.next_auto_id;
                        self.next_auto_id += 1;
                        id
                    };
                    let cfg = AgentConfig::PureTone {
                        freq,
                        amp,
                        phase: Some(phase),
                        lifecycle: lifecycle.clone(),
                        tag: tag.clone(),
                    };
                    let metadata = AgentMetadata {
                        id,
                        tag: tag.clone(),
                        group_idx,
                        member_idx: i,
                    };
                    let spawned = cfg.spawn(id, self.current_frame, metadata);
                    self.add_agent(spawned);
                }
            }
            Action::RemoveAgent { target } => {
                let ids = self.resolve_targets(&target);
                for id in ids {
                    self.remove_agent(id);
                }
            }
            Action::SetFreq { target, freq_hz } => {
                let ids = self.resolve_targets(&target);
                for id in ids {
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
            }
            Action::SetAmp { target, amp } => {
                let ids = self.resolve_targets(&target);
                for id in ids {
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
        let before_count = self.agents.len();
        self.agents.retain(|agent| agent.is_alive());
        let removed_count = before_count - self.agents.len();

        if removed_count > 0 {
            debug!(
                "Cleaned up {} dead agents. Remaining: {}",
                removed_count,
                self.agents.len()
            );
        }
        amps
    }
}
