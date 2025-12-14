use super::individual::{AgentMetadata, AudioAgent, IndividualWrapper, SoundBody};
use super::scenario::{Action, IndividualConfig, SpawnMethod};
use crate::core::landscape::LandscapeFrame;
use rand::{Rng, distr::Distribution, distr::weighted::WeightedIndex};
use std::collections::HashMap;
use tracing::{info, warn};

pub struct PopulationParams {
    pub initial_tones_hz: Vec<f32>,
    pub amplitude: f32,
}

#[derive(Default)]
struct WorkBuffers {
    audio: Vec<f32>,
    amps: Vec<f32>,
}

pub struct Population {
    pub individuals: Vec<IndividualWrapper>,
    current_frame: u64,
    pub abort_requested: bool,
    next_auto_id: u64,
    tag_counters: HashMap<String, usize>,
    buffers: WorkBuffers,
    pub global_vitality: f32,
}

impl Population {
    pub fn new(p: PopulationParams) -> Self {
        let individuals = p
            .initial_tones_hz
            .into_iter()
            .enumerate()
            .map(|(idx, f)| {
                let cfg = IndividualConfig::PureTone {
                    freq: f,
                    amp: p.amplitude,
                    phase: None,
                    lifecycle: crate::life::lifecycle::LifecycleConfig::Decay {
                        initial_energy: 1.0,
                        half_life_sec: 0.5,
                        attack_sec: crate::life::lifecycle::default_decay_attack(),
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
            individuals,
            current_frame: 0,
            abort_requested: false,
            next_auto_id: 1_000_000,
            tag_counters: HashMap::new(),
            buffers: WorkBuffers::default(),
            global_vitality: 0.1,
        }
    }

    fn find_individual_mut(&mut self, id: u64) -> Option<&mut IndividualWrapper> {
        self.individuals.iter_mut().find(|a| a.id() == id)
    }

    pub fn add_individual(&mut self, individual: IndividualWrapper) {
        let id = individual.id();
        self.individuals.retain(|a| a.id() != id);
        self.individuals.push(individual);
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

        if rest.starts_with('[')
            && let Some(end) = rest.find(']')
        {
            let grp_str = &rest[1..end];
            if let Ok(g) = grp_str.parse::<usize>() {
                group_idx = Some(g);
            }
            rest = &rest[(end + 1)..];
            if rest.starts_with('[')
                && let Some(end2) = rest.find(']')
            {
                let mem_str = &rest[1..end2];
                if let Ok(m) = mem_str.parse::<usize>() {
                    member_idx = Some(m);
                }
            }
        }

        self.individuals
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
                if let Some(g) = group_idx
                    && meta.group_idx != g
                {
                    return None;
                }
                if let Some(m) = member_idx
                    && meta.member_idx != m
                {
                    return None;
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
                // Penalize already loud bins to avoid crowding.
                let max_amp = landscape
                    .amps_last
                    .get(idx_min..=idx_max)
                    .map(|slice| slice.iter().cloned().fold(0.0f32, f32::max))
                    .unwrap_or(0.0)
                    .max(1e-6);
                let penalty_weight = 0.5;
                for i in idx_min..=idx_max {
                    if let (Some(&c_val), Some(&amp)) =
                        (landscape.c_last.get(i), landscape.amps_last.get(i))
                    {
                        let crowded = amp / max_amp;
                        let score = c_val - penalty_weight * crowded;
                        if score > best_val {
                            best_val = score;
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
                    if let Some(&v) = landscape.c_last.get(i)
                        && v < best_val
                    {
                        best_val = v;
                        best = i;
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
                let weights: Vec<f32> = (idx_min..=idx_max)
                    .map(|i| {
                        let amp = landscape.amps_last.get(i).copied().unwrap_or(0.0).max(1e-6);
                        1.0 / amp
                    })
                    .collect();
                if let Ok(dist) = WeightedIndex::new(&weights) {
                    idx_min + dist.sample(rng)
                } else {
                    // Fallback to the quietest bin.
                    let mut best = idx_min;
                    let mut best_val = f32::MAX;
                    for i in idx_min..=idx_max {
                        if let Some(&v) = landscape.amps_last.get(i)
                            && v < best_val
                        {
                            best_val = v;
                            best = i;
                        }
                    }
                    best
                }
            }
            SpawnMethod::HarmonicDensity { temperature, .. } => {
                let mut weights: Vec<f32> = (idx_min..=idx_max)
                    .map(|i| landscape.c_last.get(i).copied().unwrap_or(0.0).max(0.0))
                    .collect();
                if let Some(temp) = temperature
                    && *temp > 0.0
                {
                    for w in &mut weights {
                        *w = w.powf(1.0 / temp);
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
                self.add_individual(spawned);
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
                let mut rng = rand::rng();
                let group_idx = self.next_group_idx(tag.as_deref());
                for i in 0..count {
                    let freq = self.decide_frequency(&method, landscape, &mut rng);
                    let phase = rng.random_range(0.0..std::f32::consts::TAU);
                    let id = {
                        let id = self.next_auto_id;
                        self.next_auto_id += 1;
                        id
                    };
                    let cfg = IndividualConfig::PureTone {
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
                    self.add_individual(spawned);
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
                    if let Some(a) = self.find_individual_mut(id) {
                        match a {
                            IndividualWrapper::PureTone(ind) => {
                                ind.body.set_freq(freq_hz);
                            }
                            IndividualWrapper::Harmonic(ind) => {
                                ind.body.set_freq(freq_hz);
                            }
                        }
                    } else {
                        warn!("SetFreq: agent {id} not found");
                    }
                }
            }
            Action::SetAmp { target, amp } => {
                let ids = self.resolve_targets(&target);
                for id in ids {
                    if let Some(a) = self.find_individual_mut(id) {
                        match a {
                            IndividualWrapper::PureTone(ind) => {
                                ind.body.set_amp(amp);
                            }
                            IndividualWrapper::Harmonic(ind) => {
                                ind.body.set_amp(amp);
                            }
                        }
                    } else {
                        warn!("SetAmp: agent {id} not found");
                    }
                }
            }
            Action::SetRhythmVitality { value } => {
                self.global_vitality = value;
            }
        }
    }

    pub fn remove_agent(&mut self, id: u64) {
        self.individuals.retain(|a| a.id() != id);
    }

    /// Mix audio samples for the next hop.
    pub fn process_audio(
        &mut self,
        samples_len: usize,
        fs: f32,
        current_frame: u64,
        dt_sec: f32,
        landscape: &crate::core::landscape::Landscape,
    ) -> &[f32] {
        self.current_frame = current_frame;
        self.buffers.audio.resize(samples_len, 0.0);
        self.buffers.audio.fill(0.0);
        for agent in self.individuals.iter_mut() {
            if agent.is_alive() {
                agent.render_wave(
                    &mut self.buffers.audio,
                    fs,
                    current_frame,
                    dt_sec,
                    landscape,
                );
            }
        }
        &self.buffers.audio
    }

    /// Render spectral bodies for landscape processing.
    pub fn process_frame(
        &mut self,
        current_frame: u64,
        n_bins: usize,
        fs: f32,
        nfft: usize,
        dt_sec: f32,
        scenario_finished: bool,
    ) -> &[f32] {
        self.current_frame = current_frame;
        self.buffers.amps.resize(n_bins, 0.0);
        self.buffers.amps.fill(0.0);
        for agent in self.individuals.iter_mut() {
            if agent.is_alive() {
                agent.render_spectrum(&mut self.buffers.amps, fs, nfft, current_frame, dt_sec);
            }
        }
        let before_count = self.individuals.len();
        self.individuals.retain(|agent| agent.is_alive());
        let removed_count = before_count - self.individuals.len();

        if removed_count > 0 {
            let t = current_frame as f32 * dt_sec;
            let prefix = if scenario_finished || self.abort_requested {
                "Event after scenario close: "
            } else {
                ""
            };
            if scenario_finished || self.abort_requested {
                warn!(
                    "{prefix}[t={:.6}] Cleaned up {} dead individuals. Remaining: {} (frame_idx={})",
                    t,
                    removed_count,
                    self.individuals.len(),
                    current_frame
                );
            } else {
                info!(
                    "{prefix}[t={:.6}] Cleaned up {} dead individuals. Remaining: {} (frame_idx={})",
                    t,
                    removed_count,
                    self.individuals.len(),
                    current_frame
                );
            }
        }
        &self.buffers.amps
    }
}
