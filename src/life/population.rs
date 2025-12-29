use super::individual::{AgentMetadata, AudioAgent, Individual, ModulationCore, SoundBody};
use super::scenario::{Action, IndividualConfig, SpawnMethod};
use crate::core::landscape::{LandscapeFrame, LandscapeUpdate};
use crate::core::log2space::Log2Space;
use rand::{Rng, SeedableRng, distr::Distribution, distr::weighted::WeightedIndex, rngs::SmallRng};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use tracing::{debug, info, warn};

#[derive(Default)]
struct WorkBuffers {
    audio: Vec<f32>,
    amps: Vec<f32>,
}

pub struct Population {
    pub individuals: Vec<Individual>,
    current_frame: u64,
    pub abort_requested: bool,
    next_auto_id: u64,
    tag_counters: HashMap<String, usize>,
    buffers: WorkBuffers,
    pub global_vitality: f32,
    pub global_coupling: f32,
    shutdown_gain: f32,
    pending_update: Option<LandscapeUpdate>,
    fs: f32,
}

impl Population {
    const RELEASE_SEC_DEFAULT: f32 = 0.03;
    /// Returns true if `freq_hz` is within `min_dist_erb` (ERB scale) of any existing agent's base
    /// frequency.
    pub fn is_range_occupied(&self, freq_hz: f32, min_dist_erb: f32) -> bool {
        if !freq_hz.is_finite() || min_dist_erb <= 0.0 {
            return false;
        }
        let target_erb = crate::core::erb::hz_to_erb(freq_hz.max(1e-6));
        for agent in &self.individuals {
            let base_hz = agent.body.base_freq_hz();
            if !base_hz.is_finite() {
                continue;
            }
            let d_erb = (crate::core::erb::hz_to_erb(base_hz.max(1e-6)) - target_erb).abs();
            if d_erb < min_dist_erb {
                return true;
            }
        }
        false
    }

    pub fn new(fs: f32) -> Self {
        debug!("Population sample rate: {:.1} Hz", fs);
        Self {
            individuals: Vec::new(),
            current_frame: 0,
            abort_requested: false,
            next_auto_id: 1_000_000,
            tag_counters: HashMap::new(),
            buffers: WorkBuffers::default(),
            global_vitality: 0.1,
            global_coupling: 1.0,
            shutdown_gain: 1.0,
            pending_update: None,
            fs,
        }
    }

    fn spawn_seed(&self, tag: Option<&str>, group_idx: usize, count: usize) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.current_frame.hash(&mut hasher);
        group_idx.hash(&mut hasher);
        count.hash(&mut hasher);
        if let Some(tag) = tag {
            tag.hash(&mut hasher);
        }
        hasher.finish() ^ 0x9E37_79B9_7F4A_7C15
    }

    fn find_individual_mut(&mut self, id: u64) -> Option<&mut Individual> {
        self.individuals.iter_mut().find(|a| a.id() == id)
    }

    pub fn add_individual(&mut self, individual: Individual) {
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
                let Some(t) = &meta.tag else { return None };
                if !wildcard_match(tag, t) {
                    return None;
                };
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

        let (min_freq, max_freq) = method.freq_range_hz();

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

        let min_dist_erb = method.min_dist_erb_or_default();

        let jitter_bin = |idx: usize, rng: &mut R| -> f32 {
            let idx = idx.min(n_bins - 1);
            let center = space.freq_of_index(idx);
            let step = space.step();
            let half = step * 0.5;
            let center_log2 = center.log2();
            let sample_log2 = rng.random_range((center_log2 - half)..(center_log2 + half));
            2.0f32.powf(sample_log2).clamp(space.fmin, space.fmax)
        };

        let jitter_free_bin = |idx: usize, rng: &mut R| -> f32 {
            let center = space.freq_of_index(idx.min(n_bins - 1));
            // Try a few times to jitter within the bin while avoiding occupied bands.
            for _ in 0..16 {
                let f = jitter_bin(idx, rng);
                if !self.is_range_occupied(f, min_dist_erb) {
                    return f;
                }
            }
            center
        };

        let pick_idx = match method {
            SpawnMethod::Harmonicity { .. } => {
                let mut best = idx_min;
                let mut best_val = f32::MIN;
                let mut found = false;
                for i in idx_min..=idx_max {
                    let f = space.freq_of_index(i);
                    if self.is_range_occupied(f, min_dist_erb) {
                        continue;
                    }
                    if let Some(&c_val) = landscape.consonance01.get(i)
                        && c_val > best_val
                    {
                        found = true;
                        best_val = c_val;
                        best = i;
                    }
                }
                if found {
                    best
                } else {
                    // Fallback: everything is occupied; pick the best bin ignoring occupancy.
                    let mut best = idx_min;
                    let mut best_val = f32::MIN;
                    for i in idx_min..=idx_max {
                        if let Some(&c_val) = landscape.consonance01.get(i)
                            && c_val > best_val
                        {
                            best_val = c_val;
                            best = i;
                        }
                    }
                    best
                }
            }
            SpawnMethod::LowHarmonicity { .. } => {
                let mut best = idx_min;
                let mut best_val = f32::MAX;
                let mut found = false;
                for i in idx_min..=idx_max {
                    let f = space.freq_of_index(i);
                    if self.is_range_occupied(f, min_dist_erb) {
                        continue;
                    }
                    if let Some(&v) = landscape.consonance01.get(i)
                        && v < best_val
                    {
                        found = true;
                        best_val = v;
                        best = i;
                    }
                }
                if found { best } else { idx_min }
            }
            SpawnMethod::ZeroCrossing { .. } => {
                let mut best = idx_min;
                let mut best_val = f32::MAX;
                let mut found = false;
                for i in idx_min..=idx_max {
                    let f = space.freq_of_index(i);
                    if self.is_range_occupied(f, min_dist_erb) {
                        continue;
                    }
                    if let Some(&v) = landscape.consonance01.get(i) {
                        let d = (v - 0.5).abs();
                        if d < best_val {
                            found = true;
                            best_val = d;
                            best = i;
                        }
                    }
                }
                if found { best } else { idx_min }
            }
            SpawnMethod::SpectralGap { .. } => {
                let weights: Vec<f32> = (idx_min..=idx_max)
                    .map(|i| {
                        let f = space.freq_of_index(i);
                        if self.is_range_occupied(f, min_dist_erb) {
                            return 0.0;
                        }
                        let amp = landscape
                            .subjective_intensity
                            .get(i)
                            .copied()
                            .unwrap_or(0.0)
                            .max(1e-6);
                        (1.0f32 / amp).max(0.0)
                    })
                    .collect();
                if let Ok(dist) = WeightedIndex::new(&weights) {
                    idx_min + dist.sample(rng)
                } else {
                    // Fallback to the quietest bin.
                    let mut best = idx_min;
                    let mut best_val = f32::MAX;
                    for i in idx_min..=idx_max {
                        let f = space.freq_of_index(i);
                        if self.is_range_occupied(f, min_dist_erb) {
                            continue;
                        }
                        if let Some(&v) = landscape.subjective_intensity.get(i)
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
                    .enumerate()
                    .map(|(local_idx, i)| {
                        let f = space.freq_of_index(i);
                        let occupied = self.is_range_occupied(f, min_dist_erb);
                        let _ = local_idx;
                        let c01 = landscape.consonance01.get(i).copied().unwrap_or(0.0);
                        harmonic_density_weight(c01, occupied)
                    })
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
                    for _ in 0..32 {
                        let r = rng.random_range(min_l..max_l);
                        let f = 2.0f32.powf(r);
                        if !self.is_range_occupied(f, min_dist_erb) {
                            return f;
                        }
                    }
                    return 2.0f32.powf(rng.random_range(min_l..max_l));
                }
            }
            SpawnMethod::RandomLogUniform { .. } => {
                let min_l = min_freq.log2();
                let max_l = max_freq.log2();
                for _ in 0..32 {
                    let r = rng.random_range(min_l..max_l);
                    let f = 2.0f32.powf(r);
                    if !self.is_range_occupied(f, min_dist_erb) {
                        return f;
                    }
                }
                return 2.0f32.powf(rng.random_range(min_l..max_l));
            }
        };

        jitter_free_bin(pick_idx, rng)
    }

    pub fn apply_action(
        &mut self,
        action: Action,
        landscape: &LandscapeFrame,
        landscape_rt: Option<&mut crate::core::stream::roughness::RoughnessStream>,
    ) {
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
                let spawned = agent.spawn(id, self.current_frame, metadata, self.fs);
                self.add_individual(spawned);
            }
            Action::Finish => {
                self.abort_requested = true;
            }
            Action::SpawnAgents {
                method,
                count,
                amp,
                life,
                tag,
            } => {
                let group_idx = self.next_group_idx(tag.as_deref());
                let seed = self.spawn_seed(tag.as_deref(), group_idx, count);
                let mut rng = SmallRng::seed_from_u64(seed);
                for i in 0..count {
                    let freq = self.decide_frequency(&method, landscape, &mut rng);
                    let id = {
                        let id = self.next_auto_id;
                        self.next_auto_id += 1;
                        id
                    };
                    let cfg = IndividualConfig {
                        freq,
                        amp,
                        life: life.clone(),
                        tag: tag.clone(),
                    };
                    let metadata = AgentMetadata {
                        id,
                        tag: tag.clone(),
                        group_idx,
                        member_idx: i,
                    };
                    let spawned = cfg.spawn(id, self.current_frame, metadata, self.fs);
                    self.add_individual(spawned);
                }
            }
            Action::RemoveAgent { target } => {
                let ids = self.resolve_targets(&target);
                for id in ids {
                    self.remove_agent(id);
                }
            }
            Action::ReleaseAgent {
                target,
                release_sec,
            } => {
                let ids = self.resolve_targets(&target);
                let sec = if release_sec > 0.0 {
                    release_sec
                } else {
                    Self::RELEASE_SEC_DEFAULT
                };
                for id in ids {
                    if let Some(agent) = self.find_individual_mut(id) {
                        agent.start_release(sec);
                    }
                }
            }
            Action::SetFreq { target, freq_hz } => {
                let ids = self.resolve_targets(&target);
                let log_freq = freq_hz.max(1.0).log2();
                for id in ids {
                    if let Some(a) = self.find_individual_mut(id) {
                        a.force_set_pitch_log2(log_freq);
                    } else {
                        warn!("SetFreq: agent {id} not found");
                    }
                }
            }
            Action::SetAmp { target, amp } => {
                let ids = self.resolve_targets(&target);
                for id in ids {
                    if let Some(a) = self.find_individual_mut(id) {
                        a.body.set_amp(amp);
                    } else {
                        warn!("SetAmp: agent {id} not found");
                    }
                }
            }
            Action::SetRhythmVitality { value } => {
                self.global_vitality = value;
            }
            Action::SetGlobalCoupling { value } => {
                self.global_coupling = value;
            }
            Action::SetRoughnessTolerance { value } => {
                let upd = LandscapeUpdate {
                    roughness_k: Some(value),
                    ..Default::default()
                };
                if let Some(roughness) = landscape_rt {
                    roughness.apply_update(upd);
                }
                let mut pending = self.pending_update.unwrap_or_default();
                pending.roughness_k = Some(value);
                self.pending_update = Some(pending);
            }
            Action::SetHarmonicity { mirror, limit } => {
                let mut pending = self.pending_update.unwrap_or_default();
                pending.mirror = mirror.or(pending.mirror);
                pending.limit = limit.or(pending.limit);
                self.pending_update = Some(pending);
            }
            Action::SetCommitment { target, value } => {
                let ids = self.resolve_targets(&target);
                for id in ids {
                    if let Some(agent) = self.find_individual_mut(id) {
                        let v = value.clamp(0.0, 1.0);
                        agent.modulation.set_persistence(v);
                    } else {
                        warn!("SetCommitment: agent {id} not found");
                    }
                }
            }
            Action::SetDrift { target, value } => {
                let ids = self.resolve_targets(&target);
                for id in ids {
                    if let Some(agent) = self.find_individual_mut(id) {
                        // Map drift to exploration without coupling persistence.
                        let v = value.abs().clamp(0.0, 1.0);
                        agent.modulation.set_exploration(v);
                    } else {
                        warn!("SetDrift: agent {id} not found");
                    }
                }
            }
        }
    }

    pub fn take_pending_update(&mut self) -> Option<LandscapeUpdate> {
        self.pending_update.take()
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
        if !self.individuals.is_empty() {
            for agent in self.individuals.iter_mut() {
                if agent.is_alive() {
                    agent.render_wave(
                        &mut self.buffers.audio,
                        fs,
                        current_frame,
                        dt_sec,
                        landscape,
                        self.global_coupling,
                    );
                }
            }
        }

        if self.abort_requested {
            if !self.individuals.is_empty() {
                let step = 1.0 / (0.05 * fs.max(1.0)); // fade over ~50ms
                for s in &mut self.buffers.audio {
                    *s *= self.shutdown_gain;
                    self.shutdown_gain -= step;
                    if self.shutdown_gain <= 0.0 {
                        self.shutdown_gain = 0.0;
                    }
                }
                if self.shutdown_gain <= 0.0 {
                    self.individuals.clear();
                }
            } else {
                self.buffers.audio.fill(0.0);
            }
        }

        &self.buffers.audio
    }

    /// Render spectral bodies on the provided log2 axis (explicit for external DSP handoff).
    pub fn process_frame(
        &mut self,
        current_frame: u64,
        space: &Log2Space,
        dt_sec: f32,
        scenario_finished: bool,
    ) -> &[f32] {
        self.current_frame = current_frame;
        self.buffers.amps.resize(space.n_bins(), 0.0);
        self.buffers.amps.fill(0.0);
        for agent in self.individuals.iter_mut() {
            if agent.is_alive() {
                agent.render_spectrum(&mut self.buffers.amps, space);
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

fn harmonic_density_weight(c01: f32, occupied: bool) -> f32 {
    if occupied {
        return 0.0;
    }
    let c = c01.clamp(0.0, 1.0);
    let eps = 1e-6f32;
    eps + (1.0 - eps) * c
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn decide_frequency_uses_consonance01() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let mut landscape = LandscapeFrame::new(space.clone());
        landscape.consonance.fill(-10.0);
        landscape.consonance01.fill(0.0);

        let idx_high = space.index_of_freq(200.0).expect("idx");
        let idx_raw = space.index_of_freq(300.0).expect("idx");
        landscape.consonance01[idx_high] = 1.0;
        landscape.consonance[idx_raw] = 10.0;

        let pop = Population::new(48_000.0);
        let method = SpawnMethod::Harmonicity {
            min_freq: 100.0,
            max_freq: 400.0,
            min_dist_erb: Some(0.0),
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let freq = pop.decide_frequency(&method, &landscape, &mut rng);
        let picked_idx = space.index_of_freq(freq).expect("picked idx");
        assert_eq!(picked_idx, idx_high);
    }

    #[test]
    fn harmonic_density_weight_eps_floor() {
        let w = harmonic_density_weight(0.0, false);
        assert!(w > 0.0);
    }

    #[test]
    fn harmonic_density_weight_occupied_is_zero() {
        let w = harmonic_density_weight(1.0, true);
        assert_eq!(w, 0.0);
    }

    #[test]
    fn harmonic_density_weighted_index_accepts_zero_c01() {
        let weights = vec![
            harmonic_density_weight(0.0, false),
            harmonic_density_weight(0.0, false),
        ];
        assert!(WeightedIndex::new(&weights).is_ok());
    }

    #[test]
    fn harmonic_density_weighted_index_fails_when_all_occupied() {
        let weights = vec![
            harmonic_density_weight(1.0, true),
            harmonic_density_weight(0.2, true),
        ];
        assert!(WeightedIndex::new(&weights).is_err());
    }
}

fn wildcard_match(pattern: &str, text: &str) -> bool {
    fn helper(pat: &[u8], text: &[u8]) -> bool {
        if pat.is_empty() {
            return text.is_empty();
        }
        match pat[0] {
            b'*' => {
                if pat.len() == 1 {
                    return true;
                }
                for i in 0..=text.len() {
                    if helper(&pat[1..], &text[i..]) {
                        return true;
                    }
                }
                false
            }
            c => {
                if text.is_empty() || c != text[0] {
                    false
                } else {
                    helper(&pat[1..], &text[1..])
                }
            }
        }
    }
    if pattern == "*" {
        return true;
    }
    helper(pattern.as_bytes(), text.as_bytes())
}
