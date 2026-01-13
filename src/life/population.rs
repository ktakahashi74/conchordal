use super::control::{AgentControl, PitchConstraintMode, matches_tag_pattern, merge_json};
use super::individual::{AgentMetadata, Individual, PhonationBatch, SoundBody};
use super::scenario::{Action, IndividualConfig, SocialConfig, SpawnMethod};
use crate::core::landscape::{LandscapeFrame, LandscapeUpdate};
use crate::core::log2space::Log2Space;
use crate::core::timebase::{Tick, Timebase};
use crate::life::audio::{LifeEvent, VoiceTarget};
use crate::life::social_density::SocialDensityTrace;
use crate::life::world_model::WorldModel;
use rand::{Rng, SeedableRng, distr::Distribution, distr::weighted::WeightedIndex, rngs::SmallRng};
use std::hash::{Hash, Hasher};
use tracing::{debug, info, warn};

#[derive(Default)]
struct WorkBuffers {
    amps: Vec<f32>,
}

pub struct Population {
    pub individuals: Vec<Individual>,
    current_frame: u64,
    pub abort_requested: bool,
    buffers: WorkBuffers,
    pub global_coupling: f32,
    shutdown_gain: f32,
    pending_update: Option<LandscapeUpdate>,
    time: Timebase,
    seed: u64,
    next_agent_id: u64,
    spawn_counter: u64,
    social_trace: Option<SocialDensityTrace>,
    life_events: Vec<LifeEvent>,
}

impl Population {
    const REMOVE_FADE_SEC_DEFAULT: f32 = 0.05;
    const CONTROL_STEP_SAMPLES: usize = 64;
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

    pub fn new(time: Timebase) -> Self {
        debug!("Population sample rate: {:.1} Hz", time.fs);
        Self {
            individuals: Vec::new(),
            current_frame: 0,
            abort_requested: false,
            buffers: WorkBuffers::default(),
            global_coupling: 1.0,
            shutdown_gain: 1.0,
            pending_update: None,
            time,
            seed: rand::random::<u64>(),
            next_agent_id: 1,
            spawn_counter: 0,
            social_trace: None,
            life_events: Vec::new(),
        }
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    fn spawn_seed(&self, tag: &str, count: u32, seq: u64) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.seed.hash(&mut hasher);
        self.current_frame.hash(&mut hasher);
        seq.hash(&mut hasher);
        count.hash(&mut hasher);
        tag.hash(&mut hasher);
        hasher.finish() ^ 0x9E37_79B9_7F4A_7C15
    }

    fn find_individual_mut(&mut self, id: u64) -> Option<&mut Individual> {
        self.individuals.iter_mut().find(|a| a.id() == id)
    }

    pub fn add_individual(&mut self, individual: Individual) {
        let id = individual.id();
        if self.individuals.iter().any(|a| a.id() == id) {
            warn!("AddIndividual: id collision for {id}");
            return;
        }
        self.individuals.push(individual);
    }

    pub fn set_current_frame(&mut self, frame: u64) {
        self.current_frame = frame;
    }

    pub fn drain_life_events(&mut self, out: &mut Vec<LifeEvent>) {
        out.clear();
        out.append(&mut self.life_events);
    }

    pub fn fill_voice_targets(&self, out: &mut Vec<VoiceTarget>) {
        out.clear();
        out.reserve(self.individuals.len());
        for agent in &self.individuals {
            if !agent.is_alive() {
                continue;
            }
            let pitch_hz = agent.body.base_freq_hz();
            let amp = agent.body.amp();
            let body = agent.body.body_spec();
            out.push(VoiceTarget {
                id: agent.id(),
                pitch_hz,
                amp,
                body,
            });
        }
    }

    pub fn publish_intents(
        &mut self,
        world: &mut WorldModel,
        landscape: &LandscapeFrame,
        now: Tick,
    ) -> Vec<PhonationBatch> {
        let mut batches = Vec::new();
        let count = self.publish_intents_into(world, landscape, now, &mut batches);
        batches.truncate(count);
        batches
    }

    pub(crate) fn publish_intents_into(
        &mut self,
        world: &mut WorldModel,
        landscape: &LandscapeFrame,
        now: Tick,
        out: &mut Vec<PhonationBatch>,
    ) -> usize {
        let hop_tick = (world.time.hop as Tick).max(1);
        let tb = &world.time;
        let frame_end = now.saturating_add(hop_tick);
        let mut used = 0usize;
        let social_trace = self.social_trace.as_ref();
        for agent in &mut self.individuals {
            let social_coupling = agent.phonation_social.coupling;
            if used == out.len() {
                out.push(PhonationBatch::default());
            }
            let batch = &mut out[used];
            agent.tick_phonation_into(
                tb,
                now,
                &landscape.rhythm,
                social_trace,
                social_coupling,
                batch,
            );
            let has_output =
                !(batch.cmds.is_empty() && batch.notes.is_empty() && batch.onsets.is_empty());
            if has_output {
                used += 1;
            }
        }
        let active_batches = &out[..used];
        let social_enabled =
            social_trace_enabled_from_configs(self.individuals.iter().map(|a| a.phonation_social));
        if social_enabled {
            let (bin_ticks, smooth) = social_trace_params(&self.individuals, hop_tick);
            self.social_trace = Some(build_social_trace_from_batches(
                active_batches,
                frame_end,
                hop_tick,
                bin_ticks,
                smooth,
                self.individuals.len(),
            ));
        } else {
            self.social_trace = None;
        }
        used
    }

    fn resolve_target_ids(&self, pattern: &str) -> Vec<u64> {
        self.individuals
            .iter()
            .filter_map(|a| {
                let meta = a.metadata();
                match meta.tag.as_deref() {
                    Some(tag) if matches_tag_pattern(pattern, tag) => Some(meta.id),
                    _ => None,
                }
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
                    if !min_l.is_finite() || !max_l.is_finite() || min_l >= max_l {
                        return min_freq.max(1e-6);
                    }
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
                if !min_l.is_finite() || !max_l.is_finite() || min_l >= max_l {
                    return min_freq.max(1e-6);
                }
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
        _landscape_rt: Option<&mut crate::core::stream::roughness::RoughnessStream>,
    ) {
        match action {
            Action::Finish => {
                self.abort_requested = true;
            }
            Action::Spawn {
                tag,
                count,
                opts,
                patch,
            } => {
                let spawn_seq = self.spawn_counter;
                self.spawn_counter = self.spawn_counter.wrapping_add(1);
                let seed = self.spawn_seed(&tag, count, spawn_seq);
                let mut rng = SmallRng::seed_from_u64(seed);
                let mut patch = patch;
                let legacy_method = take_spawn_method(&mut patch, &tag);
                let opts_method = opts.and_then(|opts| opts.method);
                let mut control = match AgentControl::from_json(merge_json(
                    AgentControl::default().to_json().unwrap_or_default(),
                    patch.clone(),
                )) {
                    Ok(control) => control,
                    Err(err) => {
                        warn!("Spawn: invalid patch for tag={}: {}", tag, err);
                        return;
                    }
                };
                let center_in_patch = patch_sets_center_hz(&patch);
                if let Some(method) = opts_method {
                    let freq = self.decide_frequency(&method, landscape, &mut rng);
                    control.pitch.constraint.mode = PitchConstraintMode::Lock;
                    control.pitch.constraint.freq_hz = Some(freq.max(1.0));
                } else if !center_in_patch {
                    let method = legacy_method.unwrap_or_default();
                    let freq = self.decide_frequency(&method, landscape, &mut rng);
                    control.pitch.center_hz = freq.max(1.0);
                }
                if matches!(control.pitch.constraint.mode, PitchConstraintMode::Lock)
                    && let Some(freq) = control.pitch.constraint.freq_hz
                {
                    control.pitch.center_hz = freq.max(1.0);
                }
                for i in 0..count {
                    let id = self.next_agent_id;
                    self.next_agent_id = self.next_agent_id.wrapping_add(1);
                    let metadata = AgentMetadata {
                        id,
                        tag: Some(tag.clone()),
                        group_idx: 0,
                        member_idx: i as usize,
                    };
                    let cfg = IndividualConfig {
                        control: control.clone(),
                        tag: Some(tag.clone()),
                    };
                    let spawned =
                        cfg.spawn(id, self.current_frame, metadata, self.time.fs, self.seed);
                    self.add_individual(spawned);
                    self.life_events.push(LifeEvent::Spawned { id });
                }
            }
            Action::Set { target, patch } => {
                let ids = self.resolve_target_ids(&target);
                for id in ids {
                    if let Some(agent) = self.find_individual_mut(id) {
                        match agent.apply_control_patch(patch.clone()) {
                            Ok(()) => {}
                            Err(err) => {
                                warn!("Set: agent {id} rejected patch: {}", err);
                                continue;
                            }
                        }
                    } else {
                        warn!("Set: agent {id} not found");
                        continue;
                    }
                }
            }
            Action::Unset { target, path } => {
                let ids = self.resolve_target_ids(&target);
                for id in ids {
                    if let Some(agent) = self.find_individual_mut(id) {
                        match agent.apply_unset_path(&path) {
                            Ok(_removed) => {}
                            Err(err) => {
                                warn!("Unset: agent {id} rejected path {}: {}", path, err);
                                continue;
                            }
                        }
                    } else {
                        warn!("Unset: agent {id} not found");
                        continue;
                    }
                }
            }
            Action::Remove { target } => {
                let ids = self.resolve_target_ids(&target);
                for id in ids {
                    if let Some(agent) = self.find_individual_mut(id) {
                        agent.start_remove_fade(Self::REMOVE_FADE_SEC_DEFAULT);
                    } else {
                        warn!("Remove: agent {id} not found");
                    }
                }
            }
            Action::Release { target, fade_sec } => {
                let ids = self.resolve_target_ids(&target);
                let fade_sec = fade_sec.max(0.0);
                for id in ids {
                    if let Some(agent) = self.find_individual_mut(id) {
                        agent.start_remove_fade(fade_sec);
                    } else {
                        warn!("Release: agent {id} not found");
                    }
                }
            }
            Action::SetHarmonicity { update } => {
                self.merge_landscape_update(update);
            }
            Action::SetGlobalCoupling { value } => {
                self.global_coupling = value.max(0.0);
            }
            Action::SetRoughnessTolerance { value } => {
                let update = LandscapeUpdate {
                    roughness_k: Some(value),
                    ..LandscapeUpdate::default()
                };
                self.merge_landscape_update(update);
            }
            Action::PostIntent { .. } => {}
        }
    }

    fn merge_landscape_update(&mut self, update: LandscapeUpdate) {
        let mut merged = self.pending_update.unwrap_or_default();
        if update.mirror.is_some() {
            merged.mirror = update.mirror;
        }
        if update.limit.is_some() {
            merged.limit = update.limit;
        }
        if update.roughness_k.is_some() {
            merged.roughness_k = update.roughness_k;
        }
        self.pending_update = Some(merged);
    }

    pub fn take_pending_update(&mut self) -> Option<LandscapeUpdate> {
        self.pending_update.take()
    }

    /// Assumes `set_current_frame` has been called for the current hop.
    pub fn remove_agent(&mut self, id: u64) {
        let mut next = Vec::with_capacity(self.individuals.len());
        for agent in self.individuals.drain(..) {
            if agent.id() != id {
                next.push(agent);
            }
        }
        self.individuals = next;
    }

    /// Advance agent state without emitting audio (ScheduleRenderer is output authority).
    /// `samples_len` controls sub-stepping of control-rate updates within the block.
    pub fn advance(
        &mut self,
        samples_len: usize,
        _fs: f32,
        current_frame: u64,
        dt_sec: f32,
        landscape: &crate::core::landscape::Landscape,
    ) {
        self.current_frame = current_frame;
        if !dt_sec.is_finite() || dt_sec <= 0.0 {
            return;
        }
        // Sub-step updates to keep control-rate integration stable across hop sizes.
        let steps = (samples_len / Self::CONTROL_STEP_SAMPLES).max(1);
        let dt_step_sec = dt_sec / steps as f32;
        if !dt_step_sec.is_finite() || dt_step_sec <= 0.0 {
            return;
        }
        let mut rhythms = landscape.rhythm;
        for _ in 0..steps {
            for agent in self.individuals.iter_mut() {
                if agent.is_alive() {
                    agent.tick_control(dt_step_sec, &rhythms, landscape, self.global_coupling);
                }
            }
            rhythms.advance_in_place(dt_step_sec);
        }

        if self.abort_requested {
            let step = dt_sec / 0.05; // fade over ~50ms
            if step.is_finite() && step > 0.0 {
                self.shutdown_gain = (self.shutdown_gain - step).max(0.0);
            }
            if self.shutdown_gain <= 0.0 {
                self.individuals.clear();
            }
        }
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
        let mut next = Vec::with_capacity(self.individuals.len());
        for agent in self.individuals.drain(..) {
            if agent.should_retain() {
                next.push(agent);
            }
        }
        self.individuals = next;
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

fn build_social_trace_from_batches(
    phonation_batches: &[PhonationBatch],
    frame_end: Tick,
    hop_tick: Tick,
    bin_ticks: u32,
    smooth: f32,
    population_size: usize,
) -> SocialDensityTrace {
    let mut onset_ticks = Vec::new();
    for batch in phonation_batches {
        for onset in &batch.onsets {
            onset_ticks.push((onset.onset_tick.saturating_add(hop_tick), onset.strength));
        }
    }
    SocialDensityTrace::from_onsets(
        frame_end,
        frame_end.saturating_add(hop_tick),
        bin_ticks,
        smooth,
        population_size,
        &onset_ticks,
    )
}

fn social_trace_params(individuals: &[Individual], hop_tick: Tick) -> (u32, f32) {
    let base = individuals
        .iter()
        .find(|agent| agent.phonation_social.bin_ticks != 0 || agent.phonation_social.smooth != 0.0)
        .map(|agent| agent.phonation_social)
        .unwrap_or_default();
    if individuals.iter().any(|agent| {
        agent.phonation_social.bin_ticks != base.bin_ticks
            || agent.phonation_social.smooth != base.smooth
    }) {
        warn!("Population social trace params differ across agents; using first match settings");
    }
    let auto_bin = (hop_tick / 64).max(1);
    let bin_ticks = if base.bin_ticks == 0 {
        auto_bin.min(u32::MAX as Tick) as u32
    } else {
        base.bin_ticks
    };
    (bin_ticks, base.smooth)
}

fn social_trace_enabled_from_configs<I>(configs: I) -> bool
where
    I: IntoIterator<Item = SocialConfig>,
{
    configs.into_iter().any(|cfg| cfg.coupling != 0.0)
}

fn patch_sets_center_hz(patch: &serde_json::Value) -> bool {
    let serde_json::Value::Object(map) = patch else {
        return false;
    };
    let Some(serde_json::Value::Object(pitch)) = map.get("pitch") else {
        return false;
    };
    pitch.contains_key("center_hz")
}

fn take_spawn_method(patch: &mut serde_json::Value, tag: &str) -> Option<SpawnMethod> {
    let serde_json::Value::Object(map) = patch else {
        return None;
    };
    let method_val = map.remove("method")?;
    match serde_json::from_value(method_val) {
        Ok(method) => Some(method),
        Err(err) => {
            warn!("Spawn: invalid method for tag={}: {}", tag, err);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::landscape::Landscape;
    use crate::life::control::{BodyMethod, PhonationType};
    use crate::life::individual::{AnyArticulationCore, ArticulationWrapper, DroneCore};
    use crate::life::intent::BodySnapshot;
    use crate::life::phonation_engine::{OnsetEvent, PhonationCmd, PhonationKick};
    use crate::life::scenario::SocialConfig;
    use crate::life::world_model::WorldModel;
    use rand::SeedableRng;
    use serde_json::json;

    fn make_dummy_note_spec() -> crate::life::individual::PhonationNoteSpec {
        crate::life::individual::PhonationNoteSpec {
            note_id: 1,
            onset: 0,
            hold_ticks: None,
            freq_hz: 440.0,
            amp: 0.5,
            smoothing_tau_sec: 0.0,
            body: BodySnapshot {
                kind: "sine".to_string(),
                amp_scale: 1.0,
                brightness: 0.0,
                noise_mix: 0.0,
            },
            articulation: ArticulationWrapper::new(
                AnyArticulationCore::Drone(DroneCore {
                    phase: 0.0,
                    sway_rate: 1.0,
                }),
                1.0,
            ),
        }
    }

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

        let pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
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

    #[test]
    fn social_trace_is_delayed_by_one_hop() {
        let batch = PhonationBatch {
            source_id: 1,
            cmds: Vec::new(),
            notes: Vec::new(),
            onsets: vec![OnsetEvent {
                gate: 0,
                onset_tick: 90,
                strength: 1.0,
            }],
        };
        let trace = build_social_trace_from_batches(&[batch], 100, 10, 5, 0.0, 1);
        assert_eq!(trace.start_tick, 100);
        assert_eq!(trace.density_at(95), 0.0);
        assert_eq!(trace.density_at(100), 1.0);
    }

    #[test]
    fn social_trace_enabled_with_nonzero_coupling() {
        let base = SocialConfig {
            coupling: 0.0,
            bin_ticks: 0,
            smooth: 0.0,
        };
        let other = SocialConfig {
            coupling: 1.0,
            bin_ticks: 0,
            smooth: 0.0,
        };
        let configs = vec![base, other];
        assert!(social_trace_enabled_from_configs(configs));
    }

    #[test]
    fn glob_matches_wildcards_and_escapes() {
        assert!(matches_tag_pattern("a*", "a1"));
        assert!(matches_tag_pattern("a*", "a2"));
        assert!(!matches_tag_pattern("a*", "b1"));
        assert!(matches_tag_pattern(r"a\*", "a*"));
        assert!(!matches_tag_pattern(r"a\*", "a1"));
        assert!(matches_tag_pattern("a?b", "acb"));
        assert!(!matches_tag_pattern("a?b", "ab"));
    }

    #[test]
    fn set_applies_to_matching_tags() {
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        for tag in ["lead_1", "lead_2", "bass_1"] {
            pop.apply_action(
                Action::Spawn {
                    tag: tag.to_string(),
                    count: 1,
                    opts: None,
                    patch: json!({}),
                },
                &landscape,
                None,
            );
        }
        pop.apply_action(
            Action::Set {
                target: "lead_*".to_string(),
                patch: json!({
                    "body": { "amp": 0.42 }
                }),
            },
            &landscape,
            None,
        );
        assert_eq!(pop.individuals.len(), 3);
        for agent in &pop.individuals {
            let tag = agent.metadata.tag.as_deref().unwrap_or_default();
            let amp = agent.effective_control.body.amp;
            if tag.starts_with("lead_") {
                assert!((amp - 0.42).abs() <= 1e-6, "tag {tag} should match");
            } else {
                assert!(
                    (amp - AgentControl::default().body.amp).abs() <= 1e-6,
                    "tag {tag} should not match"
                );
            }
        }
    }

    #[test]
    fn remove_applies_to_matching_tags() {
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        for tag in ["lead_1", "lead_2", "bass_1"] {
            pop.apply_action(
                Action::Spawn {
                    tag: tag.to_string(),
                    count: 1,
                    opts: None,
                    patch: json!({}),
                },
                &landscape,
                None,
            );
        }
        pop.apply_action(
            Action::Remove {
                target: "lead_?".to_string(),
            },
            &landscape,
            None,
        );
        let dt = 0.1;
        let samples_per_hop = (pop.time.fs * dt) as usize;
        let rt_landscape = Landscape::new(Log2Space::new(55.0, 4000.0, 64));
        pop.advance(samples_per_hop, pop.time.fs, 0, dt, &rt_landscape);
        pop.process_frame(0, &rt_landscape.space, dt, false);
        assert_eq!(pop.individuals.len(), 1);
        assert_eq!(pop.individuals[0].metadata.tag.as_deref(), Some("bass_1"));
    }

    #[test]
    fn unset_reverts_to_base_control() {
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        pop.apply_action(
            Action::Spawn {
                tag: "unset".to_string(),
                count: 1,
                opts: None,
                patch: json!({
                    "body": { "amp": 0.30 }
                }),
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::Set {
                target: "unset".to_string(),
                patch: json!({
                    "body": { "amp": 0.80 }
                }),
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::Unset {
                target: "unset".to_string(),
                path: "body.amp".to_string(),
            },
            &landscape,
            None,
        );
        let agent = pop.individuals.first().expect("agent exists");
        assert!((agent.effective_control.body.amp - 0.30).abs() <= 1e-6);
        pop.apply_action(
            Action::Unset {
                target: "unset".to_string(),
                path: "body.missing".to_string(),
            },
            &landscape,
            None,
        );
        let agent = pop.individuals.first().expect("agent exists");
        assert!((agent.effective_control.body.amp - 0.30).abs() <= 1e-6);
    }

    #[test]
    fn set_cannot_change_body_method() {
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        pop.apply_action(
            Action::Spawn {
                tag: "body_method".to_string(),
                count: 1,
                opts: None,
                patch: json!({
                    "body": { "method": "harmonic" }
                }),
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::Set {
                target: "body_method".to_string(),
                patch: json!({
                    "body": { "method": "sine" }
                }),
            },
            &landscape,
            None,
        );
        let agent = pop.individuals.first().expect("agent exists");
        assert_eq!(agent.effective_control.body.method, BodyMethod::Harmonic);
    }

    #[test]
    fn set_cannot_change_phonation_type() {
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        pop.apply_action(
            Action::Spawn {
                tag: "phonation_type".to_string(),
                count: 1,
                opts: None,
                patch: json!({
                    "phonation": { "type": "interval" }
                }),
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::Set {
                target: "phonation_type".to_string(),
                patch: json!({
                    "phonation": { "type": "hold" }
                }),
            },
            &landscape,
            None,
        );
        let agent = pop.individuals.first().expect("agent exists");
        assert_eq!(
            agent.effective_control.phonation.r#type,
            PhonationType::Interval
        );
    }

    #[test]
    fn publish_intents_into_clears_unused_batch() {
        let time = Timebase {
            fs: 48_000.0,
            hop: 64,
        };
        let space = Log2Space::new(55.0, 880.0, 12);
        let landscape = LandscapeFrame::new(space.clone());
        let mut world = WorldModel::new(time, space);
        let mut pop = Population::new(time);
        pop.apply_action(
            Action::Spawn {
                tag: "silent".to_string(),
                count: 1,
                opts: None,
                patch: serde_json::json!({
                    "pitch": { "center_hz": 440.0 },
                    "phonation": { "type": "none" }
                }),
            },
            &landscape,
            None,
        );

        let mut batches = vec![PhonationBatch {
            source_id: 99,
            cmds: vec![PhonationCmd::NoteOn {
                note_id: 1,
                kick: PhonationKick::Planned { strength: 1.0 },
            }],
            notes: vec![make_dummy_note_spec()],
            onsets: vec![OnsetEvent {
                gate: 0,
                onset_tick: 0,
                strength: 1.0,
            }],
        }];

        let used = pop.publish_intents_into(&mut world, &landscape, 0, &mut batches);
        assert_eq!(used, 0);
        assert!(batches[0].cmds.is_empty());
        assert!(batches[0].notes.is_empty());
        assert!(batches[0].onsets.is_empty());
    }
}
