use super::individual::{
    AgentMetadata, AnyArticulationCore, Individual, PhonationBatch, SoundBody,
};
use super::scenario::{Action, IndividualConfig, RespawnPolicy, SpawnStrategy};
use super::telemetry::LifeRecord;
use crate::core::landscape::{LandscapeFrame, LandscapeUpdate};
use crate::core::timebase::{Tick, Timebase};
use crate::life::control::{MAX_FREQ_HZ, MIN_FREQ_HZ};
use crate::life::social_density::SocialDensityTrace;
use crate::life::sound::{AudioCommand, VoiceTarget};
use crate::life::world_model::WorldModel;
use rand::{Rng, SeedableRng, distr::Distribution, distr::weighted::WeightedIndex, rngs::SmallRng};
use std::collections::{BTreeMap, HashSet};
use std::hash::{Hash, Hasher};
use tracing::{debug, info, warn};

#[derive(Clone, Copy, Debug, Default)]
pub struct PredGateStats {
    pub raw_min: f32,
    pub raw_max: f32,
    pub raw_mean: f32,
    pub mixed_min: f32,
    pub mixed_max: f32,
    pub mixed_mean: f32,
    pub sync_mean: f32,
    pub count: u32,
}

pub struct Population {
    pub individuals: Vec<Individual>,
    current_frame: u64,
    pub abort_requested: bool,
    pub global_coupling: f32,
    birth_energy: f32,
    shutdown_gain: f32,
    pending_update: Option<LandscapeUpdate>,
    time: Timebase,
    seed: u64,
    spawn_counter: u64,
    social_trace: Option<SocialDensityTrace>,
    audio_cmds: Vec<AudioCommand>,
    groups: BTreeMap<u64, RuntimeGroupState>,
    death_observed: HashSet<u64>,
    next_runtime_id: u64,
    last_pred_gate_stats: Option<PredGateStats>,
    last_gate_boundary_in_hop: Option<bool>,
    last_phonation_onsets_in_hop: Option<u32>,
    pub death_records: Vec<LifeRecord>,
}

#[derive(Debug, Clone)]
struct RuntimeGroupState {
    template: IndividualConfig,
    strategy: Option<SpawnStrategy>,
    respawn_policy: RespawnPolicy,
    crowding_target_same: bool,
    crowding_target_other: bool,
    released: bool,
    next_member_idx: usize,
    spawn_count_hint: usize,
    telemetry_first_k: Option<u32>,
    plv_window: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct SpawnParams {
    id: u64,
    group_id: u64,
    member_idx: usize,
    resolved_freq_hz: f32,
}

impl Population {
    const CONTROL_STEP_SAMPLES: usize = 64;
    /// Returns true if `freq_hz` is within `min_dist_erb` (ERB scale) of any existing agent's base
    /// frequency.
    pub fn is_range_occupied(&self, freq_hz: f32, min_dist_erb: f32) -> bool {
        self.is_range_occupied_with(freq_hz, min_dist_erb, &[])
    }

    fn is_range_occupied_with(&self, freq_hz: f32, min_dist_erb: f32, reserved: &[f32]) -> bool {
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
        for &freq in reserved {
            if !freq.is_finite() {
                continue;
            }
            let d_erb = (crate::core::erb::hz_to_erb(freq.max(1e-6)) - target_erb).abs();
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
            global_coupling: 1.0,
            birth_energy: 1.0,
            shutdown_gain: 1.0,
            pending_update: None,
            time,
            seed: rand::random::<u64>(),
            spawn_counter: 0,
            social_trace: None,
            audio_cmds: Vec::new(),
            groups: BTreeMap::new(),
            death_observed: HashSet::new(),
            next_runtime_id: 1,
            last_pred_gate_stats: None,
            last_gate_boundary_in_hop: None,
            last_phonation_onsets_in_hop: None,
            death_records: Vec::new(),
        }
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    fn spawn_seed(&self, group_id: u64, count: usize, seq: u64) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.seed.hash(&mut hasher);
        self.current_frame.hash(&mut hasher);
        seq.hash(&mut hasher);
        count.hash(&mut hasher);
        group_id.hash(&mut hasher);
        hasher.finish() ^ 0x9E37_79B9_7F4A_7C15
    }

    fn track_runtime_id(&mut self, id: u64) {
        if id >= self.next_runtime_id {
            self.next_runtime_id = id.saturating_add(1).max(1);
        }
    }

    fn allocate_runtime_id(&mut self) -> u64 {
        loop {
            let id = self.next_runtime_id.max(1);
            self.next_runtime_id = self.next_runtime_id.wrapping_add(1).max(1);
            if self.individuals.iter().all(|agent| agent.id() != id) {
                return id;
            }
        }
    }

    fn group_member_ids(&self, group_id: u64) -> Vec<u64> {
        self.individuals
            .iter()
            .filter(|agent| agent.metadata.group_id == group_id)
            .map(|agent| agent.id())
            .collect()
    }

    fn normal_sample<R: Rng + ?Sized>(rng: &mut R) -> f32 {
        let u1 = (1.0 - rng.random::<f32>()).max(1e-7);
        let u2 = rng.random::<f32>();
        let mag = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        mag * theta.cos()
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
        self.track_runtime_id(id);
        self.individuals.push(individual);
    }

    pub fn set_current_frame(&mut self, frame: u64) {
        self.current_frame = frame;
    }

    pub fn last_pred_gate_stats(&self) -> Option<PredGateStats> {
        self.last_pred_gate_stats
    }

    pub fn last_gate_boundary_in_hop(&self) -> Option<bool> {
        self.last_gate_boundary_in_hop
    }

    pub fn last_phonation_onsets_in_hop(&self) -> Option<u32> {
        self.last_phonation_onsets_in_hop
    }

    pub fn drain_audio_cmds(&mut self, out: &mut Vec<AudioCommand>) {
        out.clear();
        out.append(&mut self.audio_cmds);
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
            out.push(VoiceTarget {
                id: agent.id(),
                pitch_hz,
                amp,
                continuous_drive: agent.effective_control.body.continuous_drive,
                pitch_smooth_tau: agent.effective_control.body.pitch_smooth_tau,
            });
        }
    }

    pub fn collect_phonation_batches(
        &mut self,
        world: &mut WorldModel,
        landscape: &LandscapeFrame,
        now: Tick,
    ) -> Vec<PhonationBatch> {
        let mut batches = Vec::new();
        let count = self.collect_phonation_batches_into(world, landscape, now, &mut batches);
        batches.truncate(count);
        batches
    }

    pub(crate) fn collect_phonation_batches_into(
        &mut self,
        world: &mut WorldModel,
        landscape: &LandscapeFrame,
        now: Tick,
        out: &mut Vec<PhonationBatch>,
    ) -> usize {
        let tb = world.time;
        let hop_tick = (tb.hop as Tick).max(1);
        let frame_end = now.saturating_add(hop_tick);
        let gate_boundary_in_hop = world
            .next_gate_tick_est
            .is_some_and(|gate_tick| gate_tick > now && gate_tick <= frame_end);
        let pred_scan =
            world
                .predict_consonance_field_level_next_gate()
                .and_then(|(gate_tick, scan)| {
                    if gate_tick >= now && gate_tick < frame_end {
                        Some(scan)
                    } else {
                        None
                    }
                });
        let mut pred_count = 0u32;
        let mut pred_raw_sum = 0.0f32;
        let mut pred_raw_min = f32::INFINITY;
        let mut pred_raw_max = f32::NEG_INFINITY;
        let mut pred_mixed_sum = 0.0f32;
        let mut pred_mixed_min = f32::INFINITY;
        let mut pred_mixed_max = f32::NEG_INFINITY;
        let mut pred_sync_sum = 0.0f32;
        let mut phonation_onsets_in_hop = 0u32;
        let mut used = 0usize;
        let social_trace = self.social_trace.as_ref();
        for agent in &mut self.individuals {
            let social_coupling = agent.phonation_coupling;
            if used == out.len() {
                out.push(PhonationBatch::default());
            }
            let batch = &mut out[used];
            let mut extra_gate_gain = 1.0;
            if let Some(scan) = pred_scan.as_ref() {
                let mut gain_raw = world.sample_scan_field_level(scan, agent.body.base_freq_hz());
                if !gain_raw.is_finite() {
                    gain_raw = 0.0;
                }
                gain_raw = gain_raw.clamp(0.0, 1.0);
                let mut sync = agent.effective_control.phonation.sync;
                if !sync.is_finite() {
                    sync = 0.0;
                }
                sync = sync.clamp(0.0, 1.0);
                let mut mixed = mix_pred_gate_gain(sync, gain_raw);
                if !mixed.is_finite() {
                    mixed = 1.0;
                }
                pred_count = pred_count.saturating_add(1);
                pred_raw_sum += gain_raw;
                pred_raw_min = pred_raw_min.min(gain_raw);
                pred_raw_max = pred_raw_max.max(gain_raw);
                pred_mixed_sum += mixed;
                pred_mixed_min = pred_mixed_min.min(mixed);
                pred_mixed_max = pred_mixed_max.max(mixed);
                pred_sync_sum += sync;
                extra_gate_gain = mixed;
            }
            agent.tick_phonation_into(
                &tb,
                now,
                &landscape.rhythm,
                social_trace,
                social_coupling,
                extra_gate_gain,
                batch,
            );
            phonation_onsets_in_hop = phonation_onsets_in_hop
                .saturating_add(batch.onsets.len().min(u32::MAX as usize) as u32);
            let has_output =
                !(batch.cmds.is_empty() && batch.notes.is_empty() && batch.onsets.is_empty());
            if has_output {
                used += 1;
            }
        }
        let active_batches = &out[..used];
        let social_enabled = social_trace_enabled_from_couplings(
            self.individuals.iter().map(|a| a.phonation_coupling),
        );
        if social_enabled {
            let (bin_ticks, smooth) = social_trace_params(hop_tick);
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
        self.last_gate_boundary_in_hop = Some(gate_boundary_in_hop);
        self.last_phonation_onsets_in_hop = Some(phonation_onsets_in_hop);
        self.last_pred_gate_stats = if pred_count > 0 {
            let inv = 1.0 / pred_count as f32;
            Some(PredGateStats {
                raw_min: pred_raw_min,
                raw_max: pred_raw_max,
                raw_mean: pred_raw_sum * inv,
                mixed_min: pred_mixed_min,
                mixed_max: pred_mixed_max,
                mixed_mean: pred_mixed_sum * inv,
                sync_mean: pred_sync_sum * inv,
                count: pred_count,
            })
        } else {
            None
        };
        used
    }

    fn decide_frequency<R: Rng + ?Sized>(
        &self,
        strategy: &SpawnStrategy,
        landscape: &LandscapeFrame,
        rng: &mut R,
        reserved: &[f32],
    ) -> f32 {
        let space = &landscape.space;
        let n_bins = space.n_bins();
        if n_bins == 0 {
            return 440.0;
        }

        let (min_freq, max_freq) = strategy.freq_range_hz();

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

        let min_dist_erb = strategy.min_dist_erb();

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
                if !self.is_range_occupied_with(f, min_dist_erb, reserved) {
                    return f;
                }
            }
            center
        };

        let pick_idx = match strategy {
            SpawnStrategy::Consonance { .. } => {
                let mut best = idx_min;
                let mut best_val = f32::MIN;
                let mut found = false;
                for i in idx_min..=idx_max {
                    let f = space.freq_of_index(i);
                    if self.is_range_occupied_with(f, min_dist_erb, reserved) {
                        continue;
                    }
                    if let Some(&c_val) = landscape.consonance_field_level.get(i)
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
                        if let Some(&c_val) = landscape.consonance_field_level.get(i)
                            && c_val > best_val
                        {
                            best_val = c_val;
                            best = i;
                        }
                    }
                    best
                }
            }
            SpawnStrategy::ConsonanceDensity { .. } => {
                let mut range_min = idx_min;
                let mut range_max = idx_max;
                if range_max < range_min {
                    std::mem::swap(&mut range_min, &mut range_max);
                }
                debug_assert!(range_min <= range_max);

                let range_len = range_max - range_min + 1;
                let mut occupied_flags = vec![false; range_len];
                let mut weights = vec![0.0f32; range_len];
                let mut sum = 0.0f32;
                let mut unoccupied_count = 0usize;

                for (offset, i) in (range_min..=range_max).enumerate() {
                    let f = space.freq_of_index(i);
                    let occupied_i = self.is_range_occupied_with(f, min_dist_erb, reserved);
                    occupied_flags[offset] = occupied_i;
                    if !occupied_i {
                        unoccupied_count += 1;
                    }
                    let raw = landscape
                        .consonance_density_mass
                        .get(i)
                        .copied()
                        .unwrap_or(0.0);
                    let mut w = if occupied_i { 0.0 } else { raw.max(0.0) };
                    if !w.is_finite() {
                        w = 0.0;
                    }
                    weights[offset] = w;
                    sum += w;
                }

                if !(sum > 0.0 && sum.is_finite()) {
                    if unoccupied_count > 0 {
                        for (offset, occupied_i) in occupied_flags.iter().enumerate() {
                            weights[offset] = if *occupied_i { 0.0 } else { 1.0 };
                        }
                    } else {
                        weights.fill(1.0);
                    }
                }

                if let Ok(dist) = WeightedIndex::new(&weights) {
                    range_min + dist.sample(rng)
                } else {
                    // Defensive fallback; this path should be unreachable after range-local
                    // fallback weights, but keep behavior safe and unbiased in-range.
                    range_min + rng.random_range(0..range_len)
                }
            }
            SpawnStrategy::RandomLog { .. } => {
                let min_l = min_freq.log2();
                let max_l = max_freq.log2();
                if !min_l.is_finite() || !max_l.is_finite() || min_l >= max_l {
                    return min_freq.max(1e-6);
                }
                for _ in 0..32 {
                    let r = rng.random_range(min_l..max_l);
                    let f = 2.0f32.powf(r);
                    if !self.is_range_occupied_with(f, min_dist_erb, reserved) {
                        return f;
                    }
                }
                return 2.0f32.powf(rng.random_range(min_l..max_l));
            }
            SpawnStrategy::Linear { .. } => idx_min,
        };

        jitter_free_bin(pick_idx, rng)
    }

    fn resolve_strategy_frequency<R: Rng + ?Sized>(
        &self,
        strategy: &SpawnStrategy,
        landscape: &LandscapeFrame,
        rng: &mut R,
        reserved: &[f32],
        member_idx: usize,
        member_count: usize,
    ) -> f32 {
        match strategy {
            SpawnStrategy::Linear {
                start_freq,
                end_freq,
            } => {
                if member_count <= 1 {
                    *start_freq
                } else {
                    let t = member_idx as f32 / (member_count - 1) as f32;
                    start_freq + (end_freq - start_freq) * t
                }
            }
            _ => self.decide_frequency(strategy, landscape, rng, reserved),
        }
    }

    fn spawn_one(
        &mut self,
        params: SpawnParams,
        spec: &IndividualConfig,
        landscape: &LandscapeFrame,
    ) {
        let SpawnParams {
            id,
            group_id,
            member_idx,
            resolved_freq_hz,
        } = params;
        if self.individuals.iter().any(|agent| agent.id() == id) {
            warn!("Spawn: id collision for {id} in group {group_id}");
            return;
        }

        let mut control = spec.control.clone();
        control.pitch.freq = resolved_freq_hz.clamp(MIN_FREQ_HZ, MAX_FREQ_HZ);
        let metadata = AgentMetadata {
            group_id,
            member_idx,
        };
        let cfg = IndividualConfig {
            control: control.clone(),
            articulation: spec.articulation.clone(),
        };
        let mut spawned = cfg.spawn_with_landscape(
            id,
            self.current_frame,
            metadata,
            self.time.fs,
            Some(landscape),
            self.seed,
        );
        // Apply group-level telemetry/PLV settings from RuntimeGroupState
        if let Some(group) = self.groups.get(&group_id) {
            if let Some(first_k) = group.telemetry_first_k {
                spawned.life_accumulator = Some(super::telemetry::LifeAccumulator::new(
                    self.current_frame,
                    first_k,
                    0.0,
                ));
            }
            if let Some(window) = group.plv_window {
                if let AnyArticulationCore::Entrain(ref mut core) = spawned.articulation.core {
                    core.enable_plv(window);
                }
            }
        }
        let body = spawned.body_snapshot();
        let pitch_hz = spawned.body.base_freq_hz();
        let amp = spawned.body.amp();
        self.individuals.push(spawned);
        self.track_runtime_id(id);
        self.audio_cmds.push(AudioCommand::EnsureVoice {
            id,
            body,
            pitch_hz,
            amp,
        });
        self.audio_cmds.push(AudioCommand::Impulse {
            id,
            energy: self.birth_energy,
        });
    }

    fn ensure_group_state(
        &mut self,
        group_id: u64,
        spec: IndividualConfig,
        strategy: Option<SpawnStrategy>,
        member_count: usize,
    ) {
        let current_members = self
            .individuals
            .iter()
            .filter(|agent| agent.metadata.group_id == group_id)
            .count();
        if let Some(group) = self.groups.get_mut(&group_id) {
            // Runtime currently allows multiple Spawn actions with the same group_id.
            // In that case we treat it as "refresh group template/strategy" while preserving
            // existing runtime policies. New Spawn implicitly re-activates the group.
            group.template = spec;
            group.strategy = strategy;
            group.released = false;
            group.spawn_count_hint = member_count.max(1);
            group.next_member_idx = group.next_member_idx.max(current_members);
            return;
        }
        self.groups.insert(
            group_id,
            RuntimeGroupState {
                template: spec,
                strategy,
                respawn_policy: RespawnPolicy::None,
                crowding_target_same: true,
                crowding_target_other: false,
                released: false,
                next_member_idx: current_members.max(member_count),
                spawn_count_hint: member_count.max(1),
                telemetry_first_k: None,
                plv_window: None,
            },
        );
    }

    fn apply_group_update(&mut self, group_id: u64, update: &super::control::ControlUpdate) {
        if let Some(group) = self.groups.get_mut(&group_id) {
            group.template.control.apply_update(update);
        }
    }

    fn mark_group_released(&mut self, group_id: u64) {
        if let Some(group) = self.groups.get_mut(&group_id) {
            group.released = true;
        }
    }

    fn set_group_respawn_policy(&mut self, group_id: u64, policy: RespawnPolicy) {
        if let Some(group) = self.groups.get_mut(&group_id) {
            group.respawn_policy = policy;
        } else {
            warn!("SetRespawnPolicy: unknown group {group_id}");
        }
    }

    fn set_group_crowding_target(
        &mut self,
        group_id: u64,
        same_group_visible: bool,
        other_group_visible: bool,
    ) {
        if let Some(group) = self.groups.get_mut(&group_id) {
            group.crowding_target_same = same_group_visible;
            group.crowding_target_other = other_group_visible;
        } else {
            warn!("SetGroupCrowdingTarget: unknown group {group_id}");
        }
    }

    fn enable_group_telemetry(&mut self, group_id: u64, first_k: u32) {
        if let Some(group) = self.groups.get_mut(&group_id) {
            group.telemetry_first_k = Some(first_k);
        }
        let current_frame = self.current_frame;
        for agent in self
            .individuals
            .iter_mut()
            .filter(|a| a.metadata.group_id == group_id)
        {
            if agent.life_accumulator.is_none() {
                let consonance = 0.0; // initial consonance unknown at enable time
                agent.life_accumulator = Some(super::telemetry::LifeAccumulator::new(
                    current_frame,
                    first_k,
                    consonance,
                ));
            }
        }
    }

    fn enable_group_plv(&mut self, group_id: u64, window: usize) {
        if let Some(group) = self.groups.get_mut(&group_id) {
            group.plv_window = Some(window);
        }
        for agent in self
            .individuals
            .iter_mut()
            .filter(|a| a.metadata.group_id == group_id)
        {
            if let AnyArticulationCore::Entrain(ref mut core) = agent.articulation.core {
                if core.sliding_plv.is_none() {
                    core.enable_plv(window);
                }
            }
        }
    }

    #[inline]
    fn pairwise_split_sign(a: u64, b: u64) -> f32 {
        if a == b {
            return 0.0;
        }
        let (lo, hi, orient) = if a < b { (a, b, 1.0) } else { (b, a, -1.0) };
        // Deterministic pair hash; orientation restores anti-symmetry:
        // sign(a,b) == -sign(b,a).
        let mut x = lo
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add(hi.rotate_left(32))
            ^ 0xA076_1D64_78BD_642F;
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 31;
        let pair_sign = if (x & 1) == 0 { 1.0 } else { -1.0 };
        orient * pair_sign
    }

    fn random_respawn_frequency<R: Rng + ?Sized>(
        &self,
        group: &RuntimeGroupState,
        landscape: &LandscapeFrame,
        rng: &mut R,
        member_idx: usize,
    ) -> f32 {
        if let Some(strategy) = group.strategy.as_ref() {
            let linear_idx = member_idx % group.spawn_count_hint.max(1);
            self.resolve_strategy_frequency(
                strategy,
                landscape,
                rng,
                &[],
                linear_idx,
                group.spawn_count_hint.max(1),
            )
            .max(MIN_FREQ_HZ)
        } else {
            group
                .template
                .control
                .pitch
                .freq
                .clamp(MIN_FREQ_HZ, MAX_FREQ_HZ)
        }
    }

    fn hereditary_respawn_frequency<R: Rng + ?Sized>(
        &self,
        group: &RuntimeGroupState,
        sigma_oct: f32,
        parent_freqs_hz: &[f32],
        landscape: &LandscapeFrame,
        rng: &mut R,
        member_idx: usize,
    ) -> f32 {
        if parent_freqs_hz.is_empty() {
            return self.random_respawn_frequency(group, landscape, rng, member_idx);
        }

        let parent_idx = rng.random_range(0..parent_freqs_hz.len());
        let parent_log2 = parent_freqs_hz[parent_idx].max(MIN_FREQ_HZ).log2();
        let noise = Self::normal_sample(rng) * sigma_oct.max(0.0);
        let child_log2 = parent_log2 + noise;
        let (min_hz, max_hz) = group
            .strategy
            .as_ref()
            .map(SpawnStrategy::freq_range_hz)
            .unwrap_or_else(|| landscape.freq_bounds());
        let lo = min_hz.clamp(MIN_FREQ_HZ, MAX_FREQ_HZ);
        let hi = max_hz.clamp(lo, MAX_FREQ_HZ);
        let child_hz = 2.0f32.powf(child_log2).clamp(lo, hi);
        child_hz.clamp(MIN_FREQ_HZ, MAX_FREQ_HZ)
    }

    fn respawn_on_new_deaths(&mut self, scenario_finished: bool, landscape: &LandscapeFrame) {
        if scenario_finished || self.abort_requested {
            return;
        }

        let mut statuses = Vec::with_capacity(self.individuals.len());
        let mut alive_by_group: BTreeMap<u64, Vec<f32>> = BTreeMap::new();
        for agent in &self.individuals {
            let alive = agent.is_alive();
            let group_id = agent.metadata.group_id;
            let id = agent.id();
            statuses.push((id, group_id, alive));
            if alive {
                alive_by_group
                    .entry(group_id)
                    .or_default()
                    .push(agent.body.base_freq_hz().clamp(MIN_FREQ_HZ, MAX_FREQ_HZ));
            }
        }

        let mut dead_candidates = Vec::new();
        for (id, group_id, alive) in statuses {
            if alive {
                self.death_observed.remove(&id);
                continue;
            }
            if self.death_observed.insert(id) {
                dead_candidates.push((id, group_id));
            }
        }

        for (_dead_id, group_id) in dead_candidates {
            let Some(group) = self.groups.get(&group_id).cloned() else {
                continue;
            };
            if group.released {
                continue;
            }

            let member_idx = group.next_member_idx;
            let spawn_seq = self.spawn_counter;
            self.spawn_counter = self.spawn_counter.wrapping_add(1);
            let seed = self.spawn_seed(group_id, 1, spawn_seq);
            let mut rng = SmallRng::seed_from_u64(seed);

            let freq_hz = match group.respawn_policy {
                RespawnPolicy::None => continue,
                RespawnPolicy::Random => {
                    self.random_respawn_frequency(&group, landscape, &mut rng, member_idx)
                }
                RespawnPolicy::Hereditary { sigma_oct } => self.hereditary_respawn_frequency(
                    &group,
                    sigma_oct,
                    alive_by_group
                        .get(&group_id)
                        .map(Vec::as_slice)
                        .unwrap_or(&[]),
                    landscape,
                    &mut rng,
                    member_idx,
                ),
            };

            let id = self.allocate_runtime_id();
            self.spawn_one(
                SpawnParams {
                    id,
                    group_id,
                    member_idx,
                    resolved_freq_hz: freq_hz,
                },
                &group.template,
                landscape,
            );

            if let Some(state) = self.groups.get_mut(&group_id) {
                state.next_member_idx = state.next_member_idx.saturating_add(1);
            }
            alive_by_group.entry(group_id).or_default().push(freq_hz);
        }
    }

    pub fn apply_action(
        &mut self,
        action: Action,
        landscape: &LandscapeFrame,
        _analysis_rt: Option<&mut crate::core::stream::analysis::AnalysisStream>,
    ) {
        match action {
            Action::Finish => {
                self.abort_requested = true;
            }
            Action::Spawn {
                group_id,
                ids,
                spec,
                strategy,
            } => {
                let spawn_seq = self.spawn_counter;
                self.spawn_counter = self.spawn_counter.wrapping_add(1);
                let seed = self.spawn_seed(group_id, ids.len(), spawn_seq);
                let mut rng = SmallRng::seed_from_u64(seed);
                let mut reserved = Vec::with_capacity(ids.len());
                let total = ids.len().max(1);
                for (member_idx, id) in ids.iter().copied().enumerate() {
                    let freq_hz = strategy
                        .as_ref()
                        .map(|strat| {
                            self.resolve_strategy_frequency(
                                strat, landscape, &mut rng, &reserved, member_idx, total,
                            )
                        })
                        .unwrap_or(spec.control.pitch.freq)
                        .max(MIN_FREQ_HZ);
                    self.spawn_one(
                        SpawnParams {
                            id,
                            group_id,
                            member_idx,
                            resolved_freq_hz: freq_hz,
                        },
                        &spec,
                        landscape,
                    );
                    reserved.push(freq_hz);
                }
                self.ensure_group_state(group_id, spec, strategy, total);
            }
            Action::Update {
                group_id,
                ids: _ids,
                update,
            } => {
                // Group-wide runtime semantics:
                // updates apply to all current members with matching group_id.
                let mut updated = 0usize;
                for agent in self
                    .individuals
                    .iter_mut()
                    .filter(|agent| agent.metadata.group_id == group_id)
                {
                    if let Err(err) = agent.apply_update(&update) {
                        warn!(
                            "Update: agent {} (group {group_id}) rejected update: {err}",
                            agent.id()
                        );
                    } else {
                        updated += 1;
                    }
                }
                if updated == 0 {
                    warn!("Update: no active members found for group {group_id}");
                }
                self.apply_group_update(group_id, &update);
            }
            Action::Release {
                group_id,
                ids: _ids,
                fade_sec,
            } => {
                // Group-wide runtime semantics:
                // release applies to all current members with matching group_id.
                let fade_sec = fade_sec.max(0.0);
                let member_ids = self.group_member_ids(group_id);
                if member_ids.is_empty() {
                    warn!("Release: no active members found for group {group_id}");
                }
                for id in member_ids {
                    if let Some(agent) = self.find_individual_mut(id) {
                        agent.start_remove_fade(fade_sec);
                    }
                }
                self.mark_group_released(group_id);
            }
            Action::SetRespawnPolicy { group_id, policy } => {
                self.set_group_respawn_policy(group_id, policy);
            }
            Action::SetGroupCrowdingTarget {
                group_id,
                same_group_visible,
                other_group_visible,
            } => {
                self.set_group_crowding_target(group_id, same_group_visible, other_group_visible);
            }
            Action::SetHarmonicityParams { update } => {
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
            Action::EnableTelemetry { group_id, first_k } => {
                self.enable_group_telemetry(group_id, first_k);
            }
            Action::EnablePlv { group_id, window } => {
                self.enable_group_plv(group_id, window);
            }
        }
    }

    fn merge_landscape_update(&mut self, update: LandscapeUpdate) {
        let mut merged = self.pending_update.unwrap_or_default();
        if update.mirror.is_some() {
            merged.mirror = update.mirror;
        }
        if update.roughness_k.is_some() {
            merged.roughness_k = update.roughness_k;
        }
        self.pending_update = Some(merged);
    }

    pub fn take_pending_update(&mut self) -> Option<LandscapeUpdate> {
        self.pending_update.take()
    }

    pub fn kuramoto_order_parameter(&self) -> Option<(f32, usize)> {
        let mut phases = Vec::with_capacity(self.individuals.len());
        for agent in &self.individuals {
            if !agent.is_alive() {
                continue;
            }
            let AnyArticulationCore::Entrain(core) = &agent.articulation.core else {
                continue;
            };
            if core.rhythm_phase.is_finite() {
                phases.push(core.rhythm_phase);
            }
        }
        let r = kuramoto_order_from_phases(&phases)?;
        Some((r, phases.len()))
    }

    /// Assumes `set_current_frame` has been called for the current hop.
    pub fn remove_agent(&mut self, id: u64) {
        self.individuals.retain(|agent| agent.id() != id);
        self.death_observed.remove(&id);
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
            let crowding_active = self.individuals.iter().any(|agent| {
                agent.is_alive() && agent.effective_control.pitch.crowding_strength > 0.0
            });
            let freq_snapshot = if crowding_active {
                // Snapshot alive frequencies once per substep to avoid order-dependent updates.
                let mut snapshot = Vec::with_capacity(self.individuals.len());
                for agent in &self.individuals {
                    if agent.is_alive() {
                        snapshot.push((
                            agent.id(),
                            agent.metadata.group_id,
                            agent.body.base_freq_hz().max(1.0).log2(),
                        ));
                    }
                }
                Some(snapshot)
            } else {
                None
            };
            let group_visibility: BTreeMap<u64, (bool, bool)> = if crowding_active {
                self.groups
                    .iter()
                    .map(|(&group_id, group)| {
                        (
                            group_id,
                            (group.crowding_target_same, group.crowding_target_other),
                        )
                    })
                    .collect()
            } else {
                BTreeMap::new()
            };
            let mut neighbor_pitch_log2 = Vec::new();
            let mut neighbor_salience = Vec::new();
            for agent in self.individuals.iter_mut() {
                if agent.is_alive() {
                    let actor_group_id = agent.metadata.group_id;
                    if let Some(snapshot) = freq_snapshot.as_ref() {
                        neighbor_pitch_log2.clear();
                        neighbor_salience.clear();
                        neighbor_pitch_log2.reserve(snapshot.len());
                        neighbor_salience.reserve(snapshot.len());
                        for &(id, neighbor_group_id, log2) in snapshot {
                            if id != agent.id() {
                                let visible = group_visibility
                                    .get(&neighbor_group_id)
                                    .map(|&(same_visible, other_visible)| {
                                        if neighbor_group_id == actor_group_id {
                                            same_visible
                                        } else {
                                            other_visible
                                        }
                                    })
                                    .unwrap_or(neighbor_group_id == actor_group_id);
                                if visible {
                                    neighbor_pitch_log2.push(log2);
                                    neighbor_salience
                                        .push(Self::pairwise_split_sign(agent.id(), id));
                                }
                            }
                        }
                    }
                    let neighbors = if freq_snapshot.is_some() {
                        neighbor_pitch_log2.as_slice()
                    } else {
                        &[]
                    };
                    let neighbor_weights = if freq_snapshot.is_some() {
                        neighbor_salience.as_slice()
                    } else {
                        &[]
                    };
                    agent.tick_control(
                        dt_step_sec,
                        &rhythms,
                        landscape,
                        neighbors,
                        neighbor_weights,
                        self.global_coupling,
                    );
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

    pub fn cleanup_dead(
        &mut self,
        current_frame: u64,
        dt_sec: f32,
        scenario_finished: bool,
        landscape: &LandscapeFrame,
    ) {
        self.current_frame = current_frame;
        self.respawn_on_new_deaths(scenario_finished, landscape);

        let before_count = self.individuals.len();
        let mut removed_ids = Vec::new();
        let death_records = &mut self.death_records;
        self.individuals.retain(|agent| {
            let keep = agent.should_retain();
            if !keep {
                removed_ids.push(agent.id());
                if let Some(ref acc) = agent.life_accumulator {
                    let plv = match &agent.articulation.core {
                        AnyArticulationCore::Entrain(core) => core.plv(),
                        _ => None,
                    };
                    death_records.push(acc.finalize(
                        agent.id(),
                        agent.metadata.group_id,
                        current_frame,
                        plv,
                    ));
                }
            }
            keep
        });
        let removed_count = before_count - self.individuals.len();
        for id in removed_ids {
            self.death_observed.remove(&id);
        }

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
    }
}

fn kuramoto_order_from_phases(phases: &[f32]) -> Option<f32> {
    if phases.is_empty() {
        return None;
    }
    let mut sum_cos = 0.0f32;
    let mut sum_sin = 0.0f32;
    let mut count = 0usize;
    for &phase in phases {
        if !phase.is_finite() {
            continue;
        }
        sum_cos += phase.cos();
        sum_sin += phase.sin();
        count += 1;
    }
    if count == 0 {
        return None;
    }
    let n = count as f32;
    if n <= 0.0 {
        return None;
    }
    let r = (sum_cos * sum_cos + sum_sin * sum_sin).sqrt() / n;
    Some(r.clamp(0.0, 1.0))
}

fn mix_pred_gate_gain(sync: f32, gain_raw: f32) -> f32 {
    let sync = sync.clamp(0.0, 1.0);
    let gain01 = 0.2 + 0.8 * gain_raw.powf(2.0);
    let gain = 1.0 + (gain01 - 1.0) * sync;
    if gain.is_finite() { gain.max(0.0) } else { 1.0 }
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

fn social_trace_params(hop_tick: Tick) -> (u32, f32) {
    let auto_bin = (hop_tick / 64).max(1);
    let bin_ticks = auto_bin.min(u32::MAX as Tick) as u32;
    (bin_ticks, 0.0)
}

fn social_trace_enabled_from_couplings<I>(couplings: I) -> bool
where
    I: IntoIterator<Item = f32>,
{
    couplings.into_iter().any(|coupling| coupling != 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::landscape::LandscapeFrame;
    use crate::core::log2space::Log2Space;
    use crate::core::timebase::Timebase;
    use crate::life::control::{AgentControl, ControlUpdate, PhonationType, PitchMode};
    use crate::life::individual::{AnyArticulationCore, ArticulationWrapper, DroneCore};
    use crate::life::lifecycle::LifecycleConfig;
    use crate::life::phonation_engine::{OnsetEvent, PhonationCmd, PhonationKick};
    use crate::life::scenario::{ArticulationCoreConfig, RespawnPolicy, SpawnSpec, SpawnStrategy};
    use crate::life::sound::{BodyKind, BodySnapshot};
    use crate::life::world_model::WorldModel;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;

    fn make_dummy_note_spec() -> crate::life::individual::PhonationNoteSpec {
        crate::life::individual::PhonationNoteSpec {
            note_id: 1,
            onset: 0,
            hold_ticks: None,
            freq_hz: 440.0,
            amp: 0.5,
            smoothing_tau_sec: 0.0,
            body: BodySnapshot {
                kind: BodyKind::Sine,
                amp_scale: 1.0,
                brightness: 0.0,
                width: 0.0,
                noise_mix: 0.0,
                ratios: None,
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

    fn spawn_spec_with_freq(freq: f32) -> SpawnSpec {
        let mut control = AgentControl::default();
        control.pitch.freq = freq;
        SpawnSpec {
            control,
            articulation: ArticulationCoreConfig::default(),
        }
    }

    fn decay_spawn_spec_with_freq(freq: f32, half_life_sec: f32) -> SpawnSpec {
        let mut control = AgentControl::default();
        control.pitch.freq = freq;
        SpawnSpec {
            control,
            articulation: ArticulationCoreConfig::Entrain {
                lifecycle: LifecycleConfig::Decay {
                    initial_energy: 1.0,
                    half_life_sec,
                    attack_sec: 0.001,
                },
                rhythm_freq: None,
                rhythm_sensitivity: None,
                rhythm_coupling: crate::life::scenario::RhythmCouplingMode::TemporalOnly,
                rhythm_reward: None,
                breath_gain_init: None,
            },
        }
    }

    fn runtime_landscape() -> LandscapeFrame {
        LandscapeFrame::new(Log2Space::new(55.0, 1760.0, 24))
    }

    fn crowding_order_landscape() -> LandscapeFrame {
        let mut landscape = LandscapeFrame::new(Log2Space::new(220.0, 440.0, 96));
        let center_log2 = 330.0f32.log2();
        let width_cents = 50.0f32;
        for (idx, &bin_log2) in landscape.space.centers_log2.iter().enumerate() {
            let d_cents = (bin_log2 - center_log2).abs() * 1200.0;
            let score = (-(d_cents * d_cents) / (2.0 * width_cents * width_cents)).exp();
            landscape.consonance_field_score[idx] = score;
            landscape.consonance_field_level[idx] = score.clamp(0.0, 1.0);
        }
        landscape.rhythm.theta.phase = 0.1;
        landscape.rhythm.theta.mag = 1.0;
        landscape
    }

    fn step_population(pop: &mut Population, frame: u64, dt_sec: f32, landscape: &LandscapeFrame) {
        let fs = 48_000.0;
        let samples_per_hop = (fs * dt_sec) as usize;
        pop.advance(samples_per_hop, fs, frame, dt_sec, landscape);
        pop.cleanup_dead(frame, dt_sec, false, landscape);
    }

    fn force_dead(pop: &mut Population, id: u64) {
        if let Some(dying) = pop.individuals.iter_mut().find(|agent| agent.id() == id) {
            dying.release_gain = 0.0;
            dying.release_pending = true;
        }
    }

    fn run_single_substep_targets(order_reversed: bool, crowding_strength: f32) -> Vec<(u64, f32)> {
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        pop.set_seed(101);
        let landscape = crowding_order_landscape();
        let mut spec = spawn_spec_with_freq(330.0);
        spec.control.pitch.mode = PitchMode::Free;
        spec.control.pitch.range_oct = 1.5;
        spec.control.pitch.exploration = 0.0;
        spec.control.pitch.persistence = 0.0;
        spec.control.pitch.crowding_strength = crowding_strength;
        spec.control.pitch.crowding_sigma_cents = 20.0;
        pop.apply_action(
            Action::Spawn {
                group_id: 66,
                ids: vec![660, 661, 662],
                spec,
                strategy: Some(SpawnStrategy::Linear {
                    start_freq: 320.0,
                    end_freq: 340.0,
                }),
            },
            &landscape,
            None,
        );
        for agent in pop.individuals.iter_mut() {
            agent.set_theta_phase_state_for_test(0.9, true);
            agent.set_accumulated_time_for_test(agent.integration_window());
        }
        if order_reversed {
            pop.individuals.reverse();
        }
        pop.advance(64, 48_000.0, 0, 1.0, &landscape);
        let mut out: Vec<(u64, f32)> = pop
            .individuals
            .iter()
            .map(|agent| (agent.id(), agent.target_pitch_log2()))
            .collect();
        out.sort_by_key(|(id, _)| *id);
        out
    }

    fn run_cross_group_visibility_trial(other_group_visible: bool) -> f32 {
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        pop.set_seed(303);
        let landscape = crowding_order_landscape();

        let mut mover_spec = spawn_spec_with_freq(330.0);
        mover_spec.control.pitch.mode = PitchMode::Free;
        mover_spec.control.pitch.range_oct = 1.5;
        mover_spec.control.pitch.exploration = 0.0;
        mover_spec.control.pitch.persistence = 0.0;
        mover_spec.control.pitch.crowding_strength = 3.0;
        mover_spec.control.pitch.crowding_sigma_cents = 20.0;
        mover_spec.control.pitch.crowding_sigma_from_roughness = false;

        let mut neighbor_spec = spawn_spec_with_freq(330.0);
        neighbor_spec.control.pitch.mode = PitchMode::Lock;

        pop.apply_action(
            Action::Spawn {
                group_id: 70,
                ids: vec![700],
                spec: mover_spec,
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::Spawn {
                group_id: 71,
                ids: vec![701],
                spec: neighbor_spec,
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetGroupCrowdingTarget {
                group_id: 71,
                same_group_visible: true,
                other_group_visible: other_group_visible,
            },
            &landscape,
            None,
        );

        for agent in pop.individuals.iter_mut() {
            agent.set_theta_phase_state_for_test(0.9, true);
            agent.set_accumulated_time_for_test(agent.integration_window());
        }
        pop.advance(64, 48_000.0, 0, 1.0, &landscape);

        let mover = pop
            .individuals
            .iter()
            .find(|agent| agent.id() == 700)
            .expect("mover");
        (mover.target_pitch_log2() - 330.0f32.log2()).abs()
    }

    #[test]
    fn decide_frequency_uses_consonance_field_level() {
        let space = Log2Space::new(100.0, 400.0, 12);
        let mut landscape = LandscapeFrame::new(space.clone());
        landscape.consonance_field_score.fill(-10.0);
        landscape.consonance_field_level.fill(0.0);

        let idx_high = space.index_of_freq(200.0).expect("idx");
        let idx_raw = space.index_of_freq(300.0).expect("idx");
        landscape.consonance_field_level[idx_high] = 1.0;
        landscape.consonance_field_score[idx_raw] = 10.0;

        let pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let strategy = SpawnStrategy::Consonance {
            root_freq: 100.0,
            min_mul: 1.0,
            max_mul: 4.0,
            min_dist_erb: 0.0,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let freq = pop.decide_frequency(&strategy, &landscape, &mut rng, &[]);
        let picked_idx = space.index_of_freq(freq).expect("picked idx");
        assert_eq!(picked_idx, idx_high);
    }

    #[test]
    fn pred_gate_gain_sync_zero_is_unity() {
        let gain = mix_pred_gate_gain(0.0, 0.3);
        assert_eq!(gain, 1.0);
    }

    #[test]
    fn consonance_density_sampling_uses_density_pmf() {
        let space = Log2Space::new(100.0, 400.0, 24);
        let mut landscape = LandscapeFrame::new(space.clone());
        landscape.consonance_density_mass.fill(0.0);
        let idx_target = space.index_of_freq(220.0).expect("idx target");
        landscape.consonance_density_mass[idx_target] = 1.0;

        let pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let strategy = SpawnStrategy::ConsonanceDensity {
            min_freq: space.fmin,
            max_freq: space.fmax,
            min_dist_erb: 0.0,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);

        for _ in 0..64 {
            let freq = pop.decide_frequency(&strategy, &landscape, &mut rng, &[]);
            let picked_idx = space.index_of_freq(freq).expect("picked idx");
            assert_eq!(picked_idx, idx_target);
        }
    }

    #[test]
    fn consonance_density_range_zero_weights_fallback_is_range_uniform() {
        let space = Log2Space::new(100.0, 400.0, 24);
        let mut landscape = LandscapeFrame::new(space.clone());
        landscape.consonance_density_mass.fill(1.0);

        let idx_min = 6usize;
        let idx_max = 12usize;
        for i in idx_min..=idx_max {
            landscape.consonance_density_mass[i] = 0.0;
        }

        let pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let strategy = SpawnStrategy::ConsonanceDensity {
            min_freq: space.freq_of_index(idx_min),
            max_freq: space.freq_of_index(idx_max),
            min_dist_erb: 0.0,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let mut seen = HashSet::new();

        for _ in 0..64 {
            let freq = pop.decide_frequency(&strategy, &landscape, &mut rng, &[]);
            let picked_idx = space.index_of_freq(freq).expect("picked idx");
            assert!(
                (idx_min..=idx_max).contains(&picked_idx),
                "picked_idx={picked_idx}, expected in [{idx_min},{idx_max}]"
            );
            seen.insert(picked_idx);
        }

        assert!(
            seen.len() > 1,
            "range fallback should not collapse to a single fixed index"
        );
    }

    #[test]
    fn consonance_density_range_all_occupied_fallback_does_not_panic() {
        let space = Log2Space::new(100.0, 400.0, 24);
        let mut landscape = LandscapeFrame::new(space.clone());
        landscape.consonance_density_mass.fill(1.0);

        let idx_min = 8usize;
        let idx_max = 14usize;
        let reserved: Vec<f32> = (idx_min..=idx_max)
            .map(|i| space.freq_of_index(i))
            .collect();

        let pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let strategy = SpawnStrategy::ConsonanceDensity {
            min_freq: space.freq_of_index(idx_min),
            max_freq: space.freq_of_index(idx_max),
            min_dist_erb: 1e-4,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(12);

        for _ in 0..64 {
            let freq = pop.decide_frequency(&strategy, &landscape, &mut rng, &reserved);
            let picked_idx = space.index_of_freq(freq).expect("picked idx");
            assert!(
                (idx_min..=idx_max).contains(&picked_idx),
                "picked_idx={picked_idx}, expected in [{idx_min},{idx_max}]"
            );
        }
    }

    #[test]
    fn consonance_density_avoids_occupied_when_unoccupied_exists() {
        let space = Log2Space::new(100.0, 400.0, 24);
        let mut landscape = LandscapeFrame::new(space.clone());
        landscape.consonance_density_mass.fill(1.0);

        let idx_min = 5usize;
        let idx_max = 11usize;
        let idx_occupied = 8usize;
        let reserved = vec![space.freq_of_index(idx_occupied)];

        let pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let strategy = SpawnStrategy::ConsonanceDensity {
            min_freq: space.freq_of_index(idx_min),
            max_freq: space.freq_of_index(idx_max),
            min_dist_erb: 1e-4,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(13);

        for _ in 0..100 {
            let freq = pop.decide_frequency(&strategy, &landscape, &mut rng, &reserved);
            let picked_idx = space.index_of_freq(freq).expect("picked idx");
            assert!(
                (idx_min..=idx_max).contains(&picked_idx),
                "picked_idx={picked_idx}, expected in [{idx_min},{idx_max}]"
            );
            assert_ne!(
                picked_idx, idx_occupied,
                "occupied index should not be chosen when unoccupied bins exist"
            );
        }
    }

    #[test]
    fn consonance_density_zero_sum_fallback_still_avoids_occupied() {
        let space = Log2Space::new(100.0, 400.0, 24);
        let mut landscape = LandscapeFrame::new(space.clone());
        landscape.consonance_density_mass.fill(1.0);

        let idx_min = 5usize;
        let idx_max = 11usize;
        for i in idx_min..=idx_max {
            landscape.consonance_density_mass[i] = 0.0;
        }
        let idx_occupied = 8usize;
        let reserved = vec![space.freq_of_index(idx_occupied)];

        let pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let strategy = SpawnStrategy::ConsonanceDensity {
            min_freq: space.freq_of_index(idx_min),
            max_freq: space.freq_of_index(idx_max),
            min_dist_erb: 1e-4,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(14);

        for _ in 0..100 {
            let freq = pop.decide_frequency(&strategy, &landscape, &mut rng, &reserved);
            let picked_idx = space.index_of_freq(freq).expect("picked idx");
            assert!(
                (idx_min..=idx_max).contains(&picked_idx),
                "picked_idx={picked_idx}, expected in [{idx_min},{idx_max}]"
            );
            assert_ne!(
                picked_idx, idx_occupied,
                "occupied index should not be chosen in zero-sum fallback"
            );
        }
    }

    #[test]
    fn consonance_density_reversed_range_is_handled_safely() {
        let space = Log2Space::new(100.0, 400.0, 24);
        let mut landscape = LandscapeFrame::new(space.clone());
        landscape.consonance_density_mass.fill(0.0);

        let idx_low = 6usize;
        let idx_high = 12usize;
        let idx_target = 9usize;
        landscape.consonance_density_mass[idx_target] = 1.0;

        let pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let strategy = SpawnStrategy::ConsonanceDensity {
            // Intentionally reversed order to emulate Rhai-side input mistakes.
            min_freq: space.freq_of_index(idx_high),
            max_freq: space.freq_of_index(idx_low),
            min_dist_erb: 0.0,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(15);

        for _ in 0..64 {
            let freq = pop.decide_frequency(&strategy, &landscape, &mut rng, &[]);
            let picked_idx = space.index_of_freq(freq).expect("picked idx");
            assert!(
                (idx_low..=idx_high).contains(&picked_idx),
                "picked_idx={picked_idx}, expected in [{idx_low},{idx_high}]"
            );
            assert_eq!(picked_idx, idx_target);
        }
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
        let couplings = vec![0.0, 1.0];
        assert!(social_trace_enabled_from_couplings(couplings));
    }

    #[test]
    fn update_applies_to_group_members() {
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        pop.apply_action(
            Action::Spawn {
                group_id: 1,
                ids: vec![10, 11],
                spec: spawn_spec_with_freq(220.0),
                strategy: None,
            },
            &landscape,
            None,
        );
        let update = ControlUpdate {
            amp: Some(0.42),
            ..ControlUpdate::default()
        };
        pop.apply_action(
            Action::Update {
                group_id: 1,
                ids: vec![10],
                update,
            },
            &landscape,
            None,
        );
        for agent in &pop.individuals {
            assert!((agent.effective_control.body.amp - 0.42).abs() <= 1e-6);
        }
    }

    #[test]
    fn neighbor_snapshot_order_independent_without_crowding() {
        let forward = run_single_substep_targets(false, 0.0);
        let reversed = run_single_substep_targets(true, 0.0);
        assert_eq!(forward.len(), reversed.len());
        for ((id_a, pitch_a), (id_b, pitch_b)) in forward.iter().zip(reversed.iter()) {
            assert_eq!(*id_a, *id_b);
            assert!((pitch_a - pitch_b).abs() <= 1e-6);
        }
    }

    #[test]
    fn neighbor_snapshot_order_independent_with_crowding() {
        let forward = run_single_substep_targets(false, 2.0);
        let reversed = run_single_substep_targets(true, 2.0);
        assert_eq!(forward.len(), reversed.len());
        for ((id_a, pitch_a), (id_b, pitch_b)) in forward.iter().zip(reversed.iter()) {
            assert_eq!(*id_a, *id_b);
            assert!((pitch_a - pitch_b).abs() <= 1e-6);
        }
    }

    #[test]
    fn cross_group_crowding_follows_target_visibility_policy() {
        let hidden = run_cross_group_visibility_trial(false);
        let visible = run_cross_group_visibility_trial(true);
        assert!(
            visible > hidden + 1e-6,
            "cross-group crowding should only affect behavior when target group allows visibility"
        );
    }

    #[test]
    fn pairwise_split_sign_is_antisymmetric() {
        let ab = Population::pairwise_split_sign(10, 42);
        let ba = Population::pairwise_split_sign(42, 10);
        assert!(ab.abs() > 0.0);
        assert!((ab + ba).abs() <= 1e-6);
    }

    #[test]
    fn release_marks_group_members() {
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        pop.apply_action(
            Action::Spawn {
                group_id: 1,
                ids: vec![21, 22],
                spec: spawn_spec_with_freq(220.0),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::Release {
                group_id: 1,
                ids: vec![21],
                fade_sec: 0.05,
            },
            &landscape,
            None,
        );
        let released: Vec<u64> = pop
            .individuals
            .iter()
            .filter(|agent| agent.remove_pending)
            .map(|agent| agent.id())
            .collect();
        assert_eq!(released.len(), 2);
        assert!(released.contains(&21));
        assert!(released.contains(&22));
    }

    #[test]
    fn spawn_without_strategy_keeps_spec_frequency() {
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        pop.apply_action(
            Action::Spawn {
                group_id: 6,
                ids: vec![60],
                spec: spawn_spec_with_freq(275.0),
                strategy: None,
            },
            &landscape,
            None,
        );
        let spawned = pop.individuals.first().expect("spawned");
        assert!((spawned.body.base_freq_hz() - 275.0).abs() <= 1e-6);
    }

    #[test]
    fn respawn_none_keeps_current_behavior() {
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        pop.set_seed(7);
        let landscape = runtime_landscape();
        pop.apply_action(
            Action::Spawn {
                group_id: 7,
                ids: vec![1],
                spec: decay_spawn_spec_with_freq(220.0, 0.02),
                strategy: None,
            },
            &landscape,
            None,
        );

        for frame in 0..300 {
            step_population(&mut pop, frame, 0.01, &landscape);
            if pop.individuals.is_empty() {
                break;
            }
        }

        assert!(pop.individuals.is_empty());
    }

    #[test]
    fn respawn_random_maintains_population() {
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        pop.set_seed(11);
        let landscape = runtime_landscape();
        pop.apply_action(
            Action::Spawn {
                group_id: 8,
                ids: vec![10],
                spec: decay_spawn_spec_with_freq(220.0, 0.02),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                group_id: 8,
                policy: RespawnPolicy::Random,
            },
            &landscape,
            None,
        );

        let mut saw_respawned = false;
        for frame in 0..300 {
            step_population(&mut pop, frame, 0.01, &landscape);
            if pop.individuals.iter().any(|a| a.id() != 10) {
                saw_respawned = true;
                break;
            }
        }

        assert!(saw_respawned, "expected at least one respawned member");
        assert!(
            !pop.individuals.is_empty(),
            "population should not collapse with random respawn"
        );
    }

    #[test]
    fn respawn_hereditary_maintains_population() {
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        pop.set_seed(13);
        let landscape = runtime_landscape();
        pop.apply_action(
            Action::Spawn {
                group_id: 9,
                ids: vec![20],
                spec: decay_spawn_spec_with_freq(330.0, 0.02),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                group_id: 9,
                policy: RespawnPolicy::Hereditary { sigma_oct: 0.01 },
            },
            &landscape,
            None,
        );

        let mut saw_respawned = false;
        for frame in 0..300 {
            step_population(&mut pop, frame, 0.01, &landscape);
            if pop.individuals.iter().any(|a| a.id() != 20) {
                saw_respawned = true;
                break;
            }
        }

        assert!(saw_respawned, "expected at least one respawned member");
        assert!(
            !pop.individuals.is_empty(),
            "population should not collapse with hereditary respawn"
        );
    }

    #[test]
    fn hereditary_respawn_without_strategy_uses_parent_pitch_regression() {
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        pop.set_seed(31);
        let landscape = runtime_landscape();

        let mut spec = decay_spawn_spec_with_freq(220.0, 0.02);
        spec.control.pitch.mode = PitchMode::Lock;
        pop.apply_action(
            Action::Spawn {
                group_id: 90,
                ids: vec![900, 901],
                spec,
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                group_id: 90,
                policy: RespawnPolicy::Hereditary { sigma_oct: 0.002 },
            },
            &landscape,
            None,
        );

        let parent_target_hz: f32 = 440.0;
        if let Some(parent) = pop.individuals.iter_mut().find(|agent| agent.id() == 901) {
            parent.force_set_pitch_log2(parent_target_hz.log2());
        }
        force_dead(&mut pop, 900);
        pop.cleanup_dead(0, 0.01, false, &landscape);

        let child = pop
            .individuals
            .iter()
            .find(|agent| agent.id() != 901)
            .expect("child exists");
        let child_log2 = child.body.base_freq_hz().log2();
        let parent_log2 = parent_target_hz.log2();
        let spec_log2 = 220.0f32.log2();
        let to_parent = (child_log2 - parent_log2).abs();
        let to_spec = (child_log2 - spec_log2).abs();
        assert!(to_parent < 0.05, "child should be close to live parent");
        assert!(
            to_parent < to_spec,
            "regression: child should follow parent, not template frequency"
        );
    }

    #[test]
    fn release_reaches_respawned_members() {
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        pop.set_seed(17);
        let landscape = runtime_landscape();
        pop.apply_action(
            Action::Spawn {
                group_id: 10,
                ids: vec![30],
                spec: decay_spawn_spec_with_freq(220.0, 0.02),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                group_id: 10,
                policy: RespawnPolicy::Random,
            },
            &landscape,
            None,
        );

        let mut respawned_id = None;
        for frame in 0..300 {
            step_population(&mut pop, frame, 0.01, &landscape);
            if let Some(id) = pop
                .individuals
                .iter()
                .find(|agent| agent.metadata.group_id == 10 && agent.id() != 30)
                .map(|agent| agent.id())
            {
                respawned_id = Some(id);
                break;
            }
        }
        let respawned_id = respawned_id.expect("respawned member should exist");

        pop.apply_action(
            Action::Release {
                group_id: 10,
                ids: vec![30],
                fade_sec: 0.05,
            },
            &landscape,
            None,
        );

        let respawned = pop
            .individuals
            .iter()
            .find(|agent| agent.id() == respawned_id)
            .expect("respawned member");
        assert!(respawned.remove_pending);
    }

    #[test]
    fn live_update_reaches_respawned_members() {
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        pop.set_seed(23);
        let landscape = runtime_landscape();
        pop.apply_action(
            Action::Spawn {
                group_id: 11,
                ids: vec![40],
                spec: decay_spawn_spec_with_freq(220.0, 0.02),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                group_id: 11,
                policy: RespawnPolicy::Random,
            },
            &landscape,
            None,
        );

        let mut respawned_id = None;
        for frame in 0..300 {
            step_population(&mut pop, frame, 0.01, &landscape);
            if let Some(id) = pop
                .individuals
                .iter()
                .find(|agent| agent.metadata.group_id == 11 && agent.id() != 40)
                .map(|agent| agent.id())
            {
                respawned_id = Some(id);
                break;
            }
        }
        let respawned_id = respawned_id.expect("respawned member should exist");

        pop.apply_action(
            Action::Update {
                group_id: 11,
                ids: vec![40],
                update: ControlUpdate {
                    amp: Some(0.17),
                    ..ControlUpdate::default()
                },
            },
            &landscape,
            None,
        );

        let respawned = pop
            .individuals
            .iter()
            .find(|agent| agent.id() == respawned_id)
            .expect("respawned member");
        assert!((respawned.effective_control.body.amp - 0.17).abs() <= 1e-6);
    }

    #[test]
    fn live_landscape_weight_update_is_inherited_by_respawn() {
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        pop.set_seed(41);
        let landscape = runtime_landscape();
        pop.apply_action(
            Action::Spawn {
                group_id: 91,
                ids: vec![910, 911],
                spec: decay_spawn_spec_with_freq(220.0, 0.02),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                group_id: 91,
                policy: RespawnPolicy::Random,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::Update {
                group_id: 91,
                ids: vec![910],
                update: ControlUpdate {
                    landscape_weight: Some(0.73),
                    ..ControlUpdate::default()
                },
            },
            &landscape,
            None,
        );

        for member in pop
            .individuals
            .iter()
            .filter(|agent| agent.metadata.group_id == 91)
        {
            assert!((member.effective_control.pitch.landscape_weight - 0.73).abs() <= 1e-6);
        }

        force_dead(&mut pop, 910);
        pop.cleanup_dead(0, 0.01, false, &landscape);

        let child = pop
            .individuals
            .iter()
            .find(|agent| agent.id() != 911)
            .expect("child exists");
        assert!((child.effective_control.pitch.landscape_weight - 0.73).abs() <= 1e-6);
    }

    #[test]
    fn release_disables_future_respawns() {
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        pop.set_seed(47);
        let landscape = runtime_landscape();
        pop.apply_action(
            Action::Spawn {
                group_id: 92,
                ids: vec![920],
                spec: decay_spawn_spec_with_freq(220.0, 0.02),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                group_id: 92,
                policy: RespawnPolicy::Random,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::Release {
                group_id: 92,
                ids: vec![920],
                fade_sec: 0.01,
            },
            &landscape,
            None,
        );

        let mut saw_new_id = false;
        for frame in 0..400 {
            step_population(&mut pop, frame, 0.01, &landscape);
            if pop.individuals.iter().any(|agent| agent.id() != 920) {
                saw_new_id = true;
                break;
            }
            if pop.individuals.is_empty() {
                break;
            }
        }

        assert!(!saw_new_id, "release must disable future respawns");
        assert!(
            pop.individuals.is_empty(),
            "released group should drain without repopulation"
        );
    }

    #[test]
    fn hereditary_respawn_child_stays_near_parent() {
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        pop.set_seed(29);
        let landscape = runtime_landscape();

        let mut spec = spawn_spec_with_freq(220.0);
        spec.control.pitch.mode = PitchMode::Lock;
        pop.apply_action(
            Action::Spawn {
                group_id: 12,
                ids: vec![100, 101],
                spec,
                strategy: Some(SpawnStrategy::Linear {
                    start_freq: 220.0,
                    end_freq: 330.0,
                }),
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                group_id: 12,
                policy: RespawnPolicy::Hereditary { sigma_oct: 0.005 },
            },
            &landscape,
            None,
        );

        let parent_freq = pop
            .individuals
            .iter()
            .find(|agent| agent.id() == 101)
            .map(|agent| agent.body.base_freq_hz())
            .expect("parent exists");

        if let Some(dying) = pop.individuals.iter_mut().find(|agent| agent.id() == 100) {
            dying.release_gain = 0.0;
            dying.release_pending = true;
        }
        pop.cleanup_dead(0, 0.01, false, &landscape);

        let child = pop
            .individuals
            .iter()
            .find(|agent| agent.id() != 101)
            .expect("child exists");
        let delta_oct = (child.body.base_freq_hz().log2() - parent_freq.log2()).abs();
        assert!(
            delta_oct < 0.05,
            "child should stay near parent in log2 space"
        );
    }

    #[test]
    fn spawn_strategy_respects_free_pitch_mode() {
        let mut pop = Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = LandscapeFrame::default();
        let mut spec = spawn_spec_with_freq(110.0);
        spec.control.pitch.mode = PitchMode::Free;
        pop.apply_action(
            Action::Spawn {
                group_id: 1,
                ids: vec![1],
                spec,
                strategy: Some(SpawnStrategy::Linear {
                    start_freq: 220.0,
                    end_freq: 220.0,
                }),
            },
            &landscape,
            None,
        );
        let agent = pop.individuals.first().expect("spawned");
        assert_eq!(agent.effective_control.pitch.mode, PitchMode::Free);
        assert!((agent.effective_control.pitch.freq - 220.0).abs() <= 1e-6);
    }

    #[test]
    fn kuramoto_order_parameter_is_bounded() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(4242);
        let phases: Vec<f32> = (0..256)
            .map(|_| rng.random_range(0.0..std::f32::consts::TAU))
            .collect();
        let r = kuramoto_order_from_phases(&phases).expect("non-empty");
        assert!((0.0..=1.0).contains(&r));
    }

    #[test]
    fn kuramoto_order_parameter_high_for_aligned_low_for_random() {
        let aligned = vec![0.0f32; 128];
        let aligned_r = kuramoto_order_from_phases(&aligned).expect("non-empty");
        assert!(aligned_r > 0.99, "aligned phase set should have high order");

        let mut rng = rand::rngs::StdRng::seed_from_u64(99);
        let random: Vec<f32> = (0..128)
            .map(|_| rng.random_range(0.0..std::f32::consts::TAU))
            .collect();
        let random_r = kuramoto_order_from_phases(&random).expect("non-empty");
        assert!(
            random_r < 0.35,
            "random phase set should have low order (got {random_r})"
        );
    }

    #[test]
    fn collect_phonation_batches_into_clears_unused_batch() {
        let time = Timebase {
            fs: 48_000.0,
            hop: 64,
        };
        let space = Log2Space::new(55.0, 880.0, 12);
        let landscape = LandscapeFrame::new(space.clone());
        let mut world = WorldModel::new(time, space);
        let mut pop = Population::new(time);
        let mut silent_spec = spawn_spec_with_freq(440.0);
        silent_spec.control.phonation.r#type = PhonationType::None;
        pop.apply_action(
            Action::Spawn {
                group_id: 2,
                ids: vec![77],
                spec: silent_spec,
                strategy: None,
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

        let used = pop.collect_phonation_batches_into(&mut world, &landscape, 0, &mut batches);
        assert_eq!(used, 0);
        assert!(batches[0].cmds.is_empty());
        assert!(batches[0].notes.is_empty());
        assert!(batches[0].onsets.is_empty());
    }
}
