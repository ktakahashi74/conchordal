use super::scenario::{
    Action, ControlUpdateMode, RespawnPeakBiasConfig, RespawnPolicy, SpawnStrategy, VoiceConfig,
};
use super::telemetry::LifeRecord;
use super::voice::{AnyArticulationCore, PhonationBatch, SoundBody, Voice, VoiceMetadata};
use crate::core::landscape::{Landscape, LandscapeFrame, LandscapeUpdate};
use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::life::control::{MAX_FREQ_HZ, MIN_FREQ_HZ};
use crate::life::social_density::SocialDensityTrace;
use crate::life::world_model::WorldModel;
use rand::{Rng, SeedableRng, distr::Distribution, distr::weighted::WeightedIndex, rngs::SmallRng};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use tracing::{debug, info, warn};

const DEFAULT_REPORT_FIRST_K: u32 = 10;
const DEFAULT_REPORT_PLV_WINDOW: usize = 200;

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

#[derive(Default)]
struct PredGateAccum {
    count: u32,
    raw_min: f32,
    raw_max: f32,
    raw_sum: f32,
    mixed_min: f32,
    mixed_max: f32,
    mixed_sum: f32,
    sync_sum: f32,
}

impl PredGateAccum {
    fn push(&mut self, raw: f32, mixed: f32, sync: f32) {
        if self.count == 0 {
            self.raw_min = raw;
            self.raw_max = raw;
            self.mixed_min = mixed;
            self.mixed_max = mixed;
        } else {
            self.raw_min = self.raw_min.min(raw);
            self.raw_max = self.raw_max.max(raw);
            self.mixed_min = self.mixed_min.min(mixed);
            self.mixed_max = self.mixed_max.max(mixed);
        }
        self.raw_sum += raw;
        self.mixed_sum += mixed;
        self.sync_sum += sync;
        self.count += 1;
    }

    fn finalize(&self) -> Option<PredGateStats> {
        if self.count == 0 {
            return None;
        }
        let inv = 1.0 / self.count as f32;
        Some(PredGateStats {
            raw_min: self.raw_min,
            raw_max: self.raw_max,
            raw_mean: self.raw_sum * inv,
            mixed_min: self.mixed_min,
            mixed_max: self.mixed_max,
            mixed_mean: self.mixed_sum * inv,
            sync_mean: self.sync_sum * inv,
            count: self.count,
        })
    }
}

pub struct Population {
    pub voices: Vec<Voice>,
    current_frame: u64,
    pub abort_requested: bool,
    pub global_coupling: f32,
    shutdown_gain: f32,
    pending_update: Option<LandscapeUpdate>,
    time: Timebase,
    seed: u64,
    spawn_counter: u64,
    social_trace: Option<SocialDensityTrace>,
    groups: BTreeMap<u64, RuntimeGroupState>,
    death_observed: HashSet<u64>,
    next_runtime_id: u64,
    control_update_mode: ControlUpdateMode,
    last_pred_gate_stats: Option<PredGateStats>,
    last_gate_boundary_in_hop: Option<bool>,
    last_phonation_onsets_in_hop: Option<u32>,
    death_records: Vec<LifeRecord>,
    auto_observe: Option<ObservationConfig>,
    runtime_events: Vec<RuntimeEvent>,
    advance_scratch: AdvanceScratch,
}

#[derive(Debug, Clone)]
struct RuntimeGroupState {
    template: VoiceConfig,
    strategy: Option<SpawnStrategy>,
    respawn_policy: RespawnPolicy,
    respawn_settle_strategy: Option<SpawnStrategy>,
    respawn_capacity: usize,
    respawn_min_c_level: Option<f32>,
    respawn_background_death_rate_per_sec: f32,
    crowding_target_same: bool,
    crowding_target_other: bool,
    released: bool,
    next_member_idx: usize,
    spawn_count_hint: usize,
}

#[derive(Debug, Clone, Copy)]
struct SpawnParams {
    id: u64,
    group_id: u64,
    member_idx: usize,
    resolved_freq_hz: f32,
    parent_id: Option<u64>,
    parent_generation: Option<u32>,
    reason: SpawnReason,
}

#[derive(Debug, Clone, Copy)]
struct ObservationConfig {
    first_k: u32,
    plv_window: usize,
}

#[derive(Clone, Copy)]
struct ParentCandidate {
    id: u64,
    freq_hz: f32,
    energy: f32,
    generation: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpawnReason {
    Initial,
    Respawn,
}

#[derive(Debug, Clone)]
pub struct RuntimeEvent {
    pub time_sec: f32,
    pub group_id: u64,
    pub voice_id: u64,
    pub member_idx: usize,
    pub freq_hz: f32,
    pub parent_id: Option<u64>,
    pub generation: u32,
    pub reason: SpawnReason,
}

#[derive(Default)]
struct AdvanceScratch {
    freq_snapshot: Vec<(u64, u64, f32)>,
    group_visibility: HashMap<u64, (bool, bool)>,
    neighbor_pitch_log2: Vec<f32>,
    neighbor_salience: Vec<f32>,
    commit_queue: Vec<CommitQueueEntry>,
}

#[derive(Clone, Copy, Debug)]
struct CommitQueueEntry {
    voice_idx: usize,
}

mod respawn;
mod social;

use respawn::*;
use social::*;

impl Population {
    const CONTROL_STEP_SAMPLES: usize = 64;
    /// Returns true if `freq_hz` is within `min_dist_erb` (ERB scale) of any existing voice's base
    /// frequency.
    pub fn is_range_occupied(&self, freq_hz: f32, min_dist_erb: f32) -> bool {
        self.is_range_occupied_with(freq_hz, min_dist_erb, &[])
    }

    fn is_range_occupied_with(&self, freq_hz: f32, min_dist_erb: f32, reserved: &[f32]) -> bool {
        if !freq_hz.is_finite() || min_dist_erb <= 0.0 {
            return false;
        }
        let target_erb = crate::core::erb::hz_to_erb(freq_hz.max(1e-6));
        for voice in &self.voices {
            let base_hz = voice.body.base_freq_hz();
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
            voices: Vec::new(),
            current_frame: 0,
            abort_requested: false,
            global_coupling: 1.0,
            shutdown_gain: 1.0,
            pending_update: None,
            time,
            seed: rand::random::<u64>(),
            spawn_counter: 0,
            social_trace: None,
            groups: BTreeMap::new(),
            death_observed: HashSet::new(),
            next_runtime_id: 1,
            control_update_mode: ControlUpdateMode::SnapshotPhased,
            last_pred_gate_stats: None,
            last_gate_boundary_in_hop: None,
            last_phonation_onsets_in_hop: None,
            death_records: Vec::new(),
            auto_observe: None,
            runtime_events: Vec::new(),
            advance_scratch: AdvanceScratch::default(),
        }
    }

    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
    }

    pub fn set_control_update_mode(&mut self, mode: ControlUpdateMode) {
        self.control_update_mode = mode;
    }

    pub fn enable_auto_observe(&mut self) {
        self.auto_observe = Some(ObservationConfig {
            first_k: DEFAULT_REPORT_FIRST_K,
            plv_window: DEFAULT_REPORT_PLV_WINDOW,
        });
    }

    pub fn drain_runtime_events(&mut self) -> Vec<RuntimeEvent> {
        std::mem::take(&mut self.runtime_events)
    }

    pub fn take_death_records(&mut self) -> Vec<LifeRecord> {
        std::mem::take(&mut self.death_records)
    }

    fn current_time_sec(&self) -> f32 {
        let tick = self.time.frame_start_tick(self.current_frame);
        self.time.tick_to_sec(tick)
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
            if self.voices.iter().all(|v| v.id() != id) {
                return id;
            }
        }
    }

    fn normal_sample<R: Rng + ?Sized>(rng: &mut R) -> f32 {
        let u1 = (1.0 - rng.random::<f32>()).max(1e-7);
        let u2 = rng.random::<f32>();
        let mag = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        mag * theta.cos()
    }

    fn background_turnover_seed(&self, substep_idx: usize) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.seed.hash(&mut hasher);
        self.current_frame.hash(&mut hasher);
        substep_idx.hash(&mut hasher);
        hasher.finish() ^ 0xBADC_0FFE_E0DD_F00D
    }

    pub fn add_voice(&mut self, voice: Voice) {
        let id = voice.id();
        if self.voices.iter().any(|a| a.id() == id) {
            warn!("AddVoice: id collision for {id}");
            return;
        }
        self.track_runtime_id(id);
        self.voices.push(voice);
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
        let mut pred_acc = PredGateAccum::default();
        let mut phonation_onsets_in_hop = 0u32;
        let mut used = 0usize;
        let social_trace = self.social_trace.as_ref();
        for voice in &mut self.voices {
            let social_coupling = voice.social_coupling;
            if used == out.len() {
                out.push(PhonationBatch::default());
            }
            let batch = &mut out[used];
            let consonance = landscape.evaluate_pitch_level(voice.body.base_freq_hz());
            let extra_gate_gain = match pred_scan.as_ref() {
                Some(scan) => {
                    let gain_raw = world
                        .sample_scan_field_level(scan, voice.body.base_freq_hz())
                        .clamp(0.0, 1.0);
                    let sync = match &voice.effective_control.phonation.spec.when {
                        crate::life::scenario::WhenSpec::Pulse { sync, .. } => sync.clamp(0.0, 1.0),
                        _ => 0.0,
                    };
                    let mixed = mix_pred_gate_gain(sync, gain_raw);
                    let mixed = if mixed.is_finite() { mixed } else { 1.0 };
                    pred_acc.push(gain_raw, mixed, sync);
                    mixed
                }
                None => 1.0,
            };
            voice.tick_phonation_into(
                &tb,
                now,
                &landscape.rhythm,
                social_trace,
                social_coupling,
                extra_gate_gain,
                consonance,
                batch,
            );
            phonation_onsets_in_hop = phonation_onsets_in_hop
                .saturating_add(batch.onsets.len().min(u32::MAX as usize) as u32);
            let has_output =
                !(batch.cmds.is_empty() && batch.tones.is_empty() && batch.onsets.is_empty());
            if has_output {
                used += 1;
            }
        }
        let active_batches = &out[..used];
        let social_enabled =
            social_trace_enabled_from_couplings(self.voices.iter().map(|a| a.social_coupling));
        if social_enabled {
            let (bin_ticks, smooth) = social_trace_params(hop_tick);
            self.social_trace = Some(build_social_trace_from_batches(
                active_batches,
                frame_end,
                hop_tick,
                bin_ticks,
                smooth,
                self.voices.len(),
            ));
        } else {
            self.social_trace = None;
        }
        self.last_gate_boundary_in_hop = Some(gate_boundary_in_hop);
        self.last_phonation_onsets_in_hop = Some(phonation_onsets_in_hop);
        self.last_pred_gate_stats = pred_acc.finalize();
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
                let mut best_free = None;
                let mut best_any = (idx_min, f32::MIN);
                for i in idx_min..=idx_max {
                    let c_val = landscape
                        .consonance_field_level
                        .get(i)
                        .copied()
                        .unwrap_or(f32::MIN);
                    if c_val > best_any.1 {
                        best_any = (i, c_val);
                    }
                    let f = space.freq_of_index(i);
                    if !self.is_range_occupied_with(f, min_dist_erb, reserved)
                        && c_val > best_free.map_or(f32::MIN, |(_, v)| v)
                    {
                        best_free = Some((i, c_val));
                    }
                }
                best_free.unwrap_or(best_any).0
            }
            SpawnStrategy::ConsonanceDensity { .. } => {
                let range_len = idx_max - idx_min + 1;
                let mut weights = Vec::with_capacity(range_len);
                let mut has_unoccupied = false;
                let mut sum = 0.0f32;
                for i in idx_min..=idx_max {
                    let f = space.freq_of_index(i);
                    let occupied = self.is_range_occupied_with(f, min_dist_erb, reserved);
                    if !occupied {
                        has_unoccupied = true;
                    }
                    let raw = landscape
                        .consonance_density_mass
                        .get(i)
                        .copied()
                        .unwrap_or(0.0);
                    let w = if occupied { 0.0 } else { raw.max(0.0) };
                    let w = if w.is_finite() { w } else { 0.0 };
                    weights.push((w, occupied));
                    sum += w;
                }
                // Fallback: if density mass sums to zero, use uniform over unoccupied bins.
                if !(sum > 0.0 && sum.is_finite()) {
                    for (w, occupied) in &mut weights {
                        *w = if *occupied && has_unoccupied {
                            0.0
                        } else {
                            1.0
                        };
                    }
                }
                let ws: Vec<f32> = weights.iter().map(|(w, _)| *w).collect();
                if let Ok(dist) = WeightedIndex::new(&ws) {
                    idx_min + dist.sample(rng)
                } else {
                    idx_min + rng.random_range(0..range_len)
                }
            }
            SpawnStrategy::RejectTargets {
                base,
                anchor_hz,
                targets_st,
                exclusion_st,
                max_tries,
            } => {
                let tries = (*max_tries).max(1);
                let mut last = self.decide_frequency(base, landscape, rng, reserved);
                if !is_rejected_target(last, *anchor_hz, targets_st, *exclusion_st) {
                    return last;
                }
                for _ in 1..tries {
                    let candidate = self.decide_frequency(base, landscape, rng, reserved);
                    last = candidate;
                    if !is_rejected_target(candidate, *anchor_hz, targets_st, *exclusion_st) {
                        return candidate;
                    }
                }
                return last;
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
            SpawnStrategy::RejectTargets {
                base,
                anchor_hz,
                targets_st,
                exclusion_st,
                max_tries,
            } => {
                let tries = (*max_tries).max(1);
                let mut last = self.resolve_strategy_frequency(
                    base,
                    landscape,
                    rng,
                    reserved,
                    member_idx,
                    member_count,
                );
                if !is_rejected_target(last, *anchor_hz, targets_st, *exclusion_st) {
                    return last;
                }
                for _ in 1..tries {
                    let candidate = self.resolve_strategy_frequency(
                        base,
                        landscape,
                        rng,
                        reserved,
                        member_idx,
                        member_count,
                    );
                    last = candidate;
                    if !is_rejected_target(candidate, *anchor_hz, targets_st, *exclusion_st) {
                        return candidate;
                    }
                }
                last
            }
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

    fn spawn_one(&mut self, params: SpawnParams, spec: &VoiceConfig, landscape: &LandscapeFrame) {
        let SpawnParams {
            id,
            group_id,
            member_idx,
            resolved_freq_hz,
            parent_id,
            parent_generation,
            reason,
        } = params;
        if self.voices.iter().any(|v| v.id() == id) {
            warn!("Spawn: id collision for {id} in group {group_id}");
            return;
        }

        let generation = parent_generation.map_or(0, |g| g + 1);
        let mut control = spec.control.clone();
        control.pitch.freq = resolved_freq_hz.clamp(MIN_FREQ_HZ, MAX_FREQ_HZ);
        let metadata = VoiceMetadata {
            group_id,
            member_idx,
            generation,
            parent_id,
        };
        let cfg = VoiceConfig {
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
        if let Some(observe) = self.auto_observe {
            spawned.life_accumulator = Some(super::telemetry::LifeAccumulator::new(
                self.current_frame,
                observe.first_k,
                landscape.evaluate_pitch_level(resolved_freq_hz),
            ));
            if let AnyArticulationCore::Entrain(ref mut core) = spawned.articulation.core {
                core.enable_plv(observe.plv_window);
            }
        }
        self.voices.push(spawned);
        self.track_runtime_id(id);
        if self.auto_observe.is_some() {
            self.runtime_events.push(RuntimeEvent {
                time_sec: self.current_time_sec(),
                group_id,
                voice_id: id,
                member_idx,
                freq_hz: resolved_freq_hz,
                parent_id,
                generation,
                reason,
            });
        }
    }

    fn ensure_group_state(
        &mut self,
        group_id: u64,
        spec: VoiceConfig,
        strategy: Option<SpawnStrategy>,
        member_count: usize,
    ) {
        let current_members = self
            .voices
            .iter()
            .filter(|v| v.metadata.group_id == group_id)
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
                respawn_settle_strategy: None,
                respawn_capacity: 1,
                respawn_min_c_level: None,
                respawn_background_death_rate_per_sec: 0.0,
                crowding_target_same: true,
                crowding_target_other: false,
                released: false,
                next_member_idx: current_members.max(member_count),
                spawn_count_hint: member_count.max(1),
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

    fn set_group_respawn_policy(
        &mut self,
        group_id: u64,
        policy: RespawnPolicy,
        settle_strategy: Option<SpawnStrategy>,
        capacity: usize,
        min_c_level: Option<f32>,
        background_death_rate_per_sec: f32,
    ) {
        if let Some(group) = self.groups.get_mut(&group_id) {
            group.respawn_policy = policy;
            group.respawn_settle_strategy = settle_strategy;
            group.respawn_capacity = capacity.max(1);
            group.respawn_min_c_level = min_c_level.map(|value| value.clamp(0.0, 1.0));
            group.respawn_background_death_rate_per_sec = background_death_rate_per_sec.max(0.0);
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

    fn peak_biased_respawn_candidate<R: Rng + ?Sized>(
        &self,
        group: &RuntimeGroupState,
        selected_parent: Option<ParentCandidate>,
        landscape: &LandscapeFrame,
        rng: &mut R,
        member_idx: usize,
        config: RespawnPeakBiasConfig,
    ) -> Option<(f32, Option<u64>, Option<u32>)> {
        let (min_hz, max_hz) = group
            .strategy
            .as_ref()
            .map(SpawnStrategy::freq_range_hz)
            .unwrap_or_else(|| landscape.freq_bounds());
        let lo = min_hz.clamp(MIN_FREQ_HZ, MAX_FREQ_HZ);
        let hi = max_hz.clamp(lo, MAX_FREQ_HZ);
        let candidate_count = group.respawn_capacity.max(1);
        let candidate_bins = peak_bias_candidate_bins(landscape, lo, hi, candidate_count);

        let fallback_freq = selected_parent
            .map(|parent| parent.freq_hz.clamp(lo, hi))
            .unwrap_or_else(|| self.random_respawn_frequency(group, landscape, rng, member_idx));

        let chosen_freq = if candidate_bins.is_empty() {
            fallback_freq
        } else {
            let scene_exp = if config.scene_score_exponent.is_finite() {
                config.scene_score_exponent.max(0.0)
            } else {
                0.35
            };
            let parent_freq_hz = selected_parent
                .map(|parent| parent.freq_hz.max(MIN_FREQ_HZ))
                .filter(|freq_hz| freq_hz.is_finite() && *freq_hz > 0.0);
            let mut scene_weights = Vec::with_capacity(candidate_bins.len());
            let mut final_weights = Vec::with_capacity(candidate_bins.len());
            for &bin_idx in &candidate_bins {
                let center_hz = landscape.space.centers_hz[bin_idx].clamp(lo, hi);
                let mut scene_weight = landscape.consonance_field_score[bin_idx].max(0.0);
                if scene_exp > 0.0 {
                    scene_weight = scene_weight.powf(scene_exp);
                }
                scene_weights.push(scene_weight);

                let mut final_weight = scene_weight;
                if let Some(parent_freq_hz) = parent_freq_hz {
                    let delta_st = 12.0 * (center_hz / parent_freq_hz).log2();
                    final_weight *= peak_bias_gaussian_weight(delta_st, config.proposal_sigma_st);
                    if peak_bias_same_band(parent_freq_hz, center_hz, config.same_band_window_cents)
                    {
                        final_weight *= config.same_band_discount.clamp(0.0, 1.0);
                    }
                    if peak_bias_parent_octave(
                        parent_freq_hz,
                        center_hz,
                        config.octave_window_cents,
                    ) {
                        final_weight *= config.octave_discount.clamp(0.0, 1.0);
                    }
                }
                final_weights.push(final_weight.max(0.0));
            }

            if final_weights
                .iter()
                .all(|weight| !weight.is_finite() || *weight <= 0.0)
            {
                final_weights.clone_from(&scene_weights);
            }

            let chosen_idx = if let Ok(dist) = WeightedIndex::new(&final_weights) {
                dist.sample(rng)
            } else {
                scene_weights
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            };
            let center_hz = landscape.space.centers_hz[candidate_bins[chosen_idx]].clamp(lo, hi);
            peak_bias_local_search_frequency(landscape, center_hz, lo, hi, config)
        };

        if let Some(min_c_level) = group.respawn_min_c_level
            && landscape.evaluate_pitch_level(chosen_freq) < min_c_level
        {
            return None;
        }

        let (parent_id, parent_gen) = match selected_parent {
            Some(parent) => (Some(parent.id), Some(parent.generation)),
            None => (None, None),
        };
        Some((chosen_freq, parent_id, parent_gen))
    }

    fn pick_respawn_candidate<R: Rng + ?Sized>(
        &self,
        group_id: u64,
        group: &RuntimeGroupState,
        alive_by_group: &BTreeMap<u64, Vec<ParentCandidate>>,
        landscape: &LandscapeFrame,
        rng: &mut R,
        member_idx: usize,
    ) -> Option<(f32, Option<u64>, Option<u32>)> {
        if let RespawnPolicy::PeakBiased { config } = group.respawn_policy {
            let pool = alive_by_group
                .get(&group_id)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let selected_parent = if pool.is_empty() {
                None
            } else {
                Some(pool[weighted_parent_select(pool, rng)])
            };
            return self.peak_biased_respawn_candidate(
                group,
                selected_parent,
                landscape,
                rng,
                member_idx,
                config,
            );
        }

        let capacity = group.respawn_capacity.max(1);

        // Step 1: Select parent ONCE before candidate generation
        let selected_parent: Option<ParentCandidate> = match group.respawn_policy {
            RespawnPolicy::None => return None,
            RespawnPolicy::Random => None,
            RespawnPolicy::Hereditary { .. } => {
                let pool = alive_by_group
                    .get(&group_id)
                    .map(Vec::as_slice)
                    .unwrap_or(&[]);
                if pool.is_empty() {
                    None
                } else {
                    Some(pool[weighted_parent_select(pool, rng)])
                }
            }
            RespawnPolicy::PeakBiased { .. } => unreachable!("handled above"),
        };

        // Step 2: Generate candidates (all share same parent lineage)
        let mut candidates = Vec::with_capacity(capacity);
        for idx in 0..capacity {
            let freq = match (idx, group.respawn_settle_strategy.as_ref()) {
                (1.., Some(strategy)) => self
                    .resolve_strategy_frequency(
                        strategy,
                        landscape,
                        rng,
                        &[],
                        member_idx + idx,
                        capacity,
                    )
                    .max(MIN_FREQ_HZ),
                _ => match group.respawn_policy {
                    RespawnPolicy::None => return None,
                    RespawnPolicy::Random => {
                        self.random_respawn_frequency(group, landscape, rng, member_idx + idx)
                    }
                    RespawnPolicy::Hereditary { sigma_oct } => {
                        if let Some(ref parent) = selected_parent {
                            let parent_log2 = parent.freq_hz.max(MIN_FREQ_HZ).log2();
                            let noise = Self::normal_sample(rng) * sigma_oct.max(0.0);
                            let child_log2 = parent_log2 + noise;
                            let (min_hz, max_hz) = group
                                .strategy
                                .as_ref()
                                .map(SpawnStrategy::freq_range_hz)
                                .unwrap_or_else(|| landscape.freq_bounds());
                            let lo = min_hz.clamp(MIN_FREQ_HZ, MAX_FREQ_HZ);
                            let hi = max_hz.clamp(lo, MAX_FREQ_HZ);
                            2.0f32.powf(child_log2).clamp(lo, hi)
                        } else {
                            self.random_respawn_frequency(group, landscape, rng, member_idx + idx)
                        }
                    }
                    RespawnPolicy::PeakBiased { .. } => unreachable!("handled above"),
                },
            };
            candidates.push(freq);
        }

        let chosen_freq = match group.respawn_policy {
            RespawnPolicy::Random => choose_candidate_by_scene_score(landscape, &candidates, rng)?,
            _ => *candidates.iter().max_by(|a, b| {
                landscape
                    .evaluate_pitch_level(**a)
                    .partial_cmp(&landscape.evaluate_pitch_level(**b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?,
        };

        if let Some(min_c_level) = group.respawn_min_c_level
            && landscape.evaluate_pitch_level(chosen_freq) < min_c_level
        {
            return None;
        }

        // All candidates share same parent lineage
        let (parent_id, parent_gen) = match selected_parent {
            Some(p) => (Some(p.id), Some(p.generation)),
            None => (None, None),
        };
        Some((chosen_freq, parent_id, parent_gen))
    }

    fn respawn_on_new_deaths(&mut self, scenario_finished: bool, landscape: &LandscapeFrame) {
        if scenario_finished || self.abort_requested {
            return;
        }

        let mut statuses = Vec::with_capacity(self.voices.len());
        let mut alive_by_group: BTreeMap<u64, Vec<ParentCandidate>> = BTreeMap::new();
        for voice in &self.voices {
            let alive = voice.is_alive();
            let group_id = voice.metadata.group_id;
            let id = voice.id();
            statuses.push((id, group_id, alive));
            if alive {
                let energy = match &voice.articulation.core {
                    AnyArticulationCore::Entrain(core) => core.energy.max(0.0),
                    _ => 0.0,
                };
                alive_by_group
                    .entry(group_id)
                    .or_default()
                    .push(ParentCandidate {
                        id,
                        freq_hz: voice.body.base_freq_hz().clamp(MIN_FREQ_HZ, MAX_FREQ_HZ),
                        energy,
                        generation: voice.metadata.generation,
                    });
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

            let Some((freq_hz, parent_id, parent_generation)) = self.pick_respawn_candidate(
                group_id,
                &group,
                &alive_by_group,
                landscape,
                &mut rng,
                member_idx,
            ) else {
                continue;
            };

            let id = self.allocate_runtime_id();
            self.spawn_one(
                SpawnParams {
                    id,
                    group_id,
                    member_idx,
                    resolved_freq_hz: freq_hz,
                    parent_id,
                    parent_generation,
                    reason: SpawnReason::Respawn,
                },
                &group.template,
                landscape,
            );

            if let Some(state) = self.groups.get_mut(&group_id) {
                state.next_member_idx = state.next_member_idx.saturating_add(1);
            }
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
            } => self.on_spawn_action(group_id, ids, spec, strategy, landscape),
            Action::UpdateGroup { group_id, patch } => {
                self.on_update_group_action(group_id, patch);
            }
            Action::ReleaseGroup { group_id, fade_sec } => {
                self.on_release_group_action(group_id, fade_sec);
            }
            Action::SetRespawnPolicy {
                group_id,
                policy,
                settle_strategy,
                capacity,
                min_c_level,
                background_death_rate_per_sec,
            } => {
                self.set_group_respawn_policy(
                    group_id,
                    policy,
                    settle_strategy,
                    capacity,
                    min_c_level,
                    background_death_rate_per_sec,
                );
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
                self.on_set_roughness_tolerance(value);
            }
        }
    }

    fn on_spawn_action(
        &mut self,
        group_id: u64,
        ids: Vec<u64>,
        spec: VoiceConfig,
        strategy: Option<SpawnStrategy>,
        landscape: &LandscapeFrame,
    ) {
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
                    parent_id: None,
                    parent_generation: None,
                    reason: SpawnReason::Initial,
                },
                &spec,
                landscape,
            );
            reserved.push(freq_hz);
        }
        self.ensure_group_state(group_id, spec, strategy, total);
    }

    fn on_update_group_action(&mut self, group_id: u64, patch: super::control::ControlUpdate) {
        // Group-wide runtime semantics:
        // updates apply to all current members with matching group_id.
        let mut updated = 0usize;
        for voice in self
            .voices
            .iter_mut()
            .filter(|v| v.metadata.group_id == group_id)
        {
            if let Err(err) = voice.apply_patch(&patch) {
                warn!(
                    "Update: voice {} (group {group_id}) rejected update: {err}",
                    voice.id()
                );
            } else {
                updated += 1;
            }
        }
        if updated == 0 {
            warn!("Update: no active members found for group {group_id}");
        }
        self.apply_group_update(group_id, &patch);
    }

    fn on_release_group_action(&mut self, group_id: u64, fade_sec: f32) {
        // Group-wide runtime semantics:
        // release applies to all current members with matching group_id.
        let fade_sec = fade_sec.max(0.0);
        let mut released = 0usize;
        for voice in self
            .voices
            .iter_mut()
            .filter(|v| v.metadata.group_id == group_id)
        {
            voice.start_remove_fade(fade_sec);
            released += 1;
        }
        if released == 0 {
            warn!("Release: no active members found for group {group_id}");
        }
        self.mark_group_released(group_id);
    }

    fn on_set_roughness_tolerance(&mut self, value: f32) {
        let update = LandscapeUpdate {
            roughness_k: Some(value),
            ..LandscapeUpdate::default()
        };
        self.merge_landscape_update(update);
    }

    fn merge_landscape_update(&mut self, update: LandscapeUpdate) {
        let mut merged = self.pending_update.unwrap_or_default();
        if update.mirror.is_some() {
            merged.mirror = update.mirror;
        }
        if update.roughness_k.is_some() {
            merged.roughness_k = update.roughness_k;
        }
        if update.pitch_objective_mode.is_some() {
            merged.pitch_objective_mode = update.pitch_objective_mode;
        }
        self.pending_update = Some(merged);
    }

    pub fn take_pending_update(&mut self) -> Option<LandscapeUpdate> {
        self.pending_update.take()
    }

    pub fn kuramoto_order_parameter(&self) -> Option<(f32, usize)> {
        let mut phases = Vec::with_capacity(self.voices.len());
        for voice in &self.voices {
            if !voice.is_alive() {
                continue;
            }
            let AnyArticulationCore::Entrain(core) = &voice.articulation.core else {
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
    pub fn remove_voice(&mut self, id: u64) {
        self.voices.retain(|v| v.id() != id);
        self.death_observed.remove(&id);
    }

    /// Advance voice state without emitting audio (ScheduleRenderer is output authority).
    /// `samples_len` controls sub-stepping of control-rate updates within the block.
    pub fn advance(
        &mut self,
        samples_len: usize,
        _fs: f32,
        current_frame: u64,
        dt_sec: f32,
        landscape: &Landscape,
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
        let global_coupling = self.global_coupling;
        for substep_idx in 0..steps {
            self.apply_background_turnover(dt_step_sec, substep_idx);
            let crowding_active = self.crowding_active();
            match self.control_update_mode {
                ControlUpdateMode::SnapshotPhased => {
                    self.prepare_substep_snapshot(crowding_active);
                    self.decide_substep(dt_step_sec, &rhythms, landscape, crowding_active);
                    self.commit_substep(dt_step_sec, &rhythms, landscape, global_coupling);
                }
                ControlUpdateMode::SequentialRotating => {
                    self.advance_substep_sequential_current(
                        dt_step_sec,
                        &rhythms,
                        landscape,
                        global_coupling,
                        crowding_active,
                        current_frame as usize + substep_idx,
                    );
                }
            }
            rhythms.advance_in_place(dt_step_sec);
        }

        self.apply_shutdown_fade(dt_sec);
    }

    fn crowding_active(&self) -> bool {
        self.voices
            .iter()
            .any(|v| v.is_alive() && v.effective_control.pitch.crowding_strength > 0.0)
    }

    fn prepare_substep_snapshot(&mut self, crowding_active: bool) {
        let scratch = &mut self.advance_scratch;
        scratch.freq_snapshot.clear();
        scratch.group_visibility.clear();
        scratch.commit_queue.clear();
        if !crowding_active {
            return;
        }
        // Snapshot alive frequencies once per substep to avoid order-dependent updates.
        scratch.freq_snapshot.reserve(self.voices.len());
        for voice in &self.voices {
            if voice.is_alive() {
                scratch.freq_snapshot.push((
                    voice.id(),
                    voice.metadata.group_id,
                    voice.body.base_freq_hz().max(1.0).log2(),
                ));
            }
        }
        scratch
            .group_visibility
            .extend(self.groups.iter().map(|(&group_id, group)| {
                (
                    group_id,
                    (group.crowding_target_same, group.crowding_target_other),
                )
            }));
    }

    fn decide_substep(
        &mut self,
        dt_step_sec: f32,
        rhythms: &NeuralRhythms,
        landscape: &Landscape,
        crowding_active: bool,
    ) {
        // Decide phase: evaluate all alive voices against a stable snapshot.
        for voice_idx in 0..self.voices.len() {
            let (vid, actor_group_id, alive) = {
                let v = &self.voices[voice_idx];
                (v.id(), v.metadata.group_id, v.is_alive())
            };
            if !alive {
                continue;
            }
            if crowding_active {
                self.fill_neighbors_from_snapshot(vid, actor_group_id);
            }
            let neighbors = if crowding_active {
                self.advance_scratch.neighbor_pitch_log2.as_slice()
            } else {
                &[]
            };
            let neighbor_weights = if crowding_active {
                self.advance_scratch.neighbor_salience.as_slice()
            } else {
                &[]
            };
            if let Some(voice) = self.voices.get_mut(voice_idx) {
                voice.decide_pitch_target(
                    dt_step_sec,
                    rhythms,
                    landscape,
                    neighbors,
                    neighbor_weights,
                );
            }
            self.advance_scratch
                .commit_queue
                .push(CommitQueueEntry { voice_idx });
        }
    }

    fn fill_neighbors_from_snapshot(&mut self, actor_id: u64, actor_group_id: u64) {
        let scratch = &mut self.advance_scratch;
        scratch.neighbor_pitch_log2.clear();
        scratch.neighbor_salience.clear();
        scratch
            .neighbor_pitch_log2
            .reserve(scratch.freq_snapshot.len());
        scratch
            .neighbor_salience
            .reserve(scratch.freq_snapshot.len());
        for &(neighbor_id, neighbor_group_id, log2) in &scratch.freq_snapshot {
            if neighbor_id == actor_id {
                continue;
            }
            let visible = scratch
                .group_visibility
                .get(&neighbor_group_id)
                .map(|&(same_visible, other_visible)| {
                    Self::is_neighbor_visible(
                        actor_group_id,
                        neighbor_group_id,
                        same_visible,
                        other_visible,
                    )
                })
                .unwrap_or(neighbor_group_id == actor_group_id);
            if visible {
                scratch.neighbor_pitch_log2.push(log2);
                scratch
                    .neighbor_salience
                    .push(Self::pairwise_split_sign(actor_id, neighbor_id));
            }
        }
    }

    fn fill_neighbors_from_current_state(&mut self, actor_id: u64, actor_group_id: u64) {
        let scratch = &mut self.advance_scratch;
        scratch.neighbor_pitch_log2.clear();
        scratch.neighbor_salience.clear();
        scratch
            .neighbor_pitch_log2
            .reserve(self.voices.len().saturating_sub(1));
        scratch
            .neighbor_salience
            .reserve(self.voices.len().saturating_sub(1));
        for voice in &self.voices {
            if !voice.is_alive() || voice.id() == actor_id {
                continue;
            }
            let neighbor_group_id = voice.metadata.group_id;
            let visible = self
                .groups
                .get(&neighbor_group_id)
                .map(|group| {
                    Self::is_neighbor_visible(
                        actor_group_id,
                        neighbor_group_id,
                        group.crowding_target_same,
                        group.crowding_target_other,
                    )
                })
                .unwrap_or(neighbor_group_id == actor_group_id);
            if visible {
                scratch
                    .neighbor_pitch_log2
                    .push(voice.body.base_freq_hz().max(1.0).log2());
                scratch
                    .neighbor_salience
                    .push(Self::pairwise_split_sign(actor_id, voice.id()));
            }
        }
    }

    #[inline]
    fn is_neighbor_visible(
        actor_group_id: u64,
        neighbor_group_id: u64,
        same_visible: bool,
        other_visible: bool,
    ) -> bool {
        if neighbor_group_id == actor_group_id {
            same_visible
        } else {
            other_visible
        }
    }

    fn commit_substep(
        &mut self,
        dt_step_sec: f32,
        rhythms: &NeuralRhythms,
        landscape: &Landscape,
        global_coupling: f32,
    ) {
        // Commit phase: apply articulation/body/lifecycle after all decisions are fixed.
        // Contract: no insertion/removal/reordering of `self.voices` is allowed between
        // decide and commit; commit entries carry stable indices for this substep only.
        for entry in &self.advance_scratch.commit_queue {
            if let Some(voice) = self.voices.get_mut(entry.voice_idx)
                && voice.is_alive()
            {
                voice.commit_decided_control(dt_step_sec, rhythms, landscape, global_coupling);
            }
        }
    }

    fn advance_substep_sequential_current(
        &mut self,
        dt_step_sec: f32,
        rhythms: &NeuralRhythms,
        landscape: &Landscape,
        global_coupling: f32,
        crowding_active: bool,
        order_offset: usize,
    ) {
        if self.voices.is_empty() {
            return;
        }
        let mut order: Vec<usize> = (0..self.voices.len()).collect();
        let start = order_offset % order.len();
        order.rotate_left(start);
        for voice_idx in order {
            let (vid, actor_group_id, alive) = {
                let v = &self.voices[voice_idx];
                (v.id(), v.metadata.group_id, v.is_alive())
            };
            if !alive {
                continue;
            }
            if crowding_active {
                self.fill_neighbors_from_current_state(vid, actor_group_id);
            } else {
                self.advance_scratch.neighbor_pitch_log2.clear();
                self.advance_scratch.neighbor_salience.clear();
            }
            let neighbors = if crowding_active {
                self.advance_scratch.neighbor_pitch_log2.as_slice()
            } else {
                &[]
            };
            let neighbor_weights = if crowding_active {
                self.advance_scratch.neighbor_salience.as_slice()
            } else {
                &[]
            };
            if let Some(voice) = self.voices.get_mut(voice_idx) {
                voice.decide_pitch_target(
                    dt_step_sec,
                    rhythms,
                    landscape,
                    neighbors,
                    neighbor_weights,
                );
                voice.commit_decided_control(dt_step_sec, rhythms, landscape, global_coupling);
            }
        }
    }

    fn apply_shutdown_fade(&mut self, dt_sec: f32) {
        if !self.abort_requested {
            return;
        }
        let step = dt_sec / 0.05; // fade over ~50ms
        if step.is_finite() && step > 0.0 {
            self.shutdown_gain = (self.shutdown_gain - step).max(0.0);
        }
        if self.shutdown_gain <= 0.0 {
            self.voices.clear();
        }
    }

    fn apply_background_turnover(&mut self, dt_step_sec: f32, substep_idx: usize) {
        if !dt_step_sec.is_finite() || dt_step_sec <= 0.0 {
            return;
        }
        let mut rng = SmallRng::seed_from_u64(self.background_turnover_seed(substep_idx));
        let mut dying_ids = Vec::new();
        for voice in &self.voices {
            if !voice.is_alive() || voice.remove_pending {
                continue;
            }
            let Some(group) = self.groups.get(&voice.metadata.group_id) else {
                continue;
            };
            if group.released {
                continue;
            }
            let rate = group.respawn_background_death_rate_per_sec;
            if !rate.is_finite() || rate <= 0.0 {
                continue;
            }
            let hazard = (rate * dt_step_sec).clamp(0.0, 1.0);
            if hazard > 0.0 && rng.random::<f32>() < hazard {
                dying_ids.push(voice.id());
            }
        }
        for id in dying_ids {
            if let Some(voice) = self.voices.iter_mut().find(|voice| voice.id() == id) {
                voice.start_remove_fade(0.0);
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

        let before_count = self.voices.len();
        let mut removed_ids = Vec::new();
        let death_records = &mut self.death_records;
        self.voices.retain(|voice| {
            let keep = voice.should_retain();
            if !keep {
                removed_ids.push(voice.id());
                if let Some(ref acc) = voice.life_accumulator {
                    let plv = match &voice.articulation.core {
                        AnyArticulationCore::Entrain(core) => core.plv(),
                        _ => None,
                    };
                    death_records.push(acc.finalize(
                        voice.id(),
                        voice.metadata.group_id,
                        current_frame,
                        plv,
                        voice.metadata.generation,
                    ));
                }
            }
            keep
        });
        let removed_count = before_count - self.voices.len();
        for id in removed_ids {
            self.death_observed.remove(&id);
        }

        if removed_count > 0 {
            let t = current_frame as f32 * dt_sec;
            if scenario_finished || self.abort_requested {
                warn!(
                    "Event after scenario close: [t={t:.6}] Cleaned up {removed_count} dead voices. Remaining: {} (frame_idx={current_frame})",
                    self.voices.len(),
                );
            } else {
                info!(
                    "[t={t:.6}] Cleaned up {removed_count} dead voices. Remaining: {} (frame_idx={current_frame})",
                    self.voices.len(),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::landscape::LandscapeFrame;
    use crate::core::log2space::Log2Space;
    use crate::core::timebase::Timebase;
    use crate::life::control::{ControlUpdate, PitchMode, VoiceControl};
    use crate::life::lifecycle::LifecycleConfig;
    use crate::life::phonation_engine::{OnsetEvent, OnsetKick, ToneCmd};
    use crate::life::scenario::{
        Action, ArticulationCoreConfig, RespawnPeakBiasConfig, RespawnPolicy, SpawnSpec,
        SpawnStrategy,
    };
    use crate::life::sound::{BodyKind, BodySnapshot};
    use crate::life::world_model::WorldModel;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;

    fn test_pop() -> Population {
        Population::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        })
    }

    fn make_dummy_tone_spec() -> crate::life::voice::ToneSpec {
        crate::life::voice::ToneSpec {
            tone_id: 1,
            onset: 0,
            hold_ticks: None,
            freq_hz: 440.0,
            amp: 0.5,
            smoothing_tau_sec: 0.0,
            body: BodySnapshot {
                kind: BodyKind::Sine,
                amp_scale: 1.0,
                brightness: 0.0,
                inharmonic: 0.0,
                spread: 0.0,
                unison: 1,
                motion: 0.0,
                ratios: None,
            },
            render_modulator: crate::life::sound::RenderModulatorSpec::DroneSway {
                phase: 0.0,
                sway_rate: 1.0,
            },
            adsr: None,
        }
    }

    fn spawn_spec_with_freq(freq: f32) -> SpawnSpec {
        let mut control = VoiceControl::default();
        control.pitch.freq = freq;
        SpawnSpec {
            control,
            articulation: ArticulationCoreConfig::default(),
        }
    }

    fn decay_spawn_spec_with_freq(freq: f32, half_life_sec: f32) -> SpawnSpec {
        let mut control = VoiceControl::default();
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
                k_omega: None,
                base_sigma: None,
                gate_thresholds: None,
                energy_cap: None,
            },
        }
    }

    fn sustain_spawn_spec_with_freq(freq: f32) -> SpawnSpec {
        let mut control = VoiceControl::default();
        control.pitch.freq = freq;
        SpawnSpec {
            control,
            articulation: ArticulationCoreConfig::Entrain {
                lifecycle: LifecycleConfig::Sustain {
                    initial_energy: 1.0,
                    metabolism_rate: 0.0,
                    recharge_rate: Some(0.0),
                    action_cost: Some(0.0),
                    continuous_recharge_rate: None,
                    continuous_recharge_score_low: None,
                    continuous_recharge_score_high: None,
                    selection_approx_loo: false,
                    dissonance_cost: None,
                    envelope: crate::life::scenario::EnvelopeConfig::default(),
                },
                rhythm_freq: None,
                rhythm_sensitivity: None,
                rhythm_coupling: crate::life::scenario::RhythmCouplingMode::TemporalOnly,
                rhythm_reward: None,
                breath_gain_init: None,
                k_omega: None,
                base_sigma: None,
                gate_thresholds: None,
                energy_cap: None,
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

    fn peak_bias_landscape() -> LandscapeFrame {
        let mut landscape = LandscapeFrame::new(Log2Space::new(220.0, 880.0, 96));
        let peak_a_log2 = 330.0f32.log2();
        let peak_b_log2 = 660.0f32.log2();
        let sigma_cents = 35.0f32;
        for (idx, &bin_log2) in landscape.space.centers_log2.iter().enumerate() {
            let da = (bin_log2 - peak_a_log2).abs() * 1200.0;
            let db = (bin_log2 - peak_b_log2).abs() * 1200.0;
            let peak_a = 0.85 * (-(da * da) / (2.0 * sigma_cents * sigma_cents)).exp();
            let peak_b = 1.00 * (-(db * db) / (2.0 * sigma_cents * sigma_cents)).exp();
            let score = peak_a.max(peak_b);
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
        if let Some(dying) = pop.voices.iter_mut().find(|v| v.id() == id) {
            dying.release_gain = 0.0;
            dying.release_pending = true;
        }
    }

    fn run_single_substep_targets_with_mode(
        order_reversed: bool,
        crowding_strength: f32,
        mode: ControlUpdateMode,
    ) -> Vec<(u64, f32)> {
        let mut pop = test_pop();
        pop.set_seed(101);
        pop.set_control_update_mode(mode);
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
        for voice in pop.voices.iter_mut() {
            voice.set_theta_phase_state_for_test(0.9, true);
            voice.set_accumulated_time_for_test(voice.integration_window());
        }
        if order_reversed {
            pop.voices.reverse();
        }
        pop.advance(64, 48_000.0, 0, 1.0, &landscape);
        let mut out: Vec<(u64, f32)> = pop
            .voices
            .iter()
            .map(|v| (v.id(), v.target_pitch_log2()))
            .collect();
        out.sort_by_key(|(id, _)| *id);
        out
    }

    fn run_single_substep_targets(order_reversed: bool, crowding_strength: f32) -> Vec<(u64, f32)> {
        run_single_substep_targets_with_mode(
            order_reversed,
            crowding_strength,
            ControlUpdateMode::SnapshotPhased,
        )
    }

    fn run_cross_group_visibility_trial(other_group_visible: bool) -> f32 {
        let mut pop = test_pop();
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

        for voice in pop.voices.iter_mut() {
            voice.set_theta_phase_state_for_test(0.9, true);
            voice.set_accumulated_time_for_test(voice.integration_window());
        }
        pop.advance(64, 48_000.0, 0, 1.0, &landscape);

        let mover = pop.voices.iter().find(|v| v.id() == 700).expect("mover");
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

        let pop = test_pop();
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
    fn decide_phase_does_not_mutate_body_or_release_state() {
        let mut pop = test_pop();
        let landscape = runtime_landscape();
        let mut spec = spawn_spec_with_freq(330.0);
        spec.control.pitch.mode = PitchMode::Free;
        spec.control.pitch.range_oct = 0.5;
        pop.apply_action(
            Action::Spawn {
                group_id: 90,
                ids: vec![900],
                spec,
                strategy: None,
            },
            &landscape,
            None,
        );

        let voice = pop.voices.first_mut().expect("spawned voice");
        voice.release_gain = 0.37;
        voice.release_pending = true;
        voice.set_accumulated_time_for_test(voice.integration_window());
        let base_before = voice.body.base_freq_hz();
        let release_gain_before = voice.release_gain;
        let release_pending_before = voice.release_pending;
        let rhythms = landscape.rhythm;

        voice.decide_pitch_target(0.05, &rhythms, &landscape, &[], &[]);

        assert_eq!(voice.body.base_freq_hz(), base_before);
        assert_eq!(voice.release_gain, release_gain_before);
        assert_eq!(voice.release_pending, release_pending_before);
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

        let pop = test_pop();
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

        let pop = test_pop();
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

        let pop = test_pop();
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

        let pop = test_pop();
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

        let pop = test_pop();
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

        let pop = test_pop();
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
            routing: crate::life::control::Routing::default(),
            cmds: Vec::new(),
            tones: Vec::new(),
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
        let mut pop = test_pop();
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
            Action::UpdateGroup {
                group_id: 1,
                patch: update,
            },
            &landscape,
            None,
        );
        for voice in &pop.voices {
            assert!((voice.effective_control.body.amp - 0.42).abs() <= 1e-6);
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
    fn sequential_rotating_updates_are_order_dependent_with_crowding() {
        let forward =
            run_single_substep_targets_with_mode(false, 2.0, ControlUpdateMode::SequentialRotating);
        let reversed =
            run_single_substep_targets_with_mode(true, 2.0, ControlUpdateMode::SequentialRotating);
        assert_eq!(forward.len(), reversed.len());
        let any_diff =
            forward
                .iter()
                .zip(reversed.iter())
                .any(|((id_a, pitch_a), (id_b, pitch_b))| {
                    assert_eq!(*id_a, *id_b);
                    (pitch_a - pitch_b).abs() > 1e-6
                });
        assert!(
            any_diff,
            "sequential rotating updates should react to current-state order under crowding"
        );
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
        let mut pop = test_pop();
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
            Action::ReleaseGroup {
                group_id: 1,
                fade_sec: 0.05,
            },
            &landscape,
            None,
        );
        let released: Vec<u64> = pop
            .voices
            .iter()
            .filter(|v| v.remove_pending)
            .map(|v| v.id())
            .collect();
        assert_eq!(released.len(), 2);
        assert!(released.contains(&21));
        assert!(released.contains(&22));
    }

    #[test]
    fn spawn_without_strategy_keeps_spec_frequency() {
        let mut pop = test_pop();
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
        let spawned = pop.voices.first().expect("spawned");
        assert!((spawned.body.base_freq_hz() - 275.0).abs() <= 1e-6);
    }

    #[test]
    fn respawn_none_keeps_current_behavior() {
        let mut pop = test_pop();
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
            if pop.voices.is_empty() {
                break;
            }
        }

        assert!(pop.voices.is_empty());
    }

    #[test]
    fn respawn_random_maintains_population() {
        let mut pop = test_pop();
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
                settle_strategy: None,
                capacity: 1,
                min_c_level: None,
                background_death_rate_per_sec: 0.0,
            },
            &landscape,
            None,
        );

        let mut saw_respawned = false;
        for frame in 0..300 {
            step_population(&mut pop, frame, 0.01, &landscape);
            if pop.voices.iter().any(|a| a.id() != 10) {
                saw_respawned = true;
                break;
            }
        }

        assert!(saw_respawned, "expected at least one respawned member");
        assert!(
            !pop.voices.is_empty(),
            "population should not collapse with random respawn"
        );
    }

    #[test]
    fn background_turnover_replaces_member_via_respawn() {
        let mut pop = test_pop();
        pop.set_seed(61);
        let landscape = runtime_landscape();
        pop.apply_action(
            Action::Spawn {
                group_id: 81,
                ids: vec![810],
                spec: sustain_spawn_spec_with_freq(220.0),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                group_id: 81,
                policy: RespawnPolicy::Random,
                settle_strategy: None,
                capacity: 1,
                min_c_level: None,
                background_death_rate_per_sec: 10_000.0,
            },
            &landscape,
            None,
        );

        step_population(&mut pop, 0, 0.01, &landscape);

        assert_eq!(
            pop.voices.len(),
            1,
            "respawn should preserve population size"
        );
        assert_ne!(
            pop.voices[0].id(),
            810,
            "background turnover should replace the member"
        );
    }

    #[test]
    fn respawn_hereditary_maintains_population() {
        let mut pop = test_pop();
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
                settle_strategy: None,
                capacity: 1,
                min_c_level: None,
                background_death_rate_per_sec: 0.0,
            },
            &landscape,
            None,
        );

        let mut saw_respawned = false;
        for frame in 0..300 {
            step_population(&mut pop, frame, 0.01, &landscape);
            if pop.voices.iter().any(|a| a.id() != 20) {
                saw_respawned = true;
                break;
            }
        }

        assert!(saw_respawned, "expected at least one respawned member");
        assert!(
            !pop.voices.is_empty(),
            "population should not collapse with hereditary respawn"
        );
    }

    #[test]
    fn hereditary_respawn_without_strategy_uses_parent_pitch_regression() {
        let mut pop = test_pop();
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
                settle_strategy: None,
                capacity: 1,
                min_c_level: None,
                background_death_rate_per_sec: 0.0,
            },
            &landscape,
            None,
        );

        let parent_target_hz: f32 = 440.0;
        if let Some(parent) = pop.voices.iter_mut().find(|v| v.id() == 901) {
            parent.force_set_pitch_log2(parent_target_hz.log2());
        }
        force_dead(&mut pop, 900);
        pop.cleanup_dead(0, 0.01, false, &landscape);

        let child = pop
            .voices
            .iter()
            .find(|v| v.id() != 901)
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
        let mut pop = test_pop();
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
                settle_strategy: None,
                capacity: 1,
                min_c_level: None,
                background_death_rate_per_sec: 0.0,
            },
            &landscape,
            None,
        );

        let mut respawned_id = None;
        for frame in 0..300 {
            step_population(&mut pop, frame, 0.01, &landscape);
            if let Some(id) = pop
                .voices
                .iter()
                .find(|v| v.metadata.group_id == 10 && v.id() != 30)
                .map(|v| v.id())
            {
                respawned_id = Some(id);
                break;
            }
        }
        let respawned_id = respawned_id.expect("respawned member should exist");

        pop.apply_action(
            Action::ReleaseGroup {
                group_id: 10,
                fade_sec: 0.05,
            },
            &landscape,
            None,
        );

        let respawned = pop
            .voices
            .iter()
            .find(|v| v.id() == respawned_id)
            .expect("respawned member");
        assert!(respawned.remove_pending);
    }

    #[test]
    fn live_update_reaches_respawned_members() {
        let mut pop = test_pop();
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
                settle_strategy: None,
                capacity: 1,
                min_c_level: None,
                background_death_rate_per_sec: 0.0,
            },
            &landscape,
            None,
        );

        let mut respawned_id = None;
        for frame in 0..300 {
            step_population(&mut pop, frame, 0.01, &landscape);
            if let Some(id) = pop
                .voices
                .iter()
                .find(|v| v.metadata.group_id == 11 && v.id() != 40)
                .map(|v| v.id())
            {
                respawned_id = Some(id);
                break;
            }
        }
        let respawned_id = respawned_id.expect("respawned member should exist");

        pop.apply_action(
            Action::UpdateGroup {
                group_id: 11,
                patch: ControlUpdate {
                    amp: Some(0.17),
                    ..ControlUpdate::default()
                },
            },
            &landscape,
            None,
        );

        let respawned = pop
            .voices
            .iter()
            .find(|v| v.id() == respawned_id)
            .expect("respawned member");
        assert!((respawned.effective_control.body.amp - 0.17).abs() <= 1e-6);
    }

    #[test]
    fn live_landscape_weight_update_is_inherited_by_respawn() {
        let mut pop = test_pop();
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
                settle_strategy: None,
                capacity: 1,
                min_c_level: None,
                background_death_rate_per_sec: 0.0,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::UpdateGroup {
                group_id: 91,
                patch: ControlUpdate {
                    landscape_weight: Some(0.73),
                    ..ControlUpdate::default()
                },
            },
            &landscape,
            None,
        );

        for member in pop.voices.iter().filter(|v| v.metadata.group_id == 91) {
            assert!((member.effective_control.pitch.landscape_weight - 0.73).abs() <= 1e-6);
        }

        force_dead(&mut pop, 910);
        pop.cleanup_dead(0, 0.01, false, &landscape);

        let child = pop
            .voices
            .iter()
            .find(|v| v.id() != 911)
            .expect("child exists");
        assert!((child.effective_control.pitch.landscape_weight - 0.73).abs() <= 1e-6);
    }

    #[test]
    fn release_disables_future_respawns() {
        let mut pop = test_pop();
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
                settle_strategy: None,
                capacity: 1,
                min_c_level: None,
                background_death_rate_per_sec: 0.0,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::ReleaseGroup {
                group_id: 92,
                fade_sec: 0.01,
            },
            &landscape,
            None,
        );

        let mut saw_new_id = false;
        for frame in 0..400 {
            step_population(&mut pop, frame, 0.01, &landscape);
            if pop.voices.iter().any(|v| v.id() != 920) {
                saw_new_id = true;
                break;
            }
            if pop.voices.is_empty() {
                break;
            }
        }

        assert!(!saw_new_id, "release must disable future respawns");
        assert!(
            pop.voices.is_empty(),
            "released group should drain without repopulation"
        );
    }

    #[test]
    fn hereditary_respawn_child_stays_near_parent() {
        let mut pop = test_pop();
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
                settle_strategy: None,
                capacity: 1,
                min_c_level: None,
                background_death_rate_per_sec: 0.0,
            },
            &landscape,
            None,
        );

        let parent_freq = pop
            .voices
            .iter()
            .find(|v| v.id() == 101)
            .map(|v| v.body.base_freq_hz())
            .expect("parent exists");

        if let Some(dying) = pop.voices.iter_mut().find(|v| v.id() == 100) {
            dying.release_gain = 0.0;
            dying.release_pending = true;
        }
        pop.cleanup_dead(0, 0.01, false, &landscape);

        let child = pop
            .voices
            .iter()
            .find(|v| v.id() != 101)
            .expect("child exists");
        let delta_oct = (child.body.base_freq_hz().log2() - parent_freq.log2()).abs();
        assert!(
            delta_oct < 0.05,
            "child should stay near parent in log2 space"
        );
    }

    #[test]
    fn peak_biased_respawn_prefers_parent_nearby_peak_family() {
        let mut pop = test_pop();
        pop.set_seed(59);
        let landscape = peak_bias_landscape();

        let mut spec = spawn_spec_with_freq(220.0);
        spec.control.pitch.mode = PitchMode::Lock;
        pop.apply_action(
            Action::Spawn {
                group_id: 13,
                ids: vec![130, 131],
                spec,
                strategy: Some(SpawnStrategy::Linear {
                    start_freq: 250.0,
                    end_freq: 700.0,
                }),
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                group_id: 13,
                policy: RespawnPolicy::PeakBiased {
                    config: RespawnPeakBiasConfig::default(),
                },
                settle_strategy: None,
                capacity: 8,
                min_c_level: None,
                background_death_rate_per_sec: 0.0,
            },
            &landscape,
            None,
        );

        if let Some(parent) = pop.voices.iter_mut().find(|v| v.id() == 131) {
            parent.force_set_pitch_log2(300.0f32.log2());
        }
        force_dead(&mut pop, 130);
        pop.cleanup_dead(0, 0.01, false, &landscape);

        let child = pop
            .voices
            .iter()
            .find(|v| v.id() != 131)
            .expect("child exists");
        let child_freq = child.body.base_freq_hz();
        let near_parent_peak = (child_freq.log2() - 330.0f32.log2()).abs();
        let far_peak = (child_freq.log2() - 660.0f32.log2()).abs();
        assert!(
            near_parent_peak < far_peak,
            "child should stay closer to the parent-aligned peak family"
        );
    }

    #[test]
    fn random_respawn_capacity_uses_weighted_scene_scores() {
        let mut landscape = LandscapeFrame::new(Log2Space::new(220.0, 880.0, 96));
        let candidate_bins = [12usize, 36usize, 60usize];
        let candidate_freqs = candidate_bins.map(|idx| landscape.space.centers_hz[idx]);
        let candidate_scores = [0.0f32, 0.5, 2.0];

        for (bin_idx, score) in candidate_bins.into_iter().zip(candidate_scores) {
            landscape.consonance_field_score[bin_idx] = score;
            landscape.consonance_field_level[bin_idx] = score.clamp(0.0, 1.0);
        }

        let mut rng = rand::rngs::StdRng::seed_from_u64(20260331);
        let mut counts = [0usize; 3];
        for _ in 0..4096 {
            let chosen = choose_candidate_by_scene_score(&landscape, &candidate_freqs, &mut rng)
                .expect("candidate should be selected");
            let idx = candidate_freqs
                .iter()
                .position(|freq_hz| (*freq_hz - chosen).abs() <= 1e-6)
                .expect("chosen candidate should come from the candidate list");
            counts[idx] += 1;
        }

        assert_eq!(
            counts[0], 0,
            "zero-score candidates should not be sampled when positive weights exist"
        );
        assert!(
            counts[1] > 0,
            "lower-score positive candidates should remain reachable"
        );
        assert!(
            counts[2] > counts[1],
            "higher scene scores should win more often than lower ones"
        );
    }

    #[test]
    fn spawn_strategy_respects_free_pitch_mode() {
        let mut pop = test_pop();
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
        let voice = pop.voices.first().expect("spawned");
        assert_eq!(voice.effective_control.pitch.mode, PitchMode::Free);
        assert!((voice.effective_control.pitch.freq - 220.0).abs() <= 1e-6);
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
    fn collect_phonation_batches_into_clears_stale_batch() {
        let time = Timebase {
            fs: 48_000.0,
            hop: 64,
        };
        let space = Log2Space::new(55.0, 880.0, 12);
        let landscape = LandscapeFrame::new(space.clone());
        let mut world = WorldModel::new(time, space);
        let mut pop = Population::new(time);
        let spec = spawn_spec_with_freq(440.0);
        pop.apply_action(
            Action::Spawn {
                group_id: 2,
                ids: vec![77],
                spec,
                strategy: None,
            },
            &landscape,
            None,
        );

        let mut batches = vec![PhonationBatch {
            source_id: 99,
            routing: crate::life::control::Routing::default(),
            cmds: vec![ToneCmd::On {
                tone_id: 1,
                kick: OnsetKick { strength: 1.0 },
            }],
            tones: vec![make_dummy_tone_spec()],
            onsets: vec![OnsetEvent {
                gate: 0,
                onset_tick: 0,
                strength: 1.0,
            }],
        }];

        let used = pop.collect_phonation_batches_into(&mut world, &landscape, 0, &mut batches);
        // Voice with default Sustain produces output, stale data is replaced
        assert!(used > 0 || batches[0].cmds.is_empty());
        // Source id is from the actual voice, not the stale 99
        if used > 0 {
            assert_eq!(batches[0].source_id, 77);
        }
    }
}
