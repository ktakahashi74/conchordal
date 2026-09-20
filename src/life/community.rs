use super::telemetry::LifeRecord;
use super::voice::{AnyArticulationCore, PhonationBatch, SoundBody, Voice, VoiceMetadata};
use crate::core::landscape::{Landscape, LandscapeFrame, LandscapeUpdate};
use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::dcc_coupler::ListenerPressure;
use crate::life::generator_model::GeneratorModel;
use crate::life::social_density::SocialDensityTrace;
use crate::scenario::control::{MAX_FREQ_HZ, MIN_FREQ_HZ};
use crate::scenario::{ControlUpdateMode, RespawnPolicy, SpawnStrategy, VoiceSpec};
use rand::{Rng, RngExt, SeedableRng, rngs::SmallRng};
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

pub struct Community {
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
    populations: BTreeMap<u64, RuntimePopulationState>,
    death_observed: HashSet<u64>,
    next_runtime_id: u64,
    control_update_mode: ControlUpdateMode,
    last_pred_gate_stats: Option<PredGateStats>,
    last_gate_boundary_in_hop: Option<bool>,
    last_phonation_onsets_in_hop: Option<u32>,
    last_phonation_onset_strength_in_hop: Option<f32>,
    death_records: Vec<LifeRecord>,
    auto_observe: Option<ObservationConfig>,
    runtime_events: Vec<RuntimeEvent>,
    phonation_gate_open_events: Vec<PhonationGateOpenEvent>,
    advance_scratch: AdvanceScratch,
    // Per-hop scratch buffers, held here so the hop paths reuse the capacity instead
    // of allocating a fresh Vec on the audio thread.
    gate_open_scratch: Vec<PhonationGateOpenEvent>,
    dead_id_scratch: Vec<u64>,
}

#[derive(Debug, Clone)]
struct RuntimePopulationState {
    template: VoiceSpec,
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
    population_id: u64,
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
    pub population_id: u64,
    pub voice_id: u64,
    pub member_idx: usize,
    pub freq_hz: f32,
    pub parent_id: Option<u64>,
    pub generation: u32,
    pub reason: SpawnReason,
}

/// A `phonate_when_viable()` gate's false->true latch moment (see
/// `Voice::update_phonation_gate`). `Immediate`-gated voices never emit this.
#[derive(Debug, Clone)]
pub struct PhonationGateOpenEvent {
    pub time_sec: f32,
    pub population_id: u64,
    pub voice_id: u64,
    pub consonance: f32,
}

#[derive(Default)]
struct AdvanceScratch {
    freq_snapshot: Vec<(u64, u64, f32)>,
    population_visibility: HashMap<u64, (bool, bool)>,
    neighbor_pitch_log2: Vec<f32>,
    neighbor_salience: Vec<f32>,
    commit_queue: Vec<CommitQueueEntry>,
}

#[derive(Clone, Copy, Debug)]
struct CommitQueueEntry {
    voice_idx: usize,
}

mod actions;
mod frequency;
mod respawn;
mod social;

use social::*;

impl Community {
    const CONTROL_STEP_SAMPLES: usize = 64;

    pub fn new(time: Timebase) -> Self {
        debug!("Community sample rate: {:.1} Hz", time.fs);
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
            populations: BTreeMap::new(),
            death_observed: HashSet::new(),
            next_runtime_id: 1,
            control_update_mode: ControlUpdateMode::SnapshotPhased,
            last_pred_gate_stats: None,
            last_gate_boundary_in_hop: None,
            last_phonation_onsets_in_hop: None,
            last_phonation_onset_strength_in_hop: None,
            death_records: Vec::new(),
            auto_observe: None,
            runtime_events: Vec::new(),
            phonation_gate_open_events: Vec::new(),
            advance_scratch: AdvanceScratch::default(),
            gate_open_scratch: Vec::new(),
            dead_id_scratch: Vec::new(),
        }
    }

    pub(crate) fn active_population_ids(&self) -> Vec<u64> {
        self.populations
            .iter()
            .filter_map(|(population_id, state)| (!state.released).then_some(*population_id))
            .collect()
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

    pub fn reserve_runtime_ids_through(&mut self, max_id: u64) {
        self.track_runtime_id(max_id);
    }

    pub fn drain_runtime_events(&mut self) -> Vec<RuntimeEvent> {
        std::mem::take(&mut self.runtime_events)
    }

    pub fn drain_phonation_gate_open_events(&mut self) -> Vec<PhonationGateOpenEvent> {
        std::mem::take(&mut self.phonation_gate_open_events)
    }

    pub fn take_death_records(&mut self) -> Vec<LifeRecord> {
        std::mem::take(&mut self.death_records)
    }

    fn current_time_sec(&self) -> f32 {
        let tick = self.time.frame_start_tick(self.current_frame);
        self.time.tick_to_sec(tick)
    }

    fn spawn_seed(&self, population_id: u64, count: usize, seq: u64) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.seed.hash(&mut hasher);
        self.current_frame.hash(&mut hasher);
        seq.hash(&mut hasher);
        count.hash(&mut hasher);
        population_id.hash(&mut hasher);
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

    /// Sum of onset strengths fired this hop. Accented onsets weigh more, so the
    /// production meter can sense a recurring downbeat (the seed of an emergent
    /// measure), not just an onset count.
    pub fn last_phonation_onset_strength_in_hop(&self) -> Option<f32> {
        self.last_phonation_onset_strength_in_hop
    }

    /// One batch per Voice, including policy facts when no sound command is emitted.
    pub fn collect_phonation_batches(
        &mut self,
        generator_model: &mut GeneratorModel,
        landscape: &LandscapeFrame,
        now: Tick,
    ) -> Vec<PhonationBatch> {
        let mut batches = Vec::new();
        let count =
            self.collect_phonation_batches_into(generator_model, landscape, now, &mut batches);
        batches.truncate(count);
        batches
    }

    pub(crate) fn collect_phonation_batches_into(
        &mut self,
        generator_model: &mut GeneratorModel,
        landscape: &LandscapeFrame,
        now: Tick,
        out: &mut Vec<PhonationBatch>,
    ) -> usize {
        let tb = generator_model.time;
        let hop_tick = (tb.hop as Tick).max(1);
        let frame_end = now.saturating_add(hop_tick);
        let gate_boundary_in_hop = generator_model
            .next_gate_tick_est
            .is_some_and(|gate_tick| gate_tick > now && gate_tick <= frame_end);
        let pred_scan = generator_model
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
        let mut phonation_onset_strength_in_hop = 0.0f32;
        let mut used = 0usize;
        let social_trace = self.social_trace.as_ref();
        let auto_observe_enabled = self.auto_observe.is_some();
        let time_sec = self.current_time_sec();
        let gate_open_events = &mut self.gate_open_scratch;
        gate_open_events.clear();
        for voice in &mut self.voices {
            let social_coupling = voice.social_coupling;
            if used == out.len() {
                out.push(PhonationBatch::default());
            }
            let batch = &mut out[used];
            let consonance = landscape.evaluate_pitch_level(voice.body.base_freq_hz());
            let was_gate_open = voice.phonation_gate_open();
            let extra_gate_gain = match pred_scan.as_ref() {
                Some(scan) => {
                    let gain_raw = generator_model
                        .sample_scan_field_level(scan, voice.body.base_freq_hz())
                        .clamp(0.0, 1.0);
                    let sync = voice.effective_control.phonation.spec.prediction_sync();
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
            if auto_observe_enabled && !was_gate_open && voice.phonation_gate_open() {
                gate_open_events.push(PhonationGateOpenEvent {
                    time_sec,
                    population_id: voice.metadata.population_id,
                    voice_id: voice.id(),
                    consonance,
                });
            }
            phonation_onsets_in_hop = phonation_onsets_in_hop
                .saturating_add(batch.onsets.len().min(u32::MAX as usize) as u32);
            phonation_onset_strength_in_hop += batch
                .onsets
                .iter()
                .map(|o| o.strength.max(0.0))
                .sum::<f32>();
            // Silent owners still carry current policy facts and must not appear retired.
            used += 1;
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
        self.last_phonation_onset_strength_in_hop = Some(phonation_onset_strength_in_hop);
        self.last_pred_gate_stats = pred_acc.finalize();
        self.phonation_gate_open_events
            .append(&mut self.gate_open_scratch);
        used
    }

    fn spawn_one(&mut self, params: SpawnParams, spec: &VoiceSpec, landscape: &LandscapeFrame) {
        let SpawnParams {
            id,
            population_id,
            member_idx,
            resolved_freq_hz,
            parent_id,
            parent_generation,
            reason,
        } = params;
        if self.voices.iter().any(|v| v.id() == id) {
            warn!("Spawn: id collision for {id} in population {population_id}");
            return;
        }

        let generation = parent_generation.map_or(0, |g| g + 1);
        let mut control = spec.control.clone();
        control.pitch.freq = resolved_freq_hz.clamp(MIN_FREQ_HZ, MAX_FREQ_HZ);
        let metadata = VoiceMetadata {
            population_id,
            member_idx,
            generation,
            parent_id,
        };
        let cfg = VoiceSpec {
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
            let endurance_sec = match &spawned.articulation.core {
                AnyArticulationCore::Entrain(core) => core.endurance_sec,
                AnyArticulationCore::Seq(_) | AnyArticulationCore::Drone(_) => None,
            };
            spawned.life_accumulator = Some(super::telemetry::LifeAccumulator::new(
                self.current_frame,
                observe.first_k,
                endurance_sec,
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
                population_id,
                voice_id: id,
                member_idx,
                freq_hz: resolved_freq_hz,
                parent_id,
                generation,
                reason,
            });
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

    pub fn take_pending_update(&mut self) -> Option<LandscapeUpdate> {
        self.pending_update.take()
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
        self.advance_with_listener_pressure(
            samples_len,
            current_frame,
            dt_sec,
            landscape,
            ListenerPressure::default(),
        );
    }

    pub(crate) fn advance_with_listener_pressure(
        &mut self,
        samples_len: usize,
        current_frame: u64,
        dt_sec: f32,
        landscape: &Landscape,
        listener_pressure: ListenerPressure,
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
                    self.decide_substep(
                        dt_step_sec,
                        &rhythms,
                        landscape,
                        crowding_active,
                        listener_pressure,
                    );
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
                        listener_pressure,
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
        scratch.population_visibility.clear();
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
                    voice.metadata.population_id,
                    voice.body.base_freq_hz().max(1.0).log2(),
                ));
            }
        }
        scratch
            .population_visibility
            .extend(self.populations.iter().map(|(&population_id, population)| {
                (
                    population_id,
                    (
                        population.crowding_target_same,
                        population.crowding_target_other,
                    ),
                )
            }));
    }

    fn decide_substep(
        &mut self,
        dt_step_sec: f32,
        rhythms: &NeuralRhythms,
        landscape: &Landscape,
        crowding_active: bool,
        listener_pressure: ListenerPressure,
    ) {
        // Decide phase: evaluate all alive voices against a stable snapshot.
        for voice_idx in 0..self.voices.len() {
            let (vid, actor_population_id, alive) = {
                let v = &self.voices[voice_idx];
                (v.id(), v.metadata.population_id, v.is_alive())
            };
            if !alive {
                continue;
            }
            if crowding_active {
                self.fill_neighbors_from_snapshot(vid, actor_population_id);
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
                voice.decide_pitch_target_with_listener_pressure(
                    dt_step_sec,
                    rhythms,
                    landscape,
                    neighbors,
                    neighbor_weights,
                    listener_pressure,
                );
            }
            self.advance_scratch
                .commit_queue
                .push(CommitQueueEntry { voice_idx });
        }
    }

    fn fill_neighbors_from_snapshot(&mut self, actor_id: u64, actor_population_id: u64) {
        let scratch = &mut self.advance_scratch;
        scratch.neighbor_pitch_log2.clear();
        scratch.neighbor_salience.clear();
        scratch
            .neighbor_pitch_log2
            .reserve(scratch.freq_snapshot.len());
        scratch
            .neighbor_salience
            .reserve(scratch.freq_snapshot.len());
        for &(neighbor_id, neighbor_population_id, log2) in &scratch.freq_snapshot {
            if neighbor_id == actor_id {
                continue;
            }
            let visible = scratch
                .population_visibility
                .get(&neighbor_population_id)
                .map(|&(same_visible, other_visible)| {
                    Self::is_neighbor_visible(
                        actor_population_id,
                        neighbor_population_id,
                        same_visible,
                        other_visible,
                    )
                })
                .unwrap_or(neighbor_population_id == actor_population_id);
            if visible {
                scratch.neighbor_pitch_log2.push(log2);
                scratch
                    .neighbor_salience
                    .push(Self::pairwise_split_sign(actor_id, neighbor_id));
            }
        }
    }

    fn fill_neighbors_from_current_state(&mut self, actor_id: u64, actor_population_id: u64) {
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
            let neighbor_population_id = voice.metadata.population_id;
            let visible = self
                .populations
                .get(&neighbor_population_id)
                .map(|population| {
                    Self::is_neighbor_visible(
                        actor_population_id,
                        neighbor_population_id,
                        population.crowding_target_same,
                        population.crowding_target_other,
                    )
                })
                .unwrap_or(neighbor_population_id == actor_population_id);
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
        actor_population_id: u64,
        neighbor_population_id: u64,
        same_visible: bool,
        other_visible: bool,
    ) -> bool {
        if neighbor_population_id == actor_population_id {
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

    #[allow(clippy::too_many_arguments)]
    fn advance_substep_sequential_current(
        &mut self,
        dt_step_sec: f32,
        rhythms: &NeuralRhythms,
        landscape: &Landscape,
        global_coupling: f32,
        crowding_active: bool,
        order_offset: usize,
        listener_pressure: ListenerPressure,
    ) {
        if self.voices.is_empty() {
            return;
        }
        let mut order: Vec<usize> = (0..self.voices.len()).collect();
        let start = order_offset % order.len();
        order.rotate_left(start);
        for voice_idx in order {
            let (vid, actor_population_id, alive) = {
                let v = &self.voices[voice_idx];
                (v.id(), v.metadata.population_id, v.is_alive())
            };
            if !alive {
                continue;
            }
            if crowding_active {
                self.fill_neighbors_from_current_state(vid, actor_population_id);
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
                voice.decide_pitch_target_with_listener_pressure(
                    dt_step_sec,
                    rhythms,
                    landscape,
                    neighbors,
                    neighbor_weights,
                    listener_pressure,
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
            let Some(population) = self.populations.get(&voice.metadata.population_id) else {
                continue;
            };
            if population.released {
                continue;
            }
            let rate = population.respawn_background_death_rate_per_sec;
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
        let removed_ids = &mut self.dead_id_scratch;
        removed_ids.clear();
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
                        voice.metadata.population_id,
                        current_frame,
                        plv,
                        voice.metadata.generation,
                    ));
                }
            }
            keep
        });
        let removed_count = before_count - self.voices.len();
        for id in self.dead_id_scratch.drain(..) {
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
    use crate::life::generator_model::GeneratorModel;
    use crate::life::phonation_engine::{OnsetEvent, OnsetKick, ToneCmd};
    use crate::life::sound::{BodyKind, BodySnapshot};
    use crate::scenario::control::{PhonationGate, PitchMode, VoiceControl};
    use crate::scenario::lifecycle::LifecycleConfig;
    use crate::scenario::{Action, ArticulationCoreConfig, SpawnStrategy, VoiceSpec};

    pub(super) fn test_pop() -> Community {
        Community::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        })
    }

    fn make_dummy_tone_spec() -> crate::life::voice::ToneSpec {
        crate::life::voice::ToneSpec {
            opportunity: None,
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

    pub(super) fn spawn_spec_with_freq(freq: f32) -> VoiceSpec {
        let mut control = VoiceControl::default();
        control.pitch.freq = freq;
        VoiceSpec {
            control,
            articulation: ArticulationCoreConfig::default(),
        }
    }

    pub(super) fn decay_spawn_spec_with_freq(freq: f32, half_life_sec: f32) -> VoiceSpec {
        let mut control = VoiceControl::default();
        control.pitch.freq = freq;
        VoiceSpec {
            control,
            articulation: ArticulationCoreConfig::Entrain {
                lifecycle: LifecycleConfig::Decay {
                    half_life_sec,
                    attack_sec: 0.001,
                },
                rhythm_freq: None,
                rhythm_coupling: crate::scenario::RhythmCouplingMode::TemporalOnly,
                rhythm_reward: None,
                breath_gain_init: None,
            },
        }
    }

    pub(super) fn sustain_spawn_spec_with_freq(freq: f32) -> VoiceSpec {
        let mut control = VoiceControl::default();
        control.pitch.freq = freq;
        VoiceSpec {
            control,
            articulation: ArticulationCoreConfig::Entrain {
                lifecycle: LifecycleConfig::Sustain {
                    endurance_sec: None,
                    recovery_sec: None,
                    attack_cost_fraction: Some(0.0),
                    attack_recharge_fraction: Some(0.0),
                    continuous_recharge_score_low: None,
                    continuous_recharge_score_high: None,
                    selection_approx_loo: false,
                    dissonance_penalty: 0.0,
                    envelope: crate::scenario::EnvelopeConfig::default(),
                },
                rhythm_freq: None,
                rhythm_coupling: crate::scenario::RhythmCouplingMode::TemporalOnly,
                rhythm_reward: None,
                breath_gain_init: None,
            },
        }
    }

    /// Like `sustain_spawn_spec_with_freq`, but with an explicit phonation
    /// gate and consonance-viability low bound, for `phonate_when_viable()`
    /// gate-latch tests.
    fn sustain_spawn_spec_with_gate(
        freq: f32,
        gate: crate::scenario::control::PhonationGate,
        viability_low: f32,
    ) -> VoiceSpec {
        let mut control = VoiceControl::default();
        control.pitch.freq = freq;
        control.phonation.gate = gate;
        VoiceSpec {
            control,
            articulation: ArticulationCoreConfig::Entrain {
                lifecycle: LifecycleConfig::Sustain {
                    endurance_sec: None,
                    recovery_sec: None,
                    attack_cost_fraction: Some(0.0),
                    attack_recharge_fraction: Some(0.0),
                    continuous_recharge_score_low: Some(viability_low),
                    continuous_recharge_score_high: Some(0.95),
                    selection_approx_loo: false,
                    dissonance_penalty: 0.0,
                    envelope: crate::scenario::EnvelopeConfig::default(),
                },
                rhythm_freq: None,
                rhythm_coupling: crate::scenario::RhythmCouplingMode::TemporalOnly,
                rhythm_reward: None,
                breath_gain_init: None,
            },
        }
    }

    pub(super) fn runtime_landscape() -> LandscapeFrame {
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

    pub(super) fn peak_bias_landscape() -> LandscapeFrame {
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
            landscape.consonance_field_score_eff[idx] = score;
            landscape.consonance_field_level_eff[idx] = score.clamp(0.0, 1.0);
        }
        landscape.rhythm.theta.phase = 0.1;
        landscape.rhythm.theta.mag = 1.0;
        landscape
    }

    pub(super) fn step_population(
        pop: &mut Community,
        frame: u64,
        dt_sec: f32,
        landscape: &LandscapeFrame,
    ) {
        let fs = 48_000.0;
        let samples_per_hop = (fs * dt_sec) as usize;
        pop.advance(samples_per_hop, fs, frame, dt_sec, landscape);
        pop.cleanup_dead(frame, dt_sec, false, landscape);
    }

    pub(super) fn force_dead(pop: &mut Community, id: u64) {
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
        spec.control.pitch.crowding_strength = crowding_strength;
        // Post-Stage-2 crowding samples a literal sigma_cents Gaussian; this width
        // must reach the ~53c-spaced voices for sequential order to matter.
        spec.control.pitch.crowding_sigma_cents = 60.0;
        pop.apply_action(
            Action::Spawn {
                population_id: 66,
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

    fn run_cross_population_visibility_trial(other_population_visible: bool) -> f32 {
        let mut pop = test_pop();
        pop.set_seed(303);
        let landscape = crowding_order_landscape();

        let mut mover_spec = spawn_spec_with_freq(330.0);
        mover_spec.control.pitch.mode = PitchMode::Free;
        mover_spec.control.pitch.range_oct = 1.5;
        mover_spec.control.pitch.crowding_strength = 3.0;
        mover_spec.control.pitch.crowding_sigma_cents = 20.0;

        let mut neighbor_spec = spawn_spec_with_freq(330.0);
        neighbor_spec.control.pitch.mode = PitchMode::Lock;

        pop.apply_action(
            Action::Spawn {
                population_id: 70,
                ids: vec![700],
                spec: mover_spec,
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::Spawn {
                population_id: 71,
                ids: vec![701],
                spec: neighbor_spec,
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetPopulationCrowdingTarget {
                population_id: 71,
                same_population_visible: true,
                other_population_visible,
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
    fn decide_phase_does_not_mutate_body_or_release_state() {
        let mut pop = test_pop();
        let landscape = runtime_landscape();
        let mut spec = spawn_spec_with_freq(330.0);
        spec.control.pitch.mode = PitchMode::Free;
        spec.control.pitch.range_oct = 0.5;
        pop.apply_action(
            Action::Spawn {
                population_id: 90,
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
    fn cross_population_crowding_follows_target_visibility_policy() {
        let hidden = run_cross_population_visibility_trial(false);
        let visible = run_cross_population_visibility_trial(true);
        assert!(
            visible > hidden + 1e-6,
            "cross-population crowding should only affect behavior when target population allows visibility"
        );
    }

    #[test]
    fn pairwise_split_sign_is_antisymmetric() {
        let ab = Community::pairwise_split_sign(10, 42);
        let ba = Community::pairwise_split_sign(42, 10);
        assert!(ab.abs() > 0.0);
        assert!((ab + ba).abs() <= 1e-6);
    }

    #[test]
    fn reserved_scenario_ids_are_not_reused_by_runtime_spawns() {
        let mut pop = test_pop();
        pop.reserve_runtime_ids_through(12);

        let id = pop.allocate_runtime_id();

        assert_eq!(id, 13);
    }

    #[test]
    fn collect_phonation_batches_into_clears_stale_batch() {
        let time = Timebase {
            fs: 48_000.0,
            hop: 64,
        };
        let space = Log2Space::new(55.0, 880.0, 12);
        let landscape = LandscapeFrame::new(space.clone());
        let mut world = GeneratorModel::new(time, space);
        let mut pop = Community::new(time);
        let spec = spawn_spec_with_freq(440.0);
        pop.apply_action(
            Action::Spawn {
                population_id: 2,
                ids: vec![77],
                spec,
                strategy: None,
            },
            &landscape,
            None,
        );

        let mut batches = vec![PhonationBatch {
            body_policy: None,
            body_opportunity: None,
            intrinsic_period_sec: None,
            source_id: 99,
            source_generation: 0,
            routing: crate::scenario::control::Routing::default(),
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
            assert_eq!(batches[0].body_policy.unwrap().at, 0);
        }
        let used = pop.collect_phonation_batches_into(&mut world, &landscape, 64, &mut batches);
        assert_eq!(used, 1);
        assert!(batches[0].onsets.is_empty());
        assert_eq!(batches[0].body_policy.unwrap().at, 64);
    }

    #[test]
    fn when_viable_gate_latch_emits_exactly_one_phonation_gate_open_event() {
        let time = Timebase {
            fs: 48_000.0,
            hop: 64,
        };
        let space = Log2Space::new(55.0, 880.0, 12);
        let mut world = GeneratorModel::new(time, space.clone());
        let mut pop = Community::new(time);
        pop.enable_auto_observe();

        let mut landscape_low = LandscapeFrame::new(space.clone());
        landscape_low.consonance_field_level.fill(0.0);
        landscape_low.consonance_field_level_eff.fill(0.0);
        let mut landscape_high = LandscapeFrame::new(space);
        landscape_high.consonance_field_level.fill(1.0);
        landscape_high.consonance_field_level_eff.fill(1.0);

        let spec = sustain_spawn_spec_with_gate(440.0, PhonationGate::WhenViable, 0.5);
        pop.apply_action(
            Action::Spawn {
                population_id: 9,
                ids: vec![501],
                spec,
                strategy: None,
            },
            &landscape_low,
            None,
        );

        let mut batches = Vec::new();
        pop.collect_phonation_batches_into(&mut world, &landscape_low, 0, &mut batches);
        assert!(
            pop.drain_phonation_gate_open_events().is_empty(),
            "gate stays closed while consonance is below the viability low bound"
        );

        pop.collect_phonation_batches_into(&mut world, &landscape_high, 64, &mut batches);
        let events = pop.drain_phonation_gate_open_events();
        assert_eq!(events.len(), 1, "exactly one latch-open record");
        assert_eq!(events[0].population_id, 9);
        assert_eq!(events[0].voice_id, 501);

        pop.collect_phonation_batches_into(&mut world, &landscape_high, 128, &mut batches);
        assert!(
            pop.drain_phonation_gate_open_events().is_empty(),
            "the one-way latch must not re-fire once open"
        );
    }

    #[test]
    fn immediate_gate_never_emits_phonation_gate_open_events() {
        let time = Timebase {
            fs: 48_000.0,
            hop: 64,
        };
        let space = Log2Space::new(55.0, 880.0, 12);
        let mut world = GeneratorModel::new(time, space.clone());
        let mut pop = Community::new(time);
        pop.enable_auto_observe();

        let mut landscape_high = LandscapeFrame::new(space);
        landscape_high.consonance_field_level.fill(1.0);
        landscape_high.consonance_field_level_eff.fill(1.0);

        let spec = sustain_spawn_spec_with_gate(440.0, PhonationGate::Immediate, 0.5);
        pop.apply_action(
            Action::Spawn {
                population_id: 9,
                ids: vec![502],
                spec,
                strategy: None,
            },
            &landscape_high,
            None,
        );

        let mut batches = Vec::new();
        pop.collect_phonation_batches_into(&mut world, &landscape_high, 0, &mut batches);
        pop.collect_phonation_batches_into(&mut world, &landscape_high, 64, &mut batches);
        assert!(
            pop.drain_phonation_gate_open_events().is_empty(),
            "Immediate gates start open and never latch, so they never emit"
        );
    }
}
