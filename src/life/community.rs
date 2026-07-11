use super::telemetry::LifeRecord;
use super::voice::{AnyArticulationCore, PhonationBatch, SoundBody, Voice, VoiceMetadata};
use crate::core::landscape::{Landscape, LandscapeFrame, LandscapeUpdate};
use crate::core::modulation::NeuralRhythms;
use crate::core::timebase::{Tick, Timebase};
use crate::dcc_coupler::ListenerPressure;
use crate::life::control::{MAX_FREQ_HZ, MIN_FREQ_HZ};
use crate::life::generator_model::GeneratorModel;
use crate::life::social_density::SocialDensityTrace;
use crate::scenario::{
    Action, ControlUpdateMode, FieldSampling, FieldTarget, RespawnPeakBiasConfig, RespawnPolicy,
    SpawnStrategy, VoiceSpec,
};
use rand::{
    Rng, RngExt, SeedableRng, distr::Distribution, distr::weighted::WeightedIndex, rngs::SmallRng,
};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{Hash, Hasher};
use tracing::{debug, info, warn};

const DEFAULT_REPORT_FIRST_K: u32 = 10;
const DEFAULT_REPORT_PLV_WINDOW: usize = 200;
const RESPAWN_CANDIDATE_COUNT: usize = 16;
/// Width (fraction of the range's field_score span) of the Gaussian that weights
/// peaks near the tension target in density placement.
const TENSION_LEVEL_SIGMA_FRAC: f32 = 0.15;

/// Peak score for a field target at bin `i` (higher = better extremum).
fn field_peak_score(target: FieldTarget, landscape: &LandscapeFrame, i: usize) -> f32 {
    match target {
        FieldTarget::Consonance => landscape
            .consonance_field_level
            .get(i)
            .copied()
            .unwrap_or(f32::MIN),
        FieldTarget::Dissonance => -landscape
            .consonance_field_level
            .get(i)
            .copied()
            .unwrap_or(f32::MAX),
        FieldTarget::Edge => {
            let v = landscape
                .consonance_field_level
                .get(i)
                .copied()
                .unwrap_or(f32::MAX);
            -(v - 0.5).abs()
        }
        FieldTarget::Gap => -landscape
            .subjective_intensity
            .get(i)
            .copied()
            .unwrap_or(f32::MAX),
        FieldTarget::Uniform => 0.0,
    }
}

/// Non-negative density mass for a field target at bin `i`. `gap_ref` is the
/// loudest in-range intensity, used only by `Gap`.
fn field_density_mass(
    target: FieldTarget,
    landscape: &LandscapeFrame,
    i: usize,
    gap_ref: f32,
) -> f32 {
    match target {
        FieldTarget::Consonance => landscape
            .consonance_density_mass
            .get(i)
            .copied()
            .unwrap_or(0.0),
        FieldTarget::Dissonance => {
            // Missing bins default to fully consonant so they get zero mass.
            let v = landscape
                .consonance_field_level
                .get(i)
                .copied()
                .unwrap_or(1.0);
            (1.0 - v).max(0.0)
        }
        FieldTarget::Edge => {
            let v = landscape
                .consonance_field_level
                .get(i)
                .copied()
                .unwrap_or(0.0);
            (1.0 - 2.0 * (v - 0.5).abs()).max(0.0)
        }
        FieldTarget::Gap => {
            let e = landscape
                .subjective_intensity
                .get(i)
                .copied()
                .unwrap_or(gap_ref);
            (gap_ref - e).max(0.0)
        }
        FieldTarget::Uniform => 1.0,
    }
}

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

mod respawn;
mod social;

use respawn::*;
use social::*;

impl Community {
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
        let mut gate_open_events: Vec<PhonationGateOpenEvent> = Vec::new();
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
        self.last_phonation_onset_strength_in_hop = Some(phonation_onset_strength_in_hop);
        self.last_pred_gate_stats = pred_acc.finalize();
        self.phonation_gate_open_events.extend(gate_open_events);
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

        // Tension (Consonance only): aim at a metastable step below the strongest
        // peak — target = L_max - tension*(L_max - L_min) in field_score over the
        // range. None when tension == 0 (plain strongest-consonance behaviour).
        let tension = match strategy {
            SpawnStrategy::Field {
                target: FieldTarget::Consonance,
                tension,
                ..
            } => (*tension).clamp(0.0, 1.0),
            _ => 0.0,
        };
        let (target_score, tension_sigma) = if tension > 0.0 {
            let mut lmax = f32::MIN;
            let mut lmin = f32::MAX;
            for i in idx_min..=idx_max {
                let s = landscape
                    .consonance_field_score
                    .get(i)
                    .copied()
                    .unwrap_or(f32::NAN);
                if s.is_finite() {
                    lmax = lmax.max(s);
                    lmin = lmin.min(s);
                }
            }
            if lmax > lmin {
                let sigma = ((lmax - lmin) * TENSION_LEVEL_SIGMA_FRAC).max(1e-6);
                (Some(lmax - tension * (lmax - lmin)), sigma)
            } else {
                (None, 1.0)
            }
        } else {
            (None, 1.0)
        };

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
            // Flat (log-uniform) measure: sample continuously, retrying occupancy.
            SpawnStrategy::Field {
                target: FieldTarget::Uniform,
                ..
            } => {
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
            // Deterministic extremum of the target (higher score = better).
            SpawnStrategy::Field {
                target,
                sampling: FieldSampling::Peak,
                ..
            } => {
                let mut best_free = None;
                let mut best_any = (idx_min, f32::MIN);
                for i in idx_min..=idx_max {
                    let score = match target_score {
                        Some(t) => {
                            let s = landscape
                                .consonance_field_score
                                .get(i)
                                .copied()
                                .unwrap_or(f32::MIN);
                            -(s - t).abs() // nearest to the tension target = best
                        }
                        None => field_peak_score(*target, landscape, i),
                    };
                    if score > best_any.1 {
                        best_any = (i, score);
                    }
                    let f = space.freq_of_index(i);
                    if !self.is_range_occupied_with(f, min_dist_erb, reserved)
                        && score > best_free.map_or(f32::MIN, |(_, v)| v)
                    {
                        best_free = Some((i, score));
                    }
                }
                best_free.unwrap_or(best_any).0
            }
            // Stochastic cloud weighted by the target mass.
            SpawnStrategy::Field { target, .. } => {
                let range_len = idx_max - idx_min + 1;
                // Gap mass is measured relative to the loudest bin in range.
                let gap_ref = if *target == FieldTarget::Gap {
                    (idx_min..=idx_max)
                        .filter_map(|i| landscape.subjective_intensity.get(i).copied())
                        .fold(0.0f32, f32::max)
                } else {
                    0.0
                };
                let mut weights = Vec::with_capacity(range_len);
                let mut has_unoccupied = false;
                let mut sum = 0.0f32;
                for i in idx_min..=idx_max {
                    let f = space.freq_of_index(i);
                    let occupied = self.is_range_occupied_with(f, min_dist_erb, reserved);
                    if !occupied {
                        has_unoccupied = true;
                    }
                    let raw = match target_score {
                        Some(t) => {
                            let s = landscape
                                .consonance_field_score
                                .get(i)
                                .copied()
                                .unwrap_or(f32::MIN);
                            let base = field_density_mass(*target, landscape, i, gap_ref);
                            base * (-(s - t).powi(2) / (2.0 * tension_sigma * tension_sigma)).exp()
                        }
                        None => field_density_mass(*target, landscape, i, gap_ref),
                    };
                    let w = if occupied { 0.0 } else { raw.max(0.0) };
                    let w = if w.is_finite() { w } else { 0.0 };
                    weights.push((w, occupied));
                    sum += w;
                }
                // Fallback: if mass sums to zero, use uniform over unoccupied bins.
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

    fn ensure_population_state(
        &mut self,
        population_id: u64,
        spec: VoiceSpec,
        strategy: Option<SpawnStrategy>,
        member_count: usize,
    ) {
        let current_members = self
            .voices
            .iter()
            .filter(|v| v.metadata.population_id == population_id)
            .count();
        if let Some(population) = self.populations.get_mut(&population_id) {
            // Runtime currently allows multiple Spawn actions with the same population_id.
            // In that case we treat it as "refresh population template/strategy" while preserving
            // existing runtime policies. New Spawn implicitly re-activates the population.
            population.template = spec;
            population.strategy = strategy;
            population.released = false;
            population.spawn_count_hint = member_count.max(1);
            population.next_member_idx = population.next_member_idx.max(current_members);
            return;
        }
        self.populations.insert(
            population_id,
            RuntimePopulationState {
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

    fn apply_population_update(
        &mut self,
        population_id: u64,
        update: &super::control::ControlUpdate,
    ) {
        if let Some(population) = self.populations.get_mut(&population_id) {
            population.template.control.apply_update(update);
        }
    }

    fn mark_population_released(&mut self, population_id: u64) {
        if let Some(population) = self.populations.get_mut(&population_id) {
            population.released = true;
        }
    }

    fn set_population_respawn_policy(
        &mut self,
        population_id: u64,
        policy: RespawnPolicy,
        settle_strategy: Option<SpawnStrategy>,
        capacity: usize,
        min_c_level: Option<f32>,
        background_death_rate_per_sec: f32,
    ) {
        if let Some(population) = self.populations.get_mut(&population_id) {
            population.respawn_policy = policy;
            population.respawn_settle_strategy = settle_strategy;
            population.respawn_capacity = capacity.max(1);
            population.respawn_min_c_level = min_c_level.map(|value| value.clamp(0.0, 1.0));
            population.respawn_background_death_rate_per_sec =
                background_death_rate_per_sec.max(0.0);
        } else {
            warn!("SetRespawnPolicy: unknown population {population_id}");
        }
    }

    fn set_population_crowding_target(
        &mut self,
        population_id: u64,
        same_population_visible: bool,
        other_population_visible: bool,
    ) {
        if let Some(population) = self.populations.get_mut(&population_id) {
            population.crowding_target_same = same_population_visible;
            population.crowding_target_other = other_population_visible;
        } else {
            warn!("SetPopulationCrowdingTarget: unknown population {population_id}");
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
        population: &RuntimePopulationState,
        landscape: &LandscapeFrame,
        rng: &mut R,
        member_idx: usize,
    ) -> f32 {
        if let Some(strategy) = population.strategy.as_ref() {
            let linear_idx = member_idx % population.spawn_count_hint.max(1);
            self.resolve_strategy_frequency(
                strategy,
                landscape,
                rng,
                &[],
                linear_idx,
                population.spawn_count_hint.max(1),
            )
            .max(MIN_FREQ_HZ)
        } else {
            population
                .template
                .control
                .pitch
                .freq
                .clamp(MIN_FREQ_HZ, MAX_FREQ_HZ)
        }
    }

    fn peak_biased_respawn_candidate<R: Rng + ?Sized>(
        &self,
        population: &RuntimePopulationState,
        selected_parent: Option<ParentCandidate>,
        landscape: &LandscapeFrame,
        rng: &mut R,
        member_idx: usize,
        config: RespawnPeakBiasConfig,
    ) -> Option<(f32, Option<u64>, Option<u32>)> {
        let (min_hz, max_hz) = population
            .strategy
            .as_ref()
            .map(SpawnStrategy::freq_range_hz)
            .unwrap_or_else(|| landscape.freq_bounds());
        let lo = min_hz.clamp(MIN_FREQ_HZ, MAX_FREQ_HZ);
        let hi = max_hz.clamp(lo, MAX_FREQ_HZ);
        let candidate_count = RESPAWN_CANDIDATE_COUNT;
        let candidate_bins = peak_bias_candidate_bins(landscape, lo, hi, candidate_count);

        let fallback_freq = selected_parent
            .map(|parent| parent.freq_hz.clamp(lo, hi))
            .unwrap_or_else(|| {
                self.random_respawn_frequency(population, landscape, rng, member_idx)
            });

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

        if let Some(min_c_level) = population.respawn_min_c_level
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
        population_id: u64,
        population: &RuntimePopulationState,
        alive_by_population: &BTreeMap<u64, Vec<ParentCandidate>>,
        landscape: &LandscapeFrame,
        rng: &mut R,
        member_idx: usize,
    ) -> Option<(f32, Option<u64>, Option<u32>)> {
        if let RespawnPolicy::PeakBiased { config } = population.respawn_policy {
            let pool = alive_by_population
                .get(&population_id)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let selected_parent = if pool.is_empty() {
                None
            } else {
                Some(pool[weighted_parent_select(pool, rng)])
            };
            return self.peak_biased_respawn_candidate(
                population,
                selected_parent,
                landscape,
                rng,
                member_idx,
                config,
            );
        }

        let candidate_count = RESPAWN_CANDIDATE_COUNT;

        // Step 1: Select parent ONCE before candidate generation
        let selected_parent: Option<ParentCandidate> = match population.respawn_policy {
            RespawnPolicy::None => return None,
            RespawnPolicy::Random => None,
            RespawnPolicy::Hereditary { .. } => {
                let pool = alive_by_population
                    .get(&population_id)
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
        let mut candidates = Vec::with_capacity(candidate_count);
        for idx in 0..candidate_count {
            let freq = match (idx, population.respawn_settle_strategy.as_ref()) {
                (1.., Some(strategy)) => self
                    .resolve_strategy_frequency(
                        strategy,
                        landscape,
                        rng,
                        &[],
                        member_idx + idx,
                        candidate_count,
                    )
                    .max(MIN_FREQ_HZ),
                _ => match population.respawn_policy {
                    RespawnPolicy::None => return None,
                    RespawnPolicy::Random => {
                        self.random_respawn_frequency(population, landscape, rng, member_idx + idx)
                    }
                    RespawnPolicy::Hereditary { sigma_oct } => {
                        if let Some(ref parent) = selected_parent {
                            let parent_log2 = parent.freq_hz.max(MIN_FREQ_HZ).log2();
                            let noise = Self::normal_sample(rng) * sigma_oct.max(0.0);
                            let child_log2 = parent_log2 + noise;
                            let (min_hz, max_hz) = population
                                .strategy
                                .as_ref()
                                .map(SpawnStrategy::freq_range_hz)
                                .unwrap_or_else(|| landscape.freq_bounds());
                            let lo = min_hz.clamp(MIN_FREQ_HZ, MAX_FREQ_HZ);
                            let hi = max_hz.clamp(lo, MAX_FREQ_HZ);
                            2.0f32.powf(child_log2).clamp(lo, hi)
                        } else {
                            self.random_respawn_frequency(
                                population,
                                landscape,
                                rng,
                                member_idx + idx,
                            )
                        }
                    }
                    RespawnPolicy::PeakBiased { .. } => unreachable!("handled above"),
                },
            };
            candidates.push(freq);
        }

        let chosen_freq = match population.respawn_policy {
            RespawnPolicy::Random => choose_candidate_by_scene_score(landscape, &candidates, rng)?,
            _ => *candidates.iter().max_by(|a, b| {
                landscape
                    .evaluate_pitch_level(**a)
                    .partial_cmp(&landscape.evaluate_pitch_level(**b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })?,
        };

        if let Some(min_c_level) = population.respawn_min_c_level
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
        let mut alive_by_population: BTreeMap<u64, Vec<ParentCandidate>> = BTreeMap::new();
        for voice in &self.voices {
            let alive = voice.is_alive();
            let population_id = voice.metadata.population_id;
            let id = voice.id();
            statuses.push((id, population_id, alive));
            if alive {
                let energy = match &voice.articulation.core {
                    AnyArticulationCore::Entrain(core) => core.energy.max(0.0),
                    _ => 0.0,
                };
                alive_by_population
                    .entry(population_id)
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
        for (id, population_id, alive) in statuses {
            if alive {
                self.death_observed.remove(&id);
                continue;
            }
            if self.death_observed.insert(id) {
                dead_candidates.push((id, population_id));
            }
        }

        let mut projected_alive: BTreeMap<u64, usize> = alive_by_population
            .iter()
            .map(|(population_id, members)| (*population_id, members.len()))
            .collect();

        for (_dead_id, population_id) in dead_candidates {
            let Some(population) = self.populations.get(&population_id).cloned() else {
                continue;
            };
            if population.released {
                continue;
            }
            let alive_count = projected_alive.get(&population_id).copied().unwrap_or(0);
            if alive_count >= population.respawn_capacity {
                continue;
            }

            let member_idx = population.next_member_idx;
            let spawn_seq = self.spawn_counter;
            self.spawn_counter = self.spawn_counter.wrapping_add(1);
            let seed = self.spawn_seed(population_id, 1, spawn_seq);
            let mut rng = SmallRng::seed_from_u64(seed);

            let Some((freq_hz, parent_id, parent_generation)) = self.pick_respawn_candidate(
                population_id,
                &population,
                &alive_by_population,
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
                    population_id,
                    member_idx,
                    resolved_freq_hz: freq_hz,
                    parent_id,
                    parent_generation,
                    reason: SpawnReason::Respawn,
                },
                &population.template,
                landscape,
            );

            if let Some(state) = self.populations.get_mut(&population_id) {
                state.next_member_idx = state.next_member_idx.saturating_add(1);
            }
            *projected_alive.entry(population_id).or_default() += 1;
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
                population_id,
                ids,
                spec,
                strategy,
            } => self.on_spawn_action(population_id, ids, spec, strategy, landscape),
            Action::UpdatePopulation {
                population_id,
                patch,
            } => {
                self.on_update_population_action(population_id, patch);
            }
            Action::ReleasePopulation {
                population_id,
                fade_sec,
            } => {
                self.on_release_population_action(population_id, fade_sec);
            }
            Action::SetRespawnPolicy {
                population_id,
                policy,
                settle_strategy,
                capacity,
                min_c_level,
                background_death_rate_per_sec,
            } => {
                self.set_population_respawn_policy(
                    population_id,
                    policy,
                    settle_strategy,
                    capacity,
                    min_c_level,
                    background_death_rate_per_sec,
                );
            }
            Action::SetPopulationCrowdingTarget {
                population_id,
                same_population_visible,
                other_population_visible,
            } => {
                self.set_population_crowding_target(
                    population_id,
                    same_population_visible,
                    other_population_visible,
                );
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
        population_id: u64,
        ids: Vec<u64>,
        spec: VoiceSpec,
        strategy: Option<SpawnStrategy>,
        landscape: &LandscapeFrame,
    ) {
        let spawn_seq = self.spawn_counter;
        self.spawn_counter = self.spawn_counter.wrapping_add(1);
        let seed = self.spawn_seed(population_id, ids.len(), spawn_seq);
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
                    population_id,
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
        self.ensure_population_state(population_id, spec, strategy, total);
    }

    fn on_update_population_action(
        &mut self,
        population_id: u64,
        patch: super::control::ControlUpdate,
    ) {
        // Population-wide runtime semantics:
        // updates apply to all current members with matching population_id.
        let mut updated = 0usize;
        for voice in self
            .voices
            .iter_mut()
            .filter(|v| v.metadata.population_id == population_id)
        {
            if let Err(err) = voice.apply_patch(&patch) {
                warn!(
                    "Update: voice {} (population {population_id}) rejected update: {err}",
                    voice.id()
                );
            } else {
                updated += 1;
            }
        }
        if updated == 0 {
            warn!("Update: no active members found for population {population_id}");
        }
        self.apply_population_update(population_id, &patch);
    }

    fn on_release_population_action(&mut self, population_id: u64, fade_sec: f32) {
        // Population-wide runtime semantics:
        // release applies to all current members with matching population_id.
        let fade_sec = fade_sec.max(0.0);
        let mut released = 0usize;
        for voice in self
            .voices
            .iter_mut()
            .filter(|v| v.metadata.population_id == population_id)
        {
            voice.start_remove_fade(fade_sec);
            released += 1;
        }
        if released == 0 {
            warn!("Release: no active members found for population {population_id}");
        }
        self.mark_population_released(population_id);
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

    /// Offset-removed entrainment phases of live voices, one per `Entrain` core.
    /// Used for the Kuramoto order parameter and the GUI phase circle.
    pub fn entrain_aligned_phases(&self) -> Vec<f32> {
        let mut phases = Vec::with_capacity(self.voices.len());
        for voice in &self.voices {
            if !voice.is_alive() {
                continue;
            }
            let AnyArticulationCore::Entrain(core) = &voice.articulation.core else {
                continue;
            };
            let aligned_phase =
                (core.rhythm_phase - core.phase_offset).rem_euclid(std::f32::consts::TAU);
            if aligned_phase.is_finite() {
                phases.push(aligned_phase);
            }
        }
        phases
    }

    pub fn kuramoto_order_parameter(&self) -> Option<(f32, usize)> {
        let phases = self.entrain_aligned_phases();
        let r = kuramoto_order_from_phases(&phases)?;
        Some((r, phases.len()))
    }

    /// Entrainment phases plus their Kuramoto order in a single voice scan, for
    /// the UI frame (avoids scanning + allocating twice per frame).
    pub fn entrain_phases_and_order(&self) -> (Vec<f32>, Option<f32>) {
        let phases = self.entrain_aligned_phases();
        let r = kuramoto_order_from_phases(&phases);
        (phases, r)
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
    use crate::life::control::{ControlUpdate, PhonationGate, PitchMode, VoiceControl};
    use crate::life::generator_model::GeneratorModel;
    use crate::life::lifecycle::LifecycleConfig;
    use crate::life::phonation_engine::{OnsetEvent, OnsetKick, ToneCmd};
    use crate::life::sound::{BodyKind, BodySnapshot};
    use crate::scenario::{
        Action, ArticulationCoreConfig, RespawnPeakBiasConfig, RespawnPolicy, SpawnStrategy,
        VoiceSpec,
    };
    use rand::{RngExt, SeedableRng};
    use std::collections::HashSet;

    fn test_pop() -> Community {
        Community::new(Timebase {
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

    fn spawn_spec_with_freq(freq: f32) -> VoiceSpec {
        let mut control = VoiceControl::default();
        control.pitch.freq = freq;
        VoiceSpec {
            control,
            articulation: ArticulationCoreConfig::default(),
        }
    }

    fn decay_spawn_spec_with_freq(freq: f32, half_life_sec: f32) -> VoiceSpec {
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

    fn sustain_spawn_spec_with_freq(freq: f32) -> VoiceSpec {
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
        gate: crate::life::control::PhonationGate,
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

    fn step_population(pop: &mut Community, frame: u64, dt_sec: f32, landscape: &LandscapeFrame) {
        let fs = 48_000.0;
        let samples_per_hop = (fs * dt_sec) as usize;
        pop.advance(samples_per_hop, fs, frame, dt_sec, landscape);
        pop.cleanup_dead(frame, dt_sec, false, landscape);
    }

    fn force_dead(pop: &mut Community, id: u64) {
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
        let strategy = SpawnStrategy::Field {
            target: FieldTarget::Consonance,
            sampling: FieldSampling::Peak,
            min_freq: 100.0,
            max_freq: 400.0,
            min_dist_erb: 0.0,
            tension: 0.0,
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
        let strategy = SpawnStrategy::Field {
            target: FieldTarget::Consonance,
            sampling: FieldSampling::Density,
            min_freq: space.fmin,
            max_freq: space.fmax,
            min_dist_erb: 0.0,
            tension: 0.0,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);

        for _ in 0..64 {
            let freq = pop.decide_frequency(&strategy, &landscape, &mut rng, &[]);
            let picked_idx = space.index_of_freq(freq).expect("picked idx");
            assert_eq!(picked_idx, idx_target);
        }
    }

    #[test]
    fn consonance_tension_peak_targets_a_lower_step() {
        let space = Log2Space::new(100.0, 800.0, 48);
        let mut landscape = LandscapeFrame::new(space.clone());
        // Background well below the three steps so L_min is the background.
        for s in landscape.consonance_field_score.iter_mut() {
            *s = -1.0;
        }
        let idx_strong = space.index_of_freq(200.0).expect("strong");
        let idx_mid = space.index_of_freq(300.0).expect("mid");
        let idx_weak = space.index_of_freq(500.0).expect("weak");
        landscape.consonance_field_score[idx_strong] = 1.0; // L_max
        landscape.consonance_field_score[idx_mid] = 0.5;
        landscape.consonance_field_score[idx_weak] = 0.0;

        let pop = test_pop();
        let mk = |t: f32| SpawnStrategy::Field {
            target: FieldTarget::Consonance,
            sampling: FieldSampling::Peak,
            min_freq: 100.0,
            max_freq: 800.0,
            min_dist_erb: 0.0,
            tension: t,
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        // L_max=1, L_min=-1: target = 1 - 2*tension.
        // tension=0.25 -> 0.5 (mid step); tension=0.5 -> 0.0 (weak step).
        let f_mid = pop.decide_frequency(&mk(0.25), &landscape, &mut rng, &[]);
        assert_eq!(space.index_of_freq(f_mid).expect("idx"), idx_mid);
        let f_weak = pop.decide_frequency(&mk(0.5), &landscape, &mut rng, &[]);
        assert_eq!(space.index_of_freq(f_weak).expect("idx"), idx_weak);
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
        let strategy = SpawnStrategy::Field {
            target: FieldTarget::Consonance,
            sampling: FieldSampling::Density,
            min_freq: space.freq_of_index(idx_min),
            max_freq: space.freq_of_index(idx_max),
            min_dist_erb: 0.0,
            tension: 0.0,
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
        let strategy = SpawnStrategy::Field {
            target: FieldTarget::Consonance,
            sampling: FieldSampling::Density,
            min_freq: space.freq_of_index(idx_min),
            max_freq: space.freq_of_index(idx_max),
            min_dist_erb: 1e-4,
            tension: 0.0,
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
        let strategy = SpawnStrategy::Field {
            target: FieldTarget::Consonance,
            sampling: FieldSampling::Density,
            min_freq: space.freq_of_index(idx_min),
            max_freq: space.freq_of_index(idx_max),
            min_dist_erb: 1e-4,
            tension: 0.0,
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
        let strategy = SpawnStrategy::Field {
            target: FieldTarget::Consonance,
            sampling: FieldSampling::Density,
            min_freq: space.freq_of_index(idx_min),
            max_freq: space.freq_of_index(idx_max),
            min_dist_erb: 1e-4,
            tension: 0.0,
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
        let strategy = SpawnStrategy::Field {
            target: FieldTarget::Consonance,
            sampling: FieldSampling::Density,
            // Intentionally reversed order to emulate Rhai-side input mistakes.
            min_freq: space.freq_of_index(idx_high),
            max_freq: space.freq_of_index(idx_low),
            min_dist_erb: 0.0,
            tension: 0.0,
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
    fn update_applies_to_population_members() {
        let mut pop = test_pop();
        let landscape = LandscapeFrame::default();
        pop.apply_action(
            Action::Spawn {
                population_id: 1,
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
            Action::UpdatePopulation {
                population_id: 1,
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
    fn release_marks_population_members() {
        let mut pop = test_pop();
        let landscape = LandscapeFrame::default();
        pop.apply_action(
            Action::Spawn {
                population_id: 1,
                ids: vec![21, 22],
                spec: spawn_spec_with_freq(220.0),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::ReleasePopulation {
                population_id: 1,
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
                population_id: 6,
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
    fn reserved_scenario_ids_are_not_reused_by_runtime_spawns() {
        let mut pop = test_pop();
        pop.reserve_runtime_ids_through(12);

        let id = pop.allocate_runtime_id();

        assert_eq!(id, 13);
    }

    #[test]
    fn respawn_none_keeps_current_behavior() {
        let mut pop = test_pop();
        pop.set_seed(7);
        let landscape = runtime_landscape();
        pop.apply_action(
            Action::Spawn {
                population_id: 7,
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
                population_id: 8,
                ids: vec![10],
                spec: decay_spawn_spec_with_freq(220.0, 0.02),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                population_id: 8,
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
    fn respawn_capacity_limits_living_members() {
        let mut pop = test_pop();
        pop.set_seed(12);
        let landscape = runtime_landscape();
        pop.apply_action(
            Action::Spawn {
                population_id: 82,
                ids: vec![820, 821],
                spec: sustain_spawn_spec_with_freq(220.0),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                population_id: 82,
                policy: RespawnPolicy::Random,
                settle_strategy: None,
                capacity: 1,
                min_c_level: None,
                background_death_rate_per_sec: 0.0,
            },
            &landscape,
            None,
        );

        force_dead(&mut pop, 820);
        pop.cleanup_dead(0, 0.01, false, &landscape);

        assert_eq!(pop.voices.len(), 1);
        assert_eq!(pop.voices[0].id(), 821);
    }

    #[test]
    fn background_turnover_replaces_member_via_respawn() {
        let mut pop = test_pop();
        pop.set_seed(61);
        let landscape = runtime_landscape();
        pop.apply_action(
            Action::Spawn {
                population_id: 81,
                ids: vec![810],
                spec: sustain_spawn_spec_with_freq(220.0),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                population_id: 81,
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
                population_id: 9,
                ids: vec![20],
                spec: decay_spawn_spec_with_freq(330.0, 0.02),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                population_id: 9,
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
                population_id: 90,
                ids: vec![900, 901],
                spec,
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                population_id: 90,
                policy: RespawnPolicy::Hereditary { sigma_oct: 0.002 },
                settle_strategy: None,
                capacity: 2,
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
                population_id: 10,
                ids: vec![30],
                spec: decay_spawn_spec_with_freq(220.0, 0.02),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                population_id: 10,
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
                .find(|v| v.metadata.population_id == 10 && v.id() != 30)
                .map(|v| v.id())
            {
                respawned_id = Some(id);
                break;
            }
        }
        let respawned_id = respawned_id.expect("respawned member should exist");

        pop.apply_action(
            Action::ReleasePopulation {
                population_id: 10,
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
                population_id: 11,
                ids: vec![40],
                spec: decay_spawn_spec_with_freq(220.0, 0.02),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                population_id: 11,
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
                .find(|v| v.metadata.population_id == 11 && v.id() != 40)
                .map(|v| v.id())
            {
                respawned_id = Some(id);
                break;
            }
        }
        let respawned_id = respawned_id.expect("respawned member should exist");

        pop.apply_action(
            Action::UpdatePopulation {
                population_id: 11,
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
                population_id: 91,
                ids: vec![910, 911],
                spec: decay_spawn_spec_with_freq(220.0, 0.02),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                population_id: 91,
                policy: RespawnPolicy::Random,
                settle_strategy: None,
                capacity: 2,
                min_c_level: None,
                background_death_rate_per_sec: 0.0,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::UpdatePopulation {
                population_id: 91,
                patch: ControlUpdate {
                    landscape_weight: Some(0.73),
                    ..ControlUpdate::default()
                },
            },
            &landscape,
            None,
        );

        for member in pop.voices.iter().filter(|v| v.metadata.population_id == 91) {
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
                population_id: 92,
                ids: vec![920],
                spec: decay_spawn_spec_with_freq(220.0, 0.02),
                strategy: None,
            },
            &landscape,
            None,
        );
        pop.apply_action(
            Action::SetRespawnPolicy {
                population_id: 92,
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
            Action::ReleasePopulation {
                population_id: 92,
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
            "released population should drain without repopulation"
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
                population_id: 12,
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
                population_id: 12,
                policy: RespawnPolicy::Hereditary { sigma_oct: 0.005 },
                settle_strategy: None,
                capacity: 2,
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
                population_id: 13,
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
                population_id: 13,
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
    fn random_respawn_selection_uses_weighted_scene_scores() {
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
                population_id: 1,
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
    fn kuramoto_order_parameter_uses_relative_phase_offset() {
        let mut pop = test_pop();
        let landscape = LandscapeFrame::default();
        pop.apply_action(
            Action::Spawn {
                population_id: 1,
                ids: vec![1, 2, 3],
                spec: spawn_spec_with_freq(220.0),
                strategy: None,
            },
            &landscape,
            None,
        );
        let shared_phase = 0.75;
        for (idx, voice) in pop.voices.iter_mut().enumerate() {
            let AnyArticulationCore::Entrain(core) = &mut voice.articulation.core else {
                panic!("expected entrain core");
            };
            core.phase_offset = idx as f32 * 2.0;
            core.rhythm_phase = shared_phase + core.phase_offset;
        }

        let (order, count) = pop.kuramoto_order_parameter().expect("order");

        assert_eq!(count, 3);
        assert!(order > 0.99);
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
        let mut landscape_high = LandscapeFrame::new(space);
        landscape_high.consonance_field_level.fill(1.0);

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
