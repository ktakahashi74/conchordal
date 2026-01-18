use super::control::PitchMode;
use super::individual::{AgentMetadata, Individual, PhonationBatch, SoundBody};
use super::scenario::{Action, IndividualConfig, SpawnStrategy};
use crate::core::landscape::{LandscapeFrame, LandscapeUpdate};
use crate::core::timebase::{Tick, Timebase};
use crate::life::social_density::SocialDensityTrace;
use crate::life::sound::{AudioCommand, VoiceTarget};
use crate::life::world_model::WorldModel;
use rand::{Rng, SeedableRng, distr::Distribution, distr::weighted::WeightedIndex, rngs::SmallRng};
use std::hash::{Hash, Hasher};
use tracing::{debug, info, warn};

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
        let pred_scan = world
            .predict_consonance01_next_gate()
            .and_then(|(gate_tick, scan)| {
                if gate_tick >= now && gate_tick < frame_end {
                    Some(scan)
                } else {
                    None
                }
            });
        let mut used = 0usize;
        let social_trace = self.social_trace.as_ref();
        for agent in &mut self.individuals {
            let social_coupling = agent.phonation_coupling;
            if used == out.len() {
                out.push(PhonationBatch::default());
            }
            let batch = &mut out[used];
            let extra_gate_gain = pred_scan
                .as_ref()
                .map(|scan| {
                    let gain_raw = world.sample_scan01(scan, agent.body.base_freq_hz());
                    let sync = agent.effective_control.phonation.sync;
                    mix_pred_gate_gain(sync, gain_raw)
                })
                .unwrap_or(1.0);
            agent.tick_phonation_into(
                &tb,
                now,
                &landscape.rhythm,
                social_trace,
                social_coupling,
                extra_gate_gain,
                batch,
            );
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
            SpawnStrategy::Harmonicity { .. } => {
                let mut best = idx_min;
                let mut best_val = f32::MIN;
                let mut found = false;
                for i in idx_min..=idx_max {
                    let f = space.freq_of_index(i);
                    if self.is_range_occupied_with(f, min_dist_erb, reserved) {
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
            SpawnStrategy::HarmonicDensity { .. } => {
                let weights: Vec<f32> = (idx_min..=idx_max)
                    .enumerate()
                    .map(|(local_idx, i)| {
                        let f = space.freq_of_index(i);
                        let occupied = self.is_range_occupied_with(f, min_dist_erb, reserved);
                        let _ = local_idx;
                        let c01 = landscape.consonance01.get(i).copied().unwrap_or(0.0);
                        harmonic_density_weight(c01, occupied)
                    })
                    .collect();
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
                for (member_idx, id) in ids.into_iter().enumerate() {
                    if self.individuals.iter().any(|agent| agent.id() == id) {
                        warn!("Spawn: id collision for {id} in group {group_id}");
                        continue;
                    }
                    let mut control = spec.control.clone();
                    if let Some(ref strat) = strategy {
                        let freq = match strat {
                            SpawnStrategy::Linear {
                                start_freq,
                                end_freq,
                            } => {
                                if total <= 1 {
                                    *start_freq
                                } else {
                                    let t = member_idx as f32 / (total - 1) as f32;
                                    start_freq + (end_freq - start_freq) * t
                                }
                            }
                            _ => self.decide_frequency(strat, landscape, &mut rng, &reserved),
                        };
                        control.pitch.freq = freq.max(1.0);
                        control.pitch.mode = PitchMode::Lock;
                    }
                    let metadata = AgentMetadata {
                        group_id,
                        member_idx,
                    };
                    let cfg = IndividualConfig {
                        control: control.clone(),
                        articulation: spec.articulation.clone(),
                    };
                    let spawned =
                        cfg.spawn(id, self.current_frame, metadata, self.time.fs, self.seed);
                    let body = spawned.body_snapshot();
                    let pitch_hz = spawned.body.base_freq_hz();
                    let amp = spawned.body.amp();
                    self.individuals.push(spawned);
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
                    reserved.push(control.pitch.freq);
                }
            }
            Action::Update {
                group_id,
                ids,
                update,
            } => {
                for id in ids {
                    if let Some(agent) = self.find_individual_mut(id) {
                        if let Err(err) = agent.apply_update(&update) {
                            warn!("Update: agent {id} (group {group_id}) rejected update: {err}");
                        }
                    } else {
                        warn!("Update: agent {id} not found for group {group_id}");
                    }
                }
            }
            Action::Release {
                group_id,
                ids,
                fade_sec,
            } => {
                let fade_sec = fade_sec.max(0.0);
                for id in ids {
                    if let Some(agent) = self.find_individual_mut(id) {
                        agent.start_remove_fade(fade_sec);
                    } else {
                        warn!("Release: agent {id} not found for group {group_id}");
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
            Action::PostNote { .. } => {}
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

    pub fn cleanup_dead(&mut self, current_frame: u64, dt_sec: f32, scenario_finished: bool) {
        self.current_frame = current_frame;
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
    }
}

fn mix_pred_gate_gain(sync: f32, gain_raw: f32) -> f32 {
    let sync = sync.clamp(0.0, 1.0);
    let gain01 = 0.2 + 0.8 * gain_raw.powf(2.0);
    let gain = 1.0 + (gain01 - 1.0) * sync;
    if gain.is_finite() { gain.max(0.0) } else { 1.0 }
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
    use crate::core::landscape::{Landscape, LandscapeFrame};
    use crate::core::log2space::Log2Space;
    use crate::core::timebase::Timebase;
    use crate::life::control::{AgentControl, ControlUpdate, PhonationType};
    use crate::life::individual::{AnyArticulationCore, ArticulationWrapper, DroneCore};
    use crate::life::note_event::BodySnapshot;
    use crate::life::phonation_engine::{OnsetEvent, PhonationCmd, PhonationKick};
    use crate::life::scenario::{ArticulationCoreConfig, SpawnSpec, SpawnStrategy};
    use crate::life::world_model::WorldModel;
    use rand::SeedableRng;

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

    fn spawn_spec_with_freq(freq: f32) -> SpawnSpec {
        let mut control = AgentControl::default();
        control.pitch.freq = freq;
        SpawnSpec {
            control,
            articulation: ArticulationCoreConfig::default(),
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
        let strategy = SpawnStrategy::Harmonicity {
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
    fn pred_gate_gain_sync_zero_is_unity() {
        let gain = mix_pred_gate_gain(0.0, 0.3);
        assert_eq!(gain, 1.0);
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
        let couplings = vec![0.0, 1.0];
        assert!(social_trace_enabled_from_couplings(couplings));
    }

    #[test]
    fn update_applies_to_ids() {
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
                ids: vec![10, 11],
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
    fn release_marks_targeted_ids() {
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
        assert_eq!(released, vec![21]);
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
