use super::{Community, RuntimePopulationState, SpawnParams, SpawnReason};
use crate::core::landscape::{LandscapeFrame, LandscapeUpdate};
use crate::scenario::control::MIN_FREQ_HZ;
use crate::scenario::{Action, RespawnPolicy, SpawnStrategy, VoiceSpec};
use rand::{SeedableRng, rngs::SmallRng};
use tracing::warn;

impl Community {
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

    fn on_update_population_action(
        &mut self,
        population_id: u64,
        patch: crate::scenario::control::ControlUpdate,
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

    fn apply_population_update(
        &mut self,
        population_id: u64,
        update: &crate::scenario::control::ControlUpdate,
    ) {
        if let Some(population) = self.populations.get_mut(&population_id) {
            population.template.control.apply_update(update);
        }
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
}

#[cfg(test)]
mod tests {
    use super::super::tests::{spawn_spec_with_freq, test_pop};
    use super::*;
    use crate::life::voice::SoundBody;
    use crate::scenario::control::{ControlUpdate, PitchMode};

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
}
