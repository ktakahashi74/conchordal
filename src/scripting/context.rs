use super::*;

// Change only after author adoption of the bundled audition.
const DEFAULT_LISTENER_PROFILE: &str = "sensory";

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum PopulationStatus {
    Live,
    ReleasePending,
    Released,
}

#[derive(Clone, Debug)]
pub(super) struct PopulationState {
    pub(super) id: u64,
    pub(super) spec: PopulationSpec,
    pub(super) respawn_policy: RespawnPolicy,
    pub(super) crowding_target_same: bool,
    pub(super) crowding_target_other: bool,
    pub(super) status: PopulationStatus,
    pub(super) live_ids: Vec<u64>,
    pub(super) pending_patch: ControlUpdate,
    pub(super) pending_crowding_target: Option<(bool, bool)>,
    pub(super) pending_release: bool,
}

#[derive(Clone, Debug)]
struct FloraState {
    spec: FloraSpec,
    status: PopulationStatus,
    pending_patch: Option<FloraPatch>,
}

#[derive(Clone, Debug)]
pub struct ScriptWarning {
    pub message: String,
    pub position: Position,
}

pub(super) type ScriptWarnings = Arc<Mutex<Vec<ScriptWarning>>>;

#[derive(Debug, Clone)]
pub(super) struct ScopeFrame {
    pub(super) created_populations: Vec<u64>,
    created_flora: Vec<u64>,
}

#[derive(Debug, Clone)]
pub struct ScriptContext {
    pub(super) expression_sections: Vec<(crate::scenario::expression::Expression, Position)>,
    pub(super) listener_profile: &'static str,
    pub(super) profile_section: Option<crate::scenario::expression::Expression>,
    pub(super) explicit_expression: bool,
    pub cursor: f32,
    pub scenario: Scenario,
    pub seed: u64,
    pub next_event_order: u64,
    pub(super) next_population_id: u64,
    next_flora_id: u64,
    flora: BTreeMap<u64, FloraState>,
    pub(super) next_voice_id: u64,
    pub(super) populations: BTreeMap<u64, PopulationState>,
    pub(super) scopes: Vec<ScopeFrame>,
    pub(super) temporal_mode_declared: bool,
    pub warnings: ScriptWarnings,
}

impl Default for ScriptContext {
    fn default() -> Self {
        Self::with_seed(random::<u64>())
    }
}

impl ScriptContext {
    pub(super) fn warn(&self, message: String, position: Position) {
        self.warnings
            .lock()
            .expect("lock script warnings")
            .push(ScriptWarning { message, position });
    }

    /// A fresh context seeded with `seed`. Scripts still override this via a
    /// `seed()` call, which wins because it runs during evaluation.
    pub(super) fn with_seed(seed: u64) -> Self {
        Self {
            listener_profile: DEFAULT_LISTENER_PROFILE,
            profile_section: None,
            explicit_expression: false,
            cursor: 0.0,
            expression_sections: Vec::new(),
            scenario: Scenario {
                expression: None,
                expression_cursor: None,
                seed,
                output_gain_db: 0.0,
                output_gain_position: None,
                temporal_mode: crate::scenario::TemporalMode::Off,
                control_update_mode: ControlUpdateMode::SnapshotPhased,
                scaffold: ScaffoldConfig::Off,
                meter_shaping: MeterShaping::default(),
                scene_markers: Vec::new(),
                events: Vec::new(),
                duration_sec: 0.0,
            },
            seed,
            next_event_order: 1,
            next_population_id: 1,
            next_flora_id: 1,
            flora: BTreeMap::new(),
            next_voice_id: 1,
            populations: BTreeMap::new(),
            scopes: Vec::new(),
            temporal_mode_declared: false,
            warnings: Arc::default(),
        }
    }

    pub(super) fn place_flora(
        &mut self,
        spec: FloraSpecHandle,
        placement: Placement,
        position: Position,
    ) -> Result<FloraHandle, Box<EvalAltResult>> {
        let error =
            |message: String| Box::new(EvalAltResult::ErrorRuntime(message.into(), position));
        let (target, sampling, min_freq, max_freq, min_dist_erb, tension) = match placement.strategy
        {
            Some(SpawnStrategy::Field {
                target,
                sampling,
                min_freq,
                max_freq,
                min_dist_erb,
                tension,
            }) => (target, sampling, min_freq, max_freq, min_dist_erb, tension),
            Some(SpawnStrategy::Linear { .. }) => {
                return Err(error("Flora rejects line placement".into()));
            }
            None => return Err(error("Flora rejects at placement".into())),
        };
        let (seeding, default_spacing) = match target {
            FieldTarget::Consonance => (FloraSeeding::Consonance, 1.0),
            FieldTarget::Uniform => (FloraSeeding::Uniform, 0.0),
            other => return Err(error(format!("Flora rejects {other:?} placement target"))),
        };
        if placement.count != 1 {
            return Err(error("Flora placement count must be 1".into()));
        }
        if sampling != FieldSampling::Density {
            return Err(error("Flora rejects peak sampling; use density".into()));
        }
        if tension != 0.0 {
            return Err(error("Flora placement tension must be 0".into()));
        }
        if min_dist_erb != default_spacing {
            return Err(error(format!(
                "Flora placement spacing must be {default_spacing}"
            )));
        }
        if [min_freq, max_freq]
            .iter()
            .any(|freq| !(MIN_FREQ_HZ..=MAX_FREQ_HZ).contains(freq))
        {
            return Err(error(format!(
                "field placement bounds must be finite and within {MIN_FREQ_HZ}..={MAX_FREQ_HZ} Hz"
            )));
        }
        // Scripts have no runtime space; use the initial grid's 96 bins/octave.
        if (f64::from(max_freq) / f64::from(min_freq)).log2() < 1.0 / 96.0 {
            return Err(error(
                "Flora extent must span at least one Log2Space bin (1/96 octave)".into(),
            ));
        }
        let id = self.next_flora_id;
        self.next_flora_id = self.next_flora_id.wrapping_add(1);
        self.flora.insert(
            id,
            FloraState {
                spec: spec.spec,
                status: PopulationStatus::Live,
                pending_patch: None,
            },
        );
        if let Some(scope) = self.scopes.last_mut() {
            scope.created_flora.push(id);
        }
        self.push_event(
            self.cursor,
            vec![Action::PlaceFlora {
                flora_id: id,
                spec: spec.spec,
                extent_hz: (min_freq, max_freq),
                seeding,
            }],
        );
        Ok(FloraHandle { id })
    }

    pub(super) fn release_flora(&mut self, flora_id: u64, position: Position) {
        let Some(flora) = self.flora.get_mut(&flora_id) else {
            self.warn(format!("release on unknown Flora {flora_id}"), position);
            return;
        };
        match flora.status {
            PopulationStatus::Live => flora.status = PopulationStatus::ReleasePending,
            PopulationStatus::ReleasePending => {
                self.warn(
                    format!("release ignored for Flora {flora_id}: release already pending"),
                    position,
                );
            }
            PopulationStatus::Released => self.warn(
                format!("release ignored for inactive Flora {flora_id}"),
                position,
            ),
        }
    }

    pub(super) fn push_event(&mut self, time_sec: f32, actions: Vec<Action>) {
        if actions.is_empty() {
            return;
        }
        let order = self.next_event_order;
        self.next_event_order += 1;
        self.scenario.events.push(TimedEvent {
            time: time_sec,
            order,
            actions,
        });
    }

    pub(super) fn push_scene_marker(&mut self, name: &str) {
        let order = self.next_event_order;
        self.next_event_order += 1;
        self.scenario.scene_markers.push(SceneMarker {
            name: name.to_string(),
            time: self.cursor,
            order,
        });
    }

    pub(super) fn push_scope(&mut self) {
        self.scopes.push(ScopeFrame {
            created_populations: Vec::new(),
            created_flora: Vec::new(),
        });
    }

    pub(super) fn pop_scope(&mut self) {
        let Some(scope) = self.scopes.pop() else {
            return;
        };
        let mut updates = Vec::new();
        let mut crowding_updates = Vec::new();
        let mut releases = Vec::new();
        for population_id in scope.created_populations {
            let Some(population) = self.populations.get_mut(&population_id) else {
                continue;
            };
            match population.status {
                PopulationStatus::Live | PopulationStatus::ReleasePending => {
                    if population.pending_patch != ControlUpdate::default() {
                        updates.push(Action::UpdatePopulation {
                            population_id,
                            patch: population.pending_patch.clone(),
                        });
                    }
                    if let Some((same_population_visible, other_population_visible)) =
                        population.pending_crowding_target
                    {
                        crowding_updates.push(Action::SetPopulationCrowdingTarget {
                            population_id,
                            same_population_visible,
                            other_population_visible,
                        });
                    }
                    if !population.live_ids.is_empty() {
                        releases.push(Action::ReleasePopulation {
                            population_id,
                            fade_sec: population.spec.release_sec(),
                        });
                    }
                    population.pending_patch = ControlUpdate::default();
                    population.pending_crowding_target = None;
                    population.pending_release = false;
                    population.status = PopulationStatus::Released;
                }
                PopulationStatus::Released => {}
            }
        }
        for flora_id in scope.created_flora {
            let flora = self.flora.get_mut(&flora_id).expect("scoped Flora");
            if flora.status == PopulationStatus::Released {
                continue;
            }
            if let Some(patch) = flora.pending_patch.take() {
                updates.push(Action::UpdateFlora { flora_id, patch });
            }
            releases.push(Action::ReleaseFlora {
                flora_id,
                fade_sec: flora.spec.fade_sec,
            });
            flora.status = PopulationStatus::Released;
        }
        if !updates.is_empty() || !crowding_updates.is_empty() || !releases.is_empty() {
            updates.extend(crowding_updates);
            updates.extend(releases);
            self.push_event(self.cursor, updates);
        }
    }

    pub(super) fn finalize_duration(&mut self) -> f32 {
        let mut duration = self.cursor.max(0.0);
        if duration <= 0.0 {
            duration = 0.01;
            self.cursor = duration;
        }
        self.scenario.duration_sec = duration;
        duration
    }

    pub(super) fn max_release_tail(&self) -> f32 {
        let mut max_end: f32 = 0.0;
        for event in &self.scenario.events {
            for action in &event.actions {
                if let Action::ReleasePopulation { fade_sec, .. }
                | Action::ReleaseFlora { fade_sec, .. } = action
                {
                    let fade = fade_sec.max(0.0);
                    max_end = max_end.max(event.time + fade);
                }
            }
        }
        max_end
    }

    pub(super) fn emit_pending_actions(&mut self) {
        let mut update_actions = Vec::new();
        let mut crowding_target_actions = Vec::new();
        for population in self.populations.values_mut() {
            if !matches!(
                population.status,
                PopulationStatus::Live | PopulationStatus::ReleasePending
            ) {
                continue;
            }
            if population.pending_patch != ControlUpdate::default() {
                if !population.live_ids.is_empty() {
                    update_actions.push(Action::UpdatePopulation {
                        population_id: population.id,
                        patch: population.pending_patch.clone(),
                    });
                }
                population.pending_patch = ControlUpdate::default();
            }
            if let Some((same_population_visible, other_population_visible)) =
                population.pending_crowding_target
            {
                crowding_target_actions.push(Action::SetPopulationCrowdingTarget {
                    population_id: population.id,
                    same_population_visible,
                    other_population_visible,
                });
                population.pending_crowding_target = None;
            }
        }

        let mut release_actions = Vec::new();
        for population in self.populations.values_mut() {
            if matches!(population.status, PopulationStatus::ReleasePending)
                && population.pending_release
            {
                if !population.live_ids.is_empty() {
                    release_actions.push(Action::ReleasePopulation {
                        population_id: population.id,
                        fade_sec: population.spec.release_sec(),
                    });
                }
                population.pending_release = false;
                population.status = PopulationStatus::Released;
            }
        }

        for (&flora_id, flora) in &mut self.flora {
            if flora.status == PopulationStatus::Released {
                continue;
            }
            if let Some(patch) = flora.pending_patch.take() {
                update_actions.push(Action::UpdateFlora { flora_id, patch });
            }
            if flora.status == PopulationStatus::ReleasePending {
                release_actions.push(Action::ReleaseFlora {
                    flora_id,
                    fade_sec: flora.spec.fade_sec,
                });
                flora.status = PopulationStatus::Released;
            }
        }

        if !update_actions.is_empty()
            || !crowding_target_actions.is_empty()
            || !release_actions.is_empty()
        {
            let mut actions = Vec::with_capacity(
                update_actions.len() + crowding_target_actions.len() + release_actions.len(),
            );
            actions.extend(update_actions);
            actions.extend(crowding_target_actions);
            actions.extend(release_actions);
            self.push_event(self.cursor, actions);
        }
    }

    pub(super) fn finish(&mut self) {
        self.emit_pending_actions();
        // Listener, metric founders and Flora read the scene's final basin.
        if let Some((a, b)) = self.scenario.meter_shaping.basin_hz {
            let (lo, hi) = sane_basin(a, b);
            let rate = 0.5 * (lo + hi);
            for event in &mut self.scenario.events {
                for action in &mut event.actions {
                    if let Action::Spawn { spec, .. } = action
                        && let PhonationTiming::Coupled(timing) =
                            &mut spec.control.phonation.spec.timing
                        && timing.relation == RhythmRelation::Synchronized
                    {
                        timing.base_rate_hz = rate;
                    }
                }
            }
        }
        let mut end_time = self.finalize_duration();
        if !self.expression_sections.is_empty() {
            let (expression, position) = self.expression_sections[0];
            let result = if self.expression_sections.len() != 1 {
                Err("only one nonzero expressive section is supported")
            } else if f64::from(self.cursor) > expression.end {
                Err("requires the final section, without a later score cursor")
            } else {
                expression.validate(&self.scenario)
            };
            match result {
                Ok(()) => {
                    self.scenario.expression = Some(expression);
                    self.scenario.expression_cursor = Some(self.cursor);
                }
                Err(reason) => self.warn(format!("expressive_section ignored: {reason}"), position),
            }
        }
        if self.listener_profile == "tonal"
            && !self.explicit_expression
            && let Some(expression) = self.profile_section
            && expression.end == f64::from(self.cursor)
            && expression.validate(&self.scenario).is_ok()
        {
            self.scenario.expression = Some(expression);
            self.scenario.expression_cursor = Some(self.cursor);
        }
        let release_tail = self.max_release_tail();
        if release_tail > end_time {
            end_time = release_tail;
            self.scenario.duration_sec = end_time;
            self.cursor = end_time;
        }
        self.push_event(end_time, vec![Action::Finish]);
    }

    pub(super) fn create_population(
        &mut self,
        mut population_spec: PopulationSpecHandle,
        count: i64,
        strategy: Option<SpawnStrategy>,
        position: Position,
    ) -> Result<PopulationHandle, Box<EvalAltResult>> {
        if count <= 0 {
            return Err(Box::new(EvalAltResult::ErrorRuntime(
                "population count must be >= 1".into(),
                position,
            )));
        }
        for strategy in [
            strategy.as_ref(),
            population_spec.spec.respawn_settle_strategy.as_ref(),
        ]
        .into_iter()
        .flatten()
        {
            if let SpawnStrategy::Field {
                min_freq, max_freq, ..
            } = strategy
                && [min_freq, max_freq]
                    .iter()
                    .any(|&&freq| !(MIN_FREQ_HZ..=MAX_FREQ_HZ).contains(&freq))
            {
                return Err(Box::new(EvalAltResult::ErrorRuntime(
                    format!(
                        "field placement bounds must be finite and within {MIN_FREQ_HZ}..={MAX_FREQ_HZ} Hz"
                    )
                    .into(),
                    position,
                )));
            }
        }
        population_spec.spec.diagnostics = self.warnings.clone();
        let specified = population_spec.spec.specified.clone();
        let warn_at = |label: &str, message: &str| {
            self.warn(
                message.to_string(),
                specified.get(label).copied().unwrap_or(position),
            );
        };
        if !matches!(population_spec.spec.brain, BrainKind::Entrain) {
            for label in [
                "endurance",
                "recovery",
                "consonance_viability",
                "viability_scope",
                "dissonance_penalty",
                "attack_cost_fraction",
                "attack_recharge_fraction",
                "rhythm_freq",
                "rhythm_reward",
                "rhythm_coupling_vitality",
            ] {
                if specified.contains_key(label) {
                    warn_at(
                        label,
                        &format!("{label}() ignored by brain(\"seq\")/brain(\"drone\")"),
                    );
                }
            }
        } else if population_spec.spec.rhythm_reward_user_enabled
            && population_spec.spec.rhythm_reward.is_none()
        {
            warn_at(
                "rhythm_reward",
                "rhythm_reward() cleared by a later timing preset; specify it after the preset",
            );
        } else if specified.contains_key("rhythm_reward")
            && population_spec.spec.rhythm_reward_user_enabled
            && (!matches!(
                population_spec.spec.phonation_spec.timing,
                PhonationTiming::Once
            ) || !matches!(
                population_spec.spec.phonation_spec.duration,
                DurationSpec::WhileAlive
            ) || population_spec
                .spec
                .spawn_spec()
                .control
                .pitch
                .pitch_apply_mode
                == PitchApplyMode::Rearticulate)
        {
            warn_at(
                "rhythm_reward",
                "rhythm_reward() only affects autonomous attacks in sustained Hold phonation; ignored here",
            );
        }
        let count = count as usize;
        if let Some(capacity) = population_spec.spec.respawn_capacity
            && count > capacity
        {
            return Err(Box::new(EvalAltResult::ErrorRuntime(
                "respawn_capacity must be >= founder count".into(),
                position,
            )));
        }
        if let Some(rate) = population_spec.spec.intrinsic_rate.take() {
            if let PhonationTiming::Coupled(timing) =
                &mut population_spec.spec.phonation_spec.timing
                && timing.relation == RhythmRelation::Participating
            {
                let max = crate::life::phonation_engine::MAX_ONSET_RATE_HZ as FLOAT;
                if !rate.is_finite() || !(0.01..=max).contains(&rate) {
                    return Err(Box::new(EvalAltResult::ErrorRuntime(
                        format!(
                            "intrinsic_rate(hz) requires a finite value within 0.01..={max} Hz; these are execution protection limits, not a musically recommended range"
                        )
                        .into(),
                        position,
                    )));
                }
                timing.base_rate_hz = rate as f32;
            } else {
                warn_at(
                    "intrinsic_rate",
                    "intrinsic_rate() ignored outside entrained()/steady()/flow()",
                );
            }
        }
        population_spec
            .spec
            .spawn_spec()
            .control
            .validate_pitch_application()
            .map_err(|message| Box::new(EvalAltResult::ErrorRuntime(message.into(), position)))?;
        if population_spec
            .spec
            .spawn_spec()
            .control
            .pitch
            .pitch_apply_mode
            != PitchApplyMode::Rearticulate
            && specified.contains_key("rearticulate_above")
        {
            warn_at(
                "rearticulate_above",
                "rearticulate_above() ignored outside pitch_apply_mode(\"rearticulate\")",
            );
            population_spec.spec.control.pitch.rearticulate_above_cents = 0.0;
        }
        if specified.contains_key("articulation_persistence")
            && !population_spec
                .spec
                .spawn_spec()
                .control
                .supports_articulation_persistence()
        {
            warn_at(
                "articulation_persistence",
                "articulation_persistence() ignored outside free-pitch sustain() with metric()/entrained()/steady()/flow(), rearticulate, and a positive rearticulate_above()",
            );
            population_spec
                .spec
                .control
                .phonation
                .articulation_persistence = 0.0;
        }
        if population_spec.spec.control.phonation.phrase_boundary > 0.0
            && !population_spec
                .spec
                .spawn_spec()
                .control
                .supports_phrase_boundary()
        {
            warn_at(
                "phrase_boundary",
                "phrase_boundary() requires Sine, sustained metric/entrained/steady/flow and discrete pitch; ignored",
            );
            population_spec.spec.control.phonation.phrase_boundary = 0.0;
        }
        if population_spec.spec.control.pitch.follow_own_sequence > 0.0
            && (!population_spec.spec.consonance_movement
                || !population_spec
                    .spec
                    .spawn_spec()
                    .control
                    .supports_own_sequence())
        {
            warn_at(
                "follow_own_sequence",
                "follow_own_sequence() requires free seek_consonance(), stepwise sustained participation, no glide or articulation persistence; ignored",
            );
            population_spec.spec.control.pitch.follow_own_sequence = 0.0;
        }
        if population_spec.spec.control.pitch.follow_own_tonality > 0.0
            && (!population_spec.spec.consonance_movement
                || !population_spec
                    .spec
                    .spawn_spec()
                    .control
                    .supports_own_sequence())
        {
            warn_at(
                "follow_own_tonality",
                "follow_own_tonality() requires free seek_consonance(), stepwise sustained participation, no glide or articulation persistence; ignored",
            );
            population_spec.spec.control.pitch.follow_own_tonality = 0.0;
        }
        if population_spec.spec.control.pitch.dwell > 0.0
            && (!population_spec.spec.consonance_movement
                || population_spec.spec.control.pitch.mode != PitchMode::Free)
        {
            warn_at(
                "dwell",
                "dwell() requires free pitch with seek_consonance(); ignored",
            );
            population_spec.spec.control.pitch.dwell = 0.0;
        }
        let weight = population_spec.spec.control.phonation.rest_on_habituation;
        if !weight.is_finite()
            || weight < 0.0
            || (weight > 0.0
                && !population_spec
                    .spec
                    .spawn_spec()
                    .control
                    .phonation
                    .spec
                    .is_sustained_participation())
        {
            warn_at(
                "rest_on_habituation",
                "rest_on_habituation() requires a finite nonnegative f32 weight and explicit sustain() with metric()/entrained()/steady()/flow(); ignored",
            );
            population_spec.spec.control.phonation.rest_on_habituation = 0.0;
        }
        if population_spec.spec.tie_user_set
            && !population_spec.spec.spawn_spec().control.supports_tie()
        {
            warn_at(
                "tie",
                "tie() ignored outside free-pitch sustain() with metric()/entrained()/steady()/flow() and gate_snap/rearticulate",
            );
            population_spec.spec.control.phonation.tie = 0.0;
        }
        // Resolve the preset after final-form validation; explicit zero also wins.
        if self.listener_profile == "tonal" {
            let control = population_spec.spec.spawn_spec().control;
            if control.supports_phrase_boundary() && !specified.contains_key("phrase_boundary") {
                population_spec.spec.control.phonation.phrase_boundary = 1.0;
            }
            if population_spec.spec.consonance_movement && control.supports_own_sequence() {
                if !specified.contains_key("follow_own_sequence") {
                    population_spec.spec.control.pitch.follow_own_sequence = 0.05;
                }
                if !specified.contains_key("follow_own_tonality") {
                    population_spec.spec.control.pitch.follow_own_tonality = 0.10;
                }
            }
        }
        let id = self.next_population_id;
        self.next_population_id = self.next_population_id.wrapping_add(1);
        let mut ids = Vec::with_capacity(count);
        for _ in 0..count {
            let voice_id = self.next_voice_id;
            self.next_voice_id = self.next_voice_id.wrapping_add(1);
            ids.push(voice_id);
        }
        let population = PopulationState {
            id,
            respawn_policy: population_spec.spec.respawn_policy,
            crowding_target_same: population_spec.spec.crowding_target_same,
            crowding_target_other: population_spec.spec.crowding_target_other,
            spec: population_spec.spec,
            status: PopulationStatus::Live,
            live_ids: ids.clone(),
            pending_patch: ControlUpdate::default(),
            pending_crowding_target: None,
            pending_release: false,
        };
        self.populations.insert(id, population);
        if let Some(scope) = self.scopes.last_mut() {
            scope.created_populations.push(id);
        }
        if !ids.is_empty() {
            let population = self.populations.get(&id).expect("inserted population");
            let mut actions = vec![
                Action::Spawn {
                    population_id: id,
                    ids,
                    spec: population.spec.spawn_spec(),
                    strategy,
                },
                Action::SetPopulationCrowdingTarget {
                    population_id: id,
                    same_population_visible: population.crowding_target_same,
                    other_population_visible: population.crowding_target_other,
                },
            ];
            if !matches!(population.respawn_policy, RespawnPolicy::None) {
                actions.push(Action::SetRespawnPolicy {
                    population_id: id,
                    policy: population.respawn_policy,
                    settle_strategy: population.spec.respawn_settle_strategy.clone(),
                    capacity: population.spec.respawn_capacity.unwrap_or(count).max(1),
                    min_c_level: population.spec.respawn_min_c_level,
                    background_death_rate_per_sec: population
                        .spec
                        .respawn_background_death_rate_per_sec,
                });
            }
            self.push_event(self.cursor, actions);
        }
        Ok(PopulationHandle { id })
    }

    pub(super) fn place_population_spec(
        &mut self,
        mut population_spec: PopulationSpecHandle,
        mut placement: Placement,
        position: Position,
    ) -> Result<PopulationHandle, Box<EvalAltResult>> {
        if let Some(freq_hz) = placement.freq_hz {
            if population_spec.spec.consonance_movement
                && population_spec.spec.control.pitch.mode == PitchMode::Free
            {
                // Explicit movers start here; their final pitch policy stays intact.
                placement = Placement::line(freq_hz, freq_hz).with_count(placement.count);
            } else {
                population_spec.spec.set_freq(freq_hz);
            }
        }
        self.create_population(
            population_spec,
            placement.count,
            placement.strategy,
            position,
        )
    }

    pub(super) fn set_output_gain(
        &mut self,
        db: f64,
        position: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        let db_f32 = db as f32;
        let gain = 10.0f32.powf(db_f32 / 20.0);
        if !db.is_finite() || !db_f32.is_finite() || !gain.is_finite() || gain <= 0.0 {
            return Err(Box::new(EvalAltResult::ErrorRuntime(
                "output_gain must be finite and yield a finite, positive f32 amplitude gain".into(),
                position,
            )));
        }
        self.scenario.output_gain_db = db_f32;
        self.scenario.output_gain_position = position.line().zip(position.position());
        Ok(())
    }

    pub(super) fn set_seed(
        &mut self,
        seed: i64,
        position: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        if seed < 0 {
            return Err(Box::new(EvalAltResult::ErrorRuntime(
                "seed must be >= 0".into(),
                position,
            )));
        }
        let seed = seed as u64;
        self.seed = seed;
        self.scenario.seed = seed;
        Ok(())
    }

    pub(super) fn wait(&mut self, sec: f32) {
        self.emit_pending_actions();
        self.cursor += sec.max(0.0);
    }

    pub(super) fn flush(&mut self) {
        self.emit_pending_actions();
    }

    pub(super) fn release_population(&mut self, population_id: u64, position: Position) {
        let Some(population) = self.populations.get_mut(&population_id) else {
            self.warn(
                format!("release on unknown population {population_id}"),
                position,
            );
            return;
        };
        match population.status {
            PopulationStatus::Live => {
                population.pending_release = true;
                population.status = PopulationStatus::ReleasePending;
            }
            PopulationStatus::ReleasePending => {
                self.warn(
                    format!(
                        "release ignored for population {population_id}: release already pending"
                    ),
                    position,
                );
            }
            PopulationStatus::Released => {
                self.warn(
                    format!("release ignored for inactive population {population_id}"),
                    position,
                );
            }
        }
    }

    pub(super) fn warn_population_inactive(
        &self,
        population_id: u64,
        label: &str,
        position: Position,
    ) {
        self.warn(
            format!("{label} ignored for inactive population {population_id}"),
            position,
        );
    }
}

pub(super) fn update_flora_numeric(
    ctx: &mut ScriptContext,
    handle: FloraHandle,
    name: &str,
    value: FLOAT,
    position: Position,
) -> Result<FloraHandle, Box<EvalAltResult>> {
    let value = validate_flora_numeric(name, value, position, &ctx.warnings)?;
    let Some(flora) = ctx.flora.get_mut(&handle.id) else {
        ctx.warn(
            format!("{name} ignored for unknown Flora {}", handle.id),
            position,
        );
        return Ok(handle);
    };
    if flora.status == PopulationStatus::Live {
        let patch = flora.pending_patch.get_or_insert_with(FloraPatch::default);
        match name {
            "pace" => patch.pace = Some(value),
            "amp" => patch.amp = Some(value),
            _ => unreachable!("registered Flora patch parameter"),
        }
    } else {
        ctx.warn(
            format!("{name} ignored for inactive Flora {}", handle.id),
            position,
        );
    }
    Ok(handle)
}

pub(super) fn update_flora_pitch_apply_mode(
    ctx: &mut ScriptContext,
    handle: FloraHandle,
    name: &str,
    position: Position,
) -> FloraHandle {
    let Some(mode) = super::spec::flora_pitch_apply_mode(name, position, &ctx.warnings) else {
        return handle;
    };
    let Some(flora) = ctx.flora.get_mut(&handle.id) else {
        ctx.warn(
            format!("pitch_apply_mode ignored for unknown Flora {}", handle.id),
            position,
        );
        return handle;
    };
    if flora.status == PopulationStatus::Live {
        flora
            .pending_patch
            .get_or_insert_with(FloraPatch::default)
            .pitch_apply_mode = Some(mode);
    } else {
        ctx.warn(
            format!("pitch_apply_mode ignored for inactive Flora {}", handle.id),
            position,
        );
    }
    handle
}
