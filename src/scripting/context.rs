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
    pub(super) spec: PopulationSpec,
    pub(super) respawn_policy: RespawnPolicy,
    pub(super) loo_at_placement: bool,
    tail_roughness_warned: bool,
    pub(super) crowding_target_same: bool,
    pub(super) crowding_target_other: bool,
    pub(super) status: PopulationStatus,
    pub(super) spawned: bool,
    harmonic_supported: bool,
    depth: usize,
    pub(super) pending_patch: ControlUpdate,
    pub(super) pending_crowding_target: Option<(bool, bool)>,
    pub(super) pending_release: bool,
}

#[derive(Clone, Debug)]
struct FloraState {
    depth: usize,
    spec: FloraSpec,
    status: PopulationStatus,
    pending_patch: Option<FloraPatch>,
}

#[derive(Clone, Debug)]
struct RainState {
    depth: usize,
    fade_sec: f32,
    status: PopulationStatus,
    pending_patch: Option<RainPatch>,
}

#[derive(Clone, Debug)]
pub struct ScriptWarning {
    pub kind: &'static str,
    pub count: u64,
    pub message: String,
    pub position: Position,
}

pub(super) type ScriptWarnings = Arc<Mutex<super::diagnostics::WarningStore>>;

#[derive(Debug, Clone)]
pub(super) struct ScopeFrame {}

#[derive(Debug, Clone)]
pub struct ScriptContext {
    pub(super) tonal_observation_position: Option<Position>,
    pub(super) harmonic_observation_position: Option<Position>,
    #[cfg(feature = "native")]
    pub(super) stream: Option<Arc<score::ScoreStream>>,
    pub(super) emission_error: Option<ScriptError>,
    pub(super) emitted_any_event: bool,
    pub(super) harmonic_position: Position,
    pub(super) harmonic_sites: Vec<(u64, f64, &'static str, Position)>,
    pub(super) expression_sections: Vec<(crate::scenario::expression::Expression, Position)>,
    pub(super) listener_profile: &'static str,
    pub(super) profile_section: Option<crate::scenario::expression::Expression>,
    pub(super) explicit_expression: bool,
    pub cursor: f64,
    pub scenario: Scenario,
    pub seed: u64,
    pub next_event_order: u64,
    pub(super) next_population_id: u64,
    next_flora_id: u64,
    flora: BTreeMap<u64, FloraState>,
    next_rain_id: u64,
    rain: BTreeMap<u64, RainState>,
    pub(super) next_voice_id: u64,
    pub(super) populations: BTreeMap<u64, PopulationState>,
    pub(super) scopes: Vec<ScopeFrame>,
    pub(super) temporal_mode_declared: bool,
    pub(super) parallel_used: bool,
    pub(super) warnings: ScriptWarnings,
}

impl Default for ScriptContext {
    fn default() -> Self {
        #[cfg(feature = "native")]
        let seed = random::<u64>();
        #[cfg(not(feature = "native"))]
        let seed = 0;
        Self::with_seed(seed)
    }
}

impl ScriptContext {
    pub(super) fn warn(&self, kind: &'static str, message: String, position: Position) {
        self.warnings
            .lock()
            .expect("lock script warnings")
            .push(ScriptWarning {
                kind,
                count: 1,
                message,
                position,
            });
    }

    /// A fresh context seeded with `seed`. Scripts still override this via a
    /// `seed()` call, which wins because it runs during evaluation.
    pub(super) fn with_seed(seed: u64) -> Self {
        Self {
            tonal_observation_position: None,
            harmonic_observation_position: None,
            harmonic_position: Position::NONE,
            harmonic_sites: Vec::new(),
            listener_profile: DEFAULT_LISTENER_PROFILE,
            profile_section: None,
            explicit_expression: false,
            #[cfg(feature = "native")]
            stream: None,
            emission_error: None,
            emitted_any_event: false,
            cursor: 0.0,
            expression_sections: Vec::new(),
            scenario: Scenario {
                observe_tonal_context: false,
                harmonic_relation_sources: None,
                expression: None,
                expression_cursor: None,
                seed,
                output_gain_db: 0.0,
                output_gain_position: None,
                temporal_mode: crate::scenario::TemporalMode::Off,
                control_update_mode: ControlUpdateMode::SnapshotPhased,
                scaffold: ScaffoldConfig::Off,
                harmonic_rhythm: 0,
                remember_pitch_set: 0.,
                pitch_set_position: None,
                meter_shaping: MeterShaping::default(),
                scene_markers: Vec::new(),
                events: Vec::new(),
                end: ScenarioEnd::Finite { duration_sec: 0.0 },
            },
            seed,
            next_event_order: 1,
            next_population_id: 1,
            next_flora_id: 1,
            flora: BTreeMap::new(),
            next_rain_id: 1,
            rain: BTreeMap::new(),
            next_voice_id: 1,
            populations: BTreeMap::new(),
            scopes: Vec::new(),
            temporal_mode_declared: false,
            parallel_used: false,
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
                depth: self.scopes.len(),
                spec: spec.spec,
                status: PopulationStatus::Live,
                pending_patch: None,
            },
        );
        self.push_event(
            self.cursor,
            vec![Action::PlaceFlora {
                flora_id: id,
                spec: spec.spec,
                extent_hz: (min_freq, max_freq),
                seeding,
            }],
            position,
        );
        Ok(FloraHandle { id })
    }

    pub(super) fn release_flora(&mut self, flora_id: u64, position: Position) {
        let Some(flora) = self.flora.get_mut(&flora_id) else {
            self.warn(
                "flora_release_absent",
                if flora_id < self.next_flora_id {
                    format!("release ignored for inactive Flora {flora_id}")
                } else {
                    format!("release on unknown Flora {flora_id}")
                },
                position,
            );
            return;
        };
        match flora.status {
            PopulationStatus::Live => flora.status = PopulationStatus::ReleasePending,
            PopulationStatus::ReleasePending => {
                self.warn(
                    "flora_release_pending",
                    format!("release ignored for Flora {flora_id}: release already pending"),
                    position,
                );
            }
            PopulationStatus::Released => self.warn(
                "flora_release_inactive",
                format!("release ignored for inactive Flora {flora_id}"),
                position,
            ),
        }
    }

    pub(super) fn place_rain(
        &mut self,
        spec: RainSpecHandle,
        placement: Placement,
        position: Position,
    ) -> Result<RainHandle, Box<EvalAltResult>> {
        let error =
            |message: String| Box::new(EvalAltResult::ErrorRuntime(message.into(), position));
        let Some(SpawnStrategy::Field {
            target,
            sampling,
            min_freq,
            max_freq,
            min_dist_erb,
            tension,
        }) = placement.strategy
        else {
            return Err(error(
                "Rain needs a field or random extent placement".into(),
            ));
        };
        let default_spacing = if target == FieldTarget::Uniform {
            0.0
        } else {
            1.0
        };
        if placement.count != 1 {
            return Err(error("Rain placement count must be 1".into()));
        }
        if sampling != FieldSampling::Density {
            return Err(error("Rain rejects peak sampling; use density".into()));
        }
        if tension != 0.0 {
            return Err(error("Rain placement tension must be 0".into()));
        }
        if min_dist_erb != default_spacing {
            return Err(error(format!(
                "Rain placement spacing must be {default_spacing}"
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
        if (f64::from(max_freq) / f64::from(min_freq)).log2() < 1.0 / 96.0 {
            return Err(error(
                "Rain extent must span at least one Log2Space bin (1/96 octave)".into(),
            ));
        }
        let id = self.next_rain_id;
        self.next_rain_id = self.next_rain_id.wrapping_add(1);
        self.rain.insert(
            id,
            RainState {
                depth: self.scopes.len(),
                fade_sec: spec.spec.fade_sec,
                status: PopulationStatus::Live,
                pending_patch: None,
            },
        );
        self.push_event(
            self.cursor,
            vec![Action::PlaceRain {
                rain_id: id,
                spec: spec.spec,
                extent_hz: (min_freq, max_freq),
                target,
            }],
            position,
        );
        self.check_emission()?;
        Ok(RainHandle { id })
    }

    pub(super) fn release_rain(&mut self, rain_id: u64, position: Position) {
        let Some(rain) = self.rain.get_mut(&rain_id) else {
            self.warn(
                "rain_release_absent",
                format!("release on unknown Rain {rain_id}"),
                position,
            );
            return;
        };
        match rain.status {
            PopulationStatus::Live => rain.status = PopulationStatus::ReleasePending,
            PopulationStatus::ReleasePending => self.warn(
                "rain_release_pending",
                format!("release ignored for Rain {rain_id}: release already pending"),
                position,
            ),
            PopulationStatus::Released => self.warn(
                "rain_release_inactive",
                format!("release ignored for inactive Rain {rain_id}"),
                position,
            ),
        }
    }

    fn fail_emission(&mut self, error: ScriptError) {
        if self.emission_error.is_none() {
            self.emission_error = Some(error.clone());
            #[cfg(feature = "native")]
            if let Some(stream) = &self.stream {
                stream.finish(Some(error));
                stream
                    .script_stop
                    .store(true, std::sync::atomic::Ordering::Release);
            }
        }
    }

    pub(super) fn push_event(
        &mut self,
        time_sec: f64,
        mut actions: Vec<Action>,
        position: Position,
    ) {
        if actions.is_empty() || self.emission_error.is_some() {
            return;
        }
        if self.scenario.end == ScenarioEnd::Open && self.scenario.harmonic_rhythm > 0 {
            for action in &mut actions {
                if let Action::UpdatePopulation {
                    population_id,
                    patch,
                } = action
                    && self
                        .populations
                        .get(population_id)
                        .is_some_and(|p| p.harmonic_supported)
                {
                    for label in filter_harmonic_patch(patch).into_iter().flatten() {
                        let at = self
                            .harmonic_sites
                            .iter()
                            .rev()
                            .find(|(id, time, name, _)| {
                                id == population_id && *time == time_sec && *name == label
                            })
                            .map_or(position, |s| s.3);
                        self.warn(label, format!("harmonic_rhythm: {label} would remove birth-time participation; previous value retained"), at);
                    }
                }
            }
        }
        if self.scenario.end == ScenarioEnd::Open {
            self.harmonic_sites.clear();
        }
        self.emitted_any_event = true;
        let order = self.next_event_order;
        self.next_event_order += 1;
        #[cfg(feature = "native")]
        if self.scenario.end == ScenarioEnd::Open && self.stream.is_some() {
            if let Some((a, b)) = self.scenario.meter_shaping.basin_hz {
                let (lo, hi) = sane_basin(a, b);
                for action in &mut actions {
                    action.set_founder_basin(0.5 * (lo + hi));
                }
            }
            let event = TimedEvent {
                time: time_sec,
                order,
                actions,
            };
            if let Err(mut error) = self
                .stream
                .as_ref()
                .unwrap()
                .push(score::ScoreItem::Event(event))
            {
                error.position = Some(position);
                self.fail_emission(error);
            }
            return;
        }
        {
            self.scenario.events.push(TimedEvent {
                time: time_sec,
                order,
                actions,
            });
        }
    }

    pub(super) fn push_scene_marker(&mut self, name: &str, position: Position) {
        let order = self.next_event_order;
        self.next_event_order += 1;
        #[cfg(feature = "native")]
        if self.scenario.end == ScenarioEnd::Open
            && let Some(stream) = &self.stream
        {
            if name.len() > stream.capacity.bytes {
                self.fail_emission(ScriptError::new(
                    "score item exceeds the ring capacity",
                    Some(position),
                ));
                return;
            }
            if let Err(mut error) = stream.push(score::ScoreItem::Marker(SceneMarker {
                name: name.to_string(),
                time: self.cursor,
                order,
            })) {
                error.position = Some(position);
                self.fail_emission(error);
            }
            return;
        }
        {
            self.scenario.scene_markers.push(SceneMarker {
                name: name.to_string(),
                time: self.cursor,
                order,
            });
        }
    }

    pub(super) fn record_harmonic_site(
        &mut self,
        id: u64,
        label: &'static str,
        position: Position,
    ) {
        if self.scenario.end == ScenarioEnd::Open {
            if !self.populations.contains_key(&id) {
                return;
            }
            self.harmonic_sites
                .retain(|(old, _, name, _)| *old != id || *name != label);
        }
        self.harmonic_sites.push((id, self.cursor, label, position));
    }

    pub(super) fn push_scope(&mut self) {
        self.scopes.push(ScopeFrame {});
    }

    pub(super) fn pop_scope(&mut self, position: Position) {
        let depth = self.scopes.len();
        if self.scopes.pop().is_none() {
            return;
        }
        self.collect_actions(Some(depth), position);
    }

    pub(super) fn finalize_duration(&mut self) -> f64 {
        let mut duration = self.cursor.max(0.0);
        if duration <= 0.0 {
            duration = 0.01;
            self.cursor = duration;
        }
        self.scenario.end = ScenarioEnd::Finite {
            duration_sec: duration,
        };
        duration
    }

    pub(super) fn max_release_tail(&self) -> f64 {
        let mut max_end: f64 = 0.0;
        for event in &self.scenario.events {
            for action in &event.actions {
                if let Action::ReleasePopulation { fade_sec, .. }
                | Action::ReleaseFlora { fade_sec, .. }
                | Action::ReleaseRain { fade_sec, .. } = action
                {
                    let fade = fade_sec.max(0.0);
                    max_end = max_end.max(event.time + fade as f64);
                }
            }
        }
        max_end
    }

    pub(super) fn emit_pending_actions(&mut self, position: Position) {
        self.collect_actions(None, position);
    }

    fn collect_actions(&mut self, depth: Option<usize>, position: Position) {
        #[cfg(feature = "native")]
        let capacity = if self.scenario.end == ScenarioEnd::Open {
            self.stream.as_ref().map(|stream| stream.capacity)
        } else {
            None
        };
        #[cfg(not(feature = "native"))]
        let capacity: Option<super::diagnostics::Credit> = None;
        let mut actions = Vec::new();
        let mut payload = 0;
        let mut append = |action: Action| -> Result<(), ScriptError> {
            let bytes = action.heap_bytes();
            let next_capacity = if actions.len() == actions.capacity() {
                (actions.capacity().max(2) * 2).min(capacity.map_or(usize::MAX, |c| c.actions))
            } else {
                actions.capacity()
            };
            if capacity.is_some_and(|c| {
                actions.len() + 1 > c.actions
                    || next_capacity * size_of::<Action>() + payload + bytes > c.bytes
            }) {
                return Err(ScriptError::new(
                    "score item exceeds the ring capacity",
                    Some(position),
                ));
            }
            if next_capacity > actions.capacity() {
                actions.reserve_exact(next_capacity - actions.len());
            }
            payload += bytes;
            actions.push(action);
            Ok(())
        };
        let result = (|| -> Result<(), ScriptError> {
            for (&id, p) in &mut self.populations {
                if depth.is_some_and(|d| p.depth != d) {
                    continue;
                }
                if p.pending_patch != ControlUpdate::default() && p.spawned {
                    if !p.tail_roughness_warned
                        && p.spec.control.pitch.tail_roughness_aversion > 0.0
                        && !p.spec.control.pitch.leave_self_out
                    {
                        p.spec.warn(
                            "tail_roughness_aversion() has no effect with leave_self_out(false)"
                                .into(),
                        );
                        p.tail_roughness_warned = true;
                    }
                    append(Action::UpdatePopulation {
                        population_id: id,
                        patch: std::mem::take(&mut p.pending_patch),
                    })?;
                }
            }
            for (&id, f) in &mut self.flora {
                if depth.is_some_and(|d| f.depth != d) {
                    continue;
                }
                if let Some(patch) = f.pending_patch.take() {
                    append(Action::UpdateFlora {
                        flora_id: id,
                        patch,
                    })?;
                }
            }
            for (&id, rain) in &mut self.rain {
                if depth.is_some_and(|d| rain.depth != d) {
                    continue;
                }
                if let Some(patch) = rain.pending_patch.take() {
                    append(Action::UpdateRain { rain_id: id, patch })?;
                }
            }
            for (&id, p) in &mut self.populations {
                if depth.is_some_and(|d| p.depth != d) {
                    continue;
                }
                if let Some((same_population_visible, other_population_visible)) =
                    p.pending_crowding_target.take()
                {
                    append(Action::SetPopulationCrowdingTarget {
                        population_id: id,
                        same_population_visible,
                        other_population_visible,
                    })?;
                }
            }
            for (&id, p) in &mut self.populations {
                if depth.is_some_and(|d| p.depth != d) {
                    continue;
                }
                if (depth.is_some() || p.pending_release) && p.spawned {
                    append(Action::ReleasePopulation {
                        population_id: id,
                        fade_sec: p.spec.release_sec(),
                    })?;
                    p.status = PopulationStatus::Released;
                }
            }
            for (&id, f) in &mut self.flora {
                if depth.is_some_and(|d| f.depth != d) {
                    continue;
                }
                if depth.is_some() || f.status == PopulationStatus::ReleasePending {
                    append(Action::ReleaseFlora {
                        flora_id: id,
                        fade_sec: f.spec.fade_sec,
                    })?;
                    f.status = PopulationStatus::Released;
                }
            }
            for (&id, rain) in &mut self.rain {
                if depth.is_some_and(|d| rain.depth != d) {
                    continue;
                }
                if depth.is_some() || rain.status == PopulationStatus::ReleasePending {
                    append(Action::ReleaseRain {
                        rain_id: id,
                        fade_sec: rain.fade_sec,
                    })?;
                    rain.status = PopulationStatus::Released;
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.fail_emission(error);
            return;
        }
        self.push_event(self.cursor, actions, position);
        self.populations
            .retain(|_, p| p.status != PopulationStatus::Released);
        self.flora
            .retain(|_, f| f.status != PopulationStatus::Released);
        self.rain
            .retain(|_, r| r.status != PopulationStatus::Released);
    }

    pub(super) fn finish(&mut self) {
        self.emit_pending_actions(Position::NONE);
        if self.scenario.end != ScenarioEnd::Open && self.scenario.harmonic_rhythm > 0 {
            let mut members = BTreeMap::new();
            let mut targets = 0;
            let mut warnings = Vec::new();
            for event in &mut self.scenario.events {
                for action in &mut event.actions {
                    match action {
                        Action::PlaceFlora { .. } => targets += 1,
                        Action::Spawn {
                            population_id,
                            spec,
                            ..
                        } => {
                            let supported = spec.control.supports_harmonic_rhythm();
                            targets += usize::from(supported);
                            members.insert(*population_id, supported);
                            if !supported
                                && spec.control.pitch.mode == PitchMode::Free
                                && matches!(
                                    spec.control.phonation.spec.timing,
                                    PhonationTiming::Once
                                )
                                && matches!(
                                    spec.control.phonation.spec.duration,
                                    DurationSpec::WhileAlive
                                )
                            {
                                warnings.push((format!("harmonic_rhythm: population {population_id} excluded; requires rearticulate with zero threshold"), self.harmonic_position));
                            }
                        }
                        Action::UpdatePopulation {
                            population_id,
                            patch,
                        } if members.get(population_id) == Some(&true) => {
                            let rejected = filter_harmonic_patch(patch);
                            for label in rejected.into_iter().flatten() {
                                let position = self
                                    .harmonic_sites
                                    .iter()
                                    .rev()
                                    .find(|(id, time, name, _)| {
                                        id == population_id && *time == event.time && *name == label
                                    })
                                    .map_or(self.harmonic_position, |s| s.3);
                                warnings.push((format!("harmonic_rhythm: {label} would remove birth-time participation; previous value retained"), position));
                            }
                        }
                        _ => {}
                    }
                }
            }
            if targets == 0 {
                warnings.push((
                    "harmonic_rhythm has no supported Flora or Voice placements".into(),
                    self.harmonic_position,
                ));
            }
            for (message, position) in warnings {
                self.warn("harmonic_membership", message, position);
            }
        }
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
        if self.scenario.end == ScenarioEnd::Open {
            if let Some((_, position)) = self.expression_sections.first() {
                self.warn(
                    "expression_open",
                    "expressive_section ignored: requires a finite final section".into(),
                    *position,
                );
            }
            return;
        }
        if let Some(position) = self.harmonic_observation_position {
            self.warn("finite_harmonic_observation", "observe_harmonic_relations ignored in a finite scenario; events determine observation and source capacity".into(), position);
            self.scenario.harmonic_relation_sources = None;
        }
        if let Some(position) = self.tonal_observation_position {
            self.warn("finite_tonal_observation", "observe_tonal_context ignored in a finite scenario; placements determine observation".into(), position);
            self.scenario.observe_tonal_context = false;
        }
        let mut end_time = self.finalize_duration();
        if !self.expression_sections.is_empty() {
            let (expression, position) = self.expression_sections[0];
            let result = if self.expression_sections.len() != 1 {
                Err("only one nonzero expressive section is supported")
            } else if self.cursor > expression.end {
                Err("requires the final section, without a later score cursor")
            } else {
                expression.validate(&self.scenario)
            };
            match result {
                Ok(()) => {
                    self.scenario.expression = Some(expression);
                    self.scenario.expression_cursor = Some(self.cursor);
                }
                Err(reason) => self.warn(
                    "expression_unsupported",
                    format!("expressive_section ignored: {reason}"),
                    position,
                ),
            }
        }
        if self.listener_profile == "tonal"
            && !self.explicit_expression
            && let Some(expression) = self.profile_section
            && expression.end == self.cursor
            && expression.validate(&self.scenario).is_ok()
        {
            self.scenario.expression = Some(expression);
            self.scenario.expression_cursor = Some(self.cursor);
        }
        let release_tail = self.max_release_tail();
        if release_tail > end_time {
            end_time = release_tail;
            self.scenario.end = ScenarioEnd::Finite {
                duration_sec: end_time,
            };
            self.cursor = end_time;
        }
        self.push_event(end_time, vec![Action::Finish], Position::NONE);
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
        if self.scenario.end == ScenarioEnd::Open
            && population_spec.spec.control.pitch.follow_tonal_context > 0.0
            && !self.scenario.observe_tonal_context
        {
            return Err(EvalAltResult::ErrorRuntime(
                "declare observe_tonal_context() before time advances to use follow_tonal_context in an endless scenario".into(), position).into());
        }
        population_spec.spec.diagnostics = self.warnings.clone();
        let specified = population_spec.spec.specified.clone();
        let warn_at = |label: &str, message: &str| {
            self.warn(
                "population_spec_ignored",
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
        if population_spec.spec.control.pitch.follow_tonal_context > 0.0
            && (!population_spec.spec.consonance_movement
                || !population_spec
                    .spec
                    .spawn_spec()
                    .control
                    .supports_own_sequence())
        {
            warn_at(
                "follow_tonal_context",
                "follow_tonal_context() requires free seek_consonance(), stepwise sustained participation, no glide or articulation persistence; ignored",
            );
            population_spec.spec.control.pitch.follow_tonal_context = 0.0;
        }
        if population_spec.spec.control.pitch.melodic_continuity > 0.0
            && (!population_spec.spec.consonance_movement
                || !population_spec
                    .spec
                    .spawn_spec()
                    .control
                    .supports_own_sequence())
        {
            warn_at(
                "melodic_continuity",
                "melodic_continuity() requires free seek_consonance(), stepwise sustained participation, no glide or articulation persistence; ignored",
            );
            population_spec.spec.control.pitch.melodic_continuity = 0.0;
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
                if !specified.contains_key("melodic_continuity") {
                    population_spec.spec.control.pitch.melodic_continuity = 0.00076066292;
                }
            }
        }
        let tail_roughness_warned = population_spec.spec.control.pitch.tail_roughness_aversion
            > 0.0
            && !population_spec.spec.control.pitch.leave_self_out;
        if tail_roughness_warned {
            warn_at(
                "tail_roughness_aversion",
                "tail_roughness_aversion() has no effect with leave_self_out(false)",
            );
        }
        let id = self.next_population_id;
        self.next_population_id = self.next_population_id.wrapping_add(1);
        let ids = if self.scenario.end == ScenarioEnd::Open {
            crate::scenario::SpawnIds::Allocate(count)
        } else {
            let ids = (self.next_voice_id..self.next_voice_id + count as u64).collect();
            self.next_voice_id += count as u64;
            crate::scenario::SpawnIds::Assigned(ids)
        };
        let population = PopulationState {
            harmonic_supported: population_spec
                .spec
                .spawn_spec()
                .control
                .supports_harmonic_rhythm(),
            respawn_policy: population_spec.spec.respawn_policy,
            loo_at_placement: population_spec.spec.control.pitch.leave_self_out,
            tail_roughness_warned,
            crowding_target_same: population_spec.spec.crowding_target_same,
            crowding_target_other: population_spec.spec.crowding_target_other,
            spec: population_spec.spec,
            status: PopulationStatus::Live,
            spawned: true,
            depth: self.scopes.len(),
            pending_patch: ControlUpdate::default(),
            pending_crowding_target: None,
            pending_release: false,
        };
        self.populations.insert(id, population);
        if !ids.is_empty() {
            let population = self.populations.get(&id).expect("inserted population");
            let respawn = !matches!(population.respawn_policy, RespawnPolicy::None);
            let action_count = if respawn { 3 } else { 2 };
            #[cfg(feature = "native")]
            if let Some(stream) = &self.stream
                && self.scenario.end == ScenarioEnd::Open
                && (action_count > stream.capacity.actions
                    || action_count * size_of::<Action>()
                        + population
                            .spec
                            .control
                            .body
                            .modes
                            .as_ref()
                            .map_or(0, |m| m.heap_bytes())
                        > stream.capacity.bytes)
            {
                self.fail_emission(ScriptError::new(
                    "score item exceeds the ring capacity",
                    Some(position),
                ));
                self.check_emission()?;
            }
            let population = self.populations.get(&id).expect("inserted population");
            let mut actions = Vec::with_capacity(action_count);
            actions.push(Action::Spawn {
                population_id: id,
                ids,
                spec: population.spec.spawn_spec(),
                strategy,
            });
            actions.push(Action::SetPopulationCrowdingTarget {
                population_id: id,
                same_population_visible: population.crowding_target_same,
                other_population_visible: population.crowding_target_other,
            });
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
            self.push_event(self.cursor, actions, position);
            self.check_emission()?;
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

    pub(super) fn check_header(
        &self,
        name: &str,
        position: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        if self.scenario.end == ScenarioEnd::Open && self.cursor > 0.0 {
            return Err(EvalAltResult::ErrorRuntime(
                format!("{name} must be declared before time advances in an endless scenario")
                    .into(),
                position,
            )
            .into());
        }
        Ok(())
    }

    pub(super) fn set_output_gain(
        &mut self,
        db: f64,
        position: Position,
    ) -> Result<(), Box<EvalAltResult>> {
        self.check_header("output_gain", position)?;
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
        self.check_header("seed", position)?;
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

    pub(super) fn wait(&mut self, sec: f64, position: Position) -> Result<(), Box<EvalAltResult>> {
        #[cfg(feature = "native")]
        use std::sync::atomic::Ordering;
        self.emit_pending_actions(position);
        self.check_emission()?;
        let next = self.cursor + sec.max(0.0);
        #[cfg(feature = "native")]
        if self.scenario.end == ScenarioEnd::Open
            && let Some(stream) = &self.stream
        {
            #[cfg(test)]
            {
                stream.state.lock().unwrap().script_sizes =
                    (self.populations.len(), self.flora.len(), self.scopes.len());
            }
            if stream.bound_reached.load(Ordering::Acquire) {
                return Err(Self::bound_token());
            }
            let crossed = stream.bound.is_some_and(|bound| next > bound);
            self.cursor = if crossed { stream.bound.unwrap() } else { next };
            if self.cursor > 0.0 {
                stream.set_header(self.scenario.clone());
            }
            stream.advance(self.cursor, !crossed);
            if crossed {
                stream.bound_reached.store(true, Ordering::Release);
                return Err(Self::bound_token());
            }
            if stream.script_stop.load(Ordering::Acquire) {
                return Err(
                    EvalAltResult::ErrorTerminated("stopped".into(), Position::NONE).into(),
                );
            }
            return Ok(());
        }
        self.cursor = next;
        Ok(())
    }

    pub(super) fn bound_token() -> Box<EvalAltResult> {
        EvalAltResult::ErrorTerminated("score bound reached".into(), Position::NONE).into()
    }

    pub(super) fn check_emission(&self) -> Result<(), Box<EvalAltResult>> {
        if let Some(error) = &self.emission_error {
            Err(EvalAltResult::ErrorTerminated(
                error.message.clone().into(),
                error.position.unwrap_or(Position::NONE),
            )
            .into())
        } else {
            Ok(())
        }
    }

    pub(super) fn flush(&mut self, position: Position) {
        self.emit_pending_actions(position);
    }

    pub(super) fn release_population(&mut self, population_id: u64, position: Position) {
        let Some(population) = self.populations.get_mut(&population_id) else {
            self.warn(
                "release_absent",
                if population_id < self.next_population_id {
                    format!("release ignored for inactive population {population_id}")
                } else {
                    format!("release on unknown population {population_id}")
                },
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
                    "release_pending",
                    format!(
                        "release ignored for population {population_id}: release already pending"
                    ),
                    position,
                );
            }
            PopulationStatus::Released => {
                self.warn(
                    "release_inactive",
                    format!("release ignored for inactive population {population_id}"),
                    position,
                );
            }
        }
    }

    pub(super) fn population_absence(&self, id: u64) -> &'static str {
        if id < self.next_population_id {
            "inactive"
        } else {
            "unknown"
        }
    }

    pub(super) fn warn_population_inactive(
        &self,
        population_id: u64,
        label: &str,
        position: Position,
    ) {
        self.warn(
            "population_update_inactive",
            format!("{label} ignored for inactive population {population_id}"),
            position,
        );
    }
}

pub(super) fn filter_harmonic_patch(patch: &mut ControlUpdate) -> [Option<&'static str>; 3] {
    let mut rejected = [None; 3];
    if patch
        .pitch_apply_mode
        .is_some_and(|p| p != PitchApplyMode::Rearticulate)
    {
        patch.pitch_apply_mode = None;
        rejected[0] = Some("pitch_apply_mode");
    }
    if patch.rearticulate_above_cents.is_some_and(|v| v > 0.0) {
        patch.rearticulate_above_cents = None;
        rejected[1] = Some("rearticulate_above");
    }
    if patch.freq.take().is_some() {
        rejected[2] = Some("freq");
    }
    rejected
}

pub(super) fn update_rain_numeric(
    ctx: &mut ScriptContext,
    handle: RainHandle,
    name: &str,
    value: FLOAT,
    position: Position,
) -> Result<RainHandle, Box<EvalAltResult>> {
    let value = super::spec::validate_rain_numeric(name, value, position)?;
    let Some(rain) = ctx.rain.get_mut(&handle.id) else {
        ctx.warn(
            "rain_update_absent",
            format!("{name} ignored for unknown Rain {}", handle.id),
            position,
        );
        return Ok(handle);
    };
    if rain.status == PopulationStatus::Live {
        let patch = rain.pending_patch.get_or_insert_with(RainPatch::default);
        match name {
            "amp" => patch.amp = Some(value),
            "rate" => patch.rate = Some(value),
            "sway" => patch.sway = Some(value),
            "hit" => patch.hit = Some(value),
            "width" => patch.width = Some(value),
            _ => unreachable!("registered Rain patch parameter"),
        }
    } else {
        ctx.warn(
            "rain_update_inactive",
            format!("{name} ignored for inactive Rain {}", handle.id),
            position,
        );
    }
    Ok(handle)
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
            "flora_update_absent",
            format!(
                "{name} ignored for {} Flora {}",
                if handle.id < ctx.next_flora_id {
                    "inactive"
                } else {
                    "unknown"
                },
                handle.id
            ),
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
            "flora_update_inactive",
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
            "flora_pitch_mode_absent",
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
            "flora_pitch_mode_inactive",
            format!("pitch_apply_mode ignored for inactive Flora {}", handle.id),
            position,
        );
    }
    handle
}
