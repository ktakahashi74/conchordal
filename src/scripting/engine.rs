// Expand numeric arguments consistently for every owner, including mixed pairs.
macro_rules! numeric_overloads {
    ($engine:expr, $name:expr, |$call:ident $(, $fixed:ident: $fixed_ty:ty)*; $($value:ident),+| $body:block $(,)?) => {{
        let operation = move |$call: NativeCallContext $(, $fixed: $fixed_ty)*, $($value: f32),+| $body;
        numeric_overloads!(@expand $engine, $name, operation, [$($fixed: $fixed_ty,)*], [], [$($value),+]);
    }};
    (@expand $engine:expr, $name:expr, $operation:ident, [$($fixed:ident: $fixed_ty:ty,)*], [$($done:ident: $ty:ty,)*], [$next:ident $(, $rest:ident)*]) => {
        numeric_overloads!(@expand $engine, $name, $operation, [$($fixed: $fixed_ty,)*], [$($done: $ty,)* $next: FLOAT,], [$($rest),*]);
        numeric_overloads!(@expand $engine, $name, $operation, [$($fixed: $fixed_ty,)*], [$($done: $ty,)* $next: INT,], [$($rest),*]);
    };
    (@expand $engine:expr, $name:expr, $operation:ident, [$($fixed:ident: $fixed_ty:ty,)*], [$($value:ident: $ty:ty,)*], []) => {{
        let operation = $operation.clone();
        $engine.register_fn($name, move |call: NativeCallContext $(, $fixed: $fixed_ty)*, $($value: $ty),*| operation(call $(, $fixed)*, $($value as f32),*));
    }};
}

mod global;
mod inhabitants;
mod placement_modes;
mod population;
mod population_live;

use super::*;

impl ScriptHost {
    #[cfg(feature = "native")]
    pub(crate) fn start_source(
        source: String,
        seed: Option<u64>,
        bound: Option<f64>,
        capacity: usize,
    ) -> score::ScriptSession {
        let stream = score::ScoreStream::new(bound, capacity);
        let producer = stream.clone();
        let thread = std::thread::Builder::new()
            .name("score".into())
            .spawn(move || {
                producer.register_producer();
                let mut context =
                    seed.map_or_else(ScriptContext::default, ScriptContext::with_seed);
                context.stream = Some(producer.clone());
                context.warnings = producer.warnings.clone();
                let ctx = Arc::new(Mutex::new(context));
                let engine = Self::create_engine(ctx.clone());
                let result = engine.eval::<Dynamic>(&source);
                let mut ctx = ctx.lock().unwrap();
                let bound = producer
                    .bound_reached
                    .load(std::sync::atomic::Ordering::Acquire);
                let error = match result {
                    Err(error) if !bound => Some(ScriptError::from_eval(error, None)),
                    Ok(value) if ctx.scenario.end != ScenarioEnd::Open && !value.is_unit() => {
                        Some(ScriptError::from_eval(
                            EvalAltResult::ErrorMismatchOutputType(
                                "()".into(),
                                engine.map_type_name(value.type_name()).into(),
                                Position::NONE,
                            )
                            .into(),
                            None,
                        ))
                    }
                    _ => None,
                };
                if error.is_none() {
                    ctx.finish();
                }
                let error = ctx.emission_error.clone().or(error).or_else(|| {
                    (!ctx.emitted_any_event)
                        .then(|| ScriptError::new("Scenario has no events", None))
                });
                if error.is_none() {
                    producer.set_header(ctx.scenario.clone());
                }
                producer.finish(error);
            })
            .expect("spawn score thread");
        score::ScriptSession {
            stream,
            thread: Some(thread),
        }
    }

    #[cfg(feature = "native")]
    pub(crate) fn load_incremental(
        path: &str,
        seed: Option<u64>,
        bound: Option<f64>,
        exiting: &std::sync::atomic::AtomicBool,
        end_request: &std::sync::atomic::AtomicBool,
    ) -> Result<(Scenario, Option<score::ScriptSession>, Vec<ScriptWarning>), ScriptError> {
        let source = fs::read_to_string(path)
            .map_err(|err| ScriptError::new(format!("{path}: {err}"), None))?;
        let session = Self::start_source(source, seed, bound, score::ACTION_CAPACITY);
        loop {
            if exiting.load(std::sync::atomic::Ordering::Acquire)
                || end_request.load(std::sync::atomic::Ordering::Acquire)
            {
                session.stream.close("stopped");
            }
            let mut state = session.stream.state.lock().unwrap();
            if let Some(header) = state.header.take() {
                // Retain a header sentinel for visibility and the pre-header admission rule.
                let mut sentinel = header.clone();
                sentinel.events.clear();
                sentinel.scene_markers.clear();
                state.header = Some(sentinel);
                drop(state);
                let warnings = session.stream.warnings.lock().unwrap().snapshot();
                if header.end == ScenarioEnd::Open {
                    return Ok((header, Some(session), warnings));
                }
                return Ok((header, None, warnings));
            }
            if let Some(terminal) = &state.terminal {
                let error = match terminal {
                    score::Terminal::Failed(error) => error.clone(),
                    score::Terminal::Closed(reason) => {
                        ScriptError::new(format!("script load cancelled: {reason}"), None)
                    }
                    score::Terminal::Finished => {
                        ScriptError::new("script finished without header", None)
                    }
                };
                drop(state);
                return Err(error.with_context(path));
            }
            drop(
                session
                    .stream
                    .changed
                    .wait_timeout(state, std::time::Duration::from_millis(10))
                    .unwrap(),
            );
        }
    }

    pub fn create_engine(ctx: Arc<Mutex<ScriptContext>>) -> Engine {
        let mut engine = Engine::new();
        global::register_scenario_builtins(&mut engine, &ctx);

        engine.register_type_with_name::<PopulationSpecHandle>("PopulationSpec");
        engine.register_type_with_name::<PopulationHandle>("Population");
        engine.register_type_with_name::<FloraSpecHandle>("FloraSpec");
        engine.register_type_with_name::<FloraHandle>("Flora");
        engine.register_type_with_name::<RainSpecHandle>("RainSpec");
        engine.register_type_with_name::<RainHandle>("Rain");
        engine.register_type_with_name::<Placement>("Placement");
        engine.register_type_with_name::<Bus>("Bus");
        engine.register_type_with_name::<BusSet>("BusSet");
        engine.register_type_with_name::<ModePattern>("ModePattern");

        // Keep registration order across owners: Rhai replaces duplicate signatures.
        global::register_bus_constants(&mut engine);
        global::register_population_constructors(&mut engine);
        inhabitants::register_flora(&mut engine, &ctx);
        inhabitants::register_rain(&mut engine, &ctx);
        global::register_bus_operators(&mut engine);
        population::register_spec_methods(&mut engine, &ctx);
        global::register_timeline(&mut engine, &ctx);
        placement_modes::register_modes(&mut engine, &ctx);
        placement_modes::register_placements(&mut engine, &ctx);
        population_live::register_live_methods(&mut engine, &ctx);
        global::register_director(&mut engine, &ctx);
        engine
    }

    /// Loads and evaluates a scenario script. `seed`, when given, seeds the
    /// fresh scenario before evaluation; a script-level `seed()` call still
    /// wins, since it runs during evaluation and overwrites it.
    #[cfg(feature = "native")]
    pub fn load_script(path: &str, seed: Option<u64>) -> Result<Scenario, ScriptError> {
        let (scenario, warnings) = Self::load_script_with_warnings(path, seed)?;
        for warning in warnings {
            eprintln!(
                "{path}:{:?}: warning: {}",
                warning.position, warning.message
            );
        }
        Ok(scenario)
    }

    /// Evaluate the script and return positioned diagnostics to CLI and test consumers.
    #[cfg(feature = "native")]
    pub fn load_script_with_warnings(
        path: &str,
        seed: Option<u64>,
    ) -> Result<(Scenario, Vec<ScriptWarning>), ScriptError> {
        let src = fs::read_to_string(path)
            .map_err(|err| ScriptError::new(format!("{path}: {err}"), None))?;
        Self::evaluate_source(&src, seed, Some(path), false)
    }

    /// Evaluate a finite scene without accessing files or starting a score producer.
    pub fn compile_source(
        source: &str,
        seed: Option<u64>,
    ) -> Result<(Scenario, Vec<ScriptWarning>), ScriptError> {
        Self::evaluate_source(source, seed, None, true)
    }

    fn evaluate_source(
        source: &str,
        seed: Option<u64>,
        path: Option<&str>,
        finite_only: bool,
    ) -> Result<(Scenario, Vec<ScriptWarning>), ScriptError> {
        #[cfg(not(feature = "native"))]
        if seed.is_none() {
            return Err(ScriptError::new(
                "provide a seed or host entropy for web execution",
                None,
            ));
        }
        let ctx = Arc::new(Mutex::new(match seed {
            Some(seed) => ScriptContext::with_seed(seed),
            None => ScriptContext::default(),
        }));
        let mut engine = Self::create_engine(ctx.clone());
        if finite_only {
            engine.register_fn(
                "endless",
                |call: NativeCallContext| -> Result<(), Box<EvalAltResult>> {
                    Err(EvalAltResult::ErrorRuntime(
                        "endless is unavailable in a step engine".into(),
                        call.call_position(),
                    )
                    .into())
                },
            );
        }
        let value = engine
            .eval::<Dynamic>(source)
            .map_err(|err| ScriptError::from_eval(err, path))?;
        let mut ctx = ctx.lock().expect("lock script context");
        if ctx.scenario.end != ScenarioEnd::Open && !value.is_unit() {
            return Err(ScriptError::from_eval(
                EvalAltResult::ErrorMismatchOutputType(
                    "()".into(),
                    engine.map_type_name(value.type_name()).into(),
                    Position::NONE,
                )
                .into(),
                path,
            ));
        }
        ctx.finish();
        let warnings = ctx
            .warnings
            .lock()
            .expect("lock script warnings")
            .snapshot();
        Ok((ctx.scenario.clone(), warnings))
    }
}
