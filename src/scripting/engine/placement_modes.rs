use super::{
    Arc, Array, Engine, EvalAltResult, FLOAT, FieldSampling, INT, ModePattern, Mutex,
    NativeCallContext, Placement, ScenarioEnd, ScriptContext, ScriptWarning, parse_enum,
    rhai_array_to_f32,
};

pub(super) fn register_modes(engine: &mut Engine, ctx: &Arc<Mutex<ScriptContext>>) {
    engine.register_fn("harmonic_modes", ModePattern::harmonic_modes);
    engine.register_fn("odd_modes", ModePattern::odd_modes);
    numeric_overloads!(engine, "power_modes", |_call; beta| { ModePattern::power_modes(beta) });
    numeric_overloads!(engine, "stiff_string_modes", |_call; stiffness| { ModePattern::stiff_string_modes(stiffness) });
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "custom_modes",
        move |call: NativeCallContext, ratios: Array| {
            ModePattern::custom_modes(rhai_array_to_f32(
                ratios,
                "custom_modes",
                &diagnostics,
                call.call_position(),
            ))
        },
    );
    engine.register_fn(
        "modal_table",
        |call: NativeCallContext, name: &str| -> Result<ModePattern, Box<EvalAltResult>> {
            let name = parse_enum(
                "modal_table",
                name,
                call.call_position(),
                &[
                    ("uniform_aluminum_bar", "uniform_aluminum_bar"),
                    ("uniform_wooden_bar", "uniform_wooden_bar"),
                    ("xylophone", "xylophone"),
                    ("vibraphone_1", "vibraphone_1"),
                    ("vibraphone_2", "vibraphone_2"),
                    ("wine_glass", "wine_glass"),
                ],
            )?;
            Ok(ModePattern::modal_table(name).expect("validated modal table"))
        },
    );
    engine.register_fn(
        "landscape_density_modes",
        ModePattern::landscape_density_modes,
    );
    engine.register_fn("landscape_peaks_modes", ModePattern::landscape_peaks_modes);
    engine.register_fn(
        "count",
        |call: NativeCallContext,
         pattern: ModePattern,
         n: INT|
         -> Result<ModePattern, Box<EvalAltResult>> {
            let max = crate::core::mode_pattern::MAX_MODE_COUNT;
            match usize::try_from(n) {
                Ok(count) if (1..=max).contains(&count) => Ok(pattern.with_count(count)),
                _ => Err(Box::new(EvalAltResult::ErrorRuntime(
                    format!("mode count must be within [1, {max}]").into(),
                    call.call_position(),
                ))),
            }
        },
    );
    let warnings = ctx.lock().unwrap().warnings.clone();
    numeric_overloads!(engine, "range", |call, pattern: ModePattern; min_mul, max_mul| { if pattern.supports_range() {pattern.with_range(min_mul, max_mul)} else {warnings.lock().unwrap().push(ScriptWarning { kind: "mode_range_ignored", count: 1, message: "range() is only supported for landscape_*_modes(); ignored".into(), position: call.call_position()}); pattern} });
    let warnings = ctx.lock().unwrap().warnings.clone();
    numeric_overloads!(engine, "spacing", |call, pattern: ModePattern; min_dist| { if pattern.supports_min_dist_erb() {pattern.with_min_dist_erb(min_dist)} else {warnings.lock().unwrap().push(ScriptWarning { kind: "mode_spacing_ignored", count: 1, message: "spacing() is only supported for landscape_*_modes(); ignored".into(), position: call.call_position()}); pattern} });
    let warnings = ctx.lock().unwrap().warnings.clone();
    numeric_overloads!(engine, "gamma", |call, pattern: ModePattern; gamma| { if pattern.supports_gamma() {pattern.with_gamma(gamma)} else {warnings.lock().unwrap().push(ScriptWarning { kind: "mode_gamma_ignored", count: 1, message: "gamma() is only supported for landscape_density_modes(); ignored".into(), position: call.call_position()}); pattern} });
    numeric_overloads!(engine, "jitter", |_call, pattern: ModePattern; cents| { pattern.with_jitter_cents(cents) });
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "seed",
        move |call_ctx: NativeCallContext, pattern: ModePattern, seed: INT| {
            if seed < 0 {
                diagnostics
                    .lock()
                    .expect("lock script warnings")
                    .push(ScriptWarning {
                        kind: "mode_seed_negative",
                        count: 1,
                        message: "seed() expects >= 0".to_string(),
                        position: call_ctx.call_position(),
                    });
                pattern
            } else {
                pattern.with_seed(seed as u64)
            }
        },
    );
}

pub(super) fn register_placements(engine: &mut Engine, ctx: &Arc<Mutex<ScriptContext>>) {
    {
        let context = ctx.clone();
        let set_site = move |call: NativeCallContext,
                             mut placement: Placement,
                             x: FLOAT,
                             y: FLOAT|
              -> Result<Placement, Box<EvalAltResult>> {
            let position = call.call_position();
            if context.lock().unwrap().scenario.end == ScenarioEnd::Open {
                return Err(EvalAltResult::ErrorRuntime(
                    "site requires a finite score".into(),
                    position,
                )
                .into());
            }
            let site = crate::scenario::Site { x, y, position };
            site.check(&[])
                .map_err(|msg| Box::new(EvalAltResult::ErrorRuntime(msg.into(), position)))?;
            placement.site = Some(site);
            Ok(placement)
        };
        engine.register_fn("site", set_site.clone());
        let f = set_site.clone();
        engine.register_fn(
            "site",
            move |c: NativeCallContext, p: Placement, x: INT, y: FLOAT| f(c, p, x as FLOAT, y),
        );
        let f = set_site.clone();
        engine.register_fn(
            "site",
            move |c: NativeCallContext, p: Placement, x: FLOAT, y: INT| f(c, p, x, y as FLOAT),
        );
        engine.register_fn(
            "site",
            move |c: NativeCallContext, p: Placement, x: INT, y: INT| {
                set_site(c, p, x as FLOAT, y as FLOAT)
            },
        );
    }
    numeric_overloads!(engine, "at", |_call; freq| {Placement::at(freq)});
    numeric_overloads!(engine, "consonance", |_call; root| {Placement::consonance_root(root)});
    for (name, constructor) in [
        (
            "consonance",
            Placement::consonance_range as fn(f32, f32) -> Placement,
        ),
        ("dissonance", Placement::dissonance),
        ("edge", Placement::edge),
        ("gap", Placement::gap),
        ("random", Placement::random),
        ("line", Placement::line),
    ] {
        numeric_overloads!(engine, name, |_call; lo, hi| {constructor(lo, hi)});
    }
    for (name, sampling) in [
        ("peak", FieldSampling::Peak),
        ("density", FieldSampling::Density),
    ] {
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            name,
            move |call: NativeCallContext, mut placement: Placement| {
                placement.diagnostics = diagnostics.clone();
                placement.position = call.call_position();
                placement.with_sampling(sampling)
            },
        );
    }
    for (name, setter) in [
        (
            "tension",
            Placement::with_tension as fn(Placement, f32) -> Placement,
        ),
        ("spacing", Placement::with_spacing),
    ] {
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        numeric_overloads!(engine, name, |call, placement: Placement; value| {
            let mut placement = placement;
            placement.diagnostics = diagnostics.clone();
            placement.position = call.call_position();
            setter(placement, value)
        });
    }
    engine.register_fn("count", |placement: Placement, count: INT| {
        placement.with_count(count)
    });
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    numeric_overloads!(engine, "range", |call, placement: Placement; min_mul, max_mul| {
        let mut placement = placement;
        placement.diagnostics = diagnostics.clone();
        placement.position = call.call_position();
        placement.with_range(min_mul, max_mul)
    });
}
