use super::{
    Arc, Bus, BusSet, Engine, EvalAltResult, FLOAT, FloraHandle, FloraSpec, FloraSpecHandle,
    FloraTone, INT, ModePattern, Mutex, NativeCallContext, Placement, RainHandle, RainSpec,
    RainSpecHandle, ScriptContext, parse_enum, update_flora_numeric, update_rain_numeric,
};

pub(super) fn register_flora(engine: &mut Engine, ctx: &Arc<Mutex<ScriptContext>>) {
    engine.register_fn("flora", || FloraSpecHandle {
        spec: FloraSpec::default(),
    });
    for name in [
        "amp",
        "fade",
        "pace",
        "rest_on_habituation",
        "respond_to_onsets",
    ] {
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            name,
            move |call: NativeCallContext, spec: FloraSpecHandle, value: FLOAT| {
                spec.with_numeric(name, value, call.call_position(), &diagnostics)
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            name,
            move |call: NativeCallContext, spec: FloraSpecHandle, value: INT| {
                spec.with_numeric(name, value as FLOAT, call.call_position(), &diagnostics)
            },
        );
    }
    for name in ["amp", "pace"] {
        let ctx_float = ctx.clone();
        engine.register_fn(
            name,
            move |call: NativeCallContext, handle: FloraHandle, value: FLOAT| {
                let mut ctx = ctx_float.lock().expect("lock script context");
                update_flora_numeric(&mut ctx, handle, name, value, call.call_position())
            },
        );
        let ctx_int = ctx.clone();
        engine.register_fn(
            name,
            move |call: NativeCallContext, handle: FloraHandle, value: INT| {
                let mut ctx = ctx_int.lock().expect("lock script context");
                update_flora_numeric(&mut ctx, handle, name, value as FLOAT, call.call_position())
            },
        );
    }
    engine.register_fn(
        "tone",
        |call: NativeCallContext,
         mut spec: FloraSpecHandle,
         name: &str|
         -> Result<FloraSpecHandle, Box<EvalAltResult>> {
            spec.spec.tone = parse_enum(
                "Flora tone",
                name,
                call.call_position(),
                &[("sine", FloraTone::Sine), ("resonant", FloraTone::Resonant)],
            )?;
            Ok(spec)
        },
    );
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "pitch_apply_mode",
        move |call: NativeCallContext, mut spec: FloraSpecHandle, name: &str| {
            if let Some(mode) =
                super::spec::flora_pitch_apply_mode(name, call.call_position(), &diagnostics)
            {
                spec.spec.pitch_apply_mode = mode;
            }
            spec
        },
    );
    let ctx_flora_pitch = ctx.clone();
    engine.register_fn(
        "pitch_apply_mode",
        move |call: NativeCallContext, handle: FloraHandle, name: &str| {
            super::context::update_flora_pitch_apply_mode(
                &mut ctx_flora_pitch.lock().expect("lock script context"),
                handle,
                name,
                call.call_position(),
            )
        },
    );
    engine.register_fn("residual", |mut spec: FloraSpecHandle, enabled: bool| {
        spec.spec.residual = enabled;
        spec
    });
    engine.register_fn(
        "leave_self_out",
        |mut spec: FloraSpecHandle, enabled: bool| {
            spec.spec.leave_self_out = enabled;
            spec
        },
    );
    engine.register_fn("send", |mut spec: FloraSpecHandle, bus: Bus| {
        spec.spec.routing = bus.routing;
        spec
    });
    engine.register_fn("send", |mut spec: FloraSpecHandle, buses: BusSet| {
        spec.spec.routing = buses.routing();
        spec
    });
    let ctx_place_flora = ctx.clone();
    engine.register_fn(
        "place",
        move |call: NativeCallContext, spec: FloraSpecHandle, placement: Placement| {
            ctx_place_flora
                .lock()
                .expect("lock script context")
                .place_flora(spec, placement, call.call_position())
        },
    );
    let ctx_release_flora = ctx.clone();
    engine.register_fn(
        "release",
        move |call: NativeCallContext, handle: FloraHandle| {
            ctx_release_flora
                .lock()
                .expect("lock script context")
                .release_flora(handle.id, call.call_position());
        },
    );
}

pub(super) fn register_rain(engine: &mut Engine, ctx: &Arc<Mutex<ScriptContext>>) {
    engine.register_fn("rain", || RainSpecHandle {
        spec: RainSpec::default(),
    });
    for name in [
        "amp",
        "fade",
        "rate",
        "sway",
        "hit",
        "width",
        "ring",
        "follow_meter",
        "object_decay",
    ] {
        engine.register_fn(
            name,
            move |call: NativeCallContext, spec: RainSpecHandle, value: FLOAT| {
                spec.with_numeric(name, value, call.call_position())
            },
        );
        engine.register_fn(
            name,
            move |call: NativeCallContext, spec: RainSpecHandle, value: INT| {
                spec.with_numeric(name, value as FLOAT, call.call_position())
            },
        );
    }
    for name in [
        "amp",
        "rate",
        "sway",
        "hit",
        "width",
        "ring",
        "follow_meter",
    ] {
        let ctx_float = ctx.clone();
        engine.register_fn(
            name,
            move |call: NativeCallContext, handle: RainHandle, value: FLOAT| {
                let mut ctx = ctx_float.lock().expect("lock script context");
                update_rain_numeric(&mut ctx, handle, name, value, call.call_position())
            },
        );
        let ctx_int = ctx.clone();
        engine.register_fn(
            name,
            move |call: NativeCallContext, handle: RainHandle, value: INT| {
                let mut ctx = ctx_int.lock().expect("lock script context");
                update_rain_numeric(&mut ctx, handle, name, value as FLOAT, call.call_position())
            },
        );
    }
    engine.register_fn(
        "objects",
        |call: NativeCallContext,
         mut spec: RainSpecHandle,
         count: INT|
         -> Result<RainSpecHandle, Box<EvalAltResult>> {
            let max = crate::life::rain::MAX_RAIN_OBJECTS;
            if !(0..=INT::from(max)).contains(&count) {
                return Err(Box::new(EvalAltResult::ErrorRuntime(
                    format!("Rain objects must be within [0, {max}]").into(),
                    call.call_position(),
                )));
            }
            spec.spec.objects = count as u32;
            Ok(spec)
        },
    );
    engine.register_fn(
        "object_modes",
        |mut spec: RainSpecHandle, pattern: ModePattern| {
            spec.spec.object_modes = pattern;
            spec
        },
    );
    engine.register_fn(
        "impact_modes",
        |mut spec: RainSpecHandle, pattern: ModePattern| {
            spec.spec.impact_modes = pattern;
            spec
        },
    );
    engine.register_fn("anchor", |mut spec: RainSpecHandle| {
        spec.spec.anchored = true;
        spec
    });
    engine.register_fn(
        "leave_drops_out",
        |mut spec: RainSpecHandle, enabled: bool| {
            spec.spec.leave_drops_out = enabled;
            spec
        },
    );
    engine.register_fn(
        "objects_follow_drops",
        |mut spec: RainSpecHandle, enabled: bool| {
            spec.spec.objects_follow_drops = enabled;
            spec
        },
    );
    engine.register_fn("send", |mut spec: RainSpecHandle, bus: Bus| {
        spec.spec.routing = bus.routing;
        spec
    });
    engine.register_fn("send", |mut spec: RainSpecHandle, buses: BusSet| {
        spec.spec.routing = buses.routing();
        spec
    });
    let ctx_place_rain = ctx.clone();
    engine.register_fn(
        "place",
        move |call: NativeCallContext, spec: RainSpecHandle, placement: Placement| {
            ctx_place_rain
                .lock()
                .expect("lock script context")
                .place_rain(spec, placement, call.call_position())
        },
    );
    let ctx_release_rain = ctx.clone();
    engine.register_fn(
        "release",
        move |call: NativeCallContext, handle: RainHandle| {
            ctx_release_rain
                .lock()
                .expect("lock script context")
                .release_rain(handle.id, call.call_position());
        },
    );
}
