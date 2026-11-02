use super::*;

#[inline]
pub(super) fn cmp_time_order(
    a_time: f64,
    a_order: u64,
    b_time: f64,
    b_order: u64,
) -> std::cmp::Ordering {
    a_time
        .partial_cmp(&b_time)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| a_order.cmp(&b_order))
}

pub(super) fn validate_scenario_script_extension(script_path: &Path) -> Result<(), String> {
    let ext = script_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ext != "rhai" {
        return Err(format!(
            "Scenario must be a .rhai script: {}",
            script_path.display()
        ));
    }
    Ok(())
}

pub fn compile_scenario_from_script(
    script_path: &Path,
    seed: Option<u64>,
) -> Result<Scenario, String> {
    crate::life::modal::register_modal();
    validate_scenario_script_extension(script_path)?;
    let path_str = script_path.to_string_lossy();
    let scenario = ScriptHost::load_script(&path_str, seed).map_err(|error| error.to_string())?;
    validate_scenario(&scenario)?;
    Ok(scenario)
}

pub(crate) fn validate_scenario(scenario: &Scenario) -> Result<(), String> {
    if scenario.events.is_empty() {
        return Err("Scenario has no events".to_string());
    }

    let mut has_finish = false;
    for event in &scenario.events {
        for action in &event.actions {
            match action {
                Action::Finish => {
                    has_finish = true;
                }
                Action::PlaceFlora { .. }
                | Action::UpdateFlora { .. }
                | Action::ReleaseFlora { .. }
                | Action::PlaceRain { .. }
                | Action::UpdateRain { .. }
                | Action::ReleaseRain { .. }
                | Action::Spawn { .. }
                | Action::UpdatePopulation { .. }
                | Action::ReleasePopulation { .. }
                | Action::SetRespawnPolicy { .. }
                | Action::SetPopulationCrowdingTarget { .. } => {}
                Action::UpdateLandscape { .. }
                | Action::SetGlobalCoupling { .. }
                | Action::SetRoughnessAversion { .. }
                | Action::SetHarmonicityPreference { .. }
                | Action::FollowHarmonicRelations { .. }
                | Action::FollowHarmonicContinuation { .. }
                | Action::SetSupportPreference { .. } => {}
            }
        }
    }

    match scenario.end {
        crate::scenario::ScenarioEnd::Open => {
            if has_finish {
                return Err("Open scenario must not contain a Finish action".into());
            }
        }
        crate::scenario::ScenarioEnd::Finite { duration_sec } => {
            if !has_finish {
                return Err("Scenario has no Finish action".into());
            }
            if duration_sec <= 0.0 {
                return Err("Scenario duration_sec must be > 0".into());
            }
            if let Some(max_time) = scenario
                .events
                .iter()
                .map(|ev| ev.time)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                && duration_sec + f64::EPSILON < max_time
            {
                return Err("Scenario duration_sec is before last event".into());
            }
        }
    }

    let mut prev_order = None;
    for event in &scenario.events {
        let order = event.order;
        if let Some(prev) = prev_order
            && order <= prev
        {
            return Err("Event order is not strictly increasing".to_string());
        }
        prev_order = Some(order);
    }

    Ok(())
}

pub fn run_compile_only(args: crate::cli::Args) -> Result<(), String> {
    let path = Path::new(&args.scenario_path);
    crate::life::modal::register_modal();
    validate_scenario_script_extension(path)?;
    let stop = AtomicBool::new(false);
    let (scenario, session, warnings) = ScriptHost::load_incremental(
        &path.to_string_lossy(),
        args.seed,
        args.duration,
        &stop,
        &args.end_request,
    )
    .map_err(|error| error.to_string())?;
    if let Some(session) = session {
        if args.duration.is_none() {
            return Err("compile-only of an endless scenario requires --duration".into());
        }
        let config =
            AppConfig::load_or_default(args.config.as_deref()).map_err(|e| e.to_string())?;
        let tb = crate::core::timebase::Timebase {
            fs: config.audio.sample_rate as f32,
            hop: config.analysis.hop_size,
        };
        let (mut events, mut actions, mut markers) = (0, 0, 0);
        let result = session.drain(tb, &args.end_request, |item| match item {
            crate::scripting::score::ScoreItem::Event(event) => {
                events += 1;
                actions += event.actions.len();
                eprint!("event t={:.3} order={}", event.time, event.order);
                for action in event.actions {
                    eprint!(" | {action}");
                }
                eprintln!();
            }
            crate::scripting::score::ScoreItem::Marker(marker) => {
                markers += 1;
                eprintln!(
                    "scene t={:.3} order={} name={}",
                    marker.time, marker.order, marker.name
                );
            }
        });
        let status = session.stream.status();
        eprintln!(
            "end: open (evaluated through t={:.9})",
            status.evaluated_through
        );
        for warning in &status.warnings {
            eprintln!(
                "{}:{:?}: warning: {} ({} occurrences)",
                path.display(),
                warning.position,
                warning.message,
                warning.count
            );
        }
        result?;
        eprintln!("Warnings: {}", status.warnings.len());
        eprintln!(
            "OK compile-only: {} (events={events}, actions={actions}, markers={markers})",
            path.display()
        );
        return Ok(());
    }
    validate_scenario(&scenario)?;
    match scenario.end {
        crate::scenario::ScenarioEnd::Open => eprintln!("end: open"),
        crate::scenario::ScenarioEnd::Finite { duration_sec } => {
            eprintln!("end: {duration_sec:.3}s")
        }
    }
    let mut markers = scenario.scene_markers.clone();
    markers.sort_by(|a, b| cmp_time_order(a.time, a.order, b.time, b.order));
    for marker in &markers {
        eprintln!(
            "scene t={:.3} order={} name={}",
            marker.time, marker.order, marker.name
        );
    }
    let mut events = scenario.events.clone();
    events.sort_by(|a, b| cmp_time_order(a.time, a.order, b.time, b.order));
    for event in &events {
        let action_descs: Vec<String> = event.actions.iter().map(ToString::to_string).collect();
        eprintln!(
            "event t={:.3} order={} {}",
            event.time,
            event.order,
            action_descs.join(" | ")
        );
    }
    let event_count = scenario.events.len();
    let action_count: usize = scenario
        .events
        .iter()
        .map(|event| event.actions.len())
        .sum();
    let marker_count = scenario.scene_markers.len();
    for warning in &warnings {
        eprintln!(
            "{}:{:?}: warning: {}",
            path.display(),
            warning.position,
            warning.message
        );
    }
    eprintln!("Warnings: {}", warnings.len());
    eprintln!(
        "OK compile-only: {} (events={}, actions={}, markers={})",
        path.display(),
        event_count,
        action_count,
        marker_count
    );
    Ok(())
}

pub(super) fn load_scenario(
    path: &Path,
    seed: Option<u64>,
    audio: &crate::config::AudioConfig,
    bound: Option<f64>,
    exiting: &AtomicBool,
    end_request: &AtomicBool,
) -> Result<
    (
        String,
        Scenario,
        f32,
        Option<crate::scripting::score::ScriptSession>,
    ),
    String,
> {
    let scenario_label = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("scenario")
        .to_string();
    crate::life::modal::register_modal();
    validate_scenario_script_extension(path)?;
    let (scenario, score, warnings) =
        ScriptHost::load_incremental(&path.to_string_lossy(), seed, bound, exiting, end_request)
            .map_err(|error| error.to_string())?;
    if score.is_none() {
        validate_scenario(&scenario)?;
    }
    for warning in warnings {
        eprintln!(
            "{}:{:?}: warning: {} ({} occurrences)",
            path.display(),
            warning.position,
            warning.message,
            warning.count
        );
    }
    if scenario.remember_pitch_set > 0. && audio.sample_rate != 48000 {
        let position = scenario
            .pitch_set_position
            .map(|(line, column)| format!(":{line}:{column}"))
            .unwrap_or_default();
        return Err(format!(
            "{}{position}: remember_pitch_set requires 48000 Hz",
            path.display()
        ));
    }
    let db = audio.output_gain_db + scenario.output_gain_db;
    let output_gain = 10.0f32.powf(db / 20.0);
    if !db.is_finite() || !output_gain.is_finite() || output_gain <= 0.0 {
        let position = scenario
            .output_gain_position
            .map(|(line, column)| format!(":{line}:{column}"))
            .unwrap_or_default();
        return Err(format!(
            "{}{position}: output_gain + audio.output_gain_db must yield a finite, positive f32 amplitude gain",
            path.display()
        ));
    }
    info!(
        "scenario seed: {} (replay with --seed {})",
        scenario.seed, scenario.seed
    );
    Ok((scenario_label, scenario, output_gain, score))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn validate_run_bounds(
    scenario: &Scenario,
    duration: Option<f64>,
    sample_rate: u32,
    hop: usize,
    render: bool,
    report: bool,
    profile: bool,
) -> Result<(), String> {
    if scenario.end == crate::scenario::ScenarioEnd::Open && duration.is_none() {
        if render {
            return Err("Open scenarios require --duration for render".into());
        }
        if report {
            return Err("--report with an open scenario requires --duration".into());
        }
        if profile {
            return Err("--profile with an open scenario requires --duration".into());
        }
    }
    if let Some(seconds) = duration {
        let samples = seconds * f64::from(sample_rate);
        if !seconds.is_finite() || seconds < 0.0 || samples >= u64::MAX as f64 {
            return Err("--duration must be finite, non-negative and fit the runtime clock".into());
        }
        if render && samples.round() * 2.0 + 36.0 > f64::from(u32::MAX) {
            return Err("--duration alone exceeds the WAV size limit".into());
        }
        if profile
            && (samples.round() as u64).div_ceil(hop as u64)
                > crate::runtime_profile::PROFILE_HOP_CAPACITY as u64
        {
            return Err("--duration exceeds PROFILE_HOP_CAPACITY; leave room for end fades".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod end_tests {
    use super::*;
    use crate::scenario::{ScenarioEnd, TimedEvent};

    #[test]
    fn open_validation_keeps_nonempty_rule_and_forbids_finish() {
        let mut scenario = crate::scripting::ScriptContext::default().scenario;
        scenario.end = ScenarioEnd::Open;
        assert_eq!(
            validate_scenario(&scenario).unwrap_err(),
            "Scenario has no events"
        );
        scenario.events.push(TimedEvent {
            time: 0.0,
            order: 1,
            actions: vec![Action::SetGlobalCoupling { value: 0.0 }],
        });
        validate_scenario(&scenario).unwrap();
        scenario.events[0].actions.push(Action::Finish);
        assert!(
            validate_scenario(&scenario)
                .unwrap_err()
                .contains("must not contain a Finish")
        );
        scenario.end = ScenarioEnd::Finite { duration_sec: 0.1 };
        validate_scenario(&scenario).unwrap();
        scenario.events[0].actions.pop();
        assert_eq!(
            validate_scenario(&scenario).unwrap_err(),
            "Scenario has no Finish action"
        );
    }

    #[test]
    fn duration_bounds_protect_the_actual_output_consumers() {
        let mut scenario = crate::scripting::ScriptContext::default().scenario;
        scenario.end = ScenarioEnd::Open;
        for (render, report, profile, error) in [
            (true, false, false, "for render"),
            (false, true, false, "--report"),
            (false, false, true, "--profile"),
        ] {
            assert!(
                validate_run_bounds(&scenario, None, 48000, 512, render, report, profile)
                    .unwrap_err()
                    .contains(error)
            );
            validate_run_bounds(&scenario, Some(1.0), 48000, 512, render, report, profile).unwrap();
        }
        validate_run_bounds(&scenario, None, 48000, 512, false, false, false).unwrap();
        let capacity = crate::runtime_profile::PROFILE_HOP_CAPACITY as f64;
        let last = capacity * 512.0 / 48000.0;
        validate_run_bounds(&scenario, Some(last), 48000, 512, false, false, true).unwrap();
        assert!(
            validate_run_bounds(
                &scenario,
                Some(last + 512.0 / 48000.0),
                48000,
                512,
                false,
                false,
                true
            )
            .unwrap_err()
            .contains("PROFILE_HOP_CAPACITY")
        );
        let max_wav = f64::from((u32::MAX - 36) / 2) / 48000.0;
        validate_run_bounds(&scenario, Some(max_wav), 48000, 512, true, false, false).unwrap();
        assert!(
            validate_run_bounds(
                &scenario,
                Some(max_wav + 1.0 / 48000.0),
                48000,
                512,
                true,
                false,
                false
            )
            .unwrap_err()
            .contains("WAV size limit")
        );
        for invalid in [f64::NAN, f64::INFINITY, -1.0, f64::MAX] {
            assert!(
                validate_run_bounds(&scenario, Some(invalid), 48000, 512, false, false, false)
                    .is_err()
            );
        }
        scenario.end = ScenarioEnd::Finite { duration_sec: 1.0 };
        validate_run_bounds(&scenario, None, 48000, 512, true, true, true).unwrap();
    }
}
