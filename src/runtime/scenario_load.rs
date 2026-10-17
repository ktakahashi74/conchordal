use super::*;

#[inline]
pub(super) fn cmp_time_order(
    a_time: f32,
    a_order: u64,
    b_time: f32,
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
    let scenario = ScriptHost::load_script(&path_str, seed).map_err(|e| {
        let pos = e
            .position
            .map(|pos| format!(" (line {})", pos.line().unwrap_or(0)))
            .unwrap_or_default();
        format!(
            "Failed to run scenario script {}: {}{pos}",
            script_path.display(),
            e.message
        )
    })?;
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
                Action::Spawn { .. }
                | Action::UpdatePopulation { .. }
                | Action::ReleasePopulation { .. }
                | Action::SetRespawnPolicy { .. }
                | Action::SetPopulationCrowdingTarget { .. } => {}
                Action::UpdateLandscape { .. }
                | Action::SetGlobalCoupling { .. }
                | Action::SetRoughnessAversion { .. } => {}
            }
        }
    }

    if !has_finish {
        return Err("Scenario has no Finish action".to_string());
    }

    if scenario.duration_sec <= 0.0 {
        return Err("Scenario duration_sec must be > 0".to_string());
    }
    if let Some(max_time) = scenario
        .events
        .iter()
        .map(|ev| ev.time)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        && scenario.duration_sec + f32::EPSILON < max_time
    {
        return Err("Scenario duration_sec is before last event".to_string());
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
    let scenario = compile_scenario_from_script(path, args.seed)?;
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
    eprintln!(
        "OK compile-only: {} (events={}, actions={}, markers={})",
        path.display(),
        event_count,
        action_count,
        marker_count
    );
    Ok(())
}

pub(super) fn load_scenario(path: &Path, seed: Option<u64>) -> Result<(String, Scenario), String> {
    let scenario_label = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("scenario")
        .to_string();
    let scenario = compile_scenario_from_script(path, seed)?;
    info!(
        "scenario seed: {} (replay with --seed {})",
        scenario.seed, scenario.seed
    );
    Ok((scenario_label, scenario))
}
