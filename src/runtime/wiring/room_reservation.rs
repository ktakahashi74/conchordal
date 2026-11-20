//! Upper bound for the union of strict room readers, before starting any DSP.
use crate::core::stream::loo::SourceKind;
use crate::core::timebase::{Tick, Timebase};
use crate::scenario::{
    Action, ArticulationCoreConfig, DurationSpec, PhonationTiming, RespawnPolicy, Scenario,
};
use std::collections::BTreeMap;

#[derive(Default)]
struct Population {
    placements: usize,
    respawn_capacity: usize,
    gated_deaths: bool,
}

fn overflow() -> String {
    "room source reservation exceeds addressable storage".into()
}

// The audio ramps end after ceil(fade * fs) samples. A Voice subtracts a
// control-rate f32 increment from a gain <= 1. When even a one-sample step is
// larger than one ulp, every step reduces it by at least half the increment.
// For longer fades we conservatively keep the slot. Allow dispatch/cleanup
// on either side of rendering, including the last self-sound reader.
fn release_end(kind: SourceKind, at: u64, fade: f32, time: Timebase) -> u64 {
    let samples = f64::from(fade.max(0.0001)) * f64::from(time.fs);
    let samples = if kind == SourceKind::Voice {
        if samples * f64::from(f32::EPSILON) >= 0.5 {
            return u64::MAX;
        }
        2. * samples
    } else {
        f64::from((fade * time.fs).ceil())
    };
    at.saturating_add((samples.ceil() as Tick).div_ceil(time.hop as Tick))
        .saturating_add(2)
}

pub(super) fn source_bound(scenario: &Scenario, fs: f32, hop: usize) -> Result<usize, String> {
    let time = Timebase { fs, hop };
    let mut events: Vec<_> = scenario.events.iter().collect();
    events.sort_by(|a, b| a.time.total_cmp(&b.time).then(a.order.cmp(&b.order)));
    let mut release = BTreeMap::new();
    let mut populations = BTreeMap::<u64, Population>::new();
    let mut edges = Vec::new();
    for event in events.into_iter().rev() {
        let at = time.sec_to_tick(event.time).div_ceil(hop as Tick);
        for action in event.actions.iter().rev() {
            let placement = match action {
                Action::Spawn {
                    population_id,
                    ids,
                    spec,
                    ..
                } => {
                    let population = populations.entry(*population_id).or_default();
                    population.placements = population
                        .placements
                        .checked_add(ids.len())
                        .ok_or_else(overflow)?;
                    // Drone removal and ordinary held-note death flush their notes
                    // at the next phonation pass. Gated Offs may remain pending.
                    let held = matches!(spec.control.phonation.spec.timing, PhonationTiming::Once)
                        && matches!(
                            spec.control.phonation.spec.duration,
                            DurationSpec::WhileAlive
                        );
                    population.gated_deaths |=
                        !held && !matches!(spec.articulation, ArticulationCoreConfig::Drone { .. });
                    Some(((SourceKind::Voice, *population_id), ids.len()))
                }
                Action::PlaceFlora { flora_id, .. } => Some(((SourceKind::Flora, *flora_id), 1)),
                Action::PlaceRain { rain_id, spec, .. }
                    if spec.leave_drops_out && spec.routing.to_habitat =>
                {
                    Some(((SourceKind::Rain, *rain_id), 1))
                }
                Action::SetRespawnPolicy {
                    population_id,
                    capacity,
                    policy,
                    ..
                } if *policy != RespawnPolicy::None => {
                    let population = populations.entry(*population_id).or_default();
                    population.respawn_capacity =
                        population.respawn_capacity.max((*capacity).max(1));
                    None
                }
                Action::ReleasePopulation {
                    population_id,
                    fade_sec,
                } => {
                    release.insert(
                        (SourceKind::Voice, *population_id),
                        release_end(SourceKind::Voice, at, *fade_sec, time),
                    );
                    None
                }
                Action::ReleaseFlora { flora_id, fade_sec } => {
                    release.insert(
                        (SourceKind::Flora, *flora_id),
                        release_end(SourceKind::Flora, at, *fade_sec, time),
                    );
                    None
                }
                Action::ReleaseRain { rain_id, fade_sec } => {
                    release.insert(
                        (SourceKind::Rain, *rain_id),
                        release_end(SourceKind::Rain, at, *fade_sec, time),
                    );
                    None
                }
                _ => None,
            };
            if let Some((key, count)) = placement {
                edges.push((at, true, count));
                if let Some(&end) = release.get(&key) {
                    edges.push((end, false, count));
                }
            }
        }
    }
    edges.sort_unstable();
    let (mut active, mut peak) = (0usize, 0usize);
    for (_, birth, count) in edges {
        if birth {
            active = active.checked_add(count).ok_or_else(overflow)?;
            peak = peak.max(active);
        } else {
            active -= count;
        }
    }
    for population in populations.into_values() {
        // At most one successor per death and at most capacity births per hop.
        let births = population.placements.min(population.respawn_capacity);
        let cohorts = if births > 0 && population.gated_deaths {
            let end = scenario
                .end
                .duration_sec()
                .ok_or("strict room needs a finite bound for retiring gated generations")?;
            usize::try_from(time.sec_to_tick(end).div_ceil(hop as Tick))
                .ok()
                .and_then(|n| n.checked_add(1))
                .ok_or_else(overflow)?
        } else {
            // Self-sound is prepared before cleanup; LOO/metabolism afterwards.
            // On consecutive replacements the oldest reader finishes its last
            // hop while its successor releases and the newest reader is born.
            2
        };
        peak = peak
            .checked_add(births.checked_mul(cohorts).ok_or_else(overflow)?)
            .ok_or_else(overflow)?;
    }
    peak.checked_add(3).ok_or_else(overflow)?;
    Ok(peak)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scripting::ScriptHost;

    #[test]
    fn strict_room_score_bound_includes_release_overlap_and_reuses_retired_slots() {
        let (s, _) = ScriptHost::compile_source(
            r#"
            let a=place(sine().sustain().anchor(),at(330));
            release(a);
            place(flora().fade(0.01),random(220,880));
            wait(6);
            place(rain().leave_drops_out(false),random(220,880));
            place(rain().leave_drops_out(true).send(presentation_bus),random(220,880));
            place(sine().sustain().anchor(),at(440)); wait(1);
        "#,
            Some(5),
        )
        .unwrap();
        assert_eq!(source_bound(&s, 48000., 512).unwrap(), 2);
        let (s, _) = ScriptHost::compile_source(
            r#"
            place(sine().brain("drone").sustain().respawn_random(),at(330).count(2));
            place(rain().leave_drops_out(true),random(220,880)); wait(1);
        "#,
            Some(5),
        )
        .unwrap();
        assert_eq!(source_bound(&s, 48000., 512).unwrap(), 7);
    }
}
