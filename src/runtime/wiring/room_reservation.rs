//! Upper bound for the union of strict room readers, before starting any DSP.
use crate::core::stream::loo::SourceKind;
use crate::core::timebase::{Tick, Timebase};
use crate::scenario::{
    Action, ArticulationCoreConfig, DurationSpec, PhonationTiming, RespawnPolicy, RhythmRelation,
    Scenario, VoiceSpec,
};
use std::collections::BTreeMap;

struct Population {
    placements: usize,
    respawn_capacity: usize,
    // None if any founder can leave a reader with an unproved deadline.
    retiring_cohorts: Option<usize>,
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

// One identity can have self-sound, LOO and metabolic readers. Count each
// lineage once plus its overlapping predecessors, irrespective of reader count.
fn retiring_cohorts(
    spec: &VoiceSpec,
    scenario: &Scenario,
    time: Timebase,
    normal_meter: bool,
) -> Option<usize> {
    let control = &spec.control.phonation;
    if matches!(spec.articulation, ArticulationCoreConfig::Drone { .. })
        || (matches!(control.spec.timing, PhonationTiming::Once)
            && matches!(control.spec.duration, DurationSpec::WhileAlive)
            && control.rest_periods.is_none())
    {
        return Some(2);
    }
    let PhonationTiming::Coupled(clock) = control.spec.timing else {
        return None;
    };
    let DurationSpec::Gates(gates) = control.spec.duration else {
        return None;
    };
    if crate::core::felt_config::FeltConfig::weight(control) != 0. {
        return None;
    }
    if !normal_meter
        || scenario.expression.is_some()
        || clock.relation != RhythmRelation::Synchronized
        || clock.flow_depth != 0.
        || control.rest_on_habituation != 0.
        || control.rest_periods.is_some()
        || control.spec.sustain_participation
        || !(1..crate::life::phonation_engine::MAX_EXTRA_GATES).contains(&u64::from(gates))
    {
        return None;
    }
    let (lo, hi) = scenario.meter_shaping.basin_hz?;
    let (lo, hi) = crate::core::meter::sane_basin(lo, hi);
    // The founder and respawn template must use the production basin.
    // sane_basin is within 0.5..4 Hz, inside the unclamped regular-IOI range.
    let base = clock.base_rate_hz;
    if !(lo..=hi).contains(&base) {
        return None;
    }
    // Both the convex blend and its reciprocal use f32. Round the
    // endpoints out to quarter Hz, including the uninitialized 2 Hz beat.
    let lower = ((lo.min(base).min(2.) - 8. * f32::EPSILON * base.max(2.)) * 4.).floor() / 4.;
    let upper =
        ((hi.max(base).max(2.) + 8. * f32::EPSILON * hi.max(base).max(2.)) * 4.).ceil() / 4.;
    let hop = time.hop as u64;
    if lower <= 0. || 1.5 * f64::from(upper) * hop as f64 >= f64::from(time.fs) {
        return None;
    }
    // At most one crossing per hop. The On's absolute PendingOff is at
    // most one current hop plus 'gates' extrapolated fixed periods away.
    // Natural death stops Ons; forced removal flushes notes next pass.
    // A population release disables births: its living members' fade is
    // already covered by the placement edges, independently of this tail.
    let period = (f64::from(time.fs) / f64::from(lower)).ceil() as u64;
    let samples = period.checked_mul(u64::from(gates))?.checked_add(hop)?;
    let d = samples.div_ceil(hop).checked_add(2)?;
    usize::try_from(d.checked_add(1)?).ok()
}

impl Default for Population {
    fn default() -> Self {
        Self {
            placements: 0,
            respawn_capacity: 0,
            retiring_cohorts: Some(0),
        }
    }
}

pub(super) fn source_bound(
    scenario: &Scenario,
    fs: f32,
    hop: usize,
    normal_meter: bool,
) -> Result<usize, String> {
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
                    // ControlUpdate has no clock, gate length or rest patch.
                    population.retiring_cohorts = population
                        .retiring_cohorts
                        .zip(retiring_cohorts(spec, scenario, time, normal_meter))
                        .map(|(a, b)| a.max(b));
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
        if births == 0 {
            continue;
        }
        let score_cohorts = scenario
            .end
            .duration_sec()
            .map(|end| time.sec_to_tick(end).div_ceil(hop as Tick));
        let cohorts = match (population.retiring_cohorts, score_cohorts) {
            (Some(n), Some(last)) => n.min(
                usize::try_from(last)
                    .unwrap_or(usize::MAX)
                    .saturating_add(1),
            ),
            (Some(n), None) => n,
            (None, Some(last)) => usize::try_from(last)
                .ok()
                .and_then(|n| n.checked_add(1))
                .ok_or_else(overflow)?,
            (None, None) => {
                return Err(
                    "strict room needs a finite bound for retiring gated generations".into(),
                );
            }
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
    fn sample_colony_deadline_and_finite_fallback() {
        let (s, _) = ScriptHost::compile_source(
            include_str!("../../../samples/12_emergence_and_resolution.rhai"),
            Some(21),
        )
        .unwrap();
        let bound = |s: &Scenario, normal| source_bound(s, 48_000., 512, normal).unwrap() + 3;
        assert_eq!(bound(&s, true), 18 + 8 * (164 + 1) + 3);
        assert_eq!(bound(&s, false), 27_181);
        let time = Timebase {
            fs: 48_000.,
            hop: 512,
        };
        let colony = s
            .events
            .iter()
            .flat_map(|e| &e.actions)
            .find_map(|action| match action {
                Action::Spawn { spec, ids, .. } if ids.len() == 8 => Some(spec),
                _ => None,
            })
            .unwrap();
        for edit in [
            #[cfg(feature = "research")]
            0,
            1,
            2,
            3,
            4,
        ] {
            let mut spec = colony.clone();
            let ctl = &mut spec.control.phonation;
            match edit {
                #[cfg(feature = "research")]
                0 => ctl.syncopate = 1.,
                1 => ctl.rest_on_habituation = 1.,
                2 => ctl.spec.sustain_participation = true,
                3 => ctl.spec.duration = DurationSpec::Field(Default::default()),
                _ => {
                    if let PhonationTiming::Coupled(timing) = &mut ctl.spec.timing {
                        timing.flow_depth = 1.;
                    }
                }
            }
            assert_eq!(retiring_cohorts(&spec, &s, time, true), None);
        }
        let mut expressive = s.clone();
        expressive.expression = Some(crate::scenario::expression::Expression {
            start: 0.,
            end: 5.,
            arch: 0.,
            ritard: 0.5,
            position: (1, 1),
            from_profile: false,
        });
        assert_eq!(bound(&expressive, true), 27_181);
        let mut faster = s.clone();
        faster.meter_shaping.basin_hz = Some((3., 4.));
        let mut fast_voice = colony.clone();
        if let PhonationTiming::Coupled(timing) = &mut fast_voice.control.phonation.spec.timing {
            timing.base_rate_hz = 3.5;
        }
        // A valid basin above 2 Hz still has the initialization fallback.
        assert!(retiring_cohorts(&fast_voice, &faster, time, true).unwrap() >= 3 * 24_000 / 512);
        let mut unknown = s.clone();
        unknown.meter_shaping.basin_hz = None;
        assert_eq!(bound(&unknown, true), 27_181);
        unknown.end = crate::scenario::ScenarioEnd::Open;
        assert!(source_bound(&unknown, 48_000., 512, true).is_err());
    }

    #[test]
    fn simultaneous_population_release_and_entry_include_all_readers() {
        let (mut s, _) = ScriptHost::compile_source(
            "let a=place(sine().sustain(),at(330).count(3)); wait(1); release(a); place(sine().sustain(),at(440).count(4)); wait(1);",
            Some(7)
        ).unwrap();
        assert_eq!(source_bound(&s, 48_000., 512, true).unwrap(), 7);
        for action in s.events.iter_mut().flat_map(|e| &mut e.actions) {
            if let Action::Spawn { ids, .. } = action {
                *ids = crate::scenario::SpawnIds::Allocate(usize::MAX);
                break;
            }
        }
        assert_eq!(source_bound(&s, 48_000., 512, true), Err(overflow()));
    }

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
        assert_eq!(source_bound(&s, 48000., 512, true).unwrap(), 2);
        let (s, _) = ScriptHost::compile_source(
            r#"
            place(sine().brain("drone").sustain().respawn_random(),at(330).count(2));
            place(rain().leave_drops_out(true),random(220,880)); wait(1);
        "#,
            Some(5),
        )
        .unwrap();
        assert_eq!(source_bound(&s, 48000., 512, true).unwrap(), 7);
    }
}
