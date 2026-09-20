//! Issue-known bodily extensions; no future policy commands enter this renderer.

use super::*;
use crate::life::sound::envelope::Envelope;

#[derive(serde::Serialize, serde::Deserialize)]
pub(super) struct Snapshot {
    issue: u64,
    period: Option<u64>,
    rhythms: NeuralRhythms,
    planned_releases: Vec<(u64, u64)>,
    envelopes: Vec<(u64, Envelope)>,
}

impl Snapshot {
    pub(super) fn capture(state: &WorkerState, issue: u64) -> Self {
        let voice = state.pop.voices.first().unwrap();
        let mut planned_releases = voice
            .phonation_engine
            .planned_releases()
            .collect::<Vec<_>>();
        planned_releases.sort_unstable_by_key(|&(tone, at)| (at, tone));
        assert!(planned_releases.iter().all(|&(_, at)| at >= issue));
        let mut rhythms = state.current_landscape.rhythm;
        for _ in 0..512 {
            rhythms.advance_in_place(1. / 48000.);
        }
        Self {
            issue,
            period: voice
                .phonation_engine
                .clock
                .intrinsic_period_sec()
                .filter(|p| p.is_finite() && *p > 0.)
                .map(|p| (p * 48000.).round() as u64)
                .filter(|p| *p > 0),
            rhythms,
            planned_releases,
            envelopes: state.schedule_renderer.source_envelopes(1, 0).collect(),
        }
    }
}

pub(super) fn acquire(
    snapshot: &Snapshot,
    actual: &ScheduleRenderer,
    replay: &ScheduleRenderer,
    directory: &Path,
    registration: &Value,
) {
    assert_eq!(registration["schema"], "i10-local-continuation-v1");
    let directory = directory.join("local-continuation");
    fs::create_dir(&directory).unwrap();
    let path = directory.join("issue.json");
    fs::write(
        &path,
        serde_json::to_vec_pretty(&serde_json::to_value(snapshot).unwrap()).unwrap(),
    )
    .unwrap();
    let snapshot: Snapshot = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    let issue = snapshot.issue;
    // A planned off is sent in its actual hop: early submission can clip an attack.
    let planned = snapshot
        .envelopes
        .iter()
        .map(|&(tone, mut envelope)| {
            for &(_, at) in snapshot
                .planned_releases
                .iter()
                .filter(|&&(id, _)| id == tone)
            {
                envelope = envelope.with_release(at);
            }
            (tone, envelope)
        })
        .collect::<Vec<_>>();
    let mut branches = vec![
        ("planned-control".to_string(), None, Vec::new()),
        ("unplanned-control".to_string(), None, Vec::new()),
    ];
    let mut candidates = Vec::new();
    for class in [Class::Continue, Class::Release, Class::Gap] {
        for offset in registration["offsets_samples"].as_array().unwrap() {
            let offset = offset.as_u64().unwrap();
            if class == Class::Continue && offset != 0 {
                continue;
            }
            let at = issue + offset;
            let active = planned
                .iter()
                .filter(|(_, e)| e.onset <= at && at < e.release_end)
                .map(|(id, _)| *id)
                .collect::<Vec<_>>();
            let input = class.input(
                issue,
                at,
                snapshot.period,
                48000,
                BodyState {
                    // Local acoustic intervention, not a live decision authorization.
                    permits_action: Some(true),
                    active_at_candidate: Some(!active.is_empty()),
                    pending_opportunity: false,
                    due_unconsumed: false,
                },
            );
            let name = format!(
                "{}-{offset}",
                serde_json::to_value(class).unwrap().as_str().unwrap()
            );
            candidates.push(json!({"name": name, "class": class, "at": at,
                "active_tones": active, "input": input}));
            if let Some(input) = input {
                let extra = input.release_at.map_or_else(Vec::new, |at| {
                    active.iter().map(|&tone| (tone, at)).collect()
                });
                branches.push((name, Some(input), extra));
            }
        }
    }
    let mut records = Vec::new();
    for (name, input, extra) in branches {
        let mut commands = snapshot.planned_releases.clone();
        if name == "unplanned-control" {
            commands.clear();
        }
        commands.extend(extra);
        commands.sort_unstable_by_key(|&(tone, at)| (at, tone));
        commands.dedup();
        let mut renderer = actual.fork_source(1);
        let mut prefix_replay = replay.fork_source(1);
        let mut rhythms = snapshot.rhythms;
        let mut pcm = [Vec::new(), Vec::new()];
        for now in (issue..issue + HORIZON).step_by(512) {
            let batch = PhonationBatch {
                source_id: 1,
                source_generation: 0,
                cmds: commands
                    .iter()
                    .filter(|&&(_, at)| now <= at && at < now + 512)
                    .map(|&(tone_id, off_tick)| ToneCmd::Off { tone_id, off_tick })
                    .collect(),
                ..Default::default()
            };
            let frame = renderer.render(std::slice::from_ref(&batch), now, &rhythms);
            let check = prefix_replay.render(std::slice::from_ref(&batch), now, &rhythms);
            assert_eq!(frame.habitat, check.habitat, "local habitat replay: {name}");
            assert_eq!(
                frame.presentation, check.presentation,
                "local presentation replay: {name}"
            );
            pcm[0].extend_from_slice(frame.habitat);
            pcm[1].extend_from_slice(frame.presentation);
            for _ in 0..512 {
                rhythms.advance_in_place(1. / 48000.);
            }
        }
        let mut buses = Vec::new();
        for (bus, samples) in pcm.iter().enumerate() {
            write_pcm(&directory.join(format!("{name}-bus{bus}.f32le")), samples);
            let energy = samples
                .chunks_exact(512)
                .map(|chunk| chunk.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / 512.)
                .collect::<Vec<_>>();
            buses.push(json!({"bus": bus, "energy_per_hop": energy}));
        }
        records.push(json!({"name": name, "input": input, "commands": commands, "buses": buses}));
    }
    fs::write(directory.join("branches.json"), serde_json::to_vec_pretty(&json!({
        "schema": "i10-local-continuation-v1", "registration": registration,
        "planned_envelopes": planned, "candidates": candidates, "branches": records,
        "not_acquired": ["onset_now", "delayed_onset", "wait", "skip"],
        "scope": "Issue-known existing body and planned releases, frozen rhythm extrapolation, no later excitation. Local acoustic interventions, not live permissions or closed-loop policy counterfactuals. Planned-control is a local continuation only when active; actual-default remains a separate future policy teacher."
    })).unwrap()).unwrap();
}
