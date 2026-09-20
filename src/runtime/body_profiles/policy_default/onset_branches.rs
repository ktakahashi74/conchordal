//! Conditional seven-class extensions at a recorded real onset opportunity.

use super::*;

#[test]
#[ignore = "explicit offline recipe intervention; requires hash-registered receipts and fresh output"]
fn acquire_onset_branches() {
    let input = std::path::PathBuf::from(std::env::var("CONCHORDAL_I10_ONSET_INPUTS").unwrap());
    let output = std::path::PathBuf::from(std::env::var("CONCHORDAL_I10_ONSET_OUTPUT").unwrap());
    let registration: Value =
        serde_json::from_slice(&fs::read(input.join("registration.json")).unwrap()).unwrap();
    assert_eq!(registration["schema"], "i10-onset-branches-v1");
    let source = Path::new(registration["source_directory"].as_str().unwrap());
    for (name, digest) in registration["source_sha256"].as_object().unwrap() {
        assert_eq!(
            format!("{:x}", Sha256::digest(fs::read(source.join(name)).unwrap())),
            digest.as_str().unwrap(),
            "registered input changed: {name}"
        );
    }
    crate::life::modal::register_modal();
    fs::create_dir(&output).expect("fresh intervention directory required");
    let mut cases = Vec::new();
    for id in registration["cases"].as_array().unwrap() {
        let id = id.as_str().unwrap();
        let source_case = source.join(id);
        let hops: Vec<Hop> = fs::read_to_string(source_case.join("policy-inputs.jsonl"))
            .unwrap()
            .lines()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect();
        let Some(index) = hops.iter().position(|hop| {
            hop.now >= registration["after_sample"].as_u64().unwrap()
                && hop
                    .batches
                    .iter()
                    .flat_map(|b| &b.tones)
                    .any(|t| t.opportunity.is_some())
        }) else {
            cases.push(json!({"id": id, "status": "no_granted_recipe_in_recorded_interval"}));
            continue;
        };
        let directory = output.join(id);
        fs::create_dir(&directory).unwrap();
        // Do not pass later policy inputs to the intervention acquisition.
        acquire(&hops[..=index], &directory, &registration);
        let metadata: Value =
            serde_json::from_slice(&fs::read(directory.join("branches.json")).unwrap()).unwrap();
        let start = metadata["render_start"].as_u64().unwrap();
        let original_issue = 24576_u64;
        assert!(start >= original_issue);
        for bus in 0..2 {
            let actual =
                fs::read(source_case.join(format!("actual-default-bus{bus}.f32le"))).unwrap();
            let local = fs::read(directory.join(format!("onset_now-0-bus{bus}.f32le"))).unwrap();
            let from = (start - original_issue) as usize * 4;
            assert_eq!(
                &local[..512 * 4],
                &actual[from..from + 512 * 4],
                "actual onset hop: {id}/{bus}"
            );
            let prefix = fs::read(source_case.join(format!("prefix-bus{bus}.f32le"))).unwrap();
            let reconstructed = fs::read(directory.join(format!("prefix-bus{bus}.f32le"))).unwrap();
            assert_eq!(&reconstructed[..prefix.len()], prefix);
            assert_eq!(&reconstructed[prefix.len()..], &actual[..from]);
        }
        println!(
            "{id}: {} legal branches; actual onset hop and prefix exact",
            metadata["branches"].as_array().unwrap().len()
        );
        cases.push(
            json!({"id": id, "status": "acquired", "decision_sample": metadata["decision_sample"],
            "branches": metadata["branches"].as_array().unwrap().len()}),
        );
    }
    fs::write(
        output.join("manifest.json"),
        serde_json::to_vec_pretty(&json!({
            "schema": "i10-onset-branches-v1", "registration": registration, "cases": cases
        }))
        .unwrap(),
    )
    .unwrap();
}

fn acquire(prefix_and_decision: &[Hop], directory: &Path, registration: &Value) {
    let current = prefix_and_decision.last().unwrap();
    assert_eq!(current.batches.len(), 1);
    let selected = &current.batches[0];
    assert_eq!((selected.source_id, selected.source_generation), (1, 0));
    assert_eq!(selected.tones.len(), 1);
    assert_eq!(
        selected.cmds.len(),
        1,
        "simultaneous policy actions need a separate registration"
    );
    let recipe = &selected.tones[0];
    let receipt = recipe.opportunity.unwrap();
    let ToneCmd::On { tone_id, kick } = selected.cmds[0] else {
        panic!("actual onset required")
    };
    assert_eq!(tone_id, recipe.tone_id);
    assert_eq!((receipt.issued_at, receipt.at), (current.now, recipe.onset));
    assert!((current.now..current.now + 512).contains(&receipt.at));
    let policy = selected.body_policy.unwrap();
    assert!(policy.is_alive && policy.gate_allows_onset && policy.at == current.now);
    let decision = receipt.at;
    let render_end = (decision + HORIZON).div_ceil(512) * 512;
    let mut renderer = ScheduleRenderer::new(crate::core::timebase::Timebase {
        fs: 48000.,
        hop: 512,
    });
    let mut prefix = [Vec::new(), Vec::new()];
    let mut planned_offs = Vec::new();
    for hop in &prefix_and_decision[..prefix_and_decision.len() - 1] {
        assert_eq!(hop.now as usize, prefix[0].len());
        for batch in &hop.batches {
            assert_eq!((batch.source_id, batch.source_generation), (1, 0));
            for tone in &batch.tones {
                if let Some(at) = tone.opportunity.and_then(|r| r.planned_release_at)
                    && at >= current.now
                {
                    planned_offs.push((tone.tone_id, at));
                }
            }
        }
        let frame = renderer.render(&hop.batches, hop.now, &hop.rhythms);
        prefix[0].extend_from_slice(frame.habitat);
        prefix[1].extend_from_slice(frame.presentation);
    }
    assert_eq!(prefix[0].len() as u64, current.now);
    for (bus, pcm) in prefix.iter().enumerate() {
        write_pcm(&directory.join(format!("prefix-bus{bus}.f32le")), pcm);
    }
    planned_offs.sort_unstable_by_key(|&(tone, at)| (at, tone));
    planned_offs.dedup();
    let envelopes = renderer.source_envelopes(1, 0).collect::<Vec<_>>();
    assert!(
        envelopes.iter().all(|(_, e)| e.onset < current.now),
        "queued onset cancellation is not registered"
    );
    let planned = envelopes
        .iter()
        .map(|&(tone, mut envelope)| {
            for &(_, at) in planned_offs.iter().filter(|&&(id, _)| id == tone) {
                envelope = envelope.with_release(at);
            }
            (tone, envelope)
        })
        .collect::<Vec<_>>();
    let mut candidates = Vec::new();
    let mut branches = Vec::new();
    let mut default_pcm: Option<[Vec<f32>; 2]> = None;
    for class in [
        Class::OnsetNow,
        Class::DelayedOnset,
        Class::Wait,
        Class::Skip,
        Class::Continue,
        Class::Release,
        Class::Gap,
    ] {
        for offset in registration["offsets_samples"].as_array().unwrap() {
            let offset = offset.as_u64().unwrap();
            assert!(offset < HORIZON);
            let at = decision + offset;
            let active = planned
                .iter()
                .filter(|(_, e)| e.onset <= at && at < e.release_end)
                .map(|(id, _)| *id)
                .collect::<Vec<_>>();
            let action = class.input(
                decision,
                at,
                receipt.intrinsic_period_ticks,
                48000,
                BodyState {
                    permits_action: Some(true),
                    active_at_candidate: Some(!active.is_empty()),
                    pending_opportunity: true,
                    due_unconsumed: receipt.intrinsic_due_at == Some(decision),
                },
            );
            let name = format!(
                "{}-{offset}",
                serde_json::to_value(class).unwrap().as_str().unwrap()
            );
            candidates.push(json!({"name": name, "active_tones": active, "input": action}));
            let Some(action) = action else { continue };
            let mut fork = renderer.fork_source(1);
            let mut rhythms = current.rhythms;
            let mut off_commands = planned_offs.clone();
            if let Some(at) = action.release_at {
                off_commands.extend(active.iter().map(|&tone| (tone, at)));
            }
            let onset = action.excitation_at.map(|at| {
                let mut spec = recipe.clone();
                spec.opportunity = None;
                spec.onset = at;
                if let Some(off) = receipt.planned_release_at {
                    off_commands.push((tone_id, at + off.checked_sub(decision).unwrap()));
                }
                spec
            });
            off_commands.sort_unstable_by_key(|&(tone, at)| (at, tone));
            off_commands.dedup();
            let mut pcm = [Vec::new(), Vec::new()];
            for now in (current.now..render_end).step_by(512) {
                let mut batch = PhonationBatch {
                    source_id: 1,
                    source_generation: 0,
                    routing: selected.routing,
                    ..Default::default()
                };
                // Deliver a pending old off in its original hop, before a new excitation.
                for &(id, tick) in &off_commands {
                    if id != tone_id && now <= tick && tick < now + 512 {
                        batch.cmds.push(ToneCmd::Off {
                            tone_id: id,
                            off_tick: tick,
                        });
                    }
                }
                if let Some(spec) = &onset
                    && now <= spec.onset
                    && spec.onset < now + 512
                {
                    batch.cmds.push(ToneCmd::On { tone_id, kick });
                    batch.tones.push(spec.clone());
                }
                for &(id, tick) in &off_commands {
                    if id == tone_id && now <= tick && tick < now + 512 {
                        batch.cmds.push(ToneCmd::Off {
                            tone_id: id,
                            off_tick: tick,
                        });
                    }
                }
                let frame = fork.render(&[batch], now, &rhythms);
                pcm[0].extend_from_slice(frame.habitat);
                pcm[1].extend_from_slice(frame.presentation);
                for _ in 0..512 {
                    rhythms.advance_in_place(1. / 48000.);
                }
            }
            if class == Class::OnsetNow {
                default_pcm = Some(pcm.clone());
            }
            let default = default_pcm
                .as_ref()
                .expect("local default must precede alternatives");
            let start = (decision - current.now) as usize;
            let mut buses = Vec::new();
            for bus in 0..2 {
                write_pcm(&directory.join(format!("{name}-bus{bus}.f32le")), &pcm[bus]);
                assert_eq!(
                    &pcm[bus][..start],
                    &default[bus][..start],
                    "pre-decision audio"
                );
                let energy = pcm[bus]
                    .chunks_exact(512)
                    .map(|chunk| chunk.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / 512.)
                    .collect::<Vec<_>>();
                let mut paired = Vec::new();
                for samples in [12000, 48000, 96000, 192000] {
                    let means = [&pcm[bus], &default[bus]].map(|wave| {
                        wave[start..start + samples]
                            .iter()
                            .map(|x| f64::from(*x).powi(2))
                            .sum::<f64>()
                            / samples as f64
                    });
                    paired.push(json!({"samples": samples, "mean_square": means[0],
                        "default_mean_square": means[1], "difference": means[0] - means[1]}));
                }
                buses.push(json!({"bus": bus, "energy_per_hop": energy, "paired": paired}));
            }
            branches.push(json!({"name": name, "input": action, "onset": onset,
                "off_commands": off_commands, "buses": buses}));
        }
    }
    fs::write(directory.join("branches.json"), serde_json::to_vec_pretty(&json!({
        "schema": "i10-onset-branches-v1", "render_start": current.now,
        "decision_sample": decision, "render_end": render_end, "sample_rate": 48000, "hop_samples": 512,
        "recipe": serde_json::to_value(recipe).unwrap(), "kick": serde_json::to_value(kick).unwrap(),
        "routing": selected.routing, "policy": policy, "rhythms": serde_json::to_value(current.rhythms).unwrap(),
        "envelopes": envelopes, "planned_envelopes": planned, "planned_offs": planned_offs,
        "local_default": "onset_now-0", "candidates": candidates, "branches": branches,
        "scope": "Fixed body recipe and current grant, issue-known existing releases, extrapolated issue rhythm, no later excitations or feedback. Wait/skip/gap bookkeeping is declared, not executed by the live policy. Mean-square differences are acoustic targets, not ordinal preference or transfer accuracy."
    })).unwrap()).unwrap();
}
