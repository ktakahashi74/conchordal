//! Offline seven-class evaluation of the ordinary fixed body-energy kernel.

use super::*;
use crate::life::action_candidates::energy::{Window, project_window};
use crate::life::self_prediction::{ScheduledRelease, ToneEnergy};
use crate::life::sound::Tone;
use std::io::BufRead;

#[test]
#[ignore = "registered offline candidate/default fidelity comparison; no live action effects"]
fn acquire_seven_class_energy() {
    let input = std::path::PathBuf::from(std::env::var("CONCHORDAL_I10_ENERGY_INPUTS").unwrap());
    let output = std::path::PathBuf::from(std::env::var("CONCHORDAL_I10_ENERGY_OUTPUT").unwrap());
    let registration: Value =
        serde_json::from_slice(&fs::read(input.join("registration.json")).unwrap()).unwrap();
    assert_eq!(registration["schema"], "i10-seven-class-energy-v1");
    assert_eq!(registration["quadrature_points"], 16);
    let use_coherent = registration["coherent_sine"].as_bool().unwrap_or(false);
    let source = Path::new(registration["source_directory"].as_str().unwrap());
    for (file, digest) in registration["source_sha256"].as_object().unwrap() {
        assert_eq!(
            format!("{:x}", Sha256::digest(fs::read(source.join(file)).unwrap())),
            digest.as_str().unwrap()
        );
    }
    // Target audio is deliberately not opened by the predictor.
    crate::life::modal::register_modal();
    fs::create_dir(&output).expect("fresh prediction directory");
    let time = crate::core::timebase::Timebase {
        fs: 48000.,
        hop: 512,
    };
    let mut cases = Vec::new();
    for id in registration["cases"].as_array().unwrap() {
        let id = id.as_str().unwrap();
        let mut renderer = ScheduleRenderer::new(time);
        let mut current = None;
        for line in std::io::BufReader::new(
            File::open(source.join(id).join("policy-inputs.jsonl")).unwrap(),
        )
        .lines()
        {
            let hop: Hop = serde_json::from_str(&line.unwrap()).unwrap();
            if hop.now >= registration["after_sample"].as_u64().unwrap()
                && hop
                    .batches
                    .iter()
                    .flat_map(|b| &b.tones)
                    .any(|t| t.opportunity.is_some())
            {
                current = Some(hop);
                break;
            }
            renderer.render(&hop.batches, hop.now, &hop.rhythms);
        }
        let Some(current) = current else {
            cases.push(json!({"id": id, "status": "no_granted_recipe_in_recorded_interval"}));
            continue;
        };
        assert_eq!(current.batches.len(), 1);
        let batch = &current.batches[0];
        assert_eq!((batch.source_id, batch.source_generation), (1, 0));
        assert_eq!(batch.tones.len(), 1);
        assert_eq!(batch.cmds.len(), 1);
        let recipe = &batch.tones[0];
        let receipt = recipe.opportunity.unwrap();
        let ToneCmd::On { tone_id, kick } = batch.cmds[0] else {
            panic!("actual grant must be On")
        };
        assert_eq!(tone_id, recipe.tone_id);
        let policy = batch.body_policy.unwrap();
        assert!(policy.at == current.now && policy.is_alive && policy.gate_allows_onset);
        assert_eq!((receipt.issued_at, receipt.at), (current.now, recipe.onset));
        let decision = receipt.at;
        assert!((current.now..current.now + 512).contains(&decision));
        let retained: Vec<_> = renderer
            .source_energy_models(1, 0, current.now, &current.rhythms)
            .take(65)
            .collect();
        assert!(
            retained.len() <= 64,
            "registered corpus exceeds bounded inventory"
        );
        assert!(
            retained
                .iter()
                .all(|(_, _, t)| t.envelope.onset < current.now),
            "queued onset cancellation is not registered"
        );
        let mut onset = Tone::from_parts(
            time,
            recipe.onset,
            recipe.hold_ticks.unwrap_or(60 * 48000),
            recipe.freq_hz,
            recipe.amp,
            Some(recipe.body.clone()),
            Some(recipe.render_modulator.clone()),
            recipe.adsr,
        )
        .unwrap();
        onset.seed_modal_phases(crate::life::schedule_renderer::modal_phase_seed(
            1,
            recipe.onset,
            tone_id,
        ));
        onset.set_smoothing_tau_sec(recipe.smoothing_tau_sec);
        onset.schedule_planned_kick(kick);
        onset.arm_onset_trigger(kick.strength.max(0.));
        let (_, amplitude, envelope) = onset.prediction_parameters(None);
        let issued = ToneEnergy {
            sine: onset.prediction_sine(current.now),
            amplitude,
            envelope,
            control: Some(onset.prediction_control(current.now, &current.rhythms)),
            scheduled_release: receipt.planned_release_at.map(|off| ScheduledRelease {
                apply_at_sample: current.now + (off - current.now) / 512 * 512,
                off_sample: off,
            }),
        };
        let routed = [batch.routing.to_habitat, batch.routing.to_presentation];
        let mut default: Option<Vec<Vec<Window>>> = None;
        let mut candidates = Vec::new();
        let mut predictions = Vec::new();
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
                let at = decision.checked_add(offset).unwrap();
                let active: Vec<_> = retained
                    .iter()
                    .filter(|(_, _, t)| {
                        let envelope = t
                            .scheduled_release
                            .map_or(t.envelope, |p| t.envelope.with_release(p.off_sample));
                        envelope.onset <= at && at < envelope.release_end
                    })
                    .map(|(id, _, _)| *id)
                    .collect();
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
                let added = action.excitation_at.map(|at| {
                    let shift = at - decision;
                    let mut tone = issued;
                    tone.sine = tone.sine.map(|s| {
                        s.for_new_onset(
                            at,
                            crate::life::schedule_renderer::modal_phase_seed(1, at, tone_id),
                        )
                    });
                    tone.envelope.onset += shift;
                    tone.envelope.hold_end += shift;
                    tone.envelope.release_end += shift;
                    let control = tone.control.as_mut().unwrap();
                    control.issued_at = current.now + (at - current.now) / 512 * 512;
                    control.starts_at = control.starts_at.map(|x| x + shift);
                    control.kick_at = control.kick_at.map(|x| x + shift);
                    tone.scheduled_release = receipt.planned_release_at.map(|off| {
                        let off = off + shift;
                        ScheduledRelease {
                            apply_at_sample: current.now + (off - current.now) / 512 * 512,
                            off_sample: off,
                        }
                    });
                    tone
                });
                let intervention = action.release_at.map(|off| ScheduledRelease {
                    apply_at_sample: current.now + (off - current.now) / 512 * 512,
                    off_sample: off,
                });
                let mut buses = Vec::new();
                for bus in 0..2 {
                    let mut windows = Vec::new();
                    for (index, width) in registration["window_samples"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .enumerate()
                    {
                        let width = width.as_u64().unwrap();
                        let mut window = project_window(
                            &retained,
                            added.map(|tone| (routed, tone)),
                            at,
                            (intervention, action.withhold_until),
                            bus,
                            [decision, decision + width],
                            use_coherent,
                        )
                        .unwrap();
                        let total = window.mean;
                        window.default_mean =
                            default.as_ref().map_or(total, |d| d[bus][index].mean);
                        window.difference = total.zip(window.default_mean).map(|(a, b)| a - b);
                        windows.push(window);
                    }
                    buses.push(windows);
                }
                if class == Class::OnsetNow {
                    default = Some(buses.clone());
                }
                assert!(default.is_some());
                predictions.push(json!({"name": name, "input": action, "added": added,
                    "intervention": intervention, "buses": buses}));
            }
        }
        let directory = output.join(id);
        fs::create_dir(&directory).unwrap();
        fs::write(directory.join("predictions.json"), serde_json::to_vec_pretty(&json!({
            "schema": "i10-seven-class-energy-v1", "model": if use_coherent { "source_energy_log1p_residual_v7" } else { "source_energy_log1p_residual_v4" },
            "render_start": current.now, "decision_sample": decision, "receipt": receipt,
            "retained": retained, "issued": issued, "recipe": recipe, "routing": routed,
            "local_default": "onset_now-0", "candidates": candidates, "predictions": predictions,
        })).unwrap()).unwrap();
        println!("{id}: {} conditional energy projections", predictions.len());
        cases.push(json!({"id": id, "status": "predicted", "branches": predictions.len()}));
    }
    fs::write(
        output.join("manifest.json"),
        serde_json::to_vec_pretty(&json!({
            "schema": "i10-seven-class-energy-v1", "registration": registration, "cases": cases,
        }))
        .unwrap(),
    )
    .unwrap();
}
