use super::*;
use crate::life::action_observation::{BusOutcome, Observer};
use crate::life::voice::PhonationBatch;
use crate::temporal_cognition::body::{Record, Snapshot};

fn snapshot(end: u64, generation: u32, mask: u8, value: f64) -> Snapshot {
    let mut records = [Record::default(); 128];
    records[0] = Record {
        source_id: 1,
        source_generation: 0,
        body_generation: generation,
        end,
        available: end + 2,
        raw_values: [value; 6],
        coverage: [1.; 6],
        mask,
        active: true,
        ..Record::default()
    };
    Snapshot {
        prototype_model_version: None,
        prototype_assignments: [None; 128],
        config: crate::config::TemporalBodyConfig {
            means: [0.; 6],
            deviations: [1.; 6],
            accent_means: [0.; 2],
            accent_deviations: [1.; 2],
        },
        records,
        version: 1,
        input_end: end,
        processed_frames: 1,
        invalid_hops: 0,
        capture_drops: 0,
        outside_voice_hops: 0,
        processing_us: 0,
        max_processing_us: 0,
        worker_resources: Default::default(),
        finished: false,
    }
}

fn input(generation: u32) -> Input {
    Input {
        scheduled_release: None,
        control: Some(crate::life::sound::control_forecast::ControlForecast {
            issued_at: 0,
            valid_until: None,
            amplitude_smoothing: None,
            amplitude_updates: None,
            sample_dt: 1. / 8000.,
            starts_at: Some(0),
            kick_at: None,
            model: crate::life::sound::control_forecast::AmplitudeModel::Unmodulated { gain: 1. },
        }),
        retained_energy: [Default::default(); 2],
        coherent_energy: [[None; 16]; 2],
        sine: None,
        bank: None,
        body_generation: generation,
        descriptors: [[None; 6]; 2],
        descriptor_support: [None; 2],
        frequency_hz: 220.,
        amplitude: 0.2,
        envelope: crate::life::sound::envelope::Envelope {
            onset: 0,
            hold_end: 80,
            release_end: 100,
            attack_ticks: 10,
            decay_ticks: 0,
            sustain_level: 1.,
            decay_lambda: 0.,
            release_ticks: 20,
        },
        descriptor_target_end: 100,
        descriptor_slot: 0,
    }
}

#[test]
fn source_energy_scores_frozen_windows_and_respects_retired_models() {
    let mut bank = ModelBank::new();
    let mut first = outcome(1, ActionKind::Onset, true);
    let mut pending = outcome(2, ActionKind::Release, true);
    first.prediction = bank.issue(&first, input(1));
    pending.prediction = bank.issue(&pending, input(1));
    let frozen = pending.prediction.unwrap().source_energy;
    first.prediction.as_mut().unwrap().source_energy[0]
        .target
        .fill(Some(0.04));
    bank.observe(&mut first);
    let later = bank
        .issue(&outcome(3, ActionKind::Onset, true), input(1))
        .unwrap();
    assert_ne!(
        later.source_energy[0].predictions,
        first.prediction.unwrap().source_energy[0].predictions
    );
    assert_eq!(later.source_energy[0].updates_at_issue, 1);
    pending.prediction.as_mut().unwrap().source_energy[0]
        .target
        .fill(Some(0.01));
    bank.issue(&outcome(4, ActionKind::Onset, true), input(2));
    bank.observe(&mut pending);
    let recorded = pending.prediction.unwrap().source_energy[0];
    assert_eq!(recorded.predictions, frozen[0].predictions);
    assert!(!recorded.learned);
    assert_eq!(bank.stats.energy.retired, 1);
    for k in 0..16 {
        for model in 0..3 {
            assert_eq!(
                recorded.squared_error[model][k],
                Some((0.01 - frozen[0].predictions[model][k].unwrap()).powi(2))
            );
        }
    }
    assert_eq!(
        bank.issue(&outcome(5, ActionKind::Onset, true), input(2))
            .unwrap()
            .source_energy[0]
            .updates_at_issue,
        0
    );
}

#[test]
fn source_energy_empty_windows_silence_cancel_and_action_blind_control() {
    let mut bank = ModelBank::new();
    let mut out = outcome(1, ActionKind::Onset, false);
    out.window_end_sample = Some(3);
    out.prediction = bank.issue(&out, input(1));
    let energy = &mut out.prediction.as_mut().unwrap().source_energy[0];
    assert_eq!(energy.windows.iter().filter(|w| w[1] > w[0]).count(), 3);
    assert_eq!(energy.windows.first().unwrap()[0], 0);
    assert_eq!(energy.windows.last().unwrap()[1], 3);
    for (k, window) in energy.windows.iter().enumerate() {
        if window[1] > window[0] {
            energy.target[k] = Some(0.);
        } else {
            assert!(energy.predictions.iter().all(|p| p[k].is_none()));
        }
    }
    bank.observe(&mut out);
    assert_eq!(bank.stats.energy.scored_windows, 3);
    assert!(out.prediction.unwrap().source_energy[0].learned);
    let mut cancelled = outcome(2, ActionKind::Onset, true);
    cancelled.prediction = bank.issue(&cancelled, input(1));
    cancelled.prediction.as_mut().unwrap().source_energy[0]
        .target
        .fill(Some(1.));
    cancelled.command_status = "cancelled";
    bank.observe(&mut cancelled);
    assert_eq!(bank.stats.energy.scored_windows, 3);
    assert!(!cancelled.prediction.unwrap().source_energy[0].learned);
    let mut changed = input(1);
    changed.amplitude = 0.8;
    let a = bank
        .issue(&outcome(3, ActionKind::Onset, true), input(1))
        .unwrap();
    let b = bank
        .issue(&outcome(4, ActionKind::Release, true), changed)
        .unwrap();
    assert_ne!(
        a.source_energy[0].predictions[1],
        b.source_energy[0].predictions[1]
    );
    assert_eq!(
        a.source_energy[0].predictions[2],
        b.source_energy[0].predictions[2]
    );
}

#[test]
fn source_energy_learns_changed_action_correspondence_against_both_controls() {
    let mut bank = ModelBank::new();
    for reversed in [false, true] {
        let mut loss = [0.; 3];
        for trial in 0..1600 {
            let onset = trial % 2 == 0;
            let mut out = outcome(
                trial + 1,
                if onset {
                    ActionKind::Onset
                } else {
                    ActionKind::Release
                },
                true,
            );
            out.prediction = bank.issue(&out, input(1));
            let target: f64 = if onset ^ reversed { 0.04 } else { 0.0001 };
            out.prediction.as_mut().unwrap().source_energy[0]
                .target
                .fill(Some(target));
            bank.observe(&mut out);
            if trial >= 1400 {
                let energy = out.prediction.unwrap().source_energy[0];
                for (comparison, sum) in loss.iter_mut().enumerate() {
                    for prediction in energy.predictions[comparison].into_iter().flatten() {
                        *sum +=
                            ((target / 1e-6).ln_1p() - (prediction / 1e-6).ln_1p()).powi(2) / 3200.;
                    }
                }
            }
        }
        assert!(
            loss[1] < 0.05 && loss[0] > 1. && loss[2] > 1.,
            "{reversed}: {loss:?}"
        );
    }
}

fn outcome(id: u64, action: ActionKind, active: bool) -> Outcome {
    Outcome {
        contiguous_observed_end_sample: Some(20),
        prediction: None,
        command_id: id,
        action,
        renderer_end_sample: None,
        source_id: 1,
        source_generation: 0,
        tone_id: id,
        issued_at_sample: 0,
        scheduled_action_sample: Some(0),
        window_end_sample: Some(20),
        available_at_sample: 20,
        observed_samples: 20,
        command_status: "accepted",
        buses: [true, false].map(|routed| BusOutcome {
            routed,
            status: if !routed {
                "not_routed"
            } else if active {
                "activity"
            } else {
                "silence"
            },
            first_activity_sample: (routed && active).then_some(2),
            last_activity_sample: (routed && active).then_some(19),
            peak: if active { 0.2 } else { 0. },
            rms: routed.then_some(if active { 0.1 } else { 0. }),
        }),
    }
}

#[test]
fn delayed_outcome_scores_the_issued_forecast_before_learning() {
    let mut bank = ModelBank::new();
    let mut first = outcome(1, ActionKind::Onset, false);
    let mut second = outcome(2, ActionKind::Onset, true);
    first.prediction = bank.issue(&first, input(1));
    second.prediction = bank.issue(&second, input(1));
    let issued_envelope = second.prediction.unwrap().envelope;
    let issued = second.prediction.unwrap().buses[0].unwrap();
    assert_eq!(issued.updates_at_issue, 0);
    bank.observe(&mut first);
    let after_update = bank
        .issue(&outcome(3, ActionKind::Onset, true), input(1))
        .unwrap();
    assert_ne!(
        after_update.buses[0].unwrap().predictions,
        issued.predictions
    );
    bank.observe(&mut second);
    assert_eq!(second.prediction.unwrap().envelope, issued_envelope);
    let scored = second.prediction.unwrap().buses[0].unwrap();
    assert_eq!(scored.predictions, issued.predictions);
    assert_eq!(scored.updates_at_issue, 0);
    assert_eq!(scored.target[0], Some(1.));
    assert_eq!(scored.target[1], Some(0.1));
    assert_eq!(scored.target[3], Some(0.));
    assert_eq!(scored.target[4], None);
    for comparison in 0..3 {
        assert_eq!(
            scored.squared_error[comparison][0],
            Some((1. - issued.predictions[comparison][0]).powi(2))
        );
    }
    assert!(scored.learned);
    assert!(second.prediction.unwrap().buses[1].is_none());
    assert_eq!(bank.stats.updated, 2);
}

#[test]
fn action_mapping_is_learned_and_relearned_against_both_controls() {
    let mut bank = ModelBank::new();
    // Synthetic correspondence test, not a fit to listening responses.
    for reversed in [false, true] {
        let mut loss = [0.; 3];
        for trial in 0..1600 {
            let onset = trial % 2 == 0;
            let mut out = outcome(
                (u64::from(reversed) * 1600) + trial + 1,
                if onset {
                    ActionKind::Onset
                } else {
                    ActionKind::Release
                },
                onset ^ reversed,
            );
            out.prediction = bank.issue(&out, input(1));
            bank.observe(&mut out);
            if trial >= 1400 {
                let scored = out.prediction.unwrap().buses[0].unwrap();
                for (comparison, sum) in loss.iter_mut().enumerate() {
                    *sum += scored.squared_error[comparison][0].unwrap() / 200.;
                }
            }
        }
        assert!(loss[1] < 0.03, "{reversed}: {loss:?}");
        assert!(loss[0] > 0.4, "{reversed}: {loss:?}");
        assert!(loss[2] > 0.2, "{reversed}: {loss:?}");
    }
}

#[test]
fn known_silence_learns_but_missing_cancelled_and_unexecuted_do_not() {
    let mut bank = ModelBank::new();
    for status in ["incomplete", "invalid_sample", "not_routed"] {
        let mut out = outcome(1, ActionKind::Onset, false);
        out.prediction = bank.issue(&out, input(1));
        out.buses[0].status = status;
        bank.observe(&mut out);
        assert_eq!(bank.stats.updated, 0);
        assert_eq!(out.prediction.unwrap().buses[0].unwrap().target, [None; 5]);
    }
    let mut cancelled = outcome(2, ActionKind::Onset, true);
    cancelled.prediction = bank.issue(&cancelled, input(1));
    cancelled.command_status = "cancelled";
    bank.observe(&mut cancelled);
    assert_eq!(bank.stats.updated, 0);
    assert!(bank.issue(&cancelled, input(1)).is_none());
    assert!(
        bank.issue(&outcome(3, ActionKind::Shutdown, true), input(1))
            .is_none()
    );
    let mut unissued = outcome(4, ActionKind::Onset, true);
    bank.observe(&mut unissued);
    assert_eq!(bank.stats.updated, 0);
    let mut silence = outcome(5, ActionKind::Onset, false);
    silence.prediction = bank.issue(&silence, input(1));
    bank.observe(&mut silence);
    let scored = silence.prediction.unwrap().buses[0].unwrap();
    assert_eq!(scored.target, [Some(0.), None, None, Some(0.), None]);
    assert!(scored.learned);
    assert_eq!(bank.stats.updated, 1);
}

#[test]
fn generation_replacement_and_eviction_cannot_train_new_owner() {
    let mut bank = ModelBank::new();
    let storage = bank.models.as_ptr();
    let mut old = outcome(1, ActionKind::Release, true);
    old.prediction = bank.issue(&old, input(1));
    let new = bank
        .issue(&outcome(2, ActionKind::Onset, true), input(2))
        .unwrap();
    bank.observe(&mut old);
    assert_eq!(bank.stats.retired, 1);
    assert_eq!(bank.models[new.slot].as_ref().unwrap().updates, [0; 2]);
    assert!(!old.prediction.unwrap().buses[0].unwrap().learned);
    for id in 2..=65 {
        let mut out = outcome(id + 1, ActionKind::Onset, true);
        out.source_id = id;
        bank.issue(&out, input(id as u32 + 1)).unwrap();
    }
    assert_eq!(bank.models.iter().flatten().count(), CAPACITY);
    assert_eq!(bank.models.as_ptr(), storage);
    assert_eq!(bank.stats.evicted, 2);
    assert!(bank.models.iter().flatten().all(|m| m.owner.0 != 1));
}

#[test]
fn one_bus_cannot_train_the_other_and_action_blind_input_excludes_commands() {
    let mut bank = ModelBank::new();
    let mut out = outcome(1, ActionKind::Onset, false);
    out.prediction = bank.issue(&out, input(1));
    bank.observe(&mut out);
    let mut next = outcome(2, ActionKind::Release, false);
    next.buses[1].routed = true;
    let mut altered = input(1);
    altered.frequency_hz = 4000.;
    altered.amplitude = 0.01;
    altered.envelope.release_end = 5;
    let same = bank.issue(&next, input(1)).unwrap();
    let different = bank.issue(&next, altered).unwrap();
    assert_eq!(same.buses[1].unwrap().updates_at_issue, 0);
    assert_eq!(same.buses[0].unwrap().updates_at_issue, 1);
    assert_eq!(
        same.buses[0].unwrap().predictions[2],
        different.buses[0].unwrap().predictions[2]
    );
    assert_ne!(
        same.buses[0].unwrap().predictions[0],
        different.buses[0].unwrap().predictions[0]
    );
    let fresh = same.buses[1].unwrap();
    assert_eq!(fresh.predictions[0], fresh.predictions[1]);
}

#[test]
fn observer_replay_and_stale_handles_cannot_issue_or_learn_twice() {
    let mut observer = Observer::new(1000);
    observer.enable_predictions();
    let batch = PhonationBatch::default();
    let slot = observer
        .command(&batch, 1, Some(0), 0, "accepted", ActionKind::Onset)
        .unwrap();
    observer.predict(slot, input(1), std::iter::empty());
    observer.predict(slot, input(1), std::iter::empty());
    observer.begin_hop(0);
    observer.sample(slot, (0, 0, 1), 2, 0.2, false);
    observer.end_hop(0, 20);
    let stats = observer.snapshot.prediction.unwrap();
    assert_eq!(stats.issued, 1);
    assert_eq!(stats.updated, 2);
    observer.begin_hop(0);
    assert!(!observer.sample(slot, (0, 0, 1), 2, 0.2, false));
    observer.end_hop(0, 20);
    observer.predict(slot, input(1), std::iter::empty());
    assert_eq!(observer.snapshot.prediction.unwrap(), stats);
    let next = observer
        .command(&batch, 2, Some(20), 20, "accepted", ActionKind::Onset)
        .unwrap();
    assert_eq!(next.0, slot.0);
    observer.predict(slot, input(1), std::iter::empty());
    observer.predict(next, input(1), std::iter::empty());
    observer.begin_hop(20);
    observer.end_hop(20, 40);
    let last = observer.drain().last().unwrap();
    assert_eq!(
        last.prediction.unwrap().buses[0].unwrap().updates_at_issue,
        1
    );
    assert_eq!(observer.snapshot.prediction.unwrap().updated, 4);
}

#[test]
fn energy_ratio_context_requires_all_owner_generations_and_issue_tick() {
    use crate::core::temporal_expectation::TemporalForecast;
    for (id, generation, body, tick, supported) in [
        (0, 0, 1, 0, true),
        (1, 0, 1, 0, false),
        (0, 1, 1, 0, false),
        (0, 0, 2, 0, false),
        (0, 0, 1, 1, false),
    ] {
        let mut observer = Observer::new(1000);
        observer.enable_predictions();
        let external = TemporalForecast::energy_fixture(1000, 0, |_| [0.; 3]);
        observer.refresh_energy_contexts(tick, std::iter::once((id, generation, body, external)));
        let frozen = observer.freeze_energy_context((0, 0, Some(1)), 0);
        assert_eq!(frozen.is_some(), supported);
        assert!(observer.freeze_energy_context((0, 0, None), 0).is_none());
        let batch = PhonationBatch::default();
        let slot = observer
            .command(&batch, 1, Some(0), 0, "accepted", ActionKind::Onset)
            .unwrap();
        observer.predict(slot, input(1), std::iter::empty());
        // Replacing the context after issue must not replace the frozen diagnostic.
        observer.refresh_energy_contexts(0, std::iter::empty());
        assert!(observer.freeze_energy_context((0, 0, Some(1)), 0).is_none());
        assert_eq!(
            frozen.map(|f| f.centered_energy(5, 0.5)),
            supported.then_some(Some(0.))
        );
        observer.predict(slot, input(1), std::iter::empty());
        observer.end_hop(0, 20);
        let outcome = observer.drain().next().unwrap();
        let ratio = outcome.prediction.unwrap().energy_ratio.unwrap();
        assert_eq!(
            ratio.status,
            if supported {
                "supported"
            } else {
                "external_unavailable"
            }
        );
        assert_eq!(ratio.overlap, supported.then_some(0.));
        let stats = observer.snapshot.prediction.unwrap().energy;
        assert_eq!(stats.ratio_previews, 1);
        assert_eq!(stats.ratio_supported, u64::from(supported));
        assert_eq!(stats.ratio_unknown, u64::from(!supported));
    }
}

#[test]
fn future_descriptors_keep_frozen_predictions_masks_and_exact_target_window() {
    let mut bank = ModelBank::new();
    for id in [1, 2] {
        bank.issue(&outcome(id, ActionKind::Onset, true), input(1))
            .unwrap();
    }
    bank.observe_body(&snapshot(99, 1, 63, 2.));
    assert_eq!(bank.stats.descriptor.pending, 2);
    let observed = snapshot(100, 1, 0b100101, 2.);
    bank.observe_body(&observed);
    let scored: Vec<_> = bank.drain_descriptors().collect();
    assert_eq!(scored.len(), 2);
    for out in scored {
        assert_eq!(out.target_end_sample, 100);
        assert_eq!(out.available_at_sample, 102);
        let bus = out.buses[0].unwrap();
        assert!(out.buses[1].is_none());
        assert_eq!(bus.updates_at_issue, 0);
        assert_eq!(bus.predictions, [[0.; 6]; 3]);
        assert_eq!(bus.target[1], None);
        assert_eq!(bus.target[0], Some(0.5_f64.tanh()));
        assert_eq!(bus.squared_error[1][0], Some(0.5_f64.tanh().powi(2)));
        assert!(bus.learned);
    }
    let stats = bank.stats.descriptor;
    bank.observe_body(&observed);
    assert_eq!(bank.stats.descriptor, stats);
    bank.issue(&outcome(3, ActionKind::Onset, true), input(1))
        .unwrap();
    bank.observe_body(&snapshot(101, 1, 63, 2.));
    let missed = bank.drain_descriptors().next().unwrap();
    assert_eq!(missed.buses[0].unwrap().status, "target_unavailable");
    assert_eq!(bank.stats.descriptor.updated, 2);
}

#[test]
fn future_descriptor_mapping_adapts_without_training_the_controls_or_other_bus() {
    let mut bank = ModelBank::new();
    for reversed in [false, true] {
        let mut loss = [0.; 3];
        for trial in 0..1600 {
            let onset = trial % 2 == 0;
            let serial = u64::from(reversed) * 1600 + trial + 1;
            let mut out = outcome(
                serial,
                if onset {
                    ActionKind::Onset
                } else {
                    ActionKind::Release
                },
                true,
            );
            out.issued_at_sample = serial * 200;
            out.scheduled_action_sample = Some(out.issued_at_sample);
            out.window_end_sample = Some(out.issued_at_sample + 20);
            let mut parameters = input(1);
            parameters.envelope.release_end += out.issued_at_sample;
            parameters.descriptor_target_end += out.issued_at_sample;
            bank.issue(&out, parameters).unwrap();
            bank.observe_body(&snapshot(
                out.issued_at_sample + 100,
                1,
                63,
                if onset ^ reversed { 2. } else { -2. },
            ));
            let scored = bank.drain_descriptors().next().unwrap();
            let bus = scored.buses[0].unwrap();
            if trial >= 1400 {
                for (comparison, sum) in loss.iter_mut().enumerate() {
                    *sum += bus.squared_error[comparison][0].unwrap() / 200.;
                }
            }
        }
        assert!(loss[1] < 0.03, "{reversed}: {loss:?}");
        assert!(loss[0] > 0.2, "{reversed}: {loss:?}");
        assert!(loss[2] > 0.2, "{reversed}: {loss:?}");
        assert_eq!(bank.models[0].as_ref().unwrap().descriptor_updates[1], 0);
    }
}

#[test]
fn future_descriptor_cancel_interference_retirement_eof_and_capacity_are_not_teacher_data() {
    for reason in [
        "cancelled",
        "intervening_command",
        "retired",
        "unfinished",
        "target_unavailable",
        "unsupported_coordinates",
    ] {
        let mut bank = ModelBank::new();
        bank.issue(&outcome(1, ActionKind::Onset, true), input(1))
            .unwrap();
        let mut observed = snapshot(100, 1, 63, 2.);
        match reason {
            "cancelled" => bank.cancel_descriptor(1),
            "intervening_command" => bank.intervene(1, 0, 99),
            "retired" => {
                bank.issue(&outcome(2, ActionKind::Onset, true), input(2))
                    .unwrap();
            }
            "unfinished" => {
                observed.input_end = 99;
                observed.records[0] = Record::default();
                observed.finished = true;
            }
            "target_unavailable" => observed.records[0].body_generation = 2,
            "unsupported_coordinates" => observed.records[0].mask = 0,
            _ => unreachable!(),
        }
        bank.observe_body(&observed);
        let out = bank.drain_descriptors().next().unwrap();
        assert!(!out.buses[0].unwrap().learned, "{reason}");
        assert_eq!(bank.stats.descriptor.updated, 0, "{reason}");
        if reason != "retired" {
            assert_eq!(out.buses[0].unwrap().status, reason);
        }
    }
    let mut bank = ModelBank::new();
    let pending_storage = bank.descriptor_pending.as_ptr();
    let output_storage = bank.descriptor_ready.as_ptr();
    for id in 0..257 {
        bank.issue(&outcome(id, ActionKind::Onset, true), input(1))
            .unwrap();
    }
    assert_eq!(bank.stats.descriptor.pending, 256);
    assert_eq!(bank.stats.descriptor.capacity_dropped, 1);
    bank.observe_body(&snapshot(100, 1, 0, 0.));
    bank.issue(&outcome(258, ActionKind::Onset, true), input(1))
        .unwrap();
    bank.observe_body(&snapshot(100, 1, 63, 2.));
    assert_eq!(bank.stats.descriptor.updated, 0);
    assert_eq!(bank.stats.descriptor.output_dropped, 1);
    assert_eq!(bank.descriptor_pending.as_ptr(), pending_storage);
    assert_eq!(bank.descriptor_ready.as_ptr(), output_storage);
}

#[test]
fn retained_inventory_overflow_is_unknown_and_never_trains_a_partial_prior() {
    use crate::core::timebase::Timebase;
    use crate::life::sound::Tone;
    use crate::scenario::control::Routing;
    let time = Timebase { fs: 8000., hop: 64 };
    let tone = Tone::from_parts(time, 0, 10000, 220., 0.2, None, None, None).unwrap();
    for entries in [64, 65] {
        let mut observer = Observer::new(8000);
        observer.enable_predictions();
        let batch = PhonationBatch {
            source_id: 1,
            routing: Routing::default(),
            ..Default::default()
        };
        let slot = observer
            .command(&batch, 1, Some(0), 0, "accepted", ActionKind::Onset)
            .unwrap();
        observer.predict(
            slot,
            input(1),
            std::iter::once(None).chain(std::iter::repeat_n(
                Some((
                    &tone,
                    Routing::default(),
                    None,
                    tone.prediction_control(0, &Default::default()),
                )),
                entries - 1,
            )),
        );
        observer.end_hop(0, 160);
        let mut out = observer.drain().next().unwrap();
        let predicted = out.prediction.as_mut().unwrap();
        for energy in &mut predicted.source_energy {
            assert_eq!(energy.retained.scanned_entries, 64);
            assert_eq!(energy.retained.included_tones, 63);
            assert_eq!(energy.retained.complete, entries == 64);
            assert!(
                energy.predictions[0]
                    .iter()
                    .all(|x| x.is_some() == (entries == 64))
            );
            energy.target.fill(Some(0.01));
        }
        if entries == 65 {
            let mut bank = ModelBank::new();
            bank.observe(&mut out);
            assert_eq!(bank.stats.energy.scored_windows, 0);
            assert_eq!(bank.stats.energy.updated, 0);
            assert_eq!(bank.stats.energy.unsupported_windows, 32);
            assert!(
                out.prediction
                    .unwrap()
                    .source_energy
                    .iter()
                    .all(|e| !e.learned)
            );
        }
    }
}

#[test]
fn unsupported_control_is_not_scored_or_learned_as_partial_source_energy() {
    use crate::life::sound::RenderModulatorStateKind;
    use crate::life::sound::control_forecast::AmplitudeModel;
    for unknown_command in [false, true] {
        let mut values = input(1);
        if unknown_command {
            values.control.as_mut().unwrap().model = AmplitudeModel::EntrainPulse {
                attack_step: 100.,
                decay_rate: 10.,
                sustain_level: 0.,
                state: RenderModulatorStateKind::Idle,
                env_level: 0.,
                autonomous_retrigger: true,
            };
        } else {
            values.retained_energy[0].control_supported.fill(false);
            values.retained_energy[1].control_supported.fill(false);
        }
        let mut out = outcome(1, ActionKind::Onset, true);
        out.buses[1] = out.buses[0];
        let mut bank = ModelBank::new();
        out.prediction = bank.issue(&out, values);
        let forecast = out.prediction.as_mut().unwrap();
        for energy in &mut forecast.source_energy {
            energy.target.fill(Some(0.01));
            for (k, [left, right]) in energy.windows.iter().enumerate() {
                if left != right {
                    assert_eq!(energy.predictions[0][k], None);
                }
            }
        }
        bank.observe(&mut out);
        assert_eq!(bank.stats.energy.scored_windows, 0);
        assert_eq!(bank.stats.energy.updated, 0);
        assert!(bank.stats.energy.unsupported_windows > 0);
    }
}
