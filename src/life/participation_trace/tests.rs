use super::*;
use crate::{
    life::{action_observation::Observer, voice::PhonationBatch},
    temporal_cognition::reference_inventory::{self as inventory, Handle},
};

fn config() -> TemporalPrivateTraceConfig {
    TemporalPrivateTraceConfig {
        tau_sec: 10.,
        kappa: 2.,
        strength_max: 3.,
    }
}

fn context(now: u64) -> Context {
    let key = trace::Key {
        epoch: 3,
        episode: 42,
        generation: 42,
        family: trace::Family::Periodic,
    };
    let mut anchors = [None; 7];
    anchors[0] = Some(inventory::Anchor {
        group: Handle {
            bus: 0,
            epoch: 3,
            generation: 9,
        },
        interval_sec: [0., 0.],
        period_sec: Some(1.),
        weight: 1.,
    });
    let mut references = [None; N];
    references[0] = Some(inventory::Reference {
        key,
        weight: 0.8,
        weight_upper: 0.9,
        anchors,
    });
    Context {
        inventory: Some(inventory::Snapshot {
            bus: 0,
            epoch: 3,
            sample_rate: 1000,
            window: [now.saturating_sub(2000), now],
            denominator_samples: now.min(2000),
            epoch_clipped: now < 2000,
            available_at_sample: now,
            credit_policy: "lower_bound",
            bias_parameter: "temporal_memory.retention.no_memory_bias",
            no_memory_bias: 0.,
            assignment_samples: 1.,
            groups: [None; 7],
            references,
            examined_bindings: 1,
            assigned: 0.8,
            unassigned: 0.2,
            search_discarded: [0.; 2],
            inventory_discarded: [0.; 2],
            anchor_unsupported: [0.; 2],
            reference_bytes: 8192,
            owned_bytes: 0,
        }),
        retained: std::sync::Arc::from([(42, 42)]),
    }
}

fn outcome(id: u64, at: u64) -> Outcome {
    let mut observer = Observer::new(1000);
    let batch = PhonationBatch {
        source_id: 7,
        source_generation: 2,
        ..Default::default()
    };
    observer.begin_hop(at);
    let slot = observer
        .command(&batch, 1, Some(at), at, "accepted", ActionKind::Onset)
        .unwrap();
    observer.sample(slot, (7, 2, 1), at + 1, 0.2, false);
    observer.end_hop(at, at + 20);
    let mut o = observer.drain().next().unwrap();
    o.command_id = id;
    o
}

#[test]
fn frozen_reference_and_prediction_survive_later_cue_changes_without_retrieval_learning() {
    let mut bank = Bank::new(1000, config());
    bank.context(context(10), 10);
    let key = context(10).inventory.unwrap().references[0].unwrap().key;
    let first = outcome(1, 10);
    bank.issue(&first, Some(1.));
    assert!(
        bank.voices[0].models[0]
            .probabilities(key, trace::Head::Onset)
            .is_none()
    );
    let mut later = context(30);
    later.inventory.as_mut().unwrap().references.fill(None);
    bank.context(later, 30);
    bank.complete(first);
    bank.seal(30, 30);
    let record = bank.drain().next().unwrap();
    assert_eq!(record.inventory_at_sample, Some(10));
    let e = record.entries[0].unwrap();
    assert!((e.credit - 0.8).abs() < 1e-12);
    assert!(e.forecasts.iter().all(|f| f.prior_used));
    assert!((e.log_loss[0].unwrap() - 32_f64.ln()).abs() < 1e-12);
    assert_eq!(record.observed_interval_sec, Some([0.011, 0.012]));
    bank.context(context(40), 40);
    let second = outcome(2, 40);
    bank.issue(&second, Some(2.));
    let frozen = bank.pending.iter().flatten().next().unwrap().predictions[0][0].unwrap();
    bank.context(context(60), 60);
    bank.complete(second);
    bank.complete(second);
    bank.seal(60, 60);
    let record = bank.drain().next().unwrap();
    assert_eq!(
        record.entries[0].unwrap().forecasts[0].probabilities,
        frozen
    );
    assert!(!record.entries[0].unwrap().forecasts[0].prior_used);
    assert_eq!(bank.stats.learned, 2);
    assert_eq!(bank.stats.ignored_deliveries, 1);
    let before = bank.voices[0].models[0].probabilities(key, trace::Head::Onset);
    bank.context(context(70), 70);
    assert_eq!(
        before,
        bank.voices[0].models[0].probabilities(key, trace::Head::Onset)
    );
}

#[test]
fn delayed_release_seals_before_later_onsets_but_forecasts_stay_at_issue() {
    let mut observer = Observer::new(1000);
    observer.enable_trace(config());
    let batch = PhonationBatch {
        intrinsic_period_sec: Some(1.),
        source_id: 7,
        source_generation: 2,
        ..Default::default()
    };
    observer.trace_context(context(0), 0);
    let release = observer
        .command(&batch, 1, Some(0), 0, "accepted", ActionKind::Release)
        .unwrap();
    observer.begin_hop(0);
    observer.sample(release, (7, 2, 1), 10, 0.2, false);
    observer.sample(release, (7, 2, 1), 20, 0., true);
    observer.end_hop(0, 100);
    for at in [100, 200] {
        observer.trace_context(context(at), at);
        let onset = observer
            .command(&batch, at, Some(at), at, "accepted", ActionKind::Onset)
            .unwrap();
        observer.begin_hop(at);
        observer.sample(onset, (7, 2, at), at + 1, 0.2, false);
        observer.end_hop(at, at + 100);
        assert_eq!(observer.drain_traces().count(), 0);
    }
    observer.trace_context(context(2000), 2000);
    observer.begin_hop(300);
    observer.end_hop(300, 2000);
    let records: Vec<_> = observer.drain_traces().collect();
    assert_eq!(
        records.iter().map(|r| r.command_id).collect::<Vec<_>>(),
        vec![1, 2, 3]
    );
    assert_eq!(records[0].head, trace::Head::Release);
    assert_eq!(records[0].observed_interval_sec, Some([0.020, 0.021]));
    assert!(
        records[1..]
            .iter()
            .all(|r| r.entries[0].unwrap().forecasts[0].prior_used)
    );
    assert!(records.iter().all(|r| r.error.is_none() && r.applied));
    assert_eq!(observer.snapshot.participation_trace.unwrap().learned, 3);
    observer.finish();
    assert_eq!(observer.drain_traces().count(), 0);
}

#[test]
fn loss_of_retained_identity_epoch_or_voice_generation_masks_pending_credit() {
    for reason in 0..3 {
        let mut bank = Bank::new(1000, config());
        bank.context(context(10), 10);
        let first = outcome(1, 10);
        bank.issue(&first, Some(1.));
        let mut changed = context(30);
        if reason == 0 {
            changed.retained = std::sync::Arc::from([]);
        }
        if reason == 1 {
            changed.inventory.as_mut().unwrap().epoch = 4;
            changed.inventory.as_mut().unwrap().references.fill(None);
        }
        bank.context(changed, 30);
        if reason == 2 {
            let mut reborn = outcome(2, 30);
            reborn.source_generation = 3;
            bank.issue(&reborn, Some(1.));
        }
        bank.complete(first);
        bank.seal(30, 30);
        let record = bank.drain().next().unwrap();
        assert_eq!(record.unassigned, 1., "reason={reason}");
        assert_eq!(bank.stats.learned, 0);
        assert_eq!(
            bank.voices
                .iter()
                .map(|v| v.models[0].keys().count())
                .sum::<usize>(),
            0
        );
    }
}

#[test]
fn routing_unknown_pace_stale_context_and_cancellation_never_create_a_trace() {
    for reason in 0..5 {
        let mut bank = Bank::new(1000, config());
        let mut c = context(10);
        if reason == 1 {
            let r = c.inventory.as_mut().unwrap().references[0]
                .as_mut()
                .unwrap();
            r.key.family = trace::Family::Nonperiodic;
            r.anchors[0].as_mut().unwrap().period_sec = None;
        }
        if reason == 2 {
            c.inventory.as_mut().unwrap().available_at_sample = 1000;
        }
        bank.context(c, 10);
        let mut o = outcome(1, if reason == 3 { 200 } else { 10 });
        bank.issue(&o, None);
        if reason == 0 {
            o.buses[0].routed = false;
        }
        if reason == 4 {
            o.command_status = "superseded";
        }
        let now = o.available_at_sample;
        bank.context(context(now), now);
        bank.complete(o);
        bank.seal(now, now);
        let r = bank.drain().next().unwrap();
        assert_eq!(r.unassigned, 1., "reason={reason}");
        assert_eq!(bank.stats.learned, 0);
        assert!(bank.voices.iter().all(|v| v.models[0].keys().count() == 0));
    }
}

#[test]
fn pending_storage_is_bounded_and_capacity_loss_does_not_change_prior_credit() {
    let mut bank = Bank::new(1000, config());
    bank.context(context(10), 10);
    let pending_capacity = bank.pending.capacity();
    let ready_capacity = bank.ready.capacity();
    for id in 1..=257 {
        bank.issue(&outcome(id, 10), Some(1.));
    }
    assert_eq!(bank.stats.pending, 256);
    assert_eq!(bank.stats.capacity_dropped, 1);
    bank.context(context(30), 30);
    for id in (1..=256).rev() {
        bank.complete(outcome(id, 10));
    }
    bank.seal(30, 30);
    assert_eq!(bank.stats.pending, 0);
    assert_eq!(bank.stats.learned, 256);
    assert_eq!(bank.pending.capacity(), pending_capacity);
    assert_eq!(bank.ready.capacity(), ready_capacity);
    let records: Vec<_> = bank.drain().collect();
    assert!(
        records
            .iter()
            .enumerate()
            .all(|(i, r)| r.command_id == i as u64 + 1 && (r.unassigned - 0.2).abs() < 1e-12)
    );
}

#[test]
fn a_gap_after_a_confirmed_onset_preserves_fractional_credit_but_an_obscured_onset_does_not() {
    for onset_before_gap in [true, false] {
        let mut observer = Observer::new(1000);
        observer.enable_trace(config());
        observer.trace_context(context(0), 0);
        let batch = PhonationBatch {
            intrinsic_period_sec: Some(1.),
            source_id: 7,
            source_generation: 2,
            ..Default::default()
        };
        let slot = observer
            .command(&batch, 1, Some(0), 0, "accepted", ActionKind::Onset)
            .unwrap();
        observer.begin_hop(0);
        observer.sample(
            slot,
            (7, 2, 1),
            2,
            if onset_before_gap { 0.2 } else { 0. },
            false,
        );
        observer.end_hop(0, 5);
        observer.trace_context(context(20), 20);
        observer.begin_hop(10);
        observer.sample(slot, (7, 2, 1), 12, 0.2, false);
        observer.end_hop(10, 20);
        let r = observer.drain_traces().next().unwrap();
        assert_eq!(r.known_fraction, 0.75);
        let credit = 1. - r.unassigned;
        assert!((credit - if onset_before_gap { 0.6 } else { 0. }).abs() < 1e-12);
        assert_eq!(r.observed_interval_sec.is_some(), onset_before_gap);
    }
}

#[test]
fn full_and_elapsed_only_forecasts_diverge_after_competing_reference_credit() {
    let mut bank = Bank::new(1000, config());
    let mut key = context(0).inventory.unwrap().references[0].unwrap().key;
    for (id, at, episode) in [(1, 10, 42), (2, 200, 43), (3, 500, 42)] {
        let mut c = context(at);
        c.retained = std::sync::Arc::from([(42, 42), (43, 43)]);
        let r = c.inventory.as_mut().unwrap().references[0]
            .as_mut()
            .unwrap();
        r.key.episode = episode;
        r.key.generation = episode;
        key = r.key;
        bank.context(c.clone(), at);
        let o = outcome(id, at);
        bank.issue(&o, Some(1.));
        c.inventory.as_mut().unwrap().available_at_sample = at + 20;
        bank.context(c, at + 20);
        bank.complete(o);
        bank.seal(at + 20, at + 20);
        bank.drain().for_each(drop);
    }
    let full = bank.voices[0].models[0]
        .probabilities(key, trace::Head::Onset)
        .unwrap();
    let elapsed = bank.voices[0].models[1]
        .probabilities(key, trace::Head::Onset)
        .unwrap();
    assert!(full.iter().zip(elapsed).any(|(a, b)| (a - b).abs() > 0.001));
    let mut c = context(600);
    c.retained = std::sync::Arc::from([(42, 42), (43, 43)]);
    bank.context(c, 600);
    bank.issue(&outcome(4, 600), Some(1.));
    assert_eq!(bank.stats.learned, 3);
    assert_eq!(
        bank.pending.iter().flatten().next().unwrap().predictions[0][0],
        Some(full)
    );
    assert_eq!(
        bank.pending.iter().flatten().next().unwrap().predictions[1][0],
        Some(elapsed)
    );
}

fn predicted_outcome(id: u64, at: u64) -> Outcome {
    let mut out = outcome(id, at);
    let input = crate::life::self_prediction::Input {
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
        body_generation: 1,
        descriptors: [[None; 6]; 2],
        descriptor_support: [None; 2],
        frequency_hz: 220.,
        amplitude: 0.2,
        envelope: crate::life::sound::envelope::Envelope {
            onset: at,
            hold_end: at + 80,
            release_end: at + 100,
            attack_ticks: 10,
            decay_ticks: 0,
            sustain_level: 1.,
            decay_lambda: 0.,
            release_ticks: 20,
        },
        descriptor_target_end: at + 100,
        descriptor_slot: 0,
    };
    out.prediction = crate::life::self_prediction::ModelBank::new().issue(&out, input);
    out
}

impl Frozen {
    pub(crate) fn release_fixture() -> Self {
        let mut bank = Bank::new(1000, config());
        bank.context(context(10), 10);
        let mut first = outcome(1, 10);
        first.action = ActionKind::Release;
        first.renderer_end_sample = Some(25);
        bank.issue(&first, Some(1.));
        bank.complete(first);
        bank.seal(30, 30);
        assert_eq!(bank.stats.learned, 1);
        bank.context(context(100), 100);
        bank.freeze_release((7, 2), 100, Some(1.)).unwrap()
    }
}

#[test]
fn current_release_freeze_matches_issue_forecasts_without_creating_or_learning_an_event() {
    for family in [trace::Family::Periodic, trace::Family::Nonperiodic] {
        let mut bank = Bank::new(1000, config());
        let mut ctx = context(10);
        ctx.inventory.as_mut().unwrap().references[0]
            .as_mut()
            .unwrap()
            .key
            .family = family;
        bank.context(ctx.clone(), 10);
        assert!(bank.freeze_release((7, 2), 10, Some(1.)).is_none());
        let mut first = outcome(1, 10);
        first.action = ActionKind::Release;
        first.renderer_end_sample = Some(25);
        bank.issue(&first, Some(1.));
        assert_eq!(
            bank.freeze_release((7, 2), 10, Some(1.))
                .unwrap()
                .fit(1000, 25, 25)[0]
                .difference,
            None
        );
        bank.complete(first);
        bank.seal(30, 30);
        assert_eq!(bank.stats.learned, 1);
        bank.drain().for_each(drop);
        ctx.inventory.as_mut().unwrap().available_at_sample = 100;
        ctx.inventory.as_mut().unwrap().references[0]
            .as_mut()
            .unwrap()
            .anchors[0]
            .as_mut()
            .unwrap()
            .period_sec = None;
        bank.context(ctx.clone(), 100);
        let stats = bank.stats;
        let frozen = bank.freeze_release((7, 2), 100, Some(2.)).unwrap();
        assert_eq!(frozen.origin.command_id, None);
        assert_eq!(frozen.head, trace::Head::Release);
        let pair = frozen.fit(1000, 25, 525);
        assert!(pair.iter().all(|f| (f.paired_support - 0.8).abs() < 1e-12));
        for _ in 0..10 {
            assert_eq!(
                bank.freeze_release((7, 2), 100, Some(2.))
                    .unwrap()
                    .fit(1000, 25, 525),
                pair
            );
        }
        assert_eq!(bank.stats, stats);
        assert!(bank.pending.iter().all(Option::is_none));
        let mut issued = outcome(2, 100);
        issued.action = ActionKind::Release;
        bank.issue(&issued, Some(2.));
        let pending = bank.pending.iter().flatten().next().unwrap();
        assert_eq!(frozen.count, pending.count);
        assert_eq!(frozen.predictions, pending.predictions);
        assert_eq!(
            serde_json::to_value(frozen.references).unwrap(),
            serde_json::to_value(pending.references).unwrap()
        );
        assert!(bank.freeze_release((7, 3), 100, Some(2.)).is_none());
        let stale = bank.freeze_release((7, 2), 201, Some(2.)).unwrap();
        assert_eq!(stale.origin.inventory_at_sample, None);
        assert_eq!(stale.fit(1000, 25, 525)[0].difference, None);
        let future = bank.freeze_release((7, 2), 99, Some(2.)).unwrap();
        assert_eq!(future.origin.inventory_at_sample, None);
        ctx.retained = std::sync::Arc::from([]);
        bank.context(ctx, 100);
        assert_eq!(bank.freeze_release((7, 2), 100, Some(2.)).unwrap().count, 0);
        assert_eq!(frozen.fit(1000, 25, 525), pair);
        assert_eq!(bank.stats.learned, 1);
    }
}

#[test]
fn conditional_body_trace_freezes_only_the_owned_issued_onset() {
    let mut bank = Bank::new(1000, config());
    bank.context(context(10), 10);
    let first = predicted_outcome(1, 10);
    bank.issue(&first, Some(1.));
    let cold = bank.freeze_onset(1, (7, 2), 10).unwrap();
    assert!(
        cold.fit(1000, 20, 11)
            .iter()
            .all(|f| f.difference.is_none() && f.unassigned == 1.)
    );
    bank.complete(first);
    bank.seal(30, 30);
    bank.drain().for_each(drop);
    bank.context(context(100), 100);
    let second = predicted_outcome(2, 100);
    bank.issue(&second, Some(1.));
    let stats = bank.stats;
    let frozen = bank.freeze_onset(2, (7, 2), 100).unwrap();
    assert_eq!(
        frozen.origin,
        Origin {
            command_id: Some(2),
            source_id: 7,
            source_generation: 2,
            epoch: 3,
            bus: 0,
            issued_at_sample: 100,
            inventory_at_sample: Some(100)
        }
    );
    assert!(std::mem::size_of::<Frozen>() <= 20_000);
    let default = frozen.fit(1000, 101, 101);
    assert_eq!(default[0].difference, Some(0.));
    assert!((default[0].paired_support - 0.8).abs() < 1e-12);
    let delayed = frozen.fit(1000, 1011, 101);
    assert!(delayed[0].difference.unwrap().abs() > 1e-6);
    assert!(bank.freeze_onset(1, (7, 2), 100).is_none());
    assert!(bank.freeze_onset(2, (8, 2), 100).is_none());
    assert!(bank.freeze_onset(2, (7, 3), 100).is_none());
    assert!(bank.freeze_onset(2, (7, 2), 101).is_none());
    assert_eq!(bank.stats, stats);
    bank.preview(&second);
    let old = bank
        .pending
        .iter()
        .flatten()
        .find(|p| p.id == 2)
        .unwrap()
        .preview
        .unwrap();
    for c in old.candidates.iter().flatten() {
        assert_eq!(frozen.fit(1000, c.event_sample, 101), c.fits);
    }
    let mut release = predicted_outcome(3, 110);
    release.action = ActionKind::Release;
    bank.issue(&release, Some(1.));
    assert!(bank.freeze_onset(3, (7, 2), 110).is_none());
    let mut renewed = predicted_outcome(4, 120);
    renewed.source_generation += 1;
    bank.issue(&renewed, Some(1.));
    assert!(bank.freeze_onset(2, (7, 2), 100).is_none());
    assert_eq!(frozen.fit(1000, 1011, 101), delayed);
    assert_eq!(bank.stats.learned, 1);
    assert_eq!(cold.fit(1000, 20, 11)[0].difference, None);
}

#[test]
fn timing_preview_reads_frozen_learned_mass_without_training_or_reassigning_it() {
    let mut bank = Bank::new(1000, config());
    bank.context(context(10), 10);
    let key = context(10).inventory.unwrap().references[0].unwrap().key;
    let first = predicted_outcome(1, 10);
    bank.issue(&first, Some(1.));
    bank.preview(&first);
    let before = bank
        .pending
        .iter()
        .flatten()
        .next()
        .unwrap()
        .preview
        .unwrap();
    assert!(before.candidates.iter().flatten().all(|c| {
        c.fits
            .iter()
            .all(|f| f.difference.is_none() && f.unassigned == 1.)
    }));
    assert!(
        bank.voices[0].models[0]
            .probabilities(key, trace::Head::Onset)
            .is_none()
    );
    bank.complete(first);
    bank.seal(30, 30);
    bank.drain().for_each(drop);
    assert_eq!(bank.stats.learned, 1);

    bank.context(context(100), 100);
    let second = predicted_outcome(2, 100);
    bank.issue(&second, Some(1.));
    let learned = bank.voices[0].models[0]
        .probabilities(key, trace::Head::Onset)
        .unwrap();
    bank.preview(&second);
    bank.preview(&second);
    let frozen = bank
        .pending
        .iter()
        .flatten()
        .next()
        .unwrap()
        .preview
        .unwrap();
    assert_eq!(bank.stats.timing_previews, 2);
    assert_eq!(bank.stats.timing_queries, 26);
    assert_eq!(bank.stats.learned, 1);
    assert_eq!(
        bank.voices[0].models[0].probabilities(key, trace::Head::Onset),
        Some(learned)
    );
    assert!(
        frozen
            .candidates
            .iter()
            .flatten()
            .any(|c| c.fits[0].difference.is_some_and(|v| v.abs() > 1e-6))
    );
    let default = frozen
        .candidates
        .iter()
        .flatten()
        .find(|c| c.event_sample == frozen.body_default_event_sample)
        .unwrap();
    assert_eq!(default.event_sample, 101);
    assert_eq!(default.fits[0].difference, Some(0.));
    assert!((default.fits[0].paired_support - 0.8).abs() < 1e-12);
    let mut changed = context(120);
    changed.inventory.as_mut().unwrap().references[0]
        .as_mut()
        .unwrap()
        .weight = 0.2;
    bank.context(changed, 120);
    bank.complete(second);
    bank.seal(120, 120);
    assert_eq!(bank.drain().next().unwrap().timing_preview, Some(frozen));
    assert_eq!(bank.stats.learned, 2);

    let third = predicted_outcome(3, 130);
    bank.issue(&third, Some(1.));
    bank.preview(&third);
    assert!(
        bank.pending
            .iter()
            .flatten()
            .find(|p| p.id == 3)
            .unwrap()
            .preview
            .is_none()
    );
    assert_eq!(bank.stats.timing_rate_limited, 1);
    let fourth = predicted_outcome(4, 150);
    bank.issue(&fourth, None);
    bank.preview(&fourth);
    let fourth = bank
        .pending
        .iter()
        .flatten()
        .find(|p| p.id == 4)
        .unwrap()
        .preview
        .unwrap();
    assert_eq!(fourth.candidates.iter().flatten().count(), 1);
    assert_eq!(fourth.body_default_event_sample, 151);
}

#[test]
fn timing_preview_uses_release_end_and_cannot_borrow_a_retired_voices_trace() {
    let mut bank = Bank::new(1000, config());
    bank.context(context(10), 10);
    let first = predicted_outcome(1, 10);
    bank.issue(&first, Some(1.));
    bank.preview(&first);
    bank.complete(first);
    bank.seal(30, 30);
    bank.drain().for_each(drop);
    let mut release = predicted_outcome(2, 100);
    release.action = ActionKind::Release;
    bank.context(context(100), 100);
    bank.issue(&release, Some(1.));
    bank.preview(&release);
    let preview = bank
        .pending
        .iter()
        .flatten()
        .next()
        .unwrap()
        .preview
        .unwrap();
    assert_eq!(preview.body_default_event_sample, 200);
    assert!(
        preview
            .candidates
            .iter()
            .flatten()
            .all(|c| c.fits[0].difference.is_none())
    );
    let mut renewed = predicted_outcome(3, 101);
    renewed.source_generation += 1;
    bank.issue(&renewed, Some(1.));
    bank.preview(&renewed);
    let preview = bank
        .pending
        .iter()
        .flatten()
        .find(|p| p.id == 3)
        .unwrap()
        .preview
        .unwrap();
    assert!(
        preview
            .candidates
            .iter()
            .flatten()
            .all(|c| c.fits[0].difference.is_none())
    );
    assert_eq!(bank.stats.timing_rate_limited, 0);
    let mut silent = predicted_outcome(4, 160);
    silent.buses[0].routed = false;
    bank.issue(&silent, Some(1.));
    bank.preview(&silent);
    assert!(
        bank.pending
            .iter()
            .flatten()
            .find(|p| p.id == 4)
            .unwrap()
            .preview
            .is_none()
    );
}
