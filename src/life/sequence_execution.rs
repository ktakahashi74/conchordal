//! Sample-ordered pitch decisions for stepwise sustained participation.
use super::*;
use crate::life::own_sequence::{ExecutionKind, ExecutionRecord};
use crate::scenario::control::PitchControl;
use std::sync::Arc;

#[derive(Debug, Default)]
pub(crate) struct DeferredPitch {
    pub(crate) field: Option<Arc<Landscape>>,
    steps: Vec<PitchStep>,
    used: usize,
    events: Vec<ExecutionRecord>,
    next_execution: u64,
    missing: bool,
    recorded_end: Option<Tick>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn produced_melodic(intervals: &[i32]) -> crate::life::melodic_continuity::MelodicContinuity {
        let mut control = VoiceControl::default();
        control.pitch.freq = 440.0;
        control.pitch.pitch_apply_mode = PitchApplyMode::Rearticulate;
        let mut voice = Voice::spawn_from_control(
            control,
            ArticulationCoreConfig::Drone {
                sway: None,
                breath_gain_init: None,
            },
            2,
            0,
            VoiceMetadata::default(),
            48000.0,
            None,
            21,
        );
        voice.configure_control_hop(512, 64, 48000.0);
        let mut observer = crate::life::melodic_continuity::MelodicContinuity::new(
            48000.0,
            1.0,
            (220.0f32.log2(), 880.0f32.log2()),
        );
        observer.trace_enabled = true;
        let mut deferred = DeferredPitch::default();
        let mut batch = PhonationBatch::default();
        voice.prepare_phonation_batch(&mut batch);
        let tb = Timebase {
            fs: 48000.0,
            hop: 512,
        };
        let mut semitones = 0;
        for i in 0..=intervals.len() {
            let at = i as u64 * 6000;
            if i > 0 {
                semitones += intervals[i - 1];
                let hz = (440.0 * 2.0f64.powf(semitones as f64 / 12.0)) as f32;
                voice.force_set_pitch_log2(hz.log2());
                voice
                    .participation_pitch
                    .rearticulations
                    .push(Rearticulation {
                        at,
                        pitch_log2: hz.log2(),
                    });
            }
            voice.tick_phonation_into(
                &tb,
                at,
                &NeuralRhythms::default(),
                None,
                0.0,
                1.0,
                0.8,
                &mut batch,
            );
            assert_eq!(batch.tones.len(), 1, "producer onset {i}");
            voice.pitch_ctl.melodic = Some(observer);
            voice.consume_sequence_execution(&mut deferred, &batch, &mut 0, &mut 0, at + 512);
            observer = voice.pitch_ctl.melodic.take().unwrap();
        }
        assert_eq!(observer.prefix.n, intervals.len() + 1);
        observer
    }

    #[test]
    fn melodic_same_hop_causal_order_and_neutral_producer() {
        let mut same_hop = 0;
        for tie in [0.0, 0.5, 1.0] {
            let mut control = VoiceControl::default();
            control.pitch.freq = 330.0;
            control.phonation.tie = tie;
            control.pitch.pitch_apply_mode = PitchApplyMode::Rearticulate;
            control.pitch.proposal_interval_sec = Some(0.001);
            control.phonation.spec = crate::scenario::PhonationSpec {
                timing: crate::scenario::PhonationTiming::Coupled(
                    crate::scenario::CoupledTimingSpec {
                        relation: crate::scenario::RhythmRelation::Participating,
                        coupling: 0.8,
                        ..Default::default()
                    },
                ),
                duration: crate::scenario::DurationSpec::WhileAlive,
                sustain_participation: true,
            };
            let make = |control| {
                Voice::spawn_from_control(
                    control,
                    ArticulationCoreConfig::Drone {
                        sway: None,
                        breath_gain_init: None,
                    },
                    2,
                    0,
                    VoiceMetadata::default(),
                    48000.0,
                    None,
                    21,
                )
            };
            let mut old = make(control.clone());
            control.pitch.melodic_continuity = 1.0;
            let mut learned = make(control.clone());
            learned.pitch_ctl.melodic.as_mut().unwrap().trace_enabled = true;
            let mut causal = make(control);
            causal.pitch_ctl.melodic = None;
            let tb = Timebase {
                fs: 48000.0,
                hop: 512,
            };
            let mut field = Landscape::new(Log2Space::new(55.0, 8000.0, 96));
            for (i, &pitch) in field.space.centers_log2.iter().enumerate() {
                field.consonance_field_score_eff[i] = (pitch * 7.0).sin();
            }
            for v in [&mut old, &mut causal, &mut learned] {
                v.configure_control_hop(512, 64, tb.fs);
            }
            let mut a = PhonationBatch::default();
            let mut b = PhonationBatch::default();
            let mut c = PhonationBatch::default();
            learned.prepare_phonation_batch(&mut c);
            old.prepare_phonation_batch(&mut a);
            causal.prepare_phonation_batch(&mut b);
            for hop in 0..512 {
                let at = hop * 512;
                for (v, batch) in [
                    (&mut old, &mut a),
                    (&mut causal, &mut b),
                    (&mut learned, &mut c),
                ] {
                    v.tick_phonation_into_with_arrival(
                        &tb,
                        at,
                        &field.rhythm,
                        None,
                        0.0,
                        1.0,
                        0.8,
                        None,
                        Some(&field.space),
                        batch,
                    );
                }
                assert_eq!(
                    serde_json::to_value(&a).unwrap(),
                    serde_json::to_value(&b).unwrap(),
                    "hop {hop}"
                );
                for step in 0..8 {
                    for v in [&mut old, &mut causal, &mut learned] {
                        v.set_control_tick(at + 512 + step * 64);
                        v.tick_control(64.0 / tb.fs, &field.rhythm, &field, &[], &[], 0.0);
                    }
                }
            }
            let me = learned.pitch_ctl.melodic.as_ref().unwrap();
            let records = &me.executions;
            for decision in me.decisions.iter().filter(|d| d.old_terms_replaced) {
                let previous = records
                    .iter()
                    .filter(|e| e.kind == ExecutionKind::On && e.sample < decision.at)
                    .max_by_key(|e| e.sample)
                    .unwrap();
                let snapshot = decision.snapshot.unwrap();
                let mut prefix = crate::life::melodic_continuity::Prefix::default();
                for e in records
                    .iter()
                    .filter(|e| e.kind == ExecutionKind::On && e.sample < decision.at)
                {
                    prefix.push(12.0 * (e.frequency_hz.unwrap() as f64).log2());
                }
                let expected = prefix.distribution().unwrap();
                assert_eq!(expected.mu, snapshot.distribution.mu);
                assert_eq!(prefix.n, snapshot.n);
                if previous.sample / 512 == decision.at / 512 {
                    same_hop += 1;
                }
            }
        }
        assert!(same_hop > 0, "same-hop causal witness required");
        println!("ME same-hop prior onset decisions={same_hop}");
    }

    #[test]
    fn melodic_real_producer_and_report_independent_statistics() {
        let mut a = produced_melodic(&[0, 2, -2]);
        let mut b = produced_melodic(&[-4, 2, -2]);
        let da = a.predict(30000).unwrap().distribution;
        let db = b.predict(30000).unwrap().distribution;
        assert!(da.variance > 0. && db.variance > 0.);
        assert_eq!(
            a.executions
                .iter()
                .filter(|r| r.kind == ExecutionKind::On)
                .count(),
            4
        );
        let mut quiet = crate::life::melodic_continuity::MelodicContinuity::new(
            48000.,
            1.,
            (220.0f32.log2(), 880.0f32.log2()),
        );
        for record in &a.executions {
            quiet.observe(record);
        }
        assert!(quiet.executions.is_empty());
        assert_eq!(
            serde_json::to_value(a.predict(30001).unwrap()).unwrap(),
            serde_json::to_value(quiet.predict(30001).unwrap()).unwrap()
        );
        assert_eq!(a.epoch, 0);
    }

    #[test]
    fn melodic_reconfigure_flushes_and_recovers_independently() {
        for sq_mode in 0..3 {
            let mut control = VoiceControl::default();
            control.pitch.freq = 330.;
            control.pitch.range_oct = 2.;
            control.pitch.pitch_apply_mode = PitchApplyMode::Rearticulate;
            control.pitch.proposal_interval_sec = Some(0.001);
            control.pitch.melodic_continuity = 1.;
            control.pitch.follow_own_sequence = if sq_mode > 0 { 1. } else { 0. };
            control.phonation.spec.timing =
                crate::scenario::PhonationTiming::Coupled(Default::default());
            control.phonation.spec.sustain_participation = true;
            control.phonation.spec.duration = crate::scenario::DurationSpec::WhileAlive;
            assert!(control.supports_own_sequence());
            let mut voice = Voice::spawn_from_control(
                control.clone(),
                ArticulationCoreConfig::Drone {
                    sway: None,
                    breath_gain_init: None,
                },
                2,
                0,
                VoiceMetadata::default(),
                48000.,
                None,
                21,
            );
            voice.configure_control_hop(512, 64, 48000.);
            if sq_mode == 2 {
                voice
                    .pitch_ctl
                    .own_sequence
                    .as_mut()
                    .unwrap()
                    .resource_stopped = true;
            }
            let me = voice.pitch_ctl.melodic.as_mut().unwrap();
            for p in [100., 100., 102., 100.] {
                me.prefix.push(p);
            }
            assert!(me.predict(0).is_some());
            me.trace_enabled = true;
            let field = Landscape::default();
            voice.set_control_tick(64);
            voice.tick_control(0.1, &field.rhythm, &field, &[], &[], 0.);
            assert!(voice.sequence_control.as_ref().unwrap().used > 0);
            control.pitch.range_oct = 1.;
            voice.apply_effective_control(
                control.clone(),
                Dirty {
                    body: false,
                    pitch: true,
                },
            );
            assert_eq!(voice.sequence_control.as_ref().unwrap().used, 0);
            let me = voice.pitch_ctl.melodic.as_ref().unwrap();
            assert_eq!(me.prefix.n, 0);
            assert_eq!(me.epoch, 1);
            assert!(me.decisions.iter().all(|d| !d.old_terms_replaced));
            control.pitch.move_cost_coeff = 17.;
            voice.apply_effective_control(
                control.clone(),
                Dirty {
                    body: false,
                    pitch: true,
                },
            );
            assert_eq!(voice.pitch_ctl.melodic.as_ref().unwrap().epoch, 1);
            control.pitch.pitch_apply_mode = PitchApplyMode::Glide;
            voice.apply_effective_control(
                control.clone(),
                Dirty {
                    body: false,
                    pitch: true,
                },
            );
            assert!(!voice.pitch_ctl.melodic.as_ref().unwrap().supported);
            control.pitch.pitch_apply_mode = PitchApplyMode::Rearticulate;
            voice.apply_effective_control(
                control,
                Dirty {
                    body: false,
                    pitch: true,
                },
            );
            let me = voice.pitch_ctl.melodic.as_mut().unwrap();
            assert!(me.supported);
            assert_eq!(me.epoch, 3);
            for p in [100., 100., 102., 100.] {
                me.prefix.push(p);
            }
            assert!(me.predict(512).is_some());
            if sq_mode == 2 {
                assert!(
                    voice
                        .pitch_ctl
                        .own_sequence
                        .as_ref()
                        .unwrap()
                        .resource_stopped
                );
            }
        }
    }

    fn produced_history(intervals: &[i32]) -> crate::life::own_sequence::OwnSequence {
        let mut control = VoiceControl::default();
        control.pitch.freq = 440.0;
        control.pitch.pitch_apply_mode = PitchApplyMode::Rearticulate;
        let mut voice = Voice::spawn_from_control(
            control,
            ArticulationCoreConfig::Drone {
                sway: None,
                breath_gain_init: None,
            },
            2,
            0,
            VoiceMetadata::default(),
            48000.0,
            None,
            21,
        );
        voice.configure_control_hop(512, 64, 48000.0);
        let mut observer = crate::life::own_sequence::OwnSequence::new(
            21,
            2,
            0,
            48000.0,
            1.0,
            (220.0f32.log2(), 880.0f32.log2()),
        );
        observer.trace_enabled = true;
        let mut deferred = DeferredPitch::default();
        let mut batch = PhonationBatch::default();
        voice.prepare_phonation_batch(&mut batch);
        let tb = Timebase {
            fs: 48000.0,
            hop: 512,
        };
        let mut semitones = 0;
        for i in 0..=intervals.len() {
            let at = i as u64 * 6000;
            if i > 0 {
                semitones += intervals[i - 1];
                let hz = (440.0 * 2.0f64.powf(semitones as f64 / 12.0)) as f32;
                voice.force_set_pitch_log2(hz.log2());
                voice
                    .participation_pitch
                    .rearticulations
                    .push(Rearticulation {
                        at,
                        pitch_log2: hz.log2(),
                    });
            }
            voice.tick_phonation_into(
                &tb,
                at,
                &NeuralRhythms::default(),
                None,
                0.0,
                1.0,
                0.8,
                &mut batch,
            );
            assert_eq!(batch.tones.len(), 1, "producer onset {i}");
            voice.pitch_ctl.own_sequence = Some(observer);
            voice.consume_sequence_execution(&mut deferred, &batch, &mut 0, &mut 0, at + 512);
            observer = voice.pitch_ctl.own_sequence.take().unwrap();
        }
        assert_eq!(observer.model.symbols.len(), intervals.len());
        observer
    }

    #[test]
    fn own_sequence_producer_order_and_field_controls() {
        // Registered contrasts: identical counts, endpoint, timestamps and paired noise.
        let a: Vec<_> = [2, -2].into_iter().cycle().take(48).collect();
        let b: Vec<_> = [-2, 2].into_iter().cycle().take(48).collect();
        let mut a = produced_history(&a);
        let mut b = produced_history(&b);
        let sa = a.predict(49 * 6000).unwrap();
        let sb = b.predict(49 * 6000).unwrap();
        assert_eq!(sa.previous_hz, sb.previous_hz);
        let current = 440.0f32.log2();
        let pitches = [current - 2.0 / 12.0, current, current + 2.0 / 12.0];
        let choose = |snapshot, field: [f32; 3]| {
            let mut core = PitchHillClimbPitchCore::new(200.0, current, 0.0);
            core.set_temperature(0.0);
            core.sequence_enabled = true;
            core.sequence_trace = true;
            core.sequence = Some(snapshot);
            let mut scores: Vec<_> = pitches.into_iter().zip(field).collect();
            let selected = core.select_with_sequence(
                current,
                field[1],
                &mut scores,
                &mut Vec::new(),
                &mut rand::rngs::SmallRng::seed_from_u64(42),
            );
            let record = core.sequence_decision.borrow_mut().take().unwrap();
            (selected.target_pitch_log2, record)
        };
        let (pa, ra) = choose(sa.clone(), [0.0; 3]);
        let (pb, rb) = choose(sb, [0.0; 3]);
        assert_eq!(pa, pitches[2]);
        assert_eq!(pb, pitches[0]);
        assert_eq!(ra.without_sequence_log2, current);
        assert_eq!(rb.without_sequence_log2, current);
        let (conflict, rc) = choose(sa, [20.0, 0.0, 0.0]);
        assert_eq!(conflict, pitches[0]);
        assert_eq!(ra.distribution, rc.distribution);
        for (pitch, observer, interval) in [(pa, &mut a, 2), (pb, &mut b, -2)] {
            let mut control = VoiceControl::default();
            control.pitch.freq = pitch.exp2();
            let mut voice = Voice::spawn_from_control(
                control,
                ArticulationCoreConfig::Drone {
                    sway: None,
                    breath_gain_init: None,
                },
                2,
                294000,
                VoiceMetadata::default(),
                48000.0,
                None,
                21,
            );
            let mut batch = PhonationBatch::default();
            voice.prepare_phonation_batch(&mut batch);
            voice.tick_phonation_into(
                &Timebase {
                    fs: 48000.0,
                    hop: 512,
                },
                294000,
                &NeuralRhythms::default(),
                None,
                0.0,
                1.0,
                0.8,
                &mut batch,
            );
            assert_eq!(batch.tones.len(), 1);
            assert_eq!(
                crate::life::own_sequence::encode(
                    batch.tones[0].freq_hz as f64,
                    observer.last_snapshot.as_ref().unwrap().previous_hz
                ),
                Some(interval)
            );
            println!(
                "SQ subsequent execution {}",
                serde_json::to_string(&batch.tones).unwrap()
            );
        }
        println!(
            "SQ order_a={} order_b={} field_conflict={} histories=48+48",
            pa.exp2(),
            pb.exp2(),
            conflict.exp2()
        );
        println!(
            "SQ contrasts {}",
            serde_json::to_string(&[ra, rb, rc]).unwrap()
        );
    }

    #[test]
    fn own_sequence_consumer_precision_boundaries_are_reported() {
        use rand::RngExt;
        let mut history = produced_history(&[2, -2]);
        let mut snapshot = history.predict(18000).unwrap();
        snapshot.weight = 1e-16;
        let current = 440.0f32.log2();
        let mut core = PitchHillClimbPitchCore::new(200.0, current, 0.0);
        core.sequence_enabled = true;
        core.sequence_trace = true;
        core.sequence = Some(snapshot);
        let mut evidence = Vec::new();
        for (temperature, scores, boundary) in [
            (
                0.0,
                vec![(current, 0.0), (current + 1.0 / 12.0, 0.1)],
                "greedy",
            ),
            (
                0.0,
                vec![(current, 0.0), (current + 1.0 / 12.0, 1e-6)],
                "rank",
            ),
            (
                0.15,
                {
                    let mut rng = rand::rngs::SmallRng::seed_from_u64(42);
                    let _ = rng.random_range(0..1usize);
                    let draw: f32 = rng.random_range(0.0..1.0);
                    vec![(current + 1.0 / 12.0, 0.15 * draw.ln())]
                },
                "acceptance",
            ),
        ] {
            core.set_temperature(temperature);
            core.select_with_sequence(
                current,
                0.0,
                &mut scores.clone(),
                &mut Vec::new(),
                &mut rand::rngs::SmallRng::seed_from_u64(42),
            );
            let record = core.sequence_decision.borrow_mut().take().unwrap();
            let audit = record.precision.as_ref().unwrap();
            assert!(
                match boundary {
                    "greedy" => audit.greedy_uncertain,
                    "rank" => audit.uncertain_rank_comparisons > 0,
                    _ => audit.acceptance_uncertain,
                },
                "{boundary}: {audit:?}"
            );
            evidence.push(record);
        }
        println!(
            "SQ consumer boundaries {}",
            serde_json::to_string(&evidence).unwrap()
        );
    }

    #[test]
    fn causal_execution_without_rescoring_preserves_commands_and_rng() {
        let mut same_hop = 0;
        for tie in [0.0, 0.5, 1.0] {
            let mut control = VoiceControl::default();
            control.pitch.freq = 330.0;
            control.phonation.tie = tie;
            control.pitch.pitch_apply_mode = PitchApplyMode::Rearticulate;
            control.pitch.proposal_interval_sec = Some(0.001);
            control.phonation.spec = crate::scenario::PhonationSpec {
                timing: crate::scenario::PhonationTiming::Coupled(
                    crate::scenario::CoupledTimingSpec {
                        relation: crate::scenario::RhythmRelation::Participating,
                        coupling: 0.8,
                        ..Default::default()
                    },
                ),
                duration: crate::scenario::DurationSpec::WhileAlive,
                sustain_participation: true,
            };
            let make = |control| {
                Voice::spawn_from_control(
                    control,
                    ArticulationCoreConfig::Drone {
                        sway: None,
                        breath_gain_init: None,
                    },
                    2,
                    0,
                    VoiceMetadata::default(),
                    48000.0,
                    None,
                    21,
                )
            };
            let mut old = make(control.clone());
            control.pitch.follow_own_sequence = 1.0;
            let mut learned = make(control.clone());
            learned
                .pitch_ctl
                .own_sequence
                .as_mut()
                .unwrap()
                .trace_enabled = true;
            let mut causal = make(control);
            causal.pitch_ctl.own_sequence = None;
            let tb = Timebase {
                fs: 48000.0,
                hop: 512,
            };
            let mut field = Landscape::new(Log2Space::new(55.0, 8000.0, 96));
            for (i, &pitch) in field.space.centers_log2.iter().enumerate() {
                field.consonance_field_score_eff[i] = (pitch * 7.0).sin();
            }
            for v in [&mut old, &mut causal, &mut learned] {
                v.configure_control_hop(512, 64, tb.fs);
            }
            let mut a = PhonationBatch::default();
            let mut b = PhonationBatch::default();
            let mut c = PhonationBatch::default();
            learned.prepare_phonation_batch(&mut c);
            old.prepare_phonation_batch(&mut a);
            causal.prepare_phonation_batch(&mut b);
            for hop in 0..512 {
                let at = hop * 512;
                for (v, batch) in [
                    (&mut old, &mut a),
                    (&mut causal, &mut b),
                    (&mut learned, &mut c),
                ] {
                    v.tick_phonation_into_with_arrival(
                        &tb,
                        at,
                        &field.rhythm,
                        None,
                        0.0,
                        1.0,
                        0.8,
                        None,
                        Some(&field.space),
                        batch,
                    );
                }
                assert_eq!(
                    serde_json::to_value(&a).unwrap(),
                    serde_json::to_value(&b).unwrap(),
                    "hop {hop}"
                );
                for step in 0..8 {
                    for v in [&mut old, &mut causal, &mut learned] {
                        v.set_control_tick(at + 512 + step * 64);
                        v.tick_control(64.0 / tb.fs, &field.rhythm, &field, &[], &[], 0.0);
                    }
                }
            }
            let records = &learned.pitch_ctl.own_sequence.as_ref().unwrap().executions;
            for decision in learned
                .pitch_ctl
                .sequence_decisions
                .iter()
                .filter(|d| d.active)
            {
                let previous = records
                    .iter()
                    .filter(|e| e.kind == ExecutionKind::On && e.sample < decision.at)
                    .max_by_key(|e| e.sample)
                    .unwrap();
                assert_eq!(decision.previous_hz, previous.frequency_hz.map(f64::from));
                if previous.sample / 512 == decision.at / 512 {
                    same_hop += 1;
                }
            }
        }
        assert!(same_hop > 0, "same-hop causal witness required");
        println!("SQ same-hop prior onset decisions={same_hop}");
    }
}

#[derive(Debug)]
struct PitchStep {
    at: Tick,
    current_freq: f32,
    rhythms: NeuralRhythms,
    dt: f32,
    pitch: PitchControl,
    neighbors: Vec<f32>,
    salience: Vec<f32>,
    pressure: ListenerPressure,
}

impl DeferredPitch {
    pub(crate) fn reserve(&mut self, capacity: usize) {
        self.steps.resize_with(capacity, || PitchStep {
            at: 0,
            current_freq: 440.0,
            rhythms: NeuralRhythms::default(),
            dt: 0.0,
            pitch: PitchControl::default(),
            neighbors: Vec::with_capacity(16),
            salience: Vec::with_capacity(16),
            pressure: ListenerPressure::default(),
        });
        self.events.reserve(2 * capacity + 8);
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn push(
        &mut self,
        at: Option<Tick>,
        current_freq: f32,
        rhythms: &NeuralRhythms,
        dt: f32,
        landscape: &Landscape,
        pitch: &PitchControl,
        neighbors: &[f32],
        salience: &[f32],
        pressure: ListenerPressure,
    ) {
        let Some(at) = at else {
            self.missing = true;
            return;
        };
        if self.field.is_none() {
            self.field = Some(Arc::new(landscape.clone()));
        }
        if self.used == self.steps.len() {
            self.steps.push(PitchStep {
                at,
                current_freq,
                rhythms: *rhythms,
                dt,
                pitch: pitch.clone(),
                neighbors: Vec::new(),
                salience: Vec::new(),
                pressure,
            });
        }
        let step = &mut self.steps[self.used];
        step.at = at;
        step.current_freq = current_freq;
        step.rhythms = *rhythms;
        step.dt = dt;
        step.pitch.clone_from(pitch);
        step.pressure = pressure;
        step.neighbors.clear();
        step.neighbors.extend_from_slice(neighbors);
        step.salience.clear();
        step.salience.extend_from_slice(salience);
        self.used += 1;
    }
}

impl Voice {
    pub(super) fn flush_sequence_controls(&mut self) {
        let Some(mut deferred) = self.sequence_control.take() else {
            return;
        };
        for step in &deferred.steps[..deferred.used] {
            if let Some(field) = &deferred.field {
                self.control_tick = Some(step.at);
                self.pitch_ctl.apply_control(&step.pitch);
                self.decide_pitch_from_frequency(
                    step.current_freq,
                    &step.rhythms,
                    step.dt,
                    field,
                    &step.neighbors,
                    &step.salience,
                    step.pressure,
                );
            }
        }
        deferred.used = 0;
        deferred.field = None;
        self.sequence_control = Some(deferred);
    }

    pub(crate) fn finish_sequence_execution(&mut self, out: &PhonationBatch, end: Tick) {
        let Some(mut deferred) = self.sequence_control.take() else {
            return;
        };
        if deferred.recorded_end != Some(end) {
            self.consume_sequence_execution(&mut deferred, out, &mut 0, &mut 0, end);
            deferred.recorded_end = Some(end);
        }
        self.sequence_control = Some(deferred);
    }

    fn consume_sequence_execution(
        &mut self,
        deferred: &mut DeferredPitch,
        out: &PhonationBatch,
        commands: &mut usize,
        tones: &mut usize,
        through_exclusive: Tick,
    ) {
        for command in &out.cmds[*commands..] {
            let (kind, tone_id, sample, frequency_hz) = match command {
                ToneCmd::Off { tone_id, off_tick } => {
                    (ExecutionKind::Off, *tone_id, *off_tick, None)
                }
                ToneCmd::Update {
                    tone_id,
                    at_tick,
                    update,
                } if update.target_freq_hz.is_some() => (
                    if at_tick.is_some() {
                        ExecutionKind::PitchUpdate
                    } else {
                        ExecutionKind::Missing
                    },
                    *tone_id,
                    at_tick.unwrap_or(through_exclusive.saturating_sub(1)),
                    update.target_freq_hz,
                ),
                _ => continue,
            };
            deferred.events.push(ExecutionRecord {
                voice_id: self.id,
                generation: self.metadata.generation,
                tone_id,
                kind,
                sample,
                available_at: sample,
                execution_sequence: 0,
                frequency_hz,
                strength: None,
            });
        }
        *commands = out.cmds.len();
        for tone in &out.tones[*tones..] {
            deferred.events.push(ExecutionRecord {
                voice_id: self.id,
                generation: self.metadata.generation,
                tone_id: tone.tone_id,
                kind: ExecutionKind::On,
                sample: tone.onset,
                available_at: tone.onset,
                execution_sequence: 0,
                frequency_hz: Some(tone.freq_hz),
                strength: out
                    .onsets
                    .iter()
                    .find(|on| on.onset_tick == tone.onset)
                    .map(|on| on.strength),
            });
        }
        *tones = out.tones.len();
        deferred.events.sort_by_key(|e| {
            (
                e.sample,
                match e.kind {
                    ExecutionKind::Off => 0,
                    ExecutionKind::PitchUpdate | ExecutionKind::Missing => 1,
                    ExecutionKind::On => 2,
                },
            )
        });
        let count = deferred
            .events
            .partition_point(|e| e.sample < through_exclusive);
        for mut record in deferred.events.drain(..count) {
            record.execution_sequence = deferred.next_execution;
            deferred.next_execution += 1;
            if let Some(me) = &mut self.pitch_ctl.melodic {
                me.observe(&record);
            }
            if let Some(sequence) = &mut self.pitch_ctl.own_sequence {
                sequence.observe(record);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_sequence_participation(
        &mut self,
        ctx: &CoreTickCtx,
        state: &CoreState,
        consonance: f32,
        target_amp: f32,
        arrival: Option<&crate::life::arrival_cost::ArrivalContext<'_>>,
        space: Option<&Log2Space>,
        out: &mut PhonationBatch,
    ) {
        let mut deferred = self.sequence_control.take().expect("sequence control");
        if deferred.missing {
            deferred.events.push(ExecutionRecord {
                voice_id: self.id,
                generation: self.metadata.generation,
                tone_id: 0,
                kind: ExecutionKind::Missing,
                sample: ctx.now_tick,
                available_at: ctx.now_tick,
                execution_sequence: 0,
                frequency_hz: None,
                strength: None,
            });
            deferred.missing = false;
        }
        if let Some(shape) = self.phonation_engine.clock.shape_mut() {
            shape.enabled = false;
            if let Some(tie) = shape.tie.as_mut() {
                tie.enabled = true;
            }
        }
        self.prepare_participation_hop(ctx, state, out);
        let mut emitted = 0;
        let mut command_cursor = 0;
        let mut tone_cursor = 0;
        let mut cursor = ctx.now_tick;
        for index in 0..deferred.used {
            let at = deferred.steps[index].at;
            if at < ctx.now_tick || at >= ctx.frame_end {
                continue;
            }
            if cursor < at {
                let segment = self.phonation_segment(ctx, cursor, at);
                self.prepare_sequence_segment(&segment, space);
                self.tick_participation_segment(&segment, state, arrival, out);
                self.materialize_sequence_events(&mut emitted, target_amp, out);
                self.consume_sequence_execution(
                    &mut deferred,
                    out,
                    &mut command_cursor,
                    &mut tone_cursor,
                    at,
                );
            }
            let step = &deferred.steps[index];
            // The preceding segment has committed every onset strictly before this decision.
            if let Some(field) = &deferred.field {
                self.control_tick = Some(step.at);
                let live_pitch =
                    std::mem::replace(&mut self.effective_control.pitch, step.pitch.clone());
                self.pitch_ctl.apply_control(&step.pitch);
                self.decide_pitch_from_frequency(
                    step.current_freq,
                    &step.rhythms,
                    step.dt,
                    field,
                    &step.neighbors,
                    &step.salience,
                    step.pressure,
                );
                self.effective_control.pitch = live_pitch;
                if let Some(adoption) = self.participation_pitch.rearticulations.last()
                    && adoption.at == step.at
                {
                    self.participation_pitch.pending = Some(adoption.pitch_log2);
                }
            }
            let segment = self.phonation_segment(ctx, step.at, step.at.saturating_add(1));
            self.prepare_sequence_segment(&segment, space);
            self.tick_participation_segment(&segment, state, arrival, out);
            self.materialize_sequence_events(&mut emitted, target_amp, out);
            cursor = step.at.saturating_add(1);
            self.consume_sequence_execution(
                &mut deferred,
                out,
                &mut command_cursor,
                &mut tone_cursor,
                cursor,
            );
        }
        if cursor < ctx.frame_end {
            let segment = self.phonation_segment(ctx, cursor, ctx.frame_end);
            self.prepare_sequence_segment(&segment, space);
            self.tick_participation_segment(&segment, state, arrival, out);
            self.materialize_sequence_events(&mut emitted, target_amp, out);
        }
        self.consume_sequence_execution(
            &mut deferred,
            out,
            &mut command_cursor,
            &mut tone_cursor,
            ctx.frame_end,
        );
        self.pitch_ctl.apply_control(&self.effective_control.pitch);
        self.apply_phonation_onsets(consonance, out);
        self.update_surviving_render_notes(ctx.now_tick, target_amp, out);
        self.consume_sequence_execution(
            &mut deferred,
            out,
            &mut command_cursor,
            &mut tone_cursor,
            ctx.frame_end,
        );
        self.finish_participation_batch(out);
        deferred.used = 0;
        deferred.field = None;
        deferred.recorded_end = Some(ctx.frame_end);
        self.sequence_control = Some(deferred);
    }

    fn prepare_sequence_segment(&mut self, ctx: &CoreTickCtx, space: Option<&Log2Space>) {
        if let Some(tie) = self
            .phonation_engine
            .clock
            .shape_mut()
            .and_then(|s| s.tie.as_mut())
        {
            tie.pending_pitch = self.participation_pitch.pending.is_some();
            let pitch = self.body.base_freq_hz();
            self.phonation_engine.set_span_pitch(
                ctx.now_tick,
                pitch,
                self.participation_pitch.pending.map_or(pitch, f32::exp2),
            );
        } else {
            self.prepare_span_segment(ctx, space);
        }
    }

    fn materialize_sequence_events(
        &mut self,
        emitted: &mut usize,
        target_amp: f32,
        out: &mut PhonationBatch,
    ) {
        let mut pitch = self.body.base_freq_hz().log2();
        while *emitted < self.phonation_scratch.events.len() {
            self.emit_bounded_on(*emitted, &mut pitch, target_amp, out);
            *emitted += 1;
        }
        self.body.set_pitch_log2(pitch);
    }
}
