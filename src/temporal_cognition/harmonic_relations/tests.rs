use super::*;
use derived::OrnamentRole;

pub(super) struct Input {
    time: f64,
    pitch: f64,
    mass: f64,
    focus: f64,
    competition: f64,
    bins: Vec<Bin>,
    trajectories: Vec<Trajectory>,
    events: Vec<Event>,
    snapshot: u64,
}
impl Input {
    pub(super) fn new(time: f64, bin: usize) -> Self {
        Self {
            time,
            pitch: bin as f64,
            mass: 1.,
            focus: 1.,
            competition: 0.,
            bins: vec![Bin {
                index: bin,
                mass: 1.,
            }],
            trajectories: vec![Trajectory {
                source: Source {
                    kind: 0,
                    id: 1,
                    generation: 0,
                },
                pitch: Value::exact(bin as f64),
                amplitude: Value::exact(1.),
                timbre: Value::exact(0.),
                articulation: 0,
                sounding: true,
            }],
            events: vec![Event {
                source: Some(Source {
                    kind: 0,
                    id: 1,
                    generation: 0,
                }),
                flags: OPPORTUNITY,
            }],
            snapshot: time as u64 + 1,
        }
    }
    pub(super) fn observation(&self) -> Observation<'_> {
        Observation {
            time: self.time,
            timing_error: 0.,
            hop: true,
            acoustic: Acoustic {
                snapshot: self.snapshot,
                sampled_at: self.time,
                coordinate_version: 1,
                mass: Value::exact(self.mass),
                focus: Value::exact(self.focus),
                competition: Value::exact(self.competition),
            },
            support: Support {
                bins: &self.bins,
                residual: 0.,
                error: 0.,
                bound: false,
                valid: self.mass > 0.,
            },
            trajectories: &self.trajectories,
            events: &self.events,
        }
    }
    pub(super) fn boundary(mut self) -> Self {
        self.events[0].flags |= BOUNDARY;
        self
    }
    fn pitch(mut self, pitch: f64) -> Self {
        self.pitch = pitch;
        self.trajectories[0].pitch = Value::exact(pitch);
        self
    }
}
pub(super) fn model() -> Model {
    Model::with_limits(
        Load {
            sample_rate: 16,
            hop: 1,
            sources: 2,
            coordinates: (0..8).map(f64::from).collect(),
            sparse_entries: 8,
        },
        Limits {
            seconds: 600.,
            records: 32,
        },
    )
    .unwrap()
}

#[test]
fn background_prediction_survives_hop_only_observation_but_not_coordinate_change() {
    let mut m = model();
    for i in 0..6 {
        m.observe(Input::new(i as f64, i % 2).observation())
            .unwrap();
    }
    let version = m.predict_latest(6., 42).unwrap().model_version;
    let mut hop = Input::new(7., 1);
    hop.events.clear();
    m.observe(hop.observation()).unwrap();
    assert_eq!(m.version, version);
    assert_eq!(m.prediction.query_version, 42);
    let mut changed = hop.observation();
    changed.time = 8.;
    changed.acoustic.coordinate_version = 2;
    m.observe(changed).unwrap();
    assert_ne!(m.version, version);
    assert_eq!(m.predict_latest(8., 43).unwrap().unknown, 1.);
    assert!(m.predict_latest(7., 44).is_err());
}
pub(super) fn observe(m: &mut Model, input: &Input) {
    m.observe(input.observation()).unwrap();
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-10, "{a} != {b}");
}

#[test]
fn support_embedding_variance_and_expiration_follow_sample_identity() {
    let mut m = model();
    for (t, b) in [(0., 0), (1., 1), (2., 0)] {
        observe(&mut m, &Input::new(t, b));
    }
    close(m.graph.shape_variance(), 1. / 3.);
    m.graph.expire(601.5);
    assert_eq!(m.graph.anchors.len(), 1);
    close(m.graph.shape_variance(), 0.);
    m.graph.expire(604.);
    assert_eq!(m.graph.shape_count_for_test(), 0);
}
#[test]
fn zero_scale_is_exact_and_bounds_do_not_become_tuning_tolerance() {
    let mut k = Kernel::default();
    k.scalar(Value::exact(1.), Value::exact(1.), 0.);
    close(k.compatibility(), 1.);
    k.scalar(Value::exact(1.), Value::exact(2.), 0.);
    close(k.compatibility(), 0.);
    let mut k = Kernel::default();
    k.scalar(
        Value {
            error: 2.,
            bound: true,
            ..Value::exact(1.)
        },
        Value::exact(3.),
        0.,
    );
    close(k.compatibility(), (-0.5f64).exp());
}
#[test]
fn removable_scalar_statistics_do_not_retain_evicted_exposure() {
    let mut s = Moments::default();
    for x in [1., 2., 4.] {
        s.add(x);
    }
    close(s.variance(), 7. / 3.);
    s.remove(2.);
    close(s.variance(), 4.5);
    s.remove(1.);
    close(s.variance(), 0.);
}
#[test]
fn hop_nodes_do_not_copy_support_and_coincident_flags_count_once() {
    let mut m = model();
    let mut input = Input::new(0., 2);
    input.events[0].flags = ONSET | OPPORTUNITY;
    observe(&mut m, &input);
    for t in 1..5 {
        let mut hop = Input::new(f64::from(t), 3);
        hop.events.clear();
        observe(&mut m, &hop);
    }
    let r = m.resources();
    assert_eq!(r.nodes, 5);
    assert_eq!(r.anchors, 1);
    assert_eq!(r.records, 1);
    assert_eq!(r.retained_support_entries, 1);
    assert_eq!(m.graph.shape_count_for_test(), 1);
}
#[test]
fn between_hop_events_keep_original_sample_timestamp() {
    let mut m = model();
    let x = Input::new(0., 2);
    observe(&mut m, &x);
    let mut event = x.observation();
    event.time = 0.25;
    event.hop = false;
    m.observe(event).unwrap();
    let a = m.graph.observation(m.graph.anchors.back().unwrap());
    assert_eq!(a.time, 0.25);
    assert_eq!(a.acoustic.sampled_at, 0.);
}
#[test]
fn malformed_input_has_no_partial_commit() {
    let mut m = model();
    let mut input = Input::new(0., 2);
    input.bins.push(Bin { index: 2, mass: 0. });
    assert!(m.observe(input.observation()).is_err());
    assert_eq!(m.version, 0);
    assert!(m.graph.nodes.is_empty());
}
#[test]
fn resource_chunk_context_is_finalized_before_the_next_query_arrives() {
    let mut m = model();
    for t in 0..SMAX {
        observe(&mut m, &Input::new(t as f64, t % 4));
    }
    let chunk = *m.graph.chunks.back().unwrap();
    assert!(chunk.closed);
    let contexts: Vec<_> = m.contexts.active.iter().flatten().copied().collect();
    assert_eq!(contexts.len(), 1);
    assert_eq!(contexts[0].representative, chunk.id);
    assert_eq!(contexts[0].last, (SMAX - 1) as f64);
    observe(&mut m, &Input::new(SMAX as f64, 7));
    assert_eq!(m.graph.chunks.back().unwrap().episode, chunk.episode);
    assert_eq!(
        m.contexts
            .active
            .iter()
            .flatten()
            .map(|c| c.mass)
            .sum::<f64>(),
        1.
    );
}
#[test]
fn resource_chunking_preserves_episode_and_identity_but_evicts_complete_chunks() {
    let mut m = model();
    for t in 0..48 {
        observe(&mut m, &Input::new(t as f64, t % 4));
    }
    assert!(m.graph.anchor(1).is_none());
    assert!(m.graph.anchor(17).is_some());
    assert_eq!(m.graph.anchors.len(), 32);
    assert_eq!(m.graph.chunks.len(), 2);
    assert_eq!(m.graph.chunks[0].episode, m.graph.chunks[1].episode);
    let r = m.resources();
    assert!(r.records <= 32 && r.nodes <= r.node_capacity && r.edges <= r.edge_capacity);
    assert!(
        m.graph
            .edges()
            .all(|e| m.graph.node(e.from).is_some() && m.graph.node(e.to).is_some())
    );
    close(m.graph.shape_variance(), 32. * 0.75 / (2. * 31.));
}
#[test]
fn source_crossings_preserve_identity_and_rearticulation_is_explicit() {
    let mut m = model();
    let mut x = Input::new(0., 1);
    let mut second = x.trajectories[0];
    second.source.id = 2;
    second.pitch = Value::exact(4.);
    x.trajectories.push(second);
    observe(&mut m, &x);
    x.time = 1.;
    x.trajectories[0].pitch = Value::exact(4.);
    x.trajectories[1].pitch = Value::exact(1.);
    observe(&mut m, &x);
    assert_eq!(m.relations.continuity[0].source.id, 1);
    assert_eq!(m.relations.continuity[0].displacement.value, 3.);
    x.time = 2.;
    x.trajectories[0].articulation = 1;
    observe(&mut m, &x);
    assert!(m.relations.continuity[0].retained && m.relations.continuity[0].rearticulated);
}
#[test]
fn silence_and_missing_support_do_not_equal_uniform_support() {
    let mut m = model();
    observe(&mut m, &Input::new(0., 1));
    observe(&mut m, &Input::new(1., 1));
    let mass = m.local.mass();
    assert!(mass > 0.);
    let mut silence = Input::new(1.1, 1);
    silence.mass = 0.;
    silence.trajectories.clear();
    observe(&mut m, &silence);
    close(m.local.mass(), mass * 0.5);
    assert_eq!(m.relations.competition.removed, 1);
    let missing_input = Input::new(1.2, 1);
    let mut missing = missing_input.observation();
    missing.support.valid = false;
    m.observe(missing).unwrap();
    assert!(m.local.unknown_mass > 0.);
}
#[test]
fn prediction_is_fixed_before_learning_and_no_successor_is_unknown() {
    let mut m = model();
    observe(&mut m, &Input::new(0., 1));
    let q = Input::new(10., 1);
    assert_eq!(m.predict(&[q.observation()], 90).unwrap().unknown, 1.);
    let frozen = m.prediction.unknown;
    observe(&mut m, &Input::new(11., 2));
    assert_eq!(m.prediction.unknown, frozen);
    let q = Input::new(20., 1);
    let p = m.predict(&[q.observation()], 91).unwrap();
    assert!(p.len > 0);
    assert_eq!(p.query_version, 91);
    assert!(p.unknown < 1.);
}
#[test]
fn order_changes_successors_with_fixed_marginals_and_freeze_ablation() {
    let mut a = model();
    let mut b = model();
    for (i, bin) in [1, 2, 3].into_iter().enumerate() {
        observe(&mut a, &Input::new(i as f64, bin));
    }
    for (i, bin) in [1, 3, 2].into_iter().enumerate() {
        observe(&mut b, &Input::new(i as f64, bin));
    }
    let q = Input::new(10., 1);
    a.predict(&[q.observation()], 99).unwrap();
    b.predict(&[q.observation()], 99).unwrap();
    assert_ne!(a.prediction.atoms[0].bins, b.prediction.atoms[0].bins);
    a.remove_correspondences();
    b.remove_correspondences();
    assert_eq!(a.predict(&[q.observation()], 100).unwrap().unknown, 1.);
    assert_eq!(b.predict(&[q.observation()], 100).unwrap().unknown, 1.);
    let mut a = model();
    let mut b = model();
    a.freeze_learning();
    b.freeze_learning();
    for (i, bin) in [1, 2, 3].into_iter().enumerate() {
        observe(&mut a, &Input::new(i as f64, bin));
    }
    for (i, bin) in [1, 3, 2].into_iter().enumerate() {
        observe(&mut b, &Input::new(i as f64, bin));
    }
    assert_eq!(
        a.predict(&[q.observation()], 101).unwrap().unknown,
        b.predict(&[q.observation()], 101).unwrap().unknown
    );
}
#[test]
fn omission_pruning_and_partial_streams_preserve_unknown_mass() {
    let mut m = model();
    for (i, b) in [1, 2, 3, 4].into_iter().enumerate() {
        observe(&mut m, &Input::new(i as f64, b));
    }
    let a = Input::new(20., 1);
    let b = Input::new(21., 7);
    let c = Input::new(22., 3);
    let p = m
        .predict(&[a.observation(), b.observation(), c.observation()], 12)
        .unwrap();
    assert!(p.unknown > 0.);
    assert!(p.len > 0);
    close(
        p.unknown + p.atoms[..p.len].iter().map(|a| a.weight).sum::<f64>(),
        1.,
    );
    assert!(
        p.atoms[..p.len]
            .iter()
            .any(|a| a.mapping.omitted_query != 0)
    );
}
#[test]
fn translation_and_time_dilation_use_only_prefix() {
    let mut m = model();
    for (t, b) in [(0., 1), (1., 2), (2., 3)] {
        observe(&mut m, &Input::new(t, b));
    }
    let a = Input::new(10., 2);
    let b = Input::new(12., 3);
    let p = m.predict(&[a.observation(), b.observation()], 1).unwrap();
    let transformed = p.atoms[..p.len]
        .iter()
        .find(|a| a.mapping.transformed && a.mapping.len == 2)
        .unwrap();
    assert_eq!(transformed.mapping.source_episode, 1);
    assert_eq!(&transformed.mapping.historical_nodes[..2], &[1, 2]);
    assert_eq!(&transformed.mapping.current_snapshots[..2], &[11, 13]);
    assert_eq!(transformed.mapping.history_paths[1], Some((1, 2)));
    close(transformed.mapping.translation.value, 1.);
    close(transformed.mapping.time_scale.value, 2.);
    close(transformed.time.value, 14.);
    assert_eq!(transformed.bins, [Bin { index: 4, mass: 1. }]);
    assert!(!transformed.acoustic.focus.valid && !transformed.acoustic.competition.valid);
}
#[test]
fn arrivals_use_measurement_envelope_not_historical_variance() {
    let mut m = model();
    observe(&mut m, &Input::new(0., 1));
    observe(&mut m, &Input::new(1., 2));
    let q = Input::new(10., 1);
    m.predict(&[q.observation()], 5).unwrap();
    observe(&mut m, &Input::new(11., 2).boundary());
    assert!(
        m.relations.arrivals[..m.relations.arrival_count]
            .iter()
            .any(|a| a.categorical)
    );
    assert!(m.relations.cadence.candidate());
    let q = Input::new(20., 1);
    m.predict(&[q.observation()], 6).unwrap();
    observe(&mut m, &Input::new(21.1, 2));
    assert!(
        !m.relations.arrivals[..m.relations.arrival_count]
            .iter()
            .any(|a| a.categorical)
    );
}
#[test]
fn retrospective_roles_wait_for_observed_successor() {
    let mut m = model();
    observe(&mut m, &Input::new(0., 1));
    observe(&mut m, &Input::new(1., 2));
    assert!(m.relations.ornaments.is_empty());
    observe(&mut m, &Input::new(2., 3));
    assert_eq!(m.relations.ornaments[0].role, OrnamentRole::Passing);
    observe(&mut m, &Input::new(3., 2));
    assert_eq!(m.relations.ornaments[0].role, OrnamentRole::Return);
    observe(&mut m, &Input::new(4., 2));
    observe(&mut m, &Input::new(5., 4));
    assert_eq!(m.relations.ornaments[0].role, OrnamentRole::DelayedApproach);
}
#[test]
fn contexts_are_created_on_completed_chunks_and_expire_with_evidence() {
    let mut m = model();
    observe(&mut m, &Input::new(0., 1));
    assert_eq!(m.contexts.active.iter().flatten().count(), 0);
    observe(&mut m, &Input::new(1., 2).boundary());
    assert_eq!(m.contexts.active.iter().flatten().count(), 1);
    m.graph.expire(1000.);
    m.contexts.expire(&m.graph);
    assert_eq!(m.contexts.active.iter().flatten().count(), 0);
}
#[test]
fn substitution_preserves_distinct_intermediate_and_shared_surroundings() {
    let mut m = model();
    for (t, b) in [(0., 1), (1., 2), (2., 4)] {
        observe(&mut m, &Input::new(t, b));
    }
    observe(&mut m, &Input::new(3., 7).boundary());
    for (t, b) in [(10., 1), (11., 3), (12., 4)] {
        observe(&mut m, &Input::new(t, b));
    }
    assert!(
        m.relations
            .substitutions
            .iter()
            .flatten()
            .any(|s| s.alternative == 2
                && s.predecessor > 0.
                && s.successor.is_some_and(|k| k > 0.)
                && s.observed_successor)
    );
}
#[test]
fn resource_report_separates_sparse_occupancy_from_dense_bound() {
    let mut m = model();
    observe(&mut m, &Input::new(0., 1));
    let r = m.resources();
    assert_eq!(r.retained_support_entries, 1);
    assert_eq!(r.dense_support_bytes, 32 * 8 * size_of::<Bin>());
    assert!(r.allocated_bytes > r.dense_support_bytes);
    println!(
        "HV5 sparse_entries={} dense_support_bytes={} allocated_bytes={} node_capacity={} edge_capacity={}",
        r.retained_support_entries,
        r.dense_support_bytes,
        r.allocated_bytes,
        r.node_capacity,
        r.edge_capacity
    );
}

#[test]
fn transported_interpolation_and_out_of_range_mass_are_conservative() {
    let mut m = model();
    for (t, b) in [(0., 4), (1., 5)] {
        observe(&mut m, &Input::new(t, b));
    }
    let mut successor = Input::new(2., 6);
    successor.bins = vec![
        Bin {
            index: 6,
            mass: 0.5,
        },
        Bin {
            index: 7,
            mass: 0.5,
        },
    ];
    observe(&mut m, &successor);
    let a = Input::new(10., 4).pitch(4.5);
    let b = Input::new(12., 5).pitch(5.5);
    let p = m.predict(&[a.observation(), b.observation()], 50).unwrap();
    let atom = p.atoms[..p.len]
        .iter()
        .find(|a| a.mapping.transformed && a.mapping.len == 2)
        .unwrap();
    close(atom.residual, 0.5);
    close(atom.bins.iter().map(|b| f64::from(b.mass)).sum(), 1.);
    assert_eq!(
        atom.bins,
        [
            Bin {
                index: 6,
                mass: 0.5
            },
            Bin {
                index: 7,
                mass: 0.5
            }
        ]
    );
    close(
        p.unknown + p.atoms[..p.len].iter().map(|a| a.weight).sum::<f64>(),
        1.,
    );
    assert!(p.for_query(51).is_none());
    assert!(p.for_query(50).is_some());
}
#[test]
fn simultaneous_stream_correspondences_keep_both_source_bindings() {
    let mut m = model();
    for t in 0..3 {
        let mut x = Input::new(t as f64, t + 1);
        let mut other = x.trajectories[0];
        other.source.id = 2;
        other.pitch = Value::exact(6. - t as f64);
        x.trajectories.push(other);
        observe(&mut m, &x);
    }
    let mut a = Input::new(10., 1);
    let mut b = Input::new(11., 2);
    for x in [&mut a, &mut b] {
        let mut other = x.trajectories[0];
        other.source.id = 2;
        other.pitch = Value::exact(if x.time == 10. { 6. } else { 5. });
        x.trajectories.push(other);
    }
    let p = m.predict(&[a.observation(), b.observation()], 1).unwrap();
    assert!(
        p.atoms[..p.len]
            .iter()
            .any(|a| a.mapping.query_source.is_some_and(|s| s.id == 1)
                && a.mapping.history_source.is_some_and(|s| s.id == 1))
    );
    assert!(
        p.atoms[..p.len]
            .iter()
            .any(|a| a.mapping.query_source.is_some_and(|s| s.id == 2)
                && a.mapping.history_source.is_some_and(|s| s.id == 2))
    );
    for a in &p.atoms[..p.len] {
        assert!(
            a.mapping.pairs[..a.mapping.len]
                .windows(2)
                .all(|w| w[0].query < w[1].query && w[0].history < w[1].history)
        );
    }
}
#[test]
fn matched_conditional_shuffle_preserves_predictive_distribution() {
    let mut a = model();
    let mut b = model();
    for (m, episodes) in [(&mut a, [(1, 2), (3, 4)]), (&mut b, [(3, 4), (1, 2)])] {
        for (i, (x, y)) in episodes.into_iter().enumerate() {
            observe(m, &Input::new((2 * i) as f64, x));
            observe(m, &Input::new((2 * i + 1) as f64, y).boundary());
        }
    }
    let q = Input::new(20., 1);
    a.predict(&[q.observation()], 1).unwrap();
    b.predict(&[q.observation()], 1).unwrap();
    let distribution = |m: &Model| {
        let mut bins = [0.; 8];
        for atom in &m.prediction.atoms[..m.prediction.len] {
            for b in &atom.bins {
                bins[b.index] += atom.weight * f64::from(b.mass);
            }
        }
        bins
    };
    for (a, b) in distribution(&a).into_iter().zip(distribution(&b)) {
        close(a, b);
    }
    close(a.prediction.unknown, b.prediction.unknown);
}
#[test]
fn silence_remains_a_known_successor_without_a_support_shape() {
    let mut m = model();
    observe(&mut m, &Input::new(0., 1));
    let mut silence = Input::new(1., 1);
    silence.mass = 0.;
    silence.focus = 0.;
    silence.trajectories[0].sounding = false;
    observe(&mut m, &silence);
    let q = Input::new(10., 1);
    let p = m.predict(&[q.observation()], 1).unwrap();
    assert!(p.atoms[..p.len].iter().any(|a| a.weight > 0.
        && a.acoustic.mass.valid
        && a.acoustic.mass.value == 0.
        && !a.shape_valid));
}
#[test]
fn coordinate_change_expires_references_and_old_prediction() {
    let mut m = model();
    observe(&mut m, &Input::new(0., 1));
    observe(&mut m, &Input::new(1., 2));
    let q = Input::new(10., 1);
    m.predict(&[q.observation()], 1).unwrap();
    let input = Input::new(11., 2);
    let mut o = input.observation();
    o.acoustic.coordinate_version = 2;
    m.observe(o).unwrap();
    assert_eq!(m.relations.arrival_count, 0);
    assert!(m.graph.anchor(1).is_none());
    assert_eq!(m.graph.shape_count_for_test(), 1);
    let later = Input::new(12., 1);
    assert_eq!(m.predict(&[later.observation()], 2).unwrap().unknown, 1.);
}
#[test]
fn edge_views_preserve_relative_and_absolute_trajectory_evidence() {
    let mut m = model();
    observe(&mut m, &Input::new(0., 1));
    observe(&mut m, &Input::new(2., 3));
    let edge = m
        .graph
        .edges()
        .find(|e| matches!(e.kind, graph::EdgeKind::Continuation(_)))
        .unwrap();
    let attrs = m.graph.edge_attributes(edge).unwrap();
    close(attrs.relative_pitch.value, 2.);
    close(attrs.relative_time.value, 2.);
    close(attrs.absolute_speed.value, 1.);
    assert_eq!(attrs.absolute_register.map(|v| v.value), [1., 3.]);
    assert_eq!(attrs.source_continuity.unwrap().id, 1);
    m.graph.expire(1000.);
    assert!(m.graph.edge_attributes(edge).is_none());
}
#[test]
fn causal_order_changes_context_likelihood_with_the_same_current_chunk() {
    let mut a = model();
    let mut b = model();
    for (m, order) in [(&mut a, [1, 2, 3]), (&mut b, [2, 1, 3])] {
        for (i, bin) in order.into_iter().enumerate() {
            let x = Input::new(i as f64, bin);
            observe(m, &if i == 2 { x.boundary() } else { x });
        }
        for (i, bin) in [1, 2, 3].into_iter().enumerate() {
            let x = Input::new(10. + i as f64, bin);
            observe(m, &if i == 2 { x.boundary() } else { x });
        }
    }
    assert_ne!(a.contexts.unknown, b.contexts.unknown);
}
#[test]
fn correspondence_removal_keeps_history_and_stops_current_context_attribution() {
    let mut m = model();
    for (i, bin) in [1, 2, 3, 1, 2, 3].into_iter().enumerate() {
        let input = Input::new(i as f64, bin);
        observe(&mut m, &if i % 3 == 2 { input.boundary() } else { input });
    }
    let retained: Vec<_> = m
        .contexts
        .active
        .iter()
        .flatten()
        .map(|c| (c.id, c.mass, c.representative))
        .collect();
    assert!(!retained.is_empty());
    let anchors = m.graph.anchors.len();
    m.remove_correspondences();
    assert_eq!(m.graph.anchors.len(), anchors);
    assert_eq!(m.contexts.unknown, 1.);
    assert!(
        m.contexts
            .active
            .iter()
            .flatten()
            .all(|c| c.responsibility == 0.)
    );
    for (i, bin) in [1, 2, 3].into_iter().enumerate() {
        let input = Input::new(10. + i as f64, bin);
        observe(&mut m, &if i == 2 { input.boundary() } else { input });
        assert_eq!(m.contexts.unknown, 1.);
        assert!(m.contexts.evidence.iter().all(|&e| e == 0.));
        assert!(
            m.contexts
                .active
                .iter()
                .flatten()
                .all(|c| c.responsibility == 0.)
        );
    }
    assert_eq!(
        retained,
        m.contexts
            .active
            .iter()
            .flatten()
            .map(|c| (c.id, c.mass, c.representative))
            .collect::<Vec<_>>()
    );
    assert_eq!(m.graph.anchors.len(), anchors + 3);
}

#[test]
fn pivot_needs_multiple_mappings_and_a_later_responsibility_change() {
    let mut m = model();
    m.contexts.active[0] = Some(context::Context {
        id: 1,
        version: 1,
        mass: 1.,
        responsibility: 0.75,
        last: 0.,
        representative: 1,
    });
    m.contexts.active[1] = Some(context::Context {
        id: 2,
        version: 1,
        mass: 1.,
        responsibility: 0.25,
        last: 0.,
        representative: 2,
    });
    m.relations.contexts(&m.contexts, 0.);
    m.contexts.evidence[0] = 0.4;
    m.contexts.evidence[1] = 0.3;
    m.contexts.evidence_ids[0] = 1;
    m.contexts.evidence_ids[1] = 2;
    m.relations.contexts(&m.contexts, 1.);
    assert!(m.relations.pivot.unwrap().transition_at.is_none());
    m.contexts.active[0].as_mut().unwrap().responsibility = 0.2;
    m.contexts.active[1].as_mut().unwrap().responsibility = 0.8;
    m.relations.contexts(&m.contexts, 1.01);
    assert_eq!(m.relations.pivot.unwrap().to, 2);
    m.relations.contexts(&m.contexts, 1.06);
    close(m.relations.pivot.unwrap().persistence, 0.05);
}
#[test]
fn production_capacity_is_declared_and_payload_tracks_sparse_occupancy() {
    let mut m = Model::new(Load {
        sample_rate: 48_000,
        hop: 512,
        sources: 2,
        coordinates: (0..690).map(|i| i as f64 / 100.).collect(),
        sparse_entries: 4,
    })
    .unwrap();
    let allocated = m.resources().allocated_bytes;
    observe(&mut m, &Input::new(0., 1));
    observe(&mut m, &Input::new(1., 2));
    let r = m.resources();
    assert_eq!(r.node_capacity, 94_651);
    assert_eq!(r.edge_capacity, 378_604);
    assert_eq!(r.anchor_capacity, 38_400);
    assert!(r.allocated_bytes - allocated >= r.live_payload_bytes);
    assert_eq!(r.retained_support_entries, 2);
    assert!(r.allocated_bytes < r.reserved_dense_graph_bytes);
    println!(
        "HV5 production Fs=48000 hop=512 sources=2 B=690 sparse_bound=4 allocated_bytes={} dense_support_bytes={} sparse_retained_entries={}",
        r.allocated_bytes, r.dense_support_bytes, r.retained_support_entries
    );
}

#[test]
fn identical_fractional_shapes_have_zero_variance_without_an_epsilon() {
    let mut m = model();
    for i in 0..30 {
        let mut input = Input::new(i as f64, 0);
        input.bins = vec![
            Bin {
                index: 0,
                mass: 0.1,
            },
            Bin {
                index: 1,
                mass: 0.9,
            },
        ];
        observe(&mut m, &input);
    }
    assert_eq!(m.graph.shape_variance(), 0.);
    m.graph.expire(620.);
    assert_eq!(m.graph.shape_variance(), 0.);
}
#[test]
fn prediction_expires_history_even_without_an_intervening_observation() {
    let mut m = model();
    observe(&mut m, &Input::new(0., 1));
    observe(&mut m, &Input::new(1., 2));
    let q = Input::new(1000., 1);
    assert_eq!(m.predict(&[q.observation()], 2).unwrap().unknown, 1.);
    assert!(m.graph.anchors.is_empty());
}

#[test]
fn open_chunk_updates_responsibilities_and_pivot_without_closing_an_episode() {
    let mut m = model();
    observe(&mut m, &Input::new(0., 1).boundary());
    observe(&mut m, &Input::new(1., 4).boundary());
    assert_eq!(m.contexts.active.iter().flatten().count(), 2);
    observe(&mut m, &Input::new(10., 2));
    assert!(!m.graph.chunks.back().unwrap().closed);
    assert_eq!(m.contexts.evidence.iter().filter(|v| **v > 0.).count(), 2);
    let pivot = m
        .relations
        .pivot
        .expect("shared state maps into two histories");
    assert_eq!(pivot.count, 2);
    assert!(pivot.transition_at.is_none());
    close(
        m.contexts
            .active
            .iter()
            .flatten()
            .map(|c| c.responsibility)
            .sum(),
        1.,
    );
    observe(&mut m, &Input::new(11., 1));
    assert_eq!(m.relations.pivot.unwrap().transition_at, Some(11.));
    assert_eq!(
        m.contexts
            .active
            .iter()
            .flatten()
            .map(|c| c.mass)
            .sum::<f64>(),
        2.
    );
}
