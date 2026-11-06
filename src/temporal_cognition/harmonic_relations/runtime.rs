//! Executed-record adapter and an independent, nonblocking prediction worker.
use super::{adapter::Adapter, consumer::Decision, *};
use crate::clock::Instant;
use crate::core::harmonic_support::{Geometry, Snapshot, preview::Observer};
#[cfg(feature = "native")]
use crate::handoff::bounded;
use crate::handoff::{Receiver, Sender, TrySendError};
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
};
#[cfg(feature = "native")]
use std::thread::JoinHandle;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Record {
    pub sample: u64,
    pub source: Option<Source>,
    pub flags: u8,
    pub trajectory: Option<Trajectory>,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Availability {
    pub decision_query: Option<u64>,
    pub decision_coordinate: Option<u64>,
    pub issued_at: Option<f64>,
    pub expires_at: Option<f64>,
    pub baseline_end: Option<u64>,
    pub baseline_coordinate: Option<u64>,
}
impl Availability {
    pub(crate) fn from_decision(decision: Option<&Decision>) -> Self {
        Self {
            decision_query: decision.map(|d| d.prediction.query_version),
            decision_coordinate: decision.map(|d| d.prediction.coordinate_version),
            issued_at: decision.map(|d| d.prediction.issued_at),
            expires_at: decision.map(|d| d.expires_at),
            ..Self::default()
        }
    }
    fn reasons(self, sample: u64, fs: f64) -> [bool; 6] {
        let now = sample as f64 / fs;
        [
            self.decision_query.is_none(),
            self.baseline_end.is_none(),
            self.decision_coordinate
                .zip(self.baseline_coordinate)
                .is_some_and(|(a, b)| a != b),
            self.baseline_end.is_some_and(|end| end > sample),
            self.issued_at.is_some_and(|issued| issued > now),
            self.expires_at.is_some_and(|expires| now > expires),
        ]
    }
}

#[derive(Debug, Default)]
struct FirstFloraMiss {
    state: AtomicU64,
    data: [AtomicU64; 12],
}
impl FirstFloraMiss {
    fn capture(
        &self,
        sample: u64,
        id: u64,
        expected: u64,
        received: u64,
        fs: f64,
        a: Availability,
    ) {
        if self
            .state
            .compare_exchange(0, 1, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return;
        }
        let valid = u64::from(a.decision_query.is_some())
            | (u64::from(a.decision_coordinate.is_some()) << 1)
            | (u64::from(a.issued_at.is_some()) << 2)
            | (u64::from(a.baseline_end.is_some()) << 3)
            | (u64::from(a.baseline_coordinate.is_some()) << 4)
            | (u64::from(a.expires_at.is_some()) << 5);
        for (slot, value) in self.data.iter().zip([
            sample,
            id,
            expected,
            received,
            valid,
            a.decision_query.unwrap_or(0),
            a.decision_coordinate.unwrap_or(0),
            a.issued_at.unwrap_or(0.).to_bits(),
            a.baseline_end.unwrap_or(0),
            a.baseline_coordinate.unwrap_or(0),
            a.expires_at.unwrap_or(0.).to_bits(),
            fs.to_bits(),
        ]) {
            slot.store(value, Ordering::Relaxed);
        }
        self.state.store(2, Ordering::Release);
    }
    fn snapshot(&self) -> serde_json::Value {
        match self.state.load(Ordering::Acquire) {
            0 => return serde_json::Value::Null,
            1 => return serde_json::json!({"status": "being_recorded"}),
            _ => {}
        }
        let data: [u64; 12] = std::array::from_fn(|i| self.data[i].load(Ordering::Relaxed));
        let optional =
            |bit: u32, index: usize| (data[4] & (1_u64 << bit) != 0).then_some(data[index]);
        serde_json::json!({"sample": data[0], "flora_id": data[1], "expected_query": data[2],
            "last_received_query": (data[3] != 0).then_some(data[3]),
            "decision_query": optional(0, 5), "decision_coordinate": optional(1, 6),
            "issued_at": optional(2, 7).map(f64::from_bits), "baseline_end": optional(3, 8),
            "baseline_coordinate": optional(4, 9), "expires_at": optional(5, 10).map(f64::from_bits),
            "sample_rate": f64::from_bits(data[11])})
    }
}

#[derive(Clone, Debug)]
pub(crate) struct Recorder {
    sender: Sender<Record>,
    pub overflow: Arc<AtomicU64>,
    pub prediction_misses: Arc<AtomicU64>,
    consumer_opportunities: Arc<[AtomicU64; 4]>,
    consumer_prediction_misses: Arc<[AtomicU64; 4]>,
    flora_unavailable_reasons: Arc<[AtomicU64; 6]>,
    first_flora_miss: Arc<FirstFloraMiss>,
    latest_received_query: Arc<AtomicU64>,
}
impl Recorder {
    #[cfg(feature = "native")]
    pub(crate) fn channel() -> (Self, Receiver<Record>) {
        let (sender, records) = bounded(UMAX);
        let recorder = Self {
            sender,
            overflow: Arc::new(AtomicU64::new(0)),
            prediction_misses: Arc::new(AtomicU64::new(0)),
            consumer_opportunities: Arc::new(std::array::from_fn(|_| AtomicU64::new(0))),
            consumer_prediction_misses: Arc::new(std::array::from_fn(|_| AtomicU64::new(0))),
            flora_unavailable_reasons: Arc::new(std::array::from_fn(|_| AtomicU64::new(0))),
            first_flora_miss: Arc::new(FirstFloraMiss::default()),
            latest_received_query: Arc::new(AtomicU64::new(0)),
        };
        (recorder, records)
    }
    pub(crate) fn opportunity_stats(&self, consumer: usize, count: u64, missing: bool) {
        self.consumer_opportunities[consumer].fetch_add(count, Ordering::Relaxed);
        if missing {
            self.prediction_misses.fetch_add(count, Ordering::Relaxed);
            self.consumer_prediction_misses[consumer].fetch_add(count, Ordering::Relaxed);
        }
    }
    pub(crate) fn flora_unavailable(
        &self,
        sample: u64,
        id: u64,
        expected_query: u64,
        fs: f64,
        count: u64,
        availability: Availability,
    ) {
        for (counter, failed) in self
            .flora_unavailable_reasons
            .iter()
            .zip(availability.reasons(sample, fs))
        {
            if failed {
                counter.fetch_add(count, Ordering::Relaxed);
            }
        }
        self.first_flora_miss.capture(
            sample,
            id,
            expected_query,
            self.latest_received_query.load(Ordering::Relaxed),
            fs,
            availability,
        );
    }
    pub(crate) fn record(&self, record: Record) {
        if self.sender.try_send(record).is_err() {
            self.overflow.fetch_add(1, Ordering::Relaxed);
        }
    }
}

pub(crate) fn record_opportunity(
    landscape: &crate::core::landscape::Landscape,
    source: Option<Source>,
    sample: u64,
) {
    if landscape.harmonic_relations_weight == 0. && landscape.harmonic_continuation_weight == 0. {
        return;
    }
    if let Some(recorder) = &landscape.harmonic_recorder {
        recorder.opportunity_stats(
            if source.is_some() { 0 } else { 3 },
            1,
            landscape.harmonic_query_version > 0 && landscape.harmonic_relations.is_none(),
        );
        recorder.record(Record {
            sample,
            source,
            flags: OPPORTUNITY,
            trajectory: None,
        });
    }
}

enum Input {
    Record(Record),
    Snapshot(Arc<Snapshot>),
    Boundary(u64, Instant),
    Finish,
}

struct Publication {
    decision: Arc<Decision>,
    ready: Instant,
    boundary: u64,
}

#[derive(Default)]
struct WorkMetrics {
    timestamp_groups: AtomicU64,
    max_timestamp_groups: AtomicU64,
    max_group_ns: AtomicU64,
    max_complete_hop_ns: AtomicU64,
    max_finish_ns: AtomicU64,
    completed_hops: AtomicU64,
    submitted_hops: AtomicU64,
    total_complete_hop_ns: AtomicU64,
    shutdown_join_ns: AtomicU64,
    max_pending_inputs: AtomicU64,
    max_pending_records: AtomicU64,
    max_boundary_wait_ns: AtomicU64,
    publications: AtomicU64,
    publications_replaced: AtomicU64,
    publications_received: AtomicU64,
    stale_publications_received: AtomicU64,
    max_request_to_publication_ns: AtomicU64,
    max_publication_to_poll_ns: AtomicU64,
    max_publication_age_samples: AtomicU64,
    max_query_lag_samples: AtomicU64,
    unavailable_evaluations: AtomicU64,
    unavailable_reasons: [AtomicU64; 6],
}

pub(crate) struct Runtime {
    pub recorder: Recorder,
    records: Receiver<Record>,
    ordered: Vec<(u64, Record)>,
    next_record_order: u64,
    collected_end: u64,
    input: Option<Sender<Input>>,
    output: Receiver<Publication>,
    metrics: Arc<WorkMetrics>,
    #[cfg(feature = "native")]
    thread: Option<JoinHandle<()>>,
    expected_query: u64,
    requested_query: Arc<AtomicU64>,
    stale_queries_skipped: Arc<AtomicU64>,
    latest: Option<Arc<Decision>>,
    last_boundary: u64,
    pub input_misses: Arc<AtomicU64>,
    pub observations: Arc<AtomicU64>,
    pub anchors: Arc<AtomicU64>,
    pub retained_support_entries: Arc<AtomicU64>,
    pub allocated_bytes: usize,
    startup_resources: Resources,
    final_resources: Arc<std::sync::Mutex<Option<Resources>>>,
    pub sources: usize,
    sample_rate: u32,
    hop: usize,
    pub startup_sec: f64,
    pub max_query_ns: Arc<AtomicU64>,
    pub max_batch_ns: Arc<AtomicU64>,
    pub max_observed_sources: Arc<AtomicU64>,
}

impl Runtime {
    #[cfg(feature = "native")]
    pub(crate) fn new(
        geometry: Arc<Geometry>,
        sample_rate: u32,
        hop: usize,
        sources: usize,
    ) -> Self {
        let startup = crate::clock::Instant::now();
        let (recorder, records) = Recorder::channel();
        let (input, receiver) = bounded(UMAX);
        let (publication, output) = bounded(1);
        let stale = output.clone();
        let metrics = Arc::new(WorkMetrics::default());
        let work_metrics = metrics.clone();
        let input_misses = Arc::new(AtomicU64::new(0));
        let failures = input_misses.clone();
        let requested_query = Arc::new(AtomicU64::new(0));
        let requested = requested_query.clone();
        let stale_queries_skipped = Arc::new(AtomicU64::new(0));
        let skipped = stale_queries_skipped.clone();
        let observations = Arc::new(AtomicU64::new(0));
        let anchors = Arc::new(AtomicU64::new(0));
        let retained_support_entries = Arc::new(AtomicU64::new(0));
        let observed = observations.clone();
        let anchored = anchors.clone();
        let retained = retained_support_entries.clone();
        let max_query_ns = Arc::new(AtomicU64::new(0));
        let max_batch_ns = Arc::new(AtomicU64::new(0));
        let query_cost = max_query_ns.clone();
        let batch_cost = max_batch_ns.clone();
        let max_observed_sources = Arc::new(AtomicU64::new(0));
        let observed_sources = max_observed_sources.clone();
        let mut snapshot = Arc::new(Snapshot::new(geometry));
        let coordinates = (0..snapshot.p.len())
            .map(|i| {
                f64::from(snapshot.geometry.root_start_log2)
                    + i as f64 * f64::from(snapshot.geometry.space.step())
            })
            .collect();
        let mut model = Model::new(Load {
            sample_rate,
            hop,
            sources,
            coordinates,
            sparse_entries: snapshot.p.len(),
        })
        .expect("validated harmonic relations load");
        let startup_resources = model.resources();
        let allocated_bytes = startup_resources.allocated_bytes;
        let final_resources = Arc::new(std::sync::Mutex::new(None));
        let finished_resources = final_resources.clone();
        let mut adapter = Adapter::new(snapshot.p.len(), sources);
        let mut pending = Vec::with_capacity(UMAX);
        let mut tones: Vec<Trajectory> = Vec::with_capacity(UMAX);
        let mut trajectories: Vec<Trajectory> = Vec::with_capacity(sources);
        let mut events: Vec<Event> = Vec::with_capacity(sources + 1);
        let thread = std::thread::Builder::new()
            .name("harmonic-relations".into())
            .spawn(move || {
                let mut boundary = 0;
                let mut query = 0;
                let mut published = 0;
                let mut published_model = 0;
                // Exclude blocking receive time; include every message's actual processing.
                let mut message_work_ns = 0;
                while let Ok(message) = receiver.recv() {
                    let dispatch_start = Instant::now();
                    work_metrics
                        .max_pending_inputs
                        .fetch_max(receiver.len() as u64, Ordering::Relaxed);
                    let (next, finish, submitted) = match message {
                        Input::Record(record) => {
                            if pending.len() < UMAX {
                                pending.push(record);
                                work_metrics
                                    .max_pending_records
                                    .fetch_max(pending.len() as u64, Ordering::Relaxed);
                            } else {
                                failures.fetch_add(1, Ordering::Relaxed);
                            }
                            message_work_ns += dispatch_start.elapsed().as_nanos() as u64;
                            continue;
                        }
                        Input::Snapshot(support) => {
                            if support.end_sample <= boundary {
                                snapshot = support;
                            } else {
                                failures.fetch_add(1, Ordering::Relaxed);
                            }
                            message_work_ns += dispatch_start.elapsed().as_nanos() as u64;
                            continue;
                        }
                        Input::Boundary(sample, submitted) => (sample, false, submitted),
                        Input::Finish => (boundary, true, dispatch_start),
                    };
                    let batch_start = Instant::now();
                    if !finish {
                        work_metrics.max_boundary_wait_ns.fetch_max(
                            batch_start.duration_since(submitted).as_nanos() as u64,
                            Ordering::Relaxed,
                        );
                    }
                    let mut groups = 0;
                    let mut cursor = 0;
                    let mut at = boundary;
                    loop {
                        let group_start = Instant::now();
                        events.clear();
                        while cursor < pending.len() && pending[cursor].sample <= at {
                            let record = pending[cursor];
                            cursor += 1;
                            if record.sample < boundary {
                                failures.fetch_add(1, Ordering::Relaxed);
                                continue;
                            }
                            if record.flags != 0 {
                                if let Some(event) =
                                    events.iter_mut().find(|e| e.source == record.source)
                                {
                                    event.flags |= record.flags;
                                } else if events.len() < events.capacity() {
                                    events.push(Event {
                                        source: record.source,
                                        flags: record.flags,
                                    });
                                } else {
                                    failures.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                            if let Some(t) = record.trajectory {
                                if let Some(old) = tones.iter_mut().find(|old| {
                                    old.source == t.source && old.articulation == t.articulation
                                }) {
                                    *old = t;
                                } else if tones.len() < UMAX {
                                    tones.push(t);
                                } else {
                                    failures.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                        trajectories.clear();
                        for tone in &tones {
                            if let Some(existing) =
                                trajectories.iter_mut().find(|t| t.source == tone.source)
                            {
                                if tone.sounding && existing.sounding {
                                    if existing.pitch != tone.pitch {
                                        existing.pitch.valid = false;
                                    }
                                    // Simultaneous tone controls do not certify the mixed amplitude.
                                    existing.amplitude.valid = false;
                                } else if tone.sounding && !existing.sounding {
                                    *existing = *tone;
                                }
                                existing.sounding |= tone.sounding;
                            } else if trajectories.len() < sources {
                                trajectories.push(*tone);
                            } else {
                                failures.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        if !events.is_empty() {
                            query = at.saturating_add(1);
                        }
                        observed_sources.fetch_max(trajectories.len() as u64, Ordering::Relaxed);
                        if let Ok(observation) = adapter.prepare(
                            &snapshot,
                            Observer::Mixed,
                            at,
                            sample_rate,
                            at == boundary,
                            &trajectories,
                            &events,
                            snapshot.numerical_errors(),
                        ) {
                            if let Err(error) = model.observe(observation) {
                                if failures.fetch_add(1, Ordering::Relaxed) == 0 {
                                    tracing::warn!(
                                        error,
                                        sample = at,
                                        "harmonic observation rejected"
                                    );
                                }
                            } else {
                                observed.fetch_add(1, Ordering::Relaxed);
                                anchored
                                    .fetch_add(u64::from(!events.is_empty()), Ordering::Relaxed);
                            }
                        } else {
                            failures.fetch_add(1, Ordering::Relaxed);
                        }
                        tones.retain(|t| t.sounding);
                        groups += 1;
                        work_metrics
                            .max_group_ns
                            .fetch_max(group_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                        let Some(record) = pending.get(cursor) else {
                            break;
                        };
                        if record.sample >= next {
                            break;
                        }
                        at = record.sample;
                    }
                    pending.drain(..cursor);
                    boundary = next;
                    let needs_prediction =
                        (query != published || model.version != published_model) && query != 0;
                    if needs_prediction && query != requested.load(Ordering::Acquire) {
                        skipped.fetch_add(1, Ordering::Relaxed);
                    } else if needs_prediction {
                        let query_start = crate::clock::Instant::now();
                        match model.predict_latest(next as f64 / f64::from(sample_rate), query) {
                            Ok(prediction) => {
                                let mut decision = Decision::new(
                                    prediction.clone(),
                                    prediction.variances,
                                    prediction.shape_variance,
                                );
                                decision.expires_at = model
                                    .graph
                                    .anchors
                                    .front()
                                    .and_then(|anchor| model.graph.node(anchor.node))
                                    .map_or(f64::INFINITY, |node| node.start + TMAX);
                                let decision = Arc::new(decision);
                                let ready = Instant::now();
                                let mut value = Publication {
                                    decision,
                                    ready,
                                    boundary: next,
                                };
                                loop {
                                    match publication.try_send(value) {
                                        Ok(()) => {
                                            work_metrics
                                                .publications
                                                .fetch_add(1, Ordering::Relaxed);
                                            work_metrics.max_request_to_publication_ns.fetch_max(
                                                submitted.elapsed().as_nanos() as u64,
                                                Ordering::Relaxed,
                                            );
                                            break;
                                        }
                                        Err(TrySendError::Full(returned)) => {
                                            value = returned;
                                            if stale.try_recv().is_ok() {
                                                work_metrics
                                                    .publications_replaced
                                                    .fetch_add(1, Ordering::Relaxed);
                                            }
                                        }
                                        Err(TrySendError::Disconnected(_)) => return,
                                    }
                                }
                                published = query;
                                published_model = model.version;
                            }
                            Err(error) => {
                                if failures.fetch_add(1, Ordering::Relaxed) == 0 {
                                    tracing::warn!(error, query, "harmonic prediction rejected");
                                }
                            }
                        }
                        query_cost
                            .fetch_max(query_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                    if finish {
                        let resources = model.resources();
                        retained
                            .store(resources.retained_support_entries as u64, Ordering::Relaxed);
                        *finished_resources
                            .lock()
                            .expect("harmonic resource snapshot") = Some(resources);
                    }
                    let elapsed = batch_start.elapsed().as_nanos() as u64;
                    batch_cost.fetch_max(elapsed, Ordering::Relaxed);
                    let complete = message_work_ns + dispatch_start.elapsed().as_nanos() as u64;
                    message_work_ns = 0;
                    work_metrics
                        .timestamp_groups
                        .fetch_add(groups, Ordering::Relaxed);
                    if finish {
                        work_metrics
                            .max_finish_ns
                            .fetch_max(complete, Ordering::Relaxed);
                    } else {
                        work_metrics.completed_hops.fetch_add(1, Ordering::Relaxed);
                        work_metrics
                            .total_complete_hop_ns
                            .fetch_add(complete, Ordering::Relaxed);
                        work_metrics
                            .max_complete_hop_ns
                            .fetch_max(complete, Ordering::Relaxed);
                        work_metrics
                            .max_timestamp_groups
                            .fetch_max(groups, Ordering::Relaxed);
                    }
                    if finish {
                        break;
                    }
                }
            })
            .expect("harmonic relation worker spawn");
        Self {
            recorder,
            records,
            ordered: Vec::with_capacity(UMAX),
            next_record_order: 0,
            collected_end: 0,
            input: Some(input),
            output,
            metrics,
            thread: Some(thread),
            expected_query: 0,
            requested_query,
            stale_queries_skipped,
            latest: None,
            last_boundary: 0,
            input_misses,
            observations,
            anchors,
            retained_support_entries,
            allocated_bytes,
            startup_resources,
            final_resources,
            sources,
            sample_rate,
            hop,
            startup_sec: startup.elapsed().as_secs_f64(),
            max_query_ns,
            max_batch_ns,
            max_observed_sources,
        }
    }

    #[cfg(feature = "native")]
    pub(crate) fn finish(&mut self) {
        if self.thread.is_none() {
            return;
        }
        let join_start = Instant::now();
        self.flush_records(self.collected_end, true);
        if let Some(input) = self.input.take() {
            let _ = input.send(Input::Finish);
        }
        #[cfg(feature = "native")]
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            self.input_misses.fetch_add(1, Ordering::Relaxed);
        }
        self.metrics
            .shutdown_join_ns
            .store(join_start.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    #[cfg(feature = "native")]
    pub(crate) fn summary(&self) -> serde_json::Value {
        let memory = |r: Resources| {
            serde_json::json!({
                "model_allocated_bytes": r.allocated_bytes,
                "live_payload_bytes": r.live_payload_bytes,
                "index_allocated_bytes": r.index_allocated_bytes,
                "reserved_dense_graph_bytes": r.reserved_dense_graph_bytes,
                "dense_graph_categories": {
                    "intervals": r.dense_graph_categories[0], "anchors": r.dense_graph_categories[1],
                    "chunks": r.dense_graph_categories[2], "trajectories": r.dense_graph_categories[3],
                    "events": r.dense_graph_categories[4], "support": r.dense_graph_categories[5],
                    "motions": r.dense_graph_categories[6], "shape_and_coordinates": r.dense_graph_categories[7],
                },
                "retained_support_entries": r.retained_support_entries,
            })
        };
        let final_resources = *self
            .final_resources
            .lock()
            .expect("harmonic resource snapshot");
        let flora_reasons = [
            "missing_prediction",
            "missing_baseline",
            "coordinate_mismatch",
            "baseline_from_future",
            "prediction_from_future",
            "prediction_expired",
        ]
        .into_iter()
        .enumerate()
        .map(|(i, name)| {
            (
                name.to_owned(),
                serde_json::json!(
                    self.recorder.flora_unavailable_reasons[i].load(Ordering::Relaxed)
                ),
            )
        })
        .collect::<serde_json::Map<_, _>>();
        let runtime_reasons = [
            "no_publication",
            "query_mismatch",
            "coordinate_mismatch",
            "prediction_from_future",
            "prediction_expired",
            "input_loss",
        ]
        .into_iter()
        .enumerate()
        .map(|(i, name)| {
            (
                name.to_owned(),
                serde_json::json!(self.metrics.unavailable_reasons[i].load(Ordering::Relaxed)),
            )
        })
        .collect::<serde_json::Map<_, _>>();
        let consumers = ["voice_pitch", "flora_lane", "flora_population", "spawn"].into_iter()
            .enumerate().map(|(i, name)| (name.to_owned(), serde_json::json!({
                "opportunities": self.recorder.consumer_opportunities[i].load(Ordering::Relaxed),
                "prediction_misses": self.recorder.consumer_prediction_misses[i].load(Ordering::Relaxed),
            }))).collect::<serde_json::Map<_, _>>();
        let unavailability_diagnostics = serde_json::json!({
            "reason_counts_overlap": true,
            "flora_reason_unit": "consumer opportunities: population plus current lanes",
            "flora_reasons": flora_reasons,
            "first_flora_miss": self.recorder.first_flora_miss.snapshot(),
            "runtime_reason_unit": "advance evaluations with positive weight and nonzero query",
            "runtime_unavailable_evaluations": self.metrics.unavailable_evaluations.load(Ordering::Relaxed),
            "runtime_reasons": runtime_reasons,
        });
        let mut summary = serde_json::json!({
            "consumers": consumers,
            "memory": {
                "startup": memory(self.startup_resources),
                "after_background_join": final_resources.map(memory),
                "audio_end_live_snapshot": null,
                "audio_end_live_snapshot_status": "not_sampled_without_blocking_background",
            },
            "sample_rate": self.sample_rate, "hop_samples": self.hop,
            "hop_deadline_ms": self.hop as f64 / f64::from(self.sample_rate) * 1000.,
            "submitted_hops": self.metrics.submitted_hops.load(Ordering::Relaxed),
            "completed_hops": self.metrics.completed_hops.load(Ordering::Relaxed),
            "pending_hops": self.metrics.submitted_hops.load(Ordering::Relaxed)
                .saturating_sub(self.metrics.completed_hops.load(Ordering::Relaxed)),
            "total_complete_hop_ms": self.metrics.total_complete_hop_ns.load(Ordering::Relaxed) as f64 / 1e6,
            "shutdown_join_ms": self.metrics.shutdown_join_ns.load(Ordering::Relaxed) as f64 / 1e6,
            "timestamp_groups": self.metrics.timestamp_groups.load(Ordering::Relaxed),
            "max_timestamp_groups_per_hop": self.metrics.max_timestamp_groups.load(Ordering::Relaxed),
            "max_timestamp_group_ms": self.metrics.max_group_ns.load(Ordering::Relaxed) as f64 / 1e6,
            "max_complete_hop_ms": self.metrics.max_complete_hop_ns.load(Ordering::Relaxed) as f64 / 1e6,
            "max_finish_batch_ms": self.metrics.max_finish_ns.load(Ordering::Relaxed) as f64 / 1e6,
            "max_pending_input_messages": self.metrics.max_pending_inputs.load(Ordering::Relaxed),
            "max_pending_records": self.metrics.max_pending_records.load(Ordering::Relaxed),
            "max_boundary_queue_wait_ms": self.metrics.max_boundary_wait_ns.load(Ordering::Relaxed) as f64 / 1e6,
            "publications": self.metrics.publications.load(Ordering::Relaxed),
            "publications_replaced": self.metrics.publications_replaced.load(Ordering::Relaxed),
            "publications_received": self.metrics.publications_received.load(Ordering::Relaxed),
            "stale_publications_received": self.metrics.stale_publications_received.load(Ordering::Relaxed),
            "max_request_to_publication_ms": self.metrics.max_request_to_publication_ns.load(Ordering::Relaxed) as f64 / 1e6,
            "max_publication_to_poll_ms": self.metrics.max_publication_to_poll_ns.load(Ordering::Relaxed) as f64 / 1e6,
            "max_publication_age_samples": self.metrics.max_publication_age_samples.load(Ordering::Relaxed),
            "max_query_lag_samples": self.metrics.max_query_lag_samples.load(Ordering::Relaxed),
            "background_joined": self.thread.is_none(),
            "pending_input_messages": self.input.as_ref().map_or(0, Sender::len),
            "sources": self.sources, "startup_sec": self.startup_sec, "allocated_model_bytes": self.allocated_bytes,
            "max_observed_sources": self.max_observed_sources.load(Ordering::Relaxed),
            "observations": self.observations.load(Ordering::Relaxed), "anchors": self.anchors.load(Ordering::Relaxed),
            "retained_support_entries": self.retained_support_entries.load(Ordering::Relaxed),
            "prediction_misses": self.recorder.prediction_misses.load(Ordering::Relaxed),
            "stale_queries_skipped": self.stale_queries_skipped.load(Ordering::Relaxed),
            "record_overflow": self.recorder.overflow.load(Ordering::Relaxed),
            "input_misses": self.input_misses.load(Ordering::Relaxed),
            "max_prediction_ms": self.max_query_ns.load(Ordering::Relaxed) as f64 / 1e6,
            "max_observation_batch_ms": self.max_batch_ns.load(Ordering::Relaxed) as f64 / 1e6,
        });
        summary["unavailability_diagnostics"] = unavailability_diagnostics;
        summary["retained_future_records"] = serde_json::json!(self.ordered.len());
        summary
    }

    fn flush_records(&mut self, end: u64, closing: bool) {
        for record in self.records.try_iter() {
            let order = self.next_record_order;
            self.next_record_order += 1;
            if self.ordered.len() < UMAX {
                self.ordered.push((order, record));
            } else {
                self.input_misses.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.ordered
            .sort_unstable_by_key(|(order, record)| (record.sample, *order));
        let eligible = self.ordered.partition_point(|(_, record)| {
            record.sample < end || (closing && record.sample == end)
        });
        if let Some((_, record)) = self.ordered[..eligible]
            .iter()
            .rev()
            .find(|(_, record)| record.flags != 0)
        {
            self.expected_query = record.sample.saturating_add(1);
        }
        // Retained boundary records must advance both sides' query in the same batch.
        self.requested_query
            .store(self.expected_query, Ordering::Release);
        let input = self.input.as_ref().expect("running harmonic worker");
        for (_, record) in self.ordered.drain(..eligible) {
            let sent = if closing {
                input.send(Input::Record(record)).is_ok()
            } else {
                input.try_send(Input::Record(record)).is_ok()
            };
            if !sent {
                self.input_misses.fetch_add(1, Ordering::Relaxed);
            }
            self.metrics
                .max_pending_inputs
                .fetch_max(input.len() as u64, Ordering::Relaxed);
        }
    }

    pub(crate) fn collect(&mut self, end: u64) {
        self.flush_records(end, false);
        let input = self.input.as_ref().expect("running harmonic worker");
        if input
            .try_send(Input::Boundary(end, Instant::now()))
            .is_err()
        {
            self.input_misses.fetch_add(1, Ordering::Relaxed);
        } else {
            self.collected_end = end;
            self.metrics.submitted_hops.fetch_add(1, Ordering::Relaxed);
        }
        self.metrics
            .max_pending_inputs
            .fetch_max(input.len() as u64, Ordering::Relaxed);
    }

    pub(crate) fn advance(
        &mut self,
        support: &Arc<Snapshot>,
        landscape: &mut crate::core::landscape::Landscape,
    ) {
        if support.end_sample > self.last_boundary {
            let input = self.input.as_ref().expect("running harmonic worker");
            if input.try_send(Input::Snapshot(support.clone())).is_err() {
                self.input_misses.fetch_add(1, Ordering::Relaxed);
            }
            self.metrics
                .max_pending_inputs
                .fetch_max(input.len() as u64, Ordering::Relaxed);
            self.last_boundary = support.end_sample;
        }
        for publication in self.output.try_iter() {
            self.metrics
                .publications_received
                .fetch_add(1, Ordering::Relaxed);
            self.metrics.max_publication_to_poll_ns.fetch_max(
                publication.ready.elapsed().as_nanos() as u64,
                Ordering::Relaxed,
            );
            self.metrics.max_publication_age_samples.fetch_max(
                landscape
                    .harmonic_now_sample
                    .saturating_sub(publication.boundary),
                Ordering::Relaxed,
            );
            let lag = self
                .expected_query
                .saturating_sub(publication.decision.prediction.query_version);
            self.metrics
                .max_query_lag_samples
                .fetch_max(lag, Ordering::Relaxed);
            if publication.decision.prediction.query_version != self.expected_query {
                self.metrics
                    .stale_publications_received
                    .fetch_add(1, Ordering::Relaxed);
            }
            self.recorder.latest_received_query.store(
                publication.decision.prediction.query_version,
                Ordering::Relaxed,
            );
            self.latest = Some(publication.decision);
        }
        if (landscape.harmonic_relations_weight != 0.
            || landscape.harmonic_continuation_weight != 0.)
            && self.expected_query > 0
        {
            let now = landscape.harmonic_now_sample as f64 / f64::from(self.sample_rate);
            let reasons = [
                self.latest.is_none(),
                self.latest
                    .as_ref()
                    .is_some_and(|d| d.prediction.query_version != self.expected_query),
                self.latest.as_ref().is_some_and(|d| {
                    d.prediction.coordinate_version
                        != support.baseline(Observer::Mixed).coordinate_version
                }),
                self.latest
                    .as_ref()
                    .is_some_and(|d| d.prediction.issued_at > now),
                self.latest.as_ref().is_some_and(|d| now > d.expires_at),
                self.input_misses.load(Ordering::Relaxed) != 0
                    || self.recorder.overflow.load(Ordering::Relaxed) != 0,
            ];
            if reasons.iter().any(|failed| *failed) {
                self.metrics
                    .unavailable_evaluations
                    .fetch_add(1, Ordering::Relaxed);
                for (counter, failed) in self.metrics.unavailable_reasons.iter().zip(reasons) {
                    if failed {
                        counter.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
        landscape.harmonic_relations = self
            .latest
            .as_ref()
            .filter(|decision| {
                decision.prediction.query_version == self.expected_query
                    && decision.prediction.coordinate_version
                        == support.baseline(Observer::Mixed).coordinate_version
                    && decision.prediction.issued_at
                        <= landscape.harmonic_now_sample as f64 / f64::from(self.sample_rate)
                    && landscape.harmonic_now_sample as f64 / f64::from(self.sample_rate)
                        <= decision.expires_at
                    && self.input_misses.load(Ordering::Relaxed) == 0
                    && self.recorder.overflow.load(Ordering::Relaxed) == 0
            })
            .cloned();
        // The worker rejects publications from an older structural query before scoring.
        landscape.harmonic_query_version = self.expected_query;
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        #[cfg(feature = "native")]
        self.finish();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{
        harmonicity_kernel::HarmonicityKernel, landscape::Landscape, log2space::Log2Space,
    };

    #[test]
    fn collector_retains_boundary_records_merges_arrivals_and_flushes_final_boundary() {
        let space = Log2Space::new(220., 880., 12);
        let geometry = Geometry::new(&space, &HarmonicityKernel::new(&space, Default::default()));
        let mut runtime = Runtime::new(geometry.clone(), 48000, 512, 1);
        let source = Source {
            kind: 0,
            id: 7,
            generation: 1,
        };
        let record = |sample, flags, hz: f64| Record {
            sample,
            source: Some(source),
            flags,
            trajectory: Some(Trajectory {
                source,
                pitch: Value::exact(hz.log2()),
                amplitude: Value::exact(0.1),
                sounding: true,
                ..Default::default()
            }),
        };
        runtime.recorder.record(record(0, ONSET, 440.));
        runtime.recorder.record(record(512, OPPORTUNITY, 330.));
        runtime.recorder.record(record(1024, OPPORTUNITY, 550.));
        runtime.collect(512);
        assert_eq!(runtime.expected_query, 1);
        assert_eq!(
            runtime
                .ordered
                .iter()
                .map(|(_, r)| r.sample)
                .collect::<Vec<_>>(),
            [512, 1024]
        );
        let retained_order = runtime.ordered[1].0;
        runtime.recorder.record(record(1024, 0, 660.));
        runtime.recorder.record(record(512, 0, 880.));
        runtime.recorder.record(record(1536, ONSET, 990.));
        runtime.collect(1024);
        assert_eq!(runtime.expected_query, 513);
        assert_eq!(
            runtime
                .ordered
                .iter()
                .map(|(_, r)| r.sample)
                .collect::<Vec<_>>(),
            [1024, 1024, 1536]
        );
        assert_eq!(runtime.ordered[0].0, retained_order);
        assert!(runtime.ordered[0].0 < runtime.ordered[1].0);
        assert_eq!(
            runtime.ordered[0].1.trajectory.unwrap().pitch.value,
            550f64.log2()
        );
        assert_eq!(
            runtime.ordered[1].1.trajectory.unwrap().pitch.value,
            660f64.log2()
        );
        let support = Arc::new(Snapshot::new(geometry));
        let mut landscape = Landscape::new(space);
        landscape.harmonic_now_sample = 1024;
        let deadline = Instant::now() + std::time::Duration::from_secs(5);
        loop {
            runtime.advance(&support, &mut landscape);
            if landscape.harmonic_relations.is_some() {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "matching boundary query publication"
            );
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert_eq!(
            landscape
                .harmonic_relations
                .as_ref()
                .unwrap()
                .prediction
                .query_version,
            513
        );
        assert_eq!(runtime.observations.load(Ordering::Relaxed), 2);
        runtime.finish();
        assert_eq!(runtime.expected_query, 1025);
        let publication = runtime
            .output
            .try_recv()
            .expect("final executed boundary publication");
        assert_eq!(publication.decision.prediction.query_version, 1025);
        assert_eq!(publication.decision.prediction.issued_at, 1024. / 48000.);
        assert_eq!(runtime.observations.load(Ordering::Relaxed), 3);
        assert_eq!(runtime.anchors.load(Ordering::Relaxed), 3);
        assert_eq!(runtime.input_misses.load(Ordering::Relaxed), 0);
        assert_eq!(runtime.ordered.len(), 1);
        assert_eq!(runtime.ordered[0].1.sample, 1536);
    }

    #[test]
    fn executed_batches_merge_hop_boundary_events_and_publish_matching_query() {
        let space = Log2Space::new(220., 880., 12);
        let geometry = Geometry::new(&space, &HarmonicityKernel::new(&space, Default::default()));
        let mut runtime = Runtime::new(geometry.clone(), 48000, 512, 1);
        let source = Source {
            kind: 0,
            id: 7,
            generation: 1,
        };
        let trajectory = Trajectory {
            source,
            pitch: Value::exact(440f64.log2()),
            amplitude: Value::exact(0.1),
            sounding: true,
            ..Default::default()
        };
        runtime.recorder.record(Record {
            sample: 0,
            source: Some(source),
            flags: ONSET,
            trajectory: Some(trajectory),
        });
        runtime.recorder.record(Record {
            sample: 0,
            source: Some(source),
            flags: OPPORTUNITY,
            trajectory: None,
        });
        runtime.collect(512);
        let mut snapshot = Snapshot::new(geometry.clone());
        snapshot.observe_projection(&vec![0.; space.n_bins()], None, 1e-6, 512);
        let mut landscape = Landscape::new(space.clone());
        runtime.advance(&Arc::new(snapshot), &mut landscape);
        runtime.recorder.record(Record {
            sample: 512,
            source: Some(source),
            flags: OFFSET,
            trajectory: Some(Trajectory {
                sounding: false,
                ..trajectory
            }),
        });
        runtime.recorder.record(Record {
            sample: 512,
            source: Some(source),
            flags: OPPORTUNITY,
            trajectory: None,
        });
        runtime.collect(1024);
        let deadline = crate::clock::Instant::now() + std::time::Duration::from_secs(5);
        let prediction = loop {
            let remaining = deadline.saturating_duration_since(crate::clock::Instant::now());
            let decision = runtime
                .output
                .recv_timeout(remaining)
                .expect("bounded test prediction")
                .decision;
            if decision.prediction.query_version == 513 {
                break decision;
            }
        };
        assert_eq!(prediction.prediction.query_version, 513);
        assert_eq!(runtime.observations.load(Ordering::Relaxed), 2);
        assert_eq!(runtime.anchors.load(Ordering::Relaxed), 2);
        assert_eq!(runtime.input_misses.load(Ordering::Relaxed), 0);
        assert_eq!(runtime.recorder.overflow.load(Ordering::Relaxed), 0);
        runtime.latest = Some(prediction);
        runtime.expected_query = 514;
        let support = Arc::new(Snapshot::new(geometry));
        runtime.advance(&support, &mut landscape);
        assert!(landscape.harmonic_relations.is_none());
        runtime.expected_query = 513;
        landscape.harmonic_now_sample = 1024;
        runtime.advance(&support, &mut landscape);
        assert!(landscape.harmonic_relations.is_some());
        landscape.harmonic_now_sample = ((TMAX + 1.) * 48000.) as u64;
        runtime.advance(&support, &mut landscape);
        assert!(landscape.harmonic_relations.is_none());
        let before = runtime.observations.load(Ordering::Relaxed);
        runtime.requested_query.store(2049, Ordering::Release);
        runtime
            .input
            .as_ref()
            .unwrap()
            .send(Input::Record(Record {
                sample: 1024,
                source: Some(source),
                flags: OPPORTUNITY,
                trajectory: None,
            }))
            .unwrap();
        runtime
            .input
            .as_ref()
            .unwrap()
            .send(Input::Boundary(1536, Instant::now()))
            .unwrap();
        runtime.collected_end = 1536;
        runtime
            .metrics
            .submitted_hops
            .fetch_add(1, Ordering::Relaxed);
        runtime.finish();
        assert!(runtime.stale_queries_skipped.load(Ordering::Relaxed) >= 1);
        assert!(runtime.observations.load(Ordering::Relaxed) > before);
        assert_eq!(runtime.input_misses.load(Ordering::Relaxed), 0);
        let measured = runtime.summary();
        assert_eq!(measured["submitted_hops"], 3);
        assert_eq!(measured["completed_hops"], 3);
        assert_eq!(measured["pending_hops"], 0);
        assert_eq!(measured["timestamp_groups"], 4);
        assert_eq!(measured["background_joined"], true);
        landscape.harmonic_relations_weight = 1.;
        landscape.harmonic_query_version = 1;
        landscape.harmonic_relations = None;
        landscape.harmonic_recorder = Some(runtime.recorder.clone());
        record_opportunity(&landscape, Some(source), 2048);
        record_opportunity(&landscape, None, 2048);
        let mut flora = crate::life::flora::Flora::new(
            9,
            Default::default(),
            (220., 880.),
            crate::scenario::FloraSeeding::Uniform,
            1,
            &landscape,
            48000.,
            512,
        );
        flora.harmonic_recorder = Some(runtime.recorder.clone());
        flora.advance(&landscape);
        flora.render_add(&mut [0.; 512], &mut [0.; 512]);
        let counters = &runtime.recorder;
        for consumer in [0, 2, 3] {
            assert_eq!(
                counters.consumer_opportunities[consumer].load(Ordering::Relaxed),
                1
            );
            assert_eq!(
                counters.consumer_prediction_misses[consumer].load(Ordering::Relaxed),
                1
            );
        }
        let diagnostics = runtime.summary()["unavailability_diagnostics"].clone();
        assert_eq!(diagnostics["flora_reasons"]["missing_prediction"], 1);
        assert_eq!(diagnostics["flora_reasons"]["missing_baseline"], 1);
        let first = diagnostics["first_flora_miss"].clone();
        assert_eq!(first["flora_id"], 9);
        assert_eq!(first["expected_query"], 1);
        assert!(first["baseline_end"].is_null());
        assert!(first["decision_query"].is_null());
        // Earlier advance calls may already have received a publication.
        let received = counters.latest_received_query.load(Ordering::Relaxed);
        assert_eq!(
            first["last_received_query"],
            serde_json::json!((received != 0).then_some(received))
        );
        counters
            .first_flora_miss
            .capture(999, 99, 999, 999, 48000., Availability::default());
        assert_eq!(counters.first_flora_miss.snapshot(), first);
        landscape.harmonic_relations_weight = 0.;
        flora.advance(&landscape);
        flora.render_add(&mut [0.; 512], &mut [0.; 512]);
        assert_eq!(
            counters.consumer_opportunities[2].load(Ordering::Relaxed),
            1
        );
        assert_eq!(
            counters.consumer_prediction_misses[2].load(Ordering::Relaxed),
            1
        );
    }
}
