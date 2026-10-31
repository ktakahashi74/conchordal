//! Bounded score transport. Credit includes items being applied by the consumer.
use super::{ScriptError, ScriptWarning};
use crate::core::timebase::{Tick, Timebase};
use crate::scenario::{Action, Scenario, SceneMarker, TimedEvent};
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::thread::{self, JoinHandle, Thread};
use std::time::{Duration, Instant};

pub(crate) const LEAD_HOPS: u64 = 64;
// Calibrated by score_patch_capacity_measurement in a release build.
pub(crate) const ACTION_CAPACITY: usize = 131072;

#[cfg(test)]
thread_local! {
    pub(super) static RETAINED_ACTION_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Credit {
    pub actions: usize,
    pub bytes: usize,
}

impl Credit {
    pub fn fits(self, available: Self) -> bool {
        self.actions <= available.actions && self.bytes <= available.bytes
    }
}

#[derive(Debug)]
pub(crate) enum ScoreItem {
    Event(TimedEvent),
    Marker(SceneMarker),
}

impl ScoreItem {
    pub fn time(&self) -> f64 {
        match self {
            Self::Event(e) => e.time,
            Self::Marker(m) => m.time,
        }
    }

    pub fn retained_bytes(&self) -> usize {
        match self {
            Self::Event(e) => {
                e.actions.capacity() * size_of::<Action>()
                    + e.actions
                        .iter()
                        .map(|action| {
                            #[cfg(test)]
                            RETAINED_ACTION_VISITS.with(|visits| visits.set(visits.get() + 1));
                            action.heap_bytes()
                        })
                        .sum::<usize>()
            }
            Self::Marker(m) => m.name.capacity(),
        }
    }

    pub fn charge(&self) -> Credit {
        Credit {
            actions: match self {
                Self::Event(e) => e.actions.len(),
                Self::Marker(_) => 1,
            },
            bytes: self.retained_bytes(),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum Terminal {
    Finished,
    Failed(ScriptError),
    Closed(String),
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ScriptStatus {
    pub evaluated_through: f64,
    pub done: bool,
    pub error: Option<String>,
    pub error_position: Option<rhai::Position>,
    pub stopped: bool,
    pub warnings: Vec<ScriptWarning>,
}

#[derive(Clone, Debug, Default)]
pub(super) struct WarningStore {
    entries: BTreeMap<(&'static str, rhai::Position), ScriptWarning>,
    outbox: Option<Weak<ScoreStream>>,
}

impl WarningStore {
    pub fn push(&mut self, warning: ScriptWarning) {
        if let Some(stream) = self.outbox.as_ref().and_then(Weak::upgrade) {
            stream
                .state
                .lock()
                .unwrap()
                .warnings
                .entry((warning.kind, warning.position))
                .and_modify(|entry| entry.count += 1)
                .or_insert(warning);
            stream.changed.notify_all();
        } else {
            self.entries
                .entry((warning.kind, warning.position))
                .and_modify(|entry| entry.count += 1)
                .or_insert(warning);
        }
    }
    pub fn snapshot(&self) -> Vec<ScriptWarning> {
        if let Some(stream) = self.outbox.as_ref().and_then(Weak::upgrade) {
            stream
                .state
                .lock()
                .unwrap()
                .warnings
                .values()
                .cloned()
                .collect()
        } else {
            self.entries.values().cloned().collect()
        }
    }
    #[cfg(test)]
    pub fn iter(&self) -> impl Iterator<Item = &ScriptWarning> {
        self.entries.values()
    }
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[derive(Debug)]
pub(crate) struct ScoreOutbox {
    pub warnings: BTreeMap<(&'static str, rhai::Position), ScriptWarning>,
    #[cfg(test)]
    pub script_sizes: (usize, usize, usize),
    pub header: Option<Scenario>,
    pub ring: VecDeque<(ScoreItem, Credit)>,
    pub watermark_sec: f64,
    pub credit: Credit,
    pub unreturned: Credit,
    pub pending: Option<Credit>,
    pub script_parked_full: bool,
    pub worker_waiting: bool,
    pub terminal: Option<Terminal>,
    pub parked_cursor_sec: f64,
}

impl ScoreOutbox {
    pub(super) fn overflow(&self) -> bool {
        self.worker_waiting
            && self.unreturned == Credit::default()
            && self
                .pending
                .is_some_and(|pending| !pending.fits(self.credit))
    }
}

#[derive(Debug)]
pub(crate) struct ScoreStream {
    pub state: Mutex<ScoreOutbox>,
    pub changed: Condvar,
    pub pace_limit_sec: AtomicU64,
    pub script_stop: AtomicBool,
    pub bound_reached: AtomicBool,
    pub bound: Option<f64>,
    pub capacity: Credit,
    pub(super) warnings: Arc<Mutex<WarningStore>>,
    producer: Mutex<Option<Thread>>,
}

impl ScoreStream {
    pub fn new(bound: Option<f64>, capacity: usize) -> Arc<Self> {
        let capacity = Credit {
            actions: capacity,
            bytes: capacity * size_of::<Action>(),
        };
        Arc::new_cyclic(|weak| Self {
            state: Mutex::new(ScoreOutbox {
                warnings: BTreeMap::new(),
                #[cfg(test)]
                script_sizes: (0, 0, 0),
                header: None,
                ring: VecDeque::with_capacity(capacity.actions),
                watermark_sec: 0.0,
                credit: capacity,
                unreturned: Credit::default(),
                pending: None,
                script_parked_full: false,
                worker_waiting: false,
                terminal: None,
                parked_cursor_sec: f64::INFINITY,
            }),
            changed: Condvar::new(),
            pace_limit_sec: AtomicU64::new(0.0f64.to_bits()),
            script_stop: AtomicBool::new(false),
            bound_reached: AtomicBool::new(false),
            bound,
            capacity,
            warnings: Arc::new(Mutex::new(WarningStore {
                entries: BTreeMap::new(),
                outbox: Some(weak.clone()),
            })),
            producer: Mutex::new(None),
        })
    }

    pub fn register_producer(&self) {
        *self.producer.lock().unwrap() = Some(thread::current());
    }
    fn wake(&self) {
        if let Some(producer) = &*self.producer.lock().unwrap() {
            producer.unpark();
        }
    }

    pub fn push(&self, item: ScoreItem) -> Result<(), ScriptError> {
        let charge = item.charge();
        loop {
            let mut state = self.state.lock().unwrap();
            if matches!(state.terminal, Some(Terminal::Closed(_))) {
                return Ok(());
            }
            if !charge.fits(self.capacity) {
                return Err(ScriptError::new(
                    "score item exceeds the ring capacity",
                    None,
                ));
            }
            if charge.fits(state.credit) {
                state.credit.actions -= charge.actions;
                state.credit.bytes -= charge.bytes;
                state.ring.push_back((item, charge));
                state.pending = None;
                state.script_parked_full = false;
                self.changed.notify_all();
                return Ok(());
            }
            if state.header.is_none() {
                return Err(ScriptError::new(
                    "score items before time advances exceed the ring capacity",
                    None,
                ));
            }
            state.pending = Some(charge);
            state.script_parked_full = true;
            self.changed.notify_all();
            drop(state);
            if self.script_stop.load(Ordering::Acquire) {
                return Ok(());
            }
            thread::park();
        }
    }

    pub fn set_header(&self, header: Scenario) {
        let mut state = self.state.lock().unwrap();
        if state.terminal.is_none() && state.header.is_none() {
            // Pre-header events are still invisible and owned by the producer.
            if let Some((a, b)) = header.meter_shaping.basin_hz {
                let (lo, hi) = crate::core::meter::sane_basin(a, b);
                for (item, _) in &mut state.ring {
                    if let ScoreItem::Event(event) = item {
                        for action in &mut event.actions {
                            action.set_founder_basin(0.5 * (lo + hi));
                        }
                    }
                }
            }
            if header.harmonic_rhythm > 0 {
                let mut members = BTreeMap::new();
                let mut warnings = Vec::new();
                for (item, _) in &mut state.ring {
                    if let ScoreItem::Event(event) = item {
                        for action in &mut event.actions {
                            match action {
                                Action::Spawn {
                                    population_id,
                                    spec,
                                    ..
                                } => {
                                    members.insert(
                                        *population_id,
                                        spec.control.supports_harmonic_rhythm(),
                                    );
                                }
                                Action::UpdatePopulation {
                                    population_id,
                                    patch,
                                } if members.get(population_id) == Some(&true) => {
                                    for label in super::context::filter_harmonic_patch(patch)
                                        .into_iter()
                                        .flatten()
                                    {
                                        warnings.push((label, ScriptWarning { kind: label, count: 1,
                                            message: format!("harmonic_rhythm: {label} would remove birth-time participation; previous value retained"), position: rhai::Position::NONE }));
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                for (kind, warning) in warnings {
                    state
                        .warnings
                        .entry((kind, warning.position))
                        .and_modify(|w| w.count += 1)
                        .or_insert(warning);
                }
            }
            state.header = Some(header);
            self.changed.notify_all();
        }
    }

    pub fn advance(&self, cursor: f64, pace: bool) {
        {
            let mut state = self.state.lock().unwrap();
            if matches!(state.terminal, Some(Terminal::Closed(_))) {
                return;
            }
            state.watermark_sec = cursor;
            self.changed.notify_all();
        }
        while pace && cursor > f64::from_bits(self.pace_limit_sec.load(Ordering::Acquire)) {
            self.state.lock().unwrap().parked_cursor_sec = cursor;
            if self.script_stop.load(Ordering::Acquire)
                || cursor <= f64::from_bits(self.pace_limit_sec.load(Ordering::Acquire))
            {
                break;
            }
            thread::park();
        }
        self.state.lock().unwrap().parked_cursor_sec = f64::INFINITY;
    }

    pub fn publish_pace(&self, tick: Tick, tb: Timebase) {
        let lead = tb.hop as u64 * LEAD_HOPS;
        let limit = tb.tick_to_sec(tick.saturating_add(lead));
        self.pace_limit_sec
            .store(limit.to_bits(), Ordering::Release);
        if limit >= self.state.lock().unwrap().parked_cursor_sec + tb.tick_to_sec(lead) / 2.0 {
            self.wake();
        }
    }

    pub fn finish(&self, error: Option<ScriptError>) {
        let mut state = self.state.lock().unwrap();
        if state.terminal.is_none() {
            state.terminal = Some(error.map_or(Terminal::Finished, Terminal::Failed));
            self.changed.notify_all();
        }
    }

    pub fn close(&self, reason: impl Into<String>) {
        {
            let mut state = self.state.lock().unwrap();
            if matches!(state.terminal, Some(Terminal::Closed(_))) {
                return;
            }
            state.terminal = Some(Terminal::Closed(reason.into()));
            state.ring.clear();
            state.credit = Credit {
                actions: self.capacity.actions - state.unreturned.actions,
                bytes: self.capacity.bytes - state.unreturned.bytes,
            };
            state.pending = None;
            state.script_parked_full = false;
        }
        self.changed.notify_all();
        self.script_stop.store(true, Ordering::Release);
        self.wake();
    }

    pub fn wait_ready(
        &self,
        tick: Tick,
        tb: Timebase,
        exiting: &AtomicBool,
        end_request: &AtomicBool,
        realtime: bool,
    ) -> Result<(), String> {
        let started = Instant::now();
        loop {
            if exiting.load(Ordering::Acquire) || end_request.load(Ordering::Acquire) {
                self.close("stopped");
                return Ok(());
            }
            let mut state = self.state.lock().unwrap();
            if state.terminal.is_some() || tb.sec_to_tick(state.watermark_sec) > tick {
                state.worker_waiting = false;
                return Ok(());
            }
            state.worker_waiting = true;
            let error = if state.overflow() {
                Some(format!(
                    "score items within one hop exceed the ring capacity at t={:.9}",
                    tb.tick_to_sec(tick)
                ))
            } else if realtime
                && started.elapsed().as_secs_f64() > tb.tick_to_sec(tb.hop as u64 * LEAD_HOPS)
            {
                Some(format!(
                    "script fell behind the performance at t={:.9}",
                    tb.tick_to_sec(tick)
                ))
            } else {
                None
            };
            drop(state);
            if let Some(error) = error {
                self.close(error.clone());
                return Err(error);
            }
            thread::sleep(Duration::from_micros(200));
        }
    }

    pub fn pop_due(&self, tick: Tick, tb: Timebase) -> Option<(ScoreItem, Credit)> {
        let mut state = self.state.lock().unwrap();
        if state.header.is_none()
            || (state.terminal.is_none() && tb.sec_to_tick(state.watermark_sec) <= tick)
        {
            return None;
        }
        if state
            .ring
            .front()
            .is_none_or(|(item, _)| tb.sec_to_tick(item.time()) > tick)
        {
            return None;
        }
        let (item, charge) = state.ring.pop_front().unwrap();
        state.unreturned.actions += charge.actions;
        state.unreturned.bytes += charge.bytes;
        Some((item, charge))
    }

    pub fn return_credit(&self, charge: Credit) {
        let mut state = self.state.lock().unwrap();
        state.unreturned.actions -= charge.actions;
        state.unreturned.bytes -= charge.bytes;
        state.credit.actions += charge.actions;
        state.credit.bytes += charge.bytes;
        self.changed.notify_all();
        if state.script_parked_full {
            self.wake();
        }
    }

    pub fn status(&self) -> ScriptStatus {
        let state = self.state.lock().unwrap();
        ScriptStatus {
            evaluated_through: state.watermark_sec,
            done: matches!(state.terminal, Some(Terminal::Finished)),
            error: match &state.terminal {
                Some(Terminal::Failed(e)) => Some(e.to_string()),
                Some(Terminal::Closed(reason)) if reason != "stopped" => Some(reason.clone()),
                _ => None,
            },
            stopped: matches!(state.terminal, Some(Terminal::Closed(_))),
            error_position: match &state.terminal {
                Some(Terminal::Failed(error)) => error.position,
                _ => None,
            },
            warnings: state.warnings.values().cloned().collect(),
        }
    }
}

#[derive(Debug)]
pub(crate) struct ScriptSession {
    pub stream: Arc<ScoreStream>,
    pub thread: Option<JoinHandle<()>>,
}

impl ScriptSession {
    pub fn drain(
        &self,
        tb: Timebase,
        stop: &AtomicBool,
        mut consume: impl FnMut(ScoreItem),
    ) -> Result<(), String> {
        self.stream
            .pace_limit_sec
            .store(f64::INFINITY.to_bits(), Ordering::Release);
        self.stream.wake();
        let mut tick = 0;
        loop {
            self.stream.wait_ready(tick, tb, stop, stop, false)?;
            {
                let state = self.stream.state.lock().unwrap();
                if state.header.is_none() {
                    match &state.terminal {
                        Some(Terminal::Failed(error)) => return Err(error.to_string()),
                        Some(Terminal::Closed(reason)) => return Err(reason.clone()),
                        _ => {}
                    }
                }
            }
            while let Some((item, charge)) = self.stream.pop_due(tick, tb) {
                consume(item);
                self.stream.return_credit(charge);
            }
            let state = self.stream.state.lock().unwrap();
            if state.ring.is_empty()
                && let Some(terminal) = &state.terminal
            {
                return match terminal {
                    Terminal::Finished => Ok(()),
                    Terminal::Failed(error) => Err(error.to_string()),
                    Terminal::Closed(reason) => Err(reason.clone()),
                };
            }
            drop(state);
            tick += tb.hop as Tick;
        }
    }
}

impl Drop for ScriptSession {
    fn drop(&mut self) {
        self.stream.close("stopped");
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
