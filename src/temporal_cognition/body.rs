//! Private routed PCM capture and acoustic descriptors. Never feeds passive relation inputs.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender, bounded};
use serde::Serialize;

use super::{features, observables, resources, ridge::Handle};
use crate::config::TemporalBodyConfig;
use crate::core::nsgt_rt::RtNsgtKernelLog2;
use crate::life::sound::BodySnapshot;

pub(crate) const VOICES: usize = 64;
const BUFFERS: usize = 4;

pub(crate) fn validate(config: TemporalBodyConfig) -> Result<(), &'static str> {
    if config
        .means
        .iter()
        .chain(&config.accent_means)
        .any(|v| !v.is_finite())
        || config
            .deviations
            .iter()
            .chain(&config.accent_deviations)
            .any(|v| !v.is_finite() || *v < 0.)
    {
        return Err("temporal_body requires finite means and nonnegative finite deviations");
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct Record {
    pub source_id: u64,
    pub source_generation: u32,
    pub body_generation: u32,
    pub start: u64,
    pub end: u64,
    pub source_start: u64,
    pub available: u64,
    pub raw_values: [f64; 6],
    pub coverage: [f32; 6],
    pub bus: u8,
    pub mask: u8,
    pub active: bool,
}

impl Record {
    pub(crate) fn standardized(&self, config: TemporalBodyConfig) -> [Option<f64>; 6] {
        observables::standardize(
            std::array::from_fn(|i| (self.mask & (1 << i) != 0).then_some(self.raw_values[i])),
            config,
        )
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Snapshot {
    pub config: TemporalBodyConfig,
    pub prototype_model_version: Option<[u8; 32]>,
    #[serde(skip)]
    pub prototype_assignments: [Option<super::body_model::Assignment>; VOICES * 2],
    #[serde(skip)]
    pub records: [Record; VOICES * 2],
    pub version: u64,
    pub input_end: u64,
    pub processed_frames: u64,
    pub invalid_hops: u64,
    pub capture_drops: u64,
    pub outside_voice_hops: u64,
    pub processing_us: u64,
    pub max_processing_us: u64,
    pub worker_resources: resources::Snapshot,
    pub finished: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Owner {
    id: u64,
    generation: u32,
    body_generation: u32,
    born: u64,
    changed: u64,
}

struct Slot {
    owner: Owner,
    recipe: BodySnapshot,
    live: bool,
    sounding: bool,
}

struct Frame {
    start: u64,
    published: Instant,
    owners: [Option<Owner>; VOICES],
    supported: [bool; VOICES * 2],
    pcm: Vec<f32>,
}

pub(crate) struct Capture {
    slots: Vec<Option<Slot>>,
    next_generation: u32,
    hop: usize,
    sample_rate: u32,
    frame: Option<Box<Frame>>,
    free: Receiver<Box<Frame>>,
    tx: Option<Sender<Box<Frame>>>,
    completed: Option<Receiver<()>>,
    handle: Option<JoinHandle<()>>,
    snapshot: Arc<Mutex<Snapshot>>,
    deterministic: bool,
    clock: Arc<AtomicU64>,
    current_end: u64,
    dropped: u64,
    outside: u64,
}

#[derive(Clone, Copy)]
struct Sample {
    raw: features::RawDescriptor,
    energy: Option<f64>,
    shape: bool,
}

struct Lane {
    owner: Option<Owner>,
    bus: u8,
    nsgt: RtNsgtKernelLog2,
    features: features::Stream,
    scan: Vec<f64>,
    history: VecDeque<Sample>,
    accents: VecDeque<features::Accent>,
    evicted_accent: Option<u64>,
    next_sample: u64,
    warmup: usize,
    next_publish: u64,
}

impl Lane {
    fn new(nsgt: RtNsgtKernelLog2, bus: u8, cfg: TemporalBodyConfig) -> Self {
        let bins = nsgt.space().n_bins();
        let frames = ((2. * nsgt.fs()) as usize).div_ceil(nsgt.hop()) + 4;
        Self {
            owner: None,
            bus,
            features: features::Stream::new(
                bins,
                features::Config {
                    means: cfg.accent_means,
                    deviations: cfg.accent_deviations,
                    threshold: 1.,
                    rms_floor: 1e-6,
                },
            )
            .expect("validated body feature scales"),
            scan: vec![0.; bins],
            history: VecDeque::with_capacity(frames),
            accents: VecDeque::with_capacity(256),
            evicted_accent: None,
            next_sample: 0,
            warmup: 0,
            next_publish: 0,
            nsgt,
        }
    }

    fn push(
        &mut self,
        owner: Owner,
        start: u64,
        pcm: &[f32],
        supported: bool,
    ) -> Result<Option<Record>, &'static str> {
        let rate = self.nsgt.fs() as u32;
        let end = start + pcm.len() as u64;
        let changed = self.owner != Some(owner);
        if changed {
            self.owner = Some(owner);
            self.history.clear();
            self.accents.clear();
            self.evicted_accent = None;
            self.next_publish = owner.changed + u64::from(rate).div_ceil(10);
        }
        if changed || start != self.next_sample || !supported {
            self.nsgt.reset();
            self.features.clear();
            self.warmup = self.nsgt.nfft();
        }
        self.next_sample = end;
        let energy = supported
            .then(|| pcm.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / pcm.len() as f64);
        let mut shape = false;
        if supported {
            for (target, value) in self.scan.iter_mut().zip(self.nsgt.process_hop(pcm)) {
                *target = f64::from(*value);
            }
            self.warmup = self.warmup.saturating_sub(pcm.len());
            shape = self.warmup == 0;
            // The same whole-source energy assignment used by the group frontend.
            let mass: f64 = self.scan.iter().sum();
            if mass > 0. {
                for value in &mut self.scan {
                    *value = *value / mass * energy.unwrap();
                }
            }
        }
        let group = Handle {
            bus: self.bus,
            epoch: 0,
            generation: u64::from(owner.body_generation),
        };
        let update = self
            .features
            .push(
                self.nsgt.space(),
                features::Input {
                    stamp: features::Stamp {
                        group,
                        association: Some(group.generation),
                        grid_id: 0,
                        start,
                        end,
                        source_start: end
                            .saturating_sub(self.nsgt.nfft() as u64)
                            .max(owner.changed),
                        source_end: end,
                        available_end: end,
                        known_samples: if supported { pcm.len() as u64 } else { 0 },
                        observed: supported,
                    },
                    energy,
                    bus_energy: energy,
                    energy_scan: shape.then_some(&self.scan),
                },
                end,
            )?
            .expect("one ordered private hop");
        if supported {
            if self.history.len() == self.history.capacity() {
                self.history.pop_front();
            }
            self.history.push_back(Sample {
                raw: update.raw,
                energy,
                shape,
            });
        }
        if let Some(accent) = update.detector.and_then(|d| d.accent) {
            if self.accents.len() == self.accents.capacity() {
                self.evicted_accent = self.accents.pop_front().map(|a| a.event_end);
            }
            self.accents.push_back(accent);
        }
        let left = owner
            .born
            .max(owner.changed)
            .max(end.saturating_sub(2 * u64::from(rate)));
        while self.history.front().is_some_and(|s| s.raw.end <= left) {
            self.history.pop_front();
        }
        while self.accents.front().is_some_and(|a| a.event_end < left) {
            self.accents.pop_front();
        }
        if end < self.next_publish {
            return Ok(None);
        }
        self.next_publish = end + u64::from(rate).div_ceil(10);
        let descriptor = observables::summarize(
            group,
            (left, end),
            rate,
            self.history.iter().map(|s| (s.raw, s.energy, s.shape)),
            self.accents.iter().copied(),
            self.evicted_accent,
        );
        let mut record = Record {
            source_id: owner.id,
            source_generation: owner.generation,
            body_generation: owner.body_generation,
            start: left,
            end,
            source_start: descriptor.source_start,
            available: descriptor.available,
            coverage: descriptor.coverage.map(|v| v as f32),
            bus: self.bus,
            active: true,
            ..Record::default()
        };
        for (i, value) in descriptor.raw_values.into_iter().enumerate() {
            if let Some(value) = value {
                record.raw_values[i] = value;
                record.mask |= 1 << i;
            }
        }
        Ok(Some(record))
    }
}

impl Capture {
    pub(crate) fn spawn(
        nsgt: RtNsgtKernelLog2,
        config: TemporalBodyConfig,
        deterministic: bool,
        prototypes: Option<super::body_model::Prototypes>,
    ) -> Self {
        validate(config).expect("validated body diagnostic configuration");
        let hop = nsgt.hop();
        let sample_rate = nsgt.fs() as u32;
        let initial = Snapshot {
            config,
            prototype_model_version: prototypes.as_ref().map(|p| p.version),
            prototype_assignments: [None; VOICES * 2],
            records: [Record::default(); VOICES * 2],
            version: 0,
            input_end: 0,
            processed_frames: 0,
            invalid_hops: 0,
            capture_drops: 0,
            outside_voice_hops: 0,
            processing_us: 0,
            max_processing_us: 0,
            worker_resources: resources::Snapshot::default(),
            finished: false,
        };
        let snapshot = Arc::new(Mutex::new(initial));
        let output = Arc::clone(&snapshot);
        let clock = Arc::new(AtomicU64::new(0));
        let worker_clock = Arc::clone(&clock);
        let (tx, rx) = bounded::<Box<Frame>>(BUFFERS);
        let (free_tx, free) = bounded(BUFFERS);
        let (completion_tx, completed) = if deterministic {
            let (tx, rx) = bounded(1);
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };
        for _ in 0..BUFFERS {
            free_tx
                .send(Box::new(Frame {
                    start: 0,
                    published: Instant::now(),
                    owners: [None; VOICES],
                    supported: [false; VOICES * 2],
                    pcm: vec![0.; VOICES * 2 * hop],
                }))
                .unwrap();
        }
        let handle = std::thread::Builder::new()
            .name("private-body-analysis".into())
            .spawn(move || {
                let startup = Instant::now();
                let mut lanes: Vec<_> = (0..VOICES * 2)
                    .map(|i| Lane::new(nsgt.clone(), (i % 2) as u8, config))
                    .collect();
                let mut state = initial;
                let mut meter =
                    resources::Meter::new(sample_rate, hop as u64, resources::elapsed_ns(startup));
                while let Ok(frame) = rx.recv() {
                    let began = Instant::now();
                    let mut publish = false;
                    for (index, lane) in lanes.iter_mut().enumerate() {
                        let Some(owner) = frame.owners[index / 2] else {
                            publish |= state.records[index].active;
                            state.records[index] = Record::default();
                            state.prototype_assignments[index] = None;
                            lane.owner = None;
                            continue;
                        };
                        if lane.owner != Some(owner) {
                            state.records[index] = Record::default();
                            state.prototype_assignments[index] = None;
                            publish = true;
                        }
                        let pcm = &frame.pcm[index * hop..(index + 1) * hop];
                        let supported = frame.supported[index] && pcm.iter().all(|x| x.is_finite());
                        if !supported {
                            state.invalid_hops += 1;
                        }
                        match lane.push(owner, frame.start, pcm, supported) {
                            Ok(Some(mut record)) => {
                                record.available =
                                    record.end.max(worker_clock.load(Ordering::Acquire));
                                state.prototype_assignments[index] =
                                    prototypes.as_ref().and_then(|p| p.assign(&record, config));
                                state.records[index] = record;
                                publish = true;
                            }
                            Ok(None) => {}
                            Err(_) => {
                                state.invalid_hops += 1;
                                state.records[index] = Record::default();
                                state.prototype_assignments[index] = None;
                                publish = true;
                            }
                        }
                    }
                    state.input_end = frame.start + hop as u64;
                    state.processed_frames += 1;
                    state.processing_us = began.elapsed().as_micros() as u64;
                    state.max_processing_us = state.max_processing_us.max(state.processing_us);
                    let mut destination =
                        publish.then(|| output.lock().expect("private descriptor snapshot"));
                    if let Some(destination) = destination.as_mut() {
                        state.version += 1;
                        **destination = state;
                    }
                    meter.frame(
                        0,
                        frame.start,
                        resources::elapsed_ns(began),
                        0,
                        resources::elapsed_ns(frame.published),
                    );
                    state.worker_resources = meter.snapshot;
                    if let Some(destination) = destination.as_mut() {
                        destination.worker_resources = state.worker_resources;
                    }
                    drop(destination);
                    if free_tx.send(frame).is_err() {
                        break;
                    }
                    if let Some(tx) = &completion_tx
                        && tx.send(()).is_err()
                    {
                        break;
                    }
                }
                let finishing = Instant::now();
                state.version += 1;
                state.finished = true;
                let mut destination = output.lock().expect("final private descriptor snapshot");
                *destination = state;
                meter.finish(state.input_end, resources::elapsed_ns(finishing));
                destination.worker_resources = meter.snapshot;
            })
            .expect("spawn private body analysis");
        Self {
            slots: (0..VOICES).map(|_| None).collect(),
            next_generation: 1,
            hop,
            sample_rate,
            frame: None,
            free,
            tx: Some(tx),
            completed,
            handle: Some(handle),
            snapshot,
            deterministic,
            clock,
            current_end: 0,
            dropped: 0,
            outside: 0,
        }
    }

    pub(crate) fn prepare(
        &mut self,
        voices: impl Iterator<Item = (u64, u32, BodySnapshot)>,
        now: u64,
    ) {
        for slot in self.slots.iter_mut().flatten() {
            slot.live = false;
        }
        for (id, generation, recipe) in voices {
            let found = self.slots.iter().position(|s| {
                s.as_ref()
                    .is_some_and(|s| s.owner.id == id && s.owner.generation == generation)
            });
            if let Some(index) = found {
                let slot = self.slots[index].as_mut().unwrap();
                slot.live = true;
                if slot.recipe.kind != recipe.kind
                    || slot.recipe.unison != recipe.unison
                    || slot.recipe.ratios != recipe.ratios
                {
                    slot.owner.body_generation = self.next_generation;
                    self.next_generation += 1;
                    slot.owner.changed = now;
                }
                slot.recipe = recipe;
            } else if let Some(index) = self.slots.iter().position(Option::is_none) {
                self.slots[index] = Some(Slot {
                    owner: Owner {
                        id,
                        generation,
                        body_generation: self.next_generation,
                        born: now,
                        changed: now,
                    },
                    recipe,
                    live: true,
                    sounding: false,
                });
                self.next_generation += 1;
            } else {
                self.outside += 1;
            }
        }
    }

    pub(crate) fn token(&self, id: u64, generation: u32) -> Option<(usize, u32)> {
        self.slots.iter().enumerate().find_map(|(i, slot)| {
            slot.as_ref()
                .filter(|s| s.owner.id == id && s.owner.generation == generation)
                .map(|s| (i, s.owner.body_generation))
        })
    }

    pub(crate) fn prediction_input(
        &self,
        token: (usize, u32),
        now: u64,
        scheduled: u64,
        parameters: (f32, f32, crate::life::sound::envelope::Envelope),
    ) -> Option<crate::life::self_prediction::Input> {
        let owner = self.slots[token.0].as_ref()?.owner;
        if owner.body_generation != token.1 {
            return None;
        }
        let snapshot = self.snapshot.lock().expect("private prediction inputs");
        let mut descriptors = [[None; 6]; 2];
        let mut descriptor_support = [None; 2];
        for record in &snapshot.records {
            if record.active
                && record.source_id == owner.id
                && record.source_generation == owner.generation
                && record.body_generation == owner.body_generation
                && record.available <= now
                && record.end <= now
                && now - record.end <= u64::from(self.sample_rate) / 2
            {
                descriptors[record.bus as usize] = record.standardized(snapshot.config);
                descriptor_support[record.bus as usize] =
                    Some([record.start, record.end, record.available]);
            }
        }
        Some(crate::life::self_prediction::Input {
            body_generation: owner.body_generation,
            descriptors,
            descriptor_support,
            frequency_hz: parameters.0,
            amplitude: parameters.1,
            envelope: parameters.2,
            scheduled_release: None,
            control: None,
            retained_energy: [Default::default(); 2],
            coherent_energy: [[None; 16]; 2],
            sine: None,
            bank: None,
            descriptor_slot: token.0,
            descriptor_target_end: {
                let period = u64::from(self.sample_rate)
                    .div_ceil(10)
                    .div_ceil(self.hop as u64)
                    * self.hop as u64;
                let earliest = now
                    .max(scheduled)
                    .saturating_add(u64::from(self.sample_rate).div_ceil(10));
                owner.changed + earliest.saturating_sub(owner.changed).div_ceil(period) * period
            },
        })
    }

    pub(crate) fn begin(&mut self, now: u64) {
        self.current_end = now + self.hop as u64;
        self.frame = if self.deterministic {
            self.free.recv().ok()
        } else {
            self.free.try_recv().ok()
        };
        for slot in self.slots.iter_mut().flatten() {
            slot.sounding = false;
        }
        if let Some(frame) = &mut self.frame {
            frame.start = now;
            frame.pcm.fill(0.);
            frame.owners = std::array::from_fn(|i| self.slots[i].as_ref().map(|s| s.owner));
            frame.supported.fill(true);
        } else {
            self.dropped += 1;
        }
    }

    pub(crate) fn sample(
        &mut self,
        token: (usize, u32),
        offset: usize,
        value: f32,
        routing: crate::scenario::control::Routing,
    ) {
        let Some(slot) = self.slots[token.0].as_mut() else {
            return;
        };
        slot.sounding = true;
        let Some(frame) = self.frame.as_mut() else {
            return;
        };
        for (bus, routed) in [routing.to_habitat, routing.to_presentation]
            .into_iter()
            .enumerate()
        {
            if routed {
                if slot.owner.body_generation != token.1 {
                    frame.supported[token.0 * 2 + bus] = false;
                } else {
                    frame.pcm[(token.0 * 2 + bus) * self.hop + offset] += value;
                }
            }
        }
    }

    pub(crate) fn end(&mut self) {
        self.clock.store(self.current_end, Ordering::Release);
        if let Some(mut frame) = self.frame.take()
            && let Some(tx) = &self.tx
        {
            // Taking a pooled buffer reserves a queue slot; this send cannot wait for capacity.
            frame.published = Instant::now();
            if tx.try_send(frame).is_err() {
                self.dropped += 1;
            } else if let Some(completed) = &self.completed {
                // Offline generation may read or advance only after this frame is published.
                completed
                    .recv()
                    .expect("private body publication completion");
            }
        }
        for slot in &mut self.slots {
            if slot.as_ref().is_some_and(|s| !s.live && !s.sounding) {
                *slot = None;
            }
        }
    }

    pub(crate) fn source_audio(
        &self,
        slot: usize,
        identity: (u64, u32, u32),
    ) -> Option<(u64, [Option<&[f32]>; 2])> {
        let frame = self.frame.as_ref()?;
        let owner = frame.owners.get(slot)?.as_ref()?;
        if (owner.id, owner.generation, owner.body_generation) != identity {
            return None;
        }
        Some((
            frame.start,
            std::array::from_fn(|bus| {
                let index = slot * 2 + bus;
                frame.supported[index]
                    .then_some(&frame.pcm[index * self.hop..(index + 1) * self.hop])
            }),
        ))
    }

    pub(crate) fn snapshot(&self) -> Snapshot {
        let mut snapshot = *self.snapshot.lock().expect("private descriptor snapshot");
        snapshot.capture_drops = self.dropped;
        snapshot.outside_voice_hops = self.outside;
        snapshot
    }

    pub(crate) fn finish(&mut self) {
        self.tx.take();
        if let Some(handle) = self.handle.take() {
            handle.join().expect("private body worker");
        }
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.finish();
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod action_targets;
