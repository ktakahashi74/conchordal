//! Ordered auditory observations with optional uncalibrated group and memory diagnostics.

use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use crossbeam_channel::{Sender, TrySendError};
use serde::Serialize;

use super::{
    gesture, phrase, proposals::frontend, recall, reference_inventory, resources, ridge, section,
    whole,
};
use crate::{
    config::{TemporalAcousticConfig, TemporalRidgeConfig},
    core::log2space::Log2Space,
};

const OBSERVATION_VERSION: u64 = 24;
const QUEUE_CAPACITY: usize = 32;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ObservationState {
    #[default]
    Off,
    Waiting,
    Warming,
    Receiving,
    Finished,
    Failed,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct Snapshot {
    pub state: ObservationState,
    pub bus: u8,
    pub source_epoch: u64,
    pub sample_rate: u32,
    pub frame_id: Option<u64>,
    pub support_start_sample: u64,
    pub support_end_sample: u64,
    pub available_sample: u64,
    pub input_end_sample: u64,
    pub source_missing_samples: u64,
    pub delivery_dropped_frames: u64,
    pub received_frames: u64,
    pub fully_supported_frames: u64,
    pub rejected_frames: u64,
    pub delivery_delay_us: u64,
    pub spectral_power_sum: Option<f64>,
    pub hop_start_sample: u64,
    pub mono_mean_square: Option<f64>,
    pub trajectories: Option<super::trajectory::TrajectoryFrame>,
    pub ridge_parameters: Option<TemporalRidgeConfig>,
    pub ridges: Option<ridge::Update>,
    pub ridge_failed: bool,
    pub acoustic_parameters: Option<TemporalAcousticConfig>,
    pub acoustic: Option<frontend::Snapshot>,
    pub acoustic_failed: bool,
    pub memory_parameters: Option<crate::config::TemporalMemoryConfig>,
    pub memory: Option<recall::Snapshot>,
    pub memory_failed: bool,
    pub memory_error: Option<&'static str>,
    pub reference_inventory: Option<reference_inventory::Snapshot>,
    pub reference_inventory_error: Option<&'static str>,
    pub gesture_parameters: Option<crate::config::TemporalGestureConfig>,
    pub gesture: Option<gesture::Snapshot>,
    pub gesture_error: Option<&'static str>,
    pub period_parameters: Option<crate::config::TemporalPeriodConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub groove_parameters: Option<crate::config::TemporalGrooveConfig>,
    pub period: Option<frontend::recurrence::Snapshot>,
    pub period_error: Option<&'static str>,
    pub phrase_parameters: Option<crate::config::TemporalPhraseConfig>,
    pub phrase: Option<phrase::Snapshot>,
    pub phrase_error: Option<&'static str>,
    pub group_prototypes: Option<super::body_model::Shared>,
    pub action_profile_features: Option<super::action_profiles::Snapshot>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub action_profile_resources: Option<super::action_profiles::Resources>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_resources: Option<resources::Snapshot>,
    pub section_parameters: Option<crate::config::TemporalSectionConfig>,
    pub section: Option<section::Snapshot>,
    pub section_error: Option<&'static str>,
    pub whole_parameters: Option<crate::config::TemporalWholeConfig>,
    pub whole: Option<whole::Snapshot>,
    pub whole_error: Option<&'static str>,
    pub relations_implemented: bool,
    pub action_enabled: bool,
}

#[derive(Clone, Default)]
pub(crate) struct Options {
    pub action_profiles: Option<Arc<super::action_profiles::Profiles>>,
    pub body_prototypes: Option<(
        crate::config::TemporalBodyConfig,
        super::body_model::Prototypes,
    )>,
    pub deterministic: bool,
    pub ridge: Option<TemporalRidgeConfig>,
    pub acoustic: Option<TemporalAcousticConfig>,
    pub memory: Option<crate::config::TemporalMemoryConfig>,
    pub gesture: Option<crate::config::TemporalGestureConfig>,
    pub period: Option<crate::config::TemporalPeriodConfig>,
    pub groove: Option<crate::config::TemporalGrooveConfig>,
    pub phrase: Option<crate::config::TemporalPhraseConfig>,
    pub section: Option<crate::config::TemporalSectionConfig>,
    pub whole: Option<crate::config::TemporalWholeConfig>,
}

struct Observation {
    version: u64,
    bus: u8,
    epoch: u64,
    frame_id: u64,
    start: u64,
    end: u64,
    support_start: u64,
    available: u64,
    complete: bool,
    source_missing_samples: u64,
    dropped_frames: u64,
    published: Instant,
    power_scan: Arc<[f32]>,
    mono_energy: f64,
}

enum Event {
    Frame(Observation),
    End {
        input_end_sample: u64,
        source_epoch: u64,
        dropped_frames: u64,
        source_missing_samples: u64,
    },
}

pub(crate) struct Tap {
    pub reference_context: Arc<Mutex<reference_inventory::Context>>,
    pub candidate_table: Arc<Mutex<Option<Arc<super::action_profiles::consumer::Publication>>>>,
    tx: Option<Sender<Event>>,
    completed: Option<crossbeam_channel::Receiver<()>>,
    handle: Option<JoinHandle<()>>,
    pub snapshot: Arc<Mutex<Snapshot>>,
    space: Arc<Log2Space>,
    bus: u8,
    hop: u64,
    window: u64,
    next_frame: u64,
    epoch: u64,
    epoch_start: u64,
    source_missing_samples: u64,
    dropped_frames: u64,
    deterministic: bool,
}

impl Tap {
    pub fn spawn(
        bus: u8,
        sample_rate: u32,
        hop: usize,
        window: usize,
        space: Log2Space,
        options: Options,
    ) -> Self {
        assert!(bus < 2 && sample_rate > 0 && hop > 0 && window >= hop);
        let (tx, rx) = crossbeam_channel::bounded(QUEUE_CAPACITY);
        let (completion_tx, completed) = if options.deterministic {
            let (tx, rx) = crossbeam_channel::bounded(1);
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };
        let initial = Snapshot {
            state: ObservationState::Waiting,
            bus,
            sample_rate,
            ridge_parameters: options.ridge,
            acoustic_parameters: options.acoustic,
            memory_parameters: options.memory,
            gesture_parameters: options.gesture,
            period_parameters: options.period,
            groove_parameters: options.groove,
            phrase_parameters: options.phrase,
            section_parameters: options.section,
            whole_parameters: options.whole,
            ..Snapshot::default()
        };
        let snapshot = Arc::new(Mutex::new(initial));
        let reference_context = Arc::new(Mutex::new(reference_inventory::Context::default()));
        let reference_output = Arc::clone(&reference_context);
        let candidate_table = Arc::new(Mutex::new(None));
        let table_output = Arc::clone(&candidate_table);
        let output = Arc::clone(&snapshot);
        let state = Box::new(initial);
        let space = Arc::new(space);
        let worker_space = Arc::clone(&space);
        let handle = std::thread::Builder::new()
            .name(format!("temporal-observation-{bus}"))
            .spawn(move || {
                let resource_started = options.action_profiles.as_ref().map(|_| Instant::now());
                let mut state = state;
                let mut next_frame = 0;
                let new_tracker = |epoch| {
                    options.ridge.map(|scales| {
                        ridge::Tracker::new(
                            bus,
                            epoch,
                            sample_rate,
                            hop as u64,
                            ridge::Config {
                                means: scales.means,
                                deviations: scales.deviations,
                                distance_limit: 1.0,
                                retirement_sec: 0.25,
                            },
                        )
                        .expect("validated ridge diagnostic scales")
                    })
                };
                let new_frontend = |epoch, start| {
                    options.acoustic.map(|config| {
                        frontend::Frontend::configured(
                            worker_space.as_ref().clone(),
                            bus,
                            (epoch, start),
                            sample_rate,
                            hop as u64,
                            options
                                .ridge
                                .expect("acoustic diagnostics require ridge scales"),
                            config,
                        )
                        .expect("validated acoustic diagnostic configuration")
                    })
                };
                let split_frontend = |f: Option<frontend::Frontend>| {
                    if let Some(config) = options.period {
                        (
                            None,
                            f.map(|f| {
                                frontend::recurrence::Recurrence::configured(
                                    f,
                                    config,
                                    options.groove,
                                )
                                .expect("validated recurrence configuration")
                            }),
                        )
                    } else {
                        (f, None)
                    }
                };
                let mut published_key = None;
                let mut action_table = options
                    .action_profiles
                    .as_ref()
                    .map(|_| super::action_profiles::Table::new());
                let new_memory = |epoch| {
                    options.memory.map(|config| {
                        recall::Recall::new(
                            bus,
                            epoch,
                            sample_rate,
                            hop as u64,
                            config,
                            options.section,
                        )
                        .expect("validated memory diagnostic configuration")
                    })
                };
                let new_gesture = |epoch| {
                    options.gesture.map(|cfg| {
                        gesture::Gesture::new(bus, epoch, sample_rate, hop as u64, cfg)
                            .expect("validated gesture diagnostic configuration")
                    })
                };
                let new_phrase = |epoch, start| {
                    options.phrase.map(|c| {
                        phrase::Phrase::new(bus, epoch, start, sample_rate, hop as u64, c)
                            .expect("validated phrase configuration")
                    })
                };
                let new_section = |epoch, start| {
                    options.section.map(|c| {
                        section::Stream::new(
                            bus,
                            epoch,
                            sample_rate,
                            start,
                            hop as u64,
                            options.memory.expect("section requires memory"),
                            c,
                        )
                        .expect("validated section configuration")
                    })
                };
                let mut section = new_section(0, 0);
                let new_inventory = |epoch, start| {
                    options.memory.and_then(|m| m.retention).map(|_| {
                        reference_inventory::Stream::new(bus, epoch, start, sample_rate, hop as u64)
                    })
                };
                let mut inventory = new_inventory(0, 0);
                let mut phrase = new_phrase(0, 0);
                let mut gesture = new_gesture(0);
                let mut memory = new_memory(0);
                let (mut acoustic, mut recurrence) = split_frontend(new_frontend(0, 0));
                let mut tracker = if acoustic.is_none() && recurrence.is_none() {
                    new_tracker(0)
                } else {
                    None
                };
                let mut resources = resource_started.map(|started| {
                    resources::Meter::new(sample_rate, hop as u64, resources::elapsed_ns(started))
                });
                while let Ok(event) = rx.recv() {
                    let started = resources.as_ref().map(|_| Instant::now());
                    let published = match &event {
                        Event::Frame(frame) => Some(frame.published),
                        Event::End { .. } => None,
                    };
                    let received_before = state.received_frames;
                    let mut table_ns = 0;
                    let mut finished = false;
                    match event {
                        Event::End {
                            input_end_sample,
                            source_epoch,
                            dropped_frames,
                            source_missing_samples,
                        } => {
                            state.input_end_sample = input_end_sample;
                            if source_epoch != state.source_epoch {
                                section = None;
                                state.section = None;
                                state.whole = None;
                                state.whole_error = None;
                                phrase = None;
                                state.phrase = None;
                                state.group_prototypes = None;
                                state.action_profile_features = None;
                                recurrence = None;
                                state.period = None;
                                memory = None;
                                state.reference_inventory = None;
                                gesture = None;
                                state.gesture = None;
                                state.memory = None;
                            }
                            state.delivery_dropped_frames = dropped_frames;
                            state.source_missing_samples = source_missing_samples;
                            if let Some(memory) = memory.as_mut().filter(|_| !state.memory_failed) {
                                if let Err(error) = memory.finish(input_end_sample) {
                                    state.memory_failed = true;
                                    state.memory_error = Some(error);
                                }
                                state.memory = Some(memory.snapshot());
                            }
                            if let Some(g) =
                                gesture.as_mut().filter(|_| state.gesture_error.is_none())
                            {
                                match g.finish(input_end_sample) {
                                    Ok(()) => state.gesture = Some(g.snapshot()),
                                    Err(error) => {
                                        state.gesture_error = Some(error);
                                        state.gesture = None;
                                    }
                                }
                            }
                            if let Some(r) =
                                recurrence.as_mut().filter(|_| state.period_error.is_none())
                            {
                                match r.finish(input_end_sample) {
                                    Ok(()) => state.period = r.diagnostics(),
                                    Err(e) => {
                                        state.period = None;
                                        state.period_error = Some(e);
                                    }
                                }
                            }
                            if let Some(p) =
                                phrase.as_mut().filter(|_| state.phrase_error.is_none())
                            {
                                match p.finish(input_end_sample) {
                                    Ok(()) => state.phrase = Some(p.snapshot()),
                                    Err(e) => {
                                        state.phrase = None;
                                        state.group_prototypes = None;
                                        state.action_profile_features = None;
                                        state.phrase_error = Some(e);
                                    }
                                }
                            }
                            if let Some(s) =
                                section.as_mut().filter(|_| state.section_error.is_none())
                            {
                                match s.finish(input_end_sample) {
                                    Ok(()) => state.section = Some(*s.snapshot()),
                                    Err(e) => {
                                        state.section = None;
                                        state.section_error = Some(e);
                                    }
                                }
                            }
                            state.state = ObservationState::Finished;
                            reference_output
                                .lock()
                                .expect("reference context")
                                .inventory = None;
                            finished = true;
                        }
                        Event::Frame(frame) => {
                            worker_space.assert_scan_len_named(
                                &frame.power_scan,
                                "temporal_observation_power_scan",
                            );
                            if frame.version != OBSERVATION_VERSION
                                || frame.bus != bus
                                || frame.epoch < state.source_epoch
                                || frame.frame_id < next_frame
                                || frame.frame_id.checked_mul(hop as u64) != Some(frame.start)
                                || frame.start.checked_add(hop as u64) != Some(frame.end)
                                || frame.support_start > frame.start
                                || frame.support_start < frame.end.saturating_sub(window as u64)
                                || frame.complete
                                    != (frame.end - frame.support_start == window as u64)
                                || frame.available != frame.end
                                || frame.source_missing_samples < state.source_missing_samples
                                || frame.dropped_frames < state.delivery_dropped_frames
                                || frame.power_scan.iter().any(|x| !x.is_finite() || *x < 0.0)
                                || !frame.mono_energy.is_finite()
                                || frame.mono_energy < 0.0
                            {
                                state.rejected_frames += 1;
                            } else {
                                if frame.epoch != state.source_epoch {
                                    (acoustic, recurrence) =
                                        split_frontend(new_frontend(frame.epoch, frame.start));
                                    state.period = None;
                                    state.period_error = None;
                                    section = new_section(frame.epoch, frame.start);
                                    state.section = None;
                                    state.section_error = None;
                                    state.whole = None;
                                    state.whole_error = None;
                                    phrase = new_phrase(frame.epoch, frame.start);
                                    state.phrase = None;
                                    state.group_prototypes = None;
                                    state.action_profile_features = None;
                                    state.phrase_error = None;
                                    memory = new_memory(frame.epoch);
                                    inventory = new_inventory(frame.epoch, frame.start);
                                    state.reference_inventory = None;
                                    state.reference_inventory_error = None;
                                    gesture = new_gesture(frame.epoch);
                                    state.gesture = None;
                                    state.gesture_error = None;
                                    state.memory = None;
                                    state.memory_failed = false;
                                    state.memory_error = None;
                                    tracker = if acoustic.is_none() && recurrence.is_none() {
                                        new_tracker(frame.epoch)
                                    } else {
                                        None
                                    };
                                    state.ridge_failed = false;
                                    state.acoustic_failed = false;
                                }
                                state.state = if frame.complete {
                                    ObservationState::Receiving
                                } else {
                                    ObservationState::Warming
                                };
                                state.source_epoch = frame.epoch;
                                state.frame_id = Some(frame.frame_id);
                                state.support_start_sample = frame.support_start;
                                state.support_end_sample = frame.end;
                                state.available_sample = frame.available;
                                state.input_end_sample = frame.available;
                                state.source_missing_samples = frame.source_missing_samples;
                                state.delivery_dropped_frames = frame.dropped_frames;
                                state.received_frames += 1;
                                state.fully_supported_frames += u64::from(frame.complete);
                                state.delivery_delay_us =
                                    frame.published.elapsed().as_micros().min(u64::MAX as u128)
                                        as u64;
                                state.spectral_power_sum = frame
                                    .complete
                                    .then(|| frame.power_scan.iter().map(|x| f64::from(*x)).sum());
                                state.hop_start_sample = frame.start;
                                state.mono_mean_square = Some(frame.mono_energy);
                                if let Some(memory) = memory.as_mut() {
                                    memory
                                        .observe_acquisition(
                                            frame.start,
                                            frame.end,
                                            frame.available,
                                        )
                                        .expect("validated current-epoch acquisition frame");
                                    state.memory = Some(memory.snapshot());
                                }
                                state.trajectories =
                                    (acoustic.is_none() && recurrence.is_none() && frame.complete)
                                        .then(|| {
                                            super::trajectory::partition(
                                                &worker_space,
                                                &frame.power_scan,
                                                frame.mono_energy,
                                            )
                                            .expect("validated trajectory input")
                                        });
                                state.ridges = None;
                                state.acoustic = None;
                                if (acoustic.is_some() || recurrence.is_some())
                                    && !state.acoustic_failed
                                {
                                    let observation =
                                        frame.complete.then_some(frontend::Observation {
                                            power_scan: &frame.power_scan,
                                            mono_energy: frame.mono_energy,
                                            source_start: frame.support_start,
                                            source_end: frame.end,
                                            available_end: frame.available,
                                        });
                                    let result = if let Some(r) = recurrence.as_mut() {
                                        r.advance(frame.end, frame.available, observation).map(
                                            |out| {
                                                let summary = r.acoustic_snapshot(&out);
                                                state.period = r.diagnostics();
                                                (out, summary)
                                            },
                                        )
                                    } else {
                                        let a = acoustic.as_mut().unwrap();
                                        a.advance(frame.end, observation).map(|out| {
                                            let summary = a.snapshot(&out);
                                            (out, summary)
                                        })
                                    };
                                    match result {
                                        Ok((out, summary)) => {
                                            state.trajectories = out.partition;
                                            state.ridges = Some(out.ridges);

                                            if let Some(memory) =
                                                memory.as_mut().filter(|_| !state.memory_failed)
                                            {
                                                let result = if options.section.is_some() {
                                                    memory.receive(&summary, frame.end)
                                                } else {
                                                    memory.advance(&summary, frame.end, None)
                                                };
                                                if let Err(error) = result {
                                                    state.memory_failed = true;
                                                    state.memory_error = Some(error);
                                                }
                                                state.memory = Some(memory.snapshot());
                                            }
                                            if let Some(g) = gesture
                                                .as_mut()
                                                .filter(|_| state.gesture_error.is_none())
                                            {
                                                match g.advance(&summary, &out.ridges, frame.end) {
                                                    Ok(()) => state.gesture = Some(g.snapshot()),
                                                    Err(error) => {
                                                        state.gesture_error = Some(error);
                                                        state.gesture = None;
                                                    }
                                                }
                                            }
                                            if phrase.is_some() && state.gesture_error.is_some() {
                                                state.phrase = None;
                                                state.phrase_error =
                                                    Some("gesture input failed for this epoch");
                                            }
                                            if let (Some(p), Some(g)) = (
                                                phrase
                                                    .as_mut()
                                                    .filter(|_| state.phrase_error.is_none()),
                                                gesture
                                                    .as_ref()
                                                    .filter(|_| state.gesture_error.is_none()),
                                            ) {
                                                match p.advance(
                                                    &summary,
                                                    g,
                                                    state.memory.filter(|_| !state.memory_failed),
                                                    state.period,
                                                    frame.end,
                                                ) {
                                                    Ok(()) => state.phrase = Some(p.snapshot()),
                                                    Err(e) => {
                                                        state.phrase = None;
                                                        state.phrase_error = Some(e);
                                                    }
                                                }
                                            }
                                            if let (Some((scales, model)), Some(p)) = (
                                                options.body_prototypes.as_ref(),
                                                phrase
                                                    .as_ref()
                                                    .filter(|_| state.phrase_error.is_none()),
                                            ) {
                                                super::body_model::Shared::refresh(
                                                    &mut state.group_prototypes,
                                                    model,
                                                    *scales,
                                                    p,
                                                    &summary,
                                                    (bus, frame.epoch),
                                                );
                                            } else {
                                                state.group_prototypes = None;
                                                state.action_profile_features = None;
                                            }
                                            let table_started =
                                                resources.as_ref().map(|_| Instant::now());
                                            state.action_profile_features = match (
                                                options.action_profiles.as_ref(),
                                                action_table.as_mut(),
                                                phrase
                                                    .as_ref()
                                                    .filter(|_| state.phrase_error.is_none()),
                                                state.group_prototypes.as_ref(),
                                                gesture
                                                    .as_ref()
                                                    .filter(|_| state.gesture_error.is_none()),
                                            ) {
                                                (
                                                    Some(profiles),
                                                    Some(table),
                                                    Some(p),
                                                    Some(shared),
                                                    Some(gesture),
                                                ) => table.refresh(
                                                    profiles,
                                                    p,
                                                    shared,
                                                    gesture,
                                                    recurrence.as_ref(),
                                                ),
                                                _ => None,
                                            };
                                            table_ns =
                                                table_started.map_or(0, resources::elapsed_ns);
                                            state.action_profile_resources = action_table
                                                .as_ref()
                                                .map(super::action_profiles::Table::resources);
                                            if section.is_some() && state.phrase_error.is_some() {
                                                state.section = None;
                                                state.section_error =
                                                    Some("phrase input failed for this epoch");
                                            }
                                            if let (Some(s), Some(p)) = (
                                                section
                                                    .as_mut()
                                                    .filter(|_| state.section_error.is_none()),
                                                state.phrase.as_ref(),
                                            ) {
                                                match s.advance(
                                                    &summary,
                                                    p,
                                                    state.period,
                                                    memory
                                                        .as_ref()
                                                        .filter(|_| !state.memory_failed),
                                                ) {
                                                    Ok(()) => state.section = Some(*s.snapshot()),
                                                    Err(e) => {
                                                        state.section = None;
                                                        state.section_error = Some(e);
                                                    }
                                                }
                                            }
                                            if options.section.is_some()
                                                && let Some(m) =
                                                    memory.as_mut().filter(|_| !state.memory_failed)
                                            {
                                                let result = if let Some(s) = section
                                                    .as_ref()
                                                    .filter(|_| state.section_error.is_none())
                                                {
                                                    m.advance(&summary, frame.end, Some(s.cues()))
                                                } else {
                                                    Err("phrase cue input failed for this epoch")
                                                };
                                                if let Err(e) = result {
                                                    state.memory_failed = true;
                                                    state.memory_error = Some(e);
                                                }
                                                state.memory = Some(m.snapshot());
                                            }
                                            state.reference_inventory = None;
                                            if let Some(inventory) =
                                                inventory.as_mut().filter(|_| {
                                                    state.reference_inventory_error.is_none()
                                                })
                                            {
                                                if let Some(m) = state
                                                    .memory
                                                    .as_ref()
                                                    .filter(|_| !state.memory_failed)
                                                {
                                                    match inventory.advance(
                                                        &summary,
                                                        state.period.as_ref(),
                                                        m,
                                                        options
                                                            .memory
                                                            .unwrap()
                                                            .retention
                                                            .unwrap()
                                                            .no_memory_bias,
                                                        frame.end,
                                                    ) {
                                                        Ok(snapshot) => {
                                                            state.reference_inventory =
                                                                Some(snapshot)
                                                        }
                                                        Err(e) => {
                                                            state.reference_inventory_error =
                                                                Some(e)
                                                        }
                                                    }
                                                } else {
                                                    state.reference_inventory_error =
                                                        Some("shared memory input unavailable");
                                                }
                                            }
                                            if let Some(c) = options.whole {
                                                if let (Some(p), Some(s)) =
                                                    (state.phrase.as_ref(), state.section.as_ref())
                                                {
                                                    match whole::observe(c, p, s, state.memory) {
                                                        Ok(w) => {
                                                            state.whole = Some(w);
                                                            state.whole_error = None;
                                                        }
                                                        Err(e) => {
                                                            state.whole = None;
                                                            state.whole_error = Some(e);
                                                        }
                                                    }
                                                } else {
                                                    state.whole = None;
                                                    state.whole_error = Some(
                                                        "whole-piece lower-head input unavailable",
                                                    );
                                                }
                                            }
                                            state.acoustic = Some(summary);
                                        }
                                        Err(error) => {
                                            state.group_prototypes = None;
                                            state.action_profile_features = None;
                                            state.reference_inventory = None;
                                            if recurrence.is_some() {
                                                state.period = None;
                                                state.period_error = Some(error);
                                            }
                                            if phrase.is_some() {
                                                state.phrase = None;
                                                state.phrase_error = Some(error);
                                            }
                                            if section.is_some() {
                                                state.section = None;
                                                state.section_error = Some(error);
                                            }
                                            state.acoustic_failed = true;
                                            state.ridge_failed = true;
                                            if gesture.is_some() {
                                                state.gesture = None;
                                                state.gesture_error =
                                                    Some("acoustic input failed for this epoch");
                                            }
                                        }
                                    }
                                }
                                if let Some(tracker) =
                                    tracker.as_mut().filter(|_| !state.ridge_failed)
                                {
                                    let mut points = [ridge::Point {
                                        frequency_log2: None,
                                        log_envelope: None,
                                    }; 7];
                                    let mut count = 0;
                                    let supported =
                                        state.trajectories.filter(|t| t.spectral_shape_supported);
                                    if let Some(trajectories) = supported {
                                        for (slot, bin) in trajectories.peak_bins.iter().enumerate()
                                        {
                                            if let Some(bin) = bin {
                                                points[count] = ridge::Point {
                                                    frequency_log2: Some(f64::from(
                                                        worker_space.centers_log2[*bin],
                                                    )),
                                                    log_envelope: Some(
                                                        trajectories.log_envelope[slot],
                                                    ),
                                                };
                                                count += 1;
                                            }
                                        }
                                    }
                                    match tracker
                                        .advance(frame.end, supported.map(|_| &points[..count]))
                                    {
                                        Ok(update) => state.ridges = Some(update),
                                        Err(_) => state.ridge_failed = true,
                                    }
                                }
                                next_frame = frame
                                    .frame_id
                                    .checked_add(1)
                                    .expect("observation frame overflow");
                            }
                            if options.memory.is_some_and(|m| m.retention.is_some()) {
                                let mut context =
                                    reference_output.lock().expect("reference context");
                                context.inventory = state.reference_inventory;
                                if let Some(memory) =
                                    memory.as_mut().filter(|_| !state.memory_failed)
                                {
                                    context.retained = memory.retained_ids();
                                } else {
                                    context.inventory = None;
                                }
                            }
                        }
                    }
                    let key = state
                        .action_profile_features
                        .as_ref()
                        .map(super::action_profiles::consumer::Key::from);
                    if key != published_key {
                        let published = key.and_then(|_| {
                            action_table
                                .as_ref()
                                .zip(options.action_profiles.as_ref())
                                .and_then(|(table, profiles)| table.publication(profiles))
                        });
                        *table_output.lock().expect("candidate table publication") = published;
                        published_key = key;
                    }
                    {
                        let mut destination = output.lock().expect("observation snapshot");
                        *destination = *state;
                        if let (Some(meter), Some(started)) = (resources.as_mut(), started) {
                            let wall_ns = resources::elapsed_ns(started);
                            if finished {
                                meter.finish(state.input_end_sample, wall_ns);
                            } else if state.received_frames > received_before {
                                meter.frame(
                                    state.source_epoch,
                                    state.hop_start_sample,
                                    wall_ns,
                                    table_ns,
                                    resources::elapsed_ns(published.unwrap()),
                                );
                            } else {
                                meter.rejected(wall_ns);
                            }
                            state.worker_resources = Some(meter.snapshot);
                            destination.worker_resources = state.worker_resources;
                        }
                    }
                    if finished {
                        return;
                    }
                    if let Some(tx) = &completion_tx
                        && tx.send(()).is_err()
                    {
                        return;
                    }
                }
                *table_output.lock().expect("candidate table publication") = None;
                state.state = ObservationState::Failed;
                state.group_prototypes = None;
                state.action_profile_features = None;
                if let Some(meter) = resources.as_mut() {
                    meter.close(state.input_end_sample);
                    state.worker_resources = Some(meter.snapshot);
                }
                reference_output
                    .lock()
                    .expect("reference context")
                    .inventory = None;
                *output.lock().expect("observation snapshot") = *state;
            })
            .expect("spawn temporal observer");
        Self {
            reference_context,
            candidate_table,
            tx: Some(tx),
            completed,
            handle: Some(handle),
            snapshot,
            space,
            bus,
            hop: hop as u64,
            window: window as u64,
            next_frame: 0,
            epoch: 0,
            epoch_start: 0,
            source_missing_samples: 0,
            dropped_frames: 0,
            deterministic: options.deterministic,
        }
    }

    pub fn observe(&mut self, frame_id: u64, power_scan: &[f32], mono_energy: f64) {
        self.space
            .assert_scan_len_named(power_scan, "temporal_tap_power_scan");
        assert!(
            frame_id >= self.next_frame,
            "temporal source frames must be ordered"
        );
        let start = frame_id
            .checked_mul(self.hop)
            .expect("observation sample overflow");
        let end = start
            .checked_add(self.hop)
            .expect("observation sample overflow");
        if frame_id != self.next_frame {
            self.source_missing_samples += (frame_id - self.next_frame) * self.hop;
            self.epoch = self
                .epoch
                .checked_add(1)
                .expect("observation epoch overflow");
            self.epoch_start = start;
        }
        self.next_frame = frame_id.checked_add(1).expect("observation frame overflow");
        let event = Event::Frame(Observation {
            version: OBSERVATION_VERSION,
            bus: self.bus,
            epoch: self.epoch,
            frame_id,
            start,
            end,
            support_start: end.saturating_sub(self.window).max(self.epoch_start),
            available: end,
            complete: end - self.epoch_start >= self.window,
            source_missing_samples: self.source_missing_samples,
            dropped_frames: self.dropped_frames,
            published: Instant::now(),
            power_scan: Arc::from(power_scan),
            mono_energy,
        });
        if self.deterministic {
            // Offline analysis may advance only after both observer outputs are published.
            if self.tx.as_ref().unwrap().send(event).is_err()
                || self.completed.as_ref().unwrap().recv().is_err()
            {
                self.snapshot.lock().expect("observation snapshot").state =
                    ObservationState::Failed;
            }
        } else {
            match self.tx.as_ref().unwrap().try_send(event) {
                Ok(()) => (),
                Err(TrySendError::Full(_)) => self.dropped_frames += 1,
                Err(TrySendError::Disconnected(_)) => {
                    self.snapshot.lock().expect("observation snapshot").state =
                        ObservationState::Failed
                }
            }
        }
    }
}

impl Drop for Tap {
    fn drop(&mut self) {
        // The analysis thread owns this shutdown; audio callbacks never join workers.
        if let Some(tx) = self.tx.take() {
            let _ = tx.send(Event::End {
                input_end_sample: self.next_frame.saturating_mul(self.hop),
                source_epoch: self.epoch,
                dropped_frames: self.dropped_frames,
                source_missing_samples: self.source_missing_samples,
            });
        }
        if let Some(handle) = self.handle.take()
            && handle.join().is_err()
        {
            self.snapshot.lock().expect("observation snapshot").state = ObservationState::Failed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_delivery_waits_for_reference_and_snapshot_publication() {
        let space = Log2Space::new(100., 200., 12);
        let power = vec![1.; space.n_bins()];
        let mut tap = Tap::spawn(
            0,
            48_000,
            128,
            512,
            space,
            Options {
                deterministic: true,
                section: Some(section::tests::runtime_config()),
                memory: Some(crate::config::TemporalMemoryConfig {
                    retention: Some(crate::config::TemporalRetentionConfig {
                        tau_sec: 20.,
                        kappa: 0.7,
                        strength_max: 1.2,
                        r_max: 1.,
                        no_memory_bias: 0.,
                    }),
                    candidates: None,
                    scales: [1.; 10],
                    span_hops: 8,
                    episodes: 16,
                    query_cadence_ms: 100,
                    deadline_ms: 100,
                }),
                ..Options::default()
            },
        );
        let context = Arc::clone(&tap.reference_context);
        let snapshot = Arc::clone(&tap.snapshot);
        let held = context.lock().unwrap();
        let (started_tx, started_rx) = crossbeam_channel::bounded(1);
        let (done_tx, done_rx) = crossbeam_channel::bounded(1);
        let worker = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            tap.observe(0, &power, 0.25);
            done_tx.send(()).unwrap();
            tap
        });
        started_rx.recv().unwrap();
        assert_eq!(
            done_rx.recv_timeout(std::time::Duration::from_millis(20)),
            Err(crossbeam_channel::RecvTimeoutError::Timeout)
        );
        drop(held);
        done_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        assert_eq!(snapshot.lock().unwrap().received_frames, 1);
        assert_eq!(snapshot.lock().unwrap().support_end_sample, 128);
        drop(worker.join().unwrap());
        assert_eq!(snapshot.lock().unwrap().state, ObservationState::Finished);
    }

    #[test]
    fn eof_uses_input_clock_and_discards_memory_after_undelivered_epoch_change() {
        let space = Log2Space::new(100., 6400., 32);
        let mut scan = vec![0.; space.n_bins()];
        scan[20] = 1.;
        for changed_epoch in [false, true] {
            let mut tap = Tap::spawn(
                1,
                48000,
                512,
                2048,
                space.clone(),
                Options {
                    deterministic: true,
                    ridge: Some(TemporalRidgeConfig {
                        means: [0.; 3],
                        deviations: [0.05, 4., 1.],
                    }),
                    acoustic: Some(TemporalAcousticConfig {
                        group_means: [0.; 3],
                        group_deviations: [0.05, 4., 1.],
                        accent_means: [0.; 2],
                        accent_deviations: [1.; 2],
                        group_retirement_sec: 2.,
                        inactive_energy_max: 1e-8,
                        correlation_window_sec: 0.25,
                        min_pairs: 8,
                        min_coverage: 0.9,
                        persistence_hops: 3,
                    }),
                    phrase: Some(phrase::tests::config()),
                    section: Some(section::tests::runtime_config()),
                    whole: Some(whole::tests::config()),
                    groove: None,
                    body_prototypes: None,
                    action_profiles: None,
                    period: Some(crate::config::TemporalPeriodConfig {
                        model: crate::config::ArrivalModel::Hazard,
                        coefficients: [0.; 18],
                        means: [0.; 8],
                        deviations: [1.; 8],
                        horizon_sec: 0.1,
                    }),
                    gesture: Some(crate::config::TemporalGestureConfig {
                        rms_reference: 0.1,
                        means: [0.; 5],
                        deviations: [1.; 5],
                        coefficients: [[[0.; 11]; 4]; 4],
                    }),
                    memory: Some(crate::config::TemporalMemoryConfig {
                        retention: None,
                        candidates: None,
                        scales: [1.; 10],
                        span_hops: 8,
                        episodes: 16,
                        query_cadence_ms: 100,
                        deadline_ms: 100,
                    }),
                },
            );
            let output = Arc::clone(&tap.snapshot);
            for frame in 0..40 {
                tap.observe(frame, &scan, 0.25);
            }
            let until = std::time::Instant::now() + std::time::Duration::from_secs(10);
            let before_eof = loop {
                let snapshot = output.lock().unwrap();
                if snapshot.support_end_sample == 40 * 512 {
                    assert!(snapshot.whole_error.is_none(), "{:?}", snapshot.whole_error);
                    break snapshot.whole;
                }
                drop(snapshot);
                assert!(
                    std::time::Instant::now() < until,
                    "observer did not publish its final audio frame"
                );
                std::thread::sleep(std::time::Duration::from_millis(1));
            };
            // Model a terminal queue overflow after the last accepted observation.
            tap.next_frame = 100;
            tap.dropped_frames = 60;
            if changed_epoch {
                tap.epoch = 1;
                tap.source_missing_samples = 512;
            }
            drop(tap);
            let state = *output.lock().unwrap();
            assert!(state.gesture_error.is_none());
            assert!(state.phrase_error.is_none(), "{:?}", state.phrase_error);
            assert!(state.section_error.is_none(), "{:?}", state.section_error);
            assert!(state.period_error.is_none(), "{:?}", state.period_error);
            assert!(!state.memory_failed, "{:?}", state.memory_error);
            assert_eq!(state.support_end_sample, 40 * 512);
            assert_eq!(state.input_end_sample, 100 * 512);
            assert_eq!(state.received_frames, 40);
            if changed_epoch {
                assert!(state.memory.is_none());
                assert!(state.gesture.is_none());
                assert!(state.period.is_none());
                assert!(state.phrase.is_none());
                assert!(state.section.is_none());
                assert!(state.whole.is_none());
            } else {
                let before = before_eof.unwrap();
                let after = state.whole.unwrap();
                assert!(after.controls.is_some());
                assert_eq!(after.observed_end_sample, 40 * 512);
                assert_eq!(
                    serde_json::to_value(before).unwrap(),
                    serde_json::to_value(after).unwrap(),
                    "EOF changed the whole-piece scoring snapshot"
                );
                assert_eq!(state.period.unwrap().received_at, 100 * 512);
                assert!(state.phrase.unwrap().censored);
                assert!(state.section.unwrap().censored);
                assert!(!state.section.unwrap().initial_context_only);
                assert_eq!(state.phrase.unwrap().end_sample, 100 * 512);
                let memory = state.memory.unwrap();
                let acquisition = memory.acquisition.unwrap();
                assert_eq!(acquisition.end_sample, 40 * 512);
                assert_eq!(acquisition.delivery_cut_sample, 40 * 512);
                assert_eq!(acquisition.missing_seconds, 0.);
                assert_eq!(
                    memory.stored_total, 0,
                    "EOF cannot seal before the half-second observation lag"
                );
                assert!(memory.queries > 0 && memory.completed > 0);
                assert!(memory.latest.is_none(), "expired result survived EOF");
                assert_eq!(state.gesture.unwrap().unresolved, 1.);
            }
        }
    }

    #[test]
    fn acoustic_diagnostics_keep_handles_and_missing_support_across_gaps() {
        let space = Log2Space::new(100., 6400., 32);
        let mut scan = vec![0.; space.n_bins()];
        let silence = scan.clone();
        scan[20] = 1.;
        for source_gap in [false, true] {
            let mut tap = Tap::spawn(
                1,
                48000,
                512,
                2048,
                space.clone(),
                Options {
                    deterministic: true,
                    ridge: Some(TemporalRidgeConfig {
                        means: [0.; 3],
                        deviations: [0.05, 4., 1.],
                    }),
                    memory: Some(crate::config::TemporalMemoryConfig {
                        retention: None,
                        candidates: None,
                        scales: [1.; 10],
                        span_hops: 8,
                        episodes: 16,
                        query_cadence_ms: 100,
                        deadline_ms: 100,
                    }),
                    gesture: None,
                    period: None,
                    groove: None,
                    phrase: None,
                    section: None,
                    whole: None,
                    body_prototypes: None,
                    action_profiles: None,
                    acoustic: Some(TemporalAcousticConfig {
                        group_means: [0.; 3],
                        group_deviations: [0.05, 4., 1.],
                        accent_means: [0.; 2],
                        accent_deviations: [1.; 2],
                        group_retirement_sec: 2.,
                        inactive_energy_max: 1e-8,
                        correlation_window_sec: 0.25,
                        min_pairs: 8,
                        min_coverage: 0.9,
                        persistence_hops: 3,
                    }),
                },
            );
            let output = Arc::clone(&tap.snapshot);
            for frame in 0..40 {
                if frame < 3 {
                    tap.observe(frame, &silence, 0.);
                } else {
                    tap.observe(frame, &scan, 0.25);
                }
            }
            if !source_gap {
                tap.next_frame = 42;
                tap.dropped_frames = 2;
            }
            tap.observe(42, &scan, 0.25);
            drop(tap);
            let s = *output.lock().unwrap();
            assert!(!s.acoustic_failed);
            assert!(!s.memory_failed, "{:?}", s.memory_error);
            let clock = s.memory.unwrap().acquisition.unwrap();
            assert_eq!(clock.end_sample, 43 * 512);
            assert_eq!(clock.record_bytes, 12288);
            if source_gap {
                assert_eq!(clock.origin_sample, 42 * 512);
                assert_eq!(clock.retained_records, 1);
                assert_eq!(clock.missing_seconds, 0.);
            } else {
                assert_eq!(clock.origin_sample, 0);
                assert_eq!(clock.retained_records, 41);
                assert!((clock.missing_seconds - 1024. / 48000.).abs() < 1e-12);
            }
            let a = s.acoustic.unwrap();
            assert_eq!(s.source_epoch, u64::from(source_gap));
            if source_gap {
                assert!(a.energy.is_none());
                assert!(a.retained_groups.iter().all(Option::is_none));
            } else {
                assert!((a.energy.unwrap().iter().sum::<f64>() - 0.25).abs() < 1e-12);
                assert!(
                    a.feature_gaps
                        .iter()
                        .flatten()
                        .any(|g| g.known_samples == 0)
                );
                assert!(a.features[..7].iter().any(Option::is_some));
            }
            for (i, f) in a
                .features
                .iter()
                .enumerate()
                .filter_map(|(i, f)| f.map(|f| (i, f)))
            {
                assert_eq!(Some(f.raw.group), a.group_handles[i]);
                assert_eq!(f.raw.group.epoch, s.source_epoch);
                assert_eq!(f.raw.group.bus, 1);
                if source_gap {
                    assert_eq!(f.raw.known_samples, 0);
                }
            }
        }
    }

    #[test]
    fn ridge_diagnostics_reset_source_epochs_and_mask_delivery_gaps() {
        let space = Log2Space::new(100.0, 200.0, 12);
        let mut scan = vec![0.0; space.n_bins()];
        scan[5] = 1.0;
        let options = Options {
            deterministic: true,
            acoustic: None,
            memory: None,
            gesture: None,
            period: None,
            groove: None,
            phrase: None,
            section: None,
            whole: None,
            body_prototypes: None,
            action_profiles: None,
            ridge: Some(TemporalRidgeConfig {
                means: [0.; 3],
                deviations: [0.05, 4., 1.],
            }),
        };
        for source_gap in [false, true] {
            for recovery in [1, 4, 5] {
                let mut tap = Tap::spawn(1, 48_000, 128, 512, space.clone(), options.clone());
                let output = Arc::clone(&tap.snapshot);
                for frame in 0..8 {
                    tap.observe(frame, &scan, 0.25);
                }
                if !source_gap {
                    // Model two packets lost after analysis without changing its source epoch.
                    tap.next_frame = 10;
                    tap.dropped_frames = 2;
                }
                for frame in 10..10 + recovery {
                    tap.observe(frame, &scan, 0.25);
                }
                drop(tap);
                let state = *output.lock().unwrap();
                assert!(!state.ridge_failed);
                let update = state.ridges.unwrap();
                assert_eq!(update.observed, !source_gap || recovery >= 4);
                if source_gap && recovery < 4 {
                    assert!(update.current.iter().all(Option::is_none));
                    continue;
                }
                let ridge = update.current[0].unwrap();
                assert_eq!(ridge.handle.bus, 1);
                assert_eq!(ridge.handle.epoch, u64::from(source_gap));
                assert_eq!(ridge.end_sample, (10 + recovery) * 128);
                assert_eq!(
                    ridge.slopes[0].is_some(),
                    recovery >= 5 || (!source_gap && recovery >= 4)
                );
                if source_gap && recovery == 4 {
                    assert!(ridge.links.iter().all(Option::is_none));
                }
                if !source_gap && recovery == 1 {
                    assert_eq!(ridge.links[0].unwrap().parent_end_sample, 8 * 128);
                    assert_eq!(ridge.links[0].unwrap().secant, None);
                }
            }
        }
    }

    #[test]
    fn zero_is_known_only_after_a_complete_source_window() {
        let space = Log2Space::new(100.0, 200.0, 12);
        let silence = vec![0.0; space.n_bins()];
        for frames in [0, 1, 3, 4, 8] {
            let mut tap = Tap::spawn(
                1,
                48_000,
                128,
                512,
                space.clone(),
                Options {
                    deterministic: true,
                    ..Options::default()
                },
            );
            let output = Arc::clone(&tap.snapshot);
            assert_eq!(output.lock().unwrap().state, ObservationState::Waiting);
            for frame in 0..frames {
                tap.observe(frame, &silence, 0.0);
            }
            drop(tap);
            let state = *output.lock().unwrap();
            assert_eq!(state.state, ObservationState::Finished);
            assert_eq!(state.received_frames, frames);
            assert_eq!(state.fully_supported_frames, frames.saturating_sub(3));
            assert_eq!(state.spectral_power_sum, (frames >= 4).then_some(0.0));
            assert_eq!(state.mono_mean_square, (frames > 0).then_some(0.0));
            assert_eq!(state.trajectories.is_some(), frames >= 4);
            if let Some(frame) = state.trajectories {
                assert_eq!(frame.energy, [0.; 8]);
                assert_eq!(frame.peak_bins, [None; 7]);
                assert!(frame.spectral_shape_supported);
            }
            assert_eq!(state.bus, 1);
            assert_eq!(state.available_sample, frames * 128);
            assert_eq!(state.source_missing_samples, 0);
            assert!(!state.relations_implemented && !state.action_enabled);
        }
    }

    #[test]
    fn source_gap_restarts_warmup_and_preserves_original_samples() {
        let space = Log2Space::new(100.0, 200.0, 12);
        let power = vec![1.0; space.n_bins()];
        for recovery in [1, 3, 4] {
            let mut tap = Tap::spawn(
                0,
                48_000,
                128,
                512,
                space.clone(),
                Options {
                    deterministic: true,
                    ..Options::default()
                },
            );
            let output = Arc::clone(&tap.snapshot);
            for frame in 0..8 {
                tap.observe(frame, &power, 0.25);
            }
            for frame in 10..10 + recovery {
                tap.observe(frame, &power, 0.25);
            }
            drop(tap);
            let state = *output.lock().unwrap();
            assert_eq!(state.source_epoch, 1);
            assert_eq!(state.source_missing_samples, 256);
            assert_eq!(state.delivery_dropped_frames, 0);
            assert_eq!(state.support_start_sample, 1280);
            assert_eq!(state.support_end_sample, (10 + recovery) * 128);
            assert_eq!(state.fully_supported_frames, 5 + u64::from(recovery == 4));
            assert_eq!(state.mono_mean_square, Some(0.25));
            assert_eq!(state.trajectories.is_some(), recovery == 4);
            assert_eq!(
                state.spectral_power_sum,
                (recovery == 4).then_some(space.n_bins() as f64)
            );
        }
    }

    #[test]
    fn saturated_delivery_is_bounded_and_counts_loss_at_eof() {
        for bus in 0..2 {
            for recover in [false, true] {
                let space = Log2Space::new(100.0, 200.0, 12);
                let power = vec![1.0; space.n_bins()];
                let mut tap = Tap::spawn(bus, 48_000, 128, 512, space, Options::default());
                let output = Arc::clone(&tap.snapshot);
                assert_eq!(tap.tx.as_ref().unwrap().capacity(), Some(32));
                // Stop publication; live sends must return even with a full queue.
                let guard = output.lock().unwrap();
                for frame in 0..256 {
                    tap.observe(frame, &power, 0.25);
                }
                assert!(tap.dropped_frames >= 256 - 32 - 1);
                assert!(tap.tx.as_ref().unwrap().len() <= 32);
                let dropped = tap.dropped_frames;
                drop(guard);
                if recover {
                    let deadline = Instant::now() + std::time::Duration::from_secs(5);
                    while output.lock().unwrap().received_frames != 256 - dropped {
                        assert!(Instant::now() < deadline, "observer failed to drain");
                        std::thread::yield_now();
                    }
                    tap.observe(256, &power, 0.25);
                }
                drop(tap);
                let state = *output.lock().unwrap();
                assert_eq!(state.state, ObservationState::Finished);
                assert_eq!(state.delivery_dropped_frames, dropped);
                assert_eq!(state.received_frames + dropped, 256 + u64::from(recover));
                assert_eq!(state.input_end_sample, (256 + u64::from(recover)) * 128);
                assert_eq!(state.source_missing_samples, 0);
                assert_eq!(state.source_epoch, 0);
                assert_eq!(state.rejected_frames, 0);
                if recover {
                    assert_eq!(state.frame_id, Some(256));
                    assert_eq!(state.support_end_sample, 257 * 128);
                } else {
                    assert_eq!(
                        state.input_end_sample - state.support_end_sample,
                        dropped * 128
                    );
                }
                println!(
                    "I10_QUEUE32 bus={bus} recovery={recover} received={} dropped={dropped}",
                    state.received_frames
                );
            }
        }
    }

    #[test]
    fn invalid_delivery_cannot_advance_or_replace_accepted_evidence() {
        let space = Log2Space::new(100.0, 200.0, 12);
        let power: Arc<[f32]> = vec![1.0; space.n_bins()].into();
        let tap = Tap::spawn(
            0,
            48_000,
            128,
            512,
            space,
            Options {
                action_profiles: Some(Arc::new(super::super::action_profiles::tests::model())),
                deterministic: true,
                ..Options::default()
            },
        );
        let output = Arc::clone(&tap.snapshot);
        let frame = |id, epoch| Observation {
            version: OBSERVATION_VERSION,
            bus: 0,
            epoch,
            frame_id: id,
            start: id * 128,
            end: (id + 1) * 128,
            support_start: id * 128,
            available: (id + 1) * 128,
            complete: false,
            source_missing_samples: 0,
            dropped_frames: 0,
            published: Instant::now(),
            power_scan: Arc::clone(&power),
            mono_energy: 0.25,
        };
        let tx = tap.tx.as_ref().unwrap();
        tx.send(Event::Frame(frame(10, 1))).unwrap();
        tap.completed.as_ref().unwrap().recv().unwrap();
        for case in 0..9 {
            let mut invalid = frame(11, 1);
            match case {
                0 => invalid.version = 0,
                1 => invalid.bus = 1,
                2 => invalid.epoch = 0,
                3 => invalid = frame(9, 1),
                4 => invalid.available -= 1,
                5 => invalid.end += 128,
                6 => invalid.power_scan = vec![f32::NAN; power.len()].into(),
                7 => invalid.mono_energy = f64::NAN,
                _ => invalid.mono_energy = -1.0,
            }
            tx.send(Event::Frame(invalid)).unwrap();
            tap.completed.as_ref().unwrap().recv().unwrap();
        }
        tx.send(Event::Frame(frame(11, 1))).unwrap();
        tap.completed.as_ref().unwrap().recv().unwrap();
        drop(tap);
        let state = *output.lock().unwrap();
        assert_eq!(state.rejected_frames, 9);
        let resources = state.worker_resources.unwrap();
        assert_eq!(resources.frames.count, state.received_frames);
        assert_eq!(resources.rejected.count, state.rejected_frames);
        assert_eq!(resources.finish.count, 1);
        assert_eq!(resources.delivery.count, state.received_frames);
        assert_eq!(resources.table.total_ns, 0);
        assert_eq!(resources.windows.total_ns, resources.frames.total_ns);
        assert!(resources.open_window.is_none());
        assert_eq!(state.received_frames, 2);
        assert_eq!(state.source_epoch, 1);
        assert_eq!(state.frame_id, Some(11));
        assert_eq!(state.support_end_sample, 12 * 128);
        assert!(state.spectral_power_sum.is_none());
    }

    #[test]
    #[should_panic(expected = "scan length mismatch: temporal_tap_power_scan")]
    fn observation_rejects_wrong_frequency_space_length() {
        let space = Log2Space::new(100.0, 200.0, 12);
        let mut tap = Tap::spawn(
            0,
            48_000,
            128,
            512,
            space,
            Options {
                deterministic: true,
                ..Options::default()
            },
        );
        tap.observe(0, &[0.0], 0.0);
    }

    #[test]
    #[should_panic(expected = "temporal source frames must be ordered")]
    fn observation_rejects_repeated_source_frames() {
        let space = Log2Space::new(100.0, 200.0, 12);
        let power = vec![1.0; space.n_bins()];
        let mut tap = Tap::spawn(
            0,
            48_000,
            128,
            512,
            space,
            Options {
                deterministic: true,
                ..Options::default()
            },
        );
        tap.observe(0, &power, 0.25);
        tap.observe(0, &power, 0.25);
    }
    #[test]
    fn real_waveform_memory_predictions_reach_independent_phrase_diagnostics() {
        let mut first_exposure = None;
        for (label, section_enabled, earlier_tone, frames) in [
            ("assay", false, true, 900),
            ("return-long", true, true, 900),
            ("first", true, false, 660),
            ("return", true, true, 660),
            ("focus", true, true, 700),
        ] {
            use crate::core::nsgt_kernel::{NsgtKernelLog2, NsgtLog2Config, PowerMode};
            use crate::core::nsgt_rt::RtNsgtKernelLog2;
            let space = Log2Space::new(100., 6400., 24);
            let kernel = NsgtKernelLog2::new(
                NsgtLog2Config {
                    fs: 48000.,
                    overlap: 0.75,
                    nfft_override: Some(2048),
                    ..Default::default()
                },
                space.clone(),
                None,
                PowerMode::Coherent,
            );
            let mut analysis = RtNsgtKernelLog2::new(kernel);
            let mut phrase_config = phrase::tests::config();
            if frames == 660 {
                phrase_config.hazard[11] = 0.;
            }
            let mut section_config = section::tests::runtime_config();
            if label == "focus" {
                section_config.recurrence[0] = 20.;
                section_config.new_context[0] = -10.;
                section_config.contrast[0] = -10.;
            }
            let mut tap = Tap::spawn(
                1,
                48000,
                512,
                2048,
                space,
                Options {
                    deterministic: true,
                    ridge: Some(TemporalRidgeConfig {
                        means: [0.; 3],
                        deviations: [0.05, 4., 1.],
                    }),
                    acoustic: Some(TemporalAcousticConfig {
                        group_means: [0.; 3],
                        group_deviations: [0.05, 4., 1.],
                        accent_means: [0.; 2],
                        accent_deviations: [1.; 2],
                        group_retirement_sec: 2.,
                        inactive_energy_max: 1e-8,
                        correlation_window_sec: 0.25,
                        min_pairs: 8,
                        min_coverage: 0.9,
                        persistence_hops: 3,
                    }),
                    phrase: Some(phrase_config),
                    section: section_enabled.then_some(section_config),
                    whole: section_enabled.then(whole::tests::config),
                    groove: None,
                    body_prototypes: None,
                    action_profiles: None,
                    period: Some(crate::config::TemporalPeriodConfig {
                        model: crate::config::ArrivalModel::Hazard,
                        coefficients: [0.; 18],
                        means: [0.; 8],
                        deviations: [1.; 8],
                        horizon_sec: 0.1,
                    }),
                    gesture: Some(crate::config::TemporalGestureConfig {
                        rms_reference: 0.1,
                        means: [0.; 5],
                        deviations: [1.; 5],
                        coefficients: [[[0.; 11]; 4]; 4],
                    }),
                    memory: Some(crate::config::TemporalMemoryConfig {
                        retention: None,
                        candidates: None,
                        scales: [1.; 10],
                        span_hops: 64,
                        episodes: 128,
                        query_cadence_ms: 100,
                        deadline_ms: 200,
                    }),
                },
            );
            let output = Arc::clone(&tap.snapshot);
            let mut audio = [0.; 512];
            let mut target_audio = Vec::new();
            for frame in 0..frames {
                for (i, x) in audio.iter_mut().enumerate() {
                    let time = (frame * 512 + i) as f64 / 48000.;
                    let phase = (frame % 48) as f64 / 48.;
                    let amp = if section_enabled && frame < 600 && (!earlier_tone || frame >= 300) {
                        0.
                    } else {
                        0.015
                            + 0.01 * (std::f64::consts::TAU * phase).sin()
                            + 0.003 * (2. * std::f64::consts::TAU * phase).sin()
                    };
                    *x = (amp * (std::f64::consts::TAU * 500. * time).sin()) as f32;
                }
                if frames == 660 && frame >= 300 {
                    target_audio.extend(audio.iter().map(|x| x.to_bits()));
                }
                let energy = audio.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / 512.;
                let power = analysis.process_hop(&audio);
                tap.observe(frame as u64, power, energy);
            }
            drop(tap);
            let state = *output.lock().unwrap();
            assert!(state.phrase_error.is_none(), "{:?}", state.phrase_error);
            assert!(state.section_error.is_none(), "{:?}", state.section_error);
            assert!(!state.acoustic_failed && !state.memory_failed);
            let p = state.phrase.unwrap();
            let scored: u64 = p
                .groups
                .iter()
                .flatten()
                .map(|g| g.prediction_comparisons)
                .sum();
            println!(
                "I7_NSGT_HANDOFF case={label} scored={scored} closure_support={} continuation_support={}",
                p.closure.supported_mass, p.continuation.supported_mass
            );
            if earlier_tone {
                assert!(scored > 0);
            } else {
                assert_eq!(scored, 0);
            }
            if section_enabled && frames == 900 {
                // The whole returning cue eventually outgrows the retained observed continuations.
                assert_eq!(p.closure.supported_mass, 0.);
                assert_eq!(p.closure.unknown, 1.);
            } else if !section_enabled {
                assert!(p.closure.supported_mass > 0.);
            }
            assert!(p.continuation.supported_mass > 0.);
            assert!(p.censored);
            if let Some(section) = state.section {
                let groups: Vec<_> = section.groups.iter().flatten().collect();
                assert!(!groups.is_empty());
                assert!(groups.iter().any(|g| g.cumulative[34].is_some()));
                assert!(
                    groups
                        .iter()
                        .all(|g| g.observed_end_sample == frames as u64 * 512)
                );
                assert!(groups.iter().all(|g| g.unknown == 1.));
                assert!(
                    groups
                        .iter()
                        .any(|g| g.cue.is_some_and(|c| c.start_sample >= 600 * 512))
                );
                assert!(groups.iter().map(|g| g.prefix_support_queries).sum::<u64>() > 0);
                if label == "focus" {
                    let focused: Vec<_> = groups
                        .iter()
                        .flat_map(|g| g.candidates.iter().flatten())
                        .filter(|c| c.focus.is_some())
                        .collect();
                    println!(
                        "I8_REAL_FOCUS committed={} candidates={:?}",
                        section.committed_phrases, focused
                    );
                    assert!(
                        !focused.is_empty(),
                        "received phrase memory did not activate an earlier context"
                    );
                    assert!(focused.iter().all(|c| c.query_id.is_some() && c.mass > 0.));
                    assert!(
                        focused
                            .iter()
                            .any(|c| c.focus.unwrap().support_end_sample < 300 * 512
                                && c.start_sample >= 600 * 512),
                        "focus has no provenance before the intervening silence"
                    );
                }
                if frames == 660 {
                    let mut local_groups: Vec<_> = groups
                        .iter()
                        .map(|g| {
                            let raw = state
                                .acoustic
                                .as_ref()
                                .unwrap()
                                .features
                                .iter()
                                .flatten()
                                .find(|u| u.raw.group == g.group)
                                .unwrap()
                                .raw
                                .values;
                            (
                                g.cue.map(|c| {
                                    (
                                        c.start_sample,
                                        c.support_end_sample,
                                        c.weighted_seconds.to_bits(),
                                    )
                                }),
                                raw.map(|v| v.map(f64::to_bits)),
                                g.covariates[..80]
                                    .iter()
                                    .map(|v| v.map(f64::to_bits))
                                    .collect::<Vec<_>>(),
                            )
                        })
                        .collect();
                    local_groups.sort();
                    let scores: Vec<_> =
                        groups.iter().map(|g| g.covariates[80..].to_vec()).collect();
                    println!(
                        "I8_RETURN_CONTROL case={label} groups={} scores={scores:?} predictions={scored}",
                        groups.len()
                    );
                    if earlier_tone {
                        let (audio, local) = first_exposure.as_ref().unwrap();
                        assert!(
                            *audio == target_audio,
                            "target and intervening silence differ"
                        );
                        assert!(
                            *local == local_groups,
                            "normalized cue, acoustic values or non-retrieval section inputs differ"
                        );
                        assert!(groups.iter().any(|g| g.covariates[80].is_some()));
                        assert!(section.committed_phrases > 0);
                        assert_eq!(
                            state.memory.unwrap().phrase_episodes,
                            state.memory.unwrap().stored_episodes
                        );
                        println!(
                            "I8_MATCHED_CONTEXT samples={} groups={} acoustic_values=10 non_retrieval_values=80 bit_exact=true",
                            target_audio.len(),
                            groups.len()
                        );
                    } else {
                        assert!(groups.iter().all(|g| g.covariates[80..] == [None; 2]));
                        first_exposure = Some((target_audio, local_groups));
                    }
                }
                println!(
                    "I8_NSGT_HANDOFF case={label} groups={} support_matched_queries={}",
                    groups.len(),
                    groups.iter().map(|g| g.prefix_support_queries).sum::<u64>()
                );
            }
            assert!(!state.action_enabled);
        }
    }
}

#[cfg(test)]
mod replay;
