//! Per-instance field dynamics and cell-based stochastic additive synthesis.

use std::f32::consts::PI;
#[cfg(test)]
use std::f32::consts::TAU;
use std::ops::Range;

use serde::Serialize;

use super::span_gap::{Phase as SpanPhase, SpanGap};
use crate::core::erb::{erb_bw_hz, erb_to_hz, hz_to_erb};
use crate::core::landscape::Landscape;
use crate::core::log2space::Log2Space;
use crate::scenario::control::PitchApplyMode;
use crate::scenario::{FloraPatch, FloraSeeding, FloraSpec, FloraTone};

#[path = "flora_metric.rs"]
mod metric;
#[path = "flora_support.rs"]
mod support;
use support::{SupportChoice, SupportScratch, SupportWindow};
#[path = "flora_respond.rs"]
mod respond;
#[path = "flora_span_gap.rs"]
mod span_gap;
use metric::MetricState;
pub(crate) use metric::TemporalObservation;
use respond::Respond;

#[path = "flora_dynamics.rs"]
mod dynamics;
#[path = "flora_synth.rs"]
mod synth;

struct ErbCell {
    lo_erb: f32,
    hi_erb: f32,
    bins: std::ops::Range<usize>,
}

/// One-ERB cells; merge short trailing edges and empty cells as in Flora.
fn partition_erb_cells(space: &Log2Space, erb_scan: &[f32], extent_hz: (f32, f32)) -> Vec<ErbCell> {
    space.assert_scan_len_named(erb_scan, "erb_scan");
    let start = space.centers_hz.partition_point(|&f| f < extent_hz.0);
    let end = space.centers_hz.partition_point(|&f| f <= extent_hz.1);
    let mut cells: Vec<ErbCell> = Vec::new();
    if start >= end {
        return cells;
    }
    let lo = hz_to_erb(extent_hz.0.max(space.fmin));
    let hi = hz_to_erb(extent_hz.1.min(space.fmax));
    let whole = (hi - lo).floor() as usize;
    let count = if hi - lo - (whole as f32) < 0.5 {
        whole.max(1)
    } else {
        whole + 1
    };
    let mut bin = start;
    let mut pending_lo = lo;
    for c in 0..count {
        let upper = if c + 1 == count {
            hi
        } else {
            lo + (c + 1) as f32
        };
        let first = bin;
        while bin < end && (c + 1 == count || erb_scan[bin] < upper) {
            bin += 1;
        }
        if bin == first {
            if let Some(prev) = cells.last_mut() {
                prev.hi_erb = upper;
            }
        } else {
            cells.push(ErbCell {
                lo_erb: pending_lo,
                hi_erb: upper,
                bins: first..bin,
            });
        }
        if !cells.is_empty() {
            pending_lo = upper;
        }
    }
    cells
}

pub(crate) fn harmonic_source_bound(
    space: &Log2Space,
    extent_hz: (f32, f32),
    residual: bool,
) -> usize {
    let erb: Vec<_> = space.centers_hz.iter().map(|&f| hz_to_erb(f)).collect();
    partition_erb_cells(space, &erb, extent_hz).len() * (2 + usize::from(residual))
}

const GROWTH_RATE: f32 = 8.0;
const RENEWAL_SEC: f32 = 10.0;

const METABOLISM: f32 = 0.3;
const CONSUMPTION_SEC: f32 = 0.3;
const REFERENCE_DENSITY: f32 = 10.0;
const ALLEE_DENSITY: f32 = 0.144;
const SEED_RAIN_DENSITY: f32 = 0.48;
const BACKGROUND_SEED_DENSITY: f32 = 9.6e-4;
const INITIAL_BIOMASS: f32 = 0.5;
const SLEW_SEC: f32 = 0.08;
/// A sounding lane reaches each new target over one nominal opportunity at the 2 Hz
/// reference beat, so window and peak reassignments are heard as swells, not steps.
pub(super) const CHANGE_SEC: f32 = 0.5;
const DEAD_MASS: f32 = 0.001;

fn ecological_pace(multiplier: f32, beat_pace: f32, opportunity_ratio: u8) -> f32 {
    // One renewal per opportunity; beat_pace is the private beat frequency / 2 Hz.
    beat_pace * multiplier.min(2.0 * RENEWAL_SEC * f32::from(opportunity_ratio))
}

#[derive(Clone, Debug)]
struct Ramp {
    value: f32,
    target: f32,
    step: f32,
    left: usize,
}

impl Ramp {
    fn new(value: f32) -> Self {
        Self {
            value,
            target: value,
            step: 0.0,
            left: 0,
        }
    }

    fn to(&mut self, target: f32, seconds: f32, fs: f32) {
        self.target = target;
        self.left = (seconds * fs).ceil() as usize;
        if self.left == 0 {
            self.value = target;
            self.step = 0.0;
        } else {
            self.step = (target - self.value) / self.left as f32;
        }
    }

    fn tick(&mut self) -> f32 {
        if self.left > 0 {
            self.left -= 1;
            self.value = if self.left == 0 {
                self.target
            } else {
                self.target - self.step * self.left as f32
            };
        }
        self.value
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Free,
    Current,
    Retiring,
}

#[derive(Clone, Debug, PartialEq)]
struct Lane {
    harmonic_hold: bool,
    harmonic_sounding: bool,
    mode: Mode,
    tonal: bool,
    noisy: bool,
    noise_index: usize,
    cell: Option<usize>,
    position: f32,
    target_position: f32,
    sounding_log2: f32,
    sigma: f32,
    r: f32,
    q: f32,
    target: f32,
    power_sum: f32,
    age_samples: usize,
    generation: u64,
    window: Range<usize>,
    pending_since: Option<u64>,
    forced_reason: Option<&'static str>,
    retirement_decay: f32,
    slew_target: f32,
    slew_left: usize,
    slew_step: f32,
    z: [f32; 2],
    w: [f32; 2],
    e: [f32; 2],
    a: f32,
    b: f32,
}

#[derive(Clone, Debug)]
struct Peak {
    bin: usize,
    position: f32,
    sigma: f32,
    window: Range<usize>,
    mass: f32,
    enter: bool,
    stay: bool,
    claimed: Option<usize>,
    suppressed: bool,
    visited: bool,
}

#[derive(Clone, Debug)]
struct Cell {
    lo_erb: f32,
    hi_erb: f32,
    bins: Range<usize>,
    floor: f32,
    excess: f32,
    tonal_mass: f32,
    residual_mass: f32,
    peak: Option<Peak>,
}

#[derive(Clone, Debug)]
pub(crate) struct FloraHarmonicState {
    consumer: Option<crate::temporal_cognition::harmonic_relations::consumer::Use>,
    cells: Vec<usize>,
    act_now: bool,
    weight: f64,
    query_version: u64,
    availability: crate::temporal_cognition::harmonic_relations::runtime::Availability,
    pub(crate) recorder: Option<crate::temporal_cognition::harmonic_relations::runtime::Recorder>,
    pub(crate) ticks: Option<Vec<u64>>,
}

#[derive(Clone, Debug)]
struct FloraSupportState {
    choice_scan: Vec<f64>,
    native_scores: Vec<f64>,
    window: Option<SupportWindow>,
    scratch: SupportScratch,
    #[cfg(test)]
    records: Option<Vec<SupportChoice>>,
    active: bool,
    action: Option<(
        std::sync::Arc<crate::core::harmonic_support::Snapshot>,
        crate::core::harmonic_support::Weights,
    )>,
    template: std::sync::Arc<[(isize, f32)]>,
    ref_power: f32,
    loudness_exp: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct Flora {
    pub(crate) harmonic: FloraHarmonicState,
    support: FloraSupportState,
    pub(crate) id: u64,
    pub(crate) span_gap: Option<SpanGap>,
    respond: Option<Respond>,
    spec: FloraSpec,
    // Temporarily taken only while allocating lanes. Every Flora owns a clock.
    temporal: Option<MetricState>,
    acoustic_observation: TemporalObservation,
    extent_hz: (f32, f32),
    space: Log2Space,
    range: Range<usize>,
    fs: f32,
    hop: usize,
    substeps: usize,
    pace: f32,
    diffusion: f32,
    rng: u64,
    growth_level_scan: Vec<f32>,
    harmonicity_state_scan: Vec<f32>,
    biomass_mass_scan: Vec<f32>,
    resource_state_scan: Vec<f32>,
    biomass_consumption_scan: Vec<f32>,
    external_consumption_scan: Vec<f32>,
    external_observed_frame: Option<u64>,
    external_current: bool,
    pub(crate) habitat_pcm: Vec<f32>,
    scratch_scan: Vec<f32>,
    erb_scan: Vec<f32>,
    consumption_rows: Vec<(usize, Range<usize>)>,
    consumption_weights: Vec<f32>,
    cells: Vec<Cell>,
    lanes: Vec<Lane>,
    tonal_len: usize,
    residual_end: usize,
    order: Vec<usize>,
    noise: Vec<f32>,
    noisy_len: usize,
    amp: Ramp,
    gain: Ramp,
    releasing: bool,
    births: u64,
    retirements: u64,
    flapping: u64,
    energy: [f64; 2],
    rendered_samples: u64,
    pub(crate) lifecycle_changed: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct FloraReport<'a> {
    #[serde(skip_serializing_if = "Option::is_none")]
    temporal: Option<&'a MetricState>,
    #[serde(skip_serializing_if = "Option::is_none")]
    acoustic_observation: Option<&'a TemporalObservation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    respond: Option<&'a Respond>,
    pace: f32,
    time_sec: f64,
    age_sec: f64,
    flora_id: u64,
    extent_hz: (f32, f32),
    frequency_hz: &'a [f32],
    biomass_mass_scan: &'a [f32],
    resource_state_scan: &'a [f32],
    biomass_consumption_scan: &'a [f32],
    external_consumption_scan: &'a [f32],
    external_observed_frame: Option<u64>,
    external_current: bool,
    total_biomass_mass: f32,
    latent_peaks_hz: Vec<f32>,
    births: u64,
    retirements: u64,
    flapping: u64,
    tonal_energy: f64,
    residual_energy: f64,
    rendered_partials: Vec<(f32, f32, bool)>,
    target_partials: Vec<(f32, f32, bool)>,
    lanes: Vec<FloraLaneReport>,
    growth_level_scan: &'a [f32],
    harmonicity_state_scan: &'a [f32],
}

#[derive(Debug, Serialize)]
struct FloraLaneReport {
    slot: usize,
    state: &'static str,
    frequency_hz: f32,
    target_mass: f32,
    #[serde(flatten)]
    timing: Option<FloraLaneTimingReport>,
    age_sec: f64,
}

#[derive(Debug, Serialize)]
struct FloraLaneTimingReport {
    generation: u64,
    pending_since: Option<u64>,
    forced_reason: Option<&'static str>,
}

impl Flora {
    pub(crate) fn site_id(&self) -> u32 {
        self.spec.site_id
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        id: u64,
        spec: FloraSpec,
        extent_hz: (f32, f32),
        seeding: FloraSeeding,
        scenario_seed: u64,
        landscape: &Landscape,
        fs: f32,
        hop: usize,
    ) -> Self {
        assert!(fs.is_finite() && fs > 0.0 && hop > 0);
        assert!(
            extent_hz.0.is_finite()
                && extent_hz.1.is_finite()
                && extent_hz.0 > 0.0
                && extent_hz.1 > extent_hz.0
        );
        let mut flora = Self {
            harmonic: FloraHarmonicState {
                consumer: None,
                cells: Vec::with_capacity(if landscape.harmonic.support_planned {
                    landscape.space.n_bins()
                } else {
                    0
                }),
                act_now: true,
                weight: 0.,
                query_version: 0,
                availability: Default::default(),
                recorder: None,
                ticks: None,
            },
            support: FloraSupportState {
                choice_scan: if landscape.harmonic.support_planned
                    || landscape.harmonic.support.is_some()
                {
                    vec![0.; landscape.space.n_bins()]
                } else {
                    Vec::new()
                },
                native_scores: if landscape.harmonic.support_planned
                    || landscape.harmonic.support.is_some()
                {
                    vec![0.; landscape.space.n_bins()]
                } else {
                    Vec::new()
                },
                window: None,
                scratch: Default::default(),
                #[cfg(test)]
                records: Default::default(),
                active: false,
                action: None,
                template: std::sync::Arc::from([]),
                ref_power: landscape.harmonic.support_ref_power,
                loudness_exp: landscape.harmonic.support_loudness_exp,
            },
            id,
            spec,
            span_gap: (spec.rest_on_habituation.is_finite() && spec.rest_on_habituation > 0.0)
                .then(|| {
                    SpanGap::new(
                        spec.rest_on_habituation,
                        scenario_seed ^ 0x464c_4f52_415f_5048,
                        id,
                        0,
                        fs,
                        0.0,
                    )
                }),
            respond: (spec.respond_to_onsets.is_finite() && spec.respond_to_onsets > 0.0)
                .then(|| Respond::new(spec.respond_to_onsets)),
            acoustic_observation: TemporalObservation::default(),
            temporal: Some(MetricState::new(scenario_seed ^ id, fs, hop)),
            extent_hz,
            space: landscape.space.clone(),
            range: 0..0,
            fs,
            hop,
            substeps: 1,
            pace: ecological_pace(spec.pace, 1.0, 1),
            diffusion: 0.0,
            rng: scenario_seed ^ id.wrapping_mul(0xD6E8_FEB8_6659_FD93),
            growth_level_scan: Vec::new(),
            harmonicity_state_scan: Vec::new(),
            biomass_mass_scan: Vec::new(),
            resource_state_scan: Vec::new(),
            biomass_consumption_scan: Vec::new(),
            external_consumption_scan: Vec::new(),
            external_observed_frame: None,
            external_current: false,
            habitat_pcm: vec![0.0; hop],
            scratch_scan: Vec::new(),
            erb_scan: Vec::new(),
            consumption_rows: Vec::new(),
            consumption_weights: Vec::new(),
            cells: Vec::new(),
            lanes: Vec::new(),
            tonal_len: 0,
            residual_end: 0,
            order: Vec::new(),
            noise: Vec::new(),
            noisy_len: 0,
            amp: Ramp::new(spec.amp),
            gain: Ramp::new(0.0),
            releasing: false,
            births: 0,
            retirements: 0,
            flapping: 0,
            energy: [0.0; 2],
            rendered_samples: 0,
            lifecycle_changed: true,
        };
        flora.on_space_change(&landscape.space);
        flora
            .growth_level_scan
            .copy_from_slice(&landscape.consonance_field_level);
        flora
            .harmonicity_state_scan
            .copy_from_slice(&landscape.harmonicity01);
        flora.seed_biomass(seeding, landscape);
        flora.assert_scans();
        flora.gain.to(1.0, spec.fade_sec, fs);
        flora
    }

    pub(crate) fn apply_patch(&mut self, patch: &FloraPatch) {
        if let Some(mode) = patch.pitch_apply_mode {
            self.spec.pitch_apply_mode = mode;
        }
        if let Some(pace) = patch.pace {
            self.spec.pace = pace;
            self.pace = ecological_pace(
                self.respond.as_ref().map_or(pace, |r| pace * r.factor),
                self.temporal.as_ref().unwrap().pace,
                self.temporal.as_ref().unwrap().opportunity_ratio,
            );
        }
        if let Some(amp) = patch.amp {
            self.spec.amp = amp;
            // A live level change is a gesture of the same scale as entry and release.
            self.amp.to(amp, self.spec.fade_sec.max(SLEW_SEC), self.fs);
            self.refresh_support_template();
        }
    }

    fn refresh_support_template(&mut self) {
        if self.support.choice_scan.is_empty() {
            return;
        }
        let hz = (self.extent_hz.0 * self.extent_hz.1).sqrt();
        let mut atom = Lane::new(self.spec.tone == FloraTone::Resonant, hz.log2());
        atom.r = 1.;
        if atom.noisy {
            atom.a = (-PI * erb_bw_hz(hz) / self.fs).exp();
        }
        self.support.template = atom
            .project_support_density(
                self.spec.amp,
                self.fs,
                &self.space,
                self.support.ref_power,
                self.support.loudness_exp,
            )
            .into();
    }

    pub(crate) fn release(&mut self, fade_sec: f32) {
        if !self.releasing {
            self.releasing = true;
            self.gain.to(0.0, fade_sec, self.fs);
        }
    }

    pub(crate) fn release_at_scenario_end(&mut self) {
        self.release(self.spec.fade_sec);
    }

    pub(crate) fn remaining_release_ticks(&self) -> u64 {
        if self.releasing {
            self.gain.left as u64
        } else {
            0
        }
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.releasing
            && self.gain.value == 0.0
            && self.lanes.iter().all(|lane| lane.mode == Mode::Free)
    }

    fn assert_scans(&self) {
        for (scan, name) in [
            (&self.growth_level_scan, "growth_level_scan"),
            (&self.harmonicity_state_scan, "harmonicity_state_scan"),
            (&self.biomass_mass_scan, "biomass_mass_scan"),
            (&self.resource_state_scan, "resource_state_scan"),
            (&self.biomass_consumption_scan, "biomass_consumption_scan"),
            (&self.external_consumption_scan, "external_consumption_scan"),
            (&self.scratch_scan, "scratch_scan"),
            (&self.erb_scan, "erb_scan"),
        ] {
            self.space.assert_scan_len_named(scan, name);
        }
    }

    pub(crate) fn needs_hr_observer(&self) -> bool {
        self.spec.routing.to_habitat && self.spec.leave_self_out
    }

    // Reporting owns these allocations; ordinary advance/render remain allocation-free.
    pub(crate) fn report<'a>(&'a self, time_sec: f64, landscape: &'a Landscape) -> FloraReport<'a> {
        self.assert_scans();
        let latent_peaks_hz = self
            .range
            .clone()
            .filter_map(|i| {
                let p = self.biomass_mass_scan[i];
                (p > 0.0
                    && (i == self.range.start || p > self.biomass_mass_scan[i - 1])
                    && (i + 1 == self.range.end || p >= self.biomass_mass_scan[i + 1]))
                    .then_some(self.space.centers_hz[i])
            })
            .collect();
        self.space
            .assert_scan_len_named(&landscape.consonance_field_level, "growth_level_scan");
        self.space
            .assert_scan_len_named(&landscape.harmonicity01, "harmonicity_state_scan");
        FloraReport {
            pace: self.pace,
            temporal: self.temporal.as_ref(),
            acoustic_observation: Some(&self.acoustic_observation),
            respond: self.respond.as_ref(),
            growth_level_scan: &self.growth_level_scan,
            harmonicity_state_scan: &self.harmonicity_state_scan,
            target_partials: self
                .lanes
                .iter()
                .filter(|l| l.mode == Mode::Current && l.target > 0.0)
                .map(|l| (l.target_position.exp2(), l.target.powi(2), l.tonal))
                .collect(),
            lanes: self
                .lanes
                .iter()
                .enumerate()
                .filter(|(_, l)| l.age_samples > 0)
                .map(|(slot, l)| FloraLaneReport {
                    slot,
                    state: match l.mode {
                        Mode::Free => "free",
                        Mode::Current => "current",
                        Mode::Retiring => "retiring",
                    },
                    frequency_hz: l.position.exp2(),
                    target_mass: l.target.powi(2),
                    timing: Some(FloraLaneTimingReport {
                        generation: l.generation,
                        pending_since: l.pending_since,
                        forced_reason: l.forced_reason,
                    }),
                    age_sec: l.age_samples as f64 / self.fs as f64,
                })
                .collect(),
            time_sec,
            age_sec: self.rendered_samples as f64 / self.fs as f64,
            flora_id: self.id,
            extent_hz: self.extent_hz,
            frequency_hz: &self.space.centers_hz,
            biomass_mass_scan: &self.biomass_mass_scan,
            resource_state_scan: &self.resource_state_scan,
            biomass_consumption_scan: &self.biomass_consumption_scan,
            external_consumption_scan: &self.external_consumption_scan,
            external_observed_frame: self.external_observed_frame,
            external_current: self.external_current,
            total_biomass_mass: self.biomass_mass_scan.iter().sum(),
            latent_peaks_hz,
            births: self.births,
            retirements: self.retirements,
            flapping: self.flapping,
            tonal_energy: self.energy[0],
            residual_energy: self.energy[1],
            rendered_partials: self
                .lanes
                .iter()
                .filter(|lane| lane.power_sum > 0.0)
                .map(|lane| {
                    (
                        lane.position.exp2(),
                        lane.power_sum / self.hop as f32,
                        lane.tonal,
                    )
                })
                .collect(),
        }
    }
}

#[cfg(test)]
#[path = "flora_pace_tests.rs"]
mod pace_tests;

#[cfg(test)]
#[path = "flora_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "flora_heard_tests.rs"]
mod heard_tests;

#[cfg(test)]
#[path = "flora_pitch_tests.rs"]
mod pitch_tests;
