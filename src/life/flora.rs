//! Per-instance field dynamics and cell-based stochastic additive synthesis.

use std::f32::consts::{PI, TAU};
use std::ops::Range;

use serde::Serialize;

use crate::core::erb::{erb_bw_hz, erb_to_hz, hz_to_erb};
use crate::core::float::{finite_or, sanitize_nonnegative_finite, sanitize01, unit_gaussian};
use crate::core::landscape::Landscape;
use crate::core::log2space::{Log2Space, sample_scan_linear_log2};
use crate::scenario::{FloraPatch, FloraSeeding, FloraSpec, FloraTone};
#[cfg(feature = "simd-wide")]
use wide::f32x8;

#[path = "flora_metric.rs"]
mod metric;
use metric::MetricState;
pub(crate) use metric::TemporalObservation;

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
const DEAD_MASS: f32 = 0.001;

#[inline(always)]
fn mul_add_fast(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(any(target_feature = "fma", target_feature = "neon"))]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(any(target_feature = "fma", target_feature = "neon")))]
    {
        a * b + c
    }
}

fn random_unit(state: &mut u64) -> f32 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)) as f32
}

fn noise_draw(state: &mut u64) -> [f32; 2] {
    let scale = 1.5_f32.sqrt();
    [
        (2.0 * random_unit(state) - 1.0) * scale,
        (2.0 * random_unit(state) - 1.0) * scale,
    ]
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

#[derive(Clone, Debug)]
struct Lane {
    mode: Mode,
    tonal: bool,
    noisy: bool,
    noise_index: usize,
    cell: Option<usize>,
    position: f32,
    target_position: f32,
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
    z: [f32; 2],
    w: [f32; 2],
    e: [f32; 2],
    a: f32,
    b: f32,
}

impl Lane {
    fn new(noisy: bool, position: f32) -> Self {
        Self {
            mode: Mode::Free,
            tonal: true,
            noisy,
            noise_index: 0,
            cell: None,
            position,
            target_position: position,
            sigma: 1.0,
            r: 0.0,
            q: 0.0,
            target: 0.0,
            power_sum: 0.0,
            age_samples: 0,
            generation: 0,
            window: 0..0,
            pending_since: None,
            forced_reason: None,
            retirement_decay: 0.0,
            slew_target: 0.0,
            slew_left: 0,
            z: [1.0, 0.0],
            w: [1.0, 0.0],
            e: [1.0, 0.0],
            a: 1.0,
            b: 0.0,
        }
    }

    fn retire(&mut self) {
        if self.mode != Mode::Free {
            self.mode = Mode::Retiring;
            self.target = 0.0;
            self.target_position = self.position;
            self.cell = None;
        }
    }

    fn slew(&mut self, fs: f32) {
        if self.mode == Mode::Retiring && self.retirement_decay > 0.0 {
            // Shift the metabolic amplitude exponential by the existing dead-mass
            // floor, giving a continuous finite tail and a reusable pool slot.
            let floor = DEAD_MASS.sqrt();
            self.r = ((self.r + floor) * self.retirement_decay - floor).max(0.0);
            self.q = self.r * self.r;
            if self.r == 0.0 {
                self.mode = Mode::Free;
            }
            return;
        }
        let duration = SLEW_SEC * fs;
        let step = 1.0 / duration;
        let target = if self.noisy {
            self.target * self.target
        } else {
            self.target
        };
        let current = if self.noisy { self.q } else { self.r };
        if target != self.slew_target || (self.slew_left == 0 && current != target) {
            self.slew_target = target;
            self.slew_left = ((target - current).abs() * duration).ceil() as usize;
        }
        // Pin the deadline rather than letting accumulated rounding add one sample.
        let value = if self.slew_left > 0 {
            self.slew_left -= 1;
            if self.slew_left == 0 {
                target
            } else {
                target - (target - current).signum() * step * self.slew_left as f32
            }
        } else {
            target
        };
        if self.noisy {
            self.q = value;
            self.r = value.sqrt();
        } else {
            self.r = value;
            self.q = value * value;
        }
        if self.mode == Mode::Retiring && self.r == 0.0 {
            self.mode = Mode::Free;
        }
    }

    fn atom(&mut self, n: [f32; 2]) -> f32 {
        let x = mul_add_fast(self.w[0], self.z[0], -self.w[1] * self.z[1]);
        self.z[1] = mul_add_fast(self.w[1], self.z[0], self.w[0] * self.z[1]);
        self.z[0] = x;
        self.e[0] = mul_add_fast(self.a, self.e[0], self.b * n[0]);
        self.e[1] = mul_add_fast(self.a, self.e[1], self.b * n[1]);
        mul_add_fast(self.e[0], self.z[0], -self.e[1] * self.z[1])
    }
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
pub(crate) struct Flora {
    pub(crate) id: u64,
    spec: FloraSpec,
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
            id,
            spec,
            acoustic_observation: TemporalObservation::default(),
            temporal: spec
                .metric
                .then(|| MetricState::new(scenario_seed ^ id, fs, hop)),
            extent_hz,
            space: landscape.space.clone(),
            range: 0..0,
            fs,
            hop,
            substeps: 1,
            pace: 1.0,
            diffusion: 0.0,
            rng: scenario_seed ^ id.wrapping_mul(0xD6E8_FEB8_6659_FD93),
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
        flora.space.assert_scan_len_named(
            &landscape.consonance_density_mass,
            "consonance_density_mass",
        );
        if !flora.range.is_empty() {
            for i in flora.range.clone() {
                flora.biomass_mass_scan[i] = match seeding {
                    FloraSeeding::Uniform => 1.0,
                    FloraSeeding::Consonance => {
                        finite_or(landscape.consonance_density_mass[i], 0.0).max(0.0)
                    }
                };
            }
            if flora.biomass_mass_scan.iter().sum::<f32>() == 0.0 {
                flora.biomass_mass_scan[flora.range.clone()].fill(1.0);
            }
            for i in flora.range.clone() {
                flora.biomass_mass_scan[i] *= 0.9 + 0.2 * random_unit(&mut flora.rng);
            }
            // Divide before scaling so tiny seeding weights cannot overflow; weights that
            // underflowed to zero during the perturbation fall back to uniform.
            let total: f32 = flora.biomass_mass_scan[flora.range.clone()].iter().sum();
            let uniform = INITIAL_BIOMASS / flora.range.len() as f32;
            for i in flora.range.clone() {
                flora.biomass_mass_scan[i] = if total > 0.0 {
                    flora.biomass_mass_scan[i] / total * INITIAL_BIOMASS
                } else {
                    uniform
                };
            }
        }
        flora.assert_scans();
        flora.gain.to(1.0, spec.fade_sec, fs);
        flora
    }

    pub(crate) fn apply_patch(&mut self, patch: &FloraPatch) {
        if let Some(amp) = patch.amp {
            self.spec.amp = amp;
            self.amp.to(amp, SLEW_SEC, self.fs);
        }
    }

    pub(crate) fn release(&mut self, fade_sec: f32) {
        if !self.releasing {
            self.releasing = true;
            self.gain.to(0.0, fade_sec, self.fs);
            if self.temporal.is_none() && self.gain.value == 0.0 {
                for lane in &mut self.lanes {
                    lane.mode = Mode::Free;
                    lane.r = 0.0;
                    lane.q = 0.0;
                    lane.target = 0.0;
                }
            }
        }
    }

    pub(crate) fn release_at_scenario_end(&mut self) {
        self.release(self.spec.fade_sec);
    }

    pub(crate) fn is_finished(&self) -> bool {
        self.releasing
            && self.gain.value == 0.0
            && self.lanes.iter().all(|lane| lane.mode == Mode::Free)
    }

    pub(crate) fn on_space_change(&mut self, space: &Log2Space) {
        if !self.biomass_mass_scan.is_empty() {
            self.assert_scans();
        }
        let old_space = std::mem::replace(&mut self.space, space.clone());
        let old_v = std::mem::replace(&mut self.biomass_mass_scan, vec![0.0; space.n_bins()]);
        let old_u = std::mem::replace(&mut self.resource_state_scan, vec![0.0; space.n_bins()]);
        let was_dormant = self.range.is_empty();
        let start = space.centers_hz.partition_point(|&f| f < self.extent_hz.0);
        let end = space.centers_hz.partition_point(|&f| f <= self.extent_hz.1);
        self.range = start..end;
        self.biomass_consumption_scan = vec![0.0; space.n_bins()];
        self.external_consumption_scan = vec![0.0; space.n_bins()];
        self.external_observed_frame = None;
        self.external_current = false;
        self.scratch_scan = vec![0.0; space.n_bins()];
        self.erb_scan = space.centers_hz.iter().map(|&f| hz_to_erb(f)).collect();
        self.assert_scans();
        self.consumption_rows.clear();
        self.consumption_weights.clear();
        if !self.range.is_empty() {
            self.resource_state_scan[self.range.clone()].fill(1.0);
            if !was_dormant && !old_v.is_empty() {
                old_space.assert_scan_len_named(&old_v, "biomass_mass_scan");
                old_space.assert_scan_len_named(&old_u, "resource_state_scan");
                let mass_scale = old_space.bins_per_oct as f32 / space.bins_per_oct as f32;
                for i in self.range.clone() {
                    let f = space.centers_hz[i];
                    if f >= old_space.centers_hz[0] && f <= *old_space.centers_hz.last().unwrap() {
                        self.biomass_mass_scan[i] =
                            sample_scan_linear_log2(&old_space, &old_v, f).max(0.0) * mass_scale;
                        self.resource_state_scan[i] =
                            sanitize01(sample_scan_linear_log2(&old_space, &old_u, f));
                    }
                }
            } else {
                self.biomass_mass_scan[self.range.clone()]
                    .fill(INITIAL_BIOMASS / self.range.len() as f32);
            }
            for i in self.range.clone() {
                let lo = self
                    .erb_scan
                    .partition_point(|&x| x < self.erb_scan[i] - 1.5)
                    .max(self.range.start);
                let hi = self
                    .erb_scan
                    .partition_point(|&x| x <= self.erb_scan[i] + 1.5)
                    .min(self.range.end);
                let first = self.consumption_weights.len();
                self.consumption_weights.extend(
                    (lo..hi).map(|j| unit_gaussian(self.erb_scan[j] - self.erb_scan[i], 0.5)),
                );
                let weights = &mut self.consumption_weights[first..];
                let total: f32 = weights.iter().sum();
                for weight in weights.iter_mut() {
                    *weight /= total;
                }
                self.consumption_rows
                    .push((lo, first..self.consumption_weights.len()));
            }
        }
        self.diffusion = 0.5 * (10.0 * space.bins_per_oct as f32 / 1200.0).powi(2);
        self.substeps = ((self.hop as f32 / self.fs)
            * self.pace
            * (GROWTH_RATE / 0.05).max(self.diffusion / 0.25))
        .ceil()
        .max(1.0) as usize;

        let mut old_lanes = std::mem::take(&mut self.lanes);
        self.cells = partition_erb_cells(space, &self.erb_scan, self.extent_hz)
            .into_iter()
            .map(|cell| Cell {
                lo_erb: cell.lo_erb,
                hi_erb: cell.hi_erb,
                bins: cell.bins,
                floor: 0.0,
                excess: 0.0,
                tonal_mass: 0.0,
                residual_mass: 0.0,
                peak: None,
            })
            .collect();
        self.tonal_len = 2 * self.cells.len();
        self.residual_end = self.tonal_len + self.cells.len();
        self.lanes = (0..self.tonal_len)
            .map(|_| {
                Lane::new(
                    self.spec.tone == FloraTone::Resonant,
                    self.extent_hz.0.log2(),
                )
            })
            .collect();
        for cell in &self.cells {
            let position = erb_to_hz(0.5 * (cell.lo_erb + cell.hi_erb)).log2();
            let mut lane = Lane::new(true, position);
            lane.tonal = false;
            lane.mode = Mode::Current;
            lane.e = noise_draw(&mut self.rng);
            self.lanes.push(lane);
        }
        for mut lane in old_lanes.drain(..) {
            if lane.mode == Mode::Free || (!lane.tonal && lane.r == 0.0) {
                continue;
            }
            if lane.tonal && !self.range.is_empty() {
                lane.cell = None;
            } else {
                lane.retire();
            }
            self.lanes.push(lane);
        }
        self.noisy_len = 0;
        for lane in &mut self.lanes {
            if lane.noisy {
                lane.noise_index = self.noisy_len;
                self.noisy_len += 1;
            }
        }
        if let Some(temporal) = &mut self.temporal {
            temporal.resize(self.cells.len(), self.lanes.len(), space.n_bins());
            for lane in &mut self.lanes {
                lane.window = 0..0;
                lane.pending_since = None;
            }
        }
        self.order = Vec::with_capacity(self.lanes.len());
        self.noise = vec![0.0; 2 * self.noisy_len * self.hop];
        self.assert_scans();
    }

    fn assert_scans(&self) {
        for (scan, name) in [
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

    pub(crate) fn observe_external(&mut self, landscape: &Landscape, frame: Option<u64>) {
        self.external_current = false;
        let Some(frame) = frame else {
            return;
        };
        let Some(power) = landscape
            .flora_power
            .iter()
            .find(|s| s.id == self.id && s.generation == 0)
            .and_then(|s| s.power_scan.as_ref())
        else {
            return;
        };
        self.space
            .assert_scan_len_named(power, "external_power_scan");
        self.external_current = true;
        if self.external_observed_frame == Some(frame) {
            return;
        }
        // Fixed author unit: nominal sine amplitude 0.03 has mean square 0.00045.
        let scale = self.space.bins_per_oct as f32 / REFERENCE_DENSITY / 0.00045;
        for (row, (lo, weights)) in self.consumption_rows.iter().enumerate() {
            self.external_consumption_scan[self.range.start + row] = self.consumption_weights
                [weights.clone()]
            .iter()
            .enumerate()
            .map(|(j, &w)| w * sanitize_nonnegative_finite(power[lo + j]))
            .sum::<f32>()
                * scale;
        }
        self.external_observed_frame = Some(frame);
    }

    pub(crate) fn advance(&mut self, landscape: &Landscape) {
        self.assert_scans();
        self.space
            .assert_scan_len_named(&landscape.consonance_field_level, "consonance_field_level");
        self.space
            .assert_scan_len_named(&landscape.harmonicity01, "harmonicity01");
        assert_eq!(
            (self.space.fmin, self.space.step()),
            (landscape.space.fmin, landscape.space.step()),
            "Flora space change was not applied"
        );
        if self.range.is_empty() {
            return;
        }
        self.substeps = ((self.hop as f32 / self.fs)
            * self.pace
            * (GROWTH_RATE / 0.05).max(self.diffusion / 0.25))
        .ceil()
        .max(1.0) as usize;
        let substeps = self.substeps;
        let dt = self.hop as f32 / self.fs * self.pace / substeps as f32;
        let bins_per_oct = self.space.bins_per_oct as f32;
        let allee_mass = ALLEE_DENSITY / bins_per_oct;
        for _ in 0..substeps {
            let maximum = self.biomass_mass_scan[self.range.clone()]
                .iter()
                .copied()
                .fold(0.0, f32::max);
            for i in self.range.clone() {
                let left = i.saturating_sub(1).max(self.range.start);
                let right = (i + 1).min(self.range.end - 1);
                let v = self.biomass_mass_scan[i];
                let u = self.resource_state_scan[i];
                let growth = GROWTH_RATE
                    * v
                    * (sanitize01(landscape.consonance_field_level[i]) * u * v / (v + allee_mass)
                        - METABOLISM);
                let spread = self.diffusion
                    * (self.biomass_mass_scan[left] - 2.0 * v + self.biomass_mass_scan[right]);
                let seeds = u
                    * (BACKGROUND_SEED_DENSITY
                        + SEED_RAIN_DENSITY * sanitize01(landscape.harmonicity01[i]))
                    / bins_per_oct;
                let next = v + dt * (growth + spread + seeds);
                debug_assert!(next >= -1e-6 * maximum, "negative Flora biomass: {next}");
                self.scratch_scan[i] = next.max(0.0);
            }
            std::mem::swap(&mut self.biomass_mass_scan, &mut self.scratch_scan);
            self.advance_resource(dt);
        }
    }

    fn advance_resource(&mut self, dt: f32) {
        self.assert_scans();
        let bins_per_oct = self.space.bins_per_oct as f32;
        let renewal = 1.0 / RENEWAL_SEC;
        for (row, (lo, weights)) in self.consumption_rows.iter().enumerate() {
            let i = self.range.start + row;
            let biomass_consumption_density: f32 = self.consumption_weights[weights.clone()]
                .iter()
                .enumerate()
                .map(|(j, &weight)| weight * self.biomass_mass_scan[lo + j])
                .sum::<f32>()
                * bins_per_oct
                / REFERENCE_DENSITY;
            self.biomass_consumption_scan[i] = biomass_consumption_density;
            // Unknown initial consumption cannot be interpreted as silence.
            if self.external_observed_frame.is_none() {
                continue;
            }
            let a = renewal
                + (biomass_consumption_density + self.external_consumption_scan[i])
                    / CONSUMPTION_SEC;
            let equilibrium = renewal / a;
            self.resource_state_scan[i] =
                equilibrium + (self.resource_state_scan[i] - equilibrium) * (-a * dt).exp();
        }
    }

    fn decompose(&mut self) {
        self.assert_scans();
        let p = &self.biomass_mass_scan;
        let maximum = p[self.range.clone()].iter().copied().fold(0.0, f32::max);
        for cell in &mut self.cells {
            cell.peak = None;
            cell.floor = p[cell.bins.clone()]
                .iter()
                .copied()
                .fold(f32::INFINITY, f32::min);
            cell.excess = cell.bins.clone().map(|i| p[i] - cell.floor).sum();
            cell.tonal_mass = 0.0;
            cell.residual_mass = cell.excess;
        }
        let mut first = self.range.start;
        while first < self.range.end {
            let mut end = first + 1;
            while end < self.range.end && p[end] == p[first] {
                end += 1;
            }
            let peak = (first == self.range.start || p[first - 1] < p[first])
                && (end == self.range.end || p[end] < p[first])
                && (first != self.range.start || end != self.range.end);
            if peak {
                let bin = first + (end - first - 1) / 2;
                let ci = self.cells.partition_point(|cell| cell.bins.end <= bin);
                let cell = &mut self.cells[ci];
                if cell.peak.as_ref().is_none_or(|old| p[bin] > p[old.bin]) {
                    let mut delta = 0.0;
                    let mut sigma = 1.0;
                    if end - first == 1 && bin > self.range.start && bin + 1 < self.range.end {
                        let l = p[bin - 1].max(1e-12).ln();
                        let c = p[bin].max(1e-12).ln();
                        let r = p[bin + 1].max(1e-12).ln();
                        let curvature = l - 2.0 * c + r;
                        if curvature < -1e-9 {
                            delta = ((l - r) / (2.0 * curvature)).clamp(-0.5, 0.5);
                            sigma = (-1.0 / curvature).sqrt().clamp(0.5, 4.0);
                        }
                    }
                    // Stop at the left bin of a minimum plateau, assigned to its left basin.
                    let mut left = first;
                    while left > self.range.start && p[left - 1] <= p[left] {
                        left -= 1;
                    }
                    if left > self.range.start {
                        left += 1;
                    }
                    let mut right = end - 1;
                    while right + 1 < self.range.end && p[right + 1] <= p[right] {
                        right += 1;
                    }
                    while right > end - 1 && p[right - 1] == p[right] {
                        right -= 1;
                    }
                    let window = ((bin as f32 - 2.0 * sigma).ceil() as usize)
                        .max(left)
                        .max(cell.bins.start)
                        ..(((bin as f32 + 2.0 * sigma).floor() as usize + 1)
                            .min(right + 1)
                            .min(cell.bins.end));
                    let position_bin = if end - first > 1 {
                        0.5 * (first + end - 1) as f32
                    } else {
                        bin as f32 + delta
                    };
                    cell.peak = Some(Peak {
                        bin,
                        position: self.space.centers_log2[0] + position_bin * self.space.step(),
                        sigma,
                        mass: window.clone().map(|i| p[i] - cell.floor).sum(),
                        window,
                        enter: p[bin] >= 0.02 * maximum,
                        stay: p[bin] >= 0.01 * maximum,
                        claimed: None,
                        suppressed: false,
                        visited: false,
                    });
                }
            }
            first = end;
        }
    }

    fn update_tones(&mut self) {
        self.order.clear();
        self.order.extend(
            (0..self.lanes.len())
                .filter(|&i| self.lanes[i].tonal && self.lanes[i].mode == Mode::Current),
        );
        self.order.sort_unstable_by(|&a, &b| {
            self.lanes[b]
                .target
                .total_cmp(&self.lanes[a].target)
                .then(a.cmp(&b))
        });
        for &i in &self.order {
            let lane = &mut self.lanes[i];
            let nearest = self
                .cells
                .iter()
                .enumerate()
                .filter_map(|(c, cell)| {
                    let peak = cell.peak.as_ref()?;
                    let distance = (peak.position - lane.position).abs();
                    (peak.stay && peak.claimed.is_none() && distance <= 50.0 / 1200.0)
                        .then_some((c, distance, peak.bin))
                })
                .min_by(|a, b| a.1.total_cmp(&b.1).then(a.2.cmp(&b.2)));
            if let Some((c, _, _)) = nearest {
                self.cells[c].peak.as_mut().unwrap().claimed = Some(i);
                lane.cell = Some(c);
            } else {
                self.retirements += 1;
                self.flapping += u64::from(lane.age_samples as f32 <= SLEW_SEC * self.fs);
                lane.retire();
            }
        }
        // Resolve adjacent conflicts in position order, independent of cell traversal.
        loop {
            self.order.clear();
            self.order.extend(
                (0..self.lanes.len())
                    .filter(|&i| self.lanes[i].tonal && self.lanes[i].mode == Mode::Current),
            );
            self.order.sort_unstable_by(|&a, &b| {
                let pa = self.cells[self.lanes[a].cell.unwrap()]
                    .peak
                    .as_ref()
                    .unwrap();
                let pb = self.cells[self.lanes[b].cell.unwrap()]
                    .peak
                    .as_ref()
                    .unwrap();
                pa.position.total_cmp(&pb.position).then(a.cmp(&b))
            });
            let conflict = self.order.windows(2).find_map(|pair| {
                let a = pair[0];
                let b = pair[1];
                let pa = self.cells[self.lanes[a].cell.unwrap()]
                    .peak
                    .as_ref()
                    .unwrap();
                let pb = self.cells[self.lanes[b].cell.unwrap()]
                    .peak
                    .as_ref()
                    .unwrap();
                if hz_to_erb(pb.position.exp2()) - hz_to_erb(pa.position.exp2()) >= 0.5 {
                    return None;
                }
                Some(if pa.mass < pb.mass || (pa.mass == pb.mass && a > b) {
                    a
                } else {
                    b
                })
            });
            let Some(i) = conflict else {
                break;
            };
            let peak = self.cells[self.lanes[i].cell.unwrap()]
                .peak
                .as_mut()
                .unwrap();
            peak.claimed = None;
            peak.suppressed = true;
            self.retirements += 1;
            self.flapping += u64::from(self.lanes[i].age_samples as f32 <= SLEW_SEC * self.fs);
            self.lanes[i].retire();
        }
        loop {
            let best = self
                .cells
                .iter()
                .enumerate()
                .filter_map(|(c, cell)| {
                    let peak = cell.peak.as_ref()?;
                    (peak.enter && peak.claimed.is_none() && !peak.suppressed && !peak.visited)
                        .then_some(c)
                })
                .max_by(|&a, &b| {
                    let pa = self.cells[a].peak.as_ref().unwrap();
                    let pb = self.cells[b].peak.as_ref().unwrap();
                    pa.mass.total_cmp(&pb.mass).then(pb.bin.cmp(&pa.bin))
                });
            let Some(c) = best else {
                break;
            };
            let position = self.cells[c].peak.as_ref().unwrap().position;
            self.cells[c].peak.as_mut().unwrap().visited = true;
            let blocked = self.lanes.iter().any(|lane| {
                lane.tonal
                    && lane.mode == Mode::Current
                    && (hz_to_erb(position.exp2())
                        - hz_to_erb(
                            self.cells[lane.cell.unwrap()]
                                .peak
                                .as_ref()
                                .unwrap()
                                .position
                                .exp2(),
                        ))
                    .abs()
                        < 0.5
            });
            if blocked {
                self.cells[c].peak.as_mut().unwrap().suppressed = true;
                continue;
            }
            let Some(i) = self.lanes[..self.tonal_len]
                .iter()
                .position(|lane| lane.mode == Mode::Free)
            else {
                break;
            };
            let mut lane = Lane::new(self.spec.tone == FloraTone::Resonant, position);
            lane.noise_index = self.lanes[i].noise_index;
            lane.retirement_decay = (-0.5 * GROWTH_RATE * self.pace * METABOLISM / self.fs).exp();
            lane.mode = Mode::Current;
            lane.cell = Some(c);
            let angle = TAU * random_unit(&mut self.rng);
            let (sin, cos) = angle.sin_cos();
            lane.z = [cos, sin];
            if lane.noisy {
                lane.e = noise_draw(&mut self.rng);
            }
            lane.generation = self.births + 1;
            self.lanes[i] = lane;
            self.births += 1;
            self.cells[c].peak.as_mut().unwrap().claimed = Some(i);
        }
        for lane in &mut self.lanes {
            if lane.tonal && lane.mode == Mode::Current {
                let peak = self.cells[lane.cell.unwrap()].peak.as_ref().unwrap();
                lane.target = peak.mass.sqrt();
                lane.target_position = peak.position;
                lane.sigma = peak.sigma;
                lane.window = peak.window.clone();
                lane.pending_since = None;
            }
        }
    }

    fn residual_targets(&mut self) {
        self.assert_scans();
        for (c, cell) in self.cells.iter_mut().enumerate() {
            let held = cell.peak.as_ref().filter(|peak| peak.claimed.is_some());
            cell.tonal_mass = held.map_or(0.0, |peak| peak.mass);
            cell.residual_mass = (cell.excess - cell.tonal_mass).max(0.0);
            let lane = &mut self.lanes[self.tonal_len + c];
            lane.target = if self.spec.residual {
                (cell.residual_mass - DEAD_MASS).max(0.0).sqrt()
            } else {
                0.0
            };
            if cell.residual_mass > 0.0 {
                let mut moment = 0.0;
                let mut mass = 0.0;
                for i in cell.bins.clone() {
                    if held.is_some_and(|peak| peak.window.contains(&i)) {
                        continue;
                    }
                    let x = self.biomass_mass_scan[i] - cell.floor;
                    moment += x * self.space.centers_log2[i];
                    mass += x;
                }
                if mass > 0.0 {
                    lane.target_position = (moment / mass)
                        .clamp(erb_to_hz(cell.lo_erb).log2(), erb_to_hz(cell.hi_erb).log2());
                }
            }
        }
        // Bound all targets together; slews and retiring amplitudes keep their history.
        let target_power: f32 = self.lanes.iter().map(|lane| lane.target.powi(2)).sum();
        let scale = target_power.max(1.0).sqrt().recip();
        for lane in &mut self.lanes {
            lane.target *= scale;
        }
    }

    pub(crate) fn render_add(&mut self, presentation: &mut [f32], habitat: &mut [f32]) {
        self.render::<true>(presentation, habitat);
    }

    fn render<const SIMD: bool>(&mut self, presentation: &mut [f32], habitat: &mut [f32]) {
        self.assert_scans();
        let before = (
            self.births,
            self.retirements,
            self.lanes.iter().filter(|l| l.mode != Mode::Free).count(),
        );
        assert_eq!(presentation.len(), self.hop);
        assert_eq!(habitat.len(), self.hop);
        if self.temporal.is_some() {
            self.metric_allocate(false, 0);
        } else if !(self.range.is_empty() || self.releasing && self.gain.value == 0.0) {
            self.decompose();
            if self.releasing {
                for lane in &mut self.lanes {
                    if lane.tonal && lane.mode == Mode::Current {
                        lane.retire();
                        self.retirements += 1;
                    }
                }
            } else {
                self.update_tones();
            }
            self.residual_targets();
        }
        let decay = (-0.5 * GROWTH_RATE * self.pace * METABOLISM / self.fs).exp();
        for lane in &mut self.lanes {
            lane.retirement_decay = decay;
            lane.power_sum = 0.0;
            let norm = lane.z[0].hypot(lane.z[1]);
            lane.z[0] /= norm;
            lane.z[1] /= norm;
            if lane.noisy && lane.mode != Mode::Retiring {
                let f = lane.target_position.exp2();
                let b = if lane.tonal {
                    (f * ((lane.sigma / self.space.bins_per_oct as f32).exp2() - 1.0))
                        .clamp(0.5, 0.5 * erb_bw_hz(f))
                } else {
                    erb_bw_hz(f)
                };
                lane.a = (-PI * b / self.fs).exp();
                lane.b = (1.0 - lane.a * lane.a).sqrt();
            }
        }
        // Include free noisy lanes, so lane indices define the stream even during turnover.
        for pair in self.noise.as_chunks_mut::<2>().0 {
            pair.copy_from_slice(&noise_draw(&mut self.rng));
        }
        self.habitat_pcm.fill(0.0);
        for t in 0..self.hop {
            if self
                .temporal
                .as_ref()
                .is_some_and(|m| m.active && m.crossings.iter().any(|c| c.tick == m.now + t as u64))
                && !self.releasing
                && !self.range.is_empty()
            {
                self.metric_allocate(true, t);
                for lane in &mut self.lanes {
                    lane.retirement_decay = decay;
                    if lane.noisy && lane.mode == Mode::Current {
                        let f = lane.target_position.exp2();
                        let b = if lane.tonal {
                            (f * ((lane.sigma / self.space.bins_per_oct as f32).exp2() - 1.0))
                                .clamp(0.5, 0.5 * erb_bw_hz(f))
                        } else {
                            erb_bw_hz(f)
                        };
                        lane.a = (-PI * b / self.fs).exp();
                        lane.b = (1.0 - lane.a * lane.a).sqrt();
                    }
                }
            }
            let amp = self.amp.tick() * self.gain.tick();
            let amp = if self.range.is_empty() { 0.0 } else { amp };
            for lane in &mut self.lanes {
                if self.releasing && self.gain.value == 0.0 {
                    lane.mode = Mode::Free;
                    lane.r = 0.0;
                    lane.q = 0.0;
                    lane.target = 0.0;
                } else {
                    lane.slew(self.fs);
                }
                if self.temporal.is_some() && lane.mode != Mode::Free {
                    lane.age_samples += 1;
                }
                lane.power_sum += (amp * lane.r).powi(2);
                if t % 64 == 0 || (self.temporal.is_some() && lane.age_samples == 1) {
                    let log2 = lane.position
                        + (lane.target_position - lane.position) * (t + 1) as f32 / self.hop as f32;
                    let (sin, cos) = (TAU * log2.exp2() / self.fs).sin_cos();
                    lane.w = [cos, sin];
                }
            }
            let noise = &self.noise[t * 2 * self.noisy_len..(t + 1) * 2 * self.noisy_len];
            let out = synthesize::<SIMD>(&mut self.lanes, noise, amp);
            if self.spec.routing.to_presentation {
                presentation[t] += out;
            }
            if self.spec.routing.to_habitat {
                habitat[t] += out;
                self.habitat_pcm[t] = out;
            }
        }
        for lane in &mut self.lanes {
            lane.position = lane.target_position;
            if self.temporal.is_none() && lane.mode != Mode::Free {
                lane.age_samples += self.hop;
            }
            self.energy[usize::from(!lane.tonal)] += lane.power_sum as f64 / self.fs as f64;
        }
        self.lifecycle_changed = before
            != (
                self.births,
                self.retirements,
                self.lanes.iter().filter(|l| l.mode != Mode::Free).count(),
            );
        self.rendered_samples += self.hop as u64;
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
        let timing = self.temporal.is_some() || self.acoustic_observation.observed_end.is_some();
        FloraReport {
            pace: self.pace,
            temporal: self.temporal.as_ref(),
            acoustic_observation: timing.then_some(&self.acoustic_observation),
            growth_level_scan: &landscape.consonance_field_level,
            harmonicity_state_scan: &landscape.harmonicity01,
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
                    timing: timing.then_some(FloraLaneTimingReport {
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

fn synthesize<const SIMD: bool>(lanes: &mut [Lane], noise: &[f32], amp: f32) -> f32 {
    #[cfg(feature = "simd-wide")]
    let mut first = 0;
    #[cfg(not(feature = "simd-wide"))]
    let first = 0;
    let mut out = 0.0;
    #[cfg(feature = "simd-wide")]
    if SIMD {
        while first + 8 <= lanes.len() {
            let block = &mut lanes[first..first + 8];
            let x = f32x8::from(std::array::from_fn(|i| block[i].z[0]));
            let y = f32x8::from(std::array::from_fn(|i| block[i].z[1]));
            let c = f32x8::from(std::array::from_fn(|i| block[i].w[0]));
            let s = f32x8::from(std::array::from_fn(|i| block[i].w[1]));
            let er = f32x8::from(std::array::from_fn(|i| block[i].e[0]));
            let ei = f32x8::from(std::array::from_fn(|i| block[i].e[1]));
            let a = f32x8::from(std::array::from_fn(|i| block[i].a));
            let b = f32x8::from(std::array::from_fn(|i| block[i].b));
            let nr = f32x8::from(std::array::from_fn(|i| {
                if block[i].noisy {
                    noise[2 * block[i].noise_index]
                } else {
                    0.0
                }
            }));
            let ni = f32x8::from(std::array::from_fn(|i| {
                if block[i].noisy {
                    noise[2 * block[i].noise_index + 1]
                } else {
                    0.0
                }
            }));
            let xr = c.mul_add(x, -(s * y));
            let yi = s.mul_add(x, c * y);
            let er = a.mul_add(er, b * nr);
            let ei = a.mul_add(ei, b * ni);
            let signal = er.mul_add(xr, -(ei * yi)).to_array();
            let xr = xr.to_array();
            let yi = yi.to_array();
            let er = er.to_array();
            let ei = ei.to_array();
            for i in 0..8 {
                block[i].z = [xr[i], yi[i]];
                block[i].e = [er[i], ei[i]];
                out = mul_add_fast(amp * block[i].r, signal[i], out);
            }
            first += 8;
        }
    }
    #[cfg(not(feature = "simd-wide"))]
    let _ = SIMD;
    for lane in &mut lanes[first..] {
        let n = if lane.noisy {
            [noise[2 * lane.noise_index], noise[2 * lane.noise_index + 1]]
        } else {
            [0.0; 2]
        };
        let signal = lane.atom(n);
        out = mul_add_fast(amp * lane.r, signal, out);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scenario::control::Routing;

    fn spec() -> FloraSpec {
        FloraSpec {
            amp: 0.4,
            fade_sec: 0.0,
            residual: true,
            ..FloraSpec::default()
        }
    }

    fn terrain() -> Landscape {
        Landscape::new(Log2Space::new(440.0, 880.0, 96))
    }

    fn flora(landscape: &Landscape) -> Flora {
        let mut flora = Flora::new(
            7,
            spec(),
            (440.0, 880.0),
            FloraSeeding::Uniform,
            42,
            landscape,
            2000.0,
            100,
        );
        // Legacy biomass-only fixtures have a known silent external environment.
        flora.external_observed_frame = Some(0);
        flora
    }

    fn uniform(f: &mut Flora) {
        f.biomass_mass_scan.fill(0.0);
        f.biomass_mass_scan[f.range.clone()].fill(1.0 / f.range.len() as f32);
    }

    fn normalize(f: &mut Flora) {
        let total: f32 = f.biomass_mass_scan.iter().sum();
        for p in &mut f.biomass_mass_scan {
            *p /= total;
        }
    }

    fn hill(f: &mut Flora, bin: usize, height: f32) {
        for (offset, weight) in [(-2, 0.1), (-1, 0.5), (0, 1.0), (1, 0.5), (2, 0.1)] {
            let i = (bin as isize + offset) as usize;
            f.biomass_mass_scan[i] = height * weight;
        }
    }

    fn allocate(f: &mut Flora) {
        f.decompose();
        f.update_tones();
        f.residual_targets();
    }

    fn hop(f: &mut Flora, l: &Landscape) {
        let mut p = vec![0.0; f.hop];
        let mut h = vec![0.0; f.hop];
        f.advance(l);
        f.render_add(&mut p, &mut h);
        assert!(p.iter().chain(&h).all(|v| v.is_finite()));
    }

    fn mass_accounting(f: &Flora) {
        let mut expected = 0.0;
        for cell in &f.cells {
            assert!((cell.tonal_mass + cell.residual_mass - cell.excess).abs() < 1e-7);
            if cell.peak.as_ref().is_none_or(|p| p.claimed.is_none()) {
                assert_eq!(cell.tonal_mass, 0.0);
                assert_eq!(cell.residual_mass, cell.excess);
            }
            expected += cell.tonal_mass;
            if f.spec.residual {
                expected += (cell.residual_mass - DEAD_MASS).max(0.0);
            }
        }
        let actual: f32 = f
            .lanes
            .iter()
            .filter(|lane| lane.mode == Mode::Current)
            .map(|lane| lane.target.powi(2))
            .sum();
        assert!(
            (actual - expected.min(1.0)).abs() < 2e-7,
            "{actual} != {expected}"
        );
        assert!(actual <= 1.0 + 1e-6);
    }

    #[test]
    fn dynamics_large_hop_high_rate_preserves_nonnegativity_and_extent() {
        let mut l = Landscape::new(Log2Space::new(110.0, 1760.0, 96));
        l.consonance_field_level.fill(1.0);
        l.harmonicity01.fill(1.0);
        let mut f = Flora::new(
            1,
            spec(),
            (440.0, 880.0),
            FloraSeeding::Uniform,
            1,
            &l,
            2000.0,
            800,
        );
        f.external_observed_frame = Some(0);
        assert_eq!(f.substeps, 64);
        for _ in 0..100 {
            f.advance(&l);
            for (i, &v) in f.biomass_mass_scan.iter().enumerate() {
                assert!(v.is_finite() && v >= 0.0);
                assert!((0.0..=1.0).contains(&f.resource_state_scan[i]));
                if !f.range.contains(&i) {
                    assert_eq!(v, 0.0);
                    assert_eq!(f.resource_state_scan[i], 0.0);
                }
            }
        }
    }

    #[test]
    fn resource_recovers_without_biomass_and_falls_under_consumption() {
        let l = terrain();
        let mut f = flora(&l);
        f.biomass_mass_scan.fill(0.0);
        f.resource_state_scan[f.range.clone()].fill(0.2);
        for _ in 0..120 {
            f.advance_resource(1.0);
        }
        assert!(
            f.resource_state_scan[f.range.clone()]
                .iter()
                .all(|&u| (u - 1.0).abs() < 1e-5)
        );
        f.biomass_mass_scan[f.range.clone()].fill(100.0 / f.space.bins_per_oct as f32);
        f.advance_resource(1.0);
        assert!(
            f.resource_state_scan[f.range.clone()]
                .iter()
                .all(|&u| u < 0.01)
        );
        for _ in 0..120 {
            f.advance_resource(1.0);
            assert!(
                f.resource_state_scan
                    .iter()
                    .all(|&u| (0.0..=1.0).contains(&u))
            );
        }
        let before = f.resource_state_scan.clone();
        f.observe_rhythm(
            TemporalObservation::default(),
            0,
            crate::core::meter::MeterShaping {
                basin_hz: Some((0.5, 1.5)),
                ..Default::default()
            },
        );
        assert_eq!(f.resource_state_scan, before);
    }

    #[test]
    fn consumption_kernel_is_truncated_row_normalized_and_extent_local() {
        let l = Landscape::new(Log2Space::new(110.0, 1760.0, 96));
        let mut f = flora(&l);
        for (row, (lo, weights)) in f.consumption_rows.iter().enumerate() {
            let i = f.range.start + row;
            let weights = &f.consumption_weights[weights.clone()];
            assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
            for (j, &weight) in weights.iter().enumerate() {
                assert!(f.range.contains(&(lo + j)));
                let distance = f.erb_scan[lo + j] - f.erb_scan[i];
                assert!(distance.abs() <= 1.5 + 1e-6);
                assert!(weight > 0.0);
                let centre_weight = weights[i - lo];
                assert!((weight / centre_weight - unit_gaussian(distance, 0.5)).abs() < 1e-6);
            }
        }
        f.biomass_mass_scan[f.range.clone()].fill(REFERENCE_DENSITY / f.space.bins_per_oct as f32);
        f.advance_resource(1.0);
        let a = 1.0 / RENEWAL_SEC + 1.0 / CONSUMPTION_SEC;
        let equilibrium = (1.0 / RENEWAL_SEC) / a;
        let expected = equilibrium + (1.0 - equilibrium) * (-a).exp();
        assert!(
            f.resource_state_scan[f.range.clone()]
                .iter()
                .all(|&u| (u - expected).abs() < 1e-6)
        );
    }

    #[test]
    fn flat_terrain_biomass_stays_below_twenty_over_120_seconds() {
        let mut l = terrain();
        l.consonance_field_level.fill(0.5);
        l.harmonicity01.fill(0.0);
        for total in [0.05, 5.0] {
            let mut f = flora(&l);
            f.biomass_mass_scan[f.range.clone()].fill(total / f.range.len() as f32);
            for _ in 0..2400 {
                f.advance(&l);
                let biomass: f32 = f.biomass_mass_scan.iter().sum();
                assert!(
                    biomass.is_finite() && biomass < 20.0,
                    "initial {total}, biomass {biomass}"
                );
            }
        }
    }

    #[test]
    fn dynamics_reflecting_spread_conserves_mass_except_metabolism_at_both_ends() {
        let l = terrain();
        let mut results = Vec::new();
        for right in [false, true] {
            let mut f = flora(&l);
            f.hop = 1;
            f.fs = 200.0;
            f.biomass_mass_scan.fill(0.0);
            let edge = if right {
                f.range.end - 1
            } else {
                f.range.start
            };
            f.biomass_mass_scan[edge] = 1.0;
            let dt = f.hop as f32 / f.fs;
            let expected = 1.0 - dt * (f.diffusion + GROWTH_RATE * METABOLISM);
            for step in 0..200 {
                // Zero resource suppresses seeds for this single-substep hop.
                f.resource_state_scan.fill(0.0);
                f.advance(&l);
                if step == 0 {
                    assert!((f.biomass_mass_scan[edge] - expected).abs() < 2e-6);
                }
                let expected_total = (1.0 - dt * GROWTH_RATE * METABOLISM).powi(step + 1);
                assert!((f.biomass_mass_scan.iter().sum::<f32>() - expected_total).abs() < 2e-6);
            }
            results.push(f.biomass_mass_scan[edge]);
        }
        assert!((results[0] - results[1]).abs() < 2e-6);
    }

    #[test]
    fn substep_count_uses_smallest_rate_and_spread_bound() {
        let l = terrain();
        let mut f = Flora::new(
            7,
            spec(),
            (440.0, 880.0),
            FloraSeeding::Uniform,
            42,
            &l,
            48000.0,
            512,
        );
        // Default rate: ceil(8 * (512 / 48000) / 0.05) = 2; spread needs only 1.
        assert_eq!(f.substeps, 2);
        for (rate, expected) in [(4.0, 1), (8.0, 2)] {
            f.observe_rhythm(
                TemporalObservation::default(),
                0,
                crate::core::meter::MeterShaping {
                    basin_hz: Some((rate / 4.0, rate / 4.0)),
                    ..Default::default()
                },
            );
            f.advance(&l);
            assert_eq!(f.substeps, expected);
            let hop_dt = f.hop as f32 / f.fs;
            let valid = |n| {
                rate * hop_dt / n as f32 <= 0.05 && f.diffusion * f.pace * hop_dt / n as f32 <= 0.25
            };
            assert!(valid(f.substeps));
            assert!(f.substeps == 1 || !valid(f.substeps - 1));
        }
        f.on_space_change(&Log2Space::new(440.0, 880.0, 1200));
        assert_eq!(f.substeps, 3);
    }

    #[test]
    fn biomass_growth_uses_harmonic_seed_rain_and_not_amplitude() {
        let mut l = terrain();
        let mut a = flora(&l);
        a.biomass_mass_scan.fill(0.0);
        let mut b = a.clone();
        a.spec.amp = 0.0;
        b.spec.amp = 1.0;
        l.harmonicity01[33] = 1.0;
        a.advance(&l);
        b.advance(&l);
        assert_eq!(a.biomass_mass_scan, b.biomass_mass_scan);
        assert_eq!(a.resource_state_scan, b.resource_state_scan);
        assert!(a.biomass_mass_scan[33] > 100.0 * a.biomass_mass_scan[73]);
    }

    #[test]
    fn target_scaling_bounds_tonal_and_residual_amplitudes_together() {
        let l = terrain();
        let mut f = flora(&l);
        f.biomass_mass_scan.fill(0.001);
        hill(&mut f, 33, 100.0);
        // An increasing cell has residual mass without another tonal candidate.
        for (offset, i) in f.cells[3].bins.clone().enumerate() {
            f.biomass_mass_scan[i] = 10.0 + offset as f32;
        }
        let mut p = vec![0.0; f.hop];
        let mut h = p.clone();
        f.render_add(&mut p, &mut h);
        assert!(f.cells.iter().map(|c| c.excess).sum::<f32>() > 1.0);
        assert!(f.lanes[..f.tonal_len].iter().any(|l| l.target > 0.0));
        assert!(
            f.lanes[f.tonal_len..f.residual_end]
                .iter()
                .any(|l| l.target > 0.0)
        );
        let target_power: f32 = f.lanes.iter().map(|l| l.target.powi(2)).sum();
        assert!(
            (target_power - 1.0).abs() <= 1e-6,
            "target power {target_power}"
        );
        mass_accounting(&f);
    }

    #[test]
    fn seeding_is_reproducible_restricted_and_has_uniform_fallback() {
        let mut l = terrain();
        l.consonance_density_mass[31..35].fill(2.0);
        let make = |l: &Landscape| {
            Flora::new(
                7,
                spec(),
                (440.0, 880.0),
                FloraSeeding::Consonance,
                42,
                l,
                2000.0,
                100,
            )
        };
        let a = make(&l);
        let b = make(&l);
        assert_eq!(a.biomass_mass_scan, b.biomass_mass_scan);
        let total: f32 = a.biomass_mass_scan.iter().sum();
        assert!((total - INITIAL_BIOMASS).abs() < 1e-6, "{total}");
        assert_eq!(total, a.biomass_mass_scan[31..35].iter().sum::<f32>());
        assert!(
            a.resource_state_scan[a.range.clone()]
                .iter()
                .all(|&u| u == 1.0)
        );
        l.consonance_density_mass.fill(0.0);
        let f = make(&l);
        assert!(
            f.biomass_mass_scan[f.range.clone()]
                .iter()
                .all(|&p| p > 0.0)
        );
        assert_ne!(f.biomass_mass_scan[0], f.biomass_mass_scan[1]);
        let total: f32 = f.biomass_mass_scan.iter().sum();
        assert!((total - INITIAL_BIOMASS).abs() < 1e-6, "{total}");
        // Tiny weights next to zero bins must not overflow to Inf or NaN.
        l.consonance_density_mass[31..33].fill(1e-40);
        let tiny = make(&l);
        assert!(tiny.biomass_mass_scan.iter().all(|v| v.is_finite()));
        let total: f32 = tiny.biomass_mass_scan.iter().sum();
        assert!((total - INITIAL_BIOMASS).abs() < 1e-6, "{total}");
    }

    #[test]
    fn habituation_changes_neither_base_seeding_nor_base_driven_growth() {
        use crate::core::consonance_kernel::ConsonanceRepresentationParams;
        use crate::core::habituation::{HabituationField, HabituationParams};

        let mut base = terrain();
        let repr = ConsonanceRepresentationParams::default();
        let n = base.space.n_bins();
        for i in 0..n {
            base.consonance_field_score[i] = 0.5 + i as f32 / n as f32;
            base.consonance_field_level[i] = repr.level(base.consonance_field_score[i]);
            base.consonance_density_mass[i] = 0.1 + i as f32 / n as f32;
            base.harmonicity01[i] = (i % 11) as f32 / 10.0;
        }
        base.apply_habituation(&vec![0.0; n], repr.theta, &repr);
        let mut habituated = base.clone();
        let mut hab = HabituationField::new(&HabituationParams::default(), repr.theta, n);
        assert!(hab.is_enabled());
        let projection: Vec<_> = (0..n)
            .map(|i| if i < n / 2 { 10.0 } else { 0.01 })
            .collect();
        for _ in 0..200 {
            hab.advance_from_parts(&base.consonance_field_level, &projection, 0.05);
        }
        habituated.apply_habituation(hab.state(), repr.theta, &repr);
        assert_ne!(
            habituated.consonance_field_level,
            habituated.consonance_field_level_eff
        );
        assert_ne!(
            habituated.consonance_density_mass,
            habituated.consonance_density_mass_eff
        );
        assert_eq!(habituated.harmonicity01, base.harmonicity01);
        let make = |l: &Landscape| {
            Flora::new(
                7,
                spec(),
                (440.0, 880.0),
                FloraSeeding::Consonance,
                42,
                l,
                2000.0,
                100,
            )
        };
        let mut expected = make(&base);
        let mut actual = make(&habituated);
        assert_eq!(actual.biomass_mass_scan, expected.biomass_mass_scan);
        let mut reduced = make(&base);
        let mut reduced_base = habituated.clone();
        reduced_base
            .consonance_field_level
            .copy_from_slice(&habituated.consonance_field_level_eff);
        for _ in 0..100 {
            expected.advance(&base);
            actual.advance(&habituated);
            reduced.advance(&reduced_base);
        }
        assert_eq!(actual.biomass_mass_scan, expected.biomass_mass_scan);
        assert_eq!(actual.resource_state_scan, expected.resource_state_scan);
        assert_ne!(actual.biomass_mass_scan, reduced.biomass_mass_scan);
    }

    #[test]
    fn advance_matches_the_closed_form_update_for_uniform_biomass() {
        // Uniform biomass removes the spread and makes consumption the local density.
        let mut l = terrain();
        l.consonance_field_level.fill(0.7);
        l.harmonicity01.fill(0.4);
        let mut f = flora(&l);
        let bins_per_oct = f.space.bins_per_oct as f32;
        let (v0, u0) = (2e-3f32, 0.8f32);
        f.biomass_mass_scan.fill(0.0);
        f.biomass_mass_scan[f.range.clone()].fill(v0);
        f.resource_state_scan[f.range.clone()].fill(u0);
        let dt = f.hop as f32 / f.fs / f.substeps as f32;
        let (mut v, mut u) = (v0, u0);
        for _ in 0..f.substeps {
            let allee = ALLEE_DENSITY / bins_per_oct;
            v += dt
                * (GROWTH_RATE * v * (0.7 * u * v / (v + allee) - METABOLISM)
                    + u * (BACKGROUND_SEED_DENSITY + SEED_RAIN_DENSITY * 0.4) / bins_per_oct);
            let a = 1.0 / RENEWAL_SEC + v * bins_per_oct / REFERENCE_DENSITY / CONSUMPTION_SEC;
            let fixed = (1.0 / RENEWAL_SEC) / a;
            u = fixed + (u - fixed) * (-a * dt).exp();
        }
        f.advance(&l);
        for i in f.range.clone() {
            assert!(
                (f.biomass_mass_scan[i] - v).abs() <= 1e-6 * v,
                "biomass {} != {v}",
                f.biomass_mass_scan[i]
            );
            assert!(
                (f.resource_state_scan[i] - u).abs() <= 1e-5,
                "resource {} != {u}",
                f.resource_state_scan[i]
            );
        }
        assert!(v != v0 && u != u0);
    }

    #[test]
    fn partition_tiles_merges_remnants_and_assigns_boundary_upward() {
        let l = Landscape::new(Log2Space::new(110.0, 1760.0, 96));
        let lo = hz_to_erb(220.0);
        for width in [0.3, 1.3, 1.6, 4.3, 4.7] {
            let f = Flora::new(
                1,
                spec(),
                (erb_to_hz(lo), erb_to_hz(lo + width)),
                FloraSeeding::Uniform,
                1,
                &l,
                4000.0,
                100,
            );
            assert!(!f.cells.is_empty());
            assert_eq!(f.cells[0].bins.start, f.range.start);
            assert_eq!(f.cells.last().unwrap().bins.end, f.range.end);
            for (i, cell) in f.cells.iter().enumerate() {
                assert!(!cell.bins.is_empty());
                for bin in cell.bins.clone() {
                    assert!(f.erb_scan[bin] >= cell.lo_erb - 1e-5);
                    assert!(f.erb_scan[bin] < cell.hi_erb || i + 1 == f.cells.len());
                }
                if i > 0 {
                    assert_eq!(f.cells[i - 1].hi_erb, cell.lo_erb);
                }
            }
            if width < 0.5 {
                assert_eq!(f.cells.len(), 1);
            }
            if width == 4.3 {
                assert_eq!(f.cells.len(), 4);
            }
            if width == 4.7 {
                assert_eq!(f.cells.len(), 5);
            }
        }
        // Choose an exactly representable ERB boundary through the real partition path.
        let space = Log2Space::new(220.0, 880.0, 96);
        let (bin, lower, boundary) = (30..80)
            .find_map(|bin| {
                let boundary = hz_to_erb(space.centers_hz[bin]);
                let approximate = erb_to_hz(boundary - 1.0).to_bits();
                (approximate - 256..=approximate + 256)
                    .map(f32::from_bits)
                    .find(|&f| f >= space.fmin && hz_to_erb(f) + 1.0 == boundary)
                    .map(|lower| (bin, lower, boundary))
            })
            .unwrap();
        let f = Flora::new(
            1,
            spec(),
            (lower, 880.0),
            FloraSeeding::Uniform,
            1,
            &Landscape::new(space),
            2000.0,
            100,
        );
        assert_eq!(f.cells[0].hi_erb, boundary);
        assert_eq!(f.erb_scan[bin], boundary);
        assert!(!f.cells[0].bins.contains(&bin));
        assert!(f.cells[1].bins.contains(&bin));
        let full = flora(&terrain());
        assert!(
            full.cells
                .last()
                .unwrap()
                .bins
                .contains(&(full.range.end - 1))
        );
    }

    #[test]
    fn partition_empty_first_merges_up_and_empty_later_merges_down() {
        let l = Landscape::new(Log2Space::new(110.0, 3520.0, 1));
        let f = Flora::new(
            1,
            spec(),
            (111.0, 3519.0),
            FloraSeeding::Uniform,
            1,
            &l,
            8000.0,
            100,
        );
        assert!(f.cells[0].lo_erb < f.erb_scan[f.range.start]);
        assert_eq!(f.cells[0].lo_erb, hz_to_erb(111.0));
        assert!(f.cells.iter().any(|c| c.hi_erb - c.lo_erb > 1.5));
        let covered: Vec<_> = f.cells.iter().flat_map(|c| c.bins.clone()).collect();
        assert_eq!(covered, f.range.clone().collect::<Vec<_>>());
    }

    #[test]
    fn decomposition_slope_is_not_a_peak_and_plateau_uses_left_centre_cell() {
        let l = terrain();
        let mut f = flora(&l);
        for i in f.range.clone() {
            f.biomass_mass_scan[i] = i as f32 + 1.0;
        }
        f.decompose();
        assert!(
            f.cells[..f.cells.len() - 1]
                .iter()
                .all(|c| c.peak.is_none())
        );
        let boundary = f.cells[1].bins.start;
        f.biomass_mass_scan.fill(0.001);
        f.biomass_mass_scan[boundary - 2..boundary + 2].fill(1.0);
        f.decompose();
        assert_eq!(f.cells[0].peak.as_ref().unwrap().bin, boundary - 1);
        assert!(f.cells[1].peak.is_none());
        assert_eq!(f.cells[0].peak.as_ref().unwrap().sigma, 1.0);
    }

    #[test]
    fn decomposition_equal_maxima_choose_lower_and_window_stops_at_minimum() {
        let l = terrain();
        let mut f = flora(&l);
        f.biomass_mass_scan.fill(0.001);
        let c = f.cells[1].bins.clone();
        let a = c.start + 3;
        let b = a + 4;
        f.biomass_mass_scan[a - 1..b + 2].copy_from_slice(&[0.7, 1.0, 0.8, 0.7, 0.8, 1.0, 0.7]);
        f.decompose();
        let peak = f.cells[1].peak.as_ref().unwrap();
        assert_eq!(peak.bin, a);
        assert!(peak.window.end <= a + 3);
        assert!(!peak.window.contains(&b));
        f.biomass_mass_scan[a] = 1.2;
        f.decompose();
        let peak = f.cells[1].peak.as_ref().unwrap();
        assert!((0.5..=4.0).contains(&peak.sigma));
        assert!((peak.position - f.space.centers_log2[a]).abs() <= 0.5 * f.space.step());
    }

    #[test]
    fn decomposition_flat_floor_zeros_and_end_peaks_are_finite() {
        let l = terrain();
        let mut f = flora(&l);
        uniform(&mut f);
        let positions: Vec<_> = f.lanes[f.tonal_len..f.residual_end]
            .iter()
            .map(|l| l.target_position)
            .collect();
        allocate(&mut f);
        assert!(f.cells.iter().all(|c| c.excess == 0.0 && c.peak.is_none()));
        assert!(f.lanes.iter().all(|l| l.target == 0.0));
        assert_eq!(
            positions,
            f.lanes[f.tonal_len..f.residual_end]
                .iter()
                .map(|l| l.target_position)
                .collect::<Vec<_>>()
        );
        mass_accounting(&f);
        let mut p = vec![0.0; f.hop];
        let mut h = p.clone();
        f.render_add(&mut p, &mut h);
        assert!(p.iter().all(|&x| x == 0.0));
        for bin in [f.range.start, f.range.end - 1] {
            f.biomass_mass_scan.fill(0.0);
            f.biomass_mass_scan[bin] = 1.0;
            allocate(&mut f);
            assert!(
                f.cells
                    .iter()
                    .any(|c| c.peak.as_ref().is_some_and(|p| p.bin == bin))
            );
            mass_accounting(&f);
            f.render_add(&mut p, &mut h);
            assert!(p.iter().all(|x| x.is_finite()));
        }
    }

    #[test]
    fn mass_accounting_no_candidate_failed_candidacy_waiting_and_residual_off() {
        let l = terrain();
        let mut f = flora(&l);
        for i in f.range.clone() {
            f.biomass_mass_scan[i] = i as f32 + 1.0;
        }
        normalize(&mut f);
        allocate(&mut f);
        mass_accounting(&f);
        f = flora(&l);
        f.biomass_mass_scan.fill(0.00001);
        hill(&mut f, 33, 1.0);
        hill(&mut f, 73, 0.005);
        normalize(&mut f);
        allocate(&mut f);
        assert_eq!(f.cells[3].tonal_mass, 0.0);
        mass_accounting(&f);
        f = flora(&l);
        f.biomass_mass_scan.fill(0.00001);
        hill(&mut f, 33, 1.0);
        normalize(&mut f);
        for lane in &mut f.lanes[..f.tonal_len] {
            lane.mode = Mode::Retiring;
            lane.r = 1.0;
            lane.q = 1.0;
        }
        allocate(&mut f);
        assert!(f.cells.iter().all(|c| c.tonal_mass == 0.0));
        mass_accounting(&f);
        f.spec.residual = false;
        allocate(&mut f);
        mass_accounting(&f);
        assert!(
            f.lanes[f.tonal_len..f.residual_end]
                .iter()
                .all(|l| l.target == 0.0)
        );
    }

    #[test]
    fn tracking_drift_boundary_crossings_phase_and_hysteresis() {
        let l = terrain();
        let mut f = flora(&l);
        let boundary = f.cells[2].bins.start;
        let mut held = None;
        let mut visited_cells = Vec::new();
        for shift in [-0.6, -0.2, 0.2, 0.6, 0.2, -0.2, -0.6] {
            for i in f.range.clone() {
                f.biomass_mass_scan[i] =
                    0.000001 + unit_gaussian(i as f32 - (boundary as f32 + shift), 1.2);
            }
            normalize(&mut f);
            allocate(&mut f);
            let lane = f.lanes[..f.tonal_len]
                .iter()
                .position(|l| l.mode == Mode::Current)
                .unwrap();
            if let Some(old) = held {
                assert_eq!(lane, old);
            } else {
                held = Some(lane);
            }
            visited_cells.push(f.lanes[lane].cell.unwrap());
            let phase = f.lanes[lane].z;
            f.decompose();
            f.update_tones();
            f.residual_targets();
            assert_eq!(phase, f.lanes[lane].z, "tracking reset the phase");
            assert!(
                !f.lanes[..f.tonal_len]
                    .iter()
                    .any(|l| l.mode == Mode::Retiring)
            );
            hop(&mut f, &l);
        }
        assert!(visited_cells.windows(2).any(|w| w[0] != w[1]));
        f = flora(&l);
        f.biomass_mass_scan.fill(0.000001);
        hill(&mut f, 33, 1.0);
        hill(&mut f, 73, 0.03);
        normalize(&mut f);
        allocate(&mut f);
        let lane = f.cells[3].peak.as_ref().unwrap().claimed.unwrap();
        for height in [0.012, 0.03, 0.012, 0.03] {
            f.biomass_mass_scan.fill(0.000001);
            hill(&mut f, 33, 1.0);
            hill(&mut f, 73, height);
            normalize(&mut f);
            allocate(&mut f);
            assert_eq!(f.cells[3].peak.as_ref().unwrap().claimed, Some(lane));
        }
    }

    #[test]
    fn tracking_jump_retires_old_lane_and_births_another() {
        let l = terrain();
        let mut f = flora(&l);
        f.biomass_mass_scan.fill(0.000001);
        hill(&mut f, 33, 1.0);
        normalize(&mut f);
        hop(&mut f, &l);
        let old = f.lanes[..f.tonal_len]
            .iter()
            .position(|l| l.mode == Mode::Current)
            .unwrap();
        f.biomass_mass_scan.fill(0.000001);
        hill(&mut f, 43, 1.0);
        normalize(&mut f);
        allocate(&mut f);
        assert_eq!(f.lanes[old].mode, Mode::Retiring);
        assert_eq!(f.lanes[old].target, 0.0);
        assert!(
            f.lanes[..f.tonal_len]
                .iter()
                .enumerate()
                .any(|(i, l)| i != old && l.mode == Mode::Current)
        );
    }

    #[test]
    fn separation_suppresses_smaller_and_preserves_mass() {
        let l = terrain();
        let mut f = flora(&l);
        let boundary = f.cells[2].bins.start;
        f.biomass_mass_scan.fill(0.000001);
        f.biomass_mass_scan[boundary - 2] = 0.6;
        f.biomass_mass_scan[boundary + 1] = 1.0;
        normalize(&mut f);
        f.decompose();
        // Start with both held to exercise separation, rather than birth rejection.
        for (i, c) in [1, 2].into_iter().enumerate() {
            let peak = f.cells[c].peak.as_ref().unwrap();
            let lane = &mut f.lanes[i];
            lane.mode = Mode::Current;
            lane.position = peak.position;
            lane.target_position = peak.position;
            lane.target = peak.mass.sqrt();
        }
        f.update_tones();
        f.residual_targets();
        assert_eq!(f.lanes[0].mode, Mode::Retiring);
        assert_eq!(f.cells[1].tonal_mass, 0.0);
        assert!(f.cells[1].peak.as_ref().unwrap().suppressed);
        mass_accounting(&f);
    }

    #[test]
    fn separation_three_cell_chain_is_deterministic_for_input_order() {
        let l = terrain();
        let mut template = flora(&l);
        template.biomass_mass_scan.fill(0.001);
        hill(&mut template, 33, 1.0);
        normalize(&mut template);
        template.decompose();
        let peak = template.cells[1].peak.clone().unwrap();
        let base = hz_to_erb(600.0);
        for (i, mass) in [0.4, 0.6, 0.5].into_iter().enumerate() {
            let mut p = peak.clone();
            p.bin = i * 20;
            p.position = erb_to_hz(base + 0.3 * i as f32).log2();
            p.mass = mass;
            template.cells[i].peak = Some(p);
        }
        template.cells.truncate(3);
        let mut sets = Vec::new();
        for order in [[0, 1, 2], [2, 0, 1], [1, 2, 0]] {
            let mut f = template.clone();
            f.cells = order.iter().map(|&i| template.cells[i].clone()).collect();
            for i in 0..3 {
                f.lanes[i].mode = Mode::Current;
                f.lanes[i].position = template.cells[i].peak.as_ref().unwrap().position;
                f.lanes[i].target = template.cells[i].peak.as_ref().unwrap().mass.sqrt();
            }
            f.update_tones();
            let set: Vec<_> = f.lanes[..f.tonal_len]
                .iter()
                .filter(|l| l.mode == Mode::Current)
                .map(|l| f.cells[l.cell.unwrap()].peak.as_ref().unwrap().bin)
                .collect();
            assert_eq!(set.len(), 1);
            sets.push(set);
        }
        assert!(sets.windows(2).all(|w| w[0] == w[1]));
    }

    #[test]
    fn births_wait_until_next_boundary_after_long_hop_and_follow_mass_order() {
        let l = terrain();
        let mut f = Flora::new(
            7,
            spec(),
            (440.0, 880.0),
            FloraSeeding::Uniform,
            42,
            &l,
            2000.0,
            400,
        );
        f.biomass_mass_scan.fill(0.000001);
        hill(&mut f, 33, 1.0);
        hill(&mut f, 73, 0.5);
        normalize(&mut f);
        for lane in &mut f.lanes[..f.tonal_len] {
            lane.mode = Mode::Retiring;
            lane.r = 1.0;
            lane.q = 1.0;
            lane.position = 100.0_f32.log2();
            lane.target_position = lane.position;
        }
        f.spec.residual = false;
        let mut p = vec![0.0; f.hop];
        let mut h = p.clone();
        f.render_add(&mut p, &mut h);
        assert!(f.cells.iter().all(|c| c.tonal_mass == 0.0));
        assert!(
            f.lanes[..f.tonal_len]
                .iter()
                .all(|l| l.mode == Mode::Retiring)
        );
        let tail_seconds = (1.0 + 1.0 / DEAD_MASS.sqrt()).ln() / (0.5 * GROWTH_RATE * METABOLISM);
        let max_hops = (tail_seconds * f.fs / f.hop as f32).ceil() as usize + 1;
        for _ in 1..max_hops {
            if f.lanes[..f.tonal_len].iter().all(|l| l.mode == Mode::Free) {
                break;
            }
            f.render_add(&mut p, &mut h);
            assert_eq!(f.births, 0);
        }
        assert!(f.lanes[..f.tonal_len].iter().all(|l| l.mode == Mode::Free));
        f.render_add(&mut p, &mut h);
        assert_eq!(f.cells[1].peak.as_ref().unwrap().claimed, Some(0));
        assert_eq!(f.cells[3].peak.as_ref().unwrap().claimed, Some(1));
    }

    #[test]
    fn slew_bounds_and_exact_deadlines_hold_even_at_amp_zero() {
        for noisy in [false, true] {
            let mut lane = Lane::new(noisy, 440.0_f32.log2());
            lane.mode = Mode::Current;
            lane.target = 1.0;
            let step = 1.0 / 160.0;
            for _ in 0..160 {
                let old = if noisy { lane.q } else { lane.r };
                lane.slew(2000.0);
                let new = if noisy { lane.q } else { lane.r };
                assert!((new - old).abs() <= step + 1e-7);
            }
            assert_eq!(lane.r, 1.0);
            lane.retire();
            for _ in 0..160 {
                lane.slew(2000.0);
            }
            assert_eq!(lane.r, 0.0);
            assert_eq!(lane.mode, Mode::Free);
        }
        let l = terrain();
        let mut f = flora(&l);
        f.hop = 400;
        f.habitat_pcm.resize(f.hop, 0.0);
        f.noise.resize(2 * f.noisy_len * f.hop, 0.0);
        f.biomass_mass_scan.fill(0.001);
        hill(&mut f, 33, 1.0);
        normalize(&mut f);
        f.apply_patch(&FloraPatch { amp: Some(0.0) });
        let mut p = vec![0.0; f.hop];
        let mut h = p.clone();
        f.render_add(&mut p, &mut h);
        assert_eq!(f.amp.value, 0.0);
        assert!(p[160..].iter().all(|&y| y == 0.0));
        assert!(f.lanes.iter().all(|l| l.r == l.target));
    }

    #[test]
    fn atoms_sine_is_unit_and_envelope_step_matches_logged_noise() {
        let mut sine = Lane::new(false, 0.0);
        sine.w = [0.8, 0.6];
        for _ in 0..100 {
            sine.atom([0.0; 2]);
            assert_eq!(sine.e, [1.0, 0.0]);
        }
        let mut lane = Lane::new(true, 0.0);
        lane.a = 0.7;
        lane.b = (1.0 - lane.a * lane.a).sqrt();
        lane.e = [0.4, -0.6];
        let n = noise_draw(&mut 123);
        let expected = [
            mul_add_fast(lane.a, lane.e[0], lane.b * n[0]),
            mul_add_fast(lane.a, lane.e[1], lane.b * n[1]),
        ];
        lane.atom(n);
        assert_eq!(lane.e, expected);
    }

    #[test]
    fn noise_order_and_resonant_bandwidth_are_canonical() {
        let l = terrain();
        let mut s = spec();
        s.tone = FloraTone::Resonant;
        let mut f = Flora::new(
            7,
            s,
            (440.0, 880.0),
            FloraSeeding::Uniform,
            42,
            &l,
            2000.0,
            100,
        );
        uniform(&mut f);
        let mut rng = f.rng;
        let expected: Vec<_> = (0..f.hop * f.noisy_len)
            .flat_map(|_| noise_draw(&mut rng))
            .collect();
        let mut p = vec![0.0; f.hop];
        let mut h = p.clone();
        f.render_add(&mut p, &mut h);
        assert_eq!(f.noise, expected);
        assert_eq!(f.rng, rng);
        for width in [0.1, 0.5, 1.0, 4.0, 12.0] {
            f = Flora::new(
                7,
                s,
                (440.0, 880.0),
                FloraSeeding::Uniform,
                42,
                &l,
                2000.0,
                100,
            );
            for bin in f.range.clone() {
                f.biomass_mass_scan[bin] = unit_gaussian(bin as f32 - 33.0, width);
            }
            normalize(&mut f);
            f.render_add(&mut p, &mut h);
            let i = f.cells[1].peak.as_ref().unwrap().claimed.unwrap();
            let lane = &f.lanes[i];
            assert!((lane.sigma - width.clamp(0.5, 4.0)).abs() < 0.002);
            let hz = lane.target_position.exp2();
            let expected = (hz * ((lane.sigma / f.space.bins_per_oct as f32).exp2() - 1.0))
                .clamp(0.5, 0.5 * erb_bw_hz(hz));
            let actual = -lane.a.ln() * f.fs / PI;
            assert!((actual - expected).abs() < 1e-4, "{actual} != {expected}");
            let residual = &f.lanes[f.tonal_len];
            assert!(
                (-residual.a.ln() * f.fs / PI - erb_bw_hz(residual.target_position.exp2())).abs()
                    < 1e-4
            );
        }
        for (lo, hi, bins_per_oct, lower_clamp) in
            [(10.0, 40.0, 96, true), (100.0, 1600.0, 1, false)]
        {
            let l = Landscape::new(Log2Space::new(lo, hi, bins_per_oct));
            let mut f = Flora::new(7, s, (lo, hi), FloraSeeding::Uniform, 42, &l, 4000.0, 100);
            f.biomass_mass_scan.fill(0.0);
            let peak = f.range.start + f.range.len() / 2;
            f.biomass_mass_scan[peak] = 1.0;
            f.render_add(&mut p, &mut h);
            let lane = f
                .lanes
                .iter()
                .find(|l| l.tonal && l.mode == Mode::Current)
                .unwrap();
            let expected = if lower_clamp {
                0.5
            } else {
                0.5 * erb_bw_hz(lane.position.exp2())
            };
            assert!((-lane.a.ln() * f.fs / PI - expected).abs() < 1e-4);
        }
    }

    #[test]
    fn atoms_seeded_ensemble_power_and_autocorrelation() {
        const LANES: usize = 1024;
        for bandwidth in [0.5, erb_bw_hz(1000.0)] {
            let fs = if bandwidth < 1.0 { 512.0 } else { 2048.0 };
            let samples = (10.0 * fs) as usize;
            let a = (-PI * bandwidth / fs).exp();
            let b = (1.0 - a * a).sqrt();
            let lags = [1, (fs / (10.0 * bandwidth)).round() as usize];
            let mut history = vec![[0.0; 2]; samples];
            let mut powers = 0.0_f64;
            let mut numerator = [0.0_f64; 2];
            let mut denominator = [0.0_f64; 2];
            for id in 0..LANES {
                let mut rng = 0xCAFE_1234 ^ (id as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
                let mut lane = Lane::new(true, 0.0);
                lane.a = a;
                lane.b = b;
                lane.e = noise_draw(&mut rng);
                let (sin, cos) = (TAU * random_unit(&mut rng)).sin_cos();
                lane.z = [cos, sin];
                let (sin, cos) = (TAU * 37.0 / fs).sin_cos();
                lane.w = [cos, sin];
                for (t, e) in history.iter_mut().enumerate() {
                    if t % 64 == 0 {
                        let norm = lane.z[0].hypot(lane.z[1]);
                        lane.z[0] /= norm;
                        lane.z[1] /= norm;
                    }
                    let y = lane.atom(noise_draw(&mut rng));
                    powers += f64::from(y * y);
                    *e = lane.e;
                }
                for (j, &k) in lags.iter().enumerate() {
                    for t in 0..samples - k {
                        let x = history[t];
                        let y = history[t + k];
                        numerator[j] += f64::from(x[0] * y[0] + x[1] * y[1]);
                        denominator[j] += f64::from(x[0] * x[0] + x[1] * x[1]);
                    }
                }
            }
            let power = powers / (LANES * samples) as f64;
            assert!(
                (power / 0.5 - 1.0).abs() <= 0.05,
                "B={bandwidth}, power={power}"
            );
            for (j, &k) in lags.iter().enumerate() {
                let rho = numerator[j] / denominator[j];
                assert!(
                    (rho - f64::from(a).powi(k as i32)).abs() <= 0.06,
                    "B={bandwidth}, lag={k}, rho={rho}, expected={}",
                    a.powi(k as i32)
                );
            }
        }
    }

    #[test]
    #[cfg(feature = "simd-wide")]
    fn simd_and_scalar_share_noise_and_agree_for_ten_seconds() {
        let l = terrain();
        for tone in [FloraTone::Sine, FloraTone::Resonant] {
            let mut f = flora(&l);
            f.spec.tone = tone;
            if tone == FloraTone::Resonant {
                f = Flora::new(
                    7,
                    f.spec,
                    f.extent_hz,
                    FloraSeeding::Uniform,
                    42,
                    &l,
                    f.fs,
                    f.hop,
                );
            }
            let mut scalar = f.clone();
            let mut p = vec![0.0; f.hop];
            let mut h = p.clone();
            let mut reference = p.clone();
            let mut error = 0.0_f64;
            let mut norm = 0.0_f64;
            let mut maximum = 0.0_f32;
            for hop in 0..200 {
                // Drift, jump beyond the tracking radius, disappear, then reuse freed lanes.
                let bin = match hop {
                    0..50 => Some(33 + hop / 10),
                    50..100 => Some(43 + (hop - 50) / 10),
                    100..180 => None,
                    _ => Some(33 + (hop - 180) / 10),
                };
                f.biomass_mass_scan.fill(0.00001);
                if let Some(bin) = bin {
                    hill(&mut f, bin, 1.0);
                    hill(&mut f, 73, 0.7);
                }
                normalize(&mut f);
                scalar
                    .biomass_mass_scan
                    .copy_from_slice(&f.biomass_mass_scan);
                let position = f.lanes[0].position;
                let births = f.births;
                let retirements = f.retirements;
                p.fill(0.0);
                h.fill(0.0);
                reference.fill(0.0);
                f.render::<true>(&mut p, &mut h);
                h.fill(0.0);
                scalar.render::<false>(&mut reference, &mut h);
                assert_eq!(f.noise, scalar.noise);
                assert_eq!(f.rng, scalar.rng);
                assert_eq!(f.births, scalar.births);
                assert_eq!(f.retirements, scalar.retirements);
                match hop {
                    10 => {
                        assert_eq!(f.lanes[0].mode, Mode::Current);
                        assert!(f.lanes[0].position > position);
                        assert_eq!(f.births, births);
                        assert_eq!(f.retirements, retirements);
                    }
                    50 => {
                        assert_ne!(f.lanes[0].mode, Mode::Current);
                        assert!(f.retirements > retirements);
                        assert!(f.births > births);
                    }
                    179 => assert!(f.lanes[..f.tonal_len].iter().all(|l| l.mode == Mode::Free)),
                    180 => {
                        assert_eq!(f.lanes[0].mode, Mode::Current);
                        assert!(f.births > births);
                    }
                    _ => {}
                }
                for (&x, &y) in p.iter().zip(&reference) {
                    error += f64::from(x - y).powi(2);
                    norm += f64::from(y).powi(2);
                    maximum = maximum.max((x - y).abs());
                }
            }
            assert!(
                error.sqrt() <= 1e-5 * norm.sqrt(),
                "relative error {}",
                (error / norm).sqrt()
            );
            assert!(maximum <= 1e-6 * f.spec.amp, "absolute error {maximum}");
        }
    }

    #[test]
    fn reconfiguration_interpolates_biomass_and_resource_without_normalizing() {
        let l = Landscape::new(Log2Space::new(256.0, 2048.0, 1));
        let mut f = Flora::new(
            7,
            spec(),
            (256.0, 2048.0),
            FloraSeeding::Uniform,
            42,
            &l,
            8000.0,
            100,
        );
        // Old log2 bins: 8, 9, 10, 11. New in-range bins: 8.5, 9.5, 10.5.
        f.biomass_mass_scan.copy_from_slice(&[1.0, 5.0, 3.0, 7.0]);
        f.resource_state_scan
            .copy_from_slice(&[0.0, 0.5, 1.0, 0.25]);
        f.on_space_change(&Log2Space::new(2.0_f32.powf(7.5), 2.0_f32.powf(11.5), 1));
        assert_eq!(f.range, 1..4);
        let interpolated = [0.0, 3.0, 4.0, 5.0, 0.0];
        let resource = [0.0, 0.25, 0.75, 0.625, 0.0];
        for i in 0..f.space.n_bins() {
            assert!((f.biomass_mass_scan[i] - interpolated[i]).abs() < 1e-6);
            assert!((f.resource_state_scan[i] - resource[i]).abs() < 1e-7);
        }
    }

    #[test]
    fn reconfiguration_preserves_biomass_density_across_bin_resolutions() {
        let l = Landscape::new(Log2Space::new(256.0, 512.0, 96));
        let mut f = Flora::new(
            7,
            spec(),
            (256.0, 512.0),
            FloraSeeding::Uniform,
            42,
            &l,
            2000.0,
            100,
        );
        f.biomass_mass_scan.fill(3.0 / 96.0);
        f.resource_state_scan.fill(0.4);
        let initial: f32 = f.biomass_mass_scan.iter().sum();
        for bins in [192, 72, 120, 96] {
            let mut resampled = f.clone();
            resampled.on_space_change(&Log2Space::new(256.0, 512.0, bins));
            let total: f32 = resampled.biomass_mass_scan.iter().sum();
            // Endpoint inclusion can change the sum by one bin on the coarser grid.
            let endpoint_error = 3.0 / 96.0_f32.min(bins as f32);
            assert!(
                (total - initial).abs() <= endpoint_error + 1e-5,
                "{bins} bins/octave: initial {initial}, resampled {total}"
            );
            for i in resampled.range.clone() {
                assert!((resampled.biomass_mass_scan[i] * bins as f32 - 3.0).abs() < 1e-5);
                assert!((resampled.resource_state_scan[i] - 0.4).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn reconfiguration_carries_orphans_across_two_changes() {
        let l = terrain();
        let mut f = flora(&l);
        f.biomass_mass_scan.fill(0.000001);
        hill(&mut f, 33, 1.0);
        normalize(&mut f);
        hop(&mut f, &l);
        f.lanes[f.tonal_len].r = 0.5;
        f.lanes[f.tonal_len].q = 0.25;
        let old_position = f.lanes[f.tonal_len].position;
        let old_tonal: Vec<_> = f.lanes[..f.tonal_len]
            .iter()
            .filter(|l| l.mode != Mode::Free)
            .map(|l| l.position)
            .collect();
        f.on_space_change(&Log2Space::new(400.0, 1000.0, 120));
        assert!(
            f.lanes[f.residual_end..]
                .iter()
                .any(|l| l.position == old_position && l.mode == Mode::Retiring)
        );
        for pos in old_tonal {
            assert!(f.lanes.iter().any(|l| l.tonal && l.position == pos));
        }
        f.on_space_change(&Log2Space::new(420.0, 960.0, 72));
        assert!(
            f.lanes[f.residual_end..]
                .iter()
                .any(|l| l.position == old_position && l.mode == Mode::Retiring)
        );
        let mut p = vec![0.0; f.hop];
        let mut h = p.clone();
        let tail_seconds = (1.0 + 0.5 / DEAD_MASS.sqrt()).ln() / (0.5 * GROWTH_RATE * METABOLISM);
        let max_hops = (tail_seconds * f.fs / f.hop as f32).ceil() as usize + 1;
        for _ in 0..max_hops {
            f.render_add(&mut p, &mut h);
        }
        assert!(
            f.lanes[f.residual_end..]
                .iter()
                .filter(|l| !l.tonal)
                .all(|l| l.mode == Mode::Free)
        );
    }

    #[test]
    fn reconfiguration_dormancy_requires_an_actual_bin_and_resumes_uniformly() {
        let l = terrain();
        let mut f = Flora::new(
            1,
            spec(),
            (450.0, 460.0),
            FloraSeeding::Uniform,
            42,
            &l,
            2000.0,
            100,
        );
        f.on_space_change(&Log2Space::new(440.0, 880.0, 1));
        assert!(f.range.is_empty());
        assert!(f.cells.is_empty());
        let mut p = vec![0.0; f.hop];
        let mut h = p.clone();
        f.render_add(&mut p, &mut h);
        assert!(p.iter().all(|&x| x == 0.0));
        f.on_space_change(&l.space);
        assert!(!f.range.is_empty());
        assert!(
            f.resource_state_scan[f.range.clone()]
                .iter()
                .all(|&u| u == 1.0)
        );
        assert!((f.biomass_mass_scan.iter().sum::<f32>() - INITIAL_BIOMASS).abs() < 1e-6);
        assert!(
            f.biomass_mass_scan[f.range.clone()]
                .windows(2)
                .all(|w| w[0] == w[1])
        );
        f.on_space_change(&Log2Space::new(1000.0, 2000.0, 96));
        assert!(f.range.is_empty());
    }

    #[test]
    fn release_uses_current_gain_and_scenario_end_keeps_existing_release() {
        let mut linear = Ramp::new(0.0);
        linear.to(1.0, 2.0, 48000.0);
        for t in 1..=96000 {
            let value = linear.tick();
            if t % 24000 == 0 {
                assert_eq!(value, t as f32 / 96000.0);
            }
        }

        let l = terrain();
        let mut s = spec();
        s.fade_sec = 1.0;
        let mut f = Flora::new(
            7,
            s,
            (440.0, 880.0),
            FloraSeeding::Uniform,
            42,
            &l,
            2000.0,
            100,
        );
        hop(&mut f, &l);
        let current = f.gain.value;
        assert!(current > 0.0 && current < 1.0);
        f.release(0.1);
        assert_eq!(f.gain.value, current);
        let left = f.gain.left;
        f.release_at_scenario_end();
        assert_eq!(f.gain.left, left);
        for _ in 0..3 {
            hop(&mut f, &l);
        }
        assert!(f.is_finished());
        let mut zero = flora(&l);
        zero.release(0.0);
        assert!(zero.is_finished());
    }

    #[test]
    fn routing_adds_to_only_the_selected_buffers() {
        let l = terrain();
        for presentation in [false, true] {
            let mut f = flora(&l);
            f.spec.routing = Routing {
                to_presentation: presentation,
                to_habitat: !presentation,
            };
            f.biomass_mass_scan.fill(0.001);
            hill(&mut f, 33, 1.0);
            normalize(&mut f);
            let mut p = vec![0.25; f.hop];
            let mut h = p.clone();
            f.render_add(&mut p, &mut h);
            let (sounding, silent) = if presentation { (&p, &h) } else { (&h, &p) };
            assert!(sounding.iter().any(|&x| x != 0.25));
            assert!(silent.iter().all(|&x| x == 0.25));
        }
    }

    #[test]
    fn f2_every_internal_scan_boundary_names_the_bad_scan() {
        let l = terrain();
        for name in [
            "biomass_mass_scan",
            "resource_state_scan",
            "scratch_scan",
            "erb_scan",
        ] {
            for operation in 0..7 {
                let mut f = flora(&l);
                match name {
                    "biomass_mass_scan" => {
                        f.biomass_mass_scan.pop();
                    }
                    "resource_state_scan" => {
                        f.resource_state_scan.pop();
                    }
                    "scratch_scan" => {
                        f.scratch_scan.pop();
                    }
                    _ => {
                        f.erb_scan.pop();
                    }
                }
                let result =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match operation {
                        0 => f.advance(&l),
                        1 => f.decompose(),
                        2 => f.residual_targets(),
                        3 => f.render_add(&mut [0.0; 100], &mut [0.0; 100]),
                        4 => {
                            f.report(0.0, &l);
                        }
                        5 => f.advance_resource(0.01),
                        _ => f.on_space_change(&l.space),
                    }));
                let error = result.expect_err(name);
                let message = error
                    .downcast_ref::<String>()
                    .map(String::as_str)
                    .or_else(|| error.downcast_ref::<&str>().copied())
                    .unwrap();
                assert!(message.contains(name), "{message}");
            }
        }
    }

    #[test]
    #[should_panic(expected = "consonance_field_level")]
    fn f2_advance_checks_landscape_input() {
        let mut l = terrain();
        let mut f = flora(&l);
        l.consonance_field_level.pop();
        f.advance(&l);
    }

    #[test]
    #[should_panic(expected = "harmonicity01")]
    fn f2_advance_checks_harmonicity_input() {
        let mut l = terrain();
        let mut f = flora(&l);
        l.harmonicity01.pop();
        f.advance(&l);
    }

    #[test]
    #[should_panic(expected = "consonance_density_mass")]
    fn f2_placement_checks_landscape_input() {
        let mut l = terrain();
        l.consonance_density_mass.pop();
        let _ = flora(&l);
    }

    #[test]
    #[cfg(feature = "profile-alloc")]
    fn advance_and_render_allocate_nothing_during_continual_turnover() {
        use crate::runtime_profile::{begin_allocations, finish_allocations};
        for tone in [FloraTone::Sine, FloraTone::Resonant] {
            let mut l = terrain();
            l.harmonicity01.fill(0.0);
            let mut f = flora(&l);
            f.spec.tone = tone;
            f.on_space_change(&l.space);
            let mut p = vec![0.0; f.hop];
            let mut h = p.clone();
            let boundary = f.cells[2].bins.start;
            let mut last_cell = None;
            let mut crossings = 0;
            begin_allocations();
            for t in 0..300 {
                l.consonance_field_level.fill(0.5);
                f.biomass_mass_scan.fill(0.000001);
                if t % 30 < 10 {
                    hill(&mut f, 33, 1.0);
                } else if t % 30 < 20 {
                    hill(&mut f, 73, 1.0);
                } else {
                    let shift = if t % 2 == 0 { -0.6 } else { 0.6 };
                    for i in f.range.clone() {
                        f.biomass_mass_scan[i] +=
                            unit_gaussian(i as f32 - (boundary as f32 + shift), 1.2);
                    }
                }
                normalize(&mut f);
                f.advance(&l);
                p.fill(0.0);
                h.fill(0.0);
                f.render_add(&mut p, &mut h);
                if t % 30 >= 20 {
                    let cell = f
                        .lanes
                        .iter()
                        .find(|l| l.tonal && l.mode == Mode::Current)
                        .and_then(|l| l.cell);
                    if last_cell.is_some() && cell != last_cell {
                        crossings += 1;
                    }
                    last_cell = cell;
                } else {
                    last_cell = None;
                }
            }
            let counts = finish_allocations().unwrap();
            assert!(f.births > 0 && f.retirements > 0 && crossings > 0);
            assert_eq!(counts.count, 0, "{counts:?}");
            assert_eq!(counts.bytes, 0, "{counts:?}");
            println!(
                "F12_FLORA_ALLOCATIONS tone={tone:?} counts={counts:?} births={} retirements={} crossings={crossings}",
                f.births, f.retirements
            );
        }
    }
}

#[cfg(test)]
#[path = "flora_heard_tests.rs"]
mod heard_tests;
