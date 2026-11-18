//! Rain: impacts drawn one by one from distributions over a frequency extent, and the
//! resonant objects they strike. Design: `docs/design-notes/rain.md`.

use crate::core::erb::hz_to_erb;
use crate::core::float::sanitize01;
use crate::core::habituation::{erode_mass, erode_score};
use crate::core::landscape::Landscape;
use crate::life::community::{field_density_mass, field_target_mass};
use crate::life::flora::TemporalObservation;
use crate::scenario::{FieldTarget, RainPatch, RainSpec};
use crate::synth::modes::ModeParams;
use crate::synth::resonator::ResonatorBank;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use std::f32::consts::{PI, TAU};

const IMPACT_SLOTS: usize = 256;
/// Most pattern modes an impact uses.
const IMPACT_MODES: usize = 8;
/// Impact constants follow the TEX-1 A2 prototype, which the author heard as natural.
/// First-mode decay: 25 ms at 3.2 kHz, longer at lower centres as (3.2 kHz / f)^0.5
/// (A2 used 25 ms over 1.5-7 kHz and 50 ms over 0.25-1.5 kHz), log-normal spread 0.4.
const IMPACT_TAU_REF_SEC: f32 = 0.025;
const IMPACT_TAU_REF_HZ: f32 = 3200.0;
const IMPACT_TAU_SPREAD: f32 = 0.4;
/// Mode k has level 0.85^k with a 3 dB spread.
const IMPACT_BRIGHTNESS: f32 = 0.85;
const IMPACT_MODE_LEVEL_SPREAD_DB: f32 = 3.0;
const IMPACT_LEVEL_SPREAD_DB: f32 = 5.0;
const IMPACT_LEVEL_CAP_DB: f32 = 9.0;
/// Raised-cosine onset length; a hard start adds a broadband click to every impact.
const IMPACT_ATTACK_MIN_SEC: f32 = 0.0005;
const IMPACT_ATTACK_MAX_SEC: f32 = 0.0015;
/// Amplitude below which a decaying impact is dropped (-80 dB).
const IMPACT_FLOOR: f32 = 1e-4;
pub(crate) const MAX_RAIN_OBJECTS: u32 = 16;
/// Upper impact rate; it bounds per-sample work, not a musical choice.
pub(crate) const MAX_RAIN_RATE: f32 = 2000.0;
const OBJECT_MIN_SPACING_ERB: f32 = 1.0;
const OBJECT_DECAY_EXPONENT: f32 = 0.7;
/// Largest retuning per strike: one semitone, so an object drifts slowly.
const OBJECT_STEP_OCT: f32 = 1.0 / 12.0;
/// Movement cost per octave and the score gain a move must exceed. Both are the Voice
/// hill-climb defaults (coefficient 0.5 over a 2 s integration window; threshold 0.1).
const OBJECT_MOVE_COST_PER_OCT: f32 = 1.0;
const OBJECT_MIN_IMPROVEMENT: f32 = 0.1;
/// Samples searched for the onset peak of an object's impulse response.
const OBJECT_ONSET_WINDOW: usize = 2400;
const SLEW_SEC: f32 = 0.08;
/// Points over one cycle of the slowest meter level that normalise the impact-rate
/// weight to mean 1. With at most 16 fastest cycles per measure (subdivision 4 of a
/// 4-beat measure) each fastest cycle keeps 128 points.
const METER_GRID: usize = 2048;
/// Largest von Mises concentration of one meter level. At 128 points per cycle the
/// normalisation's aliasing error stays near exp(-128^2 / (2 * 512)) = exp(-16); the
/// pulse is then about 2 degrees wide, so the per-sample rate cap governs before it.
pub(crate) const MAX_METER_KAPPA: f32 = 512.0;

#[derive(Clone, Copy, Debug, Default)]
struct ImpactMode {
    re: f32,
    im: f32,
    w_re: f32,
    w_im: f32,
}

/// One meter level's concentration and phasor, advanced per sample within a hop.
#[derive(Clone, Copy, Debug, Default)]
struct MeterLevel {
    kappa: f32,
    // f64 keeps the phasor's magnitude through a hop; at the largest concentration an
    // f32 drift of 1e-5 would already move the weight by percents.
    re: f64,
    im: f64,
    w_re: f64,
    w_im: f64,
}

/// Impact-rate weight `exp(sum kappa (cos phi - 1)) / norm` from the habitat meter.
/// `count == 0` leaves the rate unweighted.
#[derive(Clone, Debug, Default)]
struct MeterWeight {
    levels: [MeterLevel; 3],
    count: usize,
    inv_norm: f32,
    /// Beat phase (radians) at the hop start and its per-sample step, for reports.
    beat_phase: f32,
    beat_step: f32,
}

impl MeterWeight {
    /// The current weight; then advance every level by one sample.
    fn next(&mut self) -> f32 {
        let mut exponent = 0.0f32;
        for level in &mut self.levels[..self.count] {
            exponent += level.kappa * (level.re - 1.0) as f32;
            let re = level.re * level.w_re - level.im * level.w_im;
            let im = level.re * level.w_im + level.im * level.w_re;
            (level.re, level.im) = (re, im);
        }
        exponent.exp() * self.inv_norm
    }
}

/// Von Mises concentration whose mean resultant length is `r`: the piecewise
/// approximation of the inverse of A1 = I1/I0 given by Fisher (1993).
fn circular_concentration(r: f32) -> f32 {
    let r = sanitize01(r);
    if r < 0.53 {
        2.0 * r + r.powi(3) + 5.0 * r.powi(5) / 6.0
    } else if r < 0.85 {
        -0.4 + 1.39 * r + 0.43 / (1.0 - r)
    } else {
        1.0 / (r.powi(3) - 4.0 * r * r + 3.0 * r)
    }
}

#[derive(Clone, Debug)]
struct Impact {
    modes: [ImpactMode; IMPACT_MODES],
    /// Samples left before the slot frees.
    left: usize,
    attack_len: usize,
    attack_pos: usize,
    #[cfg(test)]
    start_energy: f32,
}

/// One impact as struck, for reports. `offset` is the sample within the rendered hop.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ImpactRecord {
    pub(crate) offset: usize,
    pub(crate) centre_hz: f32,
    pub(crate) first_mode_hz: f32,
    pub(crate) ring_gain: f32,
    pub(crate) level: f32,
    pub(crate) object: Option<usize>,
    /// Struck object's fundamental after this strike.
    pub(crate) object_hz: Option<f32>,
    /// The struck object had fallen silent and took this impact's centre.
    pub(crate) replaced: bool,
    /// The struck object had fallen silent, but the centre lay within 1 ERB of another
    /// object, so it kept its pitch.
    pub(crate) replace_blocked: bool,
    /// With `follow_meter`, the impact-rate weight and the beat phase in cycles [0, 1)
    /// at this impact; None while the rate is unweighted.
    pub(crate) meter_weight: Option<f32>,
    pub(crate) beat_phase: Option<f32>,
}

/// What the last advance read over the extent, for reports. `source` is "mixed",
/// "observed" (own observation without the Rain's impacts) or "held" (no observation
/// this hop; the previous tables stay in use and these values are theirs).
#[derive(Debug, Default)]
pub(crate) struct FieldTrace {
    pub(crate) source: &'static str,
    pub(crate) first_bin: usize,
    pub(crate) raw_score: Vec<f32>,
    pub(crate) eff_score: Vec<f32>,
    pub(crate) mass: Vec<f32>,
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
        self.left = (seconds.max(0.0) * fs).round() as usize;
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
                self.value + self.step
            };
        }
        self.value
    }
}

#[derive(Debug)]
pub(crate) struct Rain {
    id: u64,
    spec: RainSpec,
    target: FieldTarget,
    extent_hz: (f32, f32),
    fs: f32,
    rng: SmallRng,
    /// Log2 centres and cumulative mass of the extent's bins, rebuilt each hop.
    bin_log2: Vec<f32>,
    cdf: Vec<f32>,
    bin_step_log2: f32,
    /// Per extent bin, the eroded consonance score normalised to [0, 1] within the
    /// extent; filled only while `ring` is positive.
    bin_ring: Vec<f32>,
    /// Remaining unit-exponential budget before the next impact. Subtracting the current
    /// rate per sample (time rescaling) keeps the process Poisson while the rate changes.
    impact_budget: f32,
    /// Impact rate per sample after sway, updated once per rendered block.
    rate_per_sample: f32,
    sway: super::sway::Sway,
    meter: MeterWeight,
    #[cfg(test)]
    impacts_struck: u64,
    /// Strikes that took a slot still sounding, and the stolen impact's remaining energy
    /// relative to its start, in whole dB below it (index 80 collects anything lower).
    #[cfg(test)]
    impacts_stolen: u64,
    #[cfg(test)]
    stolen_db_hist: [u64; 81],
    impacts: Vec<Impact>,
    /// Impact mode ratios, taken from `impact_modes` at placement.
    impact_ratios: Vec<f32>,
    /// Scale that gives an impact of level 1 unit mode-level energy.
    impact_norm: f32,
    objects: Vec<ResonatorBank>,
    /// Mode ratios of each object, fixed at placement.
    object_ratios: Vec<Vec<f32>>,
    /// Sounding and proposed fundamental of each object, in log2 Hz.
    object_log2: Vec<f32>,
    object_target_log2: Vec<f32>,
    /// Samples rendered since placement; it times object strikes.
    sample_clock: u64,
    /// Sample of each object's last strike.
    object_struck_at: Vec<Option<u64>>,
    /// Samples after a strike by which each object's longest audible mode (its T60 at
    /// placement) has fallen 60 dB.
    object_silent_after: Vec<u64>,
    mode_scratch: Vec<ModeParams>,
    /// Per-object drive scale that makes a strike's onset peak match its impact's level.
    object_gain: Vec<f32>,
    drive: Vec<f32>,
    amp: Ramp,
    gain: Ramp,
    releasing: bool,
    /// Eroded score scan built from the Rain's own observation (full Log2Space length).
    own_score_eff_scan: Vec<f32>,
    /// This hop's impacts as added to the habitat bus; hop-long only when the Rain
    /// needs an observer.
    drops_habitat_pcm: Vec<f32>,
    /// Before the first render the mixed field holds none of the Rain's own sound.
    rendered: bool,
    /// Reports only: impacts struck in the last render and the last field read.
    pub(crate) trace_enabled: bool,
    pub(crate) impact_trace: Vec<ImpactRecord>,
    pub(crate) field_trace: FieldTrace,
}

impl Rain {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        id: u64,
        spec: RainSpec,
        extent_hz: (f32, f32),
        target: FieldTarget,
        scenario_seed: u64,
        landscape: &Landscape,
        fs: f32,
        hop: usize,
    ) -> Self {
        assert!(fs.is_finite() && fs > 0.0);
        assert!(extent_hz.0 > 0.0 && extent_hz.1 > extent_hz.0);
        let mut rng =
            SmallRng::seed_from_u64(scenario_seed ^ id.wrapping_mul(0x9E6C_63D0_676A_9A99));
        let pitches = object_pitches(&spec, extent_hz, landscape, &mut rng);
        let object_ratios: Vec<Vec<f32>> = pitches
            .iter()
            .map(|&f0| {
                let mut ratios =
                    spec.object_modes
                        .eval(f0, &landscape.space, Some(landscape), &mut rng);
                ratios.retain(|&ratio| ratio.is_finite() && ratio > 0.0);
                ratios
            })
            .collect();
        let mut mode_scratch =
            Vec::with_capacity(object_ratios.iter().map(Vec::len).max().unwrap_or(0));
        let objects: Vec<ResonatorBank> = pitches
            .iter()
            .zip(&object_ratios)
            .map(|(&f0, ratios)| {
                fill_object_modes(&spec, f0, ratios, fs, &mut mode_scratch);
                let mut bank =
                    ResonatorBank::new(fs, ratios.len().max(1)).expect("finite sample rate");
                bank.set_modes_preserve_state(&mode_scratch)
                    .expect("modes within capacity");
                bank
            })
            .collect();
        let object_gain = objects
            .iter()
            .map(|bank| onset_gain(bank.clone()))
            .collect();
        let object_log2: Vec<f32> = pitches.iter().map(|f0| f0.log2()).collect();
        // Same T60 law as fill_object_modes. The fundamental stays audible after any
        // retuning inside the extent, so the placement value holds.
        let object_silent_after = pitches
            .iter()
            .zip(&object_ratios)
            .map(|(&f0, ratios)| {
                let t60 = ratios
                    .iter()
                    .filter(|&&ratio| f0 * ratio < 0.45 * fs)
                    .map(|&ratio| spec.object_decay * ratio.powf(-OBJECT_DECAY_EXPONENT))
                    .fold(0.0f32, f32::max);
                (t60 * fs).round() as u64
            })
            .collect();
        let mut impact_ratios = spec.impact_modes.eval(
            (extent_hz.0 * extent_hz.1).sqrt(),
            &landscape.space,
            Some(landscape),
            &mut rng,
        );
        impact_ratios.retain(|&ratio| ratio.is_finite() && ratio > 0.0);
        impact_ratios.truncate(IMPACT_MODES);
        let energy: f32 = (0..impact_ratios.len())
            .map(|k| IMPACT_BRIGHTNESS.powi(2 * k as i32))
            .sum();
        let impact_norm = if energy > 0.0 {
            energy.sqrt().recip()
        } else {
            0.0
        };
        let observed = spec.leave_drops_out && spec.routing.to_habitat;
        let mut rain = Self {
            id,
            own_score_eff_scan: if observed {
                Vec::with_capacity(landscape.space.n_bins())
            } else {
                Vec::new()
            },
            drops_habitat_pcm: if observed { vec![0.0; hop] } else { Vec::new() },
            rendered: false,
            trace_enabled: false,
            impact_trace: Vec::new(),
            field_trace: FieldTrace::default(),
            drive: vec![0.0; objects.len()],
            sample_clock: 0,
            object_struck_at: vec![None; objects.len()],
            object_silent_after,
            object_gain,
            objects,
            object_ratios,
            object_target_log2: object_log2.clone(),
            object_log2,
            mode_scratch,
            target,
            extent_hz,
            fs,
            impact_budget: 0.0,
            rate_per_sample: 0.0,
            sway: super::sway::Sway::new(&mut rng),
            meter: MeterWeight::default(),
            #[cfg(test)]
            impacts_struck: 0,
            #[cfg(test)]
            impacts_stolen: 0,
            #[cfg(test)]
            stolen_db_hist: [0; 81],
            rng,
            bin_log2: Vec::new(),
            cdf: Vec::new(),
            bin_step_log2: 0.0,
            bin_ring: Vec::new(),
            impacts: vec![
                Impact {
                    modes: [ImpactMode::default(); IMPACT_MODES],
                    left: 0,
                    attack_len: 0,
                    attack_pos: 0,
                    #[cfg(test)]
                    start_energy: 0.0,
                };
                IMPACT_SLOTS
            ],
            impact_ratios,
            impact_norm,
            amp: Ramp::new(spec.amp),
            gain: Ramp::new(0.0),
            releasing: false,
            spec,
        };
        rain.impact_budget = rain.exponential();
        rain.advance(landscape);
        rain.gain.to(1.0, rain.spec.fade_sec, fs);
        rain
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Whether the Rain reads a field without its own impacts. Presentation-only Rain
    /// has no impacts in the habitat to remove.
    pub(crate) fn needs_observer(&self) -> bool {
        self.spec.leave_drops_out && self.spec.routing.to_habitat
    }

    pub(crate) fn drops_habitat_pcm(&self) -> &[f32] {
        &self.drops_habitat_pcm
    }

    /// Sounding object fundamentals in Hz.
    pub(crate) fn object_hz(&self) -> impl Iterator<Item = f32> + '_ {
        self.object_log2.iter().map(|x| x.exp2())
    }

    pub(crate) fn apply_patch(&mut self, patch: &RainPatch) {
        if let Some(amp) = patch.amp {
            self.spec.amp = amp;
            self.amp.to(amp, SLEW_SEC, self.fs);
        }
        if let Some(rate) = patch.rate {
            self.spec.rate = rate;
        }
        if let Some(sway) = patch.sway {
            self.spec.sway = sway;
        }
        if let Some(hit) = patch.hit {
            self.spec.hit = hit;
        }
        if let Some(width) = patch.width {
            self.spec.width = width;
        }
        if let Some(ring) = patch.ring {
            self.spec.ring = ring;
        }
        if let Some(focus) = patch.follow_meter {
            self.spec.follow_meter = focus;
        }
    }

    /// Read the habitat meter for the next render. With positive `follow_meter` the
    /// impact rate gathers where the beat and, when detected, its subdivision and
    /// measure are expected. Each level's confidence (the resultant of heard onset
    /// phases) gives the von Mises concentration with that resultant, scaled by the
    /// focus; the weight has mean 1 over a cycle of the slowest level, so only impact
    /// times move. An unknown, stale or unconfident meter leaves the rate unweighted.
    pub(crate) fn observe_meter(&mut self, observation: TemporalObservation, now: u64) {
        self.meter.count = 0;
        let focus = self.spec.follow_meter;
        if focus <= 0.0 || !observation.known || observation.observed_end != Some(now) {
            return;
        }
        let state = observation.state;
        let mut slowest_hz = f32::INFINITY;
        let mut phases = [(0.0f32, 0.0f32); 3];
        for (band, detected) in [
            (state.beat, true),
            (state.subdivision, state.subdivision_ratio > 0),
            (state.measure, state.measure_ratio > 0),
        ] {
            let kappa = (focus * circular_concentration(band.confidence)).min(MAX_METER_KAPPA);
            if !detected
                || kappa <= 0.0
                || !(band.freq_hz.is_finite() && band.freq_hz > 0.0)
                || !band.phase.is_finite()
            {
                continue;
            }
            let step = std::f64::consts::TAU * f64::from(band.freq_hz) / f64::from(self.fs);
            let phase = f64::from(band.phase);
            let n = self.meter.count;
            self.meter.levels[n] = MeterLevel {
                kappa,
                re: phase.cos(),
                im: phase.sin(),
                w_re: step.cos(),
                w_im: step.sin(),
            };
            phases[n] = (band.phase, band.freq_hz);
            self.meter.count += 1;
            slowest_hz = slowest_hz.min(band.freq_hz);
        }
        if self.meter.count == 0 {
            return;
        }
        // Every level is an integer multiple of the slowest, so one of its cycles is a
        // common period; the midpoint rule is spectrally accurate for it.
        let mut sum = 0.0f64;
        for j in 0..METER_GRID {
            let t = (j as f32 + 0.5) / (METER_GRID as f32 * slowest_hz);
            let exponent: f32 = self.meter.levels[..self.meter.count]
                .iter()
                .zip(&phases)
                .map(|(level, &(phase, hz))| level.kappa * ((phase + TAU * hz * t).cos() - 1.0))
                .sum();
            sum += f64::from(exponent.exp());
        }
        self.meter.inv_norm = (METER_GRID as f64 / sum) as f32;
        self.meter.beat_phase = state.beat.phase;
        self.meter.beat_step = TAU * state.beat.freq_hz / self.fs;
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
        self.releasing && self.gain.value == 0.0
    }

    /// Rebuild the impact distribution over the extent from the current Landscape and
    /// propose each object's next pitch. With `leave_drops_out` the field comes from the
    /// Rain's own observation, which excludes its impacts but keeps its objects.
    pub(crate) fn advance(&mut self, landscape: &Landscape) {
        let observation = if self.needs_observer() && self.rendered {
            let Some(observation) = landscape
                .rain_observations
                .iter()
                .find(|observation| observation.id == self.id)
            else {
                // A gap or analysis restart: keep the previous tables rather than fall
                // back to the mixed field, as Flora does.
                self.field_trace.source = "held";
                return;
            };
            Some(observation)
        } else {
            None
        };
        let space = &landscape.space;
        let mut own_scores = std::mem::take(&mut self.own_score_eff_scan);
        if let Some(observation) = observation {
            space.assert_scan_len_named(&observation.score_scan, "rain_score_scan");
            space.assert_scan_len_named(&observation.mass_scan, "rain_mass_scan");
            space.assert_scan_len_named(
                &landscape.perc_habituation_state_scan,
                "perc_habituation_state_scan",
            );
            // The shared habituation of the mixed habitat erodes the own field, as for
            // Voice self-exclusion.
            own_scores.clear();
            own_scores.extend(
                observation
                    .score_scan
                    .iter()
                    .zip(&landscape.perc_habituation_state_scan)
                    .map(|(&score, &h)| erode_score(score, h, landscape.consonance_theta)),
            );
        }
        let scores: &[f32] = if observation.is_some() {
            &own_scores
        } else {
            &landscape.consonance_field_score_eff
        };
        self.propose_object_pitches(landscape, scores);
        self.bin_step_log2 = space.step();
        let lo = space.nearest_index(self.extent_hz.0);
        let hi = space.nearest_index(self.extent_hz.1).max(lo);
        let gap_ref = if self.target == FieldTarget::Gap {
            (lo..=hi)
                .map(|i| {
                    landscape
                        .subjective_intensity
                        .get(i)
                        .copied()
                        .unwrap_or(0.0)
                })
                .fold(0.0f32, f32::max)
        } else {
            0.0
        };
        if self.trace_enabled {
            let trace = &mut self.field_trace;
            trace.source = if observation.is_some() {
                "observed"
            } else {
                "mixed"
            };
            trace.first_bin = lo;
            trace.raw_score.clear();
            trace.raw_score.extend_from_slice(match observation {
                Some(observation) => &observation.score_scan[lo..=hi],
                None => &landscape.consonance_field_score[lo..=hi],
            });
            trace.eff_score.clear();
            trace.eff_score.extend_from_slice(&scores[lo..=hi]);
            trace.mass.clear();
        }
        self.bin_log2.clear();
        self.cdf.clear();
        let mut total = 0.0f32;
        for i in lo..=hi {
            let mass = match observation {
                None => field_density_mass(self.target, landscape, i, gap_ref),
                // Gap reads mixed occupancy: a place is free only if impacts left it free.
                Some(observation) => field_target_mass(
                    self.target,
                    erode_mass(
                        observation.mass_scan[i],
                        landscape.perc_habituation_state_scan[i],
                    ),
                    landscape.consonance_representation.level(scores[i]),
                    landscape
                        .subjective_intensity
                        .get(i)
                        .copied()
                        .unwrap_or(gap_ref),
                    gap_ref,
                ),
            };
            let mass = if mass.is_finite() { mass.max(0.0) } else { 0.0 };
            if self.trace_enabled {
                self.field_trace.mass.push(mass);
            }
            total += mass;
            self.bin_log2.push(space.centers_log2[i]);
            self.cdf.push(total);
        }
        // Zero in-range mass falls back to log-uniform, as spawn placement does.
        if total <= 0.0 {
            for (k, c) in self.cdf.iter_mut().enumerate() {
                *c = (k + 1) as f32;
            }
        }
        self.bin_ring.clear();
        if self.spec.ring > 0.0 {
            let scores = &scores[lo..=hi];
            let (min, max) = scores
                .iter()
                .filter(|s| s.is_finite())
                .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &s| {
                    (min.min(s), max.max(s))
                });
            let span = max - min;
            self.bin_ring.extend(scores.iter().map(|&s| {
                let u = (s - min) / span;
                if span > 0.0 && u.is_finite() { u } else { 0.0 }
            }));
        }
        self.own_score_eff_scan = own_scores;
    }

    /// Each object proposes the best eroded consonance score within one semitone of its
    /// sounding pitch, less the Voice movement cost, keeping 1 ERB from the other objects.
    /// The object moves only when it is next struck. `scores` is an eroded score scan.
    fn propose_object_pitches(&mut self, landscape: &Landscape, scores: &[f32]) {
        let space = &landscape.space;
        let objective = landscape.pitch_objective_mode;
        let (ext_lo, ext_hi) = (self.extent_hz.0.log2(), self.extent_hz.1.log2());
        if self.spec.anchored {
            return;
        }
        for k in 0..self.object_log2.len() {
            let current = self.object_log2[k];
            let score_at = |x: f32| {
                objective.apply_consonance_score(landscape.sample_linear_log2(scores, x))
                    - OBJECT_MOVE_COST_PER_OCT * (x - current).abs()
            };
            let lo = space.nearest_index((current - OBJECT_STEP_OCT).max(ext_lo).exp2());
            let hi = space.nearest_index((current + OBJECT_STEP_OCT).min(ext_hi).exp2());
            let base = score_at(current);
            let mut best = (current, base);
            for i in lo..=hi {
                let x = space.centers_log2[i];
                let spaced = self.object_log2.iter().enumerate().all(|(j, &other)| {
                    j == k
                        || (hz_to_erb(x.exp2()) - hz_to_erb(other.exp2())).abs()
                            >= OBJECT_MIN_SPACING_ERB
                });
                let score = score_at(x);
                if spaced
                    && (current - x).abs() <= OBJECT_STEP_OCT
                    && score.is_finite()
                    && score > best.1
                {
                    best = (x, score);
                }
            }
            self.object_target_log2[k] = if best.1 - base > OBJECT_MIN_IMPROVEMENT {
                best.0
            } else {
                current
            };
        }
    }

    fn exponential(&mut self) -> f32 {
        let u: f32 = self.rng.random::<f32>().max(f32::MIN_POSITIVE);
        -u.ln()
    }

    /// Advance the sway by `dt` seconds and set the impact rate it gives:
    /// `rate * 2^(sway * g)`, where `g` has unit variance and a 1/f spectrum.
    fn advance_rate(&mut self, dt: f32) {
        let g = self.sway.advance(dt, &mut self.rng);
        let rate = (self.spec.rate * (self.spec.sway * g).exp2()).clamp(0.0, MAX_RAIN_RATE);
        self.rate_per_sample = rate / self.fs;
    }

    fn normal(&mut self) -> f32 {
        let u1: f32 = self.rng.random::<f32>().max(f32::MIN_POSITIVE);
        let u2: f32 = self.rng.random();
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }

    /// Draw a centre frequency from the current distribution.
    fn draw_centre_hz(&mut self) -> f32 {
        let total = *self.cdf.last().unwrap_or(&0.0);
        if total <= 0.0 || self.bin_log2.is_empty() {
            let (lo, hi) = (self.extent_hz.0.log2(), self.extent_hz.1.log2());
            return (lo + (hi - lo) * self.rng.random::<f32>()).exp2();
        }
        let u = self.rng.random::<f32>() * total;
        let k = self
            .cdf
            .partition_point(|&c| c <= u)
            .min(self.cdf.len() - 1);
        let jitter = (self.rng.random::<f32>() - 0.5) * self.bin_step_log2;
        (self.bin_log2[k] + jitter).exp2()
    }

    fn strike(&mut self) -> ImpactRecord {
        let centre = self.draw_centre_hz();
        let level_db = (self.normal() * IMPACT_LEVEL_SPREAD_DB).min(IMPACT_LEVEL_CAP_DB);
        let level = 10f32.powf(level_db / 20.0);
        let tau = IMPACT_TAU_REF_SEC
            * (IMPACT_TAU_REF_HZ / centre).sqrt()
            * (self.normal() * IMPACT_TAU_SPREAD).exp();
        let attack_sec = IMPACT_ATTACK_MIN_SEC
            + (IMPACT_ATTACK_MAX_SEC - IMPACT_ATTACK_MIN_SEC) * self.rng.random::<f32>();
        let attack_len = (attack_sec * self.fs).round().max(1.0) as usize;
        let slot = self
            .impacts
            .iter()
            .enumerate()
            .min_by_key(|(_, impact)| impact.left)
            .map(|(k, _)| k)
            .expect("impact pool");
        #[cfg(test)]
        {
            let stolen = &self.impacts[slot];
            if stolen.left > 0 {
                self.impacts_stolen += 1;
                let energy: f32 = stolen.modes[..self.impact_ratios.len()]
                    .iter()
                    .map(|m| m.re * m.re + m.im * m.im)
                    .sum();
                let below_db = -10.0 * (energy / stolen.start_energy).log10();
                self.stolen_db_hist[below_db.round().clamp(0.0, 80.0) as usize] += 1;
            }
        }
        #[cfg(test)]
        let mut start_energy = 0.0f32;
        // One decay factor for the whole impact, read where its first mode sounds, so
        // every impact keeps one material.
        let mut ring_gain = 1.0f32;
        let mut first_mode_hz = 0.0f32;
        let mut longest = attack_len;
        for k in 0..self.impact_ratios.len() {
            let ratio = self.impact_ratios[k];
            let offset = (2.0 * self.rng.random::<f32>() - 1.0) * self.spec.width;
            let phase = TAU * self.rng.random::<f32>();
            let mode_db = self.normal() * IMPACT_MODE_LEVEL_SPREAD_DB;
            let freq = centre * ratio * offset.exp2();
            if k == 0 {
                first_mode_hz = freq;
            }
            let mode = &mut self.impacts[slot].modes[k];
            if !(20.0..0.45 * self.fs).contains(&freq) {
                *mode = ImpactMode::default();
                continue;
            }
            if k == 0 && self.spec.ring > 0.0 && !self.bin_ring.is_empty() {
                let idx = ((freq.log2() - self.bin_log2[0]) / self.bin_step_log2).round();
                if idx >= 0.0 && (idx as usize) < self.bin_ring.len() {
                    ring_gain = (self.spec.ring * self.bin_ring[idx as usize]).exp2();
                }
            }
            let mode_tau = tau * ring_gain / ratio.sqrt();
            let decay = (-1.0 / (mode_tau * self.fs)).exp();
            let amplitude = level
                * IMPACT_BRIGHTNESS.powi(k as i32)
                * 10f32.powf(mode_db / 20.0)
                * self.impact_norm;
            let (sin, cos) = (TAU * freq / self.fs).sin_cos();
            *mode = ImpactMode {
                re: amplitude * phase.cos(),
                im: amplitude * phase.sin(),
                w_re: decay * cos,
                w_im: decay * sin,
            };
            let life = (amplitude / IMPACT_FLOOR).ln() * mode_tau * self.fs;
            longest = longest.max(life.ceil().max(1.0) as usize);
            #[cfg(test)]
            {
                start_energy += amplitude * amplitude;
            }
        }
        let impact = &mut self.impacts[slot];
        impact.left = longest;
        impact.attack_len = attack_len;
        impact.attack_pos = 0;
        #[cfg(test)]
        {
            impact.start_energy = start_energy;
        }
        let mut object = None;
        let (mut object_hz, mut replaced, mut replace_blocked) = (None, false, false);
        if !self.objects.is_empty() && self.rng.random::<f32>() < self.spec.hit {
            let k = self.rng.random_range(0..self.objects.len());
            object = Some(k);
            if self.spec.objects_follow_drops
                && !self.spec.anchored
                && self.object_struck_at[k]
                    .is_none_or(|last| self.sample_clock - last >= self.object_silent_after[k])
            {
                // A silent object holds no pitch: it takes the centre of the impact that
                // strikes it, unless that lies within 1 ERB of another object.
                let x = centre
                    .log2()
                    .clamp(self.extent_hz.0.log2(), self.extent_hz.1.log2());
                let erb = hz_to_erb(x.exp2());
                let spaced = self.object_log2.iter().enumerate().all(|(j, &other)| {
                    j == k || (erb - hz_to_erb(other.exp2())).abs() >= OBJECT_MIN_SPACING_ERB
                });
                if spaced {
                    self.object_target_log2[k] = x;
                    replaced = true;
                } else {
                    replace_blocked = true;
                }
            }
            let target = self.object_target_log2[k];
            if target != self.object_log2[k] {
                self.object_log2[k] = target;
                fill_object_modes(
                    &self.spec,
                    target.exp2(),
                    &self.object_ratios[k],
                    self.fs,
                    &mut self.mode_scratch,
                );
                self.objects[k]
                    .set_modes_preserve_state(&self.mode_scratch)
                    .expect("modes within capacity");
            }
            self.object_struck_at[k] = Some(self.sample_clock);
            self.drive[k] += level * self.object_gain[k];
            object_hz = Some(self.object_log2[k].exp2());
        }
        ImpactRecord {
            offset: 0,
            centre_hz: centre,
            first_mode_hz,
            ring_gain,
            level,
            object,
            object_hz,
            replaced,
            replace_blocked,
            meter_weight: None,
            beat_phase: None,
        }
    }

    /// Render one hop and add it to the buses selected by `send()`. A Rain that needs an
    /// observer also keeps its impacts' habitat samples, scaled exactly as added.
    pub(crate) fn render_add(&mut self, presentation: &mut [f32], habitat: &mut [f32]) {
        assert_eq!(presentation.len(), habitat.len());
        let capture = self.needs_observer();
        if capture {
            assert_eq!(
                self.drops_habitat_pcm.len(),
                presentation.len(),
                "Rain impact capture hop"
            );
        }
        self.advance_rate(presentation.len() as f32 / self.fs);
        let weighted = self.meter.count > 0;
        let mut weight = 1.0f32;
        for t in 0..presentation.len() {
            if weighted {
                weight = self.meter.next();
                self.impact_budget -= (self.rate_per_sample * weight).min(MAX_RAIN_RATE / self.fs);
            } else {
                self.impact_budget -= self.rate_per_sample;
            }
            while self.impact_budget <= 0.0 {
                let record = self.strike();
                if self.trace_enabled {
                    let beat_phase =
                        (self.meter.beat_phase + self.meter.beat_step * t as f32) / TAU;
                    self.impact_trace.push(ImpactRecord {
                        offset: t,
                        meter_weight: weighted.then_some(weight),
                        beat_phase: weighted.then_some(beat_phase.rem_euclid(1.0)),
                        ..record
                    });
                }
                #[cfg(test)]
                {
                    self.impacts_struck += 1;
                }
                self.impact_budget += self.exponential();
            }
            let mut out = 0.0f32;
            let modes = self.impact_ratios.len();
            for impact in &mut self.impacts {
                if impact.left == 0 {
                    continue;
                }
                impact.left -= 1;
                let mut sum = 0.0f32;
                for mode in &mut impact.modes[..modes] {
                    let re = mode.re * mode.w_re - mode.im * mode.w_im;
                    let im = mode.re * mode.w_im + mode.im * mode.w_re;
                    (mode.re, mode.im) = (re, im);
                    sum += im;
                }
                if impact.attack_pos < impact.attack_len {
                    let x = impact.attack_pos as f32 / impact.attack_len as f32;
                    sum *= 0.5 - 0.5 * (PI * x).cos();
                    impact.attack_pos += 1;
                }
                out += sum;
            }
            let drops = out;
            for (bank, drive) in self.objects.iter_mut().zip(&mut self.drive) {
                out += bank.process_sample(*drive);
                *drive = 0.0;
            }
            let (amp, gain) = (self.amp.tick(), self.gain.tick());
            let sample = out * amp * gain;
            if capture {
                self.drops_habitat_pcm[t] = drops * amp * gain;
            }
            if self.spec.routing.to_presentation {
                presentation[t] += sample;
            }
            if self.spec.routing.to_habitat {
                habitat[t] += sample;
            }
            self.sample_clock += 1;
        }
        self.rendered = true;
    }
}

/// Object pitches: eroded consonance-level peaks in the extent, at least 1 ERB apart,
/// filled with log-uniform positions when the Landscape offers too few peaks.
fn object_pitches(
    spec: &RainSpec,
    extent_hz: (f32, f32),
    landscape: &Landscape,
    rng: &mut SmallRng,
) -> Vec<f32> {
    let count = spec.objects.min(MAX_RAIN_OBJECTS) as usize;
    let space = &landscape.space;
    let lo = space.nearest_index(extent_hz.0);
    let hi = space.nearest_index(extent_hz.1).max(lo);
    let level = &landscape.consonance_field_level_eff;
    let mut peaks: Vec<(f32, f32)> = (lo..=hi)
        .filter(|&i| {
            let v = level.get(i).copied().unwrap_or(0.0);
            v > 0.0
                && (i == 0 || level.get(i - 1).copied().unwrap_or(0.0) < v)
                && level.get(i + 1).copied().unwrap_or(0.0) <= v
        })
        .map(|i| (level[i], space.centers_hz[i]))
        .collect();
    peaks.sort_by(|a, b| b.0.total_cmp(&a.0));
    let mut chosen: Vec<f32> = Vec::with_capacity(count);
    let far_enough = |chosen: &[f32], hz: f32| {
        chosen
            .iter()
            .all(|&c| (hz_to_erb(c) - hz_to_erb(hz)).abs() >= OBJECT_MIN_SPACING_ERB)
    };
    for &(_, hz) in &peaks {
        if chosen.len() == count {
            break;
        }
        if far_enough(&chosen, hz) {
            chosen.push(hz);
        }
    }
    let (log_lo, log_hi) = (extent_hz.0.log2(), extent_hz.1.log2());
    let mut attempts = 0;
    while chosen.len() < count {
        let hz = (log_lo + (log_hi - log_lo) * rng.random::<f32>()).exp2();
        attempts += 1;
        if far_enough(&chosen, hz) || attempts > 64 {
            chosen.push(hz);
        }
    }
    chosen
}

/// Drive scale for a unit impulse whose onset peak matches a unit-level impact. An
/// impact of level 1 peaks near 1, so a struck object starts as loud as its impact.
fn onset_gain(mut bank: ResonatorBank) -> f32 {
    let peak = std::iter::once(bank.process_sample(1.0))
        .chain((1..OBJECT_ONSET_WINDOW).map(|_| bank.process_sample(0.0)))
        .fold(0.0f32, |peak, y| peak.max(y.abs()));
    if peak > 0.0 { 1.0 / peak } else { 0.0 }
}

/// Object modes at `f0`. Modes at or above 0.45 fs are muted rather than dropped, so a
/// retuned bank keeps its mode count and state.
fn fill_object_modes(spec: &RainSpec, f0: f32, ratios: &[f32], fs: f32, out: &mut Vec<ModeParams>) {
    out.clear();
    out.extend(ratios.iter().enumerate().map(|(k, &ratio)| {
        let freq_hz = f0 * ratio;
        let audible = freq_hz < 0.45 * fs;
        ModeParams {
            freq_hz: freq_hz.min(0.45 * fs),
            t60_s: spec.object_decay * ratio.powf(-OBJECT_DECAY_EXPONENT),
            gain: if audible { 0.5f32.powi(k as i32) } else { 0.0 },
            in_gain: if audible { 1.0 } else { 0.0 },
        }
    }));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::log2space::Log2Space;

    fn landscape() -> Landscape {
        Landscape::new(Log2Space::new(55.0, 8000.0, 96))
    }

    fn render(rain: &mut Rain, hops: usize) -> Vec<f32> {
        let mut out = Vec::new();
        for _ in 0..hops {
            let (mut p, mut h) = (vec![0.0; 512], vec![0.0; 512]);
            rain.render_add(&mut p, &mut h);
            out.extend(p);
        }
        out
    }

    #[test]
    fn same_seed_renders_identically_and_seeds_differ() {
        let l = landscape();
        let spec = RainSpec::default();
        let mut a = Rain::new(
            1,
            spec.clone(),
            (300.0, 6000.0),
            FieldTarget::Uniform,
            21,
            &l,
            48_000.0,
            512,
        );
        let mut b = Rain::new(
            1,
            spec.clone(),
            (300.0, 6000.0),
            FieldTarget::Uniform,
            21,
            &l,
            48_000.0,
            512,
        );
        let mut c = Rain::new(
            1,
            spec,
            (300.0, 6000.0),
            FieldTarget::Uniform,
            22,
            &l,
            48_000.0,
            512,
        );
        let (x, y, z) = (render(&mut a, 40), render(&mut b, 40), render(&mut c, 40));
        assert!(x.iter().any(|&v| v != 0.0));
        assert!(x.iter().all(|v| v.is_finite()));
        assert_eq!(x, y);
        assert_ne!(x, z);
    }

    #[test]
    fn impacts_follow_the_landscape_mass() {
        let mut l = landscape();
        let i = l.space.nearest_index(1000.0);
        l.consonance_density_mass_eff.fill(0.0);
        l.consonance_density_mass_eff[i] = 1.0;
        let mut rain = Rain::new(
            1,
            RainSpec::default(),
            (300.0, 6000.0),
            FieldTarget::Consonance,
            21,
            &l,
            48_000.0,
            512,
        );
        for _ in 0..200 {
            let hz = rain.draw_centre_hz();
            assert!((hz.log2() - 1000f32.log2()).abs() <= rain.bin_step_log2);
        }
        let mut free = Rain::new(
            1,
            RainSpec::default(),
            (300.0, 6000.0),
            FieldTarget::Uniform,
            21,
            &l,
            48_000.0,
            512,
        );
        let draws: Vec<f32> = (0..400).map(|_| free.draw_centre_hz()).collect();
        assert!(draws.iter().all(|&hz| (295.0..=6100.0).contains(&hz)));
        assert!(draws.iter().any(|&hz| hz < 800.0) && draws.iter().any(|&hz| hz > 2500.0));
    }

    #[test]
    fn objects_take_consonance_peaks_and_ring_only_when_hit() {
        let mut l = landscape();
        l.consonance_field_level_eff.fill(0.0);
        for hz in [220.0, 330.0, 440.0] {
            let i = l.space.nearest_index(hz);
            l.consonance_field_level_eff[i] = 0.9;
        }
        let spec = RainSpec {
            objects: 3,
            ..RainSpec::default()
        };
        let pitches = object_pitches(&spec, (150.0, 2000.0), &l, &mut SmallRng::seed_from_u64(1));
        for hz in [220.0f32, 330.0, 440.0] {
            assert!(
                pitches.iter().any(|&p| (p.log2() - hz.log2()).abs() < 0.02),
                "{pitches:?}"
            );
        }
        let silent = RainSpec {
            hit: 0.0,
            ..spec.clone()
        };
        let mut rain = Rain::new(
            1,
            silent,
            (150.0, 2000.0),
            FieldTarget::Uniform,
            21,
            &l,
            48_000.0,
            512,
        );
        render(&mut rain, 40);
        assert!(
            rain.objects
                .iter()
                .all(|bank| bank.mode_state(0).unwrap()[..2] == [0.0, 0.0])
        );
        let mut struck = Rain::new(
            1,
            RainSpec { hit: 1.0, ..spec },
            (150.0, 2000.0),
            FieldTarget::Uniform,
            21,
            &l,
            48_000.0,
            512,
        );
        render(&mut struck, 40);
        assert!(
            struck
                .objects
                .iter()
                .any(|bank| bank.mode_state(0).unwrap()[1] != 0.0)
        );
    }

    #[test]
    fn struck_objects_drift_at_most_a_semitone_toward_consonance() {
        let mut l = landscape();
        l.consonance_field_level_eff.fill(0.0);
        let start = l.space.nearest_index(400.0);
        l.consonance_field_level_eff[start] = 0.9;
        l.consonance_field_score_eff.fill(0.0);
        let near = l.space.nearest_index(420.0);
        l.consonance_field_score_eff[near] = 1.0;
        let spec = RainSpec {
            objects: 1,
            hit: 0.0,
            ..RainSpec::default()
        };
        let mut rain = Rain::new(
            1,
            spec,
            (150.0, 2000.0),
            FieldTarget::Uniform,
            21,
            &l,
            48_000.0,
            512,
        );
        let placed = rain.object_log2[0];
        assert_eq!(placed, l.space.centers_log2[start]);
        render(&mut rain, 20);
        assert_eq!(
            rain.object_log2[0], placed,
            "unstruck objects keep their pitch"
        );
        assert_eq!(rain.object_target_log2[0], l.space.centers_log2[near]);
        rain.apply_patch(&RainPatch {
            hit: Some(1.0),
            ..RainPatch::default()
        });
        render(&mut rain, 20);
        assert_eq!(rain.object_log2[0], l.space.centers_log2[near]);

        // A better peak two semitones away is out of reach in one move.
        l.consonance_field_score_eff.fill(0.0);
        let far = l.space.nearest_index(420.0 * 2f32.powf(2.0 / 12.0));
        l.consonance_field_score_eff[far] = 1.0;
        rain.advance(&l);
        assert!(
            (rain.object_target_log2[0] - rain.object_log2[0]).abs() <= OBJECT_STEP_OCT,
            "{} -> {}",
            rain.object_log2[0],
            rain.object_target_log2[0]
        );
    }

    #[test]
    fn zero_width_repeats_the_impact_pattern_exactly() {
        let l = landscape();
        let mut rain = Rain::new(
            1,
            RainSpec {
                width: 0.0,
                objects: 0,
                ..RainSpec::default()
            },
            (300.0, 3000.0),
            FieldTarget::Uniform,
            21,
            &l,
            48_000.0,
            512,
        );
        assert_eq!(rain.impact_ratios, [1.0, 2.32, 4.25, 6.63, 9.38]);
        rain.strike();
        let slot = rain
            .impacts
            .iter()
            .find(|i| i.left > 0)
            .expect("struck slot");
        let angle = |m: &ImpactMode| m.w_im.atan2(m.w_re);
        let base = angle(&slot.modes[0]);
        for (mode, ratio) in slot.modes.iter().zip(&rain.impact_ratios) {
            assert!((angle(mode) / base - ratio).abs() < 1e-3, "{ratio}");
        }
    }

    #[test]
    fn anchored_objects_never_propose_a_move() {
        let mut l = landscape();
        l.consonance_field_level_eff.fill(0.0);
        l.consonance_field_level_eff[l.space.nearest_index(400.0)] = 0.9;
        l.consonance_field_score_eff.fill(0.0);
        let near = l.space.nearest_index(420.0);
        l.consonance_field_score_eff[near] = 1.0;
        let mut rain = Rain::new(
            1,
            RainSpec {
                objects: 1,
                hit: 1.0,
                anchored: true,
                ..RainSpec::default()
            },
            (150.0, 2000.0),
            FieldTarget::Uniform,
            21,
            &l,
            48_000.0,
            512,
        );
        let placed = rain.object_log2[0];
        render(&mut rain, 20);
        assert_eq!(rain.object_target_log2[0], placed);
        assert_eq!(rain.object_log2[0], placed);
    }

    #[test]
    fn impact_counts_follow_the_rate_and_its_live_patches() {
        let l = landscape();
        let mut rain = Rain::new(
            1,
            RainSpec {
                rate: 40.0,
                objects: 0,
                ..RainSpec::default()
            },
            (300.0, 6000.0),
            FieldTarget::Uniform,
            21,
            &l,
            48_000.0,
            512,
        );
        render(&mut rain, 5625); // 60 s
        let first = rain.impacts_struck;
        assert!((2155..=2645).contains(&first), "{first}");
        rain.apply_patch(&RainPatch {
            rate: Some(200.0),
            ..RainPatch::default()
        });
        render(&mut rain, 1875); // 20 s
        let second = rain.impacts_struck - first;
        assert!((3684..=4316).contains(&second), "{second}");
    }

    #[test]
    fn sway_spreads_the_rate_by_octaves_with_one_over_f_memory() {
        let l = landscape();
        let make = |sway| {
            Rain::new(
                1,
                RainSpec {
                    sway,
                    objects: 0,
                    ..RainSpec::default()
                },
                (300.0, 6000.0),
                FieldTarget::Uniform,
                21,
                &l,
                48_000.0,
                512,
            )
        };
        let dt = 512.0 / 48_000.0;
        let mut still = make(0.0);
        for _ in 0..100 {
            still.advance_rate(dt);
            assert_eq!(still.rate_per_sample * 48_000.0, 40.0);
        }
        let mut swayed = make(1.0);
        let octaves: Vec<f32> = (0..1_000_000)
            .map(|_| {
                swayed.advance_rate(dt);
                (swayed.rate_per_sample * 48_000.0 / 40.0).log2()
            })
            .collect();
        let n = octaves.len() as f32;
        let mean = octaves.iter().sum::<f32>() / n;
        let var = octaves.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        assert!((0.75..=1.25).contains(&var.sqrt()), "sd {}", var.sqrt());
        let autocorr = |seconds: f32| {
            let lag = (seconds / dt).round() as usize;
            let cov = octaves
                .iter()
                .zip(&octaves[lag..])
                .map(|(a, b)| (a - mean) * (b - mean))
                .sum::<f32>()
                / (n - lag as f32);
            cov / var
        };
        // Equal-variance poles at 0.4 s * 2^k give about 0.70 at 1 s and 0.35 at 10 s.
        let (one, ten) = (autocorr(1.0), autocorr(10.0));
        assert!((0.6..=0.8).contains(&one), "{one}");
        assert!((0.2..=0.5).contains(&ten), "{ten}");
    }

    #[test]
    fn routing_selects_buses_and_release_finishes() {
        let l = landscape();
        let spec = RainSpec {
            routing: crate::scenario::control::Routing {
                to_presentation: true,
                to_habitat: false,
            },
            fade_sec: 0.0,
            ..RainSpec::default()
        };
        let mut rain = Rain::new(
            1,
            spec,
            (300.0, 6000.0),
            FieldTarget::Uniform,
            21,
            &l,
            48_000.0,
            512,
        );
        let (mut p, mut h) = (vec![0.0; 48_000], vec![0.0; 48_000]);
        rain.render_add(&mut p, &mut h);
        assert!(p.iter().any(|&v| v != 0.0));
        assert!(h.iter().all(|&v| v == 0.0));
        rain.release(0.0);
        assert!(rain.is_finished());
    }

    /// Score 0.1 outside half an octave around 1 kHz, rising linearly to 1.0 at 1 kHz, so
    /// most of a 300-6000 Hz extent sits at u = 0 and the top reaches u = 1.
    fn peaked_landscape() -> Landscape {
        let mut l = landscape();
        let top = 1000f32.log2();
        for (s, &x) in l
            .consonance_field_score_eff
            .iter_mut()
            .zip(&l.space.centers_log2)
        {
            *s = 0.1 + 0.9 * (1.0 - (x - top).abs() / 0.5).max(0.0);
        }
        l
    }

    fn ring_rain(l: &Landscape, ring: f32, spec: RainSpec) -> Rain {
        Rain::new(
            1,
            RainSpec { ring, ..spec },
            (300.0, 6000.0),
            FieldTarget::Uniform,
            21,
            l,
            48_000.0,
            512,
        )
    }

    fn decay_tau_sec(mode: &ImpactMode) -> f64 {
        -1.0 / (48_000.0 * (mode.w_re as f64).hypot(mode.w_im as f64).ln())
    }

    #[test]
    fn ring_on_a_flat_or_silent_field_changes_nothing() {
        for (score, ring) in [(0.3, 2.0), (0.0, 4.0)] {
            let mut l = landscape();
            l.consonance_field_score_eff.fill(score);
            let spec = RainSpec {
                rate: 200.0,
                hit: 0.3,
                ..RainSpec::default()
            };
            let (mut a, mut b) = (ring_rain(&l, 0.0, spec.clone()), ring_rain(&l, ring, spec));
            assert!(!b.bin_ring.is_empty() && b.bin_ring.iter().all(|&u| u == 0.0));
            assert_eq!(render(&mut a, 60), render(&mut b, 60), "score {score}");
        }
    }

    #[test]
    fn ring_lengthens_every_mode_by_two_to_the_ring_times_u() {
        let l = peaked_landscape();
        let spec = RainSpec {
            objects: 0,
            ..RainSpec::default()
        };
        let (mut plain, mut rung) = (ring_rain(&l, 0.0, spec.clone()), ring_rain(&l, 2.0, spec));
        let modes = rung.impact_ratios.len();
        let (mut at_floor, mut near_top) = (0, 0);
        for _ in 0..600 {
            for rain in [&mut plain, &mut rung] {
                rain.impacts.iter_mut().for_each(|i| i.left = 0);
                rain.strike();
            }
            let (a, b) = (&plain.impacts[0], &rung.impacts[0]);
            let first = &b.modes[0];
            let f1 = first.w_im.atan2(first.w_re) * 48_000.0 / TAU;
            let pos = (f1.log2() - rung.bin_log2[0]) / rung.bin_step_log2;
            // Skip impacts too close to a bin edge to tell which bin they read.
            if (pos - pos.floor() - 0.5).abs() < 0.01 {
                continue;
            }
            let u = rung
                .bin_ring
                .get(pos.round() as usize)
                .filter(|_| pos.round() >= 0.0)
                .copied()
                .unwrap_or(0.0);
            let expected = (2.0 * u as f64).exp2();
            for (ma, mb) in a.modes[..modes].iter().zip(&b.modes[..modes]) {
                if ma.w_re == 0.0 && ma.w_im == 0.0 {
                    continue;
                }
                let ratio = decay_tau_sec(mb) / decay_tau_sec(ma);
                assert!(
                    (ratio / expected - 1.0).abs() < 0.02,
                    "{ratio} vs {expected}"
                );
                assert!(ratio > 0.99, "{ratio}");
            }
            at_floor += usize::from(u == 0.0);
            near_top += usize::from(u > 0.8);
        }
        assert!(at_floor > 100 && near_top > 10, "{at_floor} {near_top}");
    }

    #[test]
    fn ring_draws_no_random_numbers_and_keeps_object_drives() {
        let l = peaked_landscape();
        let spec = RainSpec {
            rate: 150.0,
            hit: 0.5,
            ..RainSpec::default()
        };
        let (mut a, mut b) = (ring_rain(&l, 0.0, spec.clone()), ring_rain(&l, 2.0, spec));
        assert_ne!(render(&mut a, 200), render(&mut b, 200));
        assert_eq!(a.impacts_struck, b.impacts_struck);
        assert_eq!(a.rng, b.rng);
        assert_eq!(a.object_log2, b.object_log2);
        for (p, q) in a.objects.iter().zip(&b.objects) {
            for k in 0.. {
                let Some(state) = p.mode_state(k) else { break };
                assert_eq!(Some(state), q.mode_state(k));
            }
        }
    }

    #[test]
    fn ring_patched_live_applies_from_the_next_hop() {
        let l = peaked_landscape();
        let spec = RainSpec {
            rate: 200.0,
            objects: 0,
            ..RainSpec::default()
        };
        let (mut base, mut live) = (ring_rain(&l, 0.0, spec.clone()), ring_rain(&l, 0.0, spec));
        let hop = |rain: &mut Rain| {
            rain.advance(&l);
            render(rain, 1)
        };
        assert_eq!(hop(&mut base), hop(&mut live));
        live.apply_patch(&RainPatch {
            ring: Some(2.0),
            ..RainPatch::default()
        });
        assert!(live.bin_ring.is_empty());
        assert_eq!(render(&mut base, 1), render(&mut live, 1));
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for _ in 0..40 {
            x.extend(hop(&mut base));
            y.extend(hop(&mut live));
        }
        assert!(!live.bin_ring.is_empty());
        assert_ne!(x, y);
    }

    #[test]
    fn ring_at_its_bound_on_the_lowest_band_still_decays() {
        let mut l = Landscape::new(Log2Space::new(20.0, 8000.0, 96));
        let low = 2;
        l.consonance_density_mass_eff.fill(0.0);
        l.consonance_density_mass_eff[low] = 1.0;
        l.consonance_field_score_eff.fill(0.0);
        l.consonance_field_score_eff[low] = 1.0;
        let mut rain = Rain::new(
            1,
            RainSpec {
                ring: 4.0,
                width: 0.0,
                objects: 0,
                rate: 0.0,
                ..RainSpec::default()
            },
            (20.0, 2000.0),
            FieldTarget::Consonance,
            21,
            &l,
            48_000.0,
            512,
        );
        let modes = rain.impact_ratios.len();
        let mut longest = 0.0f64;
        for _ in 0..400 {
            rain.impacts.iter_mut().for_each(|i| i.left = 0);
            rain.strike();
            for mode in &rain.impacts[0].modes[..modes] {
                let magnitude = mode.w_re.hypot(mode.w_im);
                assert!(magnitude < 1.0, "{magnitude}");
            }
            longest = longest.max(decay_tau_sec(&rain.impacts[0].modes[0]));
        }
        assert!(
            longest > 5.0,
            "the top of the field was not read: {longest}"
        );
        let energy = |rain: &Rain| -> f32 {
            rain.impacts[0].modes[..modes]
                .iter()
                .map(|m| m.re * m.re + m.im * m.im)
                .sum()
        };
        let (mut p, mut h) = (vec![0.0; 4800], vec![0.0; 4800]);
        let mut last = energy(&rain);
        for _ in 0..50 {
            rain.render_add(&mut p, &mut h);
            assert!(p.iter().all(|v| v.is_finite()));
            let now = energy(&rain);
            assert!(now < last, "{now} >= {last}");
            last = now;
        }
    }

    /// Pool steals at fixed rates on recorded eroded-score scans (RW-4 measurement).
    /// RAIN_RING_SCANS names a text file: "fmin_hz bins_per_octave n_bins", then one scan
    /// per line, cycled once per second. Consonance placement uses u as its mass, a proxy
    /// for the density mass the scans do not record.
    #[test]
    #[ignore = "measurement on recorded scans; set RAIN_RING_SCANS"]
    fn impact_pool_steals_on_recorded_fields() {
        let text = std::fs::read_to_string(std::env::var("RAIN_RING_SCANS").unwrap()).unwrap();
        let rows: Vec<Vec<f32>> = text
            .lines()
            .map(|line| {
                line.split_whitespace()
                    .map(|v| v.parse().unwrap())
                    .collect()
            })
            .collect();
        let (fmin, bpo, n) = (rows[0][0], rows[0][1] as u32, rows[0][2] as usize);
        let space = Log2Space::new(fmin, fmin * ((n as f32 - 0.5) / bpo as f32).exp2(), bpo);
        assert_eq!(space.n_bins(), n);
        let mut l = Landscape::new(space);
        for target in [FieldTarget::Uniform, FieldTarget::Consonance] {
            for rate in [60.0, 120.0, 240.0] {
                for ring in [0.0, 2.0, 4.0] {
                    let mut rain = Rain::new(
                        1,
                        RainSpec {
                            rate,
                            ring,
                            objects: 0,
                            ..RainSpec::default()
                        },
                        (150.0, 5000.0),
                        target,
                        21,
                        &l,
                        48_000.0,
                        512,
                    );
                    let mut sounding = 0u64;
                    let (mut p, mut h) = (vec![0.0; 512], vec![0.0; 512]);
                    for second in 0..60 {
                        let scan = &rows[1 + second % (rows.len() - 1)];
                        l.consonance_field_score_eff.copy_from_slice(scan);
                        let (lo, hi) = scan
                            .iter()
                            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &s| {
                                (lo.min(s), hi.max(s))
                            });
                        for (m, &s) in l.consonance_density_mass_eff.iter_mut().zip(scan) {
                            *m = (s - lo) / (hi - lo);
                        }
                        for _ in 0..94 {
                            rain.advance(&l);
                            rain.render_add(&mut p, &mut h);
                            sounding += rain.impacts.iter().filter(|i| i.left > 0).count() as u64;
                        }
                    }
                    let stolen = rain.impacts_stolen;
                    let quantile = |q: f64| {
                        let mut seen = 0;
                        for (db, &count) in rain.stolen_db_hist.iter().enumerate() {
                            seen += count;
                            if seen as f64 >= q * stolen as f64 {
                                return -(db as i32);
                            }
                        }
                        -80
                    };
                    let above_12: u64 = rain.stolen_db_hist[..=12].iter().sum();
                    println!(
                        "RW4 target={target:?} rate={rate} ring={ring} slots={IMPACT_SLOTS} struck={} stolen={stolen} stolen_frac={:.4} mean_sounding={:.1} stolen_db_p10={} median={} p90={} stolen_above_-12dB_frac={:.4}",
                        rain.impacts_struck,
                        stolen as f64 / rain.impacts_struck as f64,
                        sounding as f64 / (60.0 * 94.0),
                        quantile(0.1),
                        quantile(0.5),
                        quantile(0.9),
                        above_12 as f64 / stolen.max(1) as f64,
                    );
                }
            }
        }
    }

    /// Mixed field peaked at `mixed_hz`; Rain 1's own observation peaked at `own_hz`.
    /// Scores rise from 0.1 to 1.0 within a quarter octave of the peak; density mass is
    /// the score less 0.1, so it is zero away from the peak.
    fn split_landscape(mixed_hz: f32, own_hz: f32) -> Landscape {
        let mut l = landscape();
        let peak = |hz: f32, x: f32| 0.1 + 0.9 * (1.0 - (x - hz.log2()).abs() / 0.25).max(0.0);
        let n = l.space.n_bins();
        let mut own = Vec::with_capacity(n);
        for i in 0..n {
            let x = l.space.centers_log2[i];
            l.consonance_field_score_eff[i] = peak(mixed_hz, x);
            l.consonance_density_mass_eff[i] = peak(mixed_hz, x) - 0.1;
            own.push(peak(own_hz, x));
        }
        l.rain_observations
            .push(crate::core::stream::loo::RainObservation {
                id: 1,
                h_scan: vec![0.0; n],
                r_scan: vec![0.0; n],
                mass_scan: own.iter().map(|s| s - 0.1).collect(),
                score_scan: own,
            });
        l
    }

    fn drops_rain(l: &Landscape, leave_drops_out: bool, spec: RainSpec) -> Rain {
        Rain::new(
            1,
            RainSpec {
                leave_drops_out,
                ..spec
            },
            (300.0, 6000.0),
            FieldTarget::Consonance,
            21,
            l,
            48_000.0,
            512,
        )
    }

    #[test]
    fn captured_drops_equal_the_impacts_added_to_the_habitat() {
        let l = landscape();
        let spec = RainSpec {
            rate: 200.0,
            objects: 0,
            ..RainSpec::default()
        };
        let mut rain = drops_rain(&l, true, spec.clone());
        assert!(rain.needs_observer());
        let mut heard = false;
        for _ in 0..40 {
            let (mut p, mut h) = (vec![0.0; 512], vec![0.0; 512]);
            rain.render_add(&mut p, &mut h);
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(rain.drops_habitat_pcm()), bits(&h));
            heard |= h.iter().any(|&x| x != 0.0);
        }
        assert!(heard);
        let presentation_only = drops_rain(
            &l,
            true,
            RainSpec {
                routing: crate::scenario::control::Routing {
                    to_presentation: true,
                    to_habitat: false,
                },
                ..spec
            },
        );
        assert!(!presentation_only.needs_observer());
        assert!(presentation_only.drops_habitat_pcm().is_empty());
    }

    #[test]
    fn ringing_objects_without_impacts_capture_silence() {
        let l = landscape();
        let mut rain = drops_rain(
            &l,
            true,
            RainSpec {
                rate: 100.0,
                hit: 1.0,
                object_decay: 20.0,
                ..RainSpec::default()
            },
        );
        render(&mut rain, 20);
        rain.apply_patch(&RainPatch {
            rate: Some(0.0),
            ..RainPatch::default()
        });
        let mut hops = 0;
        while rain.impacts.iter().any(|impact| impact.left > 0) {
            render(&mut rain, 1);
            hops += 1;
            assert!(hops < 2000, "impacts never ended");
        }
        let (mut p, mut h) = (vec![0.0; 512], vec![0.0; 512]);
        rain.render_add(&mut p, &mut h);
        assert!(h.iter().any(|&x| x != 0.0), "objects should still ring");
        assert!(rain.drops_habitat_pcm().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn leave_drops_out_reads_its_own_observation_after_the_first_render() {
        let l = split_landscape(1000.0, 2000.0);
        let spec = RainSpec {
            rate: 200.0,
            ring: 2.0,
            objects: 1,
            hit: 0.0,
            ..RainSpec::default()
        };
        let (mut mixed, mut own) = (
            drops_rain(&l, false, spec.clone()),
            drops_rain(&l, true, spec),
        );
        // Before its first render the mixed field holds none of the Rain's own sound.
        mixed.advance(&l);
        own.advance(&l);
        assert_eq!(own.cdf, mixed.cdf);
        assert_eq!(own.bin_ring, mixed.bin_ring);
        for rain in [&mut mixed, &mut own] {
            render(rain, 1);
            rain.object_log2[0] = 1900f32.log2();
            rain.advance(&l);
        }
        let peak_hz = |rain: &Rain| {
            let k = (0..rain.bin_ring.len())
                .max_by(|&a, &b| rain.bin_ring[a].total_cmp(&rain.bin_ring[b]))
                .unwrap();
            rain.bin_log2[k].exp2()
        };
        assert!(
            (peak_hz(&mixed) / 1000.0 - 1.0).abs() < 0.01,
            "{}",
            peak_hz(&mixed)
        );
        assert!(
            (peak_hz(&own) / 2000.0 - 1.0).abs() < 0.01,
            "{}",
            peak_hz(&own)
        );
        for _ in 0..200 {
            let (a, b) = (mixed.draw_centre_hz(), own.draw_centre_hz());
            assert!((a / 1000.0).log2().abs() < 0.26, "{a}");
            assert!((b / 2000.0).log2().abs() < 0.26, "{b}");
        }
        assert_eq!(mixed.object_target_log2[0], mixed.object_log2[0]);
        let moved = own.object_target_log2[0].exp2();
        assert!((moved / 2000.0 - 1.0).abs() < 0.01, "{moved}");
    }

    #[test]
    fn leave_drops_out_moves_impacts_without_changing_times_levels_or_strikes() {
        let l = split_landscape(1000.0, 2000.0);
        let spec = RainSpec {
            rate: 150.0,
            hit: 0.5,
            ring: 2.0,
            ..RainSpec::default()
        };
        let (mut a, mut b) = (
            drops_rain(&l, false, spec.clone()),
            drops_rain(&l, true, spec),
        );
        a.trace_enabled = true;
        b.trace_enabled = true;
        for _ in 0..200 {
            for rain in [&mut a, &mut b] {
                rain.advance(&l);
                render(rain, 1);
            }
        }
        assert_eq!(a.impacts_struck, b.impacts_struck);
        assert_eq!(a.rng, b.rng);
        assert_eq!(a.object_gain, b.object_gain);
        let key = |r: &ImpactRecord| (r.offset, r.level.to_bits(), r.object);
        let keys = |rain: &Rain| rain.impact_trace.iter().map(key).collect::<Vec<_>>();
        assert!(a.impact_trace.len() > 100);
        assert_eq!(keys(&a), keys(&b));
        let centres = |rain: &Rain| {
            rain.impact_trace
                .iter()
                .map(|r| r.centre_hz)
                .collect::<Vec<_>>()
        };
        assert_ne!(centres(&a), centres(&b));
    }

    #[test]
    fn a_missing_observation_keeps_the_previous_tables() {
        let l = split_landscape(1000.0, 2000.0);
        let mut rain = drops_rain(
            &l,
            true,
            RainSpec {
                rate: 200.0,
                ring: 2.0,
                objects: 1,
                hit: 0.0,
                ..RainSpec::default()
            },
        );
        rain.trace_enabled = true;
        render(&mut rain, 1);
        rain.object_log2[0] = 1900f32.log2();
        rain.advance(&l);
        assert_eq!(rain.field_trace.source, "observed");
        let kept = (
            rain.cdf.clone(),
            rain.bin_ring.clone(),
            rain.object_target_log2.clone(),
        );
        let mut gap = l.clone();
        gap.rain_observations.clear();
        gap.consonance_field_score_eff.fill(0.5);
        gap.consonance_density_mass_eff.fill(1.0);
        for _ in 0..5 {
            rain.advance(&gap);
            assert_eq!(rain.field_trace.source, "held");
            assert_eq!(
                (&rain.cdf, &rain.bin_ring, &rain.object_target_log2),
                (&kept.0, &kept.1, &kept.2)
            );
            render(&mut rain, 1);
        }
    }

    fn follow_rain(l: &Landscape, extent_hz: (f32, f32), spec: RainSpec) -> Rain {
        Rain::new(
            1,
            RainSpec {
                objects_follow_drops: true,
                hit: 1.0,
                ..spec
            },
            extent_hz,
            FieldTarget::Uniform,
            21,
            l,
            48_000.0,
            512,
        )
    }

    fn clamped_log2(hz: f32, extent_hz: (f32, f32)) -> f32 {
        hz.log2().clamp(extent_hz.0.log2(), extent_hz.1.log2())
    }

    /// Bank state of a fresh object tuned to `log2_hz`, for comparing coefficients.
    fn tuned_bank(rain: &Rain, k: usize, log2_hz: f32) -> ResonatorBank {
        let mut modes = Vec::new();
        fill_object_modes(
            &rain.spec,
            log2_hz.exp2(),
            &rain.object_ratios[k],
            48_000.0,
            &mut modes,
        );
        let mut bank = ResonatorBank::new(48_000.0, modes.len()).unwrap();
        bank.set_modes_preserve_state(&modes).unwrap();
        bank
    }

    #[test]
    fn a_silent_object_takes_the_centre_of_the_impact_that_strikes_it() {
        let l = landscape();
        let extent = (150.0, 2000.0);
        let mut rain = follow_rain(
            &l,
            extent,
            RainSpec {
                objects: 1,
                ..RainSpec::default()
            },
        );
        // object_decay 2 s on a body whose lowest ratio is 1.
        let silent_after = rain.object_silent_after[0];
        assert_eq!(silent_after, 96_000);
        // An object never struck has no pitch to hold.
        let first = rain.strike();
        assert!(first.replaced && !first.replace_blocked);
        let x = clamped_log2(first.centre_hz, extent);
        assert_eq!(rain.object_log2[0], x);
        assert_eq!(rain.object_target_log2[0], x);
        assert_eq!(first.object_hz, Some(x.exp2()));
        assert_eq!(
            rain.objects[0].mode_state(0).unwrap()[2..4],
            tuned_bank(&rain, 0, x).mode_state(0).unwrap()[2..4],
            "the bank is retuned"
        );
        assert!(rain.drive[0] > 0.0);
        // A full T60 after its last strike the object is silent and follows again.
        rain.sample_clock += silent_after;
        let again = rain.strike();
        assert!(again.replaced);
        assert_eq!(rain.object_log2[0], clamped_log2(again.centre_hz, extent));
        // Every followed pitch stays inside the extent.
        for _ in 0..400 {
            rain.sample_clock += silent_after;
            let record = rain.strike();
            assert!(record.replaced);
            let x = rain.object_log2[0];
            assert_eq!(x, clamped_log2(record.centre_hz, extent));
            assert!((extent.0.log2()..=extent.1.log2()).contains(&x));
        }
    }

    #[test]
    fn a_ringing_object_keeps_the_drift_rule() {
        let l = landscape();
        let extent = (150.0, 2000.0);
        let mut rain = follow_rain(
            &l,
            extent,
            RainSpec {
                objects: 1,
                ..RainSpec::default()
            },
        );
        let silent_after = rain.object_silent_after[0];
        let placed = clamped_log2(rain.strike().centre_hz, extent);
        // One sample before its T60 the object still rings and holds its pitch.
        rain.sample_clock += silent_after - 1;
        let held = rain.strike();
        assert!(!held.replaced && !held.replace_blocked);
        assert_eq!(rain.object_log2[0], placed);
        assert_eq!(held.object_hz, Some(placed.exp2()));
        // A ringing object moves to its drift proposal, as without the word.
        let proposal = placed + 0.05;
        rain.object_target_log2[0] = proposal;
        rain.sample_clock += silent_after - 1;
        let drifted = rain.strike();
        assert!(!drifted.replaced);
        assert_eq!(rain.object_log2[0], proposal);
    }

    #[test]
    fn a_centre_within_one_erb_of_another_object_keeps_the_pitch() {
        let l = landscape();
        // The extent is narrower than 1 ERB, so every centre is too close to the other.
        let extent = (400.0, 420.0);
        let mut rain = follow_rain(
            &l,
            extent,
            RainSpec {
                objects: 2,
                ..RainSpec::default()
            },
        );
        let placed = rain.object_log2.clone();
        for _ in 0..20 {
            rain.drive.fill(0.0);
            rain.sample_clock += rain.object_silent_after[0].max(rain.object_silent_after[1]);
            let record = rain.strike();
            let k = record.object.unwrap();
            assert!(record.replace_blocked && !record.replaced);
            assert_eq!(rain.object_log2, placed);
            assert!(rain.drive[k] > 0.0, "the blocked object is still struck");
        }
    }

    #[test]
    fn anchored_objects_never_follow() {
        let l = landscape();
        let mut rain = follow_rain(
            &l,
            (150.0, 2000.0),
            RainSpec {
                objects: 2,
                anchored: true,
                ..RainSpec::default()
            },
        );
        let placed = rain.object_log2.clone();
        for _ in 0..50 {
            rain.sample_clock += 10 * rain.object_silent_after[0];
            let record = rain.strike();
            assert!(!record.replaced && !record.replace_blocked);
        }
        assert_eq!(rain.object_log2, placed);
    }

    #[test]
    fn following_draws_no_random_numbers() {
        let l = landscape();
        let spec = RainSpec {
            rate: 150.0,
            hit: 0.5,
            objects: 8,
            object_decay: 0.05,
            ..RainSpec::default()
        };
        let make = |follow| {
            let mut rain = Rain::new(
                1,
                RainSpec {
                    objects_follow_drops: follow,
                    ..spec.clone()
                },
                (150.0, 5000.0),
                FieldTarget::Uniform,
                21,
                &l,
                48_000.0,
                512,
            );
            rain.trace_enabled = true;
            rain
        };
        let (mut a, mut b) = (make(false), make(true));
        let (mut x, mut y) = (Vec::new(), Vec::new());
        for _ in 0..200 {
            a.advance(&l);
            b.advance(&l);
            x.extend(render(&mut a, 1));
            y.extend(render(&mut b, 1));
        }
        assert_ne!(x, y);
        assert_eq!(a.impacts_struck, b.impacts_struck);
        assert_eq!(a.rng, b.rng);
        let key = |r: &ImpactRecord| {
            (
                r.offset,
                r.centre_hz.to_bits(),
                r.first_mode_hz.to_bits(),
                r.level.to_bits(),
                r.object,
            )
        };
        let keys = |rain: &Rain| rain.impact_trace.iter().map(key).collect::<Vec<_>>();
        assert!(a.impact_trace.len() > 200);
        assert_eq!(keys(&a), keys(&b));
        assert!(
            a.impact_trace
                .iter()
                .all(|r| !r.replaced && !r.replace_blocked)
        );
        assert!(b.impact_trace.iter().filter(|r| r.replaced).count() > 20);
    }

    /// A meter at 2 Hz with the given beat confidence, its phase advanced to `now`.
    fn meter_at(
        now: u64,
        beat: f32,
        sub: Option<(u8, f32)>,
        measure: Option<(u8, f32)>,
    ) -> TemporalObservation {
        use crate::core::meter::{MeterBand, MeterState};
        let cycles = now as f64 * 2.0 / 48_000.0;
        let band = |ratio: f64, confidence: f32| MeterBand {
            phase: (TAU as f64 * (cycles * ratio).fract()) as f32,
            freq_hz: (2.0 * ratio) as f32,
            amplitude: 1.0,
            confidence,
        };
        let (sub_ratio, sub_conf) = sub.unwrap_or((0, 0.0));
        let (measure_ratio, measure_conf) = measure.unwrap_or((0, 0.0));
        TemporalObservation {
            observed_end: Some(now),
            known: true,
            flux: 0.5,
            state: MeterState {
                beat: band(1.0, beat),
                subdivision: band(f64::from(sub_ratio.max(1)), sub_conf),
                measure: band(1.0 / f64::from(measure_ratio.max(1)), measure_conf),
                subdivision_ratio: sub_ratio,
                measure_ratio,
                ..Default::default()
            },
        }
    }

    fn meter_rain(follow_meter: f32) -> Rain {
        Rain::new(
            1,
            RainSpec {
                rate: 200.0,
                objects: 0,
                follow_meter,
                ..RainSpec::default()
            },
            (300.0, 6000.0),
            FieldTarget::Uniform,
            21,
            &landscape(),
            48_000.0,
            512,
        )
    }

    /// Render `hops`, feeding `observe(now)` before each; returns the output and the
    /// beat phases of the impacts.
    fn render_with_meter(
        rain: &mut Rain,
        hops: usize,
        observe: impl Fn(u64) -> TemporalObservation,
    ) -> (Vec<f32>, Vec<f32>) {
        rain.trace_enabled = true;
        let (mut out, mut phases) = (Vec::new(), Vec::new());
        for n in 0..hops {
            let now = n as u64 * 512;
            rain.observe_meter(observe(now), now);
            let (mut p, mut h) = (vec![0.0; 512], vec![0.0; 512]);
            rain.render_add(&mut p, &mut h);
            out.extend(p);
            phases.extend(rain.impact_trace.drain(..).filter_map(|r| r.beat_phase));
        }
        (out, phases)
    }

    fn resultant(phases: &[f32]) -> f32 {
        let (c, s) = phases.iter().fold((0.0f64, 0.0f64), |(c, s), &p| {
            let a = f64::from(TAU * p);
            (c + a.cos(), s + a.sin())
        });
        ((c * c + s * s).sqrt() / phases.len() as f64) as f32
    }

    #[test]
    fn follow_meter_at_zero_or_without_a_fresh_meter_changes_nothing() {
        let (base, _) = render_with_meter(&mut meter_rain(0.0), 400, |now| {
            meter_at(now, 0.6, None, None)
        });
        // Unknown and stale observations leave the rate unweighted.
        let (unknown, phases) =
            render_with_meter(&mut meter_rain(4.0), 400, |now| TemporalObservation {
                known: false,
                ..meter_at(now, 0.6, None, None)
            });
        assert_eq!(base, unknown);
        assert!(phases.is_empty());
        let (stale, _) = render_with_meter(&mut meter_rain(4.0), 400, |now| TemporalObservation {
            observed_end: now.checked_sub(512),
            ..meter_at(now, 0.6, None, None)
        });
        assert_eq!(base, stale);
        let (silent, _) = render_with_meter(&mut meter_rain(4.0), 400, |now| {
            meter_at(now, 0.0, None, None)
        });
        assert_eq!(base, silent);
    }

    #[test]
    fn follow_meter_one_reproduces_the_heard_concentration_and_keeps_the_mean_rate() {
        // 120 s at 200 impacts per second: about 24,000 impacts.
        let hops = 11_250;
        let mut plain = meter_rain(0.0);
        render_with_meter(&mut plain, hops, |now| meter_at(now, 0.5, None, None));
        for (focus, confidence) in [(1.0, 0.3), (1.0, 0.5), (1.0, 0.8), (2.0, 0.5)] {
            let mut rain = meter_rain(focus);
            let (_, phases) =
                render_with_meter(&mut rain, hops, |now| meter_at(now, confidence, None, None));
            let ratio = rain.impacts_struck as f64 / plain.impacts_struck as f64;
            assert!(
                (ratio - 1.0).abs() < 0.03,
                "focus {focus} c {confidence}: rate ratio {ratio}"
            );
            // Resultant of a von Mises with concentration focus * A1^-1(c).
            let kappa = f64::from(focus * circular_concentration(confidence));
            let a1 = {
                let (mut i0, mut i1) = (0.0, 0.0);
                for j in 0..4096 {
                    let a = std::f64::consts::TAU * (j as f64 + 0.5) / 4096.0;
                    let w = (kappa * (a.cos() - 1.0)).exp();
                    i0 += w;
                    i1 += w * a.cos();
                }
                i1 / i0
            };
            let r = f64::from(resultant(&phases));
            assert!(
                (r - a1).abs() < 0.02,
                "focus {focus} c {confidence}: resultant {r} expected {a1}"
            );
            if focus == 1.0 {
                // Fisher's approximation keeps the heard resultant.
                assert!(
                    (a1 - f64::from(confidence)).abs() < 0.01,
                    "A1(A1^-1({confidence})) = {a1}"
                );
            }
            // The mean phase is the expected beat.
            let mean = phases
                .iter()
                .map(|&p| f64::from(TAU * p).sin())
                .sum::<f64>()
                .atan2(
                    phases
                        .iter()
                        .map(|&p| f64::from(TAU * p).cos())
                        .sum::<f64>(),
                );
            assert!(mean.abs() < 0.05, "mean phase {mean}");
        }
    }

    #[test]
    fn the_meter_weight_has_mean_one_over_a_measure_at_every_focus() {
        for focus in [0.5, 4.0, 64.0, 1e6] {
            let mut rain = meter_rain(focus);
            // Eight measures of three beats at 2 Hz, re-read every hop as in a run.
            let hops = 1125;
            let mut sum = 0.0f64;
            for n in 0..hops as u64 {
                let now = n * 512;
                rain.observe_meter(meter_at(now, 0.7, Some((4, 0.4)), Some((3, 0.5))), now);
                assert_eq!(rain.meter.count, 3);
                assert!(rain.meter.levels.iter().all(|l| l.kappa <= MAX_METER_KAPPA));
                sum += (0..512).map(|_| f64::from(rain.meter.next())).sum::<f64>();
            }
            let mean = sum / (hops * 512) as f64;
            assert!(
                (mean - 1.0).abs() < 1e-3,
                "focus {focus}: mean weight {mean}"
            );
        }
    }

    #[test]
    fn follow_meter_patched_live_applies_from_the_next_hop() {
        let mut live = meter_rain(0.0);
        let mut fixed = meter_rain(3.0);
        let observe = |now| meter_at(now, 0.6, Some((2, 0.3)), Some((4, 0.4)));
        live.apply_patch(&RainPatch {
            follow_meter: Some(3.0),
            ..RainPatch::default()
        });
        let (a, pa) = render_with_meter(&mut live, 200, observe);
        let (b, pb) = render_with_meter(&mut fixed, 200, observe);
        assert_eq!(a, b);
        assert_eq!(pa, pb);
    }

    #[cfg(feature = "profile-alloc")]
    #[test]
    fn steady_state_advance_and_render_allocate_nothing() {
        let l = split_landscape(1000.0, 2000.0);
        for (ring, leave_drops_out, objects_follow_drops, follow_meter) in [
            (0.0, false, false, 0.0),
            (2.0, false, false, 0.0),
            (0.0, true, false, 0.0),
            (2.0, true, false, 0.0),
            (2.0, true, true, 0.0),
            (2.0, true, true, 4.0),
        ] {
            // Frequent strikes on short rings exercise retuning inside the window.
            let (hit, object_decay) = if objects_follow_drops {
                (0.5, 0.05)
            } else {
                (RainSpec::default().hit, RainSpec::default().object_decay)
            };
            let mut rain = Rain::new(
                1,
                RainSpec {
                    ring,
                    leave_drops_out,
                    objects_follow_drops,
                    follow_meter,
                    hit,
                    object_decay,
                    ..RainSpec::default()
                },
                (300.0, 6000.0),
                FieldTarget::Consonance,
                21,
                &l,
                48_000.0,
                512,
            );
            let (mut p, mut h) = (vec![0.0; 512], vec![0.0; 512]);
            rain.observe_meter(meter_at(0, 0.6, Some((2, 0.3)), Some((4, 0.4))), 0);
            rain.advance(&l);
            rain.render_add(&mut p, &mut h);
            let before = rain.object_log2.clone();
            crate::runtime_profile::begin_allocations();
            for n in 1..=64u64 {
                rain.observe_meter(
                    meter_at(n * 512, 0.6, Some((2, 0.3)), Some((4, 0.4))),
                    n * 512,
                );
                rain.advance(&l);
                rain.render_add(&mut p, &mut h);
            }
            let counts = crate::runtime_profile::finish_allocations().unwrap();
            assert_eq!(
                (counts.count, counts.bytes),
                (0, 0),
                "ring {ring} leave_drops_out {leave_drops_out} follow {objects_follow_drops} meter {follow_meter}"
            );
            if objects_follow_drops {
                assert_ne!(rain.object_log2, before, "no object retuned in the window");
            }
            assert_eq!(
                rain.field_trace.source, "",
                "traces stay empty without reports"
            );
        }
    }
}
