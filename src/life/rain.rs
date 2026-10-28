//! Rain: impacts drawn one by one from distributions over a frequency extent, and the
//! resonant objects they strike. Design: `docs/design-notes/rain.md`.

use crate::core::erb::hz_to_erb;
use crate::core::landscape::Landscape;
use crate::life::community::field_density_mass;
use crate::scenario::{FieldTarget, RainPatch, RainSpec};
use crate::synth::modes::ModeParams;
use crate::synth::resonator::ResonatorBank;
use rand::{RngExt, SeedableRng, rngs::SmallRng};
use std::f32::consts::{PI, TAU};

const IMPACT_SLOTS: usize = 64;
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

#[derive(Clone, Copy, Debug, Default)]
struct ImpactMode {
    re: f32,
    im: f32,
    w_re: f32,
    w_im: f32,
}

#[derive(Clone, Debug)]
struct Impact {
    modes: [ImpactMode; IMPACT_MODES],
    /// Samples left before the slot frees.
    left: usize,
    attack_len: usize,
    attack_pos: usize,
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
    /// Remaining unit-exponential budget before the next impact. Subtracting the current
    /// rate per sample (time rescaling) keeps the process Poisson while the rate changes.
    impact_budget: f32,
    /// Impact rate per sample after sway, updated once per rendered block.
    rate_per_sample: f32,
    sway: super::sway::Sway,
    #[cfg(test)]
    impacts_struck: u64,
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
    mode_scratch: Vec<ModeParams>,
    /// Per-object drive scale that makes a strike's onset peak match its impact's level.
    object_gain: Vec<f32>,
    drive: Vec<f32>,
    amp: Ramp,
    gain: Ramp,
    releasing: bool,
}

impl Rain {
    pub(crate) fn new(
        id: u64,
        spec: RainSpec,
        extent_hz: (f32, f32),
        target: FieldTarget,
        scenario_seed: u64,
        landscape: &Landscape,
        fs: f32,
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
        let mut rain = Self {
            id,
            drive: vec![0.0; objects.len()],
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
            #[cfg(test)]
            impacts_struck: 0,
            rng,
            bin_log2: Vec::new(),
            cdf: Vec::new(),
            bin_step_log2: 0.0,
            impacts: vec![
                Impact {
                    modes: [ImpactMode::default(); IMPACT_MODES],
                    left: 0,
                    attack_len: 0,
                    attack_pos: 0,
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

    pub(crate) fn is_finished(&self) -> bool {
        self.releasing && self.gain.value == 0.0
    }

    /// Rebuild the impact distribution over the extent from the current Landscape and
    /// propose each object's next pitch.
    pub(crate) fn advance(&mut self, landscape: &Landscape) {
        self.propose_object_pitches(landscape);
        let space = &landscape.space;
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
        self.bin_log2.clear();
        self.cdf.clear();
        let mut total = 0.0f32;
        for i in lo..=hi {
            let mass = field_density_mass(self.target, landscape, i, gap_ref);
            total += if mass.is_finite() { mass.max(0.0) } else { 0.0 };
            self.bin_log2.push(space.centers_log2[i]);
            self.cdf.push(total);
        }
        // Zero in-range mass falls back to log-uniform, as spawn placement does.
        if total <= 0.0 {
            for (k, c) in self.cdf.iter_mut().enumerate() {
                *c = (k + 1) as f32;
            }
        }
    }

    /// Each object proposes the best eroded consonance score within one semitone of its
    /// sounding pitch, less the Voice movement cost, keeping 1 ERB from the other objects.
    /// The object moves only when it is next struck.
    fn propose_object_pitches(&mut self, landscape: &Landscape) {
        let space = &landscape.space;
        let objective = landscape.pitch_objective_mode;
        let (ext_lo, ext_hi) = (self.extent_hz.0.log2(), self.extent_hz.1.log2());
        if self.spec.anchored {
            return;
        }
        for k in 0..self.object_log2.len() {
            let current = self.object_log2[k];
            let score_at = |x: f32| {
                objective.apply_consonance_score(landscape.evaluate_pitch_score_log2(x))
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

    fn strike(&mut self) {
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
        let mut longest = attack_len;
        for k in 0..self.impact_ratios.len() {
            let ratio = self.impact_ratios[k];
            let offset = (2.0 * self.rng.random::<f32>() - 1.0) * self.spec.width;
            let phase = TAU * self.rng.random::<f32>();
            let mode_db = self.normal() * IMPACT_MODE_LEVEL_SPREAD_DB;
            let freq = centre * ratio * offset.exp2();
            let mode = &mut self.impacts[slot].modes[k];
            if !(20.0..0.45 * self.fs).contains(&freq) {
                *mode = ImpactMode::default();
                continue;
            }
            let mode_tau = tau / ratio.sqrt();
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
        }
        let impact = &mut self.impacts[slot];
        impact.left = longest;
        impact.attack_len = attack_len;
        impact.attack_pos = 0;
        if !self.objects.is_empty() && self.rng.random::<f32>() < self.spec.hit {
            let k = self.rng.random_range(0..self.objects.len());
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
            self.drive[k] += level * self.object_gain[k];
        }
    }

    /// Render one hop and add it to the buses selected by `send()`.
    pub(crate) fn render_add(&mut self, presentation: &mut [f32], habitat: &mut [f32]) {
        assert_eq!(presentation.len(), habitat.len());
        self.advance_rate(presentation.len() as f32 / self.fs);
        for t in 0..presentation.len() {
            self.impact_budget -= self.rate_per_sample;
            while self.impact_budget <= 0.0 {
                self.strike();
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
            for (bank, drive) in self.objects.iter_mut().zip(&mut self.drive) {
                out += bank.process_sample(*drive);
                *drive = 0.0;
            }
            let sample = out * self.amp.tick() * self.gain.tick();
            if self.spec.routing.to_presentation {
                presentation[t] += sample;
            }
            if self.spec.routing.to_habitat {
                habitat[t] += sample;
            }
        }
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
        );
        let mut b = Rain::new(
            1,
            spec.clone(),
            (300.0, 6000.0),
            FieldTarget::Uniform,
            21,
            &l,
            48_000.0,
        );
        let mut c = Rain::new(
            1,
            spec,
            (300.0, 6000.0),
            FieldTarget::Uniform,
            22,
            &l,
            48_000.0,
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
        );
        let (mut p, mut h) = (vec![0.0; 48_000], vec![0.0; 48_000]);
        rain.render_add(&mut p, &mut h);
        assert!(p.iter().any(|&v| v != 0.0));
        assert!(h.iter().all(|&v| v == 0.0));
        rain.release(0.0);
        assert!(rain.is_finished());
    }

    #[cfg(feature = "profile-alloc")]
    #[test]
    fn steady_state_advance_and_render_allocate_nothing() {
        let l = landscape();
        let mut rain = Rain::new(
            1,
            RainSpec::default(),
            (300.0, 6000.0),
            FieldTarget::Consonance,
            21,
            &l,
            48_000.0,
        );
        let (mut p, mut h) = (vec![0.0; 512], vec![0.0; 512]);
        rain.advance(&l);
        rain.render_add(&mut p, &mut h);
        crate::runtime_profile::begin_allocations();
        for _ in 0..64 {
            rain.advance(&l);
            rain.render_add(&mut p, &mut h);
        }
        let counts = crate::runtime_profile::finish_allocations().unwrap();
        assert_eq!((counts.count, counts.bytes), (0, 0));
    }
}
