//! Legacy direct-model v1 density for synchronous birth and respawn selection.

use super::mode_utils::{
    active_cluster_unison, cluster_detune_mul, cluster_gain, cluster_spread_cents_from_public,
    public_spread_from_cluster_spread_cents, sanitize_cluster_unison,
};
use super::spectral::{
    brightness_from_spectral_slope, harmonic_gain, harmonic_ratio, spectral_slope_from_brightness,
};
use super::{BodyKind, BodySnapshot};
use crate::core::consonance_kernel::ConsonanceRepresentationParams;
use crate::core::erb::hz_to_erb;
use crate::core::landscape::LandscapeFrame;
use crate::core::log2space::Log2Space;
use crate::core::mode_pattern::{DEFAULT_MODE_COUNT, ModePattern, ModePatternKind};
use crate::core::timebase::Timebase;
use crate::core::utils::a_weighting_gain_pow;
use crate::life::voice::sound_body::legacy_renderer_kind;
use crate::scenario::control::VoiceControl;
use crate::scenario::{HarmonicMode, RespawnPolicy, TimbreGenotype};
use sha2::{Digest, Sha256};

const MAX_BINS: usize = 2048;
const MAX_RATIOS: usize = 64;
const MAX_UNISON: usize = 9;
const MAX_LANES: usize = MAX_RATIOS * MAX_UNISON;
// R-1's observed same-hop birth load. Entries are reused, never grown in a hop.
const CACHE_ENTRIES: usize = 10;

#[derive(Clone, Copy)]
struct Lane {
    ratio: f32,
    power: f32,
}

struct CandidateCache {
    key: Option<[u8; 32]>,
    masses: Vec<f32>,
}

pub(crate) struct BirthSurrogate {
    time: Timebase,
    exponent: f32,
    ref_power: f32,
    pub(crate) level_repr: ConsonanceRepresentationParams,
    space_key: Option<(u32, u32, u32, usize)>,
    du_scan: Vec<f32>,
    a_power_scan: Vec<f32>,
    power_scan: Vec<f32>,
    density_scan: Vec<f32>,
    terrain_scan: Vec<f32>,
    ratios: Vec<f32>,
    pattern_weights: Vec<f32>,
    pattern_candidates: Vec<(usize, f32)>,
    lanes: Vec<Lane>,
    cache: Vec<CandidateCache>,
    next_cache: usize,
    pub(crate) selection_weights: Vec<(f32, bool)>,
    #[cfg(test)]
    preparations: usize,
}

impl BirthSurrogate {
    pub(crate) fn new(time: Timebase, exponent: f32, ref_power: f32) -> Self {
        Self {
            time,
            exponent: exponent.max(0.01),
            ref_power: ref_power.max(1e-12),
            level_repr: ConsonanceRepresentationParams::default(),
            space_key: None,
            du_scan: Vec::with_capacity(MAX_BINS),
            a_power_scan: Vec::with_capacity(MAX_BINS),
            power_scan: Vec::with_capacity(MAX_BINS),
            density_scan: Vec::with_capacity(MAX_BINS),
            terrain_scan: Vec::with_capacity(MAX_BINS),
            ratios: Vec::with_capacity(MAX_RATIOS),
            pattern_weights: Vec::with_capacity(MAX_BINS),
            pattern_candidates: Vec::with_capacity(MAX_BINS),
            lanes: Vec::with_capacity(MAX_LANES),
            cache: (0..CACHE_ENTRIES)
                .map(|_| CandidateCache {
                    key: None,
                    masses: Vec::with_capacity(MAX_BINS),
                })
                .collect(),
            next_cache: 0,
            selection_weights: Vec::with_capacity(MAX_BINS),
            #[cfg(test)]
            preparations: 0,
        }
    }

    /// The narrow Modal domain has a measured endpoint, not an inferred cutoff.
    /// Sine ignores controls which its renderer does not use.
    pub(crate) fn supports_snapshot(snapshot: &BodySnapshot) -> bool {
        match snapshot.kind {
            BodyKind::Sine => true,
            BodyKind::Harmonic | BodyKind::Modal => {
                snapshot.motion.is_finite()
                    && snapshot.motion == 0.0
                    && snapshot.spread.is_finite()
                    && active_cluster_unison(
                        cluster_spread_cents_from_public(snapshot.spread),
                        snapshot.unison,
                    ) == 1
                    && (snapshot.kind != BodyKind::Modal || snapshot.brightness == 1.0)
            }
        }
    }

    fn set_space(&mut self, space: &Log2Space) -> bool {
        let bins = space.n_bins();
        if !(2..=MAX_BINS).contains(&bins) {
            return false;
        }
        let key = (
            space.fmin.to_bits(),
            space.fmax.to_bits(),
            space.bins_per_oct,
            bins,
        );
        if self.space_key == Some(key) {
            return true;
        }
        self.space_key = None;
        self.du_scan.resize(bins, 0.0);
        self.a_power_scan.resize(bins, 0.0);
        self.power_scan.resize(bins, 0.0);
        self.density_scan.resize(bins, 0.0);
        self.terrain_scan.resize(bins, 0.0);
        for i in 0..bins {
            let left = hz_to_erb(space.centers_hz[i.saturating_sub(1)]);
            let right = hz_to_erb(space.centers_hz[(i + 1).min(bins - 1)]);
            self.du_scan[i] = if i == 0 || i == bins - 1 {
                (right - left).max(0.0)
            } else {
                0.5 * (right - left).max(0.0)
            };
            self.a_power_scan[i] = a_weighting_gain_pow(space.centers_hz[i]);
        }
        if self.du_scan.iter().any(|d| !d.is_finite() || *d <= 0.0) {
            return false;
        }
        self.space_key = Some(key);
        true
    }

    /// Borrowed ratios avoid an Arc allocation for each candidate recipe.
    fn prepare(&mut self, body: &BodySnapshot, ratios: Option<&[f32]>) -> bool {
        self.lanes.clear();
        if !self.time.fs.is_finite()
            || self.time.fs <= 0.0
            || self.time.hop == 0
            || !(1..=MAX_UNISON).contains(&body.unison)
            || [
                body.amp_scale,
                body.brightness,
                body.inharmonic,
                body.spread,
                body.motion,
            ]
            .iter()
            .any(|v| !v.is_finite())
            || ratios.is_some_and(|r| {
                r.is_empty()
                    || r.len() > MAX_RATIOS
                    || r.iter().any(|v| !v.is_finite() || *v <= 0.0)
            })
        {
            return false;
        }
        let amp = body.amp_scale.clamp(0.0, 1.0);
        match body.kind {
            BodyKind::Sine => self.lanes.push(Lane {
                ratio: 1.0,
                power: amp * amp,
            }),
            BodyKind::Harmonic => {
                let genotype = TimbreGenotype {
                    mode: HarmonicMode::Harmonic,
                    stiffness: body.inharmonic.clamp(0.0, 1.0),
                    spectral_slope: spectral_slope_from_brightness(body.brightness),
                    comb: 0.0,
                    damping: 0.5,
                    vibrato_rate: 5.0,
                    vibrato_depth: body.motion.clamp(0.0, 1.0) * 0.02,
                    jitter: body.motion.clamp(0.0, 1.0),
                };
                let spread = cluster_spread_cents_from_public(body.spread);
                let unison = active_cluster_unison(spread, body.unison);
                for idx in 0..ratios.map_or(DEFAULT_MODE_COUNT, |r| r.len()) {
                    let raw = ratios
                        .map_or_else(|| harmonic_ratio(&genotype, idx + 1), |r| r[idx].max(0.1));
                    let gain = cluster_gain(harmonic_gain(&genotype, idx + 1, 1.0), spread, unison);
                    for u in 0..unison {
                        let amplitude = amp * gain;
                        self.lanes.push(Lane {
                            ratio: raw * cluster_detune_mul(spread, unison, u),
                            power: amplitude * amplitude,
                        });
                    }
                }
            }
            BodyKind::Modal => {
                let fallback = [1.0];
                let ratios = ratios.unwrap_or(&fallback);
                let tilt = body.brightness.clamp(0.0, 1.0);
                let tilt_exp = (1.85 - tilt * 1.45).clamp(0.12, 2.2);
                let spread = cluster_spread_cents_from_public(body.spread);
                let unison = active_cluster_unison(spread, body.unison);
                let t = 72.0f64 * self.time.hop as f64 / f64::from(self.time.fs);
                // The first base mode's gain is one, so the legacy peak normalization is neutral.
                for (idx, &ratio) in ratios.iter().enumerate() {
                    let k = (idx + 1) as f32;
                    let gain = cluster_gain(1.0 / k.powf(tilt_exp), spread, unison);
                    let in_gain = (1.0 / (1.0 + 0.04 * k)).max(0.02);
                    let t60 = ((0.35 + tilt * 1.4) / (1.0 + 0.09 * k))
                        .max(0.03)
                        .clamp(0.005, 120.0);
                    let x = 2.0 * 1000.0f64.ln() * t / f64::from(t60);
                    let decay = if x == 0.0 { 1.0 } else { -(-x).exp_m1() / x };
                    let amplitude = amp * gain * in_gain;
                    let power = (f64::from(amplitude) * f64::from(amplitude) * decay) as f32;
                    for u in 0..unison {
                        self.lanes.push(Lane {
                            ratio: ratio * cluster_detune_mul(spread, unison, u),
                            power,
                        });
                    }
                }
                // The original mode compiler sorts the clustered lane frequencies.
                if unison > 1 {
                    self.lanes
                        .sort_unstable_by(|a, b| a.ratio.total_cmp(&b.ratio));
                }
            }
        }
        self.lanes
            .iter()
            .all(|l| l.ratio.is_finite() && l.ratio > 0.0 && l.power.is_finite() && l.power >= 0.0)
    }

    fn density(&mut self, kind: BodyKind, hz: f32, space: &Log2Space) -> Option<f32> {
        self.power_scan.fill(0.0);
        self.density_scan.fill(0.0);
        if !hz.is_finite() || hz <= 0.0 {
            return None;
        }
        let max_hz = (self.time.fs * 0.49).max(1.0);
        for lane in &self.lanes {
            let raw = hz * lane.ratio;
            if !raw.is_finite() || raw <= 0.0 {
                return None;
            }
            let freq = match kind {
                BodyKind::Modal => raw.clamp(1.0, max_hz),
                _ if raw > max_hz => continue,
                _ => raw,
            };
            if let Some(bin) = space.index_of_freq(freq) {
                self.power_scan[bin] += lane.power;
            }
        }
        let mut total = 0.0f64;
        for i in 0..space.n_bins() {
            let power = self.power_scan[i];
            if power == 0.0 {
                continue;
            }
            let d = (power * self.a_power_scan[i] / self.ref_power).powf(self.exponent)
                / self.du_scan[i];
            if !d.is_finite() || d < 0.0 {
                self.density_scan.fill(0.0);
                return None;
            }
            self.density_scan[i] = d;
            total += f64::from(d) * f64::from(self.du_scan[i]);
        }
        let mass = total as f32;
        (mass.is_finite() && mass > 0.0).then_some(mass)
    }

    pub(crate) fn candidates(
        &mut self,
        control: &VoiceControl,
        landscape: &LandscapeFrame,
        frame: u64,
        freq_range: (f32, f32),
        terrain: impl FnMut(usize) -> f32,
    ) -> Option<usize> {
        self.evaluate_candidates(control, landscape, frame, (freq_range, None), terrain)
    }

    pub(crate) fn respawn_scores(
        &mut self,
        control: &VoiceControl,
        policy: RespawnPolicy,
        landscape: &LandscapeFrame,
        frame: u64,
        frequencies: &[f32],
    ) -> Option<usize> {
        let kind = legacy_renderer_kind(control)?;
        // Sixteen proposals amplify the saved Sine Random discrepancy beyond TV 0.1.
        if kind == BodyKind::Sine && matches!(policy, RespawnPolicy::Random) {
            return None;
        }
        if matches!(policy, RespawnPolicy::PeakBiased { .. }) {
            // Fixed-ratio recipes fail local-search TV; their envelope variants share a body.
            // The measured implicit inharmonic and terrain-derived recipes pass.
            if kind != BodyKind::Harmonic
                || control.body.modes.as_ref().is_some_and(|p| {
                    !matches!(
                        p.kind,
                        ModePatternKind::LandscapeDensity | ModePatternKind::LandscapePeaks
                    )
                })
            {
                return None;
            }
        }
        landscape.space.assert_scan_len_named(
            &landscape.consonance_field_score_eff,
            "respawn_consonance_field_score_eff_scan",
        );
        if kind != BodyKind::Sine
            && let Some(pattern) = &control.body.modes
        {
            // Saved final-selection controls fail these domains, although Field mass passes.
            let excluded = match policy {
                RespawnPolicy::Random => matches!(pattern.kind, ModePatternKind::LandscapeDensity),
                RespawnPolicy::Hereditary { .. } => matches!(
                    pattern.kind,
                    ModePatternKind::LandscapeDensity | ModePatternKind::LandscapePeaks
                ),
                _ => false,
            };
            if excluded {
                return None;
            }
        }
        if frequencies.is_empty() || frequencies.len() > MAX_BINS {
            return None;
        }
        self.evaluate_candidates(
            control,
            landscape,
            frame,
            (landscape.freq_bounds(), Some(frequencies)),
            |i| landscape.consonance_field_score_eff[i],
        )
    }

    fn evaluate_candidates(
        &mut self,
        control: &VoiceControl,
        landscape: &LandscapeFrame,
        frame: u64,
        (freq_range, frequencies): ((f32, f32), Option<&[f32]>),
        mut terrain: impl FnMut(usize) -> f32,
    ) -> Option<usize> {
        let kind = legacy_renderer_kind(control)?;
        let t = &control.body.timbre;
        let mut snapshot = BodySnapshot {
            kind,
            amp_scale: 1.0,
            brightness: t.brightness,
            inharmonic: t.inharmonic,
            spread: t.spread,
            unison: t.unison,
            motion: t.motion,
            ratios: None,
        };
        // Match the factory's rendered body, including controls it ignores.
        match kind {
            BodyKind::Sine => {
                snapshot.brightness = 0.0;
                snapshot.inharmonic = 0.0;
                snapshot.spread = 0.0;
                snapshot.unison = 1;
                snapshot.motion = 0.0;
            }
            BodyKind::Harmonic => {
                snapshot.brightness =
                    brightness_from_spectral_slope(spectral_slope_from_brightness(t.brightness));
                snapshot.inharmonic = t.inharmonic.clamp(0.0, 1.0);
                snapshot.motion = t.motion.clamp(0.0, 1.0);
                snapshot.spread = public_spread_from_cluster_spread_cents(
                    cluster_spread_cents_from_public(t.spread),
                );
                snapshot.unison = sanitize_cluster_unison(t.unison);
            }
            BodyKind::Modal => {
                snapshot.brightness = t.brightness.clamp(0.0, 1.0);
                snapshot.inharmonic = 0.0;
                snapshot.motion = 0.0;
                snapshot.spread = public_spread_from_cluster_spread_cents(
                    cluster_spread_cents_from_public(t.spread),
                );
                snapshot.unison = sanitize_cluster_unison(t.unison);
            }
        }
        if !Self::supports_snapshot(&snapshot) {
            return None;
        }
        let space = &landscape.space;
        space.assert_scan_len_named(
            &landscape.consonance_density_mass,
            "birth_raw_density_mass_scan",
        );
        space.assert_scan_len_named(
            &landscape.consonance_field_level,
            "birth_raw_field_level_scan",
        );
        if !self.set_space(space) {
            return None;
        }
        let pattern = if kind == BodyKind::Sine {
            None
        } else {
            control.body.modes.as_ref()
        };
        if kind != BodyKind::Sine
            && pattern.is_some_and(|p| {
                p.count > MAX_RATIOS || p.jitter_cents.is_some_and(|c| !c.is_finite() || c > 0.0)
            })
        {
            return None;
        }
        let bins = space.n_bins();
        let mut hash = Sha256::new();
        hash.update(frame.to_le_bytes());
        hash.update([u8::from(frequencies.is_some())]);
        if let Some(frequencies) = frequencies {
            for &hz in frequencies {
                hash.update(hz.to_bits().to_le_bytes());
            }
        }
        hash.update([kind as u8]);
        for v in [
            space.fmin,
            space.fmax,
            freq_range.0,
            freq_range.1,
            snapshot.brightness,
            snapshot.inharmonic,
            snapshot.motion,
            snapshot.spread,
        ] {
            hash.update(v.to_bits().to_le_bytes());
        }
        hash.update(space.bins_per_oct.to_le_bytes());
        hash.update(snapshot.unison.to_le_bytes());
        if let Some(p) = pattern {
            for v in [p.min_mul, p.max_mul, p.min_dist_erb, p.gamma] {
                hash.update(v.to_bits().to_le_bytes());
            }
            hash.update(p.count.to_le_bytes());
            match &p.kind {
                ModePatternKind::Custom { ratios } | ModePatternKind::ModalTable { ratios, .. } => {
                    hash.update([0]);
                    for r in ratios {
                        hash.update(r.to_bits().to_le_bytes());
                    }
                }
                ModePatternKind::Harmonic => hash.update([1]),
                ModePatternKind::Odd => hash.update([2]),
                ModePatternKind::PowerLaw { beta } => {
                    hash.update([3]);
                    hash.update(beta.to_bits().to_le_bytes());
                }
                ModePatternKind::StiffString { stiffness } => {
                    hash.update([4]);
                    hash.update(stiffness.to_bits().to_le_bytes());
                }
                ModePatternKind::LandscapeDensity => hash.update([5]),
                ModePatternKind::LandscapePeaks => hash.update([6]),
            }
        } else {
            hash.update([255]);
        }
        for i in 0..bins {
            let mass = terrain(i);
            if !mass.is_finite() {
                return None;
            }
            self.terrain_scan[i] = mass;
            hash.update(mass.to_bits().to_le_bytes());
            // Candidate-dependent timbre reads the un-eroded production terrain.
            hash.update(landscape.consonance_density_mass[i].to_bits().to_le_bytes());
            hash.update(landscape.consonance_field_level[i].to_bits().to_le_bytes());
        }
        let key: [u8; 32] = hash.finalize().into();
        if let Some(idx) = self.cache.iter().position(|c| c.key == Some(key)) {
            return Some(idx);
        }
        let slot = self.next_cache;
        self.next_cache = (slot + 1) % CACHE_ENTRIES;
        self.cache[slot].key = None;
        self.cache[slot]
            .masses
            .resize(frequencies.map_or(bins, <[f32]>::len), 0.0);
        self.cache[slot].masses.fill(0.0);
        #[cfg(test)]
        {
            self.preparations += 1;
        }
        let lo = frequencies.map_or_else(|| space.nearest_index(freq_range.0), |_| 0);
        let hi = frequencies.map_or_else(|| space.nearest_index(freq_range.1), |f| f.len() - 1);
        for idx in lo..=hi {
            let hz = frequencies.map_or_else(
                || space.freq_of_index(idx).clamp(freq_range.0, freq_range.1),
                |f| f[idx],
            );
            self.ratios.clear();
            if kind != BodyKind::Sine {
                let fallback = ModePattern::harmonic_modes();
                // Harmonic's absent pattern leaves ratios implicit; Modal's default is a pattern.
                let actual = if kind == BodyKind::Modal {
                    Some(pattern.unwrap_or(&fallback))
                } else {
                    pattern
                };
                if let Some(p) = actual
                    && !p.eval_without_jitter_into(
                        hz,
                        landscape,
                        &mut self.ratios,
                        &mut self.pattern_weights,
                        &mut self.pattern_candidates,
                    )
                {
                    return None;
                }
            }
            let mut ratios = std::mem::take(&mut self.ratios);
            let ok = self.prepare(&snapshot, (!ratios.is_empty()).then_some(ratios.as_slice()));
            ratios.clear();
            self.ratios = ratios;
            if !ok {
                return None;
            }
            let mass = self.density(kind, hz, space)?;
            let weighted = self.density_weight(mass, &self.terrain_scan, space);
            if !weighted.is_finite() {
                return None;
            }
            self.cache[slot].masses[idx] = weighted;
        }
        self.cache[slot].key = Some(key);
        Some(slot)
    }

    fn density_weight(&self, mass: f32, terrain_scan: &[f32], space: &Log2Space) -> f32 {
        space.assert_scan_len_named(terrain_scan, "birth_terrain_scan");
        space.assert_scan_len_named(&self.density_scan, "birth_body_density_scan");
        self.density_scan
            .iter()
            .zip(&self.du_scan)
            .zip(terrain_scan)
            .map(|((&d, &w), &m)| f64::from(d) * f64::from(w) * f64::from(m))
            .sum::<f64>() as f32
            / mass
    }

    pub(crate) fn candidate_mass(&self, slot: usize, bin: usize) -> f32 {
        self.cache[slot].masses[bin]
    }
}

#[cfg(test)]
#[path = "birth_surrogate_tests.rs"]
mod tests;
