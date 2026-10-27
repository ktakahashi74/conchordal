//! Landscape-level habituation: a per-bin state that erodes the consonance
//! evaluation terrain under sustained perceived-consonance activity and
//! recovers after release. See
//! docs/superpowers/specs/2026-07-17-habituation-field-design.md.

const LN_STALE: f32 = std::f32::consts::LN_10; // "stale" = 90% eroded

/// Multi-timescale adaptation (research option). Octave-spaced linear components from
/// 0.3 s to 38.4 s cover the few hundred milliseconds to tens of seconds over which A1
/// stimulus-specific adaptation runs concurrently (Ulanovsky et al. 2004). Weights
/// follow tau^-0.15, the fractional order Lundstrom et al. (2008) measured over about
/// 1-30 s. With the span truncated, the step response is not a pure power law; it keeps
/// the overall depth near the single-time-constant default (73% eroded at 5 s) while
/// recovery time grows with exposure.
const MULTISCALE_COMPONENTS: usize = 8;
const MULTISCALE_TAU_MIN_SEC: f32 = 0.3;
const MULTISCALE_ORDER: f32 = 0.15;

/// Relax a signed score toward its level-neutral baseline `theta`.
#[inline]
pub fn erode_score(raw: f32, h: f32, theta: f32) -> f32 {
    let h = h.clamp(0.0, 1.0);
    theta + (raw - theta) * (1.0 - h)
}

/// Erode a non-negative mass multiplicatively.
#[inline]
pub fn erode_mass(raw: f32, h: f32) -> f32 {
    raw * (1.0 - h.clamp(0.0, 1.0))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HabituationParams {
    pub enabled: bool,
    pub satiation_sec: f32,
    pub recovery_sec: f32,
    pub ref_drive: f32,
    /// Replace the satiation/recovery pair with multi-timescale adaptation in the
    /// landscape field. Onset habituation keeps the single pair.
    pub multiscale: bool,
}

impl Default for HabituationParams {
    fn default() -> Self {
        Self {
            enabled: true,
            satiation_sec: 5.0,
            recovery_sec: 8.0,
            ref_drive: 0.25,
            multiscale: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct HabituationField {
    enabled: bool,
    tau_e: f32,
    tau_r: f32,
    ref_drive: f32,
    theta: f32,
    h: Vec<f32>,
    /// Bin-major component states for multi-timescale adaptation; empty when single.
    components: Vec<f32>,
    weights: [f32; MULTISCALE_COMPONENTS],
}

impl HabituationField {
    pub fn new(params: &HabituationParams, theta: f32, n_bins: usize) -> Self {
        Self {
            enabled: params.enabled,
            tau_e: (params.satiation_sec.max(1e-3)) / LN_STALE,
            tau_r: (params.recovery_sec.max(1e-3)) / LN_STALE,
            ref_drive: params.ref_drive.max(1e-6),
            theta,
            h: vec![0.0; n_bins],
            components: if params.multiscale {
                vec![0.0; n_bins * MULTISCALE_COMPONENTS]
            } else {
                Vec::new()
            },
            weights: multiscale_weights(),
        }
    }

    pub fn ensure_len(&mut self, n_bins: usize) {
        if self.h.len() != n_bins {
            self.h.resize(n_bins, 0.0);
            if !self.components.is_empty() {
                self.components.resize(n_bins * MULTISCALE_COMPONENTS, 0.0);
            }
        }
    }

    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    #[inline]
    pub fn theta(&self) -> f32 {
        self.theta
    }

    #[inline]
    pub fn state(&self) -> &[f32] {
        &self.h
    }

    #[inline]
    fn transfer(&self, drive_raw: f32) -> f32 {
        let d = drive_raw.max(0.0);
        d / (d + self.ref_drive)
    }

    /// Advance `h` one step. `drive_raw[i] = level[i] * proj[i]` (perceived-
    /// consonance activity in root coordinate); asymmetric relaxation:
    /// rising uses tau_e (satiation), falling uses tau_r (recovery).
    ///
    /// F2: both inputs must be aligned to the field's bins. Call `ensure_len`
    /// first when the space can change.
    pub fn advance_from_parts(&mut self, level: &[f32], proj: &[f32], dt: f32) {
        assert_eq!(
            level.len(),
            self.h.len(),
            "habituation level scan length mismatch"
        );
        assert_eq!(
            proj.len(),
            self.h.len(),
            "habituation proj scan length mismatch"
        );
        if !self.enabled || self.h.is_empty() {
            return;
        }
        let dt = dt.max(0.0);
        if !self.components.is_empty() {
            let a: [f32; MULTISCALE_COMPONENTS] =
                std::array::from_fn(|k| (-dt / multiscale_tau(k)).exp());
            for i in 0..self.h.len() {
                let d = self.transfer(level[i] * proj[i]).clamp(0.0, 1.0);
                let states = &mut self.components
                    [i * MULTISCALE_COMPONENTS..(i + 1) * MULTISCALE_COMPONENTS];
                let mut h = 0.0;
                for k in 0..MULTISCALE_COMPONENTS {
                    states[k] = a[k] * states[k] + (1.0 - a[k]) * d;
                    h += self.weights[k] * states[k];
                }
                self.h[i] = h.clamp(0.0, 1.0);
            }
            return;
        }
        let a_e = (-dt / self.tau_e).exp();
        let a_r = (-dt / self.tau_r).exp();
        for i in 0..self.h.len() {
            let d = self.transfer(level[i] * proj[i]).clamp(0.0, 1.0);
            let a = if d > self.h[i] { a_e } else { a_r };
            self.h[i] = (a * self.h[i] + (1.0 - a) * d).clamp(0.0, 1.0);
        }
    }
}

fn multiscale_tau(k: usize) -> f32 {
    MULTISCALE_TAU_MIN_SEC * (1u32 << k) as f32
}

fn multiscale_weights() -> [f32; MULTISCALE_COMPONENTS] {
    let raw: [f32; MULTISCALE_COMPONENTS] =
        std::array::from_fn(|k| multiscale_tau(k).powf(-MULTISCALE_ORDER));
    let total: f32 = raw.iter().sum();
    raw.map(|w| w / total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_drive_params() -> HabituationParams {
        HabituationParams::default()
    }

    #[test]
    fn disabled_is_identity() {
        let f = HabituationField::new(
            &HabituationParams {
                enabled: false,
                ..Default::default()
            },
            0.0,
            4,
        );
        assert!(!f.is_enabled());
        assert_eq!(erode_score(0.7, 0.0, 0.0), 0.7);
        assert_eq!(erode_mass(0.7, 0.0), 0.7);
    }

    #[test]
    fn disabled_advance_is_noop() {
        let mut f = HabituationField::new(
            &HabituationParams {
                enabled: false,
                ..Default::default()
            },
            0.0,
            3,
        );
        assert!(!f.is_enabled());
        for _ in 0..100 {
            f.advance_from_parts(&[1.0, 1.0, 1.0], &[1e6, 1e6, 1e6], 0.05);
        }
        for &h in f.state() {
            assert_eq!(h, 0.0, "disabled field must not change state");
        }
    }

    #[test]
    fn erode_relaxes_score_toward_theta() {
        assert!((erode_score(0.9, 1.0, 0.2) - 0.2).abs() < 1e-6);
        assert!((erode_score(0.9, 0.5, 0.2) - 0.55).abs() < 1e-6);
        assert!(erode_score(-0.5, 0.5, 0.0) > -0.5);
    }

    #[test]
    fn erode_mass_is_multiplicative() {
        assert!((erode_mass(0.8, 0.5) - 0.4).abs() < 1e-6);
        assert_eq!(erode_mass(0.8, 1.0), 0.0);
        assert_eq!(erode_mass(0.8, 0.0), 0.8);
    }

    #[test]
    fn satiation_reaches_0_9_in_satiation_sec_under_full_drive() {
        let mut f = HabituationField::new(&full_drive_params(), 0.0, 1);
        let level = [1.0f32];
        let proj = [1000.0f32]; // transfer(1000) ~= 1.0
        let dt = 0.01;
        let mut t = 0.0;
        while f.state()[0] < 0.9 {
            f.advance_from_parts(&level, &proj, dt);
            t += dt;
        }
        assert!((t - 5.0).abs() < 0.15, "reached 0.9 at t={t}");
    }

    #[test]
    fn recovery_falls_to_0_1_in_recovery_sec_at_zero_drive() {
        let mut f = HabituationField::new(&full_drive_params(), 0.0, 1);
        for _ in 0..2000 {
            f.advance_from_parts(&[1.0], &[1000.0], 0.01);
        }
        let start = f.state()[0];
        assert!(start > 0.99);
        let dt = 0.01;
        let mut t = 0.0;
        while f.state()[0] > 0.1 {
            f.advance_from_parts(&[0.0], &[0.0], dt);
            t += dt;
        }
        assert!((t - 8.0).abs() < 0.2, "fell to 0.1 at t={t}");
    }

    #[test]
    #[should_panic(expected = "habituation level scan length mismatch")]
    fn advance_panics_on_level_len_mismatch() {
        let mut f = HabituationField::new(&full_drive_params(), 0.0, 3);
        f.advance_from_parts(&[1.0, 1.0], &[1.0, 1.0, 1.0], 0.05);
    }

    #[test]
    #[should_panic(expected = "habituation proj scan length mismatch")]
    fn advance_panics_on_proj_len_mismatch() {
        let mut f = HabituationField::new(&full_drive_params(), 0.0, 3);
        f.advance_from_parts(&[1.0, 1.0, 1.0], &[1.0, 1.0], 0.05);
    }

    fn multiscale() -> HabituationParams {
        HabituationParams {
            multiscale: true,
            ..Default::default()
        }
    }

    #[test]
    fn multiscale_step_keeps_eroding_across_time_scales() {
        let mut f = HabituationField::new(&multiscale(), 0.0, 1);
        let dt = 0.01;
        let mut at = |seconds: f32| {
            let steps = (seconds / dt).round() as usize;
            for _ in 0..steps {
                f.advance_from_parts(&[1.0], &[1000.0], dt);
            }
            f.state()[0]
        };
        // Expected sums of the weighted components: 0.46 at 1 s, 0.73 at 5 s, 0.94 at 30 s.
        let h1 = at(1.0);
        let h5 = at(4.0);
        let h30 = at(25.0);
        assert!((h1 - 0.46).abs() < 0.02, "h(1 s) = {h1}");
        assert!((h5 - 0.73).abs() < 0.02, "h(5 s) = {h5}");
        assert!((h30 - 0.94).abs() < 0.02, "h(30 s) = {h30}");
    }

    #[test]
    fn multiscale_recovery_lengthens_with_exposure() {
        let recovery_after = |exposure: f32| {
            let mut f = HabituationField::new(&multiscale(), 0.0, 1);
            let dt = 0.01;
            for _ in 0..(exposure / dt) as usize {
                f.advance_from_parts(&[1.0], &[1000.0], dt);
            }
            let mut t = 0.0;
            while f.state()[0] > 0.1 {
                f.advance_from_parts(&[0.0], &[0.0], dt);
                t += dt;
            }
            t
        };
        let (short, long) = (recovery_after(2.0), recovery_after(60.0));
        assert!((short - 2.3).abs() < 0.3, "after 2 s: {short}");
        assert!((long - 17.2).abs() < 1.0, "after 60 s: {long}");
    }

    #[test]
    fn multiscale_resizes_with_the_space() {
        let mut f = HabituationField::new(&multiscale(), 0.0, 2);
        f.ensure_len(5);
        f.advance_from_parts(&[1.0; 5], &[1000.0; 5], 1.0);
        assert!(f.state().iter().all(|&h| h > 0.4 && h < 0.5));
    }

    #[test]
    fn state_stays_bounded() {
        let mut f = HabituationField::new(&full_drive_params(), 0.0, 3);
        for _ in 0..10_000 {
            f.advance_from_parts(&[1.0, 0.0, 0.5], &[1e6, 1e6, 1e6], 0.05);
        }
        for &h in f.state() {
            assert!((0.0..=1.0).contains(&h), "h={h}");
        }
    }
}
